#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Co-iterative augmented hessian second order SCF solver (CIAH-SOSCF)
'''

import sys
import time
import copy
from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf import symm
from pyscf.lib import logger
from pyscf.scf import chkfile
from pyscf.scf import addons
from pyscf.scf import hf_symm, uhf_symm, ghf_symm
from pyscf.scf import hf, rohf, uhf
from pyscf.soscf import ciah
from pyscf import __config__

WITH_EX_EY_DEGENERACY = getattr(__config__, 'soscf_newton_ah_Ex_Ey_degeneracy', True)


# http://scicomp.stackexchange.com/questions/1234/matrix-exponential-of-a-skew-hermitian-matrix-with-fortran-95-and-lapack
def expmat(a):
    return scipy.linalg.expm(a)

# kwarg with_symmetry is required the stability analysis. It controls whether
# the stability analysis can break the point group symmetry.
def gen_g_hop_rhf(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None,
                  with_symmetry=True):
    mo_coeff0 = mo_coeff
    mol = mf.mol
    if getattr(mf, '_scf', None) and mf._scf.mol != mol:
        #TODO: construct vind with dual-basis treatment, (see also JCP, 118, 9497)
        # To project Hessians from another basis if different basis sets are used
        # in newton solver and underlying mean-filed solver.
        mo_coeff = addons.project_mo_nr2nr(mf._scf.mol, mo_coeff, mol)

    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbo = mo_coeff[:,occidx]
    orbv = mo_coeff[:,viridx]
    if with_symmetry and mol.symmetry:
        orbsym = hf_symm.get_orbsym(mol, mo_coeff)
        sym_forbid = orbsym[viridx,None] != orbsym[occidx]

    if fock_ao is None:
        # dm0 is the density matrix in projected basis. Computing fock in
        # projected basis.
        if getattr(mf, '_scf', None) and mf._scf.mol != mol:
            h1e = mf.get_hcore(mol)
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(h1e, dm=dm0)
        fock = reduce(numpy.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))
    else:
        # If fock is given, it corresponds to main basis. It needs to be
        # diagonalized with the mo_coeff of the main basis.
        fock = reduce(numpy.dot, (mo_coeff0.conj().T, fock_ao, mo_coeff0))

    g = fock[viridx[:,None],occidx] * 2

    foo = fock[occidx[:,None],occidx]
    fvv = fock[viridx[:,None],viridx]

    h_diag = (fvv.diagonal().real[:,None] - foo.diagonal().real) * 2

    if with_symmetry and mol.symmetry:
        g[sym_forbid] = 0
        h_diag[sym_forbid] = 0
    vind = _gen_rhf_response(mf, mo_coeff, mo_occ, singlet=None, hermi=1)

    def h_op(x):
        x = x.reshape(nvir,nocc)
        if with_symmetry and mol.symmetry:
            x = x.copy()
            x[sym_forbid] = 0
        x2 = numpy.einsum('ps,sq->pq', fvv, x)
        #x2-= .5*numpy.einsum('ps,rp->rs', foo, x)
        #x2-= .5*numpy.einsum('sp,rp->rs', foo, x)
        x2-= numpy.einsum('ps,rp->rs', foo, x)

        # *2 for double occupancy
        d1 = reduce(numpy.dot, (orbv, x*2, orbo.conj().T))
        dm1 = d1 + d1.conj().T
        v1 = vind(dm1)
        x2 += reduce(numpy.dot, (orbv.conj().T, v1, orbo))
        if with_symmetry and mol.symmetry:
            x2[sym_forbid] = 0
        return x2.ravel() * 2

    return g.reshape(-1), h_op, h_diag.reshape(-1)

def gen_g_hop_rohf(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None,
                   with_symmetry=True):
    if getattr(fock_ao, 'focka', None) is None:
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(h1e, dm=dm0)
    fock_ao = fock_ao.focka, fock_ao.fockb
    mo_occa = occidxa = mo_occ > 0
    mo_occb = occidxb = mo_occ ==2
    ug, uh_op, uh_diag = gen_g_hop_uhf(mf, (mo_coeff,)*2, (mo_occa,mo_occb),
                                       fock_ao, None, with_symmetry)

    viridxa = ~occidxa
    viridxb = ~occidxb
    uniq_var_a = viridxa[:,None] & occidxa
    uniq_var_b = viridxb[:,None] & occidxb
    uniq_ab = uniq_var_a | uniq_var_b
    nmo = mo_coeff.shape[-1]
    nocca = numpy.count_nonzero(mo_occa)
    nvira = nmo - nocca

    def sum_ab(x):
        x1 = numpy.zeros((nmo,nmo), dtype=x.dtype)
        x1[uniq_var_a]  = x[:nvira*nocca]
        x1[uniq_var_b] += x[nvira*nocca:]
        return x1[uniq_ab]

    g = sum_ab(ug)
    h_diag = sum_ab(uh_diag)
    def h_op(x):
        x1 = numpy.zeros((nmo,nmo), dtype=x.dtype)
        # unpack ROHF rotation parameters
        x1[uniq_ab] = x
        x1 = numpy.hstack((x1[uniq_var_a],x1[uniq_var_b]))
        return sum_ab(uh_op(x1))

    return g, h_op, h_diag

def gen_g_hop_uhf(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None,
                  with_symmetry=True):
    mol = mf.mol
    mo_coeff0 = mo_coeff
    if getattr(mf, '_scf', None) and mf._scf.mol != mol:
        #TODO: construct vind with dual-basis treatment, (see also JCP, 118, 9497)
        mo_coeff = (addons.project_mo_nr2nr(mf._scf.mol, mo_coeff[0], mol),
                    addons.project_mo_nr2nr(mf._scf.mol, mo_coeff[1], mol))

    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]
    if with_symmetry and mol.symmetry:
        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
        sym_forbida = orbsyma[viridxa,None] != orbsyma[occidxa]
        sym_forbidb = orbsymb[viridxb,None] != orbsymb[occidxb]
        sym_forbid = numpy.hstack((sym_forbida.ravel(), sym_forbidb.ravel()))

    if fock_ao is None:
        if getattr(mf, '_scf', None) and mf._scf.mol != mol:
            h1e = mf.get_hcore(mol)
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(h1e, dm=dm0)
        focka = reduce(numpy.dot, (mo_coeff[0].conj().T, fock_ao[0], mo_coeff[0]))
        fockb = reduce(numpy.dot, (mo_coeff[1].conj().T, fock_ao[1], mo_coeff[1]))
    else:
        focka = reduce(numpy.dot, (mo_coeff0[0].conj().T, fock_ao[0], mo_coeff0[0]))
        fockb = reduce(numpy.dot, (mo_coeff0[1].conj().T, fock_ao[1], mo_coeff0[1]))
    fooa = focka[occidxa[:,None],occidxa]
    fvva = focka[viridxa[:,None],viridxa]
    foob = fockb[occidxb[:,None],occidxb]
    fvvb = fockb[viridxb[:,None],viridxb]

    g = numpy.hstack((focka[viridxa[:,None],occidxa].ravel(),
                      fockb[viridxb[:,None],occidxb].ravel()))

    h_diaga = fvva.diagonal().real[:,None] - fooa.diagonal().real
    h_diagb = fvvb.diagonal().real[:,None] - foob.diagonal().real
    h_diag = numpy.hstack((h_diaga.reshape(-1), h_diagb.reshape(-1)))

    if with_symmetry and mol.symmetry:
        g[sym_forbid] = 0
        h_diag[sym_forbid] = 0

    vind = _gen_uhf_response(mf, mo_coeff, mo_occ, hermi=1)

    def h_op(x):
        if with_symmetry and mol.symmetry:
            x = x.copy()
            x[sym_forbid] = 0
        x1a = x[:nvira*nocca].reshape(nvira,nocca)
        x1b = x[nvira*nocca:].reshape(nvirb,noccb)
        x2a = numpy.einsum('pr,rq->pq', fvva, x1a)
        x2a-= numpy.einsum('sq,ps->pq', fooa, x1a)
        x2b = numpy.einsum('pr,rq->pq', fvvb, x1b)
        x2b-= numpy.einsum('sq,ps->pq', foob, x1b)

        d1a = reduce(numpy.dot, (orbva, x1a, orboa.conj().T))
        d1b = reduce(numpy.dot, (orbvb, x1b, orbob.conj().T))
        dm1 = numpy.array((d1a+d1a.conj().T,d1b+d1b.conj().T))
        v1 = vind(dm1)
        x2a += reduce(numpy.dot, (orbva.conj().T, v1[0], orboa))
        x2b += reduce(numpy.dot, (orbvb.conj().T, v1[1], orbob))

        x2 = numpy.hstack((x2a.ravel(), x2b.ravel()))
        if with_symmetry and mol.symmetry:
            x2[sym_forbid] = 0
        return x2

    return g, h_op, h_diag

def gen_g_hop_ghf(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None,
                  with_symmetry=True):
    mol = mf.mol
    occidx = numpy.where(mo_occ==1)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbo = mo_coeff[:,occidx]
    orbv = mo_coeff[:,viridx]
    if with_symmetry and mol.symmetry:
        orbsym = ghf_symm.get_orbsym(mol, mo_coeff)
        sym_forbid = orbsym[viridx,None] != orbsym[occidx]

    if fock_ao is None:
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(h1e, dm=dm0)
    fock = reduce(numpy.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))

    g = fock[viridx[:,None],occidx]

    foo = fock[occidx[:,None],occidx]
    fvv = fock[viridx[:,None],viridx]

    h_diag = fvv.diagonal().real[:,None] - foo.diagonal().real

    if with_symmetry and mol.symmetry:
        g[sym_forbid] = 0
        h_diag[sym_forbid] = 0

    vind = _gen_ghf_response(mf, mo_coeff, mo_occ, hermi=1)

    def h_op(x):
        x = x.reshape(nvir,nocc)
        if with_symmetry and mol.symmetry:
            x = x.copy()
            x[sym_forbid] = 0
        x2 = numpy.einsum('ps,sq->pq', fvv, x)
        x2-= numpy.einsum('ps,rp->rs', foo, x)

        d1 = reduce(numpy.dot, (orbv, x, orbo.conj().T))
        dm1 = d1 + d1.conj().T
        v1 = vind(dm1)
        x2 += reduce(numpy.dot, (orbv.conj().T, v1, orbo))
        if with_symmetry and mol.symmetry:
            x2[sym_forbid] = 0
        return x2.ravel()

    return g.reshape(-1), h_op, h_diag.reshape(-1)


def _gen_rhf_response(mf, mo_coeff=None, mo_occ=None,
                      singlet=None, hermi=0, max_memory=None):
    assert(not isinstance(mf, (uhf.UHF, rohf.ROHF)))

    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    if _is_dft_object(mf):
        from pyscf.dft import rks
        from pyscf.dft import numint
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if getattr(mf, 'nlc', '') != '':
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'deriviative is not available. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = abs(hyb) > 1e-10

        # mf can be pbc.dft.RKS object with multigrid
        if (not hybrid and
            'MultiGridFFTDF' == getattr(mf, 'with_df', None).__class__.__name__):
            from pyscf.pbc.dft import multigrid
            dm0 = mf.make_rdm1(mo_coeff, mo_occ)
            return multigrid._gen_rhf_response(mf, dm0, singlet, hermi)

        if singlet is None:  # for newton solver
            rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                                mo_coeff, mo_occ, 0)
        else:
            rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                                [mo_coeff]*2, [mo_occ*.5]*2, spin=1)
        dm0 = None #mf.make_rdm1(mo_coeff, mo_occ)

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        if singlet is None:  # Without specify singlet, general case
            def vind(dm1):
                # The singlet hessian
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    v1 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                       rho0, vxc, fxc, max_memory=max_memory)
                if hybrid:
                    if hermi != 2:
                        vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                        vk *= hyb
                        if abs(omega) > 1e-10:  # For range separated Coulomb
                            vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                        v1 += vj - .5 * vk
                    else:
                        v1 -= .5 * hyb * mf.get_k(mol, dm1, hermi=hermi)
                elif hermi != 2:
                    v1 += mf.get_j(mol, dm1, hermi=hermi)
                return v1

        elif singlet:
            def vind(dm1):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1, dm1*.5 should be scaled
                    v1 = numint.nr_rks_fxc_st(ni, mol, mf.grids, mf.xc, dm0, dm1, 0,
                                              True, rho0, vxc, fxc,
                                              max_memory=max_memory)
                    v1 *= .5
                if hybrid:
                    if hermi != 2:
                        vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                        vk *= hyb
                        if abs(omega) > 1e-10:  # For range separated Coulomb
                            vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                        v1 += vj - .5 * vk
                    else:
                        v1 -= .5 * hyb * mf.get_k(mol, dm1, hermi=hermi)
                elif hermi != 2:
                    v1 += mf.get_j(mol, dm1, hermi=hermi)
                return v1
        else:  # triplet
            def vind(dm1):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1, dm1*.5 should be scaled
                    v1 = numint.nr_rks_fxc_st(ni, mol, mf.grids, mf.xc, dm0, dm1, 0,
                                              False, rho0, vxc, fxc,
                                              max_memory=max_memory)
                    v1 *= .5
                if hybrid:
                    vk = mf.get_k(mol, dm1, hermi=hermi)
                    vk *= hyb
                    if abs(omega) > 1e-10:  # For range separated Coulomb
                        vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                    v1 += -.5 * vk
                return v1

    else:  # HF
        if (singlet is None or singlet) and hermi != 2:
            def vind(dm1):
                vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                return vj - .5 * vk
        else:
            def vind(dm1):
                return -.5 * mf.get_k(mol, dm1, hermi=hermi)

    return vind


def _gen_uhf_response(mf, mo_coeff=None, mo_occ=None,
                      with_j=True, hermi=0, max_memory=None):
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    if _is_dft_object(mf):
        from pyscf.dft import rks
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if getattr(mf, 'nlc', '') != '':
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'deriviative is not available. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = abs(hyb) > 1e-10

        # mf can be pbc.dft.UKS object with multigrid
        if (not hybrid and
            'MultiGridFFTDF' == getattr(mf, 'with_df', None).__class__.__name__):
            from pyscf.pbc.dft import multigrid
            dm0 = mf.make_rdm1(mo_coeff, mo_occ)
            return multigrid._gen_uhf_response(mf, dm0, with_j, hermi)

        rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                            mo_coeff, mo_occ, 1)
        #dm0 =(numpy.dot(mo_coeff[0]*mo_occ[0], mo_coeff[0].T.conj()),
        #      numpy.dot(mo_coeff[1]*mo_occ[1], mo_coeff[1].T.conj()))
        dm0 = None

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        def vind(dm1):
            if hermi == 2:
                v1 = numpy.zeros_like(dm1)
            else:
                v1 = ni.nr_uks_fxc(mol, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                   rho0, vxc, fxc, max_memory=max_memory)
            if not hybrid:
                if with_j:
                    vj = mf.get_j(mol, dm1, hermi=hermi)
                    v1 += vj[0] + vj[1]
            else:
                if with_j:
                    vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                    vk *= hyb
                    if abs(omega) > 1e-10:  # For range separated Coulomb
                        vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                    v1 += vj[0] + vj[1] - vk
                else:
                    vk = mf.get_k(mol, dm1, hermi=hermi)
                    vk *= hyb
                    if abs(omega) > 1e-10:  # For range separated Coulomb
                        vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                    v1 -= vk
            return v1

    elif with_j:
        def vind(dm1):
            vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
            v1 = vj[0] + vj[1] - vk
            return v1

    else:
        def vind(dm1):
            return -mf.get_k(mol, dm1, hermi=hermi)

    return vind


def _gen_ghf_response(mf, mo_coeff=None, mo_occ=None,
                      with_j=True, hermi=0, max_memory=None):
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    if _is_dft_object(mf):
        from pyscf.dft import numint
        raise NotImplementedError

    elif with_j:
        def vind(dm1):
            vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
            return vj - vk

    else:
        def vind(dm1):
            return -mf.get_k(mol, dm1, hermi=hermi)

    return vind


# Dual basis for gradients and hessian
def project_mol(mol, dual_basis={}):
    from pyscf import df
    uniq_atoms = set([a[0] for a in mol._atom])
    newbasis = {}
    for symb in uniq_atoms:
        if gto.charge(symb) <= 10:
            newbasis[symb] = '321g'
        elif gto.charge(symb) <= 12:
            newbasis[symb] = 'dzp'
        elif gto.charge(symb) <= 18:
            newbasis[symb] = 'dz'
        elif gto.charge(symb) <= 86:
            newbasis[symb] = 'dzp'
        else:
            newbasis[symb] = 'sto3g'
    if isinstance(dual_basis, (dict, tuple, list)):
        newbasis.update(dual_basis)
    elif isinstance(dual_basis, str):
        for k in newbasis:
            newbasis[k] = dual_basis
    return df.addons.make_auxmol(mol, newbasis)


# TODO: check whether high order terms in (g_orb, h_op) affects optimization
# To include high order terms, we can generate mo_coeff every time u matrix
# changed and insert the mo_coeff to g_op, h_op.
# Seems the high order terms do not help optimization?
def _rotate_orb_cc(mf, h1e, s1e, conv_tol_grad=None, verbose=None):
    log = logger.new_logger(mf, verbose)

    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(mf.conv_tol*.1)
#TODO: dynamically adjust max_stepsize, as done in mc1step.py

    def precond(x, e):
        hdiagd = h_diag-(e-mf.ah_level_shift)
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        x = x/hdiagd
## Because of DFT, donot norm to 1 which leads 1st DM too large.
#        norm_x = numpy.linalg.norm(x)
#        if norm_x < 1e-2:
#            x *= 1e-2/norm_x
        return x

    t3m = (time.clock(), time.time())
    u = g_kf = g_orb = kfcount = jkcount = None
    dm0 = vhf0 = None
    g_op = lambda: g_orb
    while True:
        mo_coeff, mo_occ, dm0, vhf0, e_tot = (yield u, g_kf, kfcount, jkcount, dm0, vhf0)
        fock_ao = mf.get_fock(h1e, s1e, vhf0, dm0)

        g_kf, h_op, h_diag = mf.gen_g_hop(mo_coeff, mo_occ, fock_ao)
        norm_gkf = numpy.linalg.norm(g_kf)
        if g_orb is None:
            log.debug('    |g|= %4.3g (keyframe)', norm_gkf)
            kf_trust_region = mf.kf_trust_region
            x0_guess = g_kf
        else:
            norm_dg = numpy.linalg.norm(g_kf-g_orb)
            log.debug('    |g|= %4.3g (keyframe), |g-correction|= %4.3g',
                      norm_gkf, norm_dg)
            kf_trust_region = min(max(norm_gorb/(norm_dg+1e-9), mf.kf_trust_region), 10)
            log.debug1('Set  kf_trust_region = %g', kf_trust_region)
            x0_guess = dxi
        g_orb = g_kf
        norm_gorb = norm_gkf

        ah_conv_tol = min(norm_gorb**2, mf.ah_conv_tol)
        # increase the AH accuracy when approach convergence
        #ah_start_cycle = max(mf.ah_start_cycle, int(-numpy.log10(norm_gorb)))
        ah_start_cycle = mf.ah_start_cycle
        imic = 0
        dr = 0
        ukf = None
        jkcount = 0
        kfcount = 0
        ikf = 0

        for ah_end, ihop, w, dxi, hdxi, residual, seig \
                in ciah.davidson_cc(h_op, g_op, precond, x0_guess,
                                    tol=ah_conv_tol, max_cycle=mf.ah_max_cycle,
                                    lindep=mf.ah_lindep, verbose=log):
            norm_residual = numpy.linalg.norm(residual)
            ah_start_tol = min(norm_gorb*5, mf.ah_start_tol)
            if (ah_end or ihop == mf.ah_max_cycle or # make sure to use the last step
                ((norm_residual < ah_start_tol) and (ihop >= ah_start_cycle)) or
                (seig < mf.ah_lindep)):
                imic += 1
                dxmax = numpy.max(abs(dxi))
                if dxmax > mf.max_stepsize:
                    scale = mf.max_stepsize / dxmax
                    log.debug1('... scale rotation size %g', scale)
                    dxi *= scale
                    hdxi *= scale
                else:
                    scale = None

                dr = dr + dxi
                g_orb = g_orb + hdxi
                norm_dr = numpy.linalg.norm(dr)
                norm_gorb = numpy.linalg.norm(g_orb)
                norm_dxi = numpy.linalg.norm(dxi)
                log.debug('    imic %d(%d)  |g|= %4.3g  |dxi|= %4.3g  '
                          'max(|x|)= %4.3g  |dr|= %4.3g  eig= %4.3g  seig= %4.3g',
                          imic, ihop, norm_gorb, norm_dxi,
                          dxmax, norm_dr, w, seig)

                max_cycle = max(mf.max_cycle_inner,
                                mf.max_cycle_inner-int(numpy.log(norm_gkf+1e-9)*2))
                log.debug1('Set ah_start_tol %g, ah_start_cycle %d, max_cycle %d',
                           ah_start_tol, ah_start_cycle, max_cycle)
                ikf += 1
                if imic > 3 and norm_gorb > norm_gkf*mf.ah_grad_trust_region:
                    g_orb = g_orb - hdxi
                    dr -= dxi
                    norm_gorb = numpy.linalg.norm(g_orb)
                    log.debug('|g| >> keyframe, Restore previouse step')
                    break

                elif (imic >= max_cycle or norm_gorb < conv_tol_grad/mf.ah_grad_trust_region):
                    break

                elif (ikf > 2 and # avoid frequent keyframe
#TODO: replace it with keyframe_scheduler
                      (ikf >= max(mf.kf_interval, mf.kf_interval-numpy.log(norm_dr+1e-9)) or
# Insert keyframe if the keyframe and the esitimated g_orb are too different
                       norm_gorb < norm_gkf/kf_trust_region)):
                    ikf = 0
                    u = mf.update_rotate_matrix(dr, mo_occ, mo_coeff=mo_coeff)
                    if ukf is not None:
                        u = mf.rotate_mo(ukf, u)
                    ukf = u
                    dr[:] = 0
                    mo1 = mf.rotate_mo(mo_coeff, u)
                    dm = mf.make_rdm1(mo1, mo_occ)
# use mf._scf.get_veff to avoid density-fit mf polluting get_veff
                    vhf0 = mf._scf.get_veff(mf._scf.mol, dm, dm_last=dm0, vhf_last=vhf0)
                    dm0 = dm
# Use API to compute fock instead of "fock=h1e+vhf0". This is because get_fock
# is the hook being overloaded in many places.
                    fock_ao = mf.get_fock(h1e, s1e, vhf0, dm0)
                    g_kf1 = mf.get_grad(mo1, mo_occ, fock_ao)
                    norm_gkf1 = numpy.linalg.norm(g_kf1)
                    norm_dg = numpy.linalg.norm(g_kf1-g_orb)
                    jkcount += 1
                    kfcount += 1
                    if log.verbose >= logger.DEBUG:
                        e_tot, e_last = mf._scf.energy_tot(dm, h1e, vhf0), e_tot
                        log.debug('Adjust keyframe g_orb to |g|= %4.3g  '
                                  '|g-correction|=%4.3g  E=%.12g dE=%.5g',
                                  norm_gkf1, norm_dg, e_tot, e_tot-e_last)

                    if (norm_dg < norm_gorb*mf.ah_grad_trust_region  # kf not too diff
                        #or norm_gkf1 < norm_gkf  # grad is decaying
                        # close to solution
                        or norm_gkf1 < conv_tol_grad*mf.ah_grad_trust_region):
                        kf_trust_region = min(max(norm_gorb/(norm_dg+1e-9), mf.kf_trust_region), 10)
                        log.debug1('Set kf_trust_region = %g', kf_trust_region)
                        g_orb = g_kf = g_kf1
                        norm_gorb = norm_gkf = norm_gkf1
                    else:
                        g_orb = g_orb - hdxi
                        dr -= dxi
                        norm_gorb = numpy.linalg.norm(g_orb)
                        log.debug('Out of trust region. Restore previouse step')
                        break

        u = mf.update_rotate_matrix(dr, mo_occ, mo_coeff=mo_coeff)
        if ukf is not None:
            u = mf.rotate_mo(ukf, u)
        jkcount += ihop + 1
        log.debug('    tot inner=%d  %d JK  |g|= %4.3g  |u-1|= %4.3g',
                  imic, jkcount, norm_gorb, numpy.linalg.norm(dr))
        h_op = h_diag = None
        t3m = log.timer('aug_hess in %d inner iters' % imic, *t3m)


def kernel(mf, mo_coeff=None, mo_occ=None, dm=None,
           conv_tol=1e-10, conv_tol_grad=None, max_cycle=50, dump_chk=True,
           callback=None, verbose=logger.NOTE):
    cput0 = (time.clock(), time.time())
    log = logger.new_logger(mf, verbose)
    mol = mf._scf.mol
    if mol != mf.mol:
        logger.warn(mf, 'dual-basis SOSCF is an experimental feature. It is '
                    'still in testing.')

    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        log.info('Set conv_tol_grad to %g', conv_tol_grad)

# call mf._scf.get_hcore, mf._scf.get_ovlp because they might be overloaded
    h1e = mf._scf.get_hcore(mol)
    s1e = mf._scf.get_ovlp(mol)

    if mo_coeff is not None and mo_occ is not None:
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        # call mf._scf.get_veff, to avoid "newton().density_fit()" polluting get_veff
        vhf = mf._scf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        mo_energy, mo_tmp = mf.eig(fock, s1e)
        mf.get_occ(mo_energy, mo_tmp)
        mo_tmp = None

    else:
        if dm is None:
            logger.debug(mf, 'Initial guess density matrix is not given. '
                         'Generating initial guess from %s', mf.init_guess)
            dm = mf.get_init_guess(mf._scf.mol, mf.init_guess)
        vhf = mf._scf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf._scf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)

    # Save mo_coeff and mo_occ because they are needed by function rotate_mo
    mf.mo_coeff, mf.mo_occ = mo_coeff, mo_occ

    e_tot = mf._scf.energy_tot(dm, h1e, vhf)
    fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
    log.info('Initial guess E= %.15g  |g|= %g', e_tot,
             numpy.linalg.norm(mf._scf.get_grad(mo_coeff, mo_occ, fock)))

    if dump_chk and mf.chkfile:
        chkfile.save_mol(mol, mf.chkfile)

# Copy the integral file to soscf object to avoid the integrals being cached
# twice.
    if mol is mf.mol and not getattr(mf, 'with_df', None):
        mf._eri = mf._scf._eri
        # If different direct_scf_cutoff is assigned to newton_ah mf.opt
        # object, mf.opt should be different to mf._scf.opt
        #mf.opt = mf._scf.opt

    rotaiter = _rotate_orb_cc(mf, h1e, s1e, conv_tol_grad, verbose=log)
    next(rotaiter)  # start the iterator
    kftot = jktot = 0
    scf_conv = False
    cput1 = log.timer('initializing second order scf', *cput0)

    for imacro in range(max_cycle):
        u, g_orb, kfcount, jkcount, dm_last, vhf = \
                rotaiter.send((mo_coeff, mo_occ, dm, vhf, e_tot))
        kftot += kfcount + 1
        jktot += jkcount + 1

        last_hf_e = e_tot
        norm_gorb = numpy.linalg.norm(g_orb)
        mo_coeff = mf.rotate_mo(mo_coeff, u, log)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf._scf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
# NOTE: DO NOT change the initial guess mo_occ, mo_coeff
        if mf.verbose >= logger.DEBUG:
            mo_energy, mo_tmp = mf.eig(fock, s1e)
            mf.get_occ(mo_energy, mo_tmp)
# call mf._scf.energy_tot for dft, because the (dft).get_veff step saved _exc in mf._scf
        e_tot = mf._scf.energy_tot(dm, h1e, vhf)

        log.info('macro= %d  E= %.15g  delta_E= %g  |g|= %g  %d KF %d JK',
                 imacro, e_tot, e_tot-last_hf_e, norm_gorb,
                 kfcount+1, jkcount)
        cput1 = log.timer('cycle= %d'%(imacro+1), *cput1)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs((e_tot-last_hf_e)/e_tot)*1e3 < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        if scf_conv:
            break

    if callable(callback):
        callback(locals())

    rotaiter.close()
    mo_energy, mo_coeff1 = mf._scf.canonicalize(mo_coeff, mo_occ, fock)
    if mf.canonicalization:
        log.info('Canonicalize SCF orbitals')
        mo_coeff = mo_coeff1
        if dump_chk:
            mf.dump_chk(locals())
    log.info('macro X = %d  E=%.15g  |g|= %g  total %d KF %d JK',
             imacro+1, e_tot, norm_gorb, kftot+1, jktot+1)
    if (numpy.any(mo_occ==0) and
        mo_energy[mo_occ>0].max() > mo_energy[mo_occ==0].min()):
        log.warn('HOMO %s > LUMO %s was found in the canonicalized orbitals.',
                 mo_energy[mo_occ>0].max(), mo_energy[mo_occ==0].min())
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

# Note which function that "density_fit" decorated.  density-fit have been
# considered separatedly for newton_mf and newton_mf._scf, there are 3 cases
# 1. both grad and hessian by df, because the input mf obj is decorated by df
#       newton(scf.density_fit(scf.RHF(mol)))
# 2. both grad and hessian in df, because both newton_mf and the input mf objs
#    are decorated by df
#       scf.density_fit(newton(scf.density_fit(scf.RHF(mol))))
# 3. grad by explicit scheme, hessian by df, because only newton_mf obj is
#    decorated by df
#       scf.density_fit(newton(scf.RHF(mol)))
# The following function is not necessary
#def density_fit_(mf, auxbasis='weigend+etb'):
#    mfaux = mf.density_fit(auxbasis)
#    mf.gen_g_hop = mfaux.gen_g_hop
#    return mf
#def density_fit(mf, auxbasis='weigend+etb'):
#    return density_fit_(copy.copy(mf), auxbasis)


# A tag to label the derived SCF class
class _CIAH_SOSCF(hf.SCF):
    '''
    Attributes for Newton solver:
        max_cycle_inner : int
            AH iterations within eacy macro iterations. Default is 10
        max_stepsize : int
            The step size for orbital rotation.  Small step is prefered.  Default is 0.05.
        canonicalization : bool
            To control whether to canonicalize the orbitals optimized by
            Newton solver.  Default is True.
    '''

    max_cycle_inner = getattr(__config__, 'soscf_newton_ah_SOSCF_max_cycle_inner', 12)
    max_stepsize = getattr(__config__, 'soscf_newton_ah_SOSCF_max_stepsize', .05)
    canonicalization = getattr(__config__, 'soscf_newton_ah_SOSCF_canonicalization', True)

    ah_start_tol = getattr(__config__, 'soscf_newton_ah_SOSCF_ah_start_tol', 1e9)
    ah_start_cycle = getattr(__config__, 'soscf_newton_ah_SOSCF_ah_start_cycle', 1)
    ah_level_shift = getattr(__config__, 'soscf_newton_ah_SOSCF_ah_level_shift', 0)
    ah_conv_tol = getattr(__config__, 'soscf_newton_ah_SOSCF_ah_conv_tol', 1e-12)
    ah_lindep = getattr(__config__, 'soscf_newton_ah_SOSCF_ah_lindep', 1e-14)
    ah_max_cycle = getattr(__config__, 'soscf_newton_ah_SOSCF_ah_max_cycle', 40)
    ah_grad_trust_region = getattr(__config__, 'soscf_newton_ah_SOSCF_ah_grad_trust_region', 2.5)
    kf_interval = getattr(__config__, 'soscf_newton_ah_SOSCF_kf_interval', 4)
    kf_trust_region = getattr(__config__, 'soscf_newton_ah_SOSCF_kf_trust_region', 5)

    def __init__(self, mf):
        self.__dict__.update(mf.__dict__)
        self._scf = mf
        self._keys.update(('max_cycle_inner', 'max_stepsize',
                           'canonicalization', 'ah_start_tol', 'ah_start_cycle',
                           'ah_level_shift', 'ah_conv_tol', 'ah_lindep',
                           'ah_max_cycle', 'ah_grad_trust_region', 'kf_interval',
                           'kf_trust_region'))

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        self._scf.dump_flags(verbose)
        log.info('******** %s Newton solver flags ********', self._scf.__class__)
        log.info('SCF tol = %g', self.conv_tol)
        log.info('conv_tol_grad = %s',    self.conv_tol_grad)
        log.info('max. SCF cycles = %d', self.max_cycle)
        log.info('direct_scf = %s', self._scf.direct_scf)
        if self._scf.direct_scf:
            log.info('direct_scf_tol = %g', self._scf.direct_scf_tol)
        if self.chkfile:
            log.info('chkfile to save SCF result = %s', self.chkfile)
        log.info('max_cycle_inner = %d',  self.max_cycle_inner)
        log.info('max_stepsize = %g', self.max_stepsize)
        log.info('ah_start_tol = %g',     self.ah_start_tol)
        log.info('ah_level_shift = %g',   self.ah_level_shift)
        log.info('ah_conv_tol = %g',      self.ah_conv_tol)
        log.info('ah_lindep = %g',        self.ah_lindep)
        log.info('ah_start_cycle = %d',   self.ah_start_cycle)
        log.info('ah_max_cycle = %d',     self.ah_max_cycle)
        log.info('ah_grad_trust_region = %g', self.ah_grad_trust_region)
        log.info('kf_interval = %d', self.kf_interval)
        log.info('kf_trust_region = %d', self.kf_trust_region)
        log.info('canonicalization = %s', self.canonicalization)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def build(self, mol=None):
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self._scf.build(mol)
        self.opt = None
        self._eri = None
        return self

    def kernel(self, mo_coeff=None, mo_occ=None, dm0=None):
        cput0 = (time.clock(), time.time())
        if dm0 is not None:
            if isinstance(dm0, str):
                sys.stderr.write('Newton solver reads density matrix from chkfile %s\n' % dm)
                dm0 = mf.from_chk(dm0)

        elif mo_coeff is not None and mo_occ is None:
            logger.warn(self, 'Newton solver expects mo_coeff with '
                        'mo_occ as initial guess but mo_occ is not found in '
                        'the arguments.\n      The given '
                        'argument is treated as density matrix.')
            dm0 = mo_coeff
            mo_coeff = mo_occ = None

        else:
            if mo_coeff is None: mo_coeff = self.mo_coeff
            if mo_occ is None: mo_occ = self.mo_occ

            # TODO: assert mo_coeff orth-normality. If not orth-normal,
            # build dm from mo_coeff and mo_occ then unset mo_coeff and mo_occ.

        self.build(self.mol)
        self.dump_flags()

        self.converged, self.e_tot, \
                self.mo_energy, self.mo_coeff, self.mo_occ = \
                kernel(self, mo_coeff, mo_occ, dm0, conv_tol=self.conv_tol,
                       conv_tol_grad=self.conv_tol_grad,
                       max_cycle=self.max_cycle,
                       callback=self.callback, verbose=self.verbose)

        logger.timer(self, 'Second order SCF', *cput0)
        self._finalize()
        return self.e_tot

    def from_dm(self, dm):
        '''Transform the initial guess density matrix to initial orbital
        coefficients.

        Note kernel function can handle initial guess properly in pyscf-1.7 or
        newer versions. This function is kept for backward compatibility.
        '''
        mf = self._scf
        mol = mf.mol
        h1e = mf.get_hcore(mol)
        s1e = mf.get_ovlp(mol)
        vhf = mf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return mo_coeff, mo_occ

    gen_g_hop = gen_g_hop_rhf

    def update_rotate_matrix(self, dx, mo_occ, u0=1, mo_coeff=None):
        dr = hf.unpack_uniq_var(dx, mo_occ)

        if WITH_EX_EY_DEGENERACY:
            mol = self._scf.mol
            if mol.symmetry and mol.groupname in ('Dooh', 'Coov'):
                orbsym = hf_symm.get_orbsym(mol, mo_coeff)
                _force_Ex_Ey_degeneracy_(dr, orbsym)
        return numpy.dot(u0, expmat(dr))

    def rotate_mo(self, mo_coeff, u, log=None):
        mo = numpy.dot(mo_coeff, u)
        if self._scf.mol.symmetry:
            orbsym = hf_symm.get_orbsym(self._scf.mol, mo_coeff)
            mo = lib.tag_array(mo, orbsym=orbsym)
        return mo


def newton(mf):
    '''Co-iterative augmented hessian (CIAH) second order SCF solver

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1', basis='cc-pvdz')
    >>> mf = scf.RHF(mol).run(conv_tol=.5)
    >>> mf = scf.newton(mf).set(conv_tol=1e-9)
    >>> mf.kernel()
    -1.0811707843774987
    '''
    from pyscf import scf

    if isinstance(mf, _CIAH_SOSCF):
        return mf

    assert(isinstance(mf, hf.SCF))
    if mf.__doc__ is None:
        mf_doc = ''
    else:
        mf_doc = mf.__doc__

    class SecondOrderRHF(mf.__class__, _CIAH_SOSCF):
        __doc__ = mf_doc + _CIAH_SOSCF.__doc__
        __init__ = _CIAH_SOSCF.__init__
        dump_flags = _CIAH_SOSCF.dump_flags
        build = _CIAH_SOSCF.build
        kernel = _CIAH_SOSCF.kernel

        gen_g_hop = gen_g_hop_rhf

        def rotate_mo(self, mo_coeff, u, log=None):
            mo = _CIAH_SOSCF.rotate_mo(self, mo_coeff, u, log)
            if log is not None and log.verbose >= logger.DEBUG:
                idx = self.mo_occ > 0
                s = reduce(numpy.dot, (mo[:,idx].conj().T, self._scf.get_ovlp(),
                                       self.mo_coeff[:,idx]))
                log.debug('Overlap to initial guess, SVD = %s',
                          _effective_svd(s, 1e-5))
                log.debug('Overlap to last step, SVD = %s',
                          _effective_svd(u[idx][:,idx], 1e-5))
            return mo

    if isinstance(mf, rohf.ROHF):
        class SecondOrderROHF(SecondOrderRHF):
            gen_g_hop = gen_g_hop_rohf
        return SecondOrderROHF(mf)

    elif isinstance(mf, uhf.UHF):
        class SecondOrderUHF(mf.__class__, _CIAH_SOSCF):
            __doc__ = mf_doc + _CIAH_SOSCF.__doc__
            __init__ = _CIAH_SOSCF.__init__
            dump_flags = _CIAH_SOSCF.dump_flags
            build = _CIAH_SOSCF.build

            gen_g_hop = gen_g_hop_uhf

            def update_rotate_matrix(self, dx, mo_occ, u0=1, mo_coeff=None):
                occidxa = mo_occ[0] > 0
                occidxb = mo_occ[1] > 0
                viridxa = ~occidxa
                viridxb = ~occidxb

                nmo = len(occidxa)
                dr = numpy.zeros((2,nmo,nmo), dtype=dx.dtype)
                uniq = numpy.array((viridxa[:,None] & occidxa,
                                    viridxb[:,None] & occidxb))
                dr[uniq] = dx
                dr = dr - dr.conj().transpose(0,2,1)

                if WITH_EX_EY_DEGENERACY:
                    mol = self._scf.mol
                    if mol.symmetry and mol.groupname in ('Dooh', 'Coov'):
                        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
                        _force_Ex_Ey_degeneracy_(dr[0], orbsyma)
                        _force_Ex_Ey_degeneracy_(dr[1], orbsymb)

                if isinstance(u0, int) and u0 == 1:
                    return numpy.asarray((expmat(dr[0]), expmat(dr[1])))
                else:
                    return numpy.asarray((numpy.dot(u0[0], expmat(dr[0])),
                                          numpy.dot(u0[1], expmat(dr[1]))))

            def rotate_mo(self, mo_coeff, u, log=None):
                mo = numpy.asarray((numpy.dot(mo_coeff[0], u[0]),
                                    numpy.dot(mo_coeff[1], u[1])))
                if self._scf.mol.symmetry:
                    orbsym = uhf_symm.get_orbsym(self._scf.mol, mo_coeff)
                    mo = lib.tag_array(mo, orbsym=orbsym)
                return mo

            def spin_square(self, mo_coeff=None, s=None):
                if mo_coeff is None:
                    mo_coeff = (self.mo_coeff[0][:,self.mo_occ[0]>0],
                                self.mo_coeff[1][:,self.mo_occ[1]>0])
                if getattr(self, '_scf', None) and self._scf.mol != self.mol:
                    s = self._scf.get_ovlp()
                return self._scf.spin_square(mo_coeff, s)

            def kernel(self, mo_coeff=None, mo_occ=None, dm0=None):
                if isinstance(mo_coeff, numpy.ndarray) and mo_coeff.ndim == 2:
                    mo_coeff = (mo_coeff, mo_coeff)
                if isinstance(mo_occ, numpy.ndarray) and mo_occ.ndim == 1:
                    mo_occ = (numpy.asarray(mo_occ >0, dtype=numpy.double),
                              numpy.asarray(mo_occ==2, dtype=numpy.double))
                return _CIAH_SOSCF.kernel(self, mo_coeff, mo_occ, dm0)

        return SecondOrderUHF(mf)

    elif isinstance(mf, scf.ghf.GHF):
        class SecondOrderGHF(mf.__class__, _CIAH_SOSCF):
            __doc__ = mf_doc + _CIAH_SOSCF.__doc__
            __init__ = _CIAH_SOSCF.__init__
            dump_flags = _CIAH_SOSCF.dump_flags
            build = _CIAH_SOSCF.build
            kernel = _CIAH_SOSCF.kernel

            gen_g_hop = gen_g_hop_ghf

            def update_rotate_matrix(self, dx, mo_occ, u0=1, mo_coeff=None):
                dr = hf.unpack_uniq_var(dx, mo_occ)

                if WITH_EX_EY_DEGENERACY:
                    mol = self._scf.mol
                    if mol.symmetry and mol.groupname in ('Dooh', 'Coov'):
                        orbsym = scf.ghf_symm.get_orbsym(mol, mo_coeff)
                        _force_Ex_Ey_degeneracy_(dr, orbsym)
                return numpy.dot(u0, expmat(dr))

            def rotate_mo(self, mo_coeff, u, log=None):
                mo = numpy.dot(mo_coeff, u)
                if self._scf.mol.symmetry:
                    orbsym = scf.ghf_symm.get_orbsym(self._scf.mol, mo_coeff)
                    mo = lib.tag_array(mo, orbsym=orbsym)
                return mo
        return SecondOrderGHF(mf)

    elif isinstance(mf, scf.dhf.UHF):
        raise RuntimeError('Not support Dirac-HF')

    else:
        return SecondOrderRHF(mf)

SVD_TOL = getattr(__config__, 'soscf_newton_ah_effective_svd_tol', 1e-5)
def _effective_svd(a, tol=SVD_TOL):
    w = numpy.linalg.svd(a)[1]
    return w[(tol<w) & (w<1-tol)]
del(SVD_TOL)

def _force_Ex_Ey_degeneracy_(dr, orbsym):
    '''Force the Ex and Ey orbitals to use the same rotation matrix'''
    # 0,1,4,5 are 1D irreps
    E_irrep_ids = set(orbsym).difference(set((0,1,4,5)))
    orbsym = numpy.asarray(orbsym)

    for ir in E_irrep_ids:
        if ir % 2 == 0:
            Ex = orbsym == ir
            Ey = orbsym ==(ir + 1)
            dr_x = dr[Ex[:,None]&Ex]
            dr_y = dr[Ey[:,None]&Ey]
            # In certain open-shell systems, the rotation amplitudes dr_x may
            # be equal to 0 while dr_y are not. In this case, we choose the
            # larger one to represent the rotation amplitudes for both.
            if numpy.linalg.norm(dr_x) > numpy.linalg.norm(dr_y):
                dr[Ey[:,None]&Ey] = dr_x
            else:
                dr[Ex[:,None]&Ex] = dr_y
    return dr

def _is_dft_object(mf):
    return getattr(mf, 'xc', None) is not None and hasattr(mf, '_numint')


if __name__ == '__main__':
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0., 1.    , 1.   )],
        ['H', ( 0., 0.5   , 1.   )],
        ['H', ( 1., 0.    ,-1.   )],
    ]

    mol.basis = '6-31g'
    mol.build()

    nmo = mol.nao_nr()
    m = newton(scf.RHF(mol))
    e0 = m.kernel()

#####################################
    mol.basis = '6-31g'
    mol.spin = 2
    mol.build(0, 0)
    m = scf.RHF(mol)
    m.max_cycle = 1
    #m.verbose = 5
    m.scf()
    e1 = kernel(newton(m), m.mo_coeff, m.mo_occ, max_cycle=50, verbose=5)[1]

    m = scf.UHF(mol)
    m.max_cycle = 1
    #m.verbose = 5
    m.scf()
    e2 = kernel(newton(m), m.mo_coeff, m.mo_occ, max_cycle=50, verbose=5)[1]

    m = scf.UHF(mol)
    m.max_cycle = 1
    #m.verbose = 5
    m.scf()
    nrmf = scf.density_fit(newton(m), 'weigend')
    nrmf.max_cycle = 50
    nrmf.conv_tol = 1e-8
    nrmf.conv_tol_grad = 1e-5
    #nrmf.verbose = 5
    e4 = nrmf.kernel()

    m = scf.density_fit(scf.UHF(mol), 'weigend')
    m.max_cycle = 1
    #m.verbose = 5
    m.scf()
    nrmf = scf.density_fit(newton(m), 'weigend')
    nrmf.max_cycle = 50
    nrmf.conv_tol_grad = 1e-5
    e5 = nrmf.kernel()

    m = newton(scf.GHF(mol))
    e6 = m.kernel()

    print(e0 - -2.93707955256)
    print(e1 - -2.99456398848)
    print(e2 - -2.99663808314)
    print(e4 - -2.99663808186)
    print(e5 - -2.99634506072)
    print(e6 - -3.002844505604826)
