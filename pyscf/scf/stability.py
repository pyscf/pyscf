#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

'''Wave Function Stability Analysis

Ref.
JCP 66, 3045 (1977); DOI:10.1063/1.434318
JCP 104, 9047 (1996); DOI:10.1063/1.471637

See also tddft/rhf.py and scf/newton_ah.py
'''

import numpy
import scipy
from functools import reduce
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf, hf_symm, uhf_symm
from pyscf.scf import _response_functions  # noqa
from pyscf.soscf import newton_ah
from pyscf import __config__

STAB_NROOTS = getattr(__config__, 'stab_nroots', 3)
STAB_TOL = getattr(__config__, 'stab_tol', 1e-4)

def rhf_stability(mf, internal=True, external=False, verbose=None, return_status=False,
                  nroots=STAB_NROOTS, tol=STAB_TOL):
    '''
    Stability analysis for RHF/RKS method.

    Args:
        mf : RHF or RKS object

    Kwargs:
        internal : bool
            Internal stability, within the RHF space.
        external : bool
            External stability. Including the RHF -> UHF and real -> complex
            stability analysis.
        return_status: bool
            Whether to return `stable_i` and `stable_e`
        nroots : int
            Number of roots solved by Davidson solver
        tol : float
            Convergence threshold for Davidson solver

    Returns:
        If return_status is False (default), the return value includes
        two set of orbitals, which are more close to the stable condition.
        The first corresponds to the internal stability
        and the second corresponds to the external stability.

        Else, another two boolean variables (indicating current status:
        stable or unstable) are returned.
        The first corresponds to the internal stability
        and the second corresponds to the external stability.
    '''
    mo_i = mo_e = None
    stable_i = stable_e = None
    if internal:
        mo_i, stable_i = rhf_internal(mf, verbose=verbose, return_status=True, nroots=nroots, tol=tol)
    if external:
        mo_e, stable_e = rhf_external(mf, verbose=verbose, return_status=True, nroots=nroots, tol=tol)
    if return_status:
        return mo_i, mo_e, stable_i, stable_e
    else:
        return mo_i, mo_e

def uhf_stability(mf, internal=True, external=False, verbose=None, return_status=False,
                  nroots=STAB_NROOTS, tol=STAB_TOL):
    '''
    Stability analysis for UHF/UKS method.

    Args:
        mf : UHF or UKS object

    Kwargs:
        internal : bool
            Internal stability, within the UHF space.
        external : bool
            External stability. Including the UHF -> GHF and real -> complex
            stability analysis.
        return_status: bool
            Whether to return `stable_i` and `stable_e`
        nroots : int
            Number of roots solved by Davidson solver
        tol : float
            Convergence threshold for Davidson solver

    Returns:
        If return_status is False (default), the return value includes
        two set of orbitals, which are more close to the stable condition.
        The first corresponds to the internal stability
        and the second corresponds to the external stability.

        Else, another two boolean variables (indicating current status:
        stable or unstable) are returned.
        The first corresponds to the internal stability
        and the second corresponds to the external stability.
    '''
    mo_i = mo_e = None
    stable_i = stable_e = None
    if internal:
        mo_i, stable_i = uhf_internal(mf, verbose=verbose, return_status=True, nroots=nroots, tol=tol)
    if external:
        mo_e, stable_e = uhf_external(mf, verbose=verbose, return_status=True, nroots=nroots, tol=tol)
    if return_status:
        return mo_i, mo_e, stable_i, stable_e
    else:
        return mo_i, mo_e

def rohf_stability(mf, internal=True, external=False, verbose=None, return_status=False,
                   nroots=STAB_NROOTS, tol=STAB_TOL):
    '''
    Stability analysis for ROHF/ROKS method.

    Args:
        mf : ROHF or ROKS object

    Kwargs:
        internal : bool
            Internal stability, within the RHF space.
        external : bool
            External stability. It is not available in current version.
        return_status: bool
            Whether to return `stable_i` and `stable_e`
        nroots : int
            Number of roots solved by Davidson solver
        tol : float
            Convergence threshold for Davidson solver

    Returns:
        If return_status is False (default), the return value includes
        two set of orbitals, which are more close to the stable condition.
        The first corresponds to the internal stability
        and the second corresponds to the external stability.

        Else, another two boolean variables (indicating current status:
        stable or unstable) are returned.
        The first corresponds to the internal stability
        and the second corresponds to the external stability.
    '''
    mo_i = mo_e = None
    stable_i = stable_e = None
    if internal:
        mo_i, stable_i = rohf_internal(mf, verbose=verbose, return_status=True, nroots=nroots, tol=tol)
    if external:
        mo_e, stable_e = rohf_external(mf, verbose=verbose, return_status=True, nroots=nroots, tol=tol)
    if return_status:
        return mo_i, mo_e, stable_i, stable_e
    else:
        return mo_i, mo_e

def dump_status(log, stable, method_class, stab_type):
    if not stable:
        log.note(method_class + f' wavefunction has an {stab_type} instability')
    else:
        log.note(method_class + f' wavefunction is stable in the {stab_type} '
                 'stability analysis')

def ghf_stability(mf, verbose=None, return_status=False,
                  nroots=STAB_NROOTS, tol=STAB_TOL):
    '''
    Stability analysis for GHF/GKS method.

    Args:
        mf : GHF or GKS object

    Kwargs:
        return_status: bool
            Whether to return `stable_i` and `stable_e`
        nroots : int
            Number of roots solved by Davidson solver
        tol : float
            Convergence threshold for Davidson solver

    Returns:
        If return_status is False (default), the return value includes
        a new set of orbitals, which are more close to the stable condition.

        Else, another one boolean variable (indicating current status:
        stable or unstable) is returned.
    '''
    log = logger.new_logger(mf, verbose)
    with_symmetry = True
    # Hessian might be incorrect. see https://github.com/pyscf/pyscf/issues/2591
    g, hop, hdiag = newton_ah.gen_g_hop_ghf(mf, mf.mo_coeff, mf.mo_occ)
    hdiag *= 2
    stable = True
    def precond(dx, e, x0):
        hdiagd = hdiag - e
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return dx/hdiagd
    def hessian_x(x): # See comments in function rhf_internal
        return hop(x).real * 2

    x0 = numpy.zeros_like(g)
    x0[g!=0] = 1. / hdiag[g!=0]
    if not with_symmetry:  # allow to break point group symmetry
        x0[numpy.argmin(hdiag)] = 1
    e, v = lib.davidson(hessian_x, x0, precond, tol=tol, verbose=log, nroots=nroots)
    log.info('ghf_stability: lowest eigs of H = %s', e)
    if nroots != 1:
        e, v = e[0], v[0]
    stable = not (e < -1e-5)
    dump_status(log, stable, f'{mf.__class__}', 'internal')
    if stable:
        mo = mf.mo_coeff
    else:
        mo = _rotate_mo(mf.mo_coeff, mf.mo_occ, v)
    if return_status:
        return mo, stable
    else:
        return mo

def dhf_stability(mf, verbose=None, return_status=False,
                  nroots=STAB_NROOTS, tol=STAB_TOL):
    '''
    Stability analysis for DHF/DKS method.

    Args:
        mf : DHF or DKS object

    Kwargs:
        return_status: bool
            Whether to return `stable_i` and `stable_e`
        nroots : int
            Number of roots solved by Davidson solver
        tol : float
            Convergence threshold for Davidson solver

    Returns:
        If return_status is False (default), the return value includes
        a new set of orbitals, which are more close to the stable condition.

        Else, another one boolean variable (indicating current status:
        stable or unstable) is returned.
    '''
    log = logger.new_logger(mf, verbose)
    g, hop, hdiag = newton_ah.gen_g_hop_dhf(mf, mf.mo_coeff, mf.mo_occ)
    hdiag *= 2
    stable = True
    def precond(dx, e, x0):
        hdiagd = hdiag - e
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return dx/hdiagd
    def hessian_x(x): # See comments in function rhf_internal
        return hop(x).real * 2

    x0 = numpy.zeros_like(g)
    x0[g!=0] = 1. / hdiag[g!=0]
    x0[numpy.argmin(hdiag)] = 1
    e, v = lib.davidson(hessian_x, x0, precond, tol=tol, verbose=log, nroots=nroots)
    log.info('dhf_stability: lowest eigs of H = %s', e)
    if nroots != 1:
        e, v = e[0], v[0]
    stable = not (e < -1e-5)
    dump_status(log, stable, f'{mf.__class__}', 'internal')
    if stable:
        mo = mf.mo_coeff
    else:
        mo = _rotate_mo(mf.mo_coeff, mf.mo_occ, v)
    if return_status:
        return mo, stable
    else:
        return mo

def rhf_internal(mf, with_symmetry=True, verbose=None, return_status=False,
                 nroots=STAB_NROOTS, tol=STAB_TOL):
    log = logger.new_logger(mf, verbose)
    g, hop, hdiag = newton_ah.gen_g_hop_rhf(mf, mf.mo_coeff, mf.mo_occ,
                                            with_symmetry=with_symmetry)
    hdiag *= 2
    def precond(dx, e, x0):
        hdiagd = hdiag - e
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return dx/hdiagd
    # The results of hop(x) corresponds to a displacement that reduces
    # gradients g.  It is the vir-occ block of the matrix vector product
    # (Hessian*x). The occ-vir block equals to x2.T.conj(). The overall
    # Hessian for internal rotation is x2 + x2.T.conj(). This is
    # the reason we apply (.real * 2) below
    def hessian_x(x):
        return hop(x).real * 2

    x0 = numpy.zeros_like(g)
    x0[g!=0] = 1. / hdiag[g!=0]
    if not with_symmetry:  # allow to break point group symmetry
        x0[numpy.argmin(hdiag)] = 1
    e, v = lib.davidson(hessian_x, x0, precond, tol=tol, verbose=log, nroots=nroots)
    log.info('rhf_internal: lowest eigs of H = %s', e)
    if nroots != 1:
        e, v = e[0], v[0]
    stable = not (e < -1e-5)
    dump_status(log, stable, f'{mf.__class__}', 'internal')
    if stable:
        mo = mf.mo_coeff
    else:
        mo = _rotate_mo(mf.mo_coeff, mf.mo_occ, v)
    if return_status:
        return mo, stable
    else:
        return mo

def _rotate_mo(mo_coeff, mo_occ, dx):
    dr = hf.unpack_uniq_var(dx, mo_occ)
    u = newton_ah.expmat(dr)
    return numpy.dot(mo_coeff, u)

def _gen_hop_rhf_external(mf, with_symmetry=True, verbose=None):
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]

    if with_symmetry and mol.symmetry:
        orbsym = hf_symm.get_orbsym(mol, mo_coeff)
        sym_forbid = orbsym[viridx].reshape(-1,1) != orbsym[occidx]

    h1e = mf.get_hcore()
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    fock_ao = h1e + mf.get_veff(mol, dm0)
    fock = reduce(numpy.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))
    foo = fock[occidx[:,None],occidx]
    fvv = fock[viridx[:,None],viridx]

    hdiag = fvv.diagonal().reshape(-1,1) - foo.diagonal()
    if with_symmetry and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = hdiag.ravel()

    vrespz = mf.gen_response(singlet=None, hermi=2)
    def hop_real2complex(x1):
        x1 = x1.reshape(nvir,nocc)
        if with_symmetry and mol.symmetry:
            x1 = x1.copy()
            x1[sym_forbid] = 0
        x2 = numpy.einsum('ps,sq->pq', fvv, x1)
        x2-= numpy.einsum('ps,rp->rs', foo, x1)

        d1 = reduce(numpy.dot, (orbv, x1*2, orbo.conj().T))
        dm1 = d1 - d1.conj().T
# No Coulomb and fxc contribution for anti-hermitian DM
        v1 = vrespz(dm1)
        x2 += reduce(numpy.dot, (orbv.conj().T, v1, orbo))
        if with_symmetry and mol.symmetry:
            x2[sym_forbid] = 0
        return x2.ravel()

    vresp1 = mf.gen_response(singlet=False, hermi=1)
    def hop_rhf2uhf(x1):
        # See also rhf.TDA triplet excitation
        x1 = x1.reshape(nvir,nocc)
        if with_symmetry and mol.symmetry:
            x1 = x1.copy()
            x1[sym_forbid] = 0
        x2 = numpy.einsum('ps,sq->pq', fvv, x1)
        x2-= numpy.einsum('ps,rp->rs', foo, x1)

        d1 = reduce(numpy.dot, (orbv, x1*2, orbo.conj().T))
        dm1 = d1 + d1.conj().T
        v1ao = vresp1(dm1)
        x2 += reduce(numpy.dot, (orbv.conj().T, v1ao, orbo))
        if with_symmetry and mol.symmetry:
            x2[sym_forbid] = 0
        return x2.real.ravel()

    return hop_real2complex, hdiag, hop_rhf2uhf, hdiag


def rhf_external(mf, with_symmetry=True, verbose=None, return_status=False,
                 nroots=STAB_NROOTS, tol=STAB_TOL):
    log = logger.new_logger(mf, verbose)
    hop1, hdiag1, hop2, hdiag2 = _gen_hop_rhf_external(mf, with_symmetry)

    def precond(dx, e, x0):
        hdiagd = hdiag1 - e
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return dx/hdiagd
    x0 = numpy.zeros_like(hdiag1)
    x0[hdiag1>1e-5] = 1. / hdiag1[hdiag1>1e-5]
    if not with_symmetry:  # allow to break point group symmetry
        x0[numpy.argmin(hdiag1)] = 1
    e1, v1 = lib.davidson(hop1, x0, precond, tol=tol, verbose=log, nroots=nroots)
    log.info('rhf_real2complex: lowest eigs of H = %s', e1)
    if nroots != 1:
        e1, v1 = e1[0], v1[0]
    stable1 = not (e1 < -1e-5)
    dump_status(log, stable1, f'{mf.__class__}', 'real -> complex')

    def precond(dx, e, x0):
        hdiagd = hdiag2 - e
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return dx/hdiagd
    x0 = numpy.zeros_like(hdiag2)
    x0[hdiag2>1e-5] = 1. / hdiag2[hdiag2>1e-5]
    e3, v3 = lib.davidson(hop2, x0, precond, tol=tol, verbose=log, nroots=nroots)
    log.info('rhf_external: lowest eigs of H = %s', e3)
    if nroots != 1:
        e3, v3 = e3[0], v3[0]
    stable = not (e3 < -1e-5)
    dump_status(log, stable, f'{mf.__class__}', 'RHF/RKS -> UHF/UKS')
    if stable:
        mo = (mf.mo_coeff, mf.mo_coeff)
    else:
        mo = (_rotate_mo(mf.mo_coeff, mf.mo_occ, v3), mf.mo_coeff)
    if return_status:
        return mo, stable
    else:
        return mo

def rohf_internal(mf, with_symmetry=True, verbose=None, return_status=False,
                  nroots=STAB_NROOTS, tol=STAB_TOL):
    log = logger.new_logger(mf, verbose)
    g, hop, hdiag = newton_ah.gen_g_hop_rohf(mf, mf.mo_coeff, mf.mo_occ,
                                             with_symmetry=with_symmetry)
    hdiag *= 2
    def precond(dx, e, x0):
        hdiagd = hdiag - e
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return dx/hdiagd
    def hessian_x(x): # See comments in function rhf_internal
        return hop(x).real * 2

    x0 = numpy.zeros_like(g)
    x0[g!=0] = 1. / hdiag[g!=0]
    if not with_symmetry:  # allow to break point group symmetry
        x0[numpy.argmin(hdiag)] = 1
    e, v = lib.davidson(hessian_x, x0, precond, tol=tol, verbose=log, nroots=nroots)
    log.info('rohf_internal: lowest eigs of H = %s', e)
    if nroots != 1:
        e, v = e[0], v[0]
    stable = not (e < -1e-5)
    dump_status(log, stable, f'{mf.__class__}', 'internal')
    if stable:
        mo = mf.mo_coeff
    else:
        mo = _rotate_mo(mf.mo_coeff, mf.mo_occ, v)
    if return_status:
        return mo, stable
    else:
        return mo

def rohf_external(mf, with_symmetry=True, verbose=None, return_status=False,
                  nroots=STAB_NROOTS, tol=STAB_TOL):
    raise NotImplementedError

def uhf_internal(mf, with_symmetry=True, verbose=None, return_status=False,
                 nroots=STAB_NROOTS, tol=STAB_TOL):
    log = logger.new_logger(mf, verbose)
    g, hop, hdiag = newton_ah.gen_g_hop_uhf(mf, mf.mo_coeff, mf.mo_occ,
                                            with_symmetry=with_symmetry)
    hdiag *= 2
    def precond(dx, e, x0):
        hdiagd = hdiag - e
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return dx/hdiagd
    def hessian_x(x): # See comments in function rhf_internal
        return hop(x).real * 2

    x0 = numpy.zeros_like(g)
    x0[g!=0] = 1. / hdiag[g!=0]
    if not with_symmetry:  # allow to break point group symmetry
        x0[numpy.argmin(hdiag)] = 1
    e, v = lib.davidson(hessian_x, x0, precond, tol=tol, verbose=log, nroots=nroots)
    log.info('uhf_internal: lowest eigs of H = %s', e)
    if nroots != 1:
        e, v = e[0], v[0]
    stable = not (e < -1e-5)
    dump_status(log, stable, f'{mf.__class__}', 'internal')
    if stable:
        mo = mf.mo_coeff
    else:
        nocca = numpy.count_nonzero(mf.mo_occ[0]> 0)
        nvira = numpy.count_nonzero(mf.mo_occ[0]==0)
        mo = (_rotate_mo(mf.mo_coeff[0], mf.mo_occ[0], v[:nocca*nvira]),
              _rotate_mo(mf.mo_coeff[1], mf.mo_occ[1], v[nocca*nvira:]))
    if return_status:
        return mo, stable
    else:
        return mo

def _gen_hop_uhf_external(mf, with_symmetry=True, verbose=None):
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
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
        sym_forbida = orbsyma[viridxa].reshape(-1,1) != orbsyma[occidxa]
        sym_forbidb = orbsymb[viridxb].reshape(-1,1) != orbsymb[occidxb]
        sym_forbid1 = numpy.hstack((sym_forbida.ravel(), sym_forbidb.ravel()))

    h1e = mf.get_hcore()
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    fock_ao = h1e + mf.get_veff(mol, dm0)
    focka = reduce(numpy.dot, (mo_coeff[0].conj().T, fock_ao[0], mo_coeff[0]))
    fockb = reduce(numpy.dot, (mo_coeff[1].conj().T, fock_ao[1], mo_coeff[1]))
    fooa = focka[occidxa[:,None],occidxa]
    fvva = focka[viridxa[:,None],viridxa]
    foob = fockb[occidxb[:,None],occidxb]
    fvvb = fockb[viridxb[:,None],viridxb]

    h_diaga =(focka[viridxa,viridxa].reshape(-1,1) - focka[occidxa,occidxa])
    h_diagb =(fockb[viridxb,viridxb].reshape(-1,1) - fockb[occidxb,occidxb])
    hdiag1 = numpy.hstack((h_diaga.reshape(-1), h_diagb.reshape(-1)))
    if with_symmetry and mol.symmetry:
        hdiag1[sym_forbid1] = 0

    vrespz = mf.gen_response(with_j=False, hermi=2)
    def hop_real2complex(x1):
        if with_symmetry and mol.symmetry:
            x1 = x1.copy()
            x1[sym_forbid1] = 0
        x1a = x1[:nvira*nocca].reshape(nvira,nocca)
        x1b = x1[nvira*nocca:].reshape(nvirb,noccb)
        x2a = numpy.einsum('pr,rq->pq', fvva, x1a)
        x2a-= numpy.einsum('sq,ps->pq', fooa, x1a)
        x2b = numpy.einsum('pr,rq->pq', fvvb, x1b)
        x2b-= numpy.einsum('qs,ps->pq', foob, x1b)

        d1a = reduce(numpy.dot, (orbva, x1a, orboa.conj().T))
        d1b = reduce(numpy.dot, (orbvb, x1b, orbob.conj().T))
        dm1 = numpy.array((d1a-d1a.conj().T, d1b-d1b.conj().T))

        v1 = vrespz(dm1)
        x2a += reduce(numpy.dot, (orbva.conj().T, v1[0], orboa))
        x2b += reduce(numpy.dot, (orbvb.conj().T, v1[1], orbob))
        x2 = numpy.hstack((x2a.ravel(), x2b.ravel()))
        if with_symmetry and mol.symmetry:
            x2[sym_forbid1] = 0
        return x2

    if with_symmetry and mol.symmetry:
        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
        sym_forbidab = orbsyma[viridxa].reshape(-1,1) != orbsymb[occidxb]
        sym_forbidba = orbsymb[viridxb].reshape(-1,1) != orbsyma[occidxa]
        sym_forbid2 = numpy.hstack((sym_forbidab.ravel(), sym_forbidba.ravel()))
    hdiagab = fvva.diagonal().reshape(-1,1) - foob.diagonal()
    hdiagba = fvvb.diagonal().reshape(-1,1) - fooa.diagonal()
    hdiag2 = numpy.hstack((hdiagab.ravel(), hdiagba.ravel()))
    if with_symmetry and mol.symmetry:
        hdiag2[sym_forbid2] = 0

    vresp1 = mf.gen_response(with_j=False, hermi=0)
    # Spin flip GHF solution is not considered
    def hop_uhf2ghf(x1):
        if with_symmetry and mol.symmetry:
            x1 = x1.copy()
            x1[sym_forbid2] = 0
        x1ab = x1[:nvira*noccb].reshape(nvira,noccb)
        x1ba = x1[nvira*noccb:].reshape(nvirb,nocca)
        x2ab = numpy.einsum('pr,rq->pq', fvva, x1ab)
        x2ab-= numpy.einsum('sq,ps->pq', foob, x1ab)
        x2ba = numpy.einsum('pr,rq->pq', fvvb, x1ba)
        x2ba-= numpy.einsum('qs,ps->pq', fooa, x1ba)

        d1ab = reduce(numpy.dot, (orbva, x1ab, orbob.conj().T))
        d1ba = reduce(numpy.dot, (orbvb, x1ba, orboa.conj().T))
        dm1 = numpy.array((d1ab+d1ba.conj().T, d1ba+d1ab.conj().T))
        v1 = vresp1(dm1)
        x2ab += reduce(numpy.dot, (orbva.conj().T, v1[0], orbob))
        x2ba += reduce(numpy.dot, (orbvb.conj().T, v1[1], orboa))
        x2 = numpy.hstack((x2ab.real.ravel(), x2ba.real.ravel()))
        if with_symmetry and mol.symmetry:
            x2[sym_forbid2] = 0
        return x2

    return hop_real2complex, hdiag1, hop_uhf2ghf, hdiag2


def uhf_external(mf, with_symmetry=True, verbose=None, return_status=False,
                 nroots=STAB_NROOTS, tol=STAB_TOL):
    log = logger.new_logger(mf, verbose)
    hop1, hdiag1, hop2, hdiag2 = _gen_hop_uhf_external(mf, with_symmetry)

    def precond(dx, e, x0):
        hdiagd = hdiag1 - e
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return dx/hdiagd
    x0 = numpy.zeros_like(hdiag1)
    x0[hdiag1>1e-5] = 1. / hdiag1[hdiag1>1e-5]
    if not with_symmetry:  # allow to break point group symmetry
        x0[numpy.argmin(hdiag1)] = 1
    e1, v = lib.davidson(hop1, x0, precond, tol=tol, verbose=log, nroots=nroots)
    log.info('uhf_real2complex: lowest eigs of H = %s', e1)
    if nroots != 1:
        e1, v = e1[0], v[0]
    stable1 = not (e1 < -1e-5)
    dump_status(log, stable1, f'{mf.__class__}', 'real -> complex')

    def precond(dx, e, x0):
        hdiagd = hdiag2 - e
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return dx/hdiagd
    x0 = numpy.zeros_like(hdiag2)
    x0[hdiag2>1e-5] = 1. / hdiag2[hdiag2>1e-5]
    if not with_symmetry:  # allow to break point group symmetry
        x0[numpy.argmin(hdiag2)] = 1
    e3, v = lib.davidson(hop2, x0, precond, tol=tol, verbose=log, nroots=nroots)
    log.info('uhf_external: lowest eigs of H = %s', e3)
    if nroots != 1:
        e3, v = e3[0], v[0]
    stable = not (e3 < -1e-5)
    dump_status(log, stable, f'{mf.__class__}', 'UHF/UKS -> GHF/GKS')
    mo = scipy.linalg.block_diag(*mf.mo_coeff)
    if not stable:
        occidxa = numpy.where(mf.mo_occ[0]> 0)[0]
        viridxa = numpy.where(mf.mo_occ[0]==0)[0]
        occidxb = numpy.where(mf.mo_occ[1]> 0)[0]
        viridxb = numpy.where(mf.mo_occ[1]==0)[0]
        nocca = len(occidxa)
        nvira = len(viridxa)
        noccb = len(occidxb)
        nvirb = len(viridxb)
        nmo = nocca + nvira
        dx = numpy.zeros((nmo*2,nmo*2))
        dx[viridxa[:,None],nmo+occidxb] = v[:nvira*noccb].reshape(nvira,noccb)
        dx[nmo+viridxb[:,None],occidxa] = v[nvira*noccb:].reshape(nvirb,nocca)
        u = newton_ah.expmat(dx - dx.conj().T)
        mo = numpy.dot(mo, u)
        mo = numpy.hstack([mo[:,:nocca], mo[:,nmo:nmo+noccb],
                           mo[:,nocca:nmo], mo[:,nmo+noccb:]])
    if return_status:
        return mo, stable
    else:
        return mo


if __name__ == '__main__':
    from pyscf import gto, scf, dft
    mol = gto.M(atom='O 0 0 0; O 0 0 1.2222', basis='631g*')
    mf = scf.RHF(mol).run()
    rhf_stability(mf, True, True, verbose=4)

    mf = dft.RKS(mol).run(level_shift=.2)
    rhf_stability(mf, True, True, verbose=4)

    mf = scf.UHF(mol).run()
    mo1 = uhf_stability(mf, True, True, verbose=4)[0]

    mf = scf.newton(mf).run(mo1, mf.mo_occ)
    uhf_stability(mf, True, False, verbose=4)
    mf = scf.newton(scf.UHF(mol)).run()
    uhf_stability(mf, True, False, verbose=4)

    mol.spin = 2
    mf = scf.UHF(mol).run()
    uhf_stability(mf, True, True, verbose=4)

    mf = dft.UKS(mol).run()
    uhf_stability(mf, True, True, verbose=4)

    mol = gto.M(atom='''
O1
O2  1  1.2227
O3  1  1.2227  2  114.0451
                ''', basis = '631g*')
    mf = scf.RHF(mol).run()
    rhf_stability(mf, True, True, verbose=4)

    mf = scf.UHF(mol).run()
    mo1 = uhf_stability(mf, True, True, verbose=4)[0]

    mf = scf.newton(scf.UHF(mol)).run()
    uhf_stability(mf, True, True, verbose=4)
