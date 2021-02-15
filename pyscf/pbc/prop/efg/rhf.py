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

'''
Electric field gradients, nuclear quadrupolar coupling and Mossbauer
spectroscopy for non-relativistic (or sf-x2c) mean-field and post-HF methods.
See also pyscf/prop/efg/rhf.py

Ref:

[1] H. Petrilli, P. Blochl, P. Blaha, and K. Schwarz. Phys. Rev. B, 57, 14690 (1998)

[2] H. Akai et al. Prog. Theor. Phys. Suppl. 101, 11 (1990)
'''

import time
import numpy
import scipy.special
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import scf
from pyscf.pbc import tools
from pyscf.pbc import df
from pyscf.pbc.df import aft
from pyscf.prop.efg import rhf as rhf_efg

def kernel(method, efg_nuc=None):
    log = lib.logger.Logger(method.stdout, method.verbose)
    cell = method.cell
    if efg_nuc is None:
        efg_nuc = range(cell.natm)

    dm = method.make_rdm1()
    if isinstance(method, scf.khf.KSCF):
        if isinstance(dm[0][0], numpy.ndarray) and dm[0][0].ndim == 2:
            dm = dm[0] + dm[1]  # KUHF density matrix
    elif isinstance(method, scf.hf.SCF):
        if isinstance(dm[0], numpy.ndarray) and dm[0].ndim == 2:
            dm = dm[0] + dm[1]  # UHF density matrix
    else:
        mo = method.mo_coeff
        if isinstance(dm[0][0], numpy.ndarray) and dm[0][0].ndim == 2:
            dm_a = [lib.einsum('pi,ij,qj->pq', c, dm[0][k], c.conj())
                    for k, c in enumerate(mo)]
            dm_b = [lib.einsum('pi,ij,qj->pq', c, dm[1][k], c.conj())
                    for k, c in enumerate(mo)]
            dm = lib.asarray(dm_a) + lib.asarray(dm_b)
        else:
            dm = lib.asarray([lib.einsum('pi,ij,qj->pq', c, dm[k], c.conj())
                              for k, c in enumerate(mo)])

    if isinstance(method, scf.hf.SCF):
        with_df = getattr(method, 'with_df', None)
        with_x2c = getattr(method, 'with_x2c', None)
    else:
        with_df = getattr(method._scf, 'with_df', None)
        with_x2c = getattr(method._scf, 'with_x2c', None)
    if with_x2c:
        raise NotImplementedError

    log.info('\nElectric Field Gradient Tensor Results')
    if isinstance(with_df, df.fft.FFTDF):
        efg_e = _fft_quad_integrals(with_df, dm, efg_nuc)
    else:
        efg_e = _aft_quad_integrals(with_df, dm, efg_nuc)
    efg = []
    for i, atm_id in enumerate(efg_nuc):
        efg_nuc = _get_quad_nuc(cell, atm_id)
        v = efg_nuc - efg_e[i]
        efg.append(v)

        rhf_efg._analyze(cell, atm_id, v, log)

    return numpy.asarray(efg)

EFG = kernel


def _get_quad_nuc(cell, atm_id):
    ew_eta = cell.ew_eta
    ew_cut = cell.ew_cut
    chargs = cell.atom_charges()
    coords = cell.atom_coords()
    Lall = cell.get_lattice_Ls(rcut=ew_cut)

    rLij = coords[atm_id,:] - coords + Lall[:,None,:]
    rr = numpy.einsum('Ljx,Ljy->Ljxy', rLij, rLij)
    r = numpy.sqrt(numpy.einsum('Ljxx->Lj', rr))
    r[r<1e-16] = 1e60
    idx = numpy.arange(3)
    erfc_part = scipy.special.erfc(ew_eta * r) / r**5
    ewovrl = 3 * chargs[atm_id] * numpy.einsum('Ljxy,j,Lj->xy', rr, chargs, erfc_part)
    ewovrl[idx,idx] -= ewovrl.trace() / 3

    exp_part = numpy.exp(-ew_eta**2 * r**2) / r**4
    ewovrl_part = (2./numpy.sqrt(numpy.pi) * ew_eta * chargs[atm_id] *
                   numpy.einsum('Ljxy,j,Lj->xy', rr, chargs, exp_part))
    ewovrl += ewovrl_part

    ewovrl += 2*ewovrl_part
    ewovrl[idx,idx] -= ewovrl_part.trace()

    exp_part = numpy.exp(-ew_eta**2 * r**2) / r**2
    ewovrl += (4./numpy.sqrt(numpy.pi) * ew_eta**3 * chargs[atm_id] *
               numpy.einsum('Ljxy,j,Lj->xy', rr, chargs, exp_part))
    # Fermi contact term
    ewovrl_fc = 4./3*ew_eta**3/numpy.pi**.5 * numpy.exp(-ew_eta**2*r**2)
    ewovrl[idx,idx] -= numpy.einsum('j,Lj->', chargs, ewovrl_fc)

    mesh = gto.cell._cut_mesh_for_ewald(cell, cell.mesh)
    Gv, Gvbase, weights = cell.get_Gv_weights(mesh)
    GG = numpy.einsum('gx,gy->gxy', Gv, Gv)
    absG2 = numpy.einsum('gxx->g', GG)
    # Corresponding to FC term, that makes the tensor zero trace
    idx = numpy.arange(3)
    GG[:,idx,idx] -= 1./3 * absG2[:,None]

    absG2[absG2==0] = 1e200
    coulG = 4*numpy.pi / absG2
    coulG *= weights
    expG2 = numpy.exp(-absG2/(4*ew_eta**2))
    coulG *= expG2
    SI = cell.get_SI(Gv)
    ZSI = numpy.einsum("i,ij->j", chargs, SI)
    ewg =-chargs[atm_id] * numpy.einsum('ixy,i,i,i->xy',
                                        GG, SI[atm_id].conj(), ZSI, coulG).real
    return ewovrl + ewg


def _fft_quad_integrals(mydf, dm, efg_nuc):
    # Use FFTDF to compute the integrals of quadrupole operator 
    # (3 \vec{r} \vec{r} - r^2) / r^5
    cell = mydf.cell
    if cell.dimension != 3:
        raise NotImplementedError

    mesh = mydf.mesh
    kpts = mydf.kpts
    kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)
    nao = cell.nao_nr()
    dm_kpts = dm.reshape((nkpts,nao,nao), order='C')

    ni = mydf._numint
    hermi = 1
    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dm_kpts, hermi)
    ngrids = numpy.prod(mesh)
    rhoR = numpy.zeros(ngrids)
    for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts_lst):
        ao_ks, mask = ao_ks_etc[0], ao_ks_etc[2]
        rhoR[p0:p1] += make_rho(0, ao_ks, mask, 'LDA')
        ao_ks = None
    rhoG = tools.fft(rhoR, mesh)

    Gv = cell.get_Gv(mesh)
    coulG = tools.get_coulG(cell, mesh=mesh, Gv=Gv)
    GG = numpy.einsum('gx,gy->gxy', Gv, Gv)
    absG2 = numpy.einsum('gxx->g', GG)

    # Corresponding to FC term, that makes the tensor traceless
    idx = numpy.arange(3)
    GG[:,idx,idx] -= 1./3 * absG2[:,None]

    vG = 1./ngrids * numpy.einsum('g,g,gxy->gxy', rhoG, coulG, GG)
    SI = cell.get_SI(Gv)
    efg_e = lib.einsum('zg,gxy->zxy', SI[efg_nuc], vG.conj()).real
    return efg_e

def _aft_quad_integrals(mydf, dm, efg_nuc):
    # Use AFTDF to compute the integrals of quadrupole operator 
    # (3 \vec{r} \vec{r} - r^2) / r^5
    cell = mydf.cell
    if cell.dimension != 3:
        raise NotImplementedError

    log = lib.logger.new_logger(mydf)
    t0 = t1 = (time.clock(), time.time())

    mesh = mydf.mesh
    kpts = mydf.kpts
    kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)
    nao = cell.nao_nr()
    dm_kpts = dm.reshape((nkpts,nao,nao), order='C')

    ngrids = numpy.prod(mesh)
    rhoG = numpy.zeros(ngrids, dtype=numpy.complex128)
    kpt_allow = numpy.zeros(3)
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    for aoaoks, p0, p1 in mydf.ft_loop(mesh, kpt_allow, kpts_lst,
                                       max_memory=max_memory, aosym='s1'):
        rhoG[p0:p1] = numpy.einsum('kgpq,kqp->g', aoaoks.reshape(nkpts,p1-p0,nao,nao),
                                   dm_kpts)
        t1 = log.timer_debug1('contracting Vnuc [%s:%s]'%(p0, p1), *t1)
    t0 = log.timer_debug1('contracting Vnuc', *t0)
    rhoG *= 1./nkpts

    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    coulG = tools.get_coulG(cell, mesh=mesh, Gv=Gv)
    GG = numpy.einsum('gx,gy->gxy', Gv, Gv)
    absG2 = numpy.einsum('gxx->g', GG)

    # Corresponding to FC term, that makes the tensor traceless
    idx = numpy.arange(3)
    GG[:,idx,idx] -= 1./3 * absG2[:,None]
    vG = 1./cell.vol * numpy.einsum('g,g,gxy->gxy', rhoG, coulG, GG)

    if mydf.eta == 0:
        SI = cell.get_SI(Gv)
        efg_e = numpy.einsum('zg,gxy->zxy', SI[efg_nuc], vG.conj()).real

    else:
        nuccell = aft._compensate_nuccell(mydf)
        # PP-loc part1 is handled by fakenuc in _int_nuc_vloc
        efg_e = _int_nuc_vloc(mydf, nuccell, kpts_lst, dm_kpts)
        t0 = log.timer_debug1('vnuc pass1: analytic int', *t0)

        aoaux = df.ft_ao.ft_ao(nuccell, Gv)
        efg_e += numpy.einsum('gz,gxy->zxy', aoaux[:,efg_nuc], vG.conj()).real
    return efg_e


def _int_nuc_vloc(mydf, nuccell, kpts, dm_kpts):
    '''Vnuc - Vloc'''
    cell = mydf.cell
    nkpts = len(kpts)

    # Use the 3c2e code with steep s gaussians to mimic nuclear density
    fakenuc = aft._fake_nuc(cell)
    fakenuc._atm, fakenuc._bas, fakenuc._env = \
            gto.conc_env(nuccell._atm, nuccell._bas, nuccell._env,
                         fakenuc._atm, fakenuc._bas, fakenuc._env)

    kptij_lst = numpy.hstack((kpts,kpts)).reshape(-1,2,3)
    v3c = df.incore.aux_e2(cell, fakenuc, 'int3c2e_ipip1', aosym='s1', comp=9,
                           kptij_lst=kptij_lst)
    v3c += df.incore.aux_e2(cell, fakenuc, 'int3c2e_ipvip1', aosym='s1', comp=9,
                            kptij_lst=kptij_lst)

    nao = cell.nao_nr()
    natm = cell.natm
    v3c = v3c.reshape(nkpts,3,3,nao,nao,natm*2)
    efg_loc = 1./nkpts * numpy.einsum('kxypqz,kqp->zxy', v3c, dm_kpts)
    efg_loc+= 1./nkpts * numpy.einsum('kyxqpz,kqp->zxy', v3c, dm_kpts)
    v3c = None

    # Fermi contact
    fc = df.incore.aux_e2(cell, fakenuc, 'int3c1e', aosym='s1', kptij_lst=kptij_lst)
    fc = fc.reshape(nkpts,nao,nao,natm*2)
    vfc = 1./nkpts * numpy.einsum('kpqz,kqp->z', fc, dm_kpts)
    for i in range(3):
        efg_loc[:,i,i] += 4./3*numpy.pi * vfc

    nuc_part = efg_loc[:natm]
    modchg_part = efg_loc[natm:]
    efg_loc = nuc_part - modchg_part

    if cell.dimension == 3:
        fac = numpy.pi/cell.vol
        nucbar = numpy.array([fac/nuccell.bas_exp(i)[0] for i in range(natm)])

        ipip = cell.pbc_intor('int1e_ipipovlp', 9, lib.HERMITIAN, kpts)
        ipvip = cell.pbc_intor('int1e_ipovlpip', 9, lib.HERMITIAN, kpts)
        d2rho_bar = 0
        for k in range(nkpts):
            v = (ipip[k] + ipvip[k]).reshape(3,3,nao,nao)
            d2rho_bar += numpy.einsum('xypq,qp->xy', v, dm_kpts[k])
            d2rho_bar += numpy.einsum('yxqp,qp->xy', v, dm_kpts[k])
        d2rho_bar *= 1./nkpts

        efg_loc -= numpy.einsum('z,xy->zxy', nucbar, d2rho_bar)
    return efg_loc.real


if __name__ == '__main__':
    cell = gto.Cell()
    cell.verbose = 4
    cell.atom = 'H 1 0.8 0; H 0. 1.3 0'
    cell.a = numpy.eye(3) * 3
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    #cell.mesh = [24]*3
    cell.build()
    mf = scf.RHF(cell).run()
    v = EFG(mf)
