# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy
import numpy as np

from pyscf.pbc import gto as pgto
from pyscf.pbc.df import fft, aft


##################################################
#
# port from ao2mo/eris.py
#
##################################################
from pyscf import lib
from pyscf.pbc import lib as pbclib
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts_helper

#einsum = np.einsum
einsum = lib.einsum

r"""
    (ij|kl) = \int dr1 dr2 i*(r1) j(r1) v(r12) k*(r2) l(r2)
            = (ij|G) v(G) (G|kl)

    i*(r) j(r) = 1/N \sum_G e^{iGr}  (G|ij)
               = 1/N \sum_G e^{-iGr} (ij|G)

    "forward" FFT:
        (G|ij) = \sum_r e^{-iGr} i*(r) j(r) = fft[ i*(r) j(r) ]
    "inverse" FFT:
        (ij|G) = \sum_r e^{iGr} i*(r) j(r) = N * ifft[ i*(r) j(r) ]
               = conj[ \sum_r e^{-iGr} j*(r) i(r) ]
"""

def general(cell, mo_coeffs, kpts=None, compact=0):
    '''pyscf-style wrapper to get MO 2-el integrals.'''
    assert len(mo_coeffs) == 4
    if kpts is not None:
        assert len(kpts) == 4
    return get_mo_eri(cell, mo_coeffs, kpts)

def get_mo_eri(cell, mo_coeffs, kpts=None):
    '''Convenience function to return MO 2-el integrals.'''
    mo_coeff12 = mo_coeffs[:2]
    mo_coeff34 = mo_coeffs[2:]
    if kpts is None:
        kpts12 = kpts34 = q = None
    else:
        kpts12 = kpts[:2]
        kpts34 = kpts[2:]
        q = kpts12[0] - kpts12[1]
        #q = kpts34[1] - kpts34[0]
    if q is None:
        q = np.zeros(3)

    mo_pairs12_kG = get_mo_pairs_G(cell, mo_coeff12, kpts12)
    mo_pairs34_invkG = get_mo_pairs_invG(cell, mo_coeff34, kpts34, q)
    return assemble_eri(cell, mo_pairs12_kG, mo_pairs34_invkG, q)

def get_mo_pairs_G(cell, mo_coeffs, kpts=None, q=None):
    '''Calculate forward (G|ij) FFT of all MO pairs.

    TODO: - Implement simplifications for real orbitals.

    Args:
        mo_coeff: length-2 list of (nao,nmo) ndarrays
            The two sets of MO coefficients to use in calculating the
            product |ij).

    Returns:
        mo_pairs_G : (ngrids, nmoi*nmoj) ndarray
            The FFT of the real-space MO pairs.
    '''
    coords = gen_grid.gen_uniform_grids(cell)
    if kpts is None:
        q = np.zeros(3)
        aoR = numint.eval_ao(cell, coords)
        ngrids = aoR.shape[0]

        if np.array_equal(mo_coeffs[0], mo_coeffs[1]):
            nmoi = nmoj = mo_coeffs[0].shape[1]
            moiR = mojR = einsum('ri,ia->ra', aoR, mo_coeffs[0])
        else:
            nmoi = mo_coeffs[0].shape[1]
            nmoj = mo_coeffs[1].shape[1]
            moiR = einsum('ri,ia->ra', aoR, mo_coeffs[0])
            mojR = einsum('ri,ia->ra', aoR, mo_coeffs[1])

    else:
        if q is None:
            q = kpts[1]-kpts[0]
        aoR_ki = numint.eval_ao(cell, coords, kpt=kpts[0])
        aoR_kj = numint.eval_ao(cell, coords, kpt=kpts[1])
        ngrids = aoR_ki.shape[0]

        nmoi = mo_coeffs[0].shape[1]
        nmoj = mo_coeffs[1].shape[1]
        moiR = einsum('ri,ia->ra', aoR_ki, mo_coeffs[0])
        mojR = einsum('ri,ia->ra', aoR_kj, mo_coeffs[1])

    #mo_pairs_R = einsum('ri,rj->rij', np.conj(moiR), mojR)
    mo_pairs_G = np.zeros([ngrids,nmoi*nmoj], np.complex128)

    fac = np.exp(-1j*np.dot(coords, q))
    for i in range(nmoi):
        for j in range(nmoj):
            mo_pairs_R_ij = np.conj(moiR[:,i])*mojR[:,j]
            mo_pairs_G[:,i*nmoj+j] = tools.fftk(mo_pairs_R_ij, cell.mesh, fac)

    return mo_pairs_G

def get_mo_pairs_invG(cell, mo_coeffs, kpts=None, q=None):
    '''Calculate "inverse" (ij|G) FFT of all MO pairs.

    TODO: - Implement simplifications for real orbitals.

    Args:
        mo_coeff: length-2 list of (nao,nmo) ndarrays
            The two sets of MO coefficients to use in calculating the
            product |ij).

    Returns:
        mo_pairs_invG : (ngrids, nmoi*nmoj) ndarray
            The inverse FFTs of the real-space MO pairs.
    '''
    coords = gen_grid.gen_uniform_grids(cell)
    if kpts is None:
        q = np.zeros(3)
        aoR = numint.eval_ao(cell, coords)
        ngrids = aoR.shape[0]

        if np.array_equal(mo_coeffs[0], mo_coeffs[1]):
            nmoi = nmoj = mo_coeffs[0].shape[1]
            moiR = mojR = einsum('ri,ia->ra', aoR, mo_coeffs[0])
        else:
            nmoi = mo_coeffs[0].shape[1]
            nmoj = mo_coeffs[1].shape[1]
            moiR = einsum('ri,ia->ra', aoR, mo_coeffs[0])
            mojR = einsum('ri,ia->ra', aoR, mo_coeffs[1])

    else:
        if q is None:
            q = kpts[1]-kpts[0]
        aoR_ki = numint.eval_ao(cell, coords, kpt=kpts[0])
        aoR_kj = numint.eval_ao(cell, coords, kpt=kpts[1])
        ngrids = aoR_ki.shape[0]

        nmoi = mo_coeffs[0].shape[1]
        nmoj = mo_coeffs[1].shape[1]
        moiR = einsum('ri,ia->ra', aoR_ki, mo_coeffs[0])
        mojR = einsum('ri,ia->ra', aoR_kj, mo_coeffs[1])

    #mo_pairs_R = einsum('ri,rj->rij', np.conj(moiR), mojR)
    mo_pairs_invG = np.zeros([ngrids,nmoi*nmoj], np.complex128)

    fac = np.exp(1j*np.dot(coords, q))
    for i in range(nmoi):
        for j in range(nmoj):
            mo_pairs_R_ij = np.conj(moiR[:,i])*mojR[:,j]
            mo_pairs_invG[:,i*nmoj+j] = np.conj(tools.fftk(np.conj(mo_pairs_R_ij), cell.mesh, fac))

    return mo_pairs_invG

def get_mo_pairs_G_old(cell, mo_coeffs, kpts=None, q=None):
    '''Calculate forward (G|ij) and "inverse" (ij|G) FFT of all MO pairs.

    TODO: - Implement simplifications for real orbitals.

    Args:
        mo_coeff: length-2 list of (nao,nmo) ndarrays
            The two sets of MO coefficients to use in calculating the
            product |ij).

    Returns:
        mo_pairs_G, mo_pairs_invG : (ngrids, nmoi*nmoj) ndarray
            The FFTs of the real-space MO pairs.
    '''
    coords = gen_grid.gen_uniform_grids(cell)
    if kpts is None:
        q = np.zeros(3)
        aoR = numint.eval_ao(cell, coords)
        ngrids = aoR.shape[0]

        if np.array_equal(mo_coeffs[0], mo_coeffs[1]):
            nmoi = nmoj = mo_coeffs[0].shape[1]
            moiR = mojR = einsum('ri,ia->ra', aoR, mo_coeffs[0])
        else:
            nmoi = mo_coeffs[0].shape[1]
            nmoj = mo_coeffs[1].shape[1]
            moiR = einsum('ri,ia->ra', aoR, mo_coeffs[0])
            mojR = einsum('ri,ia->ra', aoR, mo_coeffs[1])

    else:
        if q is None:
            q = kpts[1]-kpts[0]
        aoR_ki = numint.eval_ao(cell, coords, kpt=kpts[0])
        aoR_kj = numint.eval_ao(cell, coords, kpt=kpts[1])
        ngrids = aoR_ki.shape[0]

        nmoi = mo_coeffs[0].shape[1]
        nmoj = mo_coeffs[1].shape[1]
        moiR = einsum('ri,ia->ra', aoR_ki, mo_coeffs[0])
        mojR = einsum('ri,ia->ra', aoR_kj, mo_coeffs[1])

    mo_pairs_R = np.einsum('ri,rj->rij', np.conj(moiR), mojR)
    mo_pairs_G = np.zeros([ngrids,nmoi*nmoj], np.complex128)
    mo_pairs_invG = np.zeros([ngrids,nmoi*nmoj], np.complex128)

    fac = np.exp(-1j*np.dot(coords, q))
    for i in range(nmoi):
        for j in range(nmoj):
            mo_pairs_G[:,i*nmoj+j] = tools.fftk(mo_pairs_R[:,i,j], cell.mesh, fac)
            mo_pairs_invG[:,i*nmoj+j] = np.conj(tools.fftk(np.conj(mo_pairs_R[:,i,j]), cell.mesh,
                                                                   fac.conj()))

    return mo_pairs_G, mo_pairs_invG

def assemble_eri(cell, orb_pair_invG1, orb_pair_G2, q=None):
    '''Assemble 4-index electron repulsion integrals.

    Returns:
        (nmo1*nmo2, nmo3*nmo4) ndarray

    '''
    if q is None:
        q = np.zeros(3)

    coulqG = tools.get_coulG(cell, -1.0*q)
    ngrids = orb_pair_invG1.shape[0]
    Jorb_pair_G2 = np.einsum('g,gn->gn',coulqG,orb_pair_G2)*(cell.vol/ngrids**2)
    eri = np.dot(orb_pair_invG1.T, Jorb_pair_G2)
    return eri

def get_ao_pairs_G(cell, kpt=np.zeros(3)):
    '''Calculate forward (G|ij) and "inverse" (ij|G) FFT of all AO pairs.

    Args:
        cell : instance of :class:`Cell`

    Returns:
        ao_pairs_G, ao_pairs_invG : (ngrids, nao*(nao+1)/2) ndarray
            The FFTs of the real-space AO pairs.

    '''
    coords = gen_grid.gen_uniform_grids(cell)
    aoR = numint.eval_ao(cell, coords, kpt) # shape = (coords, nao)
    ngrids, nao = aoR.shape
    gamma_point = abs(kpt).sum() < 1e-9
    if gamma_point:
        npair = nao*(nao+1)//2
        ao_pairs_G = np.empty([ngrids, npair], np.complex128)

        ij = 0
        for i in range(nao):
            for j in range(i+1):
                ao_ij_R = np.conj(aoR[:,i]) * aoR[:,j]
                ao_pairs_G[:,ij] = tools.fft(ao_ij_R, cell.mesh)
                #ao_pairs_invG[:,ij] = ngrids*tools.ifft(ao_ij_R, cell.mesh)
                ij += 1
        ao_pairs_invG = ao_pairs_G.conj()
    else:
        ao_pairs_G = np.zeros([ngrids, nao,nao], np.complex128)
        for i in range(nao):
            for j in range(nao):
                ao_ij_R = np.conj(aoR[:,i]) * aoR[:,j]
                ao_pairs_G[:,i,j] = tools.fft(ao_ij_R, cell.mesh)
        ao_pairs_invG = ao_pairs_G.transpose(0,2,1).conj().reshape(-1,nao**2)
        ao_pairs_G = ao_pairs_G.reshape(-1,nao**2)
    return ao_pairs_G, ao_pairs_invG

def get_ao_eri(cell, kpt=np.zeros(3)):
    '''Convenience function to return AO 2-el integrals.'''

    ao_pairs_G, ao_pairs_invG = get_ao_pairs_G(cell, kpt)
    eri = assemble_eri(cell, ao_pairs_invG, ao_pairs_G)
    if abs(kpt).sum() < 1e-9:
        eri = eri.real
    return eri

##################################################
#
# ao2mo/eris.py end
#
##################################################



##################################################
#
# port from scf/hf.py
#
##################################################
from pyscf.pbc import scf as pbcscf

def get_j(cell, dm, hermi=1, vhfopt=None, kpt=np.zeros(3), kpts_band=None):
    dm = np.asarray(dm)
    nao = dm.shape[-1]

    coords = gen_grid.gen_uniform_grids(cell)
    if kpts_band is None:
        kpt1 = kpt2 = kpt
        aoR_k1 = aoR_k2 = numint.eval_ao(cell, coords, kpt)
    else:
        kpt1 = kpts_band
        kpt2 = kpt
        aoR_k1 = numint.eval_ao(cell, coords, kpt1)
        aoR_k2 = numint.eval_ao(cell, coords, kpt2)
    ngrids, nao = aoR_k1.shape

    def contract(dm):
        vjR_k2 = get_vjR(cell, dm, aoR_k2)
        vj = (cell.vol/ngrids) * np.dot(aoR_k1.T.conj(), vjR_k2.reshape(-1,1)*aoR_k1)
        return vj

    if dm.ndim == 2:
        vj = contract(dm)
    else:
        vj = lib.asarray([contract(x) for x in dm.reshape(-1,nao,nao)])
    return vj.reshape(dm.shape)


def get_jk(mf, cell, dm, hermi=1, vhfopt=None, kpt=np.zeros(3), kpts_band=None):
    dm = np.asarray(dm)
    nao = dm.shape[-1]

    coords = gen_grid.gen_uniform_grids(cell)
    if kpts_band is None:
        kpt1 = kpt2 = kpt
        aoR_k1 = aoR_k2 = numint.eval_ao(cell, coords, kpt)
    else:
        kpt1 = kpts_band
        kpt2 = kpt
        aoR_k1 = numint.eval_ao(cell, coords, kpt1)
        aoR_k2 = numint.eval_ao(cell, coords, kpt2)

    vkR_k1k2 = get_vkR(mf, cell, aoR_k1, aoR_k2, kpt1, kpt2)

    ngrids, nao = aoR_k1.shape
    def contract(dm):
        vjR_k2 = get_vjR(cell, dm, aoR_k2)
        vj = (cell.vol/ngrids) * np.dot(aoR_k1.T.conj(), vjR_k2.reshape(-1,1)*aoR_k1)

        #:vk = (cell.vol/ngrids) * np.einsum('rs,Rp,Rqs,Rr->pq', dm, aoR_k1.conj(),
        #:                                vkR_k1k2, aoR_k2)
        aoR_dm_k2 = np.dot(aoR_k2, dm)
        tmp_Rq = np.einsum('Rqs,Rs->Rq', vkR_k1k2, aoR_dm_k2)
        vk = (cell.vol/ngrids) * np.dot(aoR_k1.T.conj(), tmp_Rq)
        return vj, vk

    if dm.ndim == 2:
        vj, vk = contract(dm)
    else:
        jk = [contract(x) for x in dm.reshape(-1,nao,nao)]
        vj = lib.asarray([x[0] for x in jk])
        vk = lib.asarray([x[1] for x in jk])
    return vj.reshape(dm.shape), vk.reshape(dm.shape)


def get_vjR(cell, dm, aoR):
    coulG = tools.get_coulG(cell)

    rhoR = numint.eval_rho(cell, aoR, dm)
    rhoG = tools.fft(rhoR, cell.mesh)

    vG = coulG*rhoG
    vR = tools.ifft(vG, cell.mesh)
    if rhoR.dtype == np.double:
        vR = vR.real
    return vR


def get_vkR(mf, cell, aoR_k1, aoR_k2, kpt1, kpt2):
    '''Get the real-space 2-index "exchange" potential V_{i,k1; j,k2}(r)
    where {i,k1} = exp^{i k1 r) |i> , {j,k2} = exp^{-i k2 r) <j|
    '''
    coords = gen_grid.gen_uniform_grids(cell)
    ngrids, nao = aoR_k1.shape

    expmikr = np.exp(-1j*np.dot(kpt1-kpt2,coords.T))
    coulG = tools.get_coulG(cell, kpt1-kpt2, exx=True, mf=mf)
    def prod(ij):
        i, j = divmod(ij, nao)
        rhoR = aoR_k1[:,i] * aoR_k2[:,j].conj()
        rhoG = tools.fftk(rhoR, cell.mesh, expmikr)
        vG = coulG*rhoG
        vR = tools.ifftk(vG, cell.mesh, expmikr.conj())
        return vR

    if aoR_k1.dtype == np.double and aoR_k2.dtype == np.double:
        vR = numpy.asarray([prod(ij).real for ij in range(nao**2)])
    else:
        vR = numpy.asarray([prod(ij) for ij in range(nao**2)])
    return vR.reshape(nao,nao,-1).transpose(2,0,1)


def get_j_kpts(mf, cell, dm_kpts, kpts, kpts_band=None):
    coords = gen_grid.gen_uniform_grids(cell)
    nkpts = len(kpts)
    ngrids = len(coords)
    dm_kpts = np.asarray(dm_kpts)
    nao = dm_kpts.shape[-1]

    ni = numint.KNumInt(kpts)
    aoR_kpts = ni.eval_ao(cell, coords, kpts)
    if kpts_band is not None:
        aoR_kband = numint.eval_ao(cell, coords, kpts_band)

    dms = dm_kpts.reshape(-1,nkpts,nao,nao)
    nset = dms.shape[0]

    vjR = [get_vjR(cell, dms[i], aoR_kpts) for i in range(nset)]
    if kpts_band is not None:
        vj_kpts = [cell.vol/ngrids * lib.dot(aoR_kband.T.conj()*vjR[i], aoR_kband)
                   for i in range(nset)]
        if dm_kpts.ndim == 3:  # One set of dm_kpts for KRHF
            vj_kpts = vj_kpts[0]
        return lib.asarray(vj_kpts)
    else:
        vj_kpts = []
        for i in range(nset):
            vj = [cell.vol/ngrids * lib.dot(aoR_k.T.conj()*vjR[i], aoR_k)
                  for aoR_k in aoR_kpts]
            vj_kpts.append(lib.asarray(vj))
        return lib.asarray(vj_kpts).reshape(dm_kpts.shape)


def get_jk_kpts(mf, cell, dm_kpts, kpts, kpts_band=None):
    coords = gen_grid.gen_uniform_grids(cell)
    nkpts = len(kpts)
    ngrids = len(coords)
    dm_kpts = np.asarray(dm_kpts)
    nao = dm_kpts.shape[-1]

    dms = dm_kpts.reshape(-1,nkpts,nao,nao)
    nset = dms.shape[0]

    ni = numint.KNumInt(kpts)
    aoR_kpts = ni.eval_ao(cell, coords, kpts)
    if kpts_band is not None:
        aoR_kband = numint.eval_ao(cell, coords, kpts_band)

# J
    vjR = [get_vjR_kpts(cell, dms[i], aoR_kpts) for i in range(nset)]
    if kpts_band is not None:
        vj_kpts = [cell.vol/ngrids * lib.dot(aoR_kband.T.conj()*vjR[i], aoR_kband)
                   for i in range(nset)]
    else:
        vj_kpts = []
        for i in range(nset):
            vj = [cell.vol/ngrids * lib.dot(aoR_k.T.conj()*vjR[i], aoR_k)
                  for aoR_k in aoR_kpts]
            vj_kpts.append(lib.asarray(vj))
    vj_kpts = lib.asarray(vj_kpts)
    vjR = None

# K
    weight = 1./nkpts * (cell.vol/ngrids)
    vk_kpts = np.zeros_like(vj_kpts)
    if kpts_band is not None:
        for k2, kpt2 in enumerate(kpts):
            aoR_dms = [lib.dot(aoR_kpts[k2], dms[i,k2]) for i in range(nset)]
            vkR_k1k2 = get_vkR(mf, cell, aoR_kband, aoR_kpts[k2],
                               kpts_band, kpt2)
            #:vk_kpts = 1./nkpts * (cell.vol/ngrids) * np.einsum('rs,Rp,Rqs,Rr->pq',
            #:            dm_kpts[k2], aoR_kband.conj(),
            #:            vkR_k1k2, aoR_kpts[k2])
            for i in range(nset):
                tmp_Rq = np.einsum('Rqs,Rs->Rq', vkR_k1k2, aoR_dms[i])
                vk_kpts[i] += weight * lib.dot(aoR_kband.T.conj(), tmp_Rq)
            vkR_k1k2 = None
        if dm_kpts.ndim == 3:
            vj_kpts = vj_kpts[0]
            vk_kpts = vk_kpts[0]
        return lib.asarray(vj_kpts), lib.asarray(vk_kpts)
    else:
        for k2, kpt2 in enumerate(kpts):
            aoR_dms = [lib.dot(aoR_kpts[k2], dms[i,k2]) for i in range(nset)]
            for k1, kpt1 in enumerate(kpts):
                vkR_k1k2 = get_vkR(mf, cell, aoR_kpts[k1], aoR_kpts[k2],
                                   kpt1, kpt2)
                for i in range(nset):
                    tmp_Rq = np.einsum('Rqs,Rs->Rq', vkR_k1k2, aoR_dms[i])
                    vk_kpts[i,k1] += weight * lib.dot(aoR_kpts[k1].T.conj(), tmp_Rq)
            vkR_k1k2 = None
        return vj_kpts.reshape(dm_kpts.shape), vk_kpts.reshape(dm_kpts.shape)


def get_vjR_kpts(cell, dm_kpts, aoR_kpts):
    nkpts = len(aoR_kpts)
    coulG = tools.get_coulG(cell)

    rhoR = 0
    for k in range(nkpts):
        rhoR += 1./nkpts*numint.eval_rho(cell, aoR_kpts[k], dm_kpts[k])
    rhoG = tools.fft(rhoR, cell.mesh)

    vG = coulG*rhoG
    vR = tools.ifft(vG, cell.mesh)
    if rhoR.dtype == np.double:
        vR = vR.real
    return vR

##################################################
#
# scf/hf.py end
#
##################################################


def get_nuc(cell, kpt=np.zeros(3)):
    '''Get the bare periodic nuc-el AO matrix, with G=0 removed.

    See Martin (12.16)-(12.21).
    '''
    coords = gen_grid.gen_uniform_grids(cell)
    aoR = numint.eval_ao(cell, coords, kpt)

    chargs = cell.atom_charges()
    SI = cell.get_SI()
    coulG = tools.get_coulG(cell)
    vneG = -np.dot(chargs,SI) * coulG
    vneR = tools.ifft(vneG, cell.mesh).real

    vne = np.dot(aoR.T.conj(), vneR.reshape(-1,1)*aoR)
    return vne


def setUpModule():
    global cell, cell1, cell2, kpts, kpt0, kpts1, mf0
    cell = pgto.Cell()
    cell.atom = 'He 1. .5 .5; C .1 1.3 2.1'
    cell.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))],
                  'C' :'gth-szv',}
    cell.pseudo = {'C':'gth-pade'}
    cell.a = np.eye(3) * 2.5
    cell.mesh = [21] * 3
    cell.build()
    np.random.seed(1)
    kpts = np.random.random((4,3))
    kpts[3] = kpts[0]-kpts[1]+kpts[2]
    kpt0 = np.zeros(3)

    cell1 = pgto.Cell()
    cell1.atom = 'He 1. .5 .5; He .1 1.3 2.1'
    cell1.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))]}
    cell1.a = np.eye(3) * 2.5
    cell1.mesh = [21] * 3
    cell1.build()

    cell2 = pgto.Cell()
    cell2.atom = '''
    He   1.3    .2       .3
    He    .1    .1      1.1 '''
    cell2.basis = {'He': [[0, [0.8, 1]],
                          [1, [0.6, 1]]]}
    cell2.mesh = [17]*3
    cell2.a = numpy.array(([2.0,  .9, 0. ],
                           [0.1, 1.9, 0.4],
                           [0.8, 0  , 2.1]))
    cell2.build()
    kpts1 = np.random.random((4,3))
    kpts1[3] = kpts1[0]-kpts1[1]+kpts1[2] + cell2.reciprocal_vectors().T.dot(np.ones(3))

    mf0 = pbcscf.RHF(cell)
    mf0.exxdiv = None

def tearDownModule():
    global cell, cell1, cell2, kpts, kpt0, kpts1, mf0
    del cell, cell1, cell2, kpts, kpt0, kpts1, mf0

class KnownValues(unittest.TestCase):
    def test_get_nuc(self):
        v0 = get_nuc(cell)
        v1 = fft.FFTDF(cell).get_nuc()
        self.assertTrue(v1.ndim == 3)
        self.assertAlmostEqual(abs(v0 - v1[0]).max(), 0, 9)

        v0 = get_nuc(cell, kpts[0])
        v1 = fft.FFTDF(cell).get_nuc(kpts)
        self.assertTrue(np.allclose(v0, v1[0], atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(lib.fp(v1[0]), (-5.7646608099493841+0.19126294430138713j), 8)

        v0 = get_nuc(cell, kpts[1])
        self.assertTrue(np.allclose(v0, v1[1], atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(lib.fp(v1[1]), (-5.6567258309199193+0.86813371243952175j), 8)
        self.assertAlmostEqual(lib.fp(v1[2]), (-6.1528952645454895+0.09517054428060109j), 8)
        self.assertAlmostEqual(lib.fp(v1[3]), (-5.7445962879770942+0.24611951427601772j), 8)

    def test_get_pp(self):
        v0 = pgto.pseudo.get_pp(cell, kpts[0])
        v1 = fft.FFTDF(cell).get_pp(kpts)
        self.assertTrue(np.allclose(v0, v1[0], atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(lib.fp(v1[0]), (-5.6240249083785869+0.22094834302524968j), 8)

        v0 = pgto.pseudo.get_pp(cell, kpts[1])
        self.assertTrue(np.allclose(v0, v1[1], atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(lib.fp(v1[1]), (-5.5387702576467603+1.0439333717227581j) , 8)
        self.assertAlmostEqual(lib.fp(v1[2]), (-6.0530899866313366+0.2817289667029651j), 8)
        self.assertAlmostEqual(lib.fp(v1[3]), (-5.6011543542444446+0.27597306418805201j), 8)

    def test_get_jk(self):
        df = fft.FFTDF(cell)
        dm = mf0.get_init_guess()
        vj0, vk0 = get_jk(mf0, cell, dm, kpt=kpts[0])
        vj1, vk1 = df.get_jk(dm, kpts=kpts[0], exxdiv=None)
        self.assertTrue(vj1.dtype == numpy.complex128)
        self.assertTrue(vk1.dtype == numpy.complex128)
        self.assertTrue(np.allclose(vj0, vj1, atol=1e-9, rtol=1e-9))
        self.assertTrue(np.allclose(vk0, vk1, atol=1e-9, rtol=1e-9))

        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 2.3002596914518700*(6/6.82991739766009)**2, 8)
        self.assertAlmostEqual(ek1, 3.3165691757797346*(6/6.82991739766009)**2, 8)

        dm = mf0.get_init_guess()
        vj0, vk0 = get_jk(mf0, cell, dm)
        vj1, vk1 = df.get_jk(dm, exxdiv=None)
        self.assertTrue(vj1.dtype == numpy.float64)
        self.assertTrue(vk1.dtype == numpy.float64)
        self.assertTrue(np.allclose(vj0, vj1, atol=1e-9, rtol=1e-9))
        self.assertTrue(np.allclose(vk0, vk1, atol=1e-9, rtol=1e-9))

        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 2.4673139106639925*(6/6.82991739766009)**2, 8)
        self.assertAlmostEqual(ek1, 3.6886674521354221*(6/6.82991739766009)**2, 8)

        # issue #1114
        dm = numpy.eye(cell.nao, dtype=int)
        vj, vk = df.get_jk(dm, exxdiv=None)
        self.assertAlmostEqual(lib.fp(vj), 3.7955873127283377, 8)
        self.assertAlmostEqual(lib.fp(vk), 4.290076429522121, 8)

    def test_get_jk_kpts(self):
        df = fft.FFTDF(cell)
        dm = mf0.get_init_guess()
        nkpts = len(kpts)
        dms = [dm] * nkpts
        vj0, vk0 = get_jk_kpts(mf0, cell, dms, kpts=kpts)
        vj1, vk1 = df.get_jk(dms, kpts=kpts, exxdiv=None)
        self.assertTrue(vj1.dtype == numpy.complex128)
        self.assertTrue(vk1.dtype == numpy.complex128)
        self.assertTrue(np.allclose(vj0, vj1, atol=1e-9, rtol=1e-9))
        self.assertTrue(np.allclose(vk0, vk1, atol=1e-9, rtol=1e-9))

        ej1 = numpy.einsum('xij,xji->', vj1, dms) / len(kpts)
        ek1 = numpy.einsum('xij,xji->', vk1, dms) / len(kpts)
        self.assertAlmostEqual(ej1, 2.3163352969873445*(6/6.82991739766009)**2, 8)
        self.assertAlmostEqual(ek1, 7.7311228144548600*(6/6.82991739766009)**2, 8)

        numpy.random.seed(1)
        kpts_band = numpy.random.random((2,3))
        vj1, vk1 = df.get_jk(dms, kpts=kpts, kpts_band=kpts_band, exxdiv=None)
        self.assertAlmostEqual(lib.fp(vj1), 6/6.82991739766009*(3.437188138446714+0.1360466492092307j), 8)
        self.assertAlmostEqual(lib.fp(vk1), 6/6.82991739766009*(7.479986541097368+1.1980593415201204j), 8)

        nao = dm.shape[0]
        mo_coeff = numpy.random.random((nkpts,nao,nao))
        mo_occ = numpy.array(numpy.random.random((nkpts,nao))>.6, dtype=numpy.double)
        dms = numpy.einsum('kpi,ki,kqi->kpq', mo_coeff, mo_occ, mo_coeff)
        dms = lib.tag_array(lib.asarray(dms), mo_coeff=mo_coeff, mo_occ=mo_occ)
        vk1 = df.get_jk(dms, kpts=kpts, kpts_band=kpts_band, exxdiv=None)[1]
        self.assertAlmostEqual(lib.fp(vk1), 10.239828255099447+2.1190549216896182j, 8)

    def test_get_j_non_hermitian(self):
        kpt = kpts[0]
        numpy.random.seed(2)
        nao = cell2.nao
        dm = numpy.random.random((nao,nao))
        mydf = fft.FFTDF(cell2)
        v1 = mydf.get_jk(dm, hermi=0, kpts=kpts[1], with_k=False)[0]
        eri = mydf.get_eri([kpts[1]]*4).reshape(nao,nao,nao,nao)
        ref = numpy.einsum('ijkl,ji->kl', eri, dm)
        self.assertAlmostEqual(abs(ref - v1).max(), 0, 12)
        self.assertTrue(abs(ref-ref.T.conj()).max() > 1e-5)

    def test_get_ao_eri(self):
        df = fft.FFTDF(cell)
        eri0 = get_ao_eri(cell)
        eri1 = df.get_ao_eri(compact=True)
        self.assertTrue(np.allclose(eri0, eri1, atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(lib.fp(eri1), 0.80425358275734926, 8)

        eri0 = get_ao_eri(cell, kpts[0])
        eri1 = df.get_ao_eri(kpts[0])
        self.assertTrue(np.allclose(eri0, eri1, atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(lib.fp(eri1), (2.9346374584901898-0.20479054936744959j), 8)

        eri4 = df.get_ao_eri(kpts)
        self.assertAlmostEqual(lib.fp(eri4), (0.33709288394542991-0.94185725001175313j), 8)

    def test_get_eri_gamma(self):
        odf = aft.AFTDF(cell1)
        ref = odf.get_eri(compact=True)
        df = fft.FFTDF(cell1)
        eri0000 = df.get_eri(compact=True)
        self.assertTrue(eri0000.dtype == numpy.double)
        self.assertTrue(np.allclose(eri0000, ref, atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(lib.fp(eri0000), 0.23714016293926865, 8)

        ref = odf.get_eri((kpts[0],kpts[0],kpts[0],kpts[0]))
        eri1111 = df.get_eri((kpts[0],kpts[0],kpts[0],kpts[0]))
        self.assertTrue(np.allclose(eri1111, ref, atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(lib.fp(eri1111), (1.2410388899583582-5.2370501878355006e-06j), 8)

        eri1111 = df.get_eri((kpts[0]+1e-8,kpts[0]+1e-8,kpts[0],kpts[0]))
        self.assertTrue(np.allclose(eri1111, ref, atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(lib.fp(eri1111), (1.2410388899583582-5.2370501878355006e-06j), 8)

    def test_get_eri_0011(self):
        odf = aft.AFTDF(cell1)
        df = fft.FFTDF(cell1)
        ref = odf.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        eri0011 = df.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        self.assertTrue(np.allclose(eri0011, ref, atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(lib.fp(eri0011), (1.2410162858084512+0.00074485383749912936j), 8)

        ref = get_mo_eri(cell1, [numpy.eye(cell1.nao_nr())]*4, (kpts[0],kpts[0],kpts[1],kpts[1]))
        eri0011 = df.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        self.assertTrue(np.allclose(eri0011, ref, atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(lib.fp(eri0011), (1.2410162860852818+0.00074485383748954838j), 8)

    def test_get_eri_0110(self):
        odf = aft.AFTDF(cell1)
        df = fft.FFTDF(cell1)
        ref = odf.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        eri0110 = df.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        self.assertTrue(np.allclose(eri0110, ref, atol=1e-9, rtol=1e-9))
        eri0110 = df.get_eri((kpts[0]+1e-8,kpts[1]+1e-8,kpts[1],kpts[0]))
        self.assertTrue(np.allclose(eri0110, ref, atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(lib.fp(eri0110), (1.2928399254827956-0.011820590601969154j), 8)

        ref = get_mo_eri(cell1, [numpy.eye(cell1.nao_nr())]*4, (kpts[0],kpts[1],kpts[1],kpts[0]))
        eri0110 = df.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        self.assertTrue(np.allclose(eri0110, ref, atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(lib.fp(eri0110), (1.2928399254827956-0.011820590601969154j), 8)
        eri0110 = df.get_eri((kpts[0]+1e-8,kpts[1]+1e-8,kpts[1],kpts[0]))
        self.assertTrue(np.allclose(eri0110, ref, atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(lib.fp(eri0110), (1.2928399254827956-0.011820590601969154j), 8)

    def test_get_eri_0123(self):
        odf = aft.AFTDF(cell1)
        df = fft.FFTDF(cell1)
        ref = odf.get_eri(kpts)
        eri1111 = df.get_eri(kpts)
        self.assertTrue(np.allclose(eri1111, ref, atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(lib.fp(eri1111), (1.2917759427391706-0.013340252488069412j), 8)

        ref = get_mo_eri(cell1, [numpy.eye(cell1.nao_nr())]*4, kpts)
        eri1111 = df.get_eri(kpts)
        self.assertTrue(np.allclose(eri1111, ref, atol=1e-9, rtol=1e-9))
        self.assertAlmostEqual(lib.fp(eri1111), (1.2917759427391706-0.013340252488069412j), 8)

    def test_get_mo_eri(self):
        df = fft.FFTDF(cell)
        nao = cell.nao_nr()
        numpy.random.seed(5)
        mo =(numpy.random.random((nao,nao)) +
             numpy.random.random((nao,nao))*1j)
        eri_mo0 = get_mo_eri(cell, (mo,)*4, kpts)
        eri_mo1 = df.get_mo_eri((mo,)*4, kpts)
        self.assertTrue(np.allclose(eri_mo1, eri_mo0, atol=1e-9, rtol=1e-9))

        kpts_t = (kpts[2],kpts[3],kpts[0],kpts[1])
        eri_mo2 = get_mo_eri(cell, (mo,)*4, kpts_t)
        eri_mo2 = eri_mo2.reshape((nao,)*4).transpose(2,3,0,1).reshape(nao**2,-1)
        self.assertTrue(np.allclose(eri_mo2, eri_mo0, atol=1e-9, rtol=1e-9))

        eri_mo0 = get_mo_eri(cell, (mo,)*4, (kpts[0],)*4)
        eri_mo1 = df.get_mo_eri((mo,)*4, (kpts[0],)*4)
        self.assertTrue(np.allclose(eri_mo1, eri_mo0, atol=1e-9, rtol=1e-9))

        eri_mo0 = get_mo_eri(cell, (mo,)*4, (kpts[0],kpts[1],kpts[1],kpts[0],))
        eri_mo1 = df.get_mo_eri((mo,)*4, (kpts[0],kpts[1],kpts[1],kpts[0],))
        self.assertTrue(np.allclose(eri_mo1, eri_mo0, atol=1e-9, rtol=1e-9))

        eri_mo0 = get_mo_eri(cell, (mo,)*4, (kpt0,kpt0,kpts[0],kpts[0],))
        eri_mo1 = df.get_mo_eri((mo,)*4, (kpt0,kpt0,kpts[0],kpts[0],))
        self.assertTrue(np.allclose(eri_mo1, eri_mo0, atol=1e-9, rtol=1e-9))

        eri_mo0 = get_mo_eri(cell, (mo,)*4, (kpts[0],kpts[0],kpt0,kpt0,))
        eri_mo1 = df.get_mo_eri((mo,)*4, (kpts[0],kpts[0],kpt0,kpt0,))
        self.assertTrue(np.allclose(eri_mo1, eri_mo0, atol=1e-9, rtol=1e-9))

        mo1 = mo[:,:nao//2+1]
        eri_mo0 = get_mo_eri(cell, (mo1,mo,mo,mo1), (kpts[0],)*4)
        eri_mo1 = df.get_mo_eri((mo1,mo,mo,mo1), (kpts[0],)*4)
        self.assertTrue(np.allclose(eri_mo1, eri_mo0, atol=1e-9, rtol=1e-9))

        eri_mo0 = get_mo_eri(cell, (mo1,mo,mo1,mo), (kpts[0],kpts[1],kpts[1],kpts[0],))
        eri_mo1 = df.get_mo_eri((mo1,mo,mo1,mo), (kpts[0],kpts[1],kpts[1],kpts[0],))
        self.assertTrue(np.allclose(eri_mo1, eri_mo0, atol=1e-9, rtol=1e-9))

    def test_get_mo_eri1(self):
        df = fft.FFTDF(cell2)
        nao = cell2.nao_nr()
        numpy.random.seed(5)
        mos =(numpy.random.random((4,nao,nao)) +
              numpy.random.random((4,nao,nao))*1j)
        eri_mo0 = get_mo_eri(cell2, mos, kpts1)
        eri_mo1 = df.get_mo_eri(mos, kpts1)
        self.assertTrue(np.allclose(eri_mo1, eri_mo0, atol=1e-9, rtol=1e-9))

    def test_ao2mo_7d(self):
        L = 3.
        n = 6
        cell = pgto.Cell()
        cell.a = numpy.diag([L,L,L])
        cell.mesh = [n,n,n]
        cell.atom = '''He    2.    2.2      2.
                       He    1.2   1.       1.'''
        cell.basis = {'He': [[0, (1.2, 1)], [1, (0.6, 1)]]}
        cell.verbose = 0
        cell.build(0,0)

        kpts = cell.make_kpts([1,3,1])
        nkpts = len(kpts)
        nao = cell.nao_nr()
        numpy.random.seed(1)
        mo =(numpy.random.random((nkpts,nao,nao)) +
             numpy.random.random((nkpts,nao,nao))*1j)

        with_df = fft.FFTDF(cell, kpts)
        out = with_df.ao2mo_7d(mo, kpts)
        ref = numpy.empty_like(out)

        kconserv = kpts_helper.get_kconserv(cell, kpts)
        for ki, kj, kk in kpts_helper.loop_kkk(nkpts):
            kl = kconserv[ki, kj, kk]
            tmp = with_df.ao2mo((mo[ki], mo[kj], mo[kk], mo[kl]), kpts[[ki,kj,kk,kl]])
            ref[ki,kj,kk] = tmp.reshape([nao]*4)

        self.assertAlmostEqual(abs(out-ref).max(), 0, 12)

    def test_get_jk_with_casscf(self):
        from pyscf import mcscf
        pcell = cell2.copy()
        pcell.verbose = 0
        pcell.mesh = [8]*3
        mf = pbcscf.RHF(pcell)
        mf.exxdiv = None
        ehf = mf.kernel()

        mc = mcscf.CASSCF(mf, 1, 2).run()
        self.assertAlmostEqual(mc.e_tot, ehf, 8)

        mc = mcscf.CASSCF(mf, 2, 0).run()
        self.assertAlmostEqual(mc.e_tot, ehf, 8)


if __name__ == '__main__':
    print("Full Tests for fft JK and ao2mo etc")
    unittest.main()
