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
This ao2mo module is kept for backward compatiblity.  It's recommended to use
pyscf.pbc.df module to get 2e MO integrals
'''

import numpy as np
from pyscf.pbc import df
from pyscf import lib
from pyscf.pbc.dft.gen_grid import gen_uniform_grids
from pyscf.pbc.dft.numint import eval_ao
from pyscf.pbc import tools
from pyscf.lib import logger

einsum = lib.einsum

def general(cell, mo_coeffs, kpts=None, compact=False):
    '''pyscf-style wrapper to get MO 2-el integrals.'''
    if kpts is not None:
        assert len(kpts) == 4
    return get_mo_eri(cell, mo_coeffs, kpts)
    #return df.FFTDF(cell).ao2mo(mo_coeffs, kpts, compact)

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
    coords = gen_uniform_grids(cell)
    if kpts is None:
        q = np.zeros(3)
        aoR = eval_ao(cell, coords)
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
        aoR_ki = eval_ao(cell, coords, kpt=kpts[0])
        aoR_kj = eval_ao(cell, coords, kpt=kpts[1])
        ngrids = aoR_ki.shape[0]

        nmoi = mo_coeffs[0].shape[1]
        nmoj = mo_coeffs[1].shape[1]
        moiR = einsum('ri,ia->ra', aoR_ki, mo_coeffs[0])
        mojR = einsum('ri,ia->ra', aoR_kj, mo_coeffs[1])

    #mo_pairs_R = einsum('ri,rj->rij', np.conj(moiR), mojR)
    mo_pairs_G = np.zeros([ngrids,nmoi*nmoj], np.complex128)

    expmikr = np.exp(-1j*np.dot(q,coords.T))
    for i in range(nmoi):
        for j in range(nmoj):
            mo_pairs_R_ij = np.conj(moiR[:,i])*mojR[:,j]
            mo_pairs_G[:,i*nmoj+j] = tools.fftk(mo_pairs_R_ij, cell.mesh,
                                                expmikr)

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
    coords = gen_uniform_grids(cell)
    if kpts is None:
        q = np.zeros(3)
        aoR = eval_ao(cell, coords)
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
        aoR_ki = eval_ao(cell, coords, kpt=kpts[0])
        aoR_kj = eval_ao(cell, coords, kpt=kpts[1])
        ngrids = aoR_ki.shape[0]

        nmoi = mo_coeffs[0].shape[1]
        nmoj = mo_coeffs[1].shape[1]
        moiR = einsum('ri,ia->ra', aoR_ki, mo_coeffs[0])
        mojR = einsum('ri,ia->ra', aoR_kj, mo_coeffs[1])

    #mo_pairs_R = einsum('ri,rj->rij', np.conj(moiR), mojR)
    mo_pairs_invG = np.zeros([ngrids,nmoi*nmoj], np.complex128)

    expmikr = np.exp(-1j*np.dot(q,coords.T))
    for i in range(nmoi):
        for j in range(nmoj):
            mo_pairs_R_ij = np.conj(moiR[:,i])*mojR[:,j]
            mo_pairs_invG[:,i*nmoj+j] = np.conj(tools.fftk(np.conj(mo_pairs_R_ij),
                                                           cell.mesh, expmikr.conj()))

    return mo_pairs_invG

def assemble_eri(cell, orb_pair_invG1, orb_pair_G2, q=None, verbose=logger.INFO):
    '''Assemble 4-index electron repulsion integrals.
    Returns:
        (nmo1*nmo2, nmo3*nmo4) ndarray
    '''
    log = logger.Logger
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cell.stdout, verbose)

    log.debug('Performing periodic ERI assembly of (%i, %i) ij,kl pairs',
              orb_pair_invG1.shape[1], orb_pair_G2.shape[1])
    if q is None:
        q = np.zeros(3)

    coulqG = tools.get_coulG(cell, -1.0*q)
    ngrids = orb_pair_invG1.shape[0]
    Jorb_pair_G2 = np.einsum('g,gn->gn',coulqG,orb_pair_G2)*(cell.vol/ngrids**2)
    eri = np.dot(orb_pair_invG1.T, Jorb_pair_G2)
    return eri

#def get_mo_eri(cell, mo_coeffs, kpts=None):
#    '''Convenience function to return MO 2-el integrals.'''
#    return general(cell, mo_coeffs, kpts)

#def get_mo_pairs_G(cell, mo_coeffs, kpts=None):
#    '''Calculate forward (G|ij) FFT of all MO pairs.
#
#    TODO: - Implement simplifications for real orbitals.
#
#    Args:
#        mo_coeff: length-2 list of (nao,nmo) ndarrays
#            The two sets of MO coefficients to use in calculating the
#            product |ij).
#
#    Returns:
#        mo_pairs_G : (ngrids, nmoi*nmoj) ndarray
#            The FFT of the real-space MO pairs.
#    '''
#    return df.FFTDF(cell).get_mo_pairs(mo_coeffs, kpts)
#
#def get_mo_pairs_invG(cell, mo_coeffs, kpts=None):
#    '''Calculate "inverse" (ij|G) FFT of all MO pairs.
#
#    TODO: - Implement simplifications for real orbitals.
#
#    Args:
#        mo_coeff: length-2 list of (nao,nmo) ndarrays
#            The two sets of MO coefficients to use in calculating the
#            product |ij).
#
#    Returns:
#        mo_pairs_invG : (ngrids, nmoi*nmoj) ndarray
#            The inverse FFTs of the real-space MO pairs.
#    '''
#    if kpts is None: kpts = numpy.zeros((2,3))
#    mo_pairs_G = df.FFTDF(cell).get_mo_pairs((mo_coeffs[1],mo_coeffs[0]),
#                                             (kpts[1],kpts[0]))
#    nmo0 = mo_coeffs[0].shape[1]
#    nmo1 = mo_coeffs[1].shape[1]
#    mo_pairs_invG = mo_pairs_G.T.reshape(nmo1,nmo0,-1).transpose(1,0,2).conj()
#    mo_pairs_invG = mo_pairs_invG.reshape(nmo0*nmo1,-1).T
#    return mo_pairs_invG

def get_ao_pairs_G(cell, kpts=None):
    '''Calculate forward (G|ij) and "inverse" (ij|G) FFT of all AO pairs.

    Args:
        cell : instance of :class:`Cell`

    Returns:
        ao_pairs_G, ao_pairs_invG : (ngrids, nao*(nao+1)/2) ndarray
            The FFTs of the real-space AO pairs.

    '''
    return df.FFTDF(cell).get_ao_pairs(kpts)

def get_ao_eri(cell, kpts=None):
    '''Convenience function to return AO 2-el integrals.'''
    if kpts is not None:
        assert len(kpts) == 4
    return df.FFTDF(cell).get_eri(kpts)

if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
    cell = pgto.Cell()
    cell.atom = 'He 1. .5 .5; He .1 1.3 2.1'
    cell.basis = 'ccpvdz'
    cell.a = np.eye(3) * 4.
    cell.mesh = [11]*3
    cell.build()
    print(get_ao_eri(cell).shape)
