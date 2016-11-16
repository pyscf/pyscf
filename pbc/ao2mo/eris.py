#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
This ao2mo module is kept for backward compatiblity.  It's recommended to use
pyscf.pbc.df module to get 2e MO integrals
'''
import numpy as np
from pyscf.pbc import df

def general(cell, mo_coeffs, kpts=None, compact=False):
    '''pyscf-style wrapper to get MO 2-el integrals.'''
    if kpts is not None:
        assert len(kpts) == 4
    return df.DF(cell).ao2mo(mo_coeffs, kpts, compact)

def get_mo_eri(cell, mo_coeffs, kpts=None):
    '''Convenience function to return MO 2-el integrals.'''
    return general(cell, mo_coeffs, kpts)

def get_mo_pairs_G(cell, mo_coeffs, kpts=None):
    '''Calculate forward (G|ij) FFT of all MO pairs.

    TODO: - Implement simplifications for real orbitals.

    Args:
        mo_coeff: length-2 list of (nao,nmo) ndarrays
            The two sets of MO coefficients to use in calculating the
            product |ij).

    Returns:
        mo_pairs_G : (ngs, nmoi*nmoj) ndarray
            The FFT of the real-space MO pairs.
    '''
    return df.DF(cell).get_mo_pairs(mo_coeffs, kpts)

def get_mo_pairs_invG(cell, mo_coeffs, kpts=None):
    '''Calculate "inverse" (ij|G) FFT of all MO pairs.

    TODO: - Implement simplifications for real orbitals.

    Args:
        mo_coeff: length-2 list of (nao,nmo) ndarrays
            The two sets of MO coefficients to use in calculating the
            product |ij).

    Returns:
        mo_pairs_invG : (ngs, nmoi*nmoj) ndarray
            The inverse FFTs of the real-space MO pairs.
    '''
    if kpts is None: kpts = numpy.zeros((2,3))
    mo_pairs_G = df.DF(cell).get_mo_pairs((mo_coeffs[1],mo_coeffs[0]),
                                          (kpts[1],kpts[0]))
    nmo0 = mo_coeffs[0].shape[1]
    nmo1 = mo_coeffs[1].shape[1]
    mo_pairs_invG = mo_pairs_G.T.reshape(nmo1,nmo0,-1).transpose(1,0,2).conj()
    mo_pairs_invG = mo_pairs_invG.reshape(nmo0*nmo1,-1).T
    return mo_pairs_invG

def get_ao_pairs_G(cell, kpts=None):
    '''Calculate forward (G|ij) and "inverse" (ij|G) FFT of all AO pairs.

    Args:
        cell : instance of :class:`Cell`

    Returns:
        ao_pairs_G, ao_pairs_invG : (ngs, nao*(nao+1)/2) ndarray
            The FFTs of the real-space AO pairs.

    '''
    return df.DF(cell).get_ao_pairs(kpts)

def get_ao_eri(cell, kpts=None):
    '''Convenience function to return AO 2-el integrals.'''
    if kpts is not None:
        assert len(kpts) == 4
    return df.DF(cell).get_eri(kpts)

if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
    cell = pgto.Cell()
    cell.atom = 'He 1. .5 .5; He .1 1.3 2.1'
    cell.basis = 'ccpvdz'
    cell.a = np.eye(3) * 4.
    cell.gs = [5,5,5]
    cell.build()
    print(get_ao_eri(cell).shape)
