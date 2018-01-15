#!/usr/bin/env python
#
# Authors: Qiming Sun <osirpt.sun@gmail.com>
#          James D. McClain <jmcclain@princeton.edu>
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#

from collections import OrderedDict
import numpy as np
import scipy.linalg

import pyscf.lib

KPT_DIFF_TOL = 1e-6

def is_zero(kpt):
    return abs(np.asarray(kpt)).sum() < KPT_DIFF_TOL
gamma_point = is_zero

def member(kpt, kpts):
    kpts = np.reshape(kpts, (len(kpts),kpt.size))
    dk = np.einsum('ki->k', abs(kpts-kpt.ravel()))
    return np.where(dk < KPT_DIFF_TOL)[0]

def unique(kpts):
    kpts = np.asarray(kpts)
    nkpts = len(kpts)
    uniq_kpts = []
    uniq_index = []
    uniq_inverse = np.zeros(nkpts, dtype=int)
    seen = np.zeros(nkpts, dtype=bool)
    n = 0
    for i, kpt in enumerate(kpts):
        if not seen[i]:
            uniq_kpts.append(kpt)
            uniq_index.append(i)
            idx = abs(kpt-kpts).sum(axis=1) < KPT_DIFF_TOL
            uniq_inverse[idx] = n
            seen[idx] = True
            n += 1
    return np.asarray(uniq_kpts), np.asarray(uniq_index), uniq_inverse

def get_kconserv(cell, kpts):
    '''Get the momentum conservation array for a set of k-points.

    Given k-point indices (k, l, m) the array kconserv[k,l,m] returns
    the index n that satifies momentum conservation,

       k(k) - k(l) = - k(m) + k(n)

    This is used for symmetry e.g. integrals of the form
        [\phi*[k](1) \phi[l](1) | \phi*[m](2) \phi[n](2)]
    are zero unless n satisfies the above.
    '''
    nkpts = kpts.shape[0]
    KLMN = np.zeros([nkpts,nkpts,nkpts], np.int)
    kvecs = cell.reciprocal_vectors()

    for K, kvK in enumerate(kpts):
        for L, kvL in enumerate(kpts):
            for M, kvM in enumerate(kpts):
                # Here we find where kvN = kvM + kvL - kvK (mod K)
                temp = range(-1,2)
                xyz = pyscf.lib.cartesian_prod((temp,temp,temp))
                found = 0
                kvMLK = kvK - kvL + kvM
                kvN = kvMLK
                for ishift in range(len(xyz)):
                    kvN = kvMLK + np.dot(xyz[ishift],kvecs)
                    finder = np.where(np.logical_and(kpts < kvN + 1.e-12,
                                                     kpts > kvN - 1.e-12).sum(axis=1)==3)
                    # The k-point should be the same in all 3 indices as kvN
                    if len(finder[0]) > 0:
                        KLMN[K, L, M] = finder[0][0]
                        found = 1
                        break

                if found == 0:
                    raise RuntimeError('Momentum-conserving k-point not found')
    return KLMN

def get_kconserv3(cell, kpts, kijkab):
    '''Get the momentum conservation array for a set of k-points.

    This function is similar to get_kconserv, but instead finds the 'kc'
    that satisfies momentum conservation for 5 k-points,

    kc = ki + kj + kk - ka - kb (mod G),

    where these kpoints are stored in kijkab[ki,kj,kk,ka,kb].
    '''
    nkpts = kpts.shape[0]
    KLMN = np.zeros([nkpts,nkpts,nkpts], np.int)
    kvecs = 2*np.pi*scipy.linalg.inv(cell._h)
    kijkab = np.array(kijkab)

    # Finds which indices in ijkab are integers and which are lists
    # TODO: try to see if it works for more than 1 list
    idx_sum = np.array([not(isinstance(x,int) or isinstance(x,np.int)) for x in kijkab])
    idx_range = kijkab[idx_sum]
    min_idx_range = np.zeros(5,dtype=int)
    min_idx_range = np.array([min(x) for x in idx_range])
    out_array_shape = tuple([len(x) for x in idx_range])
    out_array = np.zeros(shape=out_array_shape,dtype=int)
    kpqrst_idx = np.zeros(5,dtype=int)

    # Order here matters! Search for most ``obvious" translation first to
    # get into 1st BZ, i.e. no translation!
    temp = [0,-1,1,-2,2]
    xyz = pyscf.lib.cartesian_prod((temp,temp,temp))
    kshift = np.dot(xyz,kvecs)

    for L, kvL in enumerate(pyscf.lib.cartesian_prod(idx_range)):
        kpqrst_idx[idx_sum], kpqrst_idx[~idx_sum] = kvL, kijkab[~idx_sum]
        idx = tuple(kpqrst_idx[idx_sum]-min_idx_range)

        kvec = kpts[kpqrst_idx]
        kvec = kvec[0:3].sum(axis=0) - kvec[3:5].sum(axis=0)

        found = 0
        kvNs = kvec + kshift
        for ishift in range(len(xyz)):
            kvN = kvNs[ishift]
            finder = np.where(np.logical_and(kpts < kvN + 1.e-12, kpts > kvN - 1.e-12).sum(axis=1)==3)
            # The k-point kvN is the one that conserves momentum
            if len(finder[0]) > 0:
                found = 1
                out_array[idx] = finder[0][0]
                break

        if found == 0:
            raise RuntimeError('Momentum-conserving k-point not found')
    return out_array


class KptsHelper(pyscf.lib.StreamObject):
    def __init__(self, cell, kpts):
        '''Helper class for handling k-points in correlated calculations.

        Attributes:
            kconserv : (nkpts,nkpts,nkpts) ndarray
                The index of the fourth momentum-conserving k-point, given
                indices of three k-points
            symm_map : OrderedDict of list of (3,) tuples
                Keys are (3,) tuples of symmetry-unique k-point indices and
                values are lists of (3,) tuples, enumerating all
                symmetry-related k-point indices for ERI generation
        '''
        self.kconserv = get_kconserv(cell, kpts)
        nkpts = len(kpts)
        temp = range(0,nkpts)
        kptlist = pyscf.lib.cartesian_prod((temp,temp,temp))
        completed = np.zeros((nkpts,nkpts,nkpts), dtype=bool)

        self._operation = np.zeros((nkpts,nkpts,nkpts), dtype=int)
        self.symm_map = OrderedDict()

        for kpt in kptlist:
            kpt = tuple(kpt)
            kp,kq,kr = kpt
            if not completed[kp,kq,kr]:
                self.symm_map[kpt] = list()
                ks = self.kconserv[kp,kq,kr]

                completed[kp,kq,kr] = True
                self._operation[kp,kq,kr] = 0
                self.symm_map[kpt].append((kp,kq,kr))

                completed[kr,ks,kp] = True 
                self._operation[kr,ks,kp] = 1 #.transpose(2,3,0,1)
                self.symm_map[kpt].append((kr,ks,kp))

                completed[kq,kp,ks] = True 
                self._operation[kq,kp,ks] = 2 #np.conj(.transpose(1,0,3,2))
                self.symm_map[kpt].append((kq,kp,ks))

                completed[ks,kr,kq] = True 
                self._operation[ks,kr,kq] = 3 #np.conj(.transpose(3,2,1,0))
                self.symm_map[kpt].append((ks,kr,kq))


    def transform_symm(self, eri_kpt, kp, kq, kr):
        '''Return the symmetry-related ERI at any set of k-points.

        Args:
            eri_kpt : (nmo,nmo,nmo,nmo) ndarray
                An in-cell ERI calculated with a set of symmetry-unique k-points.
            kp, kq, kr : int
                The indices of the k-points at which the ERI is desired.
        '''
        operation = self._operation[kp,kq,kr]
        if operation == 0:
            return eri_kpt
        if operation == 1:
            return eri_kpt.transpose(2,3,0,1)
        if operation == 2:
            return np.conj(eri_kpt.transpose(1,0,3,2))
        if operation == 3:
            return np.conj(eri_kpt.transpose(3,2,1,0))



