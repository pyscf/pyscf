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
    r'''Get the momentum conservation array for a set of k-points.

    Given k-point indices (k, l, m) the array kconserv[k,l,m] returns
    the index n that satifies momentum conservation,

        (k(k) - k(l) + k(m) - k(n)) \dot a = 2n\pi

    This is used for symmetry e.g. integrals of the form
        [\phi*[k](1) \phi[l](1) | \phi*[m](2) \phi[n](2)]
    are zero unless n satisfies the above.
    '''
    nkpts = kpts.shape[0]
    a = cell.lattice_vectors() / (2*np.pi)

    kconserv = np.zeros((nkpts,nkpts,nkpts), dtype=int)
    kvMLK = kpts[:,None,None,:] - kpts[:,None,:] + kpts
    for N, kvN in enumerate(kpts):
        kvMLKN = np.einsum('klmx,wx->mlkw', kvMLK - kvN, a)
        # check whether  (1/(2pi) k_{KLMN} dot a)  are integer
        kvMLKN_int = np.rint(kvMLKN)
        mask = np.einsum('klmw->mlk', abs(kvMLKN - kvMLKN_int)) < 1e-9
        kconserv[mask] = N
    return kconserv

def get_kconserv3(cell, kpts, kijkab):
    '''Get the momentum conservation array for a set of k-points.

    This function is similar to get_kconserv, but instead finds the 'kc'
    that satisfies momentum conservation for 5 k-points,

        (ki + kj + kk - ka - kb - kc) dot a = 2n\pi

    where these kpoints are stored in kijkab[ki,kj,kk,ka,kb].
    '''
    nkpts = kpts.shape[0]
    a = cell.lattice_vectors() / (2*np.pi)

    kpts_i, kpts_j, kpts_k, kpts_a, kpts_b = \
            [kpts[x].reshape(-1,3) for x in kijkab]
    shape = [np.size(x) for x in kijkab]
    kconserv = np.zeros(shape, dtype=int)

    kv_kab = kpts_k[:,None,None,:] - kpts_a[:,None,:] - kpts_b
    for i, kpti in enumerate(kpts_i):
        for j, kptj in enumerate(kpts_j):
            kv_ijkab = kv_kab + kpti + kptj
            for c, kptc in enumerate(kpts):
                s = np.einsum('kabx,wx->kabw', kv_ijkab - kptc, a)
                s_int = np.rint(s)
                mask = np.einsum('kabw->kab', abs(s - s_int)) < 1e-9
                kconserv[i,j,mask] = c

    new_shape = [shape[i] for i, x in enumerate(kijkab)
                 if not isinstance(x, (int,np.int))]
    kconserv = kconserv.reshape(new_shape)
    return kconserv


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



