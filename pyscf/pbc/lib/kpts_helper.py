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
# Authors: Qiming Sun <osirpt.sun@gmail.com>
#          James D. McClain
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import itertools
from collections import OrderedDict
from numbers import Number
import numpy as np
from pyscf import lib
from pyscf import __config__

KPT_DIFF_TOL = getattr(__config__, 'pbc_lib_kpts_helper_kpt_diff_tol', 1e-6)


def is_zero(kpt):
    return abs(np.asarray(kpt)).sum() < KPT_DIFF_TOL
gamma_point = is_zero

def round_to_fbz(kpts, wrap_around=False, tol=KPT_DIFF_TOL):
    '''
    Round scaled k-points to the first Brillouin zone.

    Args:
        kpts : (...,3) ndarray
            Scaled k-points.
        wrap_around : bool
            If set to True, k-points are rounded to [-0.5, 0.5), otherwise [0.0, 1.0).
            Default value is False.
        tol : float
            K-points differ less than tol are considered as the same.
            Default value is 1e-6.
    '''
    shape = kpts.shape
    kpts = kpts.reshape(-1, 3)
    decimal = -np.log10((tol+1e-16)/10.).astype(int)
    kpts_fbz = np.mod(kpts, 1)
    kpts_fbz = lib.cleanse(kpts_fbz, axis=0, tol=tol)
    kpts_fbz = kpts_fbz.round(decimal)
    kpts_fbz = np.mod(kpts_fbz, 1)
    if wrap_around:
        kpts_fbz[kpts_fbz >= 0.5] -= 1.0
    return kpts_fbz.reshape(shape)

def member(kpt, kpts):
    '''Return the indices of the common k-points in kpts'''
    assert kpts.ndim == kpt.ndim + 1
    dk = abs(kpts.reshape(-1,kpt.size) - kpt.ravel()).max(axis=1)
    idx = np.where(dk < KPT_DIFF_TOL)[0]
    if idx.size > 0:
        idx = np.unique(idx)
    return idx

def intersection(kpts1, kpts2):
    '''Return the indices of the common k-points in kpts1'''
    assert kpts1.ndim == kpts2.ndim == 2
    dk = abs(kpts1[:,None] - kpts2).max(axis=2)
    idx = np.where(dk < KPT_DIFF_TOL)[0]
    if idx.size > 0:
        idx = np.unique(idx)
    return idx

def unique(kpts):
    '''
    Find the unique k-points

    Returns the sorted unique k-points, the indices of the input that give the
    unique elements, and the indices of the unique k-points that reconstruct the
    input.
    '''

    kpts = np.asarray(kpts)
    try:
        digits = int(-np.log10(KPT_DIFF_TOL))
        uniq_index, uniq_inverse = np.unique(
            kpts.round(digits), return_index=True, return_inverse=True, axis=0)[1:3]
        idx = uniq_index.argsort()
        rank = idx.argsort()
        return kpts[uniq_index[idx]], uniq_index[idx], rank[uniq_inverse]
    except TypeError:
        # Old numpy does not support unique of 2D array
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

def unique_with_wrap_around(cell, kpts):
    '''Search unique kpts in first Brillouin zone.'''
    scaled_kpts = cell.get_scaled_kpts(kpts).round(5)
    scaled_kpts = np.modf(scaled_kpts)[0]
    scaled_kpts[scaled_kpts >= .5] -= 1
    scaled_kpts[scaled_kpts < -.5] += 1

    uniq_index, uniq_inverse = unique(scaled_kpts)[1:3]
    uniq_kpts = kpts[uniq_index]
    return uniq_kpts, uniq_index, uniq_inverse

def members_with_wrap_around(cell, kpts, test_kpts):
    '''Search the indices of kpts in first Brillouin zone.
    [where(k==test_kpts) for k in kpts]

    kpts : array_like
        kpts to index
    test_kpts : array_like
        The values against which to index each value of `kpts`.
    '''
    scaled_kpts = cell.get_scaled_kpts(kpts)
    scaled_db = cell.get_scaled_kpts(test_kpts)
    dk = scaled_kpts[:,None] - scaled_db
    idx, idy = np.where(abs(np.modf(dk)[0]).max(axis=2) < KPT_DIFF_TOL)
    return np.asarray(idy[idx.argsort()], dtype=np.int32)

def group_by_conj_pairs(cell, kpts, wrap_around=True, return_kpts_pairs=True):
    '''Find all conjugation k-point pairs in the input kpts.
    This function lables three types of conjugation.
    1. self-conjugated. The two index in idx_pairs have the same value.
    2. conjugated to the k-point within the input kpts. The indices of conjugated
    k-points are held in idx_pairs.
    3. conjugated to the k-point not inside the input kpts. The record is (index, None).
    '''
    if wrap_around:
        scaled = cell.get_scaled_kpts(kpts)
        scaled = np.modf(scaled)[0]
        scaled_conj = -scaled
        scaled[scaled.round(5) > .5] -= 1
        scaled[scaled.round(5) <= -.5] += 1
        scaled_conj[scaled_conj.round(5) > .5] -= 1
        scaled_conj[scaled_conj.round(5) <= -.5] += 1
        kpts = cell.get_abs_kpts(scaled)
    else:
        scaled = cell.get_scaled_kpts(kpts)
        scaled_conj = -scaled

    scaled = scaled.round(5)
    scaled_conj = scaled_conj.round(5)
    self_conj_mask = abs(scaled - scaled_conj).max(axis=1) < KPT_DIFF_TOL
    idx_pairs = [(k, k) for k in np.where(self_conj_mask)[0]]

    seen = self_conj_mask
    for k, (skpt, skpt_conj) in enumerate(zip(scaled, scaled_conj)):
        if seen[k]:
            continue

        seen[k] = True
        conj_idx = member(skpt_conj, scaled)
        if conj_idx.size == 0:
            # conjugated k-point not in the kpts set
            idx_pairs.append((k, None))
        else:
            seen[conj_idx[0]] = True
            idx_pairs.append((k, conj_idx[0]))

    if return_kpts_pairs:
        kpts_pairs = []
        for i, j in idx_pairs:
            if j is None:
                kpts_pairs.append((kpts[i], None))
            else:
                kpts_pairs.append((kpts[i], kpts[j]))
        return idx_pairs, kpts_pairs
    else:
        return idx_pairs

def kk_adapted_iter(cell, kpts, kk_idx=None, time_reversal_symmetry=True):
    '''Generates kpt which is adapted to the kpt_p in (ij|p)'''
    if kk_idx is not None and time_reversal_symmetry:
        raise NotImplementedError('Time reversal symmetry with custom ki-kj pairs')

    log = lib.logger.new_logger(cell)
    nkpts = len(kpts)
    dk = (kpts[None,:,:] - kpts[:,None,:]).reshape(-1, 3)
    if kk_idx is not None:
        dk = dk[kk_idx]
    uniq_kpts, uniq_index, uniq_inverse = unique_with_wrap_around(cell, dk)
    scaled_kpts = cell.get_scaled_kpts(uniq_kpts).round(5)
    log.debug('Num uniq kpts %d', len(uniq_kpts))
    k_conj_groups = group_by_conj_pairs(cell, uniq_kpts, return_kpts_pairs=False)
    for k, k_conj in k_conj_groups:
        self_conj = k == k_conj
        if kk_idx is None:
            kpt_ij_idx = np.where(uniq_inverse == k)[0]
            assert k_conj is not None
        else:
            kpt_ij_idx = kk_idx[uniq_inverse == k]
        kpt_ij_idx = np.asarray(kpt_ij_idx, dtype=np.int32)
        kpti_idx = kpt_ij_idx // nkpts
        kptj_idx = kpt_ij_idx % nkpts
        log.debug1('ft_ao_pair for scaled kpt = %s', scaled_kpts[k])
        log.debug2('ft_ao_pair for kpti_idx = %s', kpti_idx)
        log.debug2('ft_ao_pair for kptj_idx = %s', kptj_idx)
        yield uniq_kpts[k], kpti_idx, kptj_idx, self_conj

        if self_conj or k_conj is None or time_reversal_symmetry:
            continue

        # For time-reversal symmetry
        if kk_idx is None:
            kpt_ij_idx = np.where(uniq_inverse == k_conj)[0]
        else:
            kpt_ij_idx = kk_idx[uniq_inverse == k_conj]
        kpt_ij_idx = np.asarray(kpt_ij_idx, dtype=np.int32)
        kpti_idx = kpt_ij_idx // nkpts
        kptj_idx = kpt_ij_idx % nkpts
        log.debug1('ft_ao_pair for scaled kpt = %s', scaled_kpts[k_conj])
        log.debug2('ft_ao_pair for kpti_idx = %s', kpti_idx)
        log.debug2('ft_ao_pair for kptj_idx = %s', kptj_idx)
        yield uniq_kpts[k_conj], kpti_idx, kptj_idx, self_conj

def loop_kkk(nkpts):
    range_nkpts = range(nkpts)
    return itertools.product(range_nkpts, range_nkpts, range_nkpts)

def conj_mapping(cell, kpts):
    '''Find the mapping index: -kpts = kpts[index]'''
    scaled_kpts = cell.get_scaled_kpts(kpts).round(5)
    scaled_kpts = np.modf(scaled_kpts)[0]
    scaled_kpts[scaled_kpts > .5] -= 1
    scaled_kpts[scaled_kpts <= -.5] += 1
    uniq_kpts, index, inverse = unique(scaled_kpts)
    if len(index) != len(inverse):
        raise KPointSymmetryError('duplicated k points')

    minus_kpts = -scaled_kpts
    minus_kpts[minus_kpts == -.5] = .5
    uniq_m_kpts, m_index, m_inverse = unique(minus_kpts)
    lex_order = np.lexsort(uniq_kpts.T)
    m_lex_order = np.lexsort(uniq_m_kpts.T)
    if abs(uniq_kpts[lex_order] - uniq_m_kpts[m_lex_order]).max() > KPT_DIFF_TOL:
        raise KPointSymmetryError('k points not symmetric')

    index[lex_order] = m_lex_order
    return index

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
    kvKLM = kpts[:,None,None,:] - kpts[:,None,:] + kpts
    for N, kvN in enumerate(kpts):
        kvKLMN = np.einsum('wx,klmx->wklm', a, kvKLM - kvN)
        # check whether (1/(2pi) k_{KLMN} dot a) is an integer
        kvKLMN_int = np.rint(kvKLMN)
        mask = np.einsum('wklm->klm', abs(kvKLMN - kvKLMN_int)) < 1e-9
        kconserv[mask] = N
    return kconserv


def check_kpt_antiperm_symmetry(array, idx1, idx2, tolerance=1e-8):
    '''Checks antipermutational symmetry for k-point array.

    Checks whether an array with k-point symmetry has antipermutational symmetry
    with respect to switching the particle indices `idx1`, `idx2`. The particle
    indices switches both the orbital index and k-point index associated with
    the two indices.

    Note:
        One common reason for not obeying antipermutational symmetry in a calculation
        involving FFTs is that the grid to perform the FFT may be too coarse.  This
        symmetry is present in operators in spin-orbital form and 'spin-free'
        operators.

    array (:obj:`ndarray`): array to test permutational symmetry, where for
        an n-particle array, the first (2n-1) array elements are kpoint indices
        while the final 2n array elements are orbital indices.
    idx1 (int): first index
    idx2 (int): second index

    Examples:
        For a 3-particle array, such as the T3 amplitude
            t3[ki, kj, kk, ka, kb, i, j, a, b, c],
        setting `idx1 = 0` and `idx2 = 1` would switch the orbital indices i, j as well
        as the kpoint indices ki, kj.

        >>> nkpts, nocc, nvir = 3, 4, 5
        >>> t2 = numpy.random.random_sample((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir))
        >>> t2 = t2 + t2.transpose(1,0,2,4,3,5,6)
        >>> check_kpt_antiperm_symmetry(t2, 0, 1)
        True
    '''
    # Checking to make sure bounds of idx1 and idx2 are O.K.
    assert (idx1 >= 0 and idx2 >= 0 and 'indices to swap must be non-negative!')

    array_shape_len = len(array.shape)
    nparticles = (array_shape_len + 1) / 4
    assert (idx1 < (2 * nparticles - 1) and idx2 < (2 * nparticles - 1) and
           'This function does not support the swapping of the last k-point index '
           '(This k-point is implicitly not indexed due to conservation of momentum '
           'between k-points.).')

    if (nparticles > 3):
        raise NotImplementedError('Currently set up for only up to 3 particle '
                                  'arrays. Input array has %d particles.')

    kpt_idx1 = idx1
    kpt_idx2 = idx2

    # Start of the orbital index, located after k-point indices
    orb_idx1 = (2 * nparticles - 1) + idx1
    orb_idx2 = (2 * nparticles - 1) + idx2

    # Sign of permutation
    # sign = (-1)**(abs(idx1 - idx2) + 1)
    out_array_indices = np.arange(array_shape_len)

    out_array_indices[kpt_idx1], out_array_indices[kpt_idx2] = \
            out_array_indices[kpt_idx2], out_array_indices[kpt_idx1]
    out_array_indices[orb_idx1], out_array_indices[orb_idx2] = \
            out_array_indices[orb_idx2], out_array_indices[orb_idx1]
    antisymmetric = (np.linalg.norm(array + array.transpose(out_array_indices)) <
                     tolerance)
    return antisymmetric


def get_kconserv3(cell, kpts, kijkab):
    r'''Get the momentum conservation array for a set of k-points.

    This function is similar to get_kconserv, but instead finds the 'kc'
    that satisfies momentum conservation for 5 k-points,

        (ki + kj + kk - ka - kb - kc) dot a = 2n\pi

    where these kpoints are stored in kijkab[ki, kj, kk, ka, kb].
    '''
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
                 if not isinstance(x, (int, np.integer))]
    kconserv = kconserv.reshape(new_shape)
    return kconserv


class VectorComposer(object):
    def __init__(self, dtype):
        """
        Composes vectors.
        Args:
            dtype (type): array data type;
        """
        self.__dtype__ = dtype
        self.__transactions__ = []
        self.__total_size__ = 0
        self.__data__ = None

    def put(self, a):
        """
        Puts array into vector.
        Args:
            a (ndarray): array to put;
        """
        if a.dtype != self.__dtype__:
            raise ValueError("dtype mismatch: passed %s vs expected %s" % (a.dtype, self.dtype))
        self.__transactions__.append(a)
        self.__total_size__ += a.size

    def flush(self):
        """
        Composes the vector.
        Returns:
            The composed vector.
        """
        if self.__data__ is None:
            self.__data__ = result = np.empty(self.__total_size__, dtype=self.__dtype__)
            offset = 0
        else:
            offset = self.__data__.size
            self.__data__ = result = np.empty(self.__total_size__ + self.__data__.size, dtype=self.__dtype__)

        for i in self.__transactions__:
            s = i.size
            result[offset:offset + s] = i.reshape(-1)
            offset += s
        self.__transactions__ = []

        return result


class VectorSplitter(object):
    def __init__(self, vector):
        """
        Splits vectors into pieces.
        Args:
            vector (ndarray): vector to split;
        """
        self.__data__ = vector
        self.__offset__ = 0

    def get(self, destination, slc=None):
        """
        Retrieves the next array.
        Args:
            destination: the shape of the destination array or the destination array itself;
            slc: an optional slice;

        Returns:
            The array.
        """
        if isinstance(destination, Number):
            destination = np.zeros((destination,), dtype=self.__data__.dtype)
        elif isinstance(destination, tuple):
            destination = np.zeros(destination, dtype=self.__data__.dtype)
        elif isinstance(destination, np.ndarray):
            pass
        else:
            raise ValueError("Unknown destination: %s" % str(destination))

        if slc is None:
            take_size = np.prod(destination.shape)
            take_shape = destination.shape
        else:
            slc = np.ix_(*slc)
            take_size = destination[slc].size
            take_shape = destination[slc].shape

        avail = self.__data__.size - self.__offset__
        if take_size > avail:
            raise ValueError("Insufficient # of elements: required %d %s, found %d" % (take_size, take_shape, avail))

        if slc is None:
            destination[:] = self.__data__[self.__offset__:self.__offset__ + take_size].reshape(take_shape)
        else:
            destination[slc] = self.__data__[self.__offset__:self.__offset__ + take_size].reshape(take_shape)

        self.__offset__ += take_size
        return destination

    def truncate(self):
        """
        Truncates the data vector.
        """
        self.__data__ = self.__data__[self.__offset__:].copy()
        self.__offset__ = 0


class KptsHelper(lib.StreamObject):
    def __init__(self, cell, kpts, init_symm_map=True):
        '''Helper class for handling k-points in correlated calculations.

        Attributes:
            kconserv : (nkpts,nkpts,nkpts) ndarray
                The index of the fourth momentum-conserving k-point, given
                indices of three k-points
            symm_map : OrderedDict of list of (3,) tuples
                Keys are (3,) tuples of symmetry-unique k-point indices and
                values are lists of (3,) tuples, enumerating all
                symmetry-related k-point indices for ERI generation
            init_symm_map : bool, optional
                Whether to build `symm_map` at initialization. Default is True.
        '''
        from pyscf.pbc.lib.kpts import KPoints
        if isinstance(kpts, KPoints):
            self.kconserv = kpts.get_kconserv()
            nkpts = kpts.nkpts
        else:
            self.kconserv = get_kconserv(cell, kpts)
            nkpts = len(kpts)

        self.nkpts = nkpts
        self._operation = None
        self.symm_map = None
        if init_symm_map:
            self.build_symm_map()

    def build_symm_map(self, kptlist=None):
        nkpts = self.nkpts
        if kptlist is None:
            _isin = lambda *args: True
        else:
            _isin = lambda v, vs: lib.isin_1d(v,vs)

        if kptlist is None:
            temp = range(0, nkpts)
            kptlist = lib.cartesian_prod((temp,temp,temp))

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

                if _isin([kr,ks,kp], kptlist):
                    completed[kr,ks,kp] = True
                    self._operation[kr,ks,kp] = 1 #.transpose(2,3,0,1)
                    self.symm_map[kpt].append((kr,ks,kp))

                if _isin([kq,kp,ks], kptlist):
                    completed[kq,kp,ks] = True
                    self._operation[kq,kp,ks] = 2 #np.conj(.transpose(1,0,3,2))
                    self.symm_map[kpt].append((kq,kp,ks))

                if _isin([ks,kr,kq], kptlist):
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

class KPointSymmetryError(RuntimeError):
    pass
