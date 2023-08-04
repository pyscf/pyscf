#!/usr/bin/env python
# Copyright 2022-2023 The PySCF Developers. All Rights Reserved.
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
# Authors: Xing Zhang <zhangxing.nju@gmail.com>
#

import warnings
from functools import reduce
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from pyscf import lib


def empty(shape, dtype=float, order='C', metadata=None):
    if metadata is None:
        return np.empty(shape, dtype, order)
    else:
        return KsymmArray(shape, dtype, order, metadata)

def empty_like(a, *args, **kwargs):
    if isinstance(a, KsymmArray):
        return KsymmArray(a.subarray_shape,
                          a.dtype,
                          a.subarray_order,
                          a.metadata)
    else:
        return np.empty_like(a)


class KsymmArray(NDArrayOperatorsMixin):
    def __init__(self, subarray_shape, dtype=float, subarray_order='C', metadata={},
                 init_with_zeros=False):
        self.metadata = metadata
        self._subarray_shape = list(subarray_shape)
        self._subarray_ndim = len(subarray_shape)
        self._subarray_order = subarray_order
        self._dtype = np.dtype(dtype)
        incore = metadata.get('incore', True)
        self._datafile = None
        self.data = self._init(subarray_order, incore, init_with_zeros)

    def _init(self, order, incore=True, init_with_zeros=False):
        if self.subarray_ndim == 2:
            kpts = self.metadata['kpts']
            n_subarray = kpts.nkpts_ibz
        elif self.subarray_ndim == 4:
            kqrts = self.metadata['kqrts']
            n_subarray = len(kqrts.kqrts_ibz)
        else:
            raise NotImplementedError

        data = None
        shape = [n_subarray,] + self.subarray_shape
        if incore:
            if init_with_zeros:
                fn_init = np.zeros
            else:
                fn_init = np.empty
            if order == 'C':
                data = fn_init(shape, self.dtype, order)
            else:
                data = []
                for i in range(n_subarray):
                    data.append(fn_init(self.subarray_shape, self.dtype, order))
                data = np.asarray(data, order='K')
        else:
            self._datafile = lib.H5TmpFile()
            data = self._datafile.create_dataset('data', shape, self.dtype.char)
        return data

    @property
    def shape(self):
        nkpts = self.metadata['kpts'].nkpts
        nk = [nkpts,] * (self.subarray_ndim-1)
        return tuple(nk + list(self.subarray_shape))

    @property
    def ndim(self):
        return self.subarray_ndim-1 + self.subarray_ndim

    @property
    def subarray_ndim(self):
        return self._subarray_ndim

    @property
    def subarray_shape(self):
        return self._subarray_shape

    @property
    def subarray_order(self):
        return self._subarray_order

    @property
    def dtype(self):
        return self._dtype

    def __getitem__(self, key):
        if self.subarray_ndim == 2:
            return self._getitem_2d(key)
        elif self.subarray_ndim == 4:
            return self._getitem_4d(key)
        else:
            raise NotImplementedError

    def __setitem__(self, key, value):
        if self.subarray_ndim == 2:
            return self._setitem_2d(key, value)
        elif self.subarray_ndim == 4:
            return self._setitem_4d(key, value)
        else:
            raise NotImplementedError

    def _getitem_2d(self, key):
        kpts = self.metadata['kpts']
        rmat = self.metadata['rmat']
        label = self.metadata['label']
        trans = self.metadata['trans']
        if isinstance(key, (int, np.integer)):
            return transform_2d(self.data, kpts, key, rmat, label, trans)
        elif isinstance(key, (slice, np.ndarray)):
            data = []
            for ki in np.arange(kpts.nkpts)[key]:
                data.append(transform_2d(self.data, kpts, ki, rmat, label, trans))
            return np.asarray(data)
        else:
            raise NotImplementedError

    def _getitem_4d(self, key):
        kpts = self.metadata['kpts']
        kqrts = self.metadata['kqrts']
        rmat = self.metadata['rmat']
        label = self.metadata['label']
        trans = self.metadata['trans']

        shape = [kpts.nkpts,] * 3
        coords = index_to_coords(key, shape)
        if coords.ndim == 1:
            klc = coords
            return transform_4d(self.data, kpts, kqrts, klc, rmat, label, trans)
        else:
            data = []
            for klc in coords:
                data.append(transform_4d(self.data, kpts, kqrts, klc, rmat, label, trans))
            return np.asarray(data)

    def _setitem_2d(self, key, value):
        kpts = self.metadata['kpts']
        #TODO allow broadcasting
        value = value.reshape(-1, *self.subarray_shape)
        if isinstance(key, (int, np.integer)):
            set_2d(self.data, value, kpts, key)
        elif isinstance(key, (slice, np.ndarray)):
            ki = np.arange(kpts.nkpts)[key]
            set_2d(self.data, value, kpts, ki)
        else:
            raise NotImplementedError

    def _setitem_4d(self, key, value):
        kpts = self.metadata['kpts']
        kqrts = self.metadata['kqrts']
        #TODO allow broadcasting
        value = value.reshape(-1, *self.subarray_shape)

        shape = [kpts.nkpts,] * 3
        coords = index_to_coords(key, shape)
        set_4d(self.data, value, kpts, kqrts, coords)

    def todense(self):
        #TODO allow to return a hdf5 dataset
        return self[:].reshape(self.shape)

    @staticmethod
    def fromdense(arr, shape, dtype=None, order=None, metadata=None):
        if dtype is None:
            dtype = arr.dtype
        order = _guess_input_order(arr, order)
        if metadata is None:
            raise RuntimeError('metadata not initialized')

        out = KsymmArray(shape, dtype, order, metadata)
        arr = arr.reshape(out.shape)
        if out.subarray_ndim == 2:
            kpts = out.metadata['kpts']
            for ki in kpts.ibz2bz:
                ki_ibz = kpts.bz2ibz[ki]
                out[ki_ibz] = arr[ki]
        elif out.subarray_ndim == 4:
            kqrts = out.metadata['kqrts']
            for m, kq in enumerate(kqrts.kqrts_ibz):
                ki, kj, ka, kb = kq
                out[m] = arr[ki, kj, ka]
        else:
            raise NotImplementedError
        return out

    @staticmethod
    def fromraw(arr, shape, dtype=None, order=None, metadata=None):
        if dtype is None:
            dtype = arr.dtype
        order = _guess_input_order(arr, order)
        if metadata is None:
            raise RuntimeError('metadata not initialized')

        out = KsymmArray(shape, dtype, order, metadata)
        arr = arr.reshape(-1, *out.subarray_shape)
        for i, a in enumerate(arr):
            out.data[i] = np.asarray(a, dtype=dtype, order=order)
        return out

    @staticmethod
    def zeros(shape, dtype=float, order='C', metadata=None):
        out = KsymmArray(shape, dtype, order, metadata, init_with_zeros=True)
        return out


def _guess_input_order(arr, order=None):
    if order is None:
        order = 'C'
        if isinstance(arr, np.ndarray):
            if arr.flags.c_contiguous:
                order = 'C'
            elif arr.flags.f_contiguous:
                order = 'F'
    elif order not in ('C', 'F'):
        order = 'C'
    return order

def set_2d(arr, value, kpts, ki):
    if isinstance(ki, (int, np.integer)):
        ki = np.asarray([ki,])

    mask = np.isin(ki, kpts.ibz2bz)
    if not mask.all():
        warnings.warn(f'Indices {ki[~mask]} are not in the irreducible wedge. '
                       'The corresponding data will be discarded.')

    ki_ibz = kpts.bz2ibz[ki[mask]]
    arr[ki_ibz] = value[mask]

def set_4d(arr, value, kpts, kqrts, klc):
    klc = klc.reshape(-1, 3)
    kk_bz = [kpts.ktuple_to_index(s) for s in klc]
    kk_bz = np.asarray(kk_bz)

    mask = np.isin(kk_bz, kqrts.ibz2bz)
    if not mask.all():
        kk_tmp = [kpts.index_to_ktuple(k, 3) for k in kk_bz[~mask]]
        warnings.warn(f'Indices {kk_tmp} are not in the irreducible wedge. '
                       'The corresponding data will be discarded.')

    kk_ibz = kqrts.bz2ibz[kk_bz[mask]]
    arr[kk_ibz] = value[mask]

def transform_2d(arr, kpts, ki, rmat, label, trans):
    ki_ibz = kpts.bz2ibz[ki]
    ki_ibz_bz = kpts.ibz2bz[ki_ibz]
    if ki == ki_ibz_bz:
        return arr[ki_ibz]

    pi, pj = label
    rmat_i = getattr(rmat, pi*2)
    rmat_j = getattr(rmat, pj*2)

    iop = kpts.stars_ops_bz[ki]
    rot_i = rmat_i[ki_ibz_bz][iop]
    rot_j = rmat_j[ki_ibz_bz][iop]
    ti, tj = trans
    if ti == 'c':
        rot_i = rot_i.conj()
    if tj == 'c':
        rot_j = rot_j.conj()

    #:einsum('ia,ij,ab->jb', arr[ki_ibz], rot_i, rot_j)
    out = reduce(np.dot, (rot_i.T, arr[ki_ibz], rot_j))
    return out

def transform_4d(arr, kpts, kqrts, klc, rmat, label, trans):
    kk_bz = kpts.ktuple_to_index(klc)
    kk_ibz = kqrts.bz2ibz[kk_bz]
    i,j,a,b = kqrts.kqrts_ibz[kk_ibz]
    if (i,j,a) == tuple(klc):
        return arr[kk_ibz]

    pi, pj, pa, pb = label
    rmat_i = getattr(rmat, pi*2)
    rmat_j = getattr(rmat, pj*2)
    rmat_a = getattr(rmat, pa*2)
    rmat_b = getattr(rmat, pb*2)

    iop = kqrts.stars_ops_bz[kk_bz]
    rot_i = rmat_i[i][iop]
    rot_j = rmat_j[j][iop]
    rot_a = rmat_a[a][iop]
    rot_b = rmat_b[b][iop]

    ti, tj, ta, tb = trans
    if ti == 'c':
        rot_i = rot_i.conj()
    if tj == 'c':
        rot_j = rot_j.conj()
    if ta == 'c':
        rot_a = rot_a.conj()
    if tb == 'c':
        rot_b = rot_b.conj()

    di, dj, da, db = arr[kk_ibz].shape
    #:einsum('ijab,ik,jl,ac,bd->klcd', arr[kk_ibz],
    #        rot_i, rot_j, rot_a, rot_b)

    #tmp = np.einsum('ik,ijab->jkab', rot_i, arr[kk_ibz])
    tmp = np.dot(rot_i.T, arr[kk_ibz].reshape(di,-1)) #k,jab
    tmp = tmp.reshape(di,dj,-1).transpose(1,0,2) #j,k,ab

    #tmp = np.einsum('jkab,jl->klab', tmp, rot_j)
    tmp = np.dot(rot_j.T, tmp.reshape(dj,-1)) #l,kab
    tmp = tmp.reshape(dj,di,-1).transpose(1,0,2) #k,l,ab

    #tmp = np.einsum('klab,ac->klbc', tmp, rot_a)
    tmp = tmp.reshape(-1,da,db).transpose(0,2,1).reshape(-1,da) #klb,a
    tmp = np.dot(tmp, rot_a) #klb,c

    #out = np.einsum('klbc,bd->klcd', tmp, rot_b)
    tmp = tmp.reshape(-1,db,da).transpose(0,2,1).reshape(-1,db) #klc,b
    out = np.dot(tmp, rot_b).reshape(di,dj,da,db) #k,l,c,d
    return out

def index_to_coords(key, shape):
    if not isinstance(key, tuple):
        key = (key,)

    idxs = []
    for i, k in enumerate(key):
        n = shape[i]
        if isinstance(k, slice):
            idx = slice_to_coords(k, n)
        elif isinstance(k, (int, np.integer)):
            idx = [k,]
        elif isinstance(k, np.ndarray) and k.ndim == 1:
            idx = k
        else:
            raise NotImplementedError
        idxs.append(idx)

    ndim = len(shape)
    if len(idxs) > ndim:
        raise RuntimeError
    elif len(idxs) < ndim:
        for i in range(len(idxs),ndim):
            n = shape[i]
            idxs.append(np.arange(n))

    coords = lib.cartesian_prod(idxs)
    if all(isinstance(k, (int, np.integer)) for k in key) and len(key) == ndim:
        coords = coords[0]
    return coords

def slice_to_coords(k, n):
    start, stop, step = k.start, k.stop, k.step
    if start is None:
        start = 0
    elif start < 0:
        start += n
    if stop is None:
        stop = n
    elif stop < 0:
        stop += n
    if step is None:
        step = 1
    return np.arange(start, stop, step)


zeros = KsymmArray.zeros
fromraw = KsymmArray.fromraw
fromdense = KsymmArray.fromdense
