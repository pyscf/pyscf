#!/usr/bin/env python
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import tempfile
import numpy
import h5py
from pyscf import lib

libao2mo = lib.load_library('libao2mo')

class load(object):
    '''load 2e integrals from hdf5 file

    Usage:
        with load(erifile) as eri:
            print(eri.shape)
    '''
    def __init__(self, eri, dataname='eri_mo'):
        self.eri = eri
        self.dataname = dataname
        self.feri = None

    def __enter__(self):
        if isinstance(self.eri, str):
            feri = self.feri = h5py.File(self.eri, 'r')
        elif isinstance(self.eri, h5py.Group):
            feri = self.eri
        elif isinstance(getattr(self.eri, 'name', None), str):
            feri = self.feri = h5py.File(self.eri.name, 'r')
        elif isinstance(self.eri, numpy.ndarray):
            return self.eri
        else:
            raise RuntimeError('Unknown eri type %s', type(self.eri))

        if self.dataname is None:
            return feri
        else:
            return feri[self.dataname]

    def __exit__(self, type, value, traceback):
        if self.feri is not None:
            self.feri.close()


def restore(symmetry, eri, norb, tao=None):
    r'''Convert the 2e integrals (in Chemist's notation) between different
    level of permutation symmetry (8-fold, 4-fold, or no symmetry)

    Args:
        symmetry : int or str
            code to present the target symmetry of 2e integrals

            | 's8' or '8' or 8 : 8-fold symmetry
            | 's4' or '4' or 4 : 4-fold symmetry
            | 's1' or '1' or 1 : no symmetry
            | 's2ij' or '2ij' : symmetric ij pair for (ij|kl) (TODO)
            | 's2ij' or '2kl' : symmetric kl pair for (ij|kl) (TODO)

            Note the 4-fold symmetry requires (ij|kl) == (ij|lk) == (ij|lk)
            while (ij|kl) != (kl|ij) is not required.

        eri : ndarray
            The symmetry of eri is determined by the size of eri and norb
        norb : int
            The symmetry of eri is determined by the size of eri and norb

    Returns:
        ndarray.  The shape depends on the target symmetry.

            | 8 : (norb*(norb+1)/2)*(norb*(norb+1)/2+1)/2
            | 4 : (norb*(norb+1)/2, norb*(norb+1)/2)
            | 1 : (norb, norb, norb, norb)

    Examples:

    >>> from pyscf import gto
    >>> from pyscf.scf import _vhf
    >>> from pyscf import ao2mo
    >>> mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
    >>> eri = mol.intor('int2e')
    >>> eri1 = ao2mo.restore(1, eri, mol.nao_nr())
    >>> eri4 = ao2mo.restore(4, eri, mol.nao_nr())
    >>> eri8 = ao2mo.restore(8, eri, mol.nao_nr())
    >>> print(eri1.shape)
    (7, 7, 7, 7)
    >>> print(eri1.shape)
    (28, 28)
    >>> print(eri1.shape)
    (406,)
    '''
    targetsym = _stand_sym_code(symmetry)
    if targetsym not in ('8', '4', '1', '2kl', '2ij'):
        raise ValueError('symmetry = %s' % symmetry)

    if eri.dtype != numpy.double:
        raise RuntimeError('Complex integrals not supported')

    eri = numpy.asarray(eri, order='C')
    npair = norb*(norb+1)//2
    if eri.size == norb**4:  # s1
        if targetsym == '1':
            return eri.reshape(norb,norb,norb,norb)
        elif targetsym == '2kl':
            eri = lib.pack_tril(eri.reshape(norb**2,norb,norb))
            return eri.reshape(norb,norb,npair)
        elif targetsym == '2ij':
            eri = lib.pack_tril(eri.reshape(norb,norb,norb**2), axis=0)
            return eri.reshape(npair,norb,norb)
        else:
            return _convert('1', targetsym, eri, norb)

    elif eri.size == npair**2:  # s4
        if targetsym == '4':
            return eri.reshape(npair,npair)
        elif targetsym == '8':
            return lib.pack_tril(eri.reshape(npair,npair))
        elif targetsym == '2kl':
            return lib.unpack_tril(eri, lib.SYMMETRIC, axis=0)
        elif targetsym == '2ij':
            return lib.unpack_tril(eri, lib.SYMMETRIC, axis=-1)
        else:
            return _convert('4', targetsym, eri, norb)

    elif eri.size == npair*(npair+1)//2: # 8-fold
        if targetsym == '8':
            return eri.ravel()
        elif targetsym == '4':
            return lib.unpack_tril(eri.ravel(), lib.SYMMETRIC)
        elif targetsym == '2kl':
            return lib.unpack_tril(lib.unpack_tril(eri.ravel()), lib.SYMMETRIC, axis=0)
        elif targetsym == '2ij':
            return lib.unpack_tril(lib.unpack_tril(eri.ravel()), lib.SYMMETRIC, axis=-1)
        else:
            return _convert('8', targetsym, eri, norb)

    elif eri.size == npair*norb**2 and eri.shape[0] == npair:  # s2ij
        if targetsym == '2ij':
            return eri.reshape(npair,norb,norb)
        elif targetsym == '8':
            eri = lib.pack_tril(eri.reshape(npair,norb,norb))
            return lib.pack_tril(eri)
        elif targetsym == '4':
            return lib.pack_tril(eri.reshape(npair,norb,norb))
        elif targetsym == '1':
            eri = lib.unpack_tril(eri.reshape(npair,norb**2), lib.SYMMETRIC, axis=0)
            return eri.reshape(norb,norb,norb,norb)
        elif targetsym == '2kl':
            tril2sq = lib.square_mat_in_trilu_indices(norb)
            trilidx = numpy.tril_indices(norb)
            eri = lib.take_2d(eri.reshape(npair,norb**2), tril2sq.ravel(),
                              trilidx[0]*norb+trilidx[1])
            return eri.reshape(norb,norb,npair)

    elif eri.size == npair*norb**2 and eri.shape[-1] == npair:  # s2kl
        if targetsym == '2kl':
            return eri.reshape(norb,norb,npair)
        elif targetsym == '8':
            eri = lib.pack_tril(eri.reshape(norb,norb,npair), axis=0)
            return lib.pack_tril(eri)
        elif targetsym == '4':
            return lib.pack_tril(eri.reshape(norb,norb,npair), axis=0)
        elif targetsym == '1':
            eri = lib.unpack_tril(eri.reshape(norb**2,npair), lib.SYMMETRIC, axis=-1)
            return eri.reshape(norb,norb,norb,norb)
        elif targetsym == '2ij':
            tril2sq = lib.square_mat_in_trilu_indices(norb)
            trilidx = numpy.tril_indices(norb)
            eri = lib.take_2d(eri.reshape(norb**2,npair),
                              trilidx[0]*norb+trilidx[1], tril2sq.ravel())
            return eri.reshape(npair,norb,norb)

    else:
        raise RuntimeError('eri.size = %d, norb = %d' % (eri.size, norb))

def _convert(origsym, targetsym, eri, norb):
    fn = getattr(libao2mo, 'AO2MOrestore_nr%sto%s'%(origsym,targetsym))
    npair = norb*(norb+1)//2
    if targetsym == '1':
        eri1 = numpy.empty((norb,norb,norb,norb), dtype=eri.dtype)
    elif targetsym == '4':
        eri1 = numpy.empty((npair,npair), dtype=eri.dtype)
    elif targetsym == '8':
        eri1 = numpy.empty(npair*(npair+1)//2, dtype=eri.dtype)

    fn(eri.ctypes.data_as(ctypes.c_void_p),
       eri1.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(norb))

    return eri1

def _stand_sym_code(sym):
    if isinstance(sym, int):
        return str(sym)
    elif 's' == sym[0]:
        return sym[1:]
    else:
        return sym

