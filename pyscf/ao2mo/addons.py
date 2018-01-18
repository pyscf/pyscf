#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import tempfile
import numpy
import h5py
import pyscf.lib

libao2mo = pyscf.lib.load_library('libao2mo')

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
            self.feri = h5py.File(self.eri, 'r')
            return self.feri[self.dataname]
        elif (hasattr(self.eri, 'read') or #isinstance(self.eri, file) or
              isinstance(self.eri, tempfile._TemporaryFileWrapper)):
            self.feri = h5py.File(self.eri.name)
            return self.feri[self.dataname]
        else:
            return self.eri

    def __exit__(self, type, value, traceback):
        if (isinstance(self.eri, str) or
            (hasattr(self.eri, 'read') or
             isinstance(self.eri, tempfile._TemporaryFileWrapper))):
            self.feri.close()


def restore(symmetry, eri, norb, tao=None):
    r'''Convert the 2e integrals between different level of permutation symmetry
    (8-fold, 4-fold, or no symmetry)

    Args:
        symmetry : int or str
            code to present the target symmetry of 2e integrals

            | 's8' or '8' or 8 : 8-fold symmetry
            | 's4' or '4' or 4 : 4-fold symmetry
            | 's1' or '1' or 1 : no symmetry
            | 's2ij' or '2ij' : symmetric ij pair for (ij|kl) (TODO)
            | 's2ij' or '2kl' : symmetric kl pair for (ij|kl) (TODO)

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

    eri = numpy.asarray(eri, order='C')
    npair = norb*(norb+1)//2
    if eri.size == norb**4:
        origsym = '1'
        if targetsym == '1':
            eri = eri.reshape(norb,norb,norb,norb)
        elif targetsym == '2kl':
            raise KeyError('TODO')
        elif targetsym == '2ij':
            raise KeyError('TODO')
    elif eri.size == npair**2:
        origsym = '4'
        if targetsym == '4':
            eri = eri.reshape(npair,npair)
        elif targetsym == '8':
            return pyscf.lib.pack_tril(eri.reshape(npair,-1))
        elif targetsym == '2kl':
            raise KeyError('TODO')
        elif targetsym == '2ij':
            raise KeyError('TODO')
    elif eri.size == npair*(npair+1)//2: # 8-fold
        origsym = '8'
        if targetsym == '4':
            return pyscf.lib.unpack_tril(eri.ravel())
        elif targetsym == '2kl':
            raise KeyError('TODO')
        elif targetsym == '2ij':
            raise KeyError('TODO')
    elif eri.size == npair*norb**2 and eri.shape[0] == npair:
        raise KeyError('TODO')
    elif eri.size == npair*norb**2 and eri.shape[-1] == npair:
        raise KeyError('TODO')
    else:
        raise ValueError('eri.size = %d, norb = %d' % (eri.size, norb))

    if origsym == targetsym:
        return eri

    if targetsym == '1':
        eri1 = numpy.empty((norb,norb,norb,norb), dtype=eri.dtype)
    elif targetsym == '4':
        eri1 = numpy.empty((npair,npair), dtype=eri.dtype)
    elif targetsym == '8':
        eri1 = numpy.empty(npair*(npair+1)//2, dtype=eri.dtype)

    return _call_restore(origsym, targetsym, eri, eri1, norb)

def _call_restore(origsym, targetsym, eri, eri1, norb, tao=None):
    if numpy.iscomplexobj(eri):
        raise RuntimeError('TODO')
        #if tao is None:
        #    raise RuntimeError('need time-reversal mapping')
        #fn = getattr(libao2mo, 'AO2MOrestore_r'+fname)
    else:
        fn = getattr(libao2mo, 'AO2MOrestore_nr%sto%s'%(origsym,targetsym))
    fn(eri.ctypes.data_as(ctypes.c_void_p),
       eri1.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(norb))
    return eri1

def _stand_sym_code(sym):
    if isinstance(sym, int):
        return str(sym)
    elif 's' == sym[0] or 'a' == sym[0]:
        return sym[1:]
    else:
        return sym

