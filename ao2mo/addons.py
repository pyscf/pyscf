#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import os
import ctypes
import numpy
import pyscf.lib

libao2mo = pyscf.lib.load_library('libao2mo')

def restore(symmetry, eri, norb, tao=None):
    r'''Transfer the 2e integrals between different level of permutation symmetry
    (8-fold, 4-fold, or no symmetry)

    Args:
        symmetry : int or str
            code to present the target symmetry of 2e integrals

            | 8 : 8-fold symmetry
            | 4 : 4-fold symmetry
            | 1 : no symmetry
            | '2ij' : symmetric ij pair for (ij|kl) (TODO)
            | '2kl' : symmetric kl pair for (ij|kl) (TODO)

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
    >>> mol = gto.Mole()
    >>> mol.build(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
    >>> eri = _vhf.int2e_sph(mol._atm, mol._bas, mol._env)
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
    if symmetry not in (8, 4, 1):
        raise ValueError('symmetry = %s' % symmetry)

    eri = numpy.ascontiguousarray(eri)
    npair = norb*(norb+1)//2
    if eri.size == norb**4:
        if symmetry == 1:
            return eri.reshape(norb,norb,norb,norb)
        elif symmetry == 4:
            eri1 = numpy.empty((npair,npair), dtype=eri.dtype)
            return _call_restore('1to4', eri, eri1, norb)
        else: # 8-fold
            eri1 = numpy.empty(npair*(npair+1)//2, dtype=eri.dtype)
            return _call_restore('1to8', eri, eri1, norb)
    elif eri.size == npair**2:
        if symmetry == 1:
            eri1 = numpy.empty((norb,norb,norb,norb), dtype=eri.dtype)
            return _call_restore('4to1', eri, eri1, norb)
        elif symmetry == 4:
            return eri.reshape(npair,npair)
        else: # 8-fold
            #return _ao2mo.restore_4to8(eri, norb)
            return pyscf.lib.pack_tril(eri.reshape(npair,-1))
    elif eri.size == npair*(npair+1)//2: # 8-fold
        if symmetry == 1:
            eri1 = numpy.empty((norb,norb,norb,norb), dtype=eri.dtype)
            return _call_restore('8to1', eri, eri1, norb)
        elif symmetry == 4:
            #return _ao2mo.restore_8to4(eri, norb)
            return pyscf.lib.unpack_tril(eri.ravel())
        else: # 8-fold
            return eri
    else:
        raise ValueError('eri.size = %d, norb = %d' % (eri.size, norb))

def _call_restore(fname, eri, eri1, norb, tao=None):
    if numpy.iscomplexobj(eri):
        raise RuntimeError('TODO')
        #if tao is None:
        #    raise RuntimeError('need time-reversal mapping')
        fn = getattr(libao2mo, 'AO2MOrestore_r'+fname)
    else:
        fn = getattr(libao2mo, 'AO2MOrestore_nr'+fname)
    fn(eri.ctypes.data_as(ctypes.c_void_p),
       eri1.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(norb))
    return eri1

