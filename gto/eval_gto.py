#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import ctypes
import _ctypes
import pyscf.lib

BLKSIZE = 96 # needs to be the same to lib/gto/grid_ao_drv.c
ANG_OF     = 1
NPRIM_OF   = 2
NCTR_OF    = 3
KAPPA_OF   = 4
PTR_EXP    = 5
PTR_COEFF  = 6
BAS_SLOTS  = 8

try:
    libcgto = pyscf.lib.load_library('libdft')
except ImportError:
    libcgto = pyscf.lib.load_library('libcgto')
def _fpointer(name):
    return ctypes.c_void_p(_ctypes.dlsym(libcgto._handle, name))

def eval_gto(eval_name, atm, bas, env, coords,
             comp=1, shls_slice=None, non0tab=None, out=None):
    '''Evaluate AO function value on the given grids,

    Args:
        eval_name : str

            ==========================  =========  =============
            Function                    type       Expression
            ==========================  =========  =============
            "GTOval_sph"                spheric    |AO>
            "GTOval_ip_sph"             spheric    nabla |AO>
            "GTOval_ig_sph"             spheric    (#C(0 1) g) |AO>
            "GTOval_ipig_sph"           spheric    (#C(0 1) nabla g) |AO>
            "GTOval_cart"               cart       |AO>
            "GTOval_ip_cart"            cart       nabla |AO>
            "GTOval_ig_cart"            cart       (#C(0 1) g)|AO>
            ==========================  =========  =============

        atm : int32 ndarray
            libcint integral function argument
        bas : int32 ndarray
            libcint integral function argument
        env : float64 ndarray
            libcint integral function argument

        coords : 2D array, shape (N,3)
            The coordinates of the grids.

    Kwargs:
        shls_slice : 2-element list
            (shl_start, shl_end).
            If given, only part of AOs (shl_start <= shell_id < shl_end) are
            evaluated.  By default, all shells defined in mol will be evaluated.
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`
        out : ndarray
            If provided, results are written into this array.

    Returns:
        2D array of shape (N,nao) Or 3D array of shape (*,N,nao) for AO values

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz')
    >>> coords = numpy.random.random((100,3))  # 100 random points
    >>> ao_value = eval_gto("GTOval_sph", mol._atm, mol._bas, mol._env, coords)
    >>> print(ao_value.shape)
    (100, 24)
    >>> ao_value = eval_gto("GTOval_ig_sph", mol._atm, mol._bas, mol._env, coords, comp=3)
    >>> print(ao_value.shape)
    (3, 100, 24)
    '''
    atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(env, dtype=numpy.double, order='C')
    coords = numpy.asarray(coords, order='C')
    natm = atm.shape[0]
    nbas = bas.shape[0]
    ngrids = coords.shape[0]

    if shls_slice is None:
        shls_slice = (0, nbas)
    bastart, basend = shls_slice
    bascount = basend - bastart

    if '_cart' in eval_name:
        dtype = numpy.double
        l = bas[bastart:basend,ANG_OF]
        nao = ((l+1)*(l+2)//2 * bas[bastart:basend,NCTR_OF]).sum()
    elif '_sph' in eval_name:
        dtype = numpy.double
        l = bas[bastart:basend,ANG_OF]
        nao = ((l*2+1) * bas[bastart:basend,NCTR_OF]).sum()
    else:
        raise NotImplementedError(eval_name)

    if out is None:
        ao = numpy.empty((comp,ngrids,nao))
    else:
        ao = numpy.ndarray((comp,ngrids,nao), buffer=out)

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,nbas),
                             dtype=numpy.int8)

    drv = getattr(libcgto, eval_name)
    drv(ctypes.c_int(nao), ctypes.c_int(ngrids),
        ctypes.c_int(BLKSIZE), ctypes.c_int(bastart), ctypes.c_int(bascount),
        ao.ctypes.data_as(ctypes.c_void_p),
        coords.ctypes.data_as(ctypes.c_void_p),
        non0tab.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
        env.ctypes.data_as(ctypes.c_void_p))
    if comp == 1:
        return ao.reshape(ngrids,nao)
    else:
        return ao

if __name__ == '__main__':
    from pyscf import gto
    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz')
    coords = numpy.random.random((100,3))
    ao_value = eval_gto("GTOval_sph", mol._atm, mol._bas, mol._env, coords)
    print(ao_value.shape)
