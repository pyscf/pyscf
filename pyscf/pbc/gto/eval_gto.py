#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import numpy
from pyscf import lib
from pyscf.gto.moleintor import make_loc

BLKSIZE = 128 # needs to be the same to lib/gto/grid_ao_drv.c

libpbc = lib.load_library('libpbc')

def eval_gto(cell, eval_name, coords, comp=1, kpts=None, kpt=None,
             shls_slice=None, non0tab=None, ao_loc=None, out=None):
    r'''Evaluate PBC-AO function value on the given grids,

    Args:
        eval_name : str

            ==========================  =======================
            Function                    Expression
            ==========================  =======================
            "PBCval_sph"                |AO>
            "PBCval_ip_sph"             nabla |AO>
            "PBCval_cart"               |AO>
            "PBCval_ip_cart"            nabla |AO>
            ==========================  =======================

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
            evaluated.  By default, all shells defined in cell will be evaluated.
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`dft.gen_grid.make_mask`
        out : ndarray
            If provided, results are written into this array.

    Returns:
        A list of 2D (or 3D) arrays to hold the AO values on grids.  Each
        element of the list corresponds to a k-point and it has the shape
        (N,nao) Or shape (\*,N,nao).

    Examples:

    >>> cell = pbc.gto.M(a=numpy.eye(3)*4, atom='He 1 1 1', basis='6-31g')
    >>> coords = cell.get_uniform_grids([20,20,20])
    >>> kpts = cell.make_kpts([3,3,3])
    >>> ao_value = cell.eval_gto("GTOval_sph", coords, kpts)
    >>> len(ao_value)
    27
    >>> ao_value[0].shape
    (100, 2)
    >>> ao_value = cell.eval_gto("GTOval_ig_sph", coords, kpts, comp=3)
    >>> print(ao_value.shape)
    >>> len(ao_value)
    27
    >>> ao_value[0].shape
    (3, 100, 2)
    '''
    if not ('_sph' in eval_name or '_cart' in eval_name or
            '_spinor' in eval_name):
        if cell.cart:
            eval_name = eval_name + '_cart'
        else:
            eval_name = eval_name + '_sph'

    atm = numpy.asarray(cell._atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(cell._bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(cell._env, dtype=numpy.double, order='C')
    natm = atm.shape[0]
    nbas = bas.shape[0]
    if kpts is None:
        if kpt is not None:
            kpts_lst = numpy.reshape(kpt, (1,3))
        else:
            kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)
    ngrids = len(coords)

    if non0tab is None:
        non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE, nbas),
                              dtype=numpy.uint8)
# non0tab stores the number of images to be summed in real space.
# Initializing it to 255 means all images are summed
        non0tab[:] = 0xff

    if ao_loc is None:
        ao_loc = make_loc(bas, eval_name)
    if shls_slice is None:
        shls_slice = (0, nbas)
    sh0, sh1 = shls_slice
    nao = ao_loc[sh1] - ao_loc[sh0]

    ao_kpts = [numpy.zeros((ngrids,nao,comp), dtype=numpy.complex128, order='F')
               for k in range(nkpts)]
    out_ptrs = (ctypes.c_void_p*nkpts)(
            *[x.ctypes.data_as(ctypes.c_void_p) for x in ao_kpts])
    coords = numpy.asarray(coords, order='F')
    Ls = cell.get_lattice_Ls(dimension=3)
    Ls = Ls[numpy.argsort(lib.norm(Ls, axis=1))]
    expLk = numpy.exp(1j * numpy.asarray(numpy.dot(Ls, kpts_lst.T), order='C'))

    drv = getattr(libpbc, eval_name)
    drv(ctypes.c_int(ngrids),
        (ctypes.c_int*2)(*shls_slice), ao_loc.ctypes.data_as(ctypes.c_void_p),
        Ls.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(Ls)),
        expLk.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nkpts),
        out_ptrs, coords.ctypes.data_as(ctypes.c_void_p),
        non0tab.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
        env.ctypes.data_as(ctypes.c_void_p))

    for k, kpt in enumerate(kpts_lst):
        if abs(kpt).sum() < 1e-9:
            ao_kpts[k] = ao_kpts[k].real.copy(order='F')

        ao_kpts[k] = ao_kpts[k].transpose(2,0,1)
        if comp == 1:
            ao_kpts[k] = ao_kpts[k][0]
    if kpts is None or numpy.shape(kpts) == (3,):  # A single k-point
        ao_kpts = ao_kpts[0]
    return ao_kpts


if __name__ == '__main__':
    from pyscf.pbc import gto, dft
    cell = gto.M(a=numpy.eye(3)*4, atom='He 1 1 1', basis='6-31g')
    coords = cell.get_uniform_grids([10]*3)
    ao_value = eval_gto(cell, "PBCval_sph", coords, kpts=cell.make_kpts([3]*3))
    print(lib.finger(numpy.asarray(ao_value)) - 0.542179662042965-0.12290561920251104j)
    print(ao_value[0].shape)
