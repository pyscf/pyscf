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

import warnings
import ctypes
import numpy
from pyscf import lib
from pyscf.gto.moleintor import make_loc

BLKSIZE = 128 # needs to be the same to lib/gto/grid_ao_drv.c

libcgto = lib.load_library('libcgto')

def eval_gto(mol, eval_name, coords,
             comp=None, shls_slice=None, non0tab=None, ao_loc=None, out=None):
    r'''Evaluate AO function value on the given grids,

    Args:
        eval_name : str

            ==================  ======  =======================
            Function            comp    Expression
            ==================  ======  =======================
            "GTOval_sph"        1       |AO>
            "GTOval_ip_sph"     3       nabla |AO>
            "GTOval_ig_sph"     3       (#C(0 1) g) |AO>
            "GTOval_ipig_sph"   3       (#C(0 1) nabla g) |AO>
            "GTOval_cart"       1       |AO>
            "GTOval_ip_cart"    3       nabla |AO>
            "GTOval_ig_cart"    3       (#C(0 1) g)|AO>
            ==================  ======  =======================

        atm : int32 ndarray
            libcint integral function argument
        bas : int32 ndarray
            libcint integral function argument
        env : float64 ndarray
            libcint integral function argument

        coords : 2D array, shape (N,3)
            The coordinates of the grids.

    Kwargs:
        comp : int
            Number of the components of the operator
        shls_slice : 2-element list
            (shl_start, shl_end).
            If given, only part of AOs (shl_start <= shell_id < shl_end) are
            evaluated.  By default, all shells defined in mol will be evaluated.
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`dft.gen_grid.make_mask`
        out : ndarray
            If provided, results are written into this array.

    Returns:
        2D array of shape (N,nao) Or 3D array of shape (\*,N,nao) to store AO
        values on grids.

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz')
    >>> coords = numpy.random.random((100,3))  # 100 random points
    >>> ao_value = mol.eval_gto("GTOval_sph", coords)
    >>> print(ao_value.shape)
    (100, 24)
    >>> ao_value = mol.eval_gto("GTOval_ig_sph", coords)
    >>> print(ao_value.shape)
    (3, 100, 24)
    '''
    eval_name, comp = _get_intor_and_comp(mol, eval_name, comp)

    atm = numpy.asarray(mol._atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(mol._bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(mol._env, dtype=numpy.double, order='C')
    coords = numpy.asarray(coords, dtype=numpy.double, order='F')
    natm = atm.shape[0]
    nbas = bas.shape[0]
    ngrids = coords.shape[0]

    if ao_loc is None:
        ao_loc = make_loc(bas, eval_name)

    if shls_slice is None:
        shls_slice = (0, nbas)
    sh0, sh1 = shls_slice
    nao = ao_loc[sh1] - ao_loc[sh0]
    if 'spinor' in eval_name:
        ao = numpy.ndarray((2,comp,nao,ngrids), dtype=numpy.complex128, buffer=out)
    else:
        ao = numpy.ndarray((comp,nao,ngrids), buffer=out)

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,nbas),
                             dtype=numpy.int8)

    drv = getattr(libcgto, eval_name)
    drv(ctypes.c_int(ngrids),
        (ctypes.c_int*2)(*shls_slice), ao_loc.ctypes.data_as(ctypes.c_void_p),
        ao.ctypes.data_as(ctypes.c_void_p),
        coords.ctypes.data_as(ctypes.c_void_p),
        non0tab.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
        env.ctypes.data_as(ctypes.c_void_p))

    ao = numpy.swapaxes(ao, -1, -2)
    if comp == 1:
        if 'spinor' in eval_name:
            ao = ao[:,0]
        else:
            ao = ao[0]
    return ao

def _get_intor_and_comp(mol, eval_name, comp=None):
    if not ('_sph' in eval_name or '_cart' in eval_name or
            '_spinor' in eval_name):
        if mol.cart:
            eval_name = eval_name + '_cart'
        else:
            eval_name = eval_name + '_sph'

    if comp is None:
        if '_spinor' in eval_name:
            fname = eval_name.replace('_spinor', '')
            comp = _GTO_EVAL_FUNCTIONS.get(fname, (None,None))[1]
        else:
            fname = eval_name.replace('_sph', '').replace('_cart', '')
            comp = _GTO_EVAL_FUNCTIONS.get(fname, (None,None))[0]
        if comp is None:
            warnings.warn('Function %s not found.  Set its comp to 1' % eval_name)
            comp = 1
    return eval_name, comp

_GTO_EVAL_FUNCTIONS = {
#   Functiona name          : (comp-for-scalar, comp-for-spinor)
    'GTOval'                : (1, 1 ),
    'GTOval_ip'             : (3, 3 ),
    'GTOval_ig'             : (3, 3 ),
    'GTOval_ipig'           : (3, 3 ),
    'GTOval_deriv0'         : (1, 1 ),
    'GTOval_deriv1'         : (4, 4 ),
    'GTOval_deriv2'         : (10,10),
    'GTOval_deriv3'         : (20,20),
    'GTOval_deriv4'         : (35,35),
    'GTOval_sp'             : (4, 1 ),
    'GTOval_ipsp'           : (12,3 ),
}


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz')
    coords = numpy.random.random((100,3))
    ao_value = eval_gto(mol, "GTOval_sph", coords)
    print(ao_value.shape)
