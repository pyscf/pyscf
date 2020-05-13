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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import numpy
from pyscf import lib
from pyscf.gto import moleintor
from pyscf.gto.eval_gto import _get_intor_and_comp
from pyscf.pbc.gto import _pbcintor
from pyscf import __config__

BLKSIZE = 128 # needs to be the same to lib/gto/grid_ao_drv.c
EXTRA_PREC = getattr(__config__, 'pbc_gto_eval_gto_extra_precision', 1e-2)

libpbc = _pbcintor.libpbc

def eval_gto(cell, eval_name, coords, comp=None, kpts=None, kpt=None,
             shls_slice=None, non0tab=None, ao_loc=None, out=None):
    r'''Evaluate PBC-AO function value on the given grids,

    Args:
        eval_name : str

            ==========================  =======================
            Function                    Expression
            ==========================  =======================
            "GTOval_sph"                \sum_T exp(ik*T) |AO>
            "GTOval_ip_sph"             nabla \sum_T exp(ik*T) |AO>
            "GTOval_cart"               \sum_T exp(ik*T) |AO>
            "GTOval_ip_cart"            nabla \sum_T exp(ik*T) |AO>
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
    >>> ao_value = cell.pbc_eval_gto("GTOval_sph", coords, kpts)
    >>> len(ao_value)
    27
    >>> ao_value[0].shape
    (100, 2)
    >>> ao_value = cell.pbc_eval_gto("GTOval_ig_sph", coords, kpts, comp=3)
    >>> print(ao_value.shape)
    >>> len(ao_value)
    27
    >>> ao_value[0].shape
    (3, 100, 2)
    '''
    if eval_name[:3] == 'PBC':  # PBCGTOval_xxx
        eval_name, comp = _get_intor_and_comp(cell, eval_name[3:], comp)
    else:
        eval_name, comp = _get_intor_and_comp(cell, eval_name, comp)
    eval_name = 'PBC' + eval_name

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
        ao_loc = moleintor.make_loc(bas, eval_name)
    if shls_slice is None:
        shls_slice = (0, nbas)
    sh0, sh1 = shls_slice
    nao = ao_loc[sh1] - ao_loc[sh0]

    out = numpy.empty((nkpts,comp,nao,ngrids), dtype=numpy.complex128)
    coords = numpy.asarray(coords, order='F')

    # For atoms near the boundary of the cell, it is necessary (even in low-
    # dimensional systems) to include lattice translations in all 3 dimensions.
    if cell.dimension < 2 or cell.low_dim_ft_type == 'inf_vacuum':
        Ls = cell.get_lattice_Ls(dimension=cell.dimension)
    else:
        Ls = cell.get_lattice_Ls(dimension=3)
    Ls = Ls[numpy.argsort(lib.norm(Ls, axis=1))]
    expLk = numpy.exp(1j * numpy.asarray(numpy.dot(Ls, kpts_lst.T), order='C'))
    rcut = _estimate_rcut(cell)

    drv = getattr(libpbc, eval_name)
    drv(ctypes.c_int(ngrids),
        (ctypes.c_int*2)(*shls_slice), ao_loc.ctypes.data_as(ctypes.c_void_p),
        Ls.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(Ls)),
        expLk.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nkpts),
        out.ctypes.data_as(ctypes.c_void_p),
        coords.ctypes.data_as(ctypes.c_void_p),
        rcut.ctypes.data_as(ctypes.c_void_p),
        non0tab.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
        env.ctypes.data_as(ctypes.c_void_p))

    ao_kpts = []
    for k, kpt in enumerate(kpts_lst):
        v = out[k]
        if abs(kpt).sum() < 1e-9:
            v = numpy.asarray(v.real, order='C')
        v = v.transpose(0,2,1)
        if comp == 1:
            v = v[0]
        ao_kpts.append(v)

    if kpts is None or numpy.shape(kpts) == (3,):  # A single k-point
        ao_kpts = ao_kpts[0]
    return ao_kpts

pbc_eval_gto = eval_gto

def _estimate_rcut(cell):
    '''Cutoff raidus, above which each shell decays to a value less than the
    required precsion'''
    log_prec = numpy.log(cell.precision * EXTRA_PREC)
    rcut = []
    for ib in range(cell.nbas):
        l = cell.bas_angular(ib)
        es = cell.bas_exp(ib)
        cs = abs(cell.bas_ctr_coeff(ib)).max(axis=1)
        r = 5.
        r = (((l+2)*numpy.log(r)+numpy.log(cs) - log_prec) / es)**.5
        r = (((l+2)*numpy.log(r)+numpy.log(cs) - log_prec) / es)**.5
        rcut.append(r.max())
    return numpy.array(rcut)


if __name__ == '__main__':
    from pyscf.pbc import gto
    cell = gto.M(a=numpy.eye(3)*4, atom='He 1 1 1', basis=[[2,(1,.5),(.5,.5)]])
    coords = cell.get_uniform_grids([10]*3)
    ao_value = eval_gto(cell, "GTOval_sph", coords, kpts=cell.make_kpts([3]*3))
    print(lib.finger(numpy.asarray(ao_value)) - (-0.27594803231989179+0.0064644591759109114j))

    cell = gto.M(a=numpy.eye(3)*4, atom='He 1 1 1', basis=[[2,(1,.5),(.5,.5)]])
    coords = cell.get_uniform_grids([10]*3)
    ao_value = eval_gto(cell, "GTOval_ip_cart", coords, kpts=cell.make_kpts([3]*3))
    print(lib.finger(numpy.asarray(ao_value)) - (0.38051517609460028+0.062526488684770759j))

