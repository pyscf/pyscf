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
import numpy as np
from pyscf import lib
from pyscf.gto import moleintor
from pyscf.gto.eval_gto import _get_intor_and_comp, BLKSIZE
from pyscf.pbc.gto import _pbcintor
from pyscf.gto.mole import extract_pgto_params, ANG_OF
from pyscf import __config__

EXTRA_PREC = getattr(__config__, 'pbc_gto_eval_gto_extra_precision', 1e-2)

libpbc = _pbcintor.libpbc

def eval_gto(cell, eval_name, coords, comp=None, kpts=None, kpt=None,
             shls_slice=None, non0tab=None, ao_loc=None, cutoff=None,
             out=None, Ls=None, rcut=None):
    r'''Evaluate PBC-AO function value on the given grids,

    Args:
        eval_name : str::

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
        cutoff : float
            AO values smaller than cutoff will be set to zero. The default
            cutoff threshold is ~1e-22 (defined in gto/grid_ao_drv.h)
        out : ndarray
            If provided, results are written into this array.

    Returns:
        A list of 2D (or 3D) arrays to hold the AO values on grids.  Each
        element of the list corresponds to a k-point and it has the shape
        (N,nao) Or shape (\*,N,nao).

    Examples:

    >>> cell = pbc.gto.M(a=numpy.eye(3)*4, atom='He 1 1 1', basis='6-31g')
    >>> coords = cell.get_uniform_grids([10,10,10])
    >>> kpts = cell.make_kpts([3,3,3])
    >>> ao_value = cell.pbc_eval_gto("GTOval_sph", coords, kpts)
    >>> len(ao_value)
    27
    >>> ao_value[0].shape
    (1000, 2)
    >>> ao_value = cell.pbc_eval_gto("GTOval_ig_sph", coords, kpts, comp=3)
    >>> print(ao_value.shape)
    >>> len(ao_value)
    27
    >>> ao_value[0].shape
    (3, 1000, 2)
    '''
    if eval_name[:3] == 'PBC':  # PBCGTOval_xxx
        eval_name, comp = _get_intor_and_comp(cell, eval_name[3:], comp)
    else:
        eval_name, comp = _get_intor_and_comp(cell, eval_name, comp)
    eval_name = 'PBC' + eval_name

    atm = np.asarray(cell._atm, dtype=np.int32, order='C')
    bas = np.asarray(cell._bas, dtype=np.int32, order='C')
    env = np.asarray(cell._env, dtype=np.double, order='C')
    natm = atm.shape[0]
    nbas = bas.shape[0]
    if kpts is None:
        if kpt is not None:
            kpts_lst = np.reshape(kpt, (1,3))
        else:
            kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)
    ngrids = len(coords)

    if non0tab is None:
        non0tab = np.empty(((ngrids+BLKSIZE-1)//BLKSIZE, nbas),
                              dtype=np.uint8)
# non0tab stores the number of images to be summed in real space.
# Initializing it to 255 means all images should be included
        non0tab[:] = 0xff

    if ao_loc is None:
        ao_loc = moleintor.make_loc(bas, eval_name)
    if shls_slice is None:
        shls_slice = (0, nbas)
    sh0, sh1 = shls_slice
    nao = ao_loc[sh1] - ao_loc[sh0]

    out = np.empty((nkpts,comp,nao,ngrids), dtype=np.complex128)
    coords = np.asarray(coords, order='F')

    if rcut is None:
        deriv = eval_name.count('ip')
        rcut = _estimate_rcut(cell, deriv)
    if Ls is None:
        Ls = get_lattice_Ls(cell, rcut=rcut.max())
        Ls = Ls[np.argsort(lib.norm(Ls, axis=1), kind='stable')]
    expLk = np.exp(1j * np.asarray(np.dot(Ls, kpts_lst.T), order='C'))

    with cell.with_integral_screen(cutoff):
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
            v = np.asarray(v.real, order='C')
        v = v.transpose(0,2,1)
        if comp == 1:
            v = v[0]
        ao_kpts.append(v)

    if kpts is None or np.shape(kpts) == (3,):  # A single k-point
        ao_kpts = ao_kpts[0]
    return ao_kpts

pbc_eval_gto = eval_gto

def _estimate_rcut(cell, deriv=0):
    '''Cutoff radius, above which each shell decays to a value less than the
    required precision'''
    es, cs = extract_pgto_params(cell, 'diffuse')
    ls = cell._bas[:,ANG_OF]

    vol = cell.vol
    weight_penalty = vol # ~ V[r] * (vol/ngrids) * ngrids
    rad = vol**(-1./3) * cell.rcut + 1
    surface = 4*np.pi * rad**2
    lattice_sum_factor = surface
    precision = cell.precision / max(weight_penalty*lattice_sum_factor, 1)

    norm_ang = ((2*ls+1)/(4*np.pi))**.5
    fac = 2*np.pi/vol * cs*norm_ang/es / precision

    r = cell.rcut
    r = (np.log(fac * r**(ls+1)*(2*es*r)**deriv + 1.) / es)**.5
    r = (np.log(fac * r**(ls+1)*(2*es*r)**deriv + 1.) / es)**.5
    return r

def get_lattice_Ls(cell, nimgs=None, rcut=None, dimension=None, discard=True):
    '''Get lattice-sum vectors for eval_gto
    '''
    if dimension is None:
        # For atoms near the boundary of the cell, it is necessary (even in low-
        # dimensional systems) to include lattice translations in all 3 dimensions.
        if cell.dimension < 2 or cell.low_dim_ft_type == 'inf_vacuum':
            dimension = cell.dimension
        else:
            dimension = 3
    if rcut is None:
        rcut = cell.rcut

    if dimension == 0 or rcut <= 0:
        return np.zeros((1, 3))

    a = cell.lattice_vectors()
    atom_coords = cell.atom_coords()
    scaled_atom_coords = np.linalg.solve(a.T, atom_coords.T).T
    atom_boundary_max = scaled_atom_coords[:,:dimension].max(axis=0)
    atom_boundary_min = scaled_atom_coords[:,:dimension].min(axis=0)
    atom_boundary_max[atom_boundary_max > 1] = 1
    atom_boundary_min[atom_boundary_min <-1] = -1
    atom_bound1 = np.diag(atom_boundary_max).dot(a[:dimension])
    atom_bound2 = np.diag(atom_boundary_min).dot(a[:dimension])

    def find_boundary(a):
        aR = np.vstack([a, atom_bound1, atom_bound2])
        r = np.linalg.qr(aR.T)[1]
        ub = (rcut + abs(r[2,3:]).max()) / abs(r[2,2])
        return ub

    xb = find_boundary(a[[1,2,0]])
    if dimension > 1:
        yb = find_boundary(a[[2,0,1]])
    else:
        yb = 0
    if dimension > 2:
        zb = find_boundary(a)
    else:
        zb = 0
    bounds = np.ceil([xb, yb, zb]).astype(int)
    Ts = lib.cartesian_prod((np.arange(-bounds[0], bounds[0]+1),
                             np.arange(-bounds[1], bounds[1]+1),
                             np.arange(-bounds[2], bounds[2]+1)))
    Ls = np.dot(Ts[:,:dimension], a[:dimension])

    # grids with wrap_around: grids_edge ~ [-.5, .5]
    # regular grids: grids_edge ~ [0, 1]
    grids_edge = lib.cartesian_prod([[-.5, 1.]] * dimension).dot(a[:dimension])
    edge_lb = grids_edge.min(axis=0)
    edge_ub = grids_edge.max(axis=0)

    grids2atm = Ls + atom_coords[:,None,:]
    edge_filter1 = grids2atm > edge_lb
    edge_filter2 = grids2atm < edge_ub
    grids2atm[~edge_filter1[:,:,0],0] -= edge_lb[0]
    grids2atm[~edge_filter1[:,:,1],1] -= edge_lb[1]
    grids2atm[~edge_filter1[:,:,2],2] -= edge_lb[2]
    grids2atm[~edge_filter2[:,:,0],0] -= edge_ub[0]
    grids2atm[~edge_filter2[:,:,1],1] -= edge_ub[1]
    grids2atm[~edge_filter2[:,:,2],2] -= edge_ub[2]
    grids2atm[edge_filter1 & edge_filter2] = 0.
    Ls_mask = (np.linalg.norm(grids2atm[:,:,:dimension], axis=2) < rcut).any(axis=0)
    Ls = Ls[Ls_mask]
    return np.asarray(Ls, order='C')
