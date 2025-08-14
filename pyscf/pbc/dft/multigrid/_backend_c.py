#!/usr/bin/env python
# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

import ctypes
import numpy as np
from pyscf import __config__
from pyscf import lib
from pyscf.lib import logger

libdft = lib.load_library('libdft')

def gradient_gs(f_gs, Gv):
    r'''Compute the G-space components of :math:`\nabla f(r)`
    given :math:`f(G)` and :math:`G`.

    This is equivalent to einsum('np,px->nxp', f_gs, 1j*Gv)
    with multithreading.
    '''
    ng, dim = Gv.shape
    assert dim == 3
    Gv = np.asarray(Gv, order='C', dtype=np.double)
    f_gs = np.asarray(f_gs.reshape(-1,ng), order='C', dtype=np.complex128)
    n = f_gs.shape[0]
    out = np.empty((n,dim,ng), dtype=np.complex128)
    libdft.gradient_gs(
        out.ctypes.data_as(ctypes.c_void_p),
        f_gs.ctypes.data_as(ctypes.c_void_p),
        Gv.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(n), ctypes.c_size_t(ng)
    )
    return out

def get_gga_vrho_gs(v, v1, Gv, weight, ngrid, fac=2.):
    '''Update v inplace
    v -= fac * 1j * np.einsum('px,xp->p', Gv, v1)
    v *= weight
    '''
    v = np.asarray(v, order='C', dtype=np.complex128)
    v1 = np.asarray(v1, order='C', dtype=np.complex128)
    Gv = np.asarray(Gv, order='C', dtype=np.double)

    libdft.get_gga_vrho_gs(
        v.ctypes.data_as(ctypes.c_void_p),
        v1.ctypes.data_as(ctypes.c_void_p),
        Gv.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(fac),
        ctypes.c_double(weight),
        ctypes.c_int(ngrid)
    )
    return v

def pp_loc_part1_gs(coulG, Gv, G2, G0idx, Z, coords, rloc):
    libpbc = lib.load_library('libpbc')
    coulG = np.asarray(coulG, order='C', dtype=np.double)
    Gv = np.asarray(Gv, order='C', dtype=np.double)
    G2 = np.asarray(G2, order='C', dtype=np.double)
    ngrid = len(G2)

    coords = np.asarray(coords, order='C', dtype=np.double)
    Z = np.asarray(Z, order='C', dtype=np.double)
    rloc = np.asarray(rloc, order='C', dtype=np.double)
    natm = len(Z)

    out = np.empty(ngrid, dtype=np.complex128)
    libpbc.pp_loc_part1_gs(
        out.ctypes.data_as(ctypes.c_void_p),
        coulG.ctypes.data_as(ctypes.c_void_p),
        Gv.ctypes.data_as(ctypes.c_void_p),
        G2.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(G0idx),
        ctypes.c_int(ngrid),
        Z.ctypes.data_as(ctypes.c_void_p),
        coords.ctypes.data_as(ctypes.c_void_p),
        rloc.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(natm))
    return out

def build_core_density(
    atm,
    bas,
    env,
    mesh,
    dimension,
    a,
    b,
    max_radius,
):
    if abs(a - np.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
        orth = True
    else:
        lattice_type = '_nonorth'
        orth = False
    fn_name = 'make_rho_lda' + lattice_type

    atm = np.asarray(atm, order='C', dtype=np.int32)
    bas = np.asarray(bas, order='C', dtype=np.int32)
    env = np.asarray(env, order='C', dtype=np.double)
    a = np.asarray(a, order='C', dtype=np.double)
    b = np.asarray(b, order='C', dtype=np.double)
    mesh = np.asarray(mesh, order='C', dtype=np.int32)
    rho_core = np.zeros((np.prod(mesh),))

    libdft.build_core_density(
        getattr(libdft, fn_name),
        rho_core.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p),
        bas.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(len(bas)),
        env.ctypes.data_as(ctypes.c_void_p),
        mesh.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(dimension),
        a.ctypes.data_as(ctypes.c_void_p),
        b.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(max_radius),
        ctypes.c_bool(orth)
    )
    return rho_core


def int_gauss_charge_v_rs(
    v_rs,
    comp,
    atm,
    bas,
    env,
    mesh,
    dimension,
    a,
    b,
    max_radius,
):
    if abs(a - np.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
        orth = True
    else:
        lattice_type = '_nonorth'
        orth = False

    fn_name = 'eval_mat_lda' + lattice_type
    if comp == 3:
        fn_name += '_ip1'
    elif comp != 1:
        raise NotImplementedError

    out = np.zeros((len(atm), comp), order='C', dtype=np.double)
    v_rs = np.asarray(v_rs, order='C', dtype=np.double)
    atm = np.asarray(atm, order='C', dtype=np.int32)
    bas = np.asarray(bas, order='C', dtype=np.int32)
    env = np.asarray(env, order='C', dtype=np.double)
    mesh = np.asarray(mesh, order='C', dtype=np.int32)
    a = np.asarray(a, order='C', dtype=np.double)
    b = np.asarray(b, order='C', dtype=np.double)

    libdft.int_gauss_charge_v_rs(
        getattr(libdft, fn_name),
        out.ctypes.data_as(ctypes.c_void_p),
        v_rs.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(comp),
        atm.ctypes.data_as(ctypes.c_void_p),
        bas.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(len(bas)),
        env.ctypes.data_as(ctypes.c_void_p),
        mesh.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(dimension),
        a.ctypes.data_as(ctypes.c_void_p),
        b.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(max_radius),
        ctypes.c_bool(orth)
    )
    return out


def grid_collocate(
    xctype,
    dm,
    task_list,
    hermi,
    shls_slice,
    ao_loc0,
    ao_loc1,
    dimension,
    a,
    b,
    ish_atm,
    ish_bas,
    ish_env,
    jsh_atm,
    jsh_bas,
    jsh_env,
    cart,
):
    if abs(a-np.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
        orth = True
    else:
        lattice_type = '_nonorth'
        orth = False

    if xctype.upper() == 'LDA':
        comp = 1
    else:
        raise NotImplementedError

    fn_name = 'make_rho_' + xctype.lower() + lattice_type

    rs_rho = RS_Grid(task_list.gridlevel_info, comp)
    dm = np.asarray(dm, order='C', dtype=np.double)
    i0, i1, j0, j1 = shls_slice
    ao_loc0 = np.asarray(ao_loc0, order='C', dtype=np.int32)
    ao_loc1 = np.asarray(ao_loc1, order='C', dtype=np.int32)
    Ls = np.asarray(task_list.Ls, order='C', dtype=np.double)
    a = np.asarray(a, order='C', dtype=np.double)
    b = np.asarray(b, order='C', dtype=np.double)
    ish_atm = np.asarray(ish_atm, order='C', dtype=np.int32)
    ish_bas = np.asarray(ish_bas, order='C', dtype=np.int32)
    ish_env = np.asarray(ish_env, order='C', dtype=np.double)
    jsh_atm = np.asarray(jsh_atm, order='C', dtype=np.int32)
    jsh_bas = np.asarray(jsh_bas, order='C', dtype=np.int32)
    jsh_env = np.asarray(jsh_env, order='C', dtype=np.double)

    libdft.grid_collocate_drv(
        getattr(libdft, fn_name),
        ctypes.byref(rs_rho._this),
        dm.ctypes.data_as(ctypes.c_void_p),
        ctypes.byref(task_list._this),
        ctypes.c_int(comp),
        ctypes.c_int(hermi),
        (ctypes.c_int*4)(i0, i1, j0, j1),
        ao_loc0.ctypes.data_as(ctypes.c_void_p),
        ao_loc1.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(dimension),
        Ls.ctypes.data_as(ctypes.c_void_p),
        a.ctypes.data_as(ctypes.c_void_p),
        b.ctypes.data_as(ctypes.c_void_p),
        ish_atm.ctypes.data_as(ctypes.c_void_p),
        ish_bas.ctypes.data_as(ctypes.c_void_p),
        ish_env.ctypes.data_as(ctypes.c_void_p),
        jsh_atm.ctypes.data_as(ctypes.c_void_p),
        jsh_bas.ctypes.data_as(ctypes.c_void_p),
        jsh_env.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(cart),
        ctypes.c_bool(orth)
    )
    return rs_rho

def grid_integrate(
    xctype,
    wv,
    task_list,
    comp,
    hermi,
    grid_level,
    shls_slice,
    ao_loc0,
    ao_loc1,
    dimension,
    a,
    b,
    ish_atm,
    ish_bas,
    ish_env,
    jsh_atm,
    jsh_bas,
    jsh_env,
    cart,
):
    if abs(a-np.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
        orth = True
    else:
        lattice_type = '_nonorth'
        orth = False

    fn_name = 'eval_mat_' + xctype.lower() + lattice_type
    if comp == 3:
        fn_name += '_ip1'
    elif comp != 1:
        raise NotImplementedError

    i0, i1, j0, j1 = shls_slice
    ao_loc0 = np.asarray(ao_loc0, order='C', dtype=np.int32)
    ao_loc1 = np.asarray(ao_loc1, order='C', dtype=np.int32)
    naoi = ao_loc0[i1] - ao_loc0[i0]
    naoj = ao_loc1[j1] - ao_loc1[j0]
    if comp == 1:
        mat = np.zeros((naoi, naoj))
    else:
        mat = np.zeros((comp, naoi, naoj))

    wv = np.asarray(wv, order='C', dtype=np.double)
    Ls = np.asarray(task_list.Ls, order='C', dtype=np.double)
    a = np.asarray(a, order='C', dtype=np.double)
    b = np.asarray(b, order='C', dtype=np.double)
    ish_atm = np.asarray(ish_atm, order='C', dtype=np.int32)
    ish_bas = np.asarray(ish_bas, order='C', dtype=np.int32)
    ish_env = np.asarray(ish_env, order='C', dtype=np.double)
    jsh_atm = np.asarray(jsh_atm, order='C', dtype=np.int32)
    jsh_bas = np.asarray(jsh_bas, order='C', dtype=np.int32)
    jsh_env = np.asarray(jsh_env, order='C', dtype=np.double)

    libdft.grid_integrate_drv(
        getattr(libdft, fn_name),
        mat.ctypes.data_as(ctypes.c_void_p),
        wv.ctypes.data_as(ctypes.c_void_p),
        ctypes.byref(task_list._this),
        ctypes.c_int(comp),
        ctypes.c_int(hermi),
        ctypes.c_int(grid_level),
        (ctypes.c_int*4)(i0, i1, j0, j1),
        ao_loc0.ctypes.data_as(ctypes.c_void_p),
        ao_loc1.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(dimension),
        Ls.ctypes.data_as(ctypes.c_void_p),
        a.ctypes.data_as(ctypes.c_void_p),
        b.ctypes.data_as(ctypes.c_void_p),
        ish_atm.ctypes.data_as(ctypes.c_void_p),
        ish_bas.ctypes.data_as(ctypes.c_void_p),
        ish_env.ctypes.data_as(ctypes.c_void_p),
        jsh_atm.ctypes.data_as(ctypes.c_void_p),
        jsh_bas.ctypes.data_as(ctypes.c_void_p),
        jsh_env.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(cart),
        ctypes.c_bool(orth)
    )
    return mat


class _CGridLevel_Info(ctypes.Structure):
    '''C structure for `GridLevel_Info`.
    '''
    _fields_ = [("nlevels", ctypes.c_int), # number of grid levels
                ("rel_cutoff", ctypes.c_double),
                ("cutoff", ctypes.POINTER(ctypes.c_double)),
                ("mesh", ctypes.POINTER(ctypes.c_int))]


class GridLevel_Info:
    '''Information of multigrids.
    '''
    def __init__(self, cutoff, rel_cutoff, mesh):
        self._this = ctypes.POINTER(_CGridLevel_Info)()

        if cutoff[0] < 1e-15:
            cutoff = cutoff[1:]

        self.cutoff = np.asarray(cutoff, order='C', dtype=np.double)
        self.mesh = np.asarray(np.asarray(mesh).reshape(-1,3), order='C', dtype=np.int32)
        self.rel_cutoff = rel_cutoff
        self.nlevels = len(cutoff)

        libdft.init_gridlevel_info(
            ctypes.byref(self._this),
            self.cutoff.ctypes.data_as(ctypes.c_void_p),
            self.mesh.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self.nlevels),
            ctypes.c_double(self.rel_cutoff)
        )

    def __del__(self):
        try:
            libdft.del_gridlevel_info(ctypes.byref(self._this))
        except AttributeError:
            pass


class _CRS_Grid(ctypes.Structure):
    '''C structure for `RS_Grid`.
    '''
    _fields_ = [("nlevels", ctypes.c_int),
                ("gridlevel_info", ctypes.POINTER(_CGridLevel_Info)),
                ("comp", ctypes.c_int),
                # data is list of 1d arrays
                ("data", ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))]


class RS_Grid:
    '''Density values on real-space multigrids.
    '''
    def __init__(self, gridlevel_info, comp):
        self._this = ctypes.POINTER(_CRS_Grid)()
        self.gridlevel_info = gridlevel_info
        self.nlevels = gridlevel_info.nlevels

        libdft.init_rs_grid(
            ctypes.byref(self._this),
            ctypes.byref(gridlevel_info._this),
            ctypes.c_int(comp)
        )

    def __del__(self):
        try:
            libdft.del_rs_grid(ctypes.byref(self._this))
        except AttributeError:
            pass

    def __getitem__(self, i):
        '''Get the i-th level density as a numpy array.
        '''
        ngrids = np.prod(self.gridlevel_info.mesh[i])
        return np.ctypeslib.as_array(self._this.contents.data[i], shape=(ngrids,))


class _CPGFPair(ctypes.Structure):
    '''C structure for a pair of primitive Gaussian functions.
    '''
    _fields_ = [("ish", ctypes.c_int),
                ("ipgf", ctypes.c_int),
                ("jsh", ctypes.c_int),
                ("jpgf", ctypes.c_int),
                ("iL", ctypes.c_int),
                ("radius", ctypes.c_double)]


class _CTask(ctypes.Structure):
    '''C structure for a single task.
    '''
    _fields_ = [("buf_size", ctypes.c_size_t),
                ("ntasks", ctypes.c_size_t),
                ("pgfpairs", ctypes.POINTER(ctypes.POINTER(_CPGFPair))),
                ("radius", ctypes.c_double)]


class _CTaskList(ctypes.Structure):
    '''C structure for a task list.
    '''
    _fields_ = [("nlevels", ctypes.c_int),
                ("hermi", ctypes.c_int),
                ("gridlevel_info", ctypes.POINTER(_CGridLevel_Info)),
                ("tasks", ctypes.POINTER(ctypes.POINTER(_CTask)))]


class TaskList:
    '''Task list for multigrid DFT calculations.
    '''
    grid_level_method = getattr(__config__, "pbc_dft_multigrid_grid_level_method", "pyscf")

    def __init__(self, cell, gridlevel_info,
                 cell1=None, Ls=None, hermi=0, precision=None):
        '''Build a task list.

        Arguments:
            cell : :class:`pbc.gto.cell.Cell`
                The :class:`Cell` instance for the bra basis functions.
            gridlevel_info : :class:`GridLevel_Info`
                Information of the multiple grids.
            cell1 : :class:`pbc.gto.cell.Cell`, optional
                The :class:`Cell` instance for the ket basis functions.
                If not given, both bra and ket basis functions come from `cell`.
            Ls : (*,3) array, optional
                The cartesian coordinates of the periodic images.
                Default is calculated by :func:`cell.get_lattice_Ls`.
            hermi : int, optional
                If :math:`hermi=1`, the task list is built only for
                the upper triangle of the matrix. Default is 0.
            precision : float, optional
                The integral precision. Default is :attr:`cell.precision`.
        '''
        from pyscf.pbc.gto import (
            build_neighbor_list_for_shlpairs,
            free_neighbor_list
        )

        self._this = ctypes.POINTER(_CTaskList)()
        self.gridlevel_info = gridlevel_info
        self.nlevels = gridlevel_info.nlevels

        if cell1 is None:
            cell1 = cell

        if precision is None:
            precision = cell.precision

        if hermi == 1 and cell1 is not cell:
            logger.warn(cell,
                        "Set hermi=0 because cell and cell1 are not the same.")
            hermi = 0
        self.hermi = hermi

        ish_atm = np.asarray(cell._atm, order='C', dtype=np.int32)
        ish_bas = np.asarray(cell._bas, order='C', dtype=np.int32)
        ish_env = np.asarray(cell._env, order='C', dtype=np.double)
        nish = len(ish_bas)
        ish_rcut, ipgf_rcut = cell.rcut_by_shells(precision=precision,
                                                  return_pgf_radius=True)
        assert nish == len(ish_rcut)
        ptr_ipgf_rcut = lib.ndarray_pointer_2d(ipgf_rcut)

        if cell1 is cell:
            jsh_atm = ish_atm
            jsh_bas = ish_bas
            jsh_env = ish_env
            jsh_rcut = ish_rcut
            jpgf_rcut = ipgf_rcut
            ptr_jpgf_rcut = ptr_ipgf_rcut
        else:
            jsh_atm = np.asarray(cell1._atm, order='C', dtype=np.int32)
            jsh_bas = np.asarray(cell1._bas, order='C', dtype=np.int32)
            jsh_env = np.asarray(cell1._env, order='C', dtype=np.double)
            jsh_rcut, jpgf_rcut = cell1.rcut_by_shells(precision=precision,
                                                       return_pgf_radius=True)
            ptr_jpgf_rcut = lib.ndarray_pointer_2d(jpgf_rcut)
        njsh = len(jsh_bas)
        assert njsh == len(jsh_rcut)

        if Ls is None:
            # The default cell.rcut might be insufficient to include all
            # significant pairs
            rcut = ish_rcut.max(initial=0) + jsh_rcut.max(initial=0)
            Ls = cell.get_lattice_Ls(rcut=rcut)
        # The lattice sum vectors must be consistent with the neighbor_list in
        # the backend functions
        self.Ls = Ls
        Ls = np.asarray(Ls, order='C', dtype=np.double)

        nl = build_neighbor_list_for_shlpairs(cell, cell1, Ls=Ls,
                                              ish_rcut=ish_rcut, jsh_rcut=jsh_rcut,
                                              hermi=hermi)

        if self.grid_level_method.lower() == "cp2k":
            fn_name = "get_grid_level_cp2k"
        else:
            fn_name = "get_grid_level"

        libdft.build_task_list(
            ctypes.byref(self._this),
            ctypes.byref(nl),
            ctypes.byref(gridlevel_info._this),
            getattr(libdft, fn_name),
            ish_atm.ctypes.data_as(ctypes.c_void_p),
            ish_bas.ctypes.data_as(ctypes.c_void_p),
            ish_env.ctypes.data_as(ctypes.c_void_p),
            ish_rcut.ctypes.data_as(ctypes.c_void_p),
            ptr_ipgf_rcut.ctypes,
            jsh_atm.ctypes.data_as(ctypes.c_void_p),
            jsh_bas.ctypes.data_as(ctypes.c_void_p),
            jsh_env.ctypes.data_as(ctypes.c_void_p),
            jsh_rcut.ctypes.data_as(ctypes.c_void_p),
            ptr_jpgf_rcut.ctypes,
            ctypes.c_int(nish), ctypes.c_int(njsh),
            Ls.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(precision),
            ctypes.c_int(hermi)
        )

    @property
    def ntasks(self):
        return [self._this.contents.tasks[i].contents.ntasks for i in range(self.nlevels)]

    def __del__(self):
        try:
            libdft.del_task_list(ctypes.byref(self._this))
        except AttributeError:
            pass

