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
from pyscf.gto import moleintor
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.df import fft
from pyscf.pbc.df.df_jk import (
    _format_dms,
    _format_kpts_band,
    _format_jks,
)
from pyscf.pbc.dft.multigrid.pp import (
    _get_vpplocG_part1,
    _get_pp_without_erf,
    vpploc_part1_nuc_grad,
)
from pyscf.pbc.dft.multigrid.utils import (
    _take_4d,
    _take_5d,
    _takebak_4d,
    _takebak_5d,
)
from pyscf.pbc.dft.multigrid.multigrid import MultiGridFFTDF

NGRIDS = getattr(__config__, 'pbc_dft_multigrid_ngrids', 4)
KE_RATIO = getattr(__config__, 'pbc_dft_multigrid_ke_ratio', 3.0)
REL_CUTOFF = getattr(__config__, 'pbc_dft_multigrid_rel_cutoff', 20.0)
GGA_METHOD = getattr(__config__, 'pbc_dft_multigrid_gga_method', 'FFT')

EXTRA_PREC = getattr(__config__, 'pbc_gto_eval_gto_extra_precision', 1e-2)
RHOG_HIGH_ORDER = getattr(__config__, 'pbc_dft_multigrid_rhog_high_order', False)
PTR_EXPDROP = 16
EXPDROP = getattr(__config__, 'pbc_dft_multigrid_expdrop', 1e-12)
IMAG_TOL = 1e-9

libdft = lib.load_library('libdft')

def gradient_gs(f_gs, Gv):
    r'''Compute the G-space components of :math:`\nabla f(r)`
    given :math:`f(G)` and :math:`G`,
    which is equivalent to einsum('np,px->nxp', f_gs, 1j*Gv)
    '''
    ng, dim = Gv.shape
    assert dim == 3
    Gv = np.asarray(Gv, order='C', dtype=np.double)
    f_gs = np.asarray(f_gs.reshape(-1,ng), order='C', dtype=np.complex128)
    n = f_gs.shape[0]
    out = np.empty((n,dim,ng), dtype=np.complex128)

    fn = getattr(libdft, 'gradient_gs', None)
    try:
        fn(out.ctypes.data_as(ctypes.c_void_p),
           f_gs.ctypes.data_as(ctypes.c_void_p),
           Gv.ctypes.data_as(ctypes.c_void_p),
           ctypes.c_int(n), ctypes.c_size_t(ng))
    except Exception as e:
        raise RuntimeError(f'Error in gradient_gs: {e}')
    return out


class GridLevel_Info(ctypes.Structure):
    '''
    Info about the grid levels.
    '''
    __slots__ = []
    _fields_ = [("nlevels", ctypes.c_int), # number of grid levels
                ("rel_cutoff", ctypes.c_double),
                ("cutoff", ctypes.POINTER(ctypes.c_double)),
                ("mesh", ctypes.POINTER(ctypes.c_int))]

class RS_Grid(ctypes.Structure):
    '''
    Values on real space multigrid.
    '''
    __slots__ = []
    _fields_ = [("nlevels", ctypes.c_int),
                ("gridlevel_info", ctypes.POINTER(GridLevel_Info)),
                ("comp", ctypes.c_int),
                # data is list of 1d arrays
                ("data", ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))]

class PGFPair(ctypes.Structure):
    '''
    A primitive Gaussian function pair.
    '''
    __slots__ = []
    _fields_ = [("ish", ctypes.c_int),
                ("ipgf", ctypes.c_int),
                ("jsh", ctypes.c_int),
                ("jpgf", ctypes.c_int),
                ("iL", ctypes.c_int),
                ("radius", ctypes.c_double)]


class Task(ctypes.Structure):
    '''
    A single task.
    '''
    __slots__ = []
    _fields_ = [("buf_size", ctypes.c_size_t),
                ("ntasks", ctypes.c_size_t),
                ("pgfpairs", ctypes.POINTER(ctypes.POINTER(PGFPair))),
                ("radius", ctypes.c_double)]


class TaskList(ctypes.Structure):
    '''
    A task list.
    '''
    __slots__ = []
    _fields_ = [("nlevels", ctypes.c_int),
                ("hermi", ctypes.c_int),
                ("gridlevel_info", ctypes.POINTER(GridLevel_Info)),
                ("tasks", ctypes.POINTER(ctypes.POINTER(Task)))]


def multi_grids_tasks(cell, ke_cutoff=None, hermi=0,
                      ngrids=NGRIDS, ke_ratio=KE_RATIO, rel_cutoff=REL_CUTOFF):
    if ke_cutoff is None:
        ke_cutoff = cell.ke_cutoff
    if ke_cutoff is None:
        raise ValueError("cell.ke_cutoff is not set.")
    ke1 = ke_cutoff
    cutoff = [ke1,]
    for i in range(ngrids-1):
        ke1 /= ke_ratio
        cutoff.append(ke1)
    cutoff.reverse()
    a = cell.lattice_vectors()
    mesh = []
    for ke in cutoff:
        mesh.append(tools.cutoff_to_mesh(a, ke))
    logger.info(cell, 'ke_cutoff for multigrid tasks:\n%s', cutoff)
    logger.info(cell, 'meshes for multigrid tasks:\n%s', mesh)
    gridlevel_info = init_gridlevel_info(cutoff, rel_cutoff, mesh)
    task_list = build_task_list(cell, gridlevel_info, hermi=hermi)
    return task_list


def _update_task_list(mydf, hermi=0, ngrids=None, ke_ratio=None, rel_cutoff=None):
    '''
    Update :attr:`task_list` if necessary.
    '''
    cell = mydf.cell
    if ngrids is None:
        ngrids = mydf.ngrids
    if ke_ratio is None:
        ke_ratio = mydf.ke_ratio
    if rel_cutoff is None:
        rel_cutoff = mydf.rel_cutoff

    need_update = False
    task_list = getattr(mydf, 'task_list', None)
    if task_list is None:
        need_update = True
    else:
        hermi_orig = task_list.contents.hermi
        nlevels = task_list.contents.nlevels
        rel_cutoff_orig = task_list.contents.gridlevel_info.contents.rel_cutoff
        #TODO also need to check kinetic energy cutoff change
        if (hermi_orig > hermi or
                nlevels != ngrids or
                abs(rel_cutoff_orig-rel_cutoff) > 1e-12):
            need_update = True

    if need_update:
        if task_list is not None:
            free_task_list(task_list)
        task_list = multi_grids_tasks(cell, hermi=hermi, ngrids=ngrids,
                                      ke_ratio=ke_ratio, rel_cutoff=rel_cutoff)
        mydf.task_list = task_list
    return task_list


def init_gridlevel_info(cutoff, rel_cutoff, mesh):
    if cutoff[0] < 1e-15:
        cutoff = cutoff[1:]
    cutoff = np.asarray(cutoff, order='C', dtype=np.double)
    mesh = np.asarray(np.asarray(mesh).reshape(-1,3), order='C', dtype=np.int32)
    nlevels = len(cutoff)
    gridlevel_info = ctypes.POINTER(GridLevel_Info)()
    fn = getattr(libdft, "init_gridlevel_info", None)
    try:
        fn(ctypes.byref(gridlevel_info),
           cutoff.ctypes.data_as(ctypes.c_void_p),
           mesh.ctypes.data_as(ctypes.c_void_p),
           ctypes.c_int(nlevels), ctypes.c_double(rel_cutoff))
    except Exception as e:
        raise RuntimeError("Failed to init grid level info. %s" % e)
    return gridlevel_info


def free_gridlevel_info(gridlevel_info):
    fn = getattr(libdft, "del_gridlevel_info", None)
    try:
        fn(ctypes.byref(gridlevel_info))
    except Exception as e:
        raise RuntimeError("Failed to free grid level info. %s" % e)


def init_rs_grid(gridlevel_info, comp):
    '''
    Initialize values on real space multigrid
    '''
    rs_grid = ctypes.POINTER(RS_Grid)()
    fn = getattr(libdft, "init_rs_grid", None)
    try:
        fn(ctypes.byref(rs_grid),
           ctypes.byref(gridlevel_info),
           ctypes.c_int(comp))
    except Exception as e:
        raise RuntimeError("Failed to initialize real space multigrid data. %s" % e)
    return rs_grid


def free_rs_grid(rs_grid):
    fn = getattr(libdft, "del_rs_grid", None)
    try:
        fn(ctypes.byref(rs_grid))
    except Exception as e:
        raise RuntimeError("Failed to free real space multigrid data. %s" % e)


def build_task_list(cell, gridlevel_info, cell1=None, Ls=None, hermi=0, precision=None):
    '''
    Build the task list for multigrid DFT calculations.

    Arguments:
        cell : :class:`pbc.gto.cell.Cell`
            The :class:`Cell` instance for the bra basis functions.
        gridlevel_info : :class:`ctypes.POINTER`
            The C pointer of the :class:`GridLevel_Info` structure.
        cell1 : :class:`pbc.gto.cell.Cell`, optional
            The :class:`Cell` instance for the ket basis functions.
            If not given, both bra and ket basis functions come from cell.
        Ls : ``(*,3)`` array, optional
            The cartesian coordinates of the periodic images.
            Default is calculated by :func:`cell.get_lattice_Ls`.
        hermi : int, optional
            If :math:`hermi=1`, the task list is built only for
            the upper triangle of the matrix. Default is 0.
        precision : float, optional
            The integral precision. Default is :attr:`cell.precision`.

    Returns: :class:`ctypes.POINTER`
        The C pointer of the :class:`TaskList` structure.
    '''
    from pyscf.pbc.gto import build_neighbor_list_for_shlpairs, free_neighbor_list
    if cell1 is None:
        cell1 = cell
    if Ls is None:
        Ls = cell.get_lattice_Ls()
    if precision is None:
        precision = cell.precision

    if hermi == 1 and cell1 is not cell:
        logger.warn(cell,
                    "Set hermi=0 because cell and cell1 are not the same.")
        hermi = 0

    ish_atm = np.asarray(cell._atm, order='C', dtype=np.int32)
    ish_bas = np.asarray(cell._bas, order='C', dtype=np.int32)
    ish_env = np.asarray(cell._env, order='C', dtype=float)
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
        jsh_env = np.asarray(cell1._env, order='C', dtype=float)
        jsh_rcut, jpgf_rcut = cell1.rcut_by_shells(precision=precision,
                                                   return_pgf_radius=True)
        ptr_jpgf_rcut = lib.ndarray_pointer_2d(jpgf_rcut)
    njsh = len(jsh_bas)
    assert njsh == len(jsh_rcut)

    nl = build_neighbor_list_for_shlpairs(cell, cell1, Ls=Ls,
                                          ish_rcut=ish_rcut, jsh_rcut=jsh_rcut,
                                          hermi=hermi)

    task_list = ctypes.POINTER(TaskList)()
    func = getattr(libdft, "build_task_list", None)
    try:
        func(ctypes.byref(task_list),
             ctypes.byref(nl), ctypes.byref(gridlevel_info),
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
             ctypes.c_double(precision), ctypes.c_int(hermi))
    except Exception as e:
        raise RuntimeError("Failed to build task list. %s" % e)
    free_neighbor_list(nl)
    return task_list


def free_task_list(task_list):
    '''
    Note:
        This will also free task_list.contents.gridlevel_info.
    '''
    if task_list is None:
        return
    func = getattr(libdft, "del_task_list", None)
    try:
        func(ctypes.byref(task_list))
    except Exception as e:
        raise RuntimeError("Failed to free task list. %s" % e)


def eval_rho(cell, dm, task_list, shls_slice=None, hermi=0, xctype='LDA', kpts=None,
             dimension=None, cell1=None, shls_slice1=None, Ls=None,
             a=None, ignore_imag=False):
    '''
    Collocate density (opt. gradients) on the real-space grid.
    The two sets of Gaussian functions can be different.

    Returns:
        rho: RS_Grid object
            Densities on real space multigrids.
    '''
    cell0 = cell
    shls_slice0 = shls_slice
    if cell1 is None:
        cell1 = cell0

    #TODO mixture of cartesian and spherical bases
    assert cell0.cart == cell1.cart

    ish_atm = np.asarray(cell0._atm, order='C', dtype=np.int32)
    ish_bas = np.asarray(cell0._bas, order='C', dtype=np.int32)
    ish_env = np.asarray(cell0._env, order='C', dtype=np.double)
    ish_env[PTR_EXPDROP] = min(cell0.precision*EXTRA_PREC, EXPDROP)

    if cell1 is cell0:
        jsh_atm = ish_atm
        jsh_bas = ish_bas
        jsh_env = ish_env
    else:
        jsh_atm = np.asarray(cell1._atm, order='C', dtype=np.int32)
        jsh_bas = np.asarray(cell1._bas, order='C', dtype=np.int32)
        jsh_env = np.asarray(cell1._env, order='C', dtype=np.double)
        jsh_env[PTR_EXPDROP] = min(cell1.precision*EXTRA_PREC, EXPDROP)

    if shls_slice0 is None:
        shls_slice0 = (0, cell0.nbas)
    i0, i1 = shls_slice0
    if shls_slice1 is None:
        shls_slice1 = shls_slice0
    j0, j1 = shls_slice1

    if hermi == 1:
        assert cell1 is cell0
        assert i0 == j0 and i1 == j1

    key0 = 'cart' if cell0.cart else 'sph'
    ao_loc0 = moleintor.make_loc(ish_bas, key0)
    naoi = ao_loc0[i1] - ao_loc0[i0]
    if hermi == 1:
        ao_loc1 = ao_loc0
    else:
        key1 = 'cart' if cell1.cart else 'sph'
        ao_loc1 = moleintor.make_loc(jsh_bas, key1)
    naoj = ao_loc1[j1] - ao_loc1[j0]

    dm = np.asarray(dm, order='C')
    assert dm.shape[-2:] == (naoi, naoj)

    if dimension is None:
        dimension = cell0.dimension
    assert dimension == getattr(cell1, "dimension", None)

    if Ls is None and dimension > 0:
        Ls = np.asarray(cell0.get_lattice_Ls(), order='C')
    elif Ls is None and dimension == 0:
        Ls = np.zeros((1,3))

    if dimension == 0 or kpts is None or gamma_point(kpts):
        nkpts, nimgs = 1, Ls.shape[0]
        dm = dm.reshape(-1,1,naoi,naoj)
    else:
        expkL = np.exp(1j*kpts.reshape(-1,3).dot(Ls.T))
        nkpts, nimgs = expkL.shape
        dm = dm.reshape(-1,nkpts,naoi,naoj)
    n_dm = dm.shape[0]

    #TODO check if cell1 has the same lattice vectors
    if a is None:
        a = cell0.lattice_vectors()
    b = np.linalg.inv(a.T)

    if abs(a-np.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
    else:
        lattice_type = '_nonorth'
    xctype = xctype.upper()
    if xctype == 'LDA':
        comp = 1
    elif xctype == 'GGA':
        if hermi == 1:
            raise RuntimeError('hermi=1 is not supported for GGA functional')
        comp = 4
    else:
        raise NotImplementedError('meta-GGA')

    eval_fn = 'make_rho_' + xctype.lower() + lattice_type
    drv = getattr(libdft, "grid_collocate_drv", None)

    def make_rho_(rs_rho, dm):
        try:
            drv(getattr(libdft, eval_fn, None),
                ctypes.byref(rs_rho),
                dm.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(task_list),
                ctypes.c_int(comp), ctypes.c_int(hermi),
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
                ctypes.c_int(cell0.cart))
        except Exception as e:
            raise RuntimeError("Failed to compute rho. %s" % e)
        return rs_rho

    gridlevel_info = task_list.contents.gridlevel_info
    rho = []
    for i, dm_i in enumerate(dm):
        rs_rho = init_rs_grid(gridlevel_info, comp)
        if dimension == 0 or kpts is None or gamma_point(kpts):
            make_rho_(rs_rho, dm_i)
        else:
            raise NotImplementedError
        rho.append(rs_rho)

    if n_dm == 1:
        rho = rho[0]
    return rho


def _eval_rhoG(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), deriv=0,
               rhog_high_order=RHOG_HIGH_ORDER):
    assert(deriv < 2)
    cell = mydf.cell

    dm_kpts = np.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    task_list = _update_task_list(mydf, hermi=hermi, ngrids=mydf.ngrids,
                                  ke_ratio=mydf.ke_ratio, rel_cutoff=mydf.rel_cutoff)

    gga_high_order = False
    if deriv == 0:
        xctype = 'LDA'
        rhodim = 1
    elif deriv == 1:
        if rhog_high_order:
            xctype = 'GGA'
            rhodim = 4
        else:  # approximate high order derivatives in reciprocal space
            gga_high_order = True
            xctype = 'LDA'
            rhodim = 1
            deriv = 0
        assert(hermi == 1 or gamma_point(kpts))
    elif deriv == 2:  # meta-GGA
        raise NotImplementedError
        assert(hermi == 1 or gamma_point(kpts))

    ignore_imag = (hermi == 1)

    rs_rho = eval_rho(cell, dms, task_list, hermi=hermi, xctype=xctype, kpts=kpts,
                      ignore_imag=ignore_imag)

    nx, ny, nz = mydf.mesh
    rhoG = np.zeros((nset*rhodim,nx,ny,nz), dtype=np.complex128)
    nlevels = task_list.contents.nlevels
    meshes = task_list.contents.gridlevel_info.contents.mesh
    meshes = np.ctypeslib.as_array(meshes, shape=(nlevels,3))
    for ilevel in range(nlevels):
        mesh = meshes[ilevel]
        ngrids = np.prod(mesh)
        if nset > 1:
            rho = []
            for i in range(nset):
                rho.append(np.ctypeslib.as_array(rs_rho[i].contents.data[ilevel], shape=(ngrids,)))
            rho = np.asarray(rho)
        else:
            rho = np.ctypeslib.as_array(rs_rho.contents.data[ilevel], shape=(ngrids,))

        weight = 1./nkpts * cell.vol/ngrids
        rho_freq = tools.fft(rho.reshape(nset*rhodim, -1), mesh)
        rho = None
        rho_freq *= weight
        gx = np.fft.fftfreq(mesh[0], 1./mesh[0]).astype(np.int32)
        gy = np.fft.fftfreq(mesh[1], 1./mesh[1]).astype(np.int32)
        gz = np.fft.fftfreq(mesh[2], 1./mesh[2]).astype(np.int32)
        _takebak_4d(rhoG, rho_freq.reshape((-1,) + tuple(mesh)), (None, gx, gy, gz))
        rho_freq = None

    if nset > 1:
        for i in range(nset):
            free_rs_grid(rs_rho[i])
    else:
        free_rs_grid(rs_rho)
    rs_rho = None

    rhoG = rhoG.reshape(nset,rhodim,-1)
    if gga_high_order:
        Gv = cell.get_Gv(mydf.mesh)
        #:rhoG1 = np.einsum('np,px->nxp', 1j*rhoG[:,0], Gv)
        rhoG1 = gradient_gs(rhoG[:,0], Gv)
        rhoG = np.concatenate([rhoG, rhoG1], axis=1)
        Gv = rhoG1 = None
    return rhoG


def eval_mat(cell, weights, task_list, shls_slice=None, comp=1, hermi=0, deriv=0,
             xctype='LDA', kpts=None, grid_level=None, dimension=None, mesh=None,
             cell1=None, shls_slice1=None, Ls=None, a=None):

    cell0 = cell
    shls_slice0 = shls_slice
    if cell1 is None:
        cell1 = cell0

    if mesh is None:
        mesh = cell0.mesh

    #TODO mixture of cartesian and spherical bases
    assert cell0.cart == cell1.cart

    ish_atm = np.asarray(cell0._atm, order='C', dtype=np.int32)
    ish_bas = np.asarray(cell0._bas, order='C', dtype=np.int32)
    ish_env = np.asarray(cell0._env, order='C', dtype=np.double)
    ish_env[PTR_EXPDROP] = min(cell0.precision*EXTRA_PREC, EXPDROP)

    if cell1 is cell0:
        jsh_atm = ish_atm
        jsh_bas = ish_bas
        jsh_env = ish_env
    else:
        jsh_atm = np.asarray(cell1._atm, order='C', dtype=np.int32)
        jsh_bas = np.asarray(cell1._bas, order='C', dtype=np.int32)
        jsh_env = np.asarray(cell1._env, order='C', dtype=np.double)
        jsh_env[PTR_EXPDROP] = min(cell1.precision*EXTRA_PREC, EXPDROP)

    if shls_slice0 is None:
        shls_slice0 = (0, cell0.nbas)
    i0, i1 = shls_slice0
    if shls_slice1 is None:
        shls_slice1 = (0, cell1.nbas)
    j0, j1 = shls_slice1

    if hermi == 1:
        assert cell1 is cell0
        assert i0 == j0 and i1 == j1

    key0 = 'cart' if cell0.cart else 'sph'
    ao_loc0 = moleintor.make_loc(ish_bas, key0)
    naoi = ao_loc0[i1] - ao_loc0[i0]
    if hermi == 1:
        ao_loc1 = ao_loc0
    else:
        key1 = 'cart' if cell1.cart else 'sph'
        ao_loc1 = moleintor.make_loc(jsh_bas, key1)
    naoj = ao_loc1[j1] - ao_loc1[j0]

    if dimension is None:
        dimension = cell0.dimension
    assert dimension == getattr(cell1, "dimension", None)

    if Ls is None and dimension > 0:
        Ls = np.asarray(cell0.get_lattice_Ls(), order='C')
    elif Ls is None and dimension == 0:
        Ls = np.zeros((1,3))

    if dimension == 0 or kpts is None or gamma_point(kpts):
        nkpts, nimgs = 1, Ls.shape[0]
    else:
        expkL = np.exp(1j*kpts.reshape(-1,3).dot(Ls.T))
        nkpts, nimgs = expkL.shape

    #TODO check if cell1 has the same lattice vectors
    if a is None:
        a = cell0.lattice_vectors()
    b = np.linalg.inv(a.T)

    if abs(a-np.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
    else:
        lattice_type = '_nonorth'

    weights = np.asarray(weights, order='C')
    assert(weights.dtype == np.double)
    xctype = xctype.upper()
    n_mat = None
    if xctype == 'LDA':
        if weights.ndim == 1:
            weights = weights.reshape(-1, np.prod(mesh))
        else:
            n_mat = weights.shape[0]
    elif xctype == 'GGA':
        if weights.ndim == 2:
            weights = weights.reshape(-1, 4, np.prod(mesh))
        else:
            n_mat = weights.shape[0]
    else:
        raise NotImplementedError

    eval_fn = 'eval_mat_' + xctype.lower() + lattice_type
    if deriv > 0:
        if deriv == 1:
            assert comp == 3
            assert hermi == 0
            eval_fn += '_ip1'
        else:
            raise NotImplementedError
    drv = getattr(libdft, "grid_integrate_drv", None)

    def make_mat(wv):
        if comp == 1:
            mat = np.zeros((naoi, naoj))
        else:
            mat = np.zeros((comp, naoi, naoj))

        try:
            drv(getattr(libdft, eval_fn, None),
                mat.ctypes.data_as(ctypes.c_void_p),
                wv.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(task_list),
                ctypes.c_int(comp), ctypes.c_int(hermi),
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
                ctypes.c_int(cell0.cart))
        except Exception as e:
            raise RuntimeError("Failed to compute rho. %s" % e)
        return mat

    out = []
    for wv in weights:
        if dimension == 0 or kpts is None or gamma_point(kpts):
            mat = make_mat(wv)
        else:
            raise NotImplementedError
        out.append(mat)

    if n_mat is None:
        out = out[0]
    return out


def _get_j_pass2(mydf, vG, kpts=np.zeros((1,3)), hermi=1, verbose=None):
    cell = mydf.cell
    nkpts = len(kpts)
    nao = cell.nao_nr()
    nx, ny, nz = mydf.mesh
    vG = vG.reshape(-1,nx,ny,nz)
    nset = vG.shape[0]

    task_list = _update_task_list(mydf, hermi=hermi, ngrids=mydf.ngrids,
                                  ke_ratio=mydf.ke_ratio, rel_cutoff=mydf.rel_cutoff)

    at_gamma_point = gamma_point(kpts)
    if at_gamma_point:
        vj_kpts = np.zeros((nset,nkpts,nao,nao))
    else:
        vj_kpts = np.zeros((nset,nkpts,nao,nao), dtype=np.complex128)

    nlevels = task_list.contents.nlevels
    meshes = task_list.contents.gridlevel_info.contents.mesh
    meshes = np.ctypeslib.as_array(meshes, shape=(nlevels,3))
    for ilevel in range(nlevels):
        mesh = meshes[ilevel]
        ngrids = np.prod(mesh)

        gx = np.fft.fftfreq(mesh[0], 1./mesh[0]).astype(np.int32)
        gy = np.fft.fftfreq(mesh[1], 1./mesh[1]).astype(np.int32)
        gz = np.fft.fftfreq(mesh[2], 1./mesh[2]).astype(np.int32)
        sub_vG = _take_4d(vG, (None, gx, gy, gz)).reshape(nset,ngrids)

        v_rs = tools.ifft(sub_vG, mesh).reshape(nset,ngrids)
        vR = np.asarray(v_rs.real, order='C')
        vI = np.asarray(v_rs.imag, order='C')
        if at_gamma_point:
            v_rs = vR

        mat = eval_mat(cell, vR, task_list, comp=1, hermi=hermi,
                       xctype='LDA', kpts=kpts, grid_level=ilevel, mesh=mesh)
        vj_kpts += np.asarray(mat).reshape(nset,-1,nao,nao)
        if not at_gamma_point and abs(vI).max() > IMAG_TOL:
            raise NotImplementedError

    if nset == 1:
        vj_kpts = vj_kpts[0]
    return vj_kpts


def _get_j_pass2_ip1(mydf, vG, kpts=np.zeros((1,3)), hermi=0, deriv=1, verbose=None):
    if deriv == 1:
        comp = 3
        assert hermi == 0
    else:
        raise NotImplementedError

    cell = mydf.cell
    nkpts = len(kpts)
    nao = cell.nao_nr()
    nx, ny, nz = mydf.mesh
    vG = vG.reshape(-1,nx,ny,nz)
    nset = vG.shape[0]

    task_list = _update_task_list(mydf, hermi=hermi, ngrids=mydf.ngrids,
                                  ke_ratio=mydf.ke_ratio, rel_cutoff=mydf.rel_cutoff)

    at_gamma_point = gamma_point(kpts)
    if at_gamma_point:
        vj_kpts = np.zeros((nset,nkpts,comp,nao,nao))
    else:
        vj_kpts = np.zeros((nset,nkpts,comp,nao,nao), dtype=np.complex128)

    nlevels = task_list.contents.nlevels
    meshes = task_list.contents.gridlevel_info.contents.mesh
    meshes = np.ctypeslib.as_array(meshes, shape=(nlevels,3))
    for ilevel in range(nlevels):
        mesh = meshes[ilevel]
        ngrids = np.prod(mesh)

        gx = np.fft.fftfreq(mesh[0], 1./mesh[0]).astype(np.int32)
        gy = np.fft.fftfreq(mesh[1], 1./mesh[1]).astype(np.int32)
        gz = np.fft.fftfreq(mesh[2], 1./mesh[2]).astype(np.int32)
        sub_vG = _take_4d(vG, (None, gx, gy, gz)).reshape(nset,ngrids)

        v_rs = tools.ifft(sub_vG, mesh).reshape(nset,ngrids)
        if at_gamma_point:
            vR = np.asarray(v_rs.real, order='C', dtype=float)
            #vI = None
        else:
            raise NotImplementedError

        mat = eval_mat(cell, vR, task_list, comp=comp, hermi=hermi, deriv=deriv,
                       xctype='LDA', kpts=kpts, grid_level=ilevel, mesh=mesh)
        mat = np.asarray(mat).reshape(nset,-1,comp,nao,nao)
        vj_kpts = np.add(vj_kpts, mat, out=vj_kpts)

    if nset == 1:
        vj_kpts = vj_kpts[0]
    return vj_kpts


def _get_gga_pass2(mydf, vG, kpts=np.zeros((1,3)), hermi=1, verbose=None):
    cell = mydf.cell
    nkpts = len(kpts)
    nao = cell.nao_nr()
    nx, ny, nz = mydf.mesh
    vG = vG.reshape(-1,4,nx,ny,nz)
    nset = vG.shape[0]

    task_list = _update_task_list(mydf, hermi=hermi, ngrids=mydf.ngrids,
                                  ke_ratio=mydf.ke_ratio, rel_cutoff=mydf.rel_cutoff)

    if gamma_point(kpts):
        veff = np.zeros((nset,nkpts,nao,nao))
    else:
        veff = np.zeros((nset,nkpts,nao,nao), dtype=np.complex128)

    nlevels = task_list.contents.nlevels
    meshes = task_list.contents.gridlevel_info.contents.mesh
    meshes = np.ctypeslib.as_array(meshes, shape=(nlevels,3))
    for ilevel in range(nlevels):
        mesh = meshes[ilevel]
        ngrids = np.prod(mesh)

        gx = np.fft.fftfreq(mesh[0], 1./mesh[0]).astype(np.int32)
        gy = np.fft.fftfreq(mesh[1], 1./mesh[1]).astype(np.int32)
        gz = np.fft.fftfreq(mesh[2], 1./mesh[2]).astype(np.int32)
        sub_vG = _take_5d(vG, (None, None, gx, gy, gz)).reshape(-1,ngrids)
        wv = tools.ifft(sub_vG, mesh).real.reshape(nset,4,ngrids)
        wv = np.asarray(wv, order='C')

        mat = eval_mat(cell, wv, task_list, comp=1, hermi=hermi,
                       xctype='GGA', kpts=kpts, grid_level=ilevel, mesh=mesh)
        mat = np.asarray(mat).reshape(nset,-1,nao,nao)
        veff = np.add(veff, mat, out=veff)
        if not gamma_point(kpts):
            raise NotImplementedError

    if nset == 1:
        veff = veff[0]
    return veff


def _get_gga_pass2_ip1(mydf, vG, kpts=np.zeros((1,3)), hermi=0, deriv=1, verbose=None):
    if deriv == 1:
        comp = 3
        assert hermi == 0
    else:
        raise NotImplementedError

    cell = mydf.cell
    nkpts = len(kpts)
    nao = cell.nao_nr()
    nx, ny, nz = mydf.mesh
    vG = vG.reshape(-1,4,nx,ny,nz)
    nset = vG.shape[0]

    task_list = _update_task_list(mydf, hermi=hermi, ngrids=mydf.ngrids,
                                  ke_ratio=mydf.ke_ratio, rel_cutoff=mydf.rel_cutoff)

    at_gamma_point = gamma_point(kpts)
    if at_gamma_point:
        vj_kpts = np.zeros((nset,nkpts,comp,nao,nao))
    else:
        vj_kpts = np.zeros((nset,nkpts,comp,nao,nao), dtype=np.complex128)

    nlevels = task_list.contents.nlevels
    meshes = task_list.contents.gridlevel_info.contents.mesh
    meshes = np.ctypeslib.as_array(meshes, shape=(nlevels,3))
    for ilevel in range(nlevels):
        mesh = meshes[ilevel]
        ngrids = np.prod(mesh)

        gx = np.fft.fftfreq(mesh[0], 1./mesh[0]).astype(np.int32)
        gy = np.fft.fftfreq(mesh[1], 1./mesh[1]).astype(np.int32)
        gz = np.fft.fftfreq(mesh[2], 1./mesh[2]).astype(np.int32)
        sub_vG = _take_5d(vG, (None, None, gx, gy, gz)).reshape(-1,ngrids)

        v_rs = tools.ifft(sub_vG, mesh).reshape(nset,4,ngrids)
        vR = np.asarray(v_rs.real, order='C')
        vI = np.asarray(v_rs.imag, order='C')
        if at_gamma_point:
            v_rs = vR

        mat = eval_mat(cell, vR, task_list, comp=comp, hermi=hermi, deriv=deriv,
                       xctype='GGA', kpts=kpts, grid_level=ilevel, mesh=mesh)
        vj_kpts += np.asarray(mat).reshape(nset,-1,comp,nao,nao)
        if not at_gamma_point and abs(vI).max() > IMAG_TOL:
            raise NotImplementedError

    if nset == 1:
        vj_kpts = vj_kpts[0]
    return vj_kpts


def _rks_gga_wv0(rho, vxc, weight):
    vrho, vgamma = vxc[:2]
    ngrid = vrho.size
    wv = np.empty((4,ngrid))
    wv[0]  = np.multiply(weight, vrho, out=wv[0])
    for i in range(1, 4):
        wv[i] = np.multiply(weight * 2, np.multiply(vgamma, rho[i], out=wv[i]), out=wv[i])
    return wv


def _uks_gga_wv0(rho, vxc, weight):
    rhoa, rhob = rho
    vrho, vsigma = vxc[:2]
    ngrids = vrho.shape[0]
    wv = np.empty((2, 4, ngrids))
    wv[0,0]  = np.multiply(weight, vrho[:,0], out=wv[0,0])
    for i in range(1,4):
        wv[0,i] = np.multiply(2., np.multiply(rhoa[i], vsigma[:,0], out=wv[0,i]), out=wv[0,i])
        wv[0,i] = np.add(wv[0,i], np.multiply(rhob[i], vsigma[:,1]), out=wv[0,i])
        wv[0,i] = np.multiply(weight, wv[0,i], out=wv[0,i])
    wv[1,0]  = np.multiply(weight, vrho[:,1], out=wv[1,0])
    for i in range(1,4):
        wv[1,i] = np.multiply(2., np.multiply(rhob[i], vsigma[:,2], out=wv[1,i]), out=wv[1,i])
        wv[1,i] = np.add(wv[1,i], np.multiply(rhoa[i], vsigma[:,1]), out=wv[1,i])
        wv[1,i] = np.multiply(weight, wv[1,i], out=wv[1,i])
    return wv


def _rks_gga_wv0_pw(cell, rho, vxc, weight, mesh):
    vrho, vgamma = vxc[:2]
    ngrid = vrho.size
    buf = np.empty((3,ngrid))
    for i in range(1, 4):
        buf[i-1] = np.multiply(vgamma, rho[i], out=buf[i-1])

    vrho_freq = tools.fft(vrho, mesh).reshape((1,ngrid))
    buf_freq = tools.fft(buf, mesh).reshape((3,ngrid))
    Gv = cell.get_Gv(mesh)
    #out  = vrho_freq - 2j * np.einsum('px,xp->p', Gv, buf_freq)
    #out *= weight

    out = np.empty((ngrid,), order="C", dtype=np.complex128)
    func = getattr(libdft, 'get_gga_vrho_gs', None)
    func(out.ctypes.data_as(ctypes.c_void_p),
         vrho_freq.ctypes.data_as(ctypes.c_void_p),
         buf_freq.ctypes.data_as(ctypes.c_void_p),
         Gv.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_double(weight), ctypes.c_int(ngrid))
    return out


def _uks_gga_wv0_pw(cell, rho, vxc, weight, mesh):
    rhoa, rhob = rho
    vrho, vgamma = vxc[:2]
    ngrid = vrho.shape[0]
    buf = np.empty((2,3,ngrid))
    for i in range(1, 4):
        buf[0,i-1] = np.multiply(vgamma[:,0], rhoa[i], out=buf[0,i-1])
        tmp = np.multiply(vgamma[:,1], rhob[i])
        tmp = np.multiply(.5, tmp, out=tmp)
        buf[0,i-1] = np.add(buf[0,i-1], tmp, out=buf[0,i-1])

        buf[1,i-1] = np.multiply(vgamma[:,2], rhob[i], out=buf[1,i-1])
        tmp = np.multiply(vgamma[:,1], rhoa[i])
        tmp = np.multiply(.5, tmp, out=tmp)
        buf[1,i-1] = np.add(buf[1,i-1], tmp, out=buf[1,i-1])


    vrho_freq = tools.fft(vrho.T, mesh).reshape((2,ngrid))
    buf_freq = tools.fft(buf.reshape(-1,ngrid), mesh).reshape((2,3,ngrid))
    Gv = cell.get_Gv(mesh)
    #out  = vrho_freq - 2j * np.einsum('px,xp->p', Gv, buf_freq)
    #out *= weight

    out = np.empty((2,ngrid), order="C", dtype=np.complex128)
    func = getattr(libdft, 'get_gga_vrho_gs')
    for s in range(2):
        func(out[s].ctypes.data_as(ctypes.c_void_p),
             vrho_freq[s].ctypes.data_as(ctypes.c_void_p),
             buf_freq[s].ctypes.data_as(ctypes.c_void_p),
             Gv.ctypes.data_as(ctypes.c_void_p),
             ctypes.c_double(weight), ctypes.c_int(ngrid))
    return out


def nr_rks(mydf, xc_code, dm_kpts, hermi=1, kpts=None,
           kpts_band=None, with_j=False, return_j=False, verbose=None):
    '''
    Same as multigrid.nr_rks, but considers Hermitian symmetry also for GGA
    '''
    if kpts is None: kpts = mydf.kpts
    log = logger.new_logger(mydf, verbose)
    cell = mydf.cell
    dm_kpts = np.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band

    ni = mydf._numint
    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv)

    mesh = mydf.mesh
    ngrids = np.prod(mesh)

    coulG = tools.get_coulG(cell, mesh=mesh)
    #vG = np.einsum('ng,g->ng', rhoG[:,0], coulG)
    vG = np.empty_like(rhoG[:,0], dtype=np.result_type(rhoG[:,0], coulG))
    for i, rhoG_i in enumerate(rhoG[:,0]):
        vG[i] = np.multiply(rhoG_i, coulG, out=vG[i])
    coulG = None

    if mydf.vpplocG_part1 is not None:
        for i in range(nset):
            #vG[i] += mydf.vpplocG_part1 * 2
            vG[i] = np.add(vG[i], np.multiply(2., mydf.vpplocG_part1), out=vG[i])

    #ecoul = .5 * np.einsum('ng,ng->n', rhoG[:,0].real, vG.real)
    #ecoul+= .5 * np.einsum('ng,ng->n', rhoG[:,0].imag, vG.imag)
    ecoul = np.zeros((rhoG.shape[0],))
    for i in range(rhoG.shape[0]):
        ecoul[i] = .5 * np.vdot(rhoG[i,0], vG[i]).real

    ecoul /= cell.vol
    log.debug('Multigrid Coulomb energy %s', ecoul)

    if mydf.vpplocG_part1 is not None:
        for i in range(nset):
            #vG[i] -= mydf.vpplocG_part1
            vG[i] = np.subtract(vG[i], mydf.vpplocG_part1, out=vG[i])

    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    rhoR = rhoR.reshape(nset,-1,ngrids)
    wv_freq = []
    nelec = np.zeros(nset)
    excsum = np.zeros(nset)
    for i in range(nset):
        exc, vxc = ni.eval_xc(xc_code, rhoR[i], spin=0, deriv=1)[:2]
        if xctype == 'LDA':
            wv = np.multiply(weight, vxc[0])
            wv_freq.append(tools.fft(wv, mesh))
            wv = None
        elif xctype == 'GGA':
            if GGA_METHOD.upper() == 'FFT':
                wv_freq.append(_rks_gga_wv0_pw(cell, rhoR[i], vxc, weight, mesh).reshape(1,ngrids))
            else:
                wv = _rks_gga_wv0(rhoR[i], vxc, weight)
                wv_freq.append(tools.fft(wv, mesh))
                wv = None
        else:
            raise NotImplementedError

        nelec[i]  += np.sum(rhoR[i,0]) * weight
        excsum[i] += np.sum(np.multiply(rhoR[i,0], exc)) * weight
        exc = vxc = None

    rhoR = rhoG = None

    if len(wv_freq) == 1:
        wv_freq = wv_freq[0].reshape(nset,-1,*mesh)
    else:
        wv_freq = np.asarray(wv_freq).reshape(nset,-1,*mesh)

    if nset == 1:
        ecoul = ecoul[0]
        nelec = nelec[0]
        excsum = excsum[0]
    log.debug('Multigrid exc %s  nelec %s', excsum, nelec)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    if xctype == 'LDA':
        if with_j:
            wv_freq[:,0] += vG.reshape(nset,*mesh)
        veff = _get_j_pass2(mydf, wv_freq, kpts_band, verbose=log)
    elif xctype == 'GGA':
        if with_j:
            #wv_freq[:,0] += vG.reshape(nset,*mesh)
            wv_freq[:,0] = np.add(wv_freq[:,0], vG.reshape(nset,*mesh), out=wv_freq[:,0])
        if GGA_METHOD.upper() == 'FFT':
            veff = _get_j_pass2(mydf, wv_freq, kpts_band, verbose=log)
        else:
            veff = _get_gga_pass2(mydf, wv_freq, kpts_band, hermi=hermi, verbose=log)
    wv_freq = None
    veff = _format_jks(veff, dm_kpts, input_band, kpts)

    if return_j:
        vj = _get_j_pass2(mydf, vG, kpts_band, verbose=log)
        vj = _format_jks(veff, dm_kpts, input_band, kpts)
    else:
        vj = None
    vG = None

    veff = lib.tag_array(veff, ecoul=ecoul, exc=excsum, vj=vj, vk=None)
    return nelec, excsum, veff

def nr_uks(mydf, xc_code, dm_kpts, hermi=1, kpts=None,
           kpts_band=None, with_j=False, return_j=False, verbose=None):
    if kpts is None: kpts = mydf.kpts
    log = logger.new_logger(mydf, verbose)
    cell = mydf.cell
    dm_kpts = np.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    nset //= 2
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band

    mesh = mydf.mesh
    ngrids = np.prod(mesh)
    ni = mydf._numint
    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1

    rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv)
    rhoG = rhoG.reshape(nset,2,-1,ngrids)

    coulG = tools.get_coulG(cell, mesh=mesh)
    #vG = np.einsum('nsg,g->ng', rhoG[:,:,0], coulG)
    vG = np.empty((nset,ngrids), dtype=np.result_type(rhoG[:,:,0], coulG))
    for i, rhoG_i in enumerate(rhoG[:,:,0]):
        vG[i] = np.multiply(np.add(rhoG_i[0], rhoG_i[1]), coulG, out=vG[i])
    coulG = None

    if mydf.vpplocG_part1 is not None:
        for i in range(nset):
            #vG[i] += mydf.vpplocG_part1 * 2
            vG[i] = np.add(vG[i], np.multiply(2., mydf.vpplocG_part1), out=vG[i])

    ecoul = np.zeros(nset)
    for i in range(nset):
        ecoul[i] = .5 * np.vdot(np.add(rhoG[i,0,0], rhoG[i,1,0]), vG[i]).real

    ecoul /= cell.vol
    log.debug('Multigrid Coulomb energy %s', ecoul)

    if mydf.vpplocG_part1 is not None:
        for i in range(nset):
            #vG[i] -= mydf.vpplocG_part1
            vG[i] = np.subtract(vG[i], mydf.vpplocG_part1, out=vG[i])

    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    rhoR = rhoR.reshape(nset,2,-1,ngrids)
    wv_freq = []
    nelec = np.zeros(nset)
    excsum = np.zeros(nset)
    for i in range(nset):
        exc, vxc = ni.eval_xc(xc_code, rhoR[i], spin=1, deriv=1)[:2]
        if xctype == 'LDA':
            wv = np.multiply(weight, vxc[0].T)
            wv_freq.append(tools.fft(wv, mesh))
            wv = None
        elif xctype == 'GGA':
            if GGA_METHOD.upper() == 'FFT':
                wv_freq.append(_uks_gga_wv0_pw(cell, rhoR[i], vxc, weight, mesh))
            else:
                wv = _uks_gga_wv0(rhoR[i], vxc, weight)
                wv_freq.append(tools.fft(wv.reshape(-1,*mesh), mesh))
                wv = None
        else:
            raise NotImplementedError

        nelec[i]  += np.sum(rhoR[i,:,0]).sum() * weight
        excsum[i] += np.sum(np.multiply(np.add(rhoR[i,0,0],rhoR[i,1,0]), exc)) * weight
        exc = vxc = None

    rhoR = rhoG = None

    if len(wv_freq) == 1:
        wv_freq = wv_freq[0].reshape(nset,2,-1,*mesh)
    else:
        wv_freq = np.asarray(wv_freq).reshape(nset,2,-1,*mesh)

    if nset == 1:
        ecoul = ecoul[0]
        nelec = nelec[0]
        excsum = excsum[0]
    log.debug('Multigrid exc %s  nelec %s', excsum, nelec)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    if xctype == 'LDA':
        if with_j:
            for s in range(2):
                wv_freq[:,s,0] += vG.reshape(nset,*mesh)
        veff = _get_j_pass2(mydf, wv_freq, kpts_band, verbose=log)
    elif xctype == 'GGA':
        if with_j:
            #wv_freq[:,:,0] += vG.reshape(nset,*mesh)
            for s in range(2):
                wv_freq[:,s,0] = np.add(wv_freq[:,s,0], vG.reshape(nset,*mesh), out=wv_freq[:,s,0])
        if GGA_METHOD.upper() == 'FFT':
            veff = _get_j_pass2(mydf, wv_freq, kpts_band, verbose=log)
        else:
            veff = _get_gga_pass2(mydf, wv_freq, kpts_band, hermi=hermi, verbose=log)
    wv_freq = None
    veff = _format_jks(veff, dm_kpts, input_band, kpts)

    if return_j:
        vj = _get_j_pass2(mydf, vG, kpts_band, verbose=log)
        vj = _format_jks(veff, dm_kpts, input_band, kpts)
    else:
        vj = None
    vG = None

    veff = lib.tag_array(veff, ecoul=ecoul, exc=excsum, vj=vj, vk=None)
    return nelec, excsum, veff

def get_veff_ip1(mydf, dm_kpts, xc_code=None, kpts=np.zeros((1,3)), kpts_band=None, spin=0):
    cell = mydf.cell
    dm_kpts = np.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    kpts_band = _format_kpts_band(kpts_band, kpts)
    if spin == 1:
        nset //= 2

    mesh = mydf.mesh
    ngrids = np.prod(mesh)
    ni = mydf._numint
    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    rhoG = _eval_rhoG(mydf, dm_kpts, hermi=1, kpts=kpts_band, deriv=deriv)
    if spin == 1:
        rhoG = rhoG.reshape(nset,2,-1,ngrids)
    # cache rhoG for core density gradients
    mydf.rhoG = rhoG

    coulG = tools.get_coulG(cell, mesh=mesh)
    vG = np.empty((nset,ngrids), dtype=np.result_type(rhoG, coulG))
    for i in range(nset):
        if spin == 0:
            vG[i] = np.multiply(rhoG[i,0], coulG, out=vG[i])
        elif spin == 1:
            tmp = np.add(rhoG[i,0,0], rhoG[i,1,0])
            vG[i] = np.multiply(tmp, coulG, out=vG[i])

    if mydf.vpplocG_part1 is not None:
        for i in range(nset):
            vG[i] = np.add(vG[i], mydf.vpplocG_part1, out=vG[i])

    weight = cell.vol / ngrids

    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    if spin == 0:
        rhoR = rhoR.reshape(nset,-1,ngrids)
    elif spin == 1:
        rhoR = rhoR.reshape(nset,2,-1,ngrids)

    wv_freq = []
    for i in range(nset):
        exc, vxc = ni.eval_xc(xc_code, rhoR[i], spin=spin, deriv=1)[:2]
        if spin == 0:
            if xctype == 'LDA':
                wv = np.multiply(weight, vxc[0])
                wv_freq.append(tools.fft(wv, mesh))
                wv = None
            elif xctype == 'GGA':
                if GGA_METHOD.upper() == 'FFT':
                    wv_freq.append(_rks_gga_wv0_pw(cell, rhoR[i], vxc, weight, mesh).reshape(1,ngrids))
                else:
                    wv = _rks_gga_wv0(rhoR[i], vxc, weight)
                    wv_freq.append(tools.fft(wv, mesh))
            else:
                raise NotImplementedError
        elif spin == 1:
            if xctype == 'LDA':
                wv = np.multiply(weight, vxc[0].T)
                wv_freq.append(tools.fft(wv, mesh))
                wv = None
            elif xctype == 'GGA':
                if GGA_METHOD.upper() == 'FFT':
                    wv_freq.append(_uks_gga_wv0_pw(cell, rhoR[i], vxc, weight, mesh))
                else:
                    wv = _uks_gga_wv0(rhoR[i], vxc, weight)
                    wv_freq.append(tools.fft(wv.reshape(-1,*mesh), mesh))
                wv = None
            else:
                raise NotImplementedError

    rhoR = rhoG = None
    if spin == 0:
        if len(wv_freq) == 1:
            wv_freq = wv_freq[0].reshape(nset,-1,*mesh)
        else:
            wv_freq = np.asarray(wv_freq).reshape(nset,-1,*mesh)
    elif spin == 1:
        if len(wv_freq) == 1:
            wv_freq = wv_freq[0].reshape(nset,2,-1,*mesh)
        else:
            wv_freq = np.asarray(wv_freq).reshape(nset,2,-1,*mesh)

    for i in range(nset):
        if spin == 0:
            wv_freq[i,0] = np.add(wv_freq[i,0], vG[i].reshape(*mesh), out=wv_freq[i,0])
        elif spin == 1:
            for s in range(2):
                wv_freq[i,s,0] = np.add(wv_freq[i,s,0], vG[i].reshape(*mesh), out=wv_freq[i,s,0])

    if xctype == 'LDA':
        vj_kpts = _get_j_pass2_ip1(mydf, wv_freq, kpts_band, hermi=0, deriv=1)
    elif xctype == 'GGA':
        if GGA_METHOD.upper() == 'FFT':
            vj_kpts = _get_j_pass2_ip1(mydf, wv_freq, kpts_band, hermi=0, deriv=1)
        else:
            vj_kpts = _get_gga_pass2_ip1(mydf, wv_freq, kpts_band, hermi=0, deriv=1)
    else:
        raise NotImplementedError

    comp = 3
    nao = cell.nao
    if spin == 0:
        vj_kpts = vj_kpts.reshape(nset,nkpts,comp,nao,nao)
    elif spin == 1:
        vj_kpts = vj_kpts.reshape(nset,2,nkpts,comp,nao,nao)
    vj_kpts = np.moveaxis(vj_kpts, -3, -4)

    if nkpts == 1:
        vj_kpts = vj_kpts[...,0,:,:]
    if nset == 1:
        vj_kpts = vj_kpts[0]
    return vj_kpts


class MultiGridFFTDF2(MultiGridFFTDF):
    '''
    Base class for multigrid DFT (version 2).

    Attributes:
        task_list : TaskList instance
            Task list recording which primitive basis function pairs
            need to be considered.
        vpplocG_part1 : arrary
            Short-range part of the local pseudopotential represented
            in the reciprocal space. It is cached to reduce cost.
        rhoG : array
            Electronic density represented in the reciprocal space.
            It is cached in nuclear gradient calculations to reduce cost.
    '''
    ngrids = getattr(__config__, 'pbc_dft_multigrid_ngrids', 4)
    ke_ratio = getattr(__config__, 'pbc_dft_multigrid_ke_ratio', 3.0)
    rel_cutoff = getattr(__config__, 'pbc_dft_multigrid_rel_cutoff', 20.0)
    _keys = {'ngrids', 'ke_ratio', 'rel_cutoff',
             'task_list', 'vpplocG_part1', 'rhoG'}

    def __init__(self, cell, kpts=np.zeros((1,3))):
        fft.FFTDF.__init__(self, cell, kpts)
        self.task_list = None
        self.vpplocG_part1 = None
        self.rhoG = None
        if not gamma_point(kpts):
            raise NotImplementedError('MultiGridFFTDF2 only supports Gamma-point calculations.')
        a = cell.lattice_vectors()
        if abs(a-np.diag(a.diagonal())).max() > 1e-12:
            raise NotImplementedError('MultiGridFFTDF2 only supports orthorhombic lattices.')

    def reset(self, cell=None):
        self.vpplocG_part1 = None
        self.rhoG = None
        if self.task_list is not None:
            free_task_list(self.task_list)
            self.task_list = None
        fft.FFTDF.reset(self, cell=cell)

    def __del__(self):
        self.reset()

    def get_veff_ip1(self, dm, xc_code=None, kpts=None, kpts_band=None, spin=0):
        if kpts is None:
            if self.kpts is None:
                kpts = np.zeros(1,3)
            else:
                kpts = self.kpts
        kpts = kpts.reshape(-1,3)
        vj = get_veff_ip1(self, dm, xc_code=xc_code,
                          kpts=kpts, kpts_band=kpts_band, spin=spin)
        return vj

    def get_pp(self, kpts=None):
        '''Compute the GTH pseudopotential matrix, which includes
        the second part of the local potential and the non-local potential.
        The first part of the local potential is cached as `vpplocG_part1`,
        which is the reciprocal space representation, to be added to the electron
        density for computing the Coulomb matrix.
        In order to get the full PP matrix, the potential due to `vpplocG_part1`
        needs to be added.
        '''
        self.vpplocG_part1 = _get_vpplocG_part1(self, with_rho_core=True)
        return _get_pp_without_erf(self, kpts)

    vpploc_part1_nuc_grad = vpploc_part1_nuc_grad
