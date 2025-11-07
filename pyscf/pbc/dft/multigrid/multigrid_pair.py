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

import numpy as np
from pyscf import __config__
from pyscf import lib
from pyscf.lib import logger
from pyscf.gto import moleintor
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.df import fft
from pyscf.pbc import tools as pbctools
from pyscf.pbc.df.df_jk import (
    _format_dms,
    _format_kpts_band,
    _format_jks,
)
from pyscf.pbc.dft.multigrid.pp import (
    _get_vpplocG_part1,
    _get_pp_without_erf,
    vpploc_part1_nuc_grad,
    get_vpploc_part1_ip1,
)
from pyscf.pbc.dft.multigrid.utils import (
    _take_4d,
    _take_5d,
    _takebak_4d,
    _takebak_5d,
)
from pyscf.pbc.dft.multigrid.multigrid import MultiGridNumInt as MultiGridNumInt_v1
from pyscf.pbc.dft.multigrid import _backend_c as backend

NTASKS = getattr(__config__, 'pbc_dft_multigrid_ntasks', 4)
KE_RATIO = getattr(__config__, 'pbc_dft_multigrid_ke_ratio', 3.0)
REL_CUTOFF = getattr(__config__, 'pbc_dft_multigrid_rel_cutoff', 20.0)
GGA_METHOD = getattr(__config__, 'pbc_dft_multigrid_gga_method', 'FFT')

EXTRA_PREC = getattr(__config__, 'pbc_gto_eval_gto_extra_precision', 1e-2)
RHOG_HIGH_ORDER = getattr(__config__, 'pbc_dft_multigrid_rhog_high_order', False)
PTR_EXPDROP = 16


def multi_grids_tasks(cell, ke_cutoff=None, hermi=0,
                      ntasks=NTASKS, ke_ratio=KE_RATIO, rel_cutoff=REL_CUTOFF):
    if ke_cutoff is None:
        ke_cutoff = cell.ke_cutoff
    a = cell.lattice_vectors()
    if ke_cutoff is None:
        ke_cutoff = pbctools.mesh_to_cutoff(a, cell.mesh).max()
    ke1 = ke_cutoff
    cutoff = [ke1,]
    for i in range(ntasks-1):
        ke1 /= ke_ratio
        cutoff.append(ke1)
    cutoff.reverse()
    mesh = []
    for ke in cutoff[:-1]:
        mesh.append(tools.cutoff_to_mesh(a, ke))
    mesh.append(cell.mesh)
    logger.info(cell, 'ke_cutoff for multigrid tasks:\n%s', cutoff)
    logger.info(cell, 'meshes for multigrid tasks:\n%s', mesh)
    gridlevel_info = backend.GridLevel_Info(cutoff, rel_cutoff, mesh)
    task_list = backend.TaskList(cell, gridlevel_info, hermi=hermi)
    return task_list


def _update_task_list(mydf, hermi=0, ntasks=None, ke_ratio=None, rel_cutoff=None):
    '''Update :attr:`task_list` if necessary.
    '''
    cell = mydf.cell
    if ntasks is None:
        ntasks = mydf.ntasks
    if ke_ratio is None:
        ke_ratio = mydf.ke_ratio
    if rel_cutoff is None:
        rel_cutoff = mydf.rel_cutoff

    need_update = False
    task_list = getattr(mydf, 'task_list', None)
    if task_list is None:
        need_update = True
    else:
        ke_cutoff = cell.ke_cutoff
        if ke_cutoff is None:
            a = cell.lattice_vectors()
            ke_cutoff = pbctools.mesh_to_cutoff(a, cell.mesh).max()
        hermi_orig = task_list.hermi
        nlevels = task_list.nlevels
        rel_cutoff_orig = task_list.gridlevel_info.rel_cutoff
        ke_cutoff_orig = task_list.gridlevel_info.cutoff[-1]
        if (hermi_orig > hermi or
                nlevels != ntasks or
                abs(rel_cutoff_orig-rel_cutoff) > 1e-12 or
                abs(ke_cutoff_orig - ke_cutoff) > 1e-12):
            need_update = True
            logger.debug(mydf, 'Hermiticity or cutoffs changed; will update the task list!')

    if need_update:
        task_list = multi_grids_tasks(cell, hermi=hermi, ntasks=ntasks,
                                      ke_ratio=ke_ratio, rel_cutoff=rel_cutoff)
        mydf.task_list = task_list
    return task_list


def eval_rho(cell, dm, task_list, shls_slice=None, hermi=0, xctype='LDA', kpts=None,
             dimension=None, cell1=None, shls_slice1=None,
             a=None, ignore_imag=False):
    '''Collocate density (and gradients) on the real-space grid.

    The two sets of Gaussian basis functions can be different.

    Returns:
        rho: `RS_Grid` object
            Densities on real space multigrids.
    '''
    cell0 = cell
    shls_slice0 = shls_slice
    if cell1 is None:
        cell1 = cell0

    #TODO mixture of cartesian and spherical bases
    assert cell0.cart == cell1.cart

    ish_atm = cell0._atm
    ish_bas = cell0._bas
    ish_env = cell0._env
    ish_env[PTR_EXPDROP] = cell0.precision * EXTRA_PREC

    if cell1 is cell0:
        jsh_atm = ish_atm
        jsh_bas = ish_bas
        jsh_env = ish_env
    else:
        jsh_atm = cell1._atm
        jsh_bas = cell1._bas
        jsh_env = cell1._env
        jsh_env[PTR_EXPDROP] = cell1.precision * EXTRA_PREC

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

    dm = np.asarray(dm)
    assert dm.shape[-2:] == (naoi, naoj)

    if dimension is None:
        dimension = cell0.dimension
    assert dimension == getattr(cell1, "dimension", None)

    if dimension == 0 or kpts is None or gamma_point(kpts):
        dm = dm.reshape(-1,1,naoi,naoj)
    else:
        raise NotImplementedError
    n_dm = dm.shape[0]

    if a is None:
        a = cell0.lattice_vectors()
        if cell1 is not cell:
            a1 = cell1.lattice_vectors()
            if abs(a-a1).max() > 1e-12:
                raise RuntimeError('The two cell objects must have the same lattice vectors.')
    b = np.linalg.inv(a.T)

    rho = []
    for i, dm_i in enumerate(dm):
        rho_i = backend.grid_collocate(
                    xctype, dm_i,
                    task_list, hermi,
                    (i0, i1, j0, j1),
                    ao_loc0, ao_loc1, dimension,
                    a, b,
                    ish_atm, ish_bas, ish_env,
                    jsh_atm, jsh_bas, jsh_env,
                    cell0.cart)
        rho.append(rho_i)

    if n_dm == 1:
        rho = rho[0]
    return rho


def _eval_rhoG(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), deriv=0,
               rhog_high_order=RHOG_HIGH_ORDER):
    if deriv >= 2:
        raise NotImplementedError
    cell = mydf.cell

    dm_kpts = np.asarray(dm_kpts)
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    task_list = _update_task_list(mydf, hermi=hermi, ntasks=mydf.ntasks,
                                  ke_ratio=mydf.ke_ratio, rel_cutoff=mydf.rel_cutoff)

    gga_high_order = False
    if deriv == 0:
        xctype = 'LDA'
        rhodim = 1
    elif deriv == 1:
        if rhog_high_order:
            raise NotImplementedError
        else:  # approximate high order derivatives in reciprocal space
            gga_high_order = True
            xctype = 'LDA'
            rhodim = 1
            deriv = 0
        assert hermi == 1 or gamma_point(kpts)

    ignore_imag = (hermi == 1)

    nx, ny, nz = mydf.mesh
    mem_avail = mydf.max_memory - lib.current_memory()[0]
    mem_needed = rhodim * nx * ny * nz * lib.num_threads() * 8 / 1e6
    if mem_needed > mem_avail:
        logger.warn(mydf, f'At least {mem_needed} MB of memory is needed for eval_rho. '
                    f'Currently {mem_avail} MB of memory is available.')
    rs_rho = eval_rho(cell, dms, task_list, hermi=hermi, xctype=xctype, kpts=kpts,
                      ignore_imag=ignore_imag)

    rhoG = np.zeros((nset*rhodim,nx,ny,nz), dtype=np.complex128)
    for ilevel, mesh in enumerate(task_list.gridlevel_info.mesh):
        ngrids = np.prod(mesh)
        if nset > 1:
            rho = []
            for i in range(nset):
                rho.append(rs_rho[i][ilevel])
            rho = np.asarray(rho)
        else:
            rho = rs_rho[ilevel]

        weight = 1./nkpts * cell.vol/ngrids
        rho_freq = tools.fft(rho.reshape(nset*rhodim, -1), mesh)
        rho = None
        rho_freq *= weight
        gx = np.fft.fftfreq(mesh[0], 1./mesh[0]).astype(np.int32)
        gy = np.fft.fftfreq(mesh[1], 1./mesh[1]).astype(np.int32)
        gz = np.fft.fftfreq(mesh[2], 1./mesh[2]).astype(np.int32)
        _takebak_4d(rhoG, rho_freq.reshape((-1,) + tuple(mesh)), (None, gx, gy, gz))
        rho_freq = None

    rs_rho = None

    rhoG = rhoG.reshape(nset,rhodim,-1)
    if gga_high_order:
        Gv = cell.get_Gv(mydf.mesh)
        #:rhoG1 = np.einsum('np,px->nxp', rhoG[:,0], 1j*Gv)
        rhoG1 = backend.gradient_gs(rhoG[:,0], Gv)
        rhoG = np.concatenate([rhoG, rhoG1], axis=1)
        Gv = rhoG1 = None
    return rhoG


def eval_mat(cell, weights, task_list, shls_slice=None, comp=1, hermi=0, deriv=0,
             xctype='LDA', kpts=None, grid_level=None, dimension=None, mesh=None,
             cell1=None, shls_slice1=None, a=None):
    if deriv == 1:
        assert comp == 3
        assert hermi == 0
    elif deriv > 1:
        raise NotImplementedError

    cell0 = cell
    shls_slice0 = shls_slice
    if cell1 is None:
        cell1 = cell0

    if mesh is None:
        mesh = cell0.mesh

    #TODO mixture of cartesian and spherical bases
    assert cell0.cart == cell1.cart

    ish_atm = cell0._atm
    ish_bas = cell0._bas
    ish_env = cell0._env
    ish_env[PTR_EXPDROP] = cell0.precision * EXTRA_PREC

    if cell1 is cell0:
        jsh_atm = ish_atm
        jsh_bas = ish_bas
        jsh_env = ish_env
    else:
        jsh_atm = cell1._atm
        jsh_bas = cell1._bas
        jsh_env = cell1._env
        jsh_env[PTR_EXPDROP] = cell1.precision * EXTRA_PREC

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
    if hermi == 1:
        ao_loc1 = ao_loc0
    else:
        key1 = 'cart' if cell1.cart else 'sph'
        ao_loc1 = moleintor.make_loc(jsh_bas, key1)

    if dimension is None:
        dimension = cell0.dimension
    assert dimension == getattr(cell1, "dimension", None)

    weights = np.asarray(weights)
    if dimension == 0 or kpts is None or gamma_point(kpts):
        assert weights.dtype == np.double
    else:
        raise NotImplementedError

    if a is None:
        a = cell0.lattice_vectors()
        if cell1 is not cell:
            a1 = cell1.lattice_vectors()
            if abs(a-a1).max() > 1e-12:
                raise RuntimeError('The two cell objects must have the same lattice vectors.')
    b = np.linalg.inv(a.T)

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

    out = []
    for wv in weights:
        mat = backend.grid_integrate(
                xctype, wv,
                task_list, comp, hermi, grid_level,
                (i0, i1, j0, j1),
                ao_loc0, ao_loc1, dimension,
                a, b,
                ish_atm, ish_bas, ish_env,
                jsh_atm, jsh_bas, jsh_env,
                cell0.cart)
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

    task_list = _update_task_list(mydf, hermi=hermi, ntasks=mydf.ntasks,
                                  ke_ratio=mydf.ke_ratio, rel_cutoff=mydf.rel_cutoff)

    if gamma_point(kpts):
        vj_kpts = np.zeros((nset,nkpts,nao,nao))
    else:
        raise NotImplementedError

    nlevels = task_list.nlevels
    meshes = task_list.gridlevel_info.mesh
    for ilevel in range(nlevels):
        mesh = meshes[ilevel]
        ngrids = np.prod(mesh)

        gx = np.fft.fftfreq(mesh[0], 1./mesh[0]).astype(np.int32)
        gy = np.fft.fftfreq(mesh[1], 1./mesh[1]).astype(np.int32)
        gz = np.fft.fftfreq(mesh[2], 1./mesh[2]).astype(np.int32)
        sub_vG = _take_4d(vG, (None, gx, gy, gz)).reshape(nset,ngrids)

        v_rs = tools.ifft(sub_vG, mesh).reshape(nset,ngrids)
        vR = np.asarray(v_rs.real, order='C')
        mat = eval_mat(cell, vR, task_list, comp=1, hermi=hermi,
                       xctype='LDA', kpts=kpts, grid_level=ilevel, mesh=mesh)
        vj_kpts += np.asarray(mat).reshape(nset,-1,nao,nao)

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

    task_list = _update_task_list(mydf, hermi=hermi, ntasks=mydf.ntasks,
                                  ke_ratio=mydf.ke_ratio, rel_cutoff=mydf.rel_cutoff)

    if gamma_point(kpts):
        vj_kpts = np.zeros((nset,nkpts,comp,nao,nao))
    else:
        raise NotImplementedError

    nlevels = task_list.nlevels
    meshes = task_list.gridlevel_info.mesh
    for ilevel in range(nlevels):
        mesh = meshes[ilevel]
        ngrids = np.prod(mesh)

        gx = np.fft.fftfreq(mesh[0], 1./mesh[0]).astype(np.int32)
        gy = np.fft.fftfreq(mesh[1], 1./mesh[1]).astype(np.int32)
        gz = np.fft.fftfreq(mesh[2], 1./mesh[2]).astype(np.int32)
        sub_vG = _take_4d(vG, (None, gx, gy, gz)).reshape(nset,ngrids)

        v_rs = tools.ifft(sub_vG, mesh).reshape(nset,ngrids)
        vR = np.asarray(v_rs.real, order='C')
        mat = eval_mat(cell, vR, task_list, comp=comp, hermi=hermi, deriv=deriv,
                       xctype='LDA', kpts=kpts, grid_level=ilevel, mesh=mesh)
        mat = np.asarray(mat).reshape(nset,-1,comp,nao,nao)
        vj_kpts += mat

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

    task_list = _update_task_list(mydf, hermi=hermi, ntasks=mydf.ntasks,
                                  ke_ratio=mydf.ke_ratio, rel_cutoff=mydf.rel_cutoff)

    if gamma_point(kpts):
        veff = np.zeros((nset,nkpts,nao,nao))
    else:
        raise NotImplementedError

    nlevels = task_list.nlevels
    meshes = task_list.gridlevel_info.mesh
    for ilevel in range(nlevels):
        mesh = meshes[ilevel]
        ngrids = np.prod(mesh)

        gx = np.fft.fftfreq(mesh[0], 1./mesh[0]).astype(np.int32)
        gy = np.fft.fftfreq(mesh[1], 1./mesh[1]).astype(np.int32)
        gz = np.fft.fftfreq(mesh[2], 1./mesh[2]).astype(np.int32)
        sub_vG = _take_5d(vG, (None, None, gx, gy, gz)).reshape(-1,ngrids)
        wv = tools.ifft(sub_vG, mesh).reshape(nset,4,ngrids)
        wv = np.asarray(wv.real, order='C')
        mat = eval_mat(cell, wv, task_list, comp=1, hermi=hermi,
                       xctype='GGA', kpts=kpts, grid_level=ilevel, mesh=mesh)
        mat = np.asarray(mat).reshape(nset,-1,nao,nao)
        veff += mat

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

    task_list = _update_task_list(mydf, hermi=hermi, ntasks=mydf.ntasks,
                                  ke_ratio=mydf.ke_ratio, rel_cutoff=mydf.rel_cutoff)

    if gamma_point(kpts):
        vj_kpts = np.zeros((nset,nkpts,comp,nao,nao))
    else:
        raise NotImplementedError

    for ilevel, mesh in enumerate(task_list.gridlevel_info.mesh):
        ngrids = np.prod(mesh)

        gx = np.fft.fftfreq(mesh[0], 1./mesh[0]).astype(np.int32)
        gy = np.fft.fftfreq(mesh[1], 1./mesh[1]).astype(np.int32)
        gz = np.fft.fftfreq(mesh[2], 1./mesh[2]).astype(np.int32)
        sub_vG = _take_5d(vG, (None, None, gx, gy, gz)).reshape(-1,ngrids)

        v_rs = tools.ifft(sub_vG, mesh).reshape(nset,4,ngrids)
        vR = np.asarray(v_rs.real, order='C')
        mat = eval_mat(cell, vR, task_list, comp=comp, hermi=hermi, deriv=deriv,
                       xctype='GGA', kpts=kpts, grid_level=ilevel, mesh=mesh)
        vj_kpts += np.asarray(mat).reshape(nset,-1,comp,nao,nao)

    if nset == 1:
        vj_kpts = vj_kpts[0]
    return vj_kpts


def nr_rks(mydf, xc_code, dm_kpts, hermi=1, kpts=None,
           kpts_band=None, with_j=False, return_j=False, verbose=None):
    '''Compute the XC energy and RKS XC matrix using the multigrid algorithm.

    See also `multigrid.nr_rks`.
    '''
    if kpts is None:
        kpts = np.zeros((1, 3))
    elif isinstance(kpts, KPoints):
        if kpts.kpts.size > 3: # multiple k points
            dm_kpts = kpts.transform_dm(dm_kpts)
        kpts = kpts.kpts
    log = logger.new_logger(mydf, verbose)
    cell = mydf.cell
    dm_kpts = np.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    assert nset == 1
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band

    ni = mydf
    xctype = ni._xc_type(xc_code)
    if xctype in (None, 'LDA', 'HF'):
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    else:
        raise NotImplementedError
    rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv)

    mesh = mydf.mesh
    ngrids = np.prod(mesh)

    coulG = tools.get_coulG(cell, mesh=mesh)

    #:vG = np.einsum('ng,g->ng', rhoG[:,0], coulG)
    vG = np.multiply(rhoG[0,0], coulG)
    coulG = None

    if mydf.vpplocG_part1 is not None:
        vG += mydf.vpplocG_part1 * 2

    #:ecoul = .5 * np.einsum('ng,ng->n', rhoG[:,0].real, vG.real)
    #:ecoul+= .5 * np.einsum('ng,ng->n', rhoG[:,0].imag, vG.imag)
    ecoul = .5 * np.vdot(rhoG[0,0], vG).real

    ecoul /= cell.vol
    log.debug('Multigrid Coulomb energy %s', ecoul)

    if mydf.vpplocG_part1 is not None:
        vG -= mydf.vpplocG_part1

    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    rhoR = rhoR.reshape(-1,ngrids)
    exc, vxc = ni.eval_xc_eff(xc_code, rhoR, deriv=1, xctype=xctype)[:2]
    wv_freq = tools.fft(vxc, mesh).reshape(-1,ngrids)
    if xctype == 'GGA' and GGA_METHOD.upper() == 'FFT':
        #:wv_freq = (wv_freq[0] - 1j * np.einsum('px,xp->p', Gv, wv_freq[1:4])) * weight
        Gv = cell.get_Gv(ni.mesh)
        wv_freq = backend.get_gga_vrho_gs(wv_freq[:1], wv_freq[1:4], Gv, weight, ngrids, 1.)
    else:
        wv_freq *= weight

    nelec = np.sum(rhoR[0]) * weight
    excsum = np.dot(rhoR[0], exc) * weight
    exc = vxc = None
    rhoR = rhoG = None
    log.debug('Multigrid exc %s  nelec %s', excsum, nelec)

    if with_j:
        wv_freq[0] += vG

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    if xctype in (None, 'LDA', 'HF'):
        veff = _get_j_pass2(mydf, wv_freq, kpts_band, verbose=log)
    elif xctype == 'GGA':
        if GGA_METHOD.upper() == 'FFT':
            veff = _get_j_pass2(mydf, wv_freq, kpts_band, verbose=log)
        else:
            veff = _get_gga_pass2(mydf, wv_freq, kpts_band, hermi=hermi, verbose=log)
    else:
        raise NotImplementedError
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
    if kpts is None:
        kpts = np.zeros((1, 3))
    elif isinstance(kpts, KPoints):
        if kpts.kpts.size > 3: # multiple k points
            dm_kpts = kpts.transform_dm(dm_kpts)
        kpts = kpts.kpts
    log = logger.new_logger(mydf, verbose)
    cell = mydf.cell
    dm_kpts = np.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    nset //= 2
    assert nset == 1
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band

    mesh = mydf.mesh
    ngrids = np.prod(mesh)
    ni = mydf
    xctype = ni._xc_type(xc_code)
    if xctype in (None, 'LDA', 'HF'):
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    else:
        raise NotImplementedError

    rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv)
    rhoG = rhoG.reshape(2,-1,ngrids)
    rhoG_sf = rhoG[0,0] + rhoG[1,0]

    coulG = tools.get_coulG(cell, mesh=mesh)
    vG = rhoG_sf * coulG
    coulG = None

    if mydf.vpplocG_part1 is not None:
        vG += mydf.vpplocG_part1 * 2

    ecoul = .5 * np.vdot(rhoG_sf, vG).real

    ecoul /= cell.vol
    log.debug('Multigrid Coulomb energy %s', ecoul)

    if mydf.vpplocG_part1 is not None:
        vG -= mydf.vpplocG_part1

    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    rhoR = rhoR.reshape(2,-1,ngrids)
    exc, vxc = ni.eval_xc_eff(xc_code, rhoR, deriv=1, xctype=xctype)[:2]
    wv_freq = tools.fft(vxc.reshape(-1,ngrids), mesh).reshape(2,-1,ngrids)
    if xctype == 'GGA' and GGA_METHOD.upper() == 'FFT':
        #:wv_freq = (wv_freq[:,0] - 1j * np.einsum('px,nxp->np', Gv, wv_freq[:,1:4])) * weight
        Gv = cell.get_Gv(ni.mesh)
        backend.get_gga_vrho_gs(wv_freq[0,:1], wv_freq[0,1:4], Gv, weight, ngrids, 1.)
        backend.get_gga_vrho_gs(wv_freq[1,:1], wv_freq[1,1:4], Gv, weight, ngrids, 1.)
        wv_freq = wv_freq[:,:1]
    else:
        wv_freq *= weight

    nelec = rhoR[:,0].sum(axis=1) * weight
    excsum = (rhoR[0,0] + rhoR[1,0]).dot(exc) * weight
    exc = vxc = None

    rhoR = rhoG = None
    log.debug('Multigrid exc %s  nelec %s', excsum, nelec)

    if with_j:
        wv_freq[:,0] += vG

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    if xctype in (None, 'LDA', 'HF'):
        veff = _get_j_pass2(mydf, wv_freq, kpts_band, verbose=log)
    elif xctype == 'GGA':
        if GGA_METHOD.upper() == 'FFT':
            veff = _get_j_pass2(mydf, wv_freq, kpts_band, verbose=log)
        else:
            veff = _get_gga_pass2(mydf, wv_freq, kpts_band, hermi=hermi, verbose=log)
    else:
        raise NotImplementedError

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
    if isinstance(kpts, KPoints):
        raise NotImplementedError

    cell = mydf.cell
    dm_kpts = np.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    kpts_band = _format_kpts_band(kpts_band, kpts)
    if spin == 1:
        nset //= 2

    mesh = mydf.mesh
    ngrids = np.prod(mesh)
    ni = mydf
    xctype = ni._xc_type(xc_code)
    if xctype in (None, 'LDA', 'HF'):
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    else:
        raise NotImplementedError
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
            tmp = rhoG[i,0,0] + rhoG[i,1,0]
            vG[i] = np.multiply(tmp, coulG, out=vG[i])

    if mydf.vpplocG_part1 is not None:
        for i in range(nset):
            vG[i] += mydf.vpplocG_part1

    weight = cell.vol / ngrids

    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    if spin == 0:
        rhoR = rhoR.reshape(nset,-1,ngrids)
    elif spin == 1:
        rhoR = rhoR.reshape(nset,2,-1,ngrids)

    wv_freq = []
    Gv = cell.get_Gv(mesh)
    for i in range(nset):
        vxc = ni.eval_xc_eff(xc_code, rhoR[i], deriv=1, xctype=xctype, spin=spin)[1]
        wv = weight * vxc
        wv = tools.fft(wv.reshape(-1,ngrids), mesh)
        if spin == 0:
            wv = wv.reshape(-1,ngrids)
            if xctype == 'GGA' and GGA_METHOD.upper() == 'FFT':
                wv[0] -= np.einsum('xg,gx->g', wv[1:4], 1j*Gv)
                wv = wv[:1]
        elif spin == 1:
            wv = wv.reshape(2,-1,ngrids)
            if xctype == 'GGA' and GGA_METHOD.upper() == 'FFT':
                wv[:,0] -= np.einsum('nxg,gx->ng', wv[:,1:4], 1j*Gv)
                wv = wv[:,:1]
        wv_freq.append(wv)

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
            wv_freq[i,0] += vG[i].reshape(*mesh)
        elif spin == 1:
            for s in range(2):
                wv_freq[i,s,0] += vG[i].reshape(*mesh)

    if xctype in (None, 'LDA', 'HF'):
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

def get_nuc(mydf, kpts=None, deriv=0):
    if kpts is None:
        kpts = np.zeros((1, 3))
    elif isinstance(kpts, KPoints):
        kpts = kpts.kpts_ibz
    kpts, is_single_kpt = fft._check_kpts(mydf, kpts)
    cell = mydf.cell
    mesh = mydf.mesh
    charge = -cell.atom_charges()

    Gv, Gvbase, _ = cell.get_Gv_weights(mesh)
    basex, basey, basez = Gvbase
    b = cell.reciprocal_vectors()
    rb = np.dot(cell.atom_coords(), b.T)
    SIx = np.exp(-1j*np.einsum('z,g->zg', rb[:,0], basex))
    SIy = np.exp(-1j*np.einsum('z,g->zg', rb[:,1], basey))
    SIz = np.exp(-1j*np.einsum('z,g->zg', rb[:,2], basez))
    rhoG = np.einsum('i,ix,iy,iz->xyz', charge, SIx, SIy, SIz).ravel()

    coulG = tools.get_coulG(cell, mesh=mesh, Gv=Gv)
    vneG = rhoG * coulG
    if deriv == 0:
        vne = _get_j_pass2(mydf, vneG, kpts=kpts, hermi=1)
    elif deriv == 1: # ip1
        vne = _get_j_pass2_ip1(mydf, vneG, kpts=kpts, hermi=0, deriv=deriv)
    else:
        raise NotImplementedError

    if is_single_kpt:
        vne = vne[0]
    return np.asarray(vne)

def get_nuc_ip1(mydf, kpts=None):
    return get_nuc(mydf, kpts=kpts, deriv=1)

def get_nuc_nuc_grad(mydf, dm, kpts=None):
    # < p | d/dR (-Z / |r-R|) | q > D_{pq}
    if kpts is None:
        kpts = np.zeros((1, 3))
    elif isinstance(kpts, KPoints):
        if kpts.kpts.size > 3: # multiple k points
            dm = kpts.transform_dm(dm)
        kpts = kpts.kpts
    kpts, is_single_kpt = fft._check_kpts(mydf, kpts)
    cell = mydf.cell
    mesh = mydf.mesh
    dms = _format_dms(dm, kpts)

    Gv = cell.get_Gv(mesh)
    coulG = tools.get_coulG(cell, mesh=mesh, Gv=Gv)
    coords = cell.atom_coords()
    charge = -cell.atom_charges()
    grad = np.zeros((cell.natm,3))
    # TODO improve performance
    rhoG = _eval_rhoG(mydf, dms, hermi=1, kpts=kpts).ravel()
    for ia in range(cell.natm):
        vG = 1j * np.exp(1j * np.dot(Gv, coords[ia])) * charge[ia] * coulG * Gv.T
        grad[ia] = np.dot(vG, rhoG).real
    grad *= 1 / cell.vol
    return grad

class MultiGridNumInt(MultiGridNumInt_v1):
    '''Base class for multigrid DFT (version 2).

    Intermediate Attributes (These attributes are generated during computations
    and should not be modified. Furthermore, they may not be compatible between
    GPU and CPU implementations):
        task_list : `TaskList` instance
            Task list recording which primitive basis function pairs
            need to be considered.
        vpplocG_part1 : ndarray
            Long-range part of the local pseudopotential represented
            in the reciprocal space. It is cached to reduce cost.
        rhoG : ndarray
            Electronic density represented in the reciprocal space.
            It is cached in nuclear gradient calculations to reduce cost.
    '''
    ntasks = NTASKS
    ke_ratio = KE_RATIO
    rel_cutoff = REL_CUTOFF
    _keys = {'ntasks', 'ke_ratio', 'rel_cutoff',
             'task_list', 'vpplocG_part1', 'rhoG'}

    def __init__(self, cell):
        MultiGridNumInt_v1.__init__(self, cell)
        self.task_list = None
        self.vpplocG_part1 = None
        self.rhoG = None

    def reset(self, cell=None):
        MultiGridNumInt_v1.reset(self, cell)
        self.task_list = None
        self.vpplocG_part1 = None
        self.rhoG = None
        return self

    @property
    def ngrids(self):
        return np.prod(self.mesh)

    @ngrids.setter
    def ngrids(self, x):
        raise AttributeError('The MultiGridNumInt attribute .ngrids is deprecated. '
                             'It is replaced by the attribute .ntasks.')

    def get_veff_ip1(self, dm, xc_code=None, kpts=None, kpts_band=None, spin=0):
        if kpts is None:
            kpts = np.zeros(1,3)
        kpts = kpts.reshape(-1,3)
        if not gamma_point(kpts):
            raise NotImplementedError('MultiGridNumInt2 only supports Gamma-point calculations.')
        vj = get_veff_ip1(self, dm, xc_code=xc_code,
                          kpts=kpts, kpts_band=kpts_band, spin=spin)
        return vj

    def get_pp(self, kpts=None, return_full=False):
        '''Return the GTH pseudopotential (PP) matrix in AO basis,
        with contribution from G=0 removed.

        By default, the returned PP includes
        the short-range part of the local potential and the non-local potential.
        The long-range part of the local potential is cached as `vpplocG_part1`,
        which is the reciprocal space representation, to be added to the electron
        density Coulomb potential for computing the Coulomb matrix.
        In order to get the full PP matrix, set return_full to True.

        Kwargs:
            return_full: bool
                If True, the returned PP also contains the long-range part.
                Default is False.
        '''
        self.vpplocG_part1 = _get_vpplocG_part1(self, with_rho_core=True)

        vpp = _get_pp_without_erf(self, kpts)
        if return_full:
            if kpts is None:
                kpts = np.zeros((1,3))
            kpts, is_single_kpt = fft._check_kpts(self, kpts)
            vpp1 = _get_j_pass2(self, self.vpplocG_part1, kpts)
            if is_single_kpt:
                vpp1 = vpp1[0]
            vpp += vpp1

        return vpp

    get_nuc = get_nuc
    get_nuc_ip1 = get_nuc_ip1
    get_nuc_nuc_grad = get_nuc_nuc_grad
    vpploc_part1_nuc_grad = vpploc_part1_nuc_grad
    get_vpploc_part1_ip1 = get_vpploc_part1_ip1

    def nr_rks(self, cell, grids, xc_code, dm_kpts, relativity=0, hermi=1,
               kpts=None, kpts_band=None, max_memory=2000, verbose=None):
        if kpts is not None and not gamma_point(kpts):
            raise NotImplementedError('MultiGridNumInt2 only supports Gamma-point calculations.')
        return_j = False
        return nr_rks(self, xc_code, dm_kpts, hermi, kpts, kpts_band, self.xc_with_j,
                      return_j, verbose)

    def nr_uks(self, cell, grids, xc_code, dm_kpts, relativity=0, hermi=1,
               kpts=None, kpts_band=None, max_memory=2000, verbose=None):
        if kpts is not None and not gamma_point(kpts):
            raise NotImplementedError('MultiGridNumInt2 only supports Gamma-point calculations.')
        return_j = False
        return nr_uks(self, xc_code, dm_kpts, hermi, kpts, kpts_band, self.xc_with_j,
                      return_j, verbose)

    nr_rks_fxc = NotImplemented
    nr_uks_fxc = NotImplemented
    nr_rks_fxc_st = NotImplemented
    cache_xc_kernel = NotImplemented
    cache_xc_kernel1 = NotImplemented
    get_rho = NotImplemented
