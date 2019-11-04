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

'''Multigrid to compute DFT integrals'''

import time
import ctypes
import copy
import numpy
import scipy.linalg
from functools import reduce

from pyscf import lib
from pyscf.lib import logger
from pyscf.gto import ATOM_OF, ANG_OF, NPRIM_OF, PTR_EXP, PTR_COEFF
from pyscf.dft.numint import libdft
from pyscf.pbc import tools
from pyscf.pbc import gto
from pyscf.pbc.gto import pseudo
from pyscf.pbc.dft import numint, gen_grid
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from pyscf.pbc.df import fft
from pyscf.pbc.df import ft_ao
from pyscf import __config__

#sys.stderr.write('WARN: multigrid is an experimental feature. It is still in '
#                 'testing\nFeatures and APIs may be changed in the future.\n')

BLKSIZE = numint.BLKSIZE
EXTRA_PREC = getattr(__config__, 'pbc_gto_eval_gto_extra_precision', 1e-2)
TO_EVEN_GRIDS = getattr(__config__, 'pbc_dft_multigrid_to_even', False)
RMAX_FACTOR_ORTH = getattr(__config__, 'pbc_dft_multigrid_rmax_factor_orth', 1.1)
RMAX_FACTOR_NONORTH = getattr(__config__, 'pbc_dft_multigrid_rmax_factor_nonorth', 0.5)
RMAX_RATIO = getattr(__config__, 'pbc_dft_multigrid_rmax_ratio', 0.7)
R_RATIO_SUBLOOP = getattr(__config__, 'pbc_dft_multigrid_r_ratio_subloop', 0.6)
INIT_MESH_ORTH = getattr(__config__, 'pbc_dft_multigrid_init_mesh_orth', (12,12,12))
INIT_MESH_NONORTH = getattr(__config__, 'pbc_dft_multigrid_init_mesh_nonorth', (32,32,32))
KE_RATIO = getattr(__config__, 'pbc_dft_multigrid_ke_ratio', 1.3)
TASKS_TYPE = getattr(__config__, 'pbc_dft_multigrid_tasks_type', 'ke_cut') # 'rcut'

# RHOG_HIGH_ORDER=True will compute the high order derivatives of electron
# density in real space and FT to reciprocal space.  Set RHOG_HIGH_ORDER=False
# to approximate the density derivatives in reciprocal space (without
# evaluating the high order derivatives in real space).
RHOG_HIGH_ORDER = getattr(__config__, 'pbc_dft_multigrid_rhog_high_order', False)

PTR_EXPDROP = 16
EXPDROP = getattr(__config__, 'pbc_dft_multigrid_expdrop', 1e-12)
IMAG_TOL = 1e-9


def eval_mat(cell, weights, shls_slice=None, comp=1, hermi=0,
             xctype='LDA', kpts=None, mesh=None, offset=None, submesh=None):
    assert(all(cell._bas[:,NPRIM_OF] == 1))
    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    env[PTR_EXPDROP] = min(cell.precision*EXTRA_PREC, EXPDROP)
    ao_loc = gto.moleintor.make_loc(bas, 'cart')
    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas)
    i0, i1, j0, j1 = shls_slice
    j0 += cell.nbas
    j1 += cell.nbas
    naoi = ao_loc[i1] - ao_loc[i0]
    naoj = ao_loc[j1] - ao_loc[j0]

    if cell.dimension > 0:
        Ls = numpy.asarray(cell.get_lattice_Ls(), order='C')
    else:
        Ls = numpy.zeros((1,3))
    nimgs = len(Ls)

    if mesh is None:
        mesh = cell.mesh
    weights = numpy.asarray(weights, order='C')
    assert(weights.dtype == numpy.double)
    xctype = xctype.upper()
    n_mat = None
    if xctype == 'LDA':
        if weights.ndim == 1:
            weights = weights.reshape(-1, numpy.prod(mesh))
        else:
            n_mat = weights.shape[0]
    elif xctype == 'GGA':
        if hermi == 1:
            raise RuntimeError('hermi=1 is not supported for GGA functional')
        if weights.ndim == 2:
            weights = weights.reshape(-1, 4, numpy.prod(mesh))
        else:
            n_mat = weights.shape[0]
    else:
        raise NotImplementedError

    a = cell.lattice_vectors()
    b = numpy.linalg.inv(a.T)
    if offset is None:
        offset = (0, 0, 0)
    if submesh is None:
        submesh = mesh
    # log_prec is used to estimate the gto_rcut. Add EXTRA_PREC to count
    # other possible factors and coefficients in the integral.
    log_prec = numpy.log(cell.precision * EXTRA_PREC)

    if abs(a-numpy.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
    else:
        lattice_type = '_nonorth'
    eval_fn = 'NUMINTeval_' + xctype.lower() + lattice_type
    drv = libdft.NUMINT_fill2c
    def make_mat(weights):
        if comp == 1:
            mat = numpy.zeros((nimgs,naoj,naoi))
        else:
            mat = numpy.zeros((nimgs,comp,naoj,naoi))
        drv(getattr(libdft, eval_fn),
            weights.ctypes.data_as(ctypes.c_void_p),
            mat.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(comp), ctypes.c_int(hermi),
            (ctypes.c_int*4)(i0, i1, j0, j1),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(log_prec),
            ctypes.c_int(cell.dimension),
            ctypes.c_int(nimgs),
            Ls.ctypes.data_as(ctypes.c_void_p),
            a.ctypes.data_as(ctypes.c_void_p),
            b.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*3)(*offset), (ctypes.c_int*3)(*submesh),
            (ctypes.c_int*3)(*mesh),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(env)))
        return mat

    out = []
    for wv in weights:
        if cell.dimension == 0:
            mat = numpy.rollaxis(make_mat(wv)[0], -1, -2)
        elif kpts is None or gamma_point(kpts):
            mat = numpy.rollaxis(make_mat(wv).sum(axis=0), -1, -2)
            if getattr(kpts, 'ndim', None) == 2:
                mat = mat.reshape((1,)+mat.shape)
        else:
            mat = make_mat(wv)
            mat_shape = mat.shape
            expkL = numpy.exp(1j*kpts.reshape(-1,3).dot(Ls.T))
            mat = numpy.dot(expkL, mat.reshape(nimgs,-1))
            mat = numpy.rollaxis(mat.reshape((-1,)+mat_shape[1:]), -1, -2)
        out.append(mat)

    if n_mat is None:
        out = out[0]
    return out

def eval_rho(cell, dm, shls_slice=None, hermi=0, xctype='LDA', kpts=None,
             mesh=None, offset=None, submesh=None, ignore_imag=False,
             out=None):
    '''Collocate the *real* density (opt. gradients) on the real-space grid.
    '''
    assert(all(cell._bas[:,NPRIM_OF] == 1))
    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    env[PTR_EXPDROP] = min(cell.precision*EXTRA_PREC, EXPDROP)
    ao_loc = gto.moleintor.make_loc(bas, 'cart')
    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas)
    i0, i1, j0, j1 = shls_slice
    if hermi:
        assert(i0 == j0 and i1 == j1)
    j0 += cell.nbas
    j1 += cell.nbas
    naoi = ao_loc[i1] - ao_loc[i0]
    naoj = ao_loc[j1] - ao_loc[j0]
    dm = numpy.asarray(dm, order='C')
    assert(dm.shape[-2:] == (naoi, naoj))

    if cell.dimension > 0:
        Ls = numpy.asarray(cell.get_lattice_Ls(), order='C')
    else:
        Ls = numpy.zeros((1,3))

    if cell.dimension == 0 or kpts is None or gamma_point(kpts):
        nkpts, nimgs = 1, Ls.shape[0]
        dm = dm.reshape(-1,1,naoi,naoj).transpose(0,1,3,2)
    else:
        expkL = numpy.exp(1j*kpts.reshape(-1,3).dot(Ls.T))
        nkpts, nimgs = expkL.shape
        dm = dm.reshape(-1,nkpts,naoi,naoj).transpose(0,1,3,2)
    n_dm = dm.shape[0]

    a = cell.lattice_vectors()
    b = numpy.linalg.inv(a.T)
    if mesh is None:
        mesh = cell.mesh
    if offset is None:
        offset = (0, 0, 0)
    if submesh is None:
        submesh = mesh
    log_prec = numpy.log(cell.precision * EXTRA_PREC)

    if abs(a-numpy.diag(a.diagonal())).max() < 1e-12:
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
    if comp == 1:
        shape = (numpy.prod(submesh),)
    else:
        shape = (comp, numpy.prod(submesh))
    eval_fn = 'NUMINTrho_' + xctype.lower() + lattice_type
    drv = libdft.NUMINT_rho_drv
    def make_rho_(rho, dm):
        drv(getattr(libdft, eval_fn),
            rho.ctypes.data_as(ctypes.c_void_p),
            dm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(comp), ctypes.c_int(hermi),
            (ctypes.c_int*4)(i0, i1, j0, j1),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(log_prec),
            ctypes.c_int(cell.dimension),
            ctypes.c_int(nimgs),
            Ls.ctypes.data_as(ctypes.c_void_p),
            a.ctypes.data_as(ctypes.c_void_p),
            b.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*3)(*offset), (ctypes.c_int*3)(*submesh),
            (ctypes.c_int*3)(*mesh),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(env)))
        return rho

    rho = []
    for i, dm_i in enumerate(dm):
        if out is None:
            rho_i = numpy.zeros(shape)
        else:
            rho_i = out[i]
            assert(rho_i.size == numpy.prod(shape))

        if cell.dimension == 0:
            # make a copy because the dm may be overwritten in the
            # NUMINT_rho_drv inplace
            make_rho_(rho_i, numpy.array(dm_i, order='C', copy=True))
        elif kpts is None or gamma_point(kpts):
            make_rho_(rho_i, numpy.repeat(dm_i, nimgs, axis=0))
        else:
            dm_i = lib.dot(expkL.T, dm_i.reshape(nkpts,-1)).reshape(nimgs,naoj,naoi)
            dmR = numpy.asarray(dm_i.real, order='C')

            if ignore_imag:
                has_imag = False
            else:
                dmI = numpy.asarray(dm_i.imag, order='C')
                has_imag = (hermi == 0 and abs(dmI).max() > 1e-8)
                if (has_imag and xctype == 'LDA' and
                    naoi == naoj and
# For hermitian density matrices, the anti-symmetry character of the imaginary
# part of the density matrices can be found by rearranging the repeated images.
                    abs(dmI + dmI[::-1].transpose(0,2,1)).max() < 1e-8):
                    has_imag = False
            dm_i = None

            if has_imag:
                if out is None:
                    rho_i  = make_rho_(rho_i, dmI)*1j
                    rho_i += make_rho_(numpy.zeros(shape), dmR)
                else:
                    out[i]  = make_rho_(numpy.zeros(shape), dmI)*1j
                    out[i] += make_rho_(numpy.zeros(shape), dmR)
            else:
                assert(rho_i.dtype == numpy.double)
                make_rho_(rho_i, dmR)
            dmR = dmI = None

        rho.append(rho_i)

    if n_dm == 1:
        rho = rho[0]
    return rho

def get_nuc(mydf, kpts=None):
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    mesh = mydf.mesh
    charge = -cell.atom_charges()
    Gv = cell.get_Gv(mesh)
    SI = cell.get_SI(Gv)
    rhoG = numpy.dot(charge, SI)

    coulG = tools.get_coulG(cell, mesh=mesh, Gv=Gv)
    vneG = rhoG * coulG
    vne = _get_j_pass2(mydf, vneG, kpts_lst)[0]

    if kpts is None or numpy.shape(kpts) == (3,):
        vne = vne[0]
    return numpy.asarray(vne)

def get_pp(mydf, kpts=None):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    from pyscf import gto
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    mesh = mydf.mesh
    SI = cell.get_SI()
    Gv = cell.get_Gv(mesh)
    vpplocG = pseudo.get_vlocG(cell, Gv)
    vpplocG = -numpy.einsum('ij,ij->j', SI, vpplocG)
    # from get_jvloc_G0 function
    vpplocG[0] = numpy.sum(pseudo.get_alphas(cell))
    ngrids = len(vpplocG)

    vpp = _get_j_pass2(mydf, vpplocG, kpts_lst)[0]

    # vppnonloc evaluated in reciprocal space
    fakemol = gto.Mole()
    fakemol._atm = numpy.zeros((1,gto.ATM_SLOTS), dtype=numpy.int32)
    fakemol._bas = numpy.zeros((1,gto.BAS_SLOTS), dtype=numpy.int32)
    ptr = gto.PTR_ENV_START
    fakemol._env = numpy.zeros(ptr+10)
    fakemol._bas[0,gto.NPRIM_OF ] = 1
    fakemol._bas[0,gto.NCTR_OF  ] = 1
    fakemol._bas[0,gto.PTR_EXP  ] = ptr+3
    fakemol._bas[0,gto.PTR_COEFF] = ptr+4

    # buf for SPG_lmi upto l=0..3 and nl=3
    buf = numpy.empty((48,ngrids), dtype=numpy.complex128)
    def vppnl_by_k(kpt):
        Gk = Gv + kpt
        G_rad = lib.norm(Gk, axis=1)
        aokG = ft_ao.ft_ao(cell, Gv, kpt=kpt) * (ngrids/cell.vol)
        vppnl = 0
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                continue
            pp = cell._pseudo[symb]
            p1 = 0
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                if nl > 0:
                    fakemol._bas[0,gto.ANG_OF] = l
                    fakemol._env[ptr+3] = .5*rl**2
                    fakemol._env[ptr+4] = rl**(l+1.5)*numpy.pi**1.25
                    pYlm_part = fakemol.eval_gto('GTOval', Gk)

                    p0, p1 = p1, p1+nl*(l*2+1)
                    # pYlm is real, SI[ia] is complex
                    pYlm = numpy.ndarray((nl,l*2+1,ngrids), dtype=numpy.complex128, buffer=buf[p0:p1])
                    for k in range(nl):
                        qkl = pseudo.pp._qli(G_rad*rl, l, k)
                        pYlm[k] = pYlm_part.T * qkl
                    #:SPG_lmi = numpy.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                    #:SPG_lm_aoG = numpy.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                    #:tmp = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                    #:vppnl += numpy.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
            if p1 > 0:
                SPG_lmi = buf[:p1]
                SPG_lmi *= SI[ia].conj()
                SPG_lm_aoGs = lib.zdot(SPG_lmi, aokG)
                p1 = 0
                for l, proj in enumerate(pp[5:]):
                    rl, nl, hl = proj
                    if nl > 0:
                        p0, p1 = p1, p1+nl*(l*2+1)
                        hl = numpy.asarray(hl)
                        SPG_lm_aoG = SPG_lm_aoGs[p0:p1].reshape(nl,l*2+1,-1)
                        tmp = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                        vppnl += numpy.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
        return vppnl * (1./ngrids**2)

    for k, kpt in enumerate(kpts_lst):
        vppnl = vppnl_by_k(kpt)
        if gamma_point(kpt):
            vpp[k] = vpp[k].real + vppnl.real
        else:
            vpp[k] += vppnl

    if kpts is None or numpy.shape(kpts) == (3,):
        vpp = vpp[0]
    return numpy.asarray(vpp)


def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpts_band=None):
    '''Get the Coulomb (J) AO matrix at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.  If a list of k-point DMs, eg,
            UHF alpha and beta DM, the alpha and beta DMs are contracted
            separately.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        or list of vj if the input dm_kpts is a list of DMs
    '''
    cell = mydf.cell
    dm_kpts = numpy.asarray(dm_kpts)
    rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv=0)
    coulG = tools.get_coulG(cell, mesh=cell.mesh)
    #:vG = numpy.einsum('ng,g->ng', rhoG[:,0], coulG)
    vG = rhoG[:,0]
    vG *= coulG

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    vj_kpts = _get_j_pass2(mydf, vG, kpts_band)
    return _format_jks(vj_kpts, dm_kpts, input_band, kpts)


def _eval_rhoG(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), deriv=0,
               rhog_high_order=RHOG_HIGH_ORDER):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    cell = mydf.cell

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    tasks = getattr(mydf, 'tasks', None)
    if tasks is None:
        mydf.tasks = tasks = multi_grids_tasks(cell, mydf.mesh, log)
        log.debug('Multigrid ntasks %s', len(tasks))

    assert(deriv < 2)
    #hermi = hermi and abs(dms - dms.transpose(0,1,3,2).conj()).max() < 1e-9
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

    ni = mydf._numint
    nx, ny, nz = mydf.mesh
    rhoG = numpy.zeros((nset*rhodim,nx,ny,nz), dtype=numpy.complex128)
    for grids_dense, grids_sparse in tasks:
        h_cell = grids_dense.cell
        mesh = tuple(grids_dense.mesh)
        ngrids = numpy.prod(mesh)
        log.debug('mesh %s  rcut %g', mesh, h_cell.rcut)

        if grids_sparse is None:
            # The first pass handles all diffused functions using the regular
            # matrix multiplication code.
            rho = numpy.zeros((nset,rhodim,ngrids), dtype=numpy.complex128)
            idx_h = grids_dense.ao_idx
            dms_hh = numpy.asarray(dms[:,:,idx_h[:,None],idx_h], order='C')
            for ao_h_etc, p0, p1 in mydf.aoR_loop(grids_dense, kpts, deriv):
                ao_h, mask = ao_h_etc[0], ao_h_etc[2]
                for k in range(nkpts):
                    for i in range(nset):
                        if xctype == 'LDA':
                            ao_dm = lib.dot(ao_h[k], dms_hh[i,k])
                            rho_sub = numpy.einsum('xi,xi->x', ao_dm, ao_h[k].conj())
                        else:
                            rho_sub = numint.eval_rho(h_cell, ao_h[k], dms_hh[i,k],
                                                      mask, xctype, hermi)
                        rho[i,:,p0:p1] += rho_sub
                ao_h = ao_h_etc = ao_dm = None
            if ignore_imag:
                rho = rho.real
        else:
            idx_h = grids_dense.ao_idx
            idx_l = grids_sparse.ao_idx
            idx_t = numpy.append(idx_h, idx_l)
            dms_ht = numpy.asarray(dms[:,:,idx_h[:,None],idx_t], order='C')
            dms_lh = numpy.asarray(dms[:,:,idx_l[:,None],idx_h], order='C')

            t_cell = h_cell + grids_sparse.cell
            nshells_h = _pgto_shells(h_cell)
            nshells_t = _pgto_shells(t_cell)
            t_cell, t_coeff = t_cell.to_uncontracted_cartesian_basis()

            if deriv == 0:
                h_coeff = scipy.linalg.block_diag(*t_coeff[:h_cell.nbas])
                l_coeff = scipy.linalg.block_diag(*t_coeff[h_cell.nbas:])
                t_coeff = scipy.linalg.block_diag(*t_coeff)

                if hermi == 1:
                    naol, naoh = dms_lh.shape[2:]
                    dms_ht[:,:,:,naoh:] += dms_lh.transpose(0,1,3,2)
                    pgto_dms = lib.einsum('nkij,pi,qj->nkpq', dms_ht, h_coeff, t_coeff)
                    shls_slice = (0, nshells_h, 0, nshells_t)
                    #:rho = eval_rho(t_cell, pgto_dms, shls_slice, 0, 'LDA', kpts,
                    #:               offset=None, submesh=None, ignore_imag=True)
                    rho = _eval_rho_bra(t_cell, pgto_dms, shls_slice, 0,
                                        'LDA', kpts, grids_dense, True, log)

                else:
                    pgto_dms = lib.einsum('nkij,pi,qj->nkpq', dms_ht, h_coeff, t_coeff)
                    shls_slice = (0, nshells_h, 0, nshells_t)
                    #:rho = eval_rho(t_cell, pgto_dms, shls_slice, 0, 'LDA', kpts,
                    #:               offset=None, submesh=None)
                    rho = _eval_rho_bra(t_cell, pgto_dms, shls_slice, 0,
                                        'LDA', kpts, grids_dense, True, log)
                    pgto_dms = lib.einsum('nkij,pi,qj->nkpq', dms_lh, l_coeff, h_coeff)
                    shls_slice = (nshells_h, nshells_t, 0, nshells_h)
                    #:rho += eval_rho(t_cell, pgto_dms, shls_slice, 0, 'LDA', kpts,
                    #:                offset=None, submesh=None)
                    rho += _eval_rho_ket(t_cell, pgto_dms, shls_slice, 0,
                                         'LDA', kpts, grids_dense, True, log)

            elif deriv == 1:
                h_coeff = scipy.linalg.block_diag(*t_coeff[:h_cell.nbas])
                l_coeff = scipy.linalg.block_diag(*t_coeff[h_cell.nbas:])
                t_coeff = scipy.linalg.block_diag(*t_coeff)

                pgto_dms = lib.einsum('nkij,pi,qj->nkpq', dms_ht, h_coeff, t_coeff)
                shls_slice = (0, nshells_h, 0, nshells_t)
                #:rho = eval_rho(t_cell, pgto_dms, shls_slice, 0, 'GGA', kpts,
                #:               ignore_imag=ignore_imag)
                rho = _eval_rho_bra(t_cell, pgto_dms, shls_slice, 0, 'GGA',
                                    kpts, grids_dense, ignore_imag, log)

                pgto_dms = lib.einsum('nkij,pi,qj->nkpq', dms_lh, l_coeff, h_coeff)
                shls_slice = (nshells_h, nshells_t, 0, nshells_h)
                #:rho += eval_rho(t_cell, pgto_dms, shls_slice, 0, 'GGA', kpts,
                #:                ignore_imag=ignore_imag)
                rho += _eval_rho_ket(t_cell, pgto_dms, shls_slice, 0, 'GGA',
                                     kpts, grids_dense, ignore_imag, log)
                if hermi == 1:
                    # \nabla \chi_i DM(i,j) \chi_j was computed above.
                    # *2 for \chi_i DM(i,j) \nabla \chi_j
                    rho[:,1:4] *= 2
                else:
                    raise NotImplementedError

        weight = 1./nkpts * cell.vol/ngrids
        rho_freq = tools.fft(rho.reshape(nset*rhodim, -1), mesh)
        rho_freq *= weight
        gx = numpy.fft.fftfreq(mesh[0], 1./mesh[0]).astype(numpy.int32)
        gy = numpy.fft.fftfreq(mesh[1], 1./mesh[1]).astype(numpy.int32)
        gz = numpy.fft.fftfreq(mesh[2], 1./mesh[2]).astype(numpy.int32)
        #:rhoG[:,gx[:,None,None],gy[:,None],gz] += rho_freq.reshape((-1,)+mesh)
        _takebak_4d(rhoG, rho_freq.reshape((-1,) + mesh), (None, gx, gy, gz))

    rhoG = rhoG.reshape(nset,rhodim,-1)

    if gga_high_order:
        Gv = cell.get_Gv(mydf.mesh)
        rhoG1 = numpy.einsum('np,px->nxp', 1j*rhoG[:,0], Gv)
        rhoG = numpy.concatenate([rhoG, rhoG1], axis=1)
    return rhoG


def _eval_rho_bra(cell, dms, shls_slice, hermi, xctype, kpts, grids,
                  ignore_imag, log):
    a = cell.lattice_vectors()
    rmax = a.max()
    mesh = numpy.asarray(grids.mesh)
    rcut = grids.cell.rcut
    nset = dms.shape[0]
    if xctype == 'LDA':
        rhodim = 1
    else:
        rhodim = 4

    if rcut > rmax * R_RATIO_SUBLOOP:
        rho = eval_rho(cell, dms, shls_slice, hermi, xctype, kpts,
                       mesh, ignore_imag=ignore_imag)
        return numpy.reshape(rho, (nset, rhodim, numpy.prod(mesh)))

    if hermi == 1 or ignore_imag:
        rho = numpy.zeros((nset, rhodim) + tuple(mesh))
    else:
        rho = numpy.zeros((nset, rhodim) + tuple(mesh), dtype=numpy.complex128)

    b = numpy.linalg.inv(a.T)
    ish0, ish1, jsh0, jsh1 = shls_slice
    nshells_j = jsh1 - jsh0
    pcell = copy.copy(cell)
    rest_dms = []
    rest_bas = []
    i1 = 0
    for atm_id in set(cell._bas[ish0:ish1,ATOM_OF]):
        atm_bas_idx = numpy.where(cell._bas[ish0:ish1,ATOM_OF] == atm_id)[0]
        _bas_i = cell._bas[atm_bas_idx]
        l = _bas_i[:,ANG_OF]
        i0, i1 = i1, i1 + sum((l+1)*(l+2)//2)
        sub_dms = dms[:,:,i0:i1]

        atom_position = cell.atom_coord(atm_id)
        frac_edge0 = b.dot(atom_position - rcut)
        frac_edge1 = b.dot(atom_position + rcut)

        if (numpy.all(0 < frac_edge0) and numpy.all(frac_edge1 < 1)):
            pcell._bas = numpy.vstack((_bas_i, cell._bas[jsh0:jsh1]))
            nshells_i = len(atm_bas_idx)
            sub_slice = (0, nshells_i, nshells_i, nshells_i+nshells_j)

            offset = (frac_edge0 * mesh).astype(int)
            mesh1 = numpy.ceil(frac_edge1 * mesh).astype(int)
            submesh = mesh1 - offset
            log.debug1('atm %d  rcut %f  offset %s submesh %s',
                       atm_id, rcut, offset, submesh)
            rho1 = eval_rho(pcell, sub_dms, sub_slice, hermi, xctype, kpts,
                            mesh, offset, submesh, ignore_imag=ignore_imag)
            #:rho[:,:,offset[0]:mesh1[0],offset[1]:mesh1[1],offset[2]:mesh1[2]] += \
            #:        numpy.reshape(rho1, (nset, rhodim) + tuple(submesh))
            gx = numpy.arange(offset[0], mesh1[0], dtype=numpy.int32)
            gy = numpy.arange(offset[1], mesh1[1], dtype=numpy.int32)
            gz = numpy.arange(offset[2], mesh1[2], dtype=numpy.int32)
            _takebak_5d(rho, numpy.reshape(rho1, (nset,rhodim)+tuple(submesh)),
                        (None, None, gx, gy, gz))
        else:
            log.debug1('atm %d  rcut %f  over 2 images', atm_id, rcut)
            #:rho1 = eval_rho(pcell, sub_dms, sub_slice, hermi, xctype, kpts,
            #:                mesh, ignore_imag=ignore_imag)
            #:rho += numpy.reshape(rho1, rho.shape)
            # or
            #:eval_rho(pcell, sub_dms, sub_slice, hermi, xctype, kpts,
            #:         mesh, ignore_imag=ignore_imag, out=rho)
            rest_bas.append(_bas_i)
            rest_dms.append(sub_dms)
    if rest_bas:
        pcell._bas = numpy.vstack(rest_bas + [cell._bas[jsh0:jsh1]])
        nshells_i = sum(len(x) for x in rest_bas)
        sub_slice = (0, nshells_i, nshells_i, nshells_i+nshells_j)
        sub_dms = numpy.concatenate(rest_dms, axis=2)
        eval_rho(pcell, sub_dms, sub_slice, hermi, xctype, kpts,
                 mesh, ignore_imag=ignore_imag, out=rho)
    return rho.reshape((nset, rhodim, numpy.prod(mesh)))

def _eval_rho_ket(cell, dms, shls_slice, hermi, xctype, kpts, grids,
                  ignore_imag, log):
    a = cell.lattice_vectors()
    rmax = a.max()
    mesh = numpy.asarray(grids.mesh)
    rcut = grids.cell.rcut
    nset = dms.shape[0]
    if xctype == 'LDA':
        rhodim = 1
    else:
        rhodim = 4

    if rcut > rmax * R_RATIO_SUBLOOP:
        rho = eval_rho(cell, dms, shls_slice, hermi, xctype, kpts,
                       mesh, ignore_imag=ignore_imag)
        return numpy.reshape(rho, (nset, rhodim, numpy.prod(mesh)))

    if hermi == 1 or ignore_imag:
        rho = numpy.zeros((nset, rhodim) + tuple(mesh))
    else:
        rho = numpy.zeros((nset, rhodim) + tuple(mesh), dtype=numpy.complex128)

    b = numpy.linalg.inv(a.T)
    ish0, ish1, jsh0, jsh1 = shls_slice
    nshells_i = ish1 - ish0
    pcell = copy.copy(cell)
    rest_dms = []
    rest_bas = []
    j1 = 0
    for atm_id in set(cell._bas[jsh0:jsh1,ATOM_OF]):
        atm_bas_idx = numpy.where(cell._bas[jsh0:jsh1,ATOM_OF] == atm_id)[0]
        _bas_j = cell._bas[atm_bas_idx]
        l = _bas_j[:,ANG_OF]
        j0, j1 = j1, j1 + sum((l+1)*(l+2)//2)
        sub_dms = dms[:,:,:,j0:j1]

        atom_position = cell.atom_coord(atm_id)
        frac_edge0 = b.dot(atom_position - rcut)
        frac_edge1 = b.dot(atom_position + rcut)

        if (numpy.all(0 < frac_edge0) and numpy.all(frac_edge1 < 1)):
            pcell._bas = numpy.vstack((cell._bas[ish0:ish1], _bas_j))
            nshells_j = len(atm_bas_idx)
            sub_slice = (0, nshells_i, nshells_i, nshells_i+nshells_j)

            offset = (frac_edge0 * mesh).astype(int)
            mesh1 = numpy.ceil(frac_edge1 * mesh).astype(int)
            submesh = mesh1 - offset
            log.debug1('atm %d  rcut %f  offset %s submesh %s',
                       atm_id, rcut, offset, submesh)
            rho1 = eval_rho(pcell, sub_dms, sub_slice, hermi, xctype, kpts,
                            mesh, offset, submesh, ignore_imag=ignore_imag)
            #:rho[:,:,offset[0]:mesh1[0],offset[1]:mesh1[1],offset[2]:mesh1[2]] += \
            #:        numpy.reshape(rho1, (nset, rhodim) + tuple(submesh))
            gx = numpy.arange(offset[0], mesh1[0], dtype=numpy.int32)
            gy = numpy.arange(offset[1], mesh1[1], dtype=numpy.int32)
            gz = numpy.arange(offset[2], mesh1[2], dtype=numpy.int32)
            _takebak_5d(rho, numpy.reshape(rho1, (nset,rhodim)+tuple(submesh)),
                        (None, None, gx, gy, gz))
        else:
            log.debug1('atm %d  rcut %f  over 2 images', atm_id, rcut)
            #:rho1 = eval_rho(pcell, sub_dms, sub_slice, hermi, xctype, kpts,
            #:                mesh, ignore_imag=ignore_imag)
            #:rho += numpy.reshape(rho1, rho.shape)
            #:eval_rho(pcell, sub_dms, sub_slice, hermi, xctype, kpts,
            #:         mesh, ignore_imag=ignore_imag, out=rho)
            rest_bas.append(_bas_j)
            rest_dms.append(sub_dms)
    if rest_bas:
        pcell._bas = numpy.vstack([cell._bas[ish0:ish1]] + rest_bas)
        nshells_j = sum(len(x) for x in rest_bas)
        sub_slice = (0, nshells_i, nshells_i, nshells_i+nshells_j)
        sub_dms = numpy.concatenate(rest_dms, axis=3)
        eval_rho(pcell, sub_dms, sub_slice, hermi, xctype, kpts,
                 mesh, ignore_imag=ignore_imag, out=rho)
    return rho.reshape((nset, rhodim, numpy.prod(mesh)))


def _get_j_pass2(mydf, vG, kpts=numpy.zeros((1,3)), verbose=None):
    log = logger.new_logger(mydf, verbose)
    cell = mydf.cell
    nkpts = len(kpts)
    nao = cell.nao_nr()
    nx, ny, nz = mydf.mesh
    vG = vG.reshape(-1,nx,ny,nz)
    nset = vG.shape[0]

    tasks = getattr(mydf, 'tasks', None)
    if tasks is None:
        mydf.tasks = tasks = multi_grids_tasks(cell, mydf.mesh, log)
        log.debug('Multigrid ntasks %s', len(tasks))

    at_gamma_point = gamma_point(kpts)
    if at_gamma_point:
        vj_kpts = numpy.zeros((nset,nkpts,nao,nao))
    else:
        vj_kpts = numpy.zeros((nset,nkpts,nao,nao), dtype=numpy.complex128)

    ni = mydf._numint
    for grids_dense, grids_sparse in tasks:
        mesh = grids_dense.mesh
        ngrids = numpy.prod(mesh)
        log.debug('mesh %s', mesh)

        gx = numpy.fft.fftfreq(mesh[0], 1./mesh[0]).astype(numpy.int32)
        gy = numpy.fft.fftfreq(mesh[1], 1./mesh[1]).astype(numpy.int32)
        gz = numpy.fft.fftfreq(mesh[2], 1./mesh[2]).astype(numpy.int32)
        #:sub_vG = vG[:,gx[:,None,None],gy[:,None],gz].reshape(nset,ngrids)
        sub_vG = _take_4d(vG, (None, gx, gy, gz)).reshape(nset,ngrids)

        v_rs = tools.ifft(sub_vG, mesh).reshape(nset,ngrids)
        vR = numpy.asarray(v_rs.real, order='C')
        vI = numpy.asarray(v_rs.imag, order='C')
        if at_gamma_point:
            v_rs = vR

        idx_h = grids_dense.ao_idx
        if grids_sparse is None:
            for ao_h_etc, p0, p1 in mydf.aoR_loop(grids_dense, kpts):
                ao_h = ao_h_etc[0]
                for k in range(nkpts):
                    for i in range(nset):
                        vj_sub = lib.dot(ao_h[k].conj().T*v_rs[i,p0:p1], ao_h[k])
                        vj_kpts[i,k,idx_h[:,None],idx_h] += vj_sub
                ao_h = ao_h_etc = None
        else:
            idx_h = grids_dense.ao_idx
            idx_l = grids_sparse.ao_idx
            idx_t = numpy.append(idx_h, idx_l)
            naoh = len(idx_h)

            h_cell = grids_dense.cell
            l_cell = grids_sparse.cell
            t_cell = h_cell + l_cell
            t_cell, t_coeff = t_cell.to_uncontracted_cartesian_basis()
            nshells_h = _pgto_shells(h_cell)
            nshells_t = _pgto_shells(t_cell)

            h_coeff = scipy.linalg.block_diag(*t_coeff[:h_cell.nbas])
            l_coeff = scipy.linalg.block_diag(*t_coeff[h_cell.nbas:])
            t_coeff = scipy.linalg.block_diag(*t_coeff)
            shls_slice = (0, nshells_h, 0, nshells_t)
            vp = eval_mat(t_cell, vR, shls_slice, 1, 0, 'LDA', kpts)
            vp = lib.einsum('nkpq,pi,qj->nkij', vp, h_coeff, t_coeff)

            # Imaginary part may contribute
            if not at_gamma_point and abs(vI).max() > IMAG_TOL:
                vpI = eval_mat(t_cell, vI, shls_slice, 1, 0, 'LDA', kpts)
                vpI = lib.einsum('nkpq,pi,qj->nkij', vpI, h_coeff, t_coeff)
                vp = vp + vpI * 1j
                vpI = None

            vj_kpts[:,:,idx_h[:,None],idx_h] += vp[:,:,:,:naoh]
            vj_kpts[:,:,idx_h[:,None],idx_l] += vp[:,:,:,naoh:]

            #:shls_slice = (nshells_h, nshells_t, 0, nshells_h)
            #:vp = eval_mat(t_cell, vR, shls_slice, 1, 0, 'LDA', kpts)
            #:vp = lib.einsum('nkpq,pi,qj->nkij', vp, l_coeff, h_coeff)
            #:vj_kpts[:,:,idx_l[:,None],idx_h] += vp
            vj_kpts[:,:,idx_l[:,None],idx_h] += \
                    vp[:,:,:,naoh:].transpose(0,1,3,2).conj()

    return vj_kpts


def _get_gga_pass2(mydf, vG, kpts=numpy.zeros((1,3)), verbose=None):
    log = logger.new_logger(mydf, verbose)
    cell = mydf.cell
    nkpts = len(kpts)
    nao = cell.nao_nr()
    nx, ny, nz = mydf.mesh
    vG = vG.reshape(-1,4,nx,ny,nz)
    nset = vG.shape[0]

    if gamma_point(kpts):
        veff = numpy.zeros((nset,nkpts,nao,nao))
    else:
        veff = numpy.zeros((nset,nkpts,nao,nao), dtype=numpy.complex128)

    for grids_dense, grids_sparse in mydf.tasks:
        mesh = grids_dense.mesh
        ngrids = numpy.prod(mesh)
        log.debug('mesh %s', mesh)

        gx = numpy.fft.fftfreq(mesh[0], 1./mesh[0]).astype(numpy.int32)
        gy = numpy.fft.fftfreq(mesh[1], 1./mesh[1]).astype(numpy.int32)
        gz = numpy.fft.fftfreq(mesh[2], 1./mesh[2]).astype(numpy.int32)
        #:sub_vG = vG[:,:,gx[:,None,None],gy[:,None],gz].reshape(-1,ngrids)
        sub_vG = _take_5d(vG, (None, None, gx, gy, gz)).reshape(-1,ngrids)
        wv = tools.ifft(sub_vG, mesh).real.reshape(nset,4,ngrids)
        wv = numpy.asarray(wv, order='C')

        if grids_sparse is None:
            idx_h = grids_dense.ao_idx
            naoh = len(idx_h)
            for ao_h_etc, p0, p1 in mydf.aoR_loop(grids_dense, kpts, deriv=1):
                ao_h = ao_h_etc[0]
                for k in range(nkpts):
                    for i in range(nset):
                        aow = numint._scale_ao(ao_h[k], wv[i])
                        v = lib.dot(aow.conj().T, ao_h[k][0])
                        veff[i,k,idx_h[:,None],idx_h] += v + v.conj().T
                ao_h = ao_h_etc = None
        else:
            idx_h = grids_dense.ao_idx
            idx_l = grids_sparse.ao_idx
            idx_t = numpy.append(idx_h, idx_l)
            naoh = len(idx_h)

            h_cell = grids_dense.cell
            l_cell = grids_sparse.cell
            t_cell = h_cell + l_cell
            t_cell, t_coeff = t_cell.to_uncontracted_cartesian_basis()
            nshells_h = _pgto_shells(h_cell)
            nshells_t = _pgto_shells(t_cell)

            h_coeff = scipy.linalg.block_diag(*t_coeff[:h_cell.nbas])
            l_coeff = scipy.linalg.block_diag(*t_coeff[h_cell.nbas:])
            t_coeff = scipy.linalg.block_diag(*t_coeff)

            shls_slice = (0, nshells_h, 0, nshells_t)
            v = eval_mat(t_cell, wv, shls_slice, 1, 0, 'GGA', kpts)
            v = lib.einsum('nkpq,pi,qj->nkij', v, h_coeff, t_coeff)
            veff[:,:,idx_h[:,None],idx_h] += v[:,:,:,:naoh]
            veff[:,:,idx_h[:,None],idx_h] += v[:,:,:,:naoh].conj().transpose(0,1,3,2)
            veff[:,:,idx_h[:,None],idx_l] += v[:,:,:,naoh:]
            veff[:,:,idx_l[:,None],idx_h] += v[:,:,:,naoh:].conj().transpose(0,1,3,2)

            shls_slice = (nshells_h, nshells_t, 0, nshells_h)
            v = eval_mat(t_cell, wv, shls_slice, 1, 0, 'GGA', kpts)#, offset, submesh)
            v = lib.einsum('nkpq,pi,qj->nkij', v, l_coeff.conj(), h_coeff)
            veff[:,:,idx_l[:,None],idx_h] += v
            veff[:,:,idx_h[:,None],idx_l] += v.conj().transpose(0,1,3,2)

    return veff


def nr_rks(mydf, xc_code, dm_kpts, hermi=1, kpts=None,
           kpts_band=None, with_j=False, return_j=False, verbose=None):
    '''Compute the XC energy and RKS XC matrix at sampled k-points.
    multigrid version of function pbc.dft.numint.nr_rks.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        exc : XC energy
        nelec : number of electrons obtained from the numerical integration
        veff : (nkpts, nao, nao) ndarray
            or list of veff if the input dm_kpts is a list of DMs
        vj : (nkpts, nao, nao) ndarray
            or list of vj if the input dm_kpts is a list of DMs
    '''
    if kpts is None: kpts = mydf.kpts
    log = logger.new_logger(mydf, verbose)
    cell = mydf.cell
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    ni = mydf._numint
    xctype = ni._xc_type(xc_code)

    if xctype == 'LDA':
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv)

    mesh = mydf.mesh
    ngrids = numpy.prod(mesh)
    coulG = tools.get_coulG(cell, mesh=mesh)
    vG = numpy.einsum('ng,g->ng', rhoG[:,0], coulG)
    ecoul = .5 * numpy.einsum('ng,ng->n', rhoG[:,0].real, vG.real)
    ecoul+= .5 * numpy.einsum('ng,ng->n', rhoG[:,0].imag, vG.imag)
    ecoul /= cell.vol
    log.debug('Multigrid Coulomb energy %s', ecoul)

    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    rhoR = rhoR.reshape(nset,-1,ngrids)
    wv_freq = []
    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    for i in range(nset):
        exc, vxc = ni.eval_xc(xc_code, rhoR[i], 0, deriv=1)[:2]
        if xctype == 'LDA':
            wv = vxc[0].reshape(1,ngrids) * weight
        elif xctype == 'GGA':
            wv = numint._rks_gga_wv0(rhoR[i], vxc, weight)

        nelec[i] += rhoR[i,0].sum() * weight
        excsum[i] += (rhoR[i,0]*exc).sum() * weight
        wv_freq.append(tools.fft(wv, mesh))
    rhoR = rhoG = None
    wv_freq = numpy.asarray(wv_freq).reshape(nset,-1,*mesh)

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
        if with_j:  # *.5 because v+v.T.conj() is evaluated in _get_gga_pass2
            wv_freq[:,0] += vG.reshape(nset,*mesh) * .5
        veff = _get_gga_pass2(mydf, wv_freq, kpts_band, verbose=log)
    veff = _format_jks(veff, dm_kpts, input_band, kpts)

    if return_j:
        vj = _get_j_pass2(mydf, vG, kpts_band, verbose=log)
        vj = _format_jks(veff, dm_kpts, input_band, kpts)
    else:
        vj = None

    veff = lib.tag_array(veff, ecoul=ecoul, exc=excsum, vj=vj, vk=None)
    return nelec, excsum, veff


# Note nr_uks handles only one set of KUKS density matrices (alpha, beta) in
# each call (nr_rks supports multiple sets of KRKS density matrices)
def nr_uks(mydf, xc_code, dm_kpts, hermi=1, kpts=None,
           kpts_band=None, with_j=False, return_j=False, verbose=None):
    '''Compute the XC energy and UKS XC matrix at sampled k-points.
    multigrid version of function pbc.dft.numint.nr_uks

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        exc : XC energy
        nelec : number of electrons obtained from the numerical integration
        veff : (2, nkpts, nao, nao) ndarray
            or list of veff if the input dm_kpts is a list of DMs
        vj : (nkpts, nao, nao) ndarray
            or list of vj if the input dm_kpts is a list of DMs
    '''
    if kpts is None:
        kpts = mydf.kpts
    log = logger.new_logger(mydf, verbose)
    cell = mydf.cell
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    assert(nset == 2)
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    ni = mydf._numint
    xctype = ni._xc_type(xc_code)

    if xctype == 'LDA':
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv)

    mesh = mydf.mesh
    ngrids = numpy.prod(mesh)
    coulG = tools.get_coulG(cell, mesh=mesh)
    vG = numpy.einsum('ng,g->g', rhoG[:,0], coulG)
    ecoul = .5 * numpy.einsum('ng,g->', rhoG[:,0].real, vG.real)
    ecoul+= .5 * numpy.einsum('ng,g->', rhoG[:,0].imag, vG.imag)
    ecoul /= cell.vol
    log.debug('Multigrid Coulomb energy %s', ecoul)

    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    rhoR = rhoR.reshape(2,-1,ngrids)
    wv_freq = []
    nelec = numpy.zeros((2))
    excsum = 0

    exc, vxc = ni.eval_xc(xc_code, rhoR, 1, deriv=1)[:2]
    if xctype == 'LDA':
        vrho = vxc[0]
        wva = vrho[:,0].reshape(1,ngrids) * weight
        wvb = vrho[:,1].reshape(1,ngrids) * weight
    elif xctype == 'GGA':
        wva, wvb = numint._uks_gga_wv0(rhoR, vxc, weight)

    nelec[0] += rhoR[0,0].sum() * weight
    nelec[1] += rhoR[1,0].sum() * weight
    excsum += (rhoR[0,0]*exc).sum() * weight
    excsum += (rhoR[1,0]*exc).sum() * weight
    wv_freq = tools.fft(numpy.vstack((wva,wvb)), mesh)
    wv_freq = wv_freq.reshape(2,-1,*mesh)
    rhoR = rhoG = None
    log.debug('Multigrid exc %g  nelec %s', excsum, nelec)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    if xctype == 'LDA':
        if with_j:
            wv_freq[:,0] += vG.reshape(*mesh)
        veff = _get_j_pass2(mydf, wv_freq, kpts_band, verbose=log)
    elif xctype == 'GGA':
        if with_j:  # *.5 because v+v.T.conj() is evaluated in _get_gga_pass2
            wv_freq[:,0] += vG.reshape(*mesh) * .5
        veff = _get_gga_pass2(mydf, wv_freq, kpts_band, verbose=log)
    veff = _format_jks(veff, dm_kpts, input_band, kpts)

    if return_j:
        vj = _get_j_pass2(mydf, vG, kpts_band, verbose=log)
        vj = _format_jks(veff, dm_kpts, input_band, kpts)
    else:
        vj = None

    veff = lib.tag_array(veff, ecoul=ecoul, exc=excsum, vj=vj, vk=None)
    return nelec, excsum, veff


def nr_rks_fxc(mydf, xc_code, dm0, dms, hermi=1, with_j=False,
               rho0=None, vxc=None, fxc=None, kpts=None, verbose=None):
    '''multigrid version of function pbc.dft.numint.nr_rks_fxc
    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))
    log = logger.new_logger(mydf, verbose)
    cell = mydf.cell
    mesh = mydf.mesh
    ngrids = numpy.prod(mesh)

    dm_kpts = lib.asarray(dms, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    ni = mydf._numint
    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    else:
        deriv = 2

    weight = cell.vol / ngrids
    if rho0 is None:
        rhoG = _eval_rhoG(mydf, dm0, hermi, kpts, deriv)
        rho0 = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)

    if vxc is None or fxc is None:
        vxc, fxc = ni.eval_xc(xc_code, rho0, spin=0, deriv=2)[1:3]

    rhoG = _eval_rhoG(mydf, dms, hermi, kpts, deriv)
    rho1 = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    rho1 = rho1.reshape(nset,-1,ngrids)
    if with_j:
        coulG = tools.get_coulG(cell, mesh=mesh)
        vG = rhoG[:,0] * coulG
        vG = vG.reshape(nset, *mesh)

    if xctype == 'LDA':
        frr = fxc[0]
        wv = weight * frr * rho1
        wv = tools.fft(wv.reshape(-1,ngrids), mesh).reshape(nset,-1,*mesh)
        if with_j:
            wv[:,0] += vG
        veff = _get_j_pass2(mydf, wv, kpts, verbose=log)

    elif xctype == 'GGA':
        wv = [numint._rks_gga_wv1(rho0, rho1[i], vxc, fxc, weight)
              for i in range(nset)]
        wv = numpy.vstack(wv).reshape(-1,ngrids)
        wv = tools.fft(wv, mesh).reshape(nset,-1,*mesh)
        if with_j:
            wv[:,0] += vG * .5
        veff = _get_gga_pass2(mydf, wv, kpts, verbose=log)

    return veff.reshape(dm_kpts.shape)


def nr_rks_fxc_st(mydf, xc_code, dm0, dms_alpha, hermi=1, singlet=True, with_j=False,
                  rho0=None, vxc=None, fxc=None, kpts=None, verbose=None):
    '''multigrid version of function pbc.dft.numint.nr_rks_fxc_st
    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))
    log = logger.new_logger(mydf, verbose)
    cell = mydf.cell
    mesh = mydf.mesh
    ngrids = numpy.prod(mesh)

    dm_kpts = lib.asarray(dms_alpha, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    ni = mydf._numint
    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    else:
        deriv = 2

    weight = cell.vol / ngrids
    if rho0 is None:
        rhoG = _eval_rhoG(mydf, dm0, hermi, kpts, deriv)
        # *.5 to get alpha density
        rho0 = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (.5/weight)
        rho0 = (rho0, rho0)

    if vxc is None or fxc is None:
        vxc, fxc = ni.eval_xc(xc_code, rho0, spin=1, deriv=2)[1:3]

    rhoG = _eval_rhoG(mydf, dms, hermi, kpts, deriv)
    rho1 = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    rho1 = rho1.reshape(nset,-1,ngrids)
    if with_j:
        coulG = tools.get_coulG(cell, mesh=mesh)
        vG = rhoG[:,0] * coulG
        vG = vG.reshape(nset, *mesh)

    if xctype == 'LDA':
        u_u, u_d, d_d = fxc[0].T
        if singlet:
            frho = u_u + u_d
        else:
            frho = u_u - u_d
        wv = weight * frho * rho1
        wv = tools.fft(wv.reshape(-1,ngrids), mesh).reshape(nset,-1,*mesh)
        if with_j:
            wv[:,0] += vG
        veff = _get_j_pass2(mydf, wv, kpts, verbose=log)

    elif xctype == 'GGA':
        vsigma = vxc[1].T
        u_u, u_d, d_d = fxc[0].T  # v2rho2
        u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc[1].T  # v2rhosigma
        uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc[2].T  # v2sigma2
        if singlet:
            fgamma = vsigma[0] + vsigma[1] * .5
            frho = u_u + u_d
            fgg = uu_uu + .5*ud_ud + 2*uu_ud + uu_dd
            frhogamma = u_uu + u_dd + u_ud
        else:
            fgamma = vsigma[0] - vsigma[1] * .5
            frho = u_u - u_d
            fgg = uu_uu - uu_dd
            frhogamma = u_uu - u_dd

        wv = [numint._rks_gga_wv1(rho0[0], rho1[i], (None,fgamma),
                                  (frho,frhogamma,fgg), weight)
              for i in range(nset)]
        wv = numpy.asarray(wv).reshape(-1,ngrids)
        wv = tools.fft(wv, mesh).reshape(nset,-1,*mesh)
        if with_j:
            wv[:,0] += vG * .5
        veff = _get_gga_pass2(mydf, wv, kpts, verbose=log)

    return veff.reshape(dm_kpts.shape)


def nr_uks_fxc(mydf, xc_code, dm0, dms, hermi=1, with_j=False,
               rho0=None, vxc=None, fxc=None, kpts=None, verbose=None):
    '''multigrid version of function pbc.dft.numint.nr_uks_fxc
    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))
    log = logger.new_logger(mydf, verbose)
    cell = mydf.cell
    mesh = mydf.mesh
    ngrids = numpy.prod(mesh)

    dm_kpts = lib.asarray(dms, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    assert(nset == 2)

    ni = mydf._numint
    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    else:
        deriv = 2

    weight = cell.vol / ngrids
    if rho0 is None:
        rhoG = _eval_rhoG(mydf, dm0, hermi, kpts, deriv)
        rho0 = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
        rho0 = rho0.reshape(nset,-1,ngrids)

    if vxc is None or fxc is None:
        vxc, fxc = ni.eval_xc(xc_code, rho0, spin=1, deriv=2)[1:3]

    rhoG = _eval_rhoG(mydf, dms, hermi, kpts, deriv)
    rho1 = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    rho1 = rho1.reshape(nset,-1,ngrids)
    if with_j:
        coulG = tools.get_coulG(cell, mesh=mesh)
        vG = (rhoG[0,0] + rhoG[1,0]) * coulG
        vG = vG.reshape(mesh)

    if xctype == 'LDA':
        u_u, u_d, d_d = fxc[0].T
        wv = numpy.asarray([u_u * rho1[0] + u_d * rho1[1],
                            u_d * rho1[0] + d_d * rho1[1]])
        wv *= weight
        wv = tools.fft(wv.reshape(-1,ngrids), mesh).reshape(nset,-1,*mesh)
        if with_j:
            wv[:,0] += vG
        veff = _get_j_pass2(mydf, wv, kpts, verbose=log)

    elif xctype == 'GGA':
        wv = numint._uks_gga_wv1(rho0, rho1, vxc, fxc, weight)
        wv = numpy.vstack(wv).reshape(-1,ngrids)
        wv = tools.fft(wv, mesh).reshape(nset,-1,*mesh)
        if with_j:
            wv[:,0] += vG * .5
        veff = _get_gga_pass2(mydf, wv, kpts, verbose=log)

    return veff.reshape(dm_kpts.shape)


def cache_xc_kernel(mydf, xc_code, dm, spin=0, kpts=None):
    '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
    DFT hessian module etc.
    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))
    cell = mydf.cell
    mesh = mydf.mesh
    ngrids = numpy.prod(mesh)

    ni = mydf._numint
    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        deriv = 0
        comp = 1
    elif xctype == 'GGA':
        deriv = 1
        comp = 4
    else:
        deriv = 2

    hermi = 1
    weight = cell.vol / ngrids
    rhoG = _eval_rhoG(mydf, dm, hermi, kpts, deriv)
    rho = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    if spin == 0:
        rho = rho.reshape(comp,ngrids)
    else:
        rho = rho.reshape(2,comp,ngrids)

    vxc, fxc = ni.eval_xc(xc_code, rho, spin=spin, deriv=2)[1:3]
    return rho, vxc, fxc

def _gen_rhf_response(mf, dm0, singlet=None, hermi=0):
    '''multigrid version of function pbc.scf.newton_ah._gen_rhf_response
    '''
    #assert(isinstance(mf, dft.krks.KRKS))
    cell = mf.cell
    if getattr(mf, 'kpts', None) is not None:
        kpts = mf.kpts
    else:
        kpts = mf.kpt.reshape(1,3)
    ni = mf._numint
    if singlet is None:  # for newton solver
        rho0, vxc, fxc = cache_xc_kernel(mf.with_df, mf.xc, dm0, 0, kpts)
    else:
        rho0, vxc, fxc = cache_xc_kernel(mf.with_df, mf.xc, [dm0*.5]*2, 1, kpts)
    dm0 = None

    def vind(dm1):
        if hermi == 2:
            return numpy.zeros_like(dm1)

        if singlet is None:  # Without specify singlet, general case
            v1 = nr_rks_fxc(mf.with_df, mf.xc, dm0, dm1, hermi,
                            True, rho0, vxc, fxc, kpts)
        elif singlet:
            v1 = nr_rks_fxc_st(mf.with_df, mf.xc, dm0, dm1, hermi, singlet,
                               True, rho0, vxc, fxc, kpts)
        else:
            v1 = nr_rks_fxc_st(mf.with_df, mf.xc, dm0, dm1, hermi, singlet,
                               False, rho0, vxc, fxc, kpts)
        return v1
    return vind

def _gen_uhf_response(mf, dm0, with_j=True, hermi=0):
    '''multigrid version of function pbc.scf.newton_ah._gen_uhf_response
    '''
    #assert(isinstance(mf, dft.kuks.KUKS))
    cell = mf.cell
    if getattr(mf, 'kpts', None) is not None:
        kpts = mf.kpts
    else:
        kpts = mf.kpt.reshape(1,3)
    ni = mf._numint
    rho0, vxc, fxc = cache_xc_kernel(mf.with_df, mf.xc, dm0, 1, kpts)
    dm0 = None

    def vind(dm1):
        if hermi == 2:
            return numpy.zeros_like(dm1)

        v1 = nr_uks_fxc(mf.with_df, mf.xc, dm0, dm1, hermi,
                        with_j, rho0, vxc, fxc, kpts)
        return v1
    return vind


def get_rho(mydf, dm, kpts=numpy.zeros((1,3))):
    '''Density in real space
    '''
    cell = mydf.cell
    hermi = 1
    rhoG = _eval_rhoG(mydf, numpy.asarray(dm), hermi, kpts, deriv=0)

    mesh = mydf.mesh
    ngrids = numpy.prod(mesh)
    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = tools.ifft(rhoG.reshape(ngrids), mesh).real * (1./weight)
    return rhoR


def multi_grids_tasks(cell, fft_mesh=None, verbose=None):
    if TASKS_TYPE == 'rcut':
        return multi_grids_tasks_for_rcut(cell, fft_mesh, verbose)
    else:
        return multi_grids_tasks_for_ke_cut(cell, fft_mesh, verbose)

def multi_grids_tasks_for_rcut(cell, fft_mesh=None, verbose=None):
    log = logger.new_logger(cell, verbose)
    if fft_mesh is None:
        fft_mesh = cell.mesh

    # Split shells based on rcut
    rcuts_pgto, kecuts_pgto = _primitive_gto_cutoff(cell)
    ao_loc = cell.ao_loc_nr()

    def make_cell_dense_exp(shls_dense, r0, r1):
        cell_dense = copy.copy(cell)
        cell_dense._bas = cell._bas.copy()
        cell_dense._env = cell._env.copy()

        rcut_atom = [0] * cell.natm
        ke_cutoff = 0
        for ib in shls_dense:
            rc = rcuts_pgto[ib]
            idx = numpy.where((r1 <= rc) & (rc < r0))[0]
            np1 = len(idx)
            cs = cell._libcint_ctr_coeff(ib)
            np, nc = cs.shape
            if np1 < np:  # no pGTO splitting within the shell
                pexp = cell._bas[ib,PTR_EXP]
                pcoeff = cell._bas[ib,PTR_COEFF]
                cs1 = cs[idx]
                cell_dense._env[pcoeff:pcoeff+cs1.size] = cs1.T.ravel()
                cell_dense._env[pexp:pexp+np1] = cell.bas_exp(ib)[idx]
                cell_dense._bas[ib,NPRIM_OF] = np1

            ke_cutoff = max(ke_cutoff, kecuts_pgto[ib][idx].max())

            ia = cell.bas_atom(ib)
            rcut_atom[ia] = max(rcut_atom[ia], rc[idx].max())
        cell_dense._bas = cell_dense._bas[shls_dense]
        ao_idx = numpy.hstack([numpy.arange(ao_loc[i], ao_loc[i+1])
                               for i in shls_dense])
        cell_dense.rcut = max(rcut_atom)
        return cell_dense, ao_idx, ke_cutoff, rcut_atom

    def make_cell_sparse_exp(shls_sparse, r0, r1):
        cell_sparse = copy.copy(cell)
        cell_sparse._bas = cell._bas.copy()
        cell_sparse._env = cell._env.copy()

        for ib in shls_sparse:
            idx = numpy.where(r0 <= rcuts_pgto[ib])[0]
            np1 = len(idx)
            cs = cell._libcint_ctr_coeff(ib)
            np, nc = cs.shape
            if np1 < np:  # no pGTO splitting within the shell
                pexp = cell._bas[ib,PTR_EXP]
                pcoeff = cell._bas[ib,PTR_COEFF]
                cs1 = cs[idx]
                cell_sparse._env[pcoeff:pcoeff+cs1.size] = cs1.T.ravel()
                cell_sparse._env[pexp:pexp+np1] = cell.bas_exp(ib)[idx]
                cell_sparse._bas[ib,NPRIM_OF] = np1
        cell_sparse._bas = cell_sparse._bas[shls_sparse]
        ao_idx = numpy.hstack([numpy.arange(ao_loc[i], ao_loc[i+1])
                               for i in shls_sparse])
        return cell_sparse, ao_idx

    tasks = []
    a = cell.lattice_vectors()
    if abs(a-numpy.diag(a.diagonal())).max() < 1e-12:
        rmax = a.max() * RMAX_FACTOR_ORTH
    else:
        rmax = a.max() * RMAX_FACTOR_NONORTH
    n_delimeter = int(numpy.log(0.005/rmax) / numpy.log(RMAX_RATIO))
    rcut_delimeter = rmax * (RMAX_RATIO ** numpy.arange(n_delimeter))
    for r0, r1 in zip(numpy.append(1e9, rcut_delimeter),
                      numpy.append(rcut_delimeter, 0)):
        # shells which have high exps (small rcut)
        shls_dense = [ib for ib, rc in enumerate(rcuts_pgto)
                     if numpy.any((r1 <= rc) & (rc < r0))]
        if len(shls_dense) == 0:
            continue
        cell_dense, ao_idx_dense, ke_cutoff, rcut_atom = \
                make_cell_dense_exp(shls_dense, r0, r1)

        mesh = tools.cutoff_to_mesh(a, ke_cutoff)
        if TO_EVEN_GRIDS:
            mesh = (mesh+1)//2 * 2  # to the nearest even number
        if numpy.all(mesh >= fft_mesh):
            # Including all rest shells
            shls_dense = [ib for ib, rc in enumerate(rcuts_pgto)
                          if numpy.any(rc < r0)]
            cell_dense, ao_idx_dense = make_cell_dense_exp(shls_dense, r0, 0)[:2]
        cell_dense.mesh = mesh = numpy.min([mesh, fft_mesh], axis=0)

        grids_dense = gen_grid.UniformGrids(cell_dense)
        grids_dense.ao_idx = ao_idx_dense
        #grids_dense.rcuts_pgto = [rcuts_pgto[i] for i in shls_dense]

        # shells which have low exps (big rcut)
        shls_sparse = [ib for ib, rc in enumerate(rcuts_pgto)
                       if numpy.any(r0 <= rc)]
        if len(shls_sparse) == 0:
            cell_sparse = None
            ao_idx_sparse = []
        else:
            cell_sparse, ao_idx_sparse = make_cell_sparse_exp(shls_sparse, r0, r1)
            cell_sparse.mesh = mesh

        if cell_sparse is None:
            grids_sparse = None
        else:
            grids_sparse = gen_grid.UniformGrids(cell_sparse)
            grids_sparse.ao_idx = ao_idx_sparse

        log.debug('mesh %s nao dense/sparse %d %d  rcut %g',
                  mesh, len(ao_idx_dense), len(ao_idx_sparse), cell_dense.rcut)

        tasks.append([grids_dense, grids_sparse])
        if numpy.all(mesh >= fft_mesh):
            break
    return tasks

def multi_grids_tasks_for_ke_cut(cell, fft_mesh=None, verbose=None):
    log = logger.new_logger(cell, verbose)
    if fft_mesh is None:
        fft_mesh = cell.mesh

    # Split shells based on rcut
    rcuts_pgto, kecuts_pgto = _primitive_gto_cutoff(cell)
    ao_loc = cell.ao_loc_nr()

    def make_cell_dense_exp(shls_dense, ke0, ke1):
        cell_dense = copy.copy(cell)
        cell_dense._bas = cell._bas.copy()
        cell_dense._env = cell._env.copy()

        rcut_atom = [0] * cell.natm
        ke_cutoff = 0
        for ib in shls_dense:
            ke = kecuts_pgto[ib]
            idx = numpy.where((ke0 < ke) & (ke <= ke1))[0]
            np1 = len(idx)
            cs = cell._libcint_ctr_coeff(ib)
            np, nc = cs.shape
            if np1 < np:  # no pGTO splitting within the shell
                pexp = cell._bas[ib,PTR_EXP]
                pcoeff = cell._bas[ib,PTR_COEFF]
                cs1 = cs[idx]
                cell_dense._env[pcoeff:pcoeff+cs1.size] = cs1.T.ravel()
                cell_dense._env[pexp:pexp+np1] = cell.bas_exp(ib)[idx]
                cell_dense._bas[ib,NPRIM_OF] = np1

            ke_cutoff = max(ke_cutoff, ke[idx].max())

            ia = cell.bas_atom(ib)
            rcut_atom[ia] = max(rcut_atom[ia], rcuts_pgto[ib][idx].max())
        cell_dense._bas = cell_dense._bas[shls_dense]
        ao_idx = numpy.hstack([numpy.arange(ao_loc[i], ao_loc[i+1])
                               for i in shls_dense])
        cell_dense.rcut = max(rcut_atom)
        return cell_dense, ao_idx, ke_cutoff, rcut_atom

    def make_cell_sparse_exp(shls_sparse, ke0, ke1):
        cell_sparse = copy.copy(cell)
        cell_sparse._bas = cell._bas.copy()
        cell_sparse._env = cell._env.copy()

        for ib in shls_sparse:
            idx = numpy.where(kecuts_pgto[ib] <= ke0)[0]
            np1 = len(idx)
            cs = cell._libcint_ctr_coeff(ib)
            np, nc = cs.shape
            if np1 < np:  # no pGTO splitting within the shell
                pexp = cell._bas[ib,PTR_EXP]
                pcoeff = cell._bas[ib,PTR_COEFF]
                cs1 = cs[idx]
                cell_sparse._env[pcoeff:pcoeff+cs1.size] = cs1.T.ravel()
                cell_sparse._env[pexp:pexp+np1] = cell.bas_exp(ib)[idx]
                cell_sparse._bas[ib,NPRIM_OF] = np1
        cell_sparse._bas = cell_sparse._bas[shls_sparse]
        ao_idx = numpy.hstack([numpy.arange(ao_loc[i], ao_loc[i+1])
                               for i in shls_sparse])
        return cell_sparse, ao_idx

    a = cell.lattice_vectors()
    if abs(a-numpy.diag(a.diagonal())).max() < 1e-12:
        init_mesh = INIT_MESH_ORTH
    else:
        init_mesh = INIT_MESH_NONORTH
    ke_cutoff_min = tools.mesh_to_cutoff(cell.lattice_vectors(), init_mesh)
    ke_cutoff_max = max([ke.max() for ke in kecuts_pgto])
    ke1 = ke_cutoff_min.min()
    ke_delimeter = [0, ke1]
    while ke1 < ke_cutoff_max:
        ke1 *= KE_RATIO
        ke_delimeter.append(ke1)

    print(kecuts_pgto)
    print(ke_delimeter)
    tasks = []
    for ke0, ke1 in zip(ke_delimeter[:-1], ke_delimeter[1:]):
        # shells which have high exps (small rcut)
        shls_dense = [ib for ib, ke in enumerate(kecuts_pgto)
                     if numpy.any((ke0 < ke) & (ke <= ke1))]
        if len(shls_dense) == 0:
            continue

        print(ke0, ke1, shls_dense)
        mesh = tools.cutoff_to_mesh(a, ke1)
        if TO_EVEN_GRIDS:
            mesh = (mesh+1)//2 * 2  # to the nearest even number

        if numpy.all(mesh >= fft_mesh):
            # Including all rest shells
            shls_dense = [ib for ib, ke in enumerate(kecuts_pgto)
                          if numpy.any(ke0 < ke)]
            cell_dense, ao_idx_dense = make_cell_dense_exp(shls_dense, ke0,
                                                           ke_cutoff_max+1)[:2]
        else:
            cell_dense, ao_idx_dense, ke_cutoff, rcut_atom = \
                    make_cell_dense_exp(shls_dense, ke0, ke1)

        cell_dense.mesh = mesh = numpy.min([mesh, fft_mesh], axis=0)

        grids_dense = gen_grid.UniformGrids(cell_dense)
        grids_dense.ao_idx = ao_idx_dense
        #grids_dense.rcuts_pgto = [rcuts_pgto[i] for i in shls_dense]

        # shells which have low exps (big rcut)
        shls_sparse = [ib for ib, ke in enumerate(kecuts_pgto)
                       if numpy.any(ke <= ke0)]
        if len(shls_sparse) == 0:
            cell_sparse = None
            ao_idx_sparse = []
        else:
            cell_sparse, ao_idx_sparse = make_cell_sparse_exp(shls_sparse, ke0, ke1)
            cell_sparse.mesh = mesh

        if cell_sparse is None:
            grids_sparse = None
        else:
            grids_sparse = gen_grid.UniformGrids(cell_sparse)
            grids_sparse.ao_idx = ao_idx_sparse

        log.debug('mesh %s nao dense/sparse %d %d  rcut %g',
                  mesh, len(ao_idx_dense), len(ao_idx_sparse), cell_dense.rcut)

        tasks.append([grids_dense, grids_sparse])
        if numpy.all(mesh >= fft_mesh):
            break
    return tasks

def _primitive_gto_cutoff(cell):
    '''Cutoff raidus, above which each shell decays to a value less than the
    required precsion'''
    precision = cell.precision * EXTRA_PREC

    log_prec = numpy.log(precision)
    b = cell.reciprocal_vectors(norm_to=1)
    rcut = []
    ke_cutoff = []
    for ib in range(cell.nbas):
        l = cell.bas_angular(ib)
        es = cell.bas_exp(ib)
        cs = abs(cell.bas_ctr_coeff(ib)).max(axis=1)
        r = 5.
        r = (((l+2)*numpy.log(r)+numpy.log(cs) - log_prec) / es)**.5
        r = (((l+2)*numpy.log(r)+numpy.log(cs) - log_prec) / es)**.5

# Errors in total number of electrons were observed with the default
# precision. The energy cutoff (or the integration mesh) is not enough to
# produce the desired accuracy. Scale precision by 0.1 to decrease the error.
        ke_guess = gto.cell._estimate_ke_cutoff(es, l, cs, precision*0.1)

        rcut.append(r)
        ke_cutoff.append(ke_guess)
    return rcut, ke_cutoff


class MultiGridFFTDF(fft.FFTDF):
    def __init__(self, cell, kpts=numpy.zeros((1,3))):
        fft.FFTDF.__init__(self, cell, kpts)
        self.tasks = None
        self._keys = self._keys.union(['tasks'])

    def build(self):
        self.tasks = multi_grids_tasks(self.cell, self.mesh, self.verbose)

    get_pp = get_pp
    get_nuc = get_nuc

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, exxdiv='ewald', **kwargs):
        from pyscf.pbc.df import fft_jk
        if with_k:
            logger.warn(self, 'MultiGridFFTDF does not support HFX. '
                        'HFX is computed by FFTDF.get_k_kpts function.')

        if kpts is None:
            if numpy.all(self.kpts == 0): # Gamma-point J/K by default
                kpts = numpy.zeros(3)
            else:
                kpts = self.kpts
        else:
            kpts = numpy.asarray(kpts)

        vj = vk = None
        if kpts.shape == (3,):
            if with_k:
                vk = fft_jk.get_jk(self, dm, hermi, kpts, kpts_band,
                                   False, True, exxdiv)[1]
            vj = get_j_kpts(self, dm, hermi, kpts.reshape(1,3), kpts_band)
            if kpts_band is None:
                vj = vj[...,0,:,:]
        else:
            if with_k:
                vk = fft_jk.get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
            if with_j:
                vj = get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk

    get_rho = get_rho


def multigrid(mf):
    '''Use MultiGridFFTDF to replace the default FFTDF integration method in
    the DFT object.
    '''
    from pyscf.pbc import dft
    mf.with_df, old_df = MultiGridFFTDF(mf.cell), mf.with_df
    keys = mf.with_df._keys
    mf.with_df.__dict__.update(old_df.__dict__)
    mf.with_df._keys = keys
    return mf


def _pgto_shells(cell):
    return cell._bas[:,NPRIM_OF].sum()

def _take_4d(a, indices):
    a_shape = a.shape
    ranges = []
    for i, s in enumerate(indices):
        if s is None:
            idx = numpy.arange(a_shape[i], dtype=numpy.int32)
        else:
            idx = numpy.asarray(s, dtype=numpy.int32)
            idx[idx < 0] += a_shape[i]
        ranges.append(idx)
    idx = ranges[0][:,None] * a_shape[1] + ranges[1]
    idy = ranges[2][:,None] * a_shape[3] + ranges[3]
    a = a.reshape(a_shape[0]*a_shape[1], a_shape[2]*a_shape[3])
    out = lib.take_2d(a, idx.ravel(), idy.ravel())
    return out.reshape([len(s) for s in ranges])

def _takebak_4d(out, a, indices):
    out_shape = out.shape
    a_shape = a.shape
    ranges = []
    for i, s in enumerate(indices):
        if s is None:
            idx = numpy.arange(a_shape[i], dtype=numpy.int32)
        else:
            idx = numpy.asarray(s, dtype=numpy.int32)
            idx[idx < 0] += out_shape[i]
        assert(len(idx) == a_shape[i])
        ranges.append(idx)
    idx = ranges[0][:,None] * out_shape[1] + ranges[1]
    idy = ranges[2][:,None] * out_shape[3] + ranges[3]
    nx = idx.size
    ny = idy.size
    out = out.reshape(out_shape[0]*out_shape[1], out_shape[2]*out_shape[3])
    lib.takebak_2d(out, a.reshape(nx,ny), idx.ravel(), idy.ravel())
    return out

def _take_5d(a, indices):
    a_shape = a.shape
    a = a.reshape((a_shape[0]*a_shape[1],) + a_shape[2:])
    indices = (None,) + indices[2:]
    return _take_4d(a, indices)

def _takebak_5d(out, a, indices):
    a_shape = a.shape
    out_shape = out.shape
    a = a.reshape((a_shape[0]*a_shape[1],) + a_shape[2:])
    out = out.reshape((out_shape[0]*out_shape[1],) + out_shape[2:])
    indices = (None,) + indices[2:]
    return _takebak_4d(out, a, indices)


if __name__ == '__main__':
    from pyscf.pbc import gto, scf, dft
    from pyscf.pbc import df
    from pyscf.pbc.df import fft_jk
    numpy.random.seed(22)
    cell = gto.M(
        a = numpy.eye(3)*3.5668,
        atom = '''C     0.      0.      0.    
                  C     0.8917  0.8917  0.8917
                  C     1.7834  1.7834  0.    
                  C     2.6751  2.6751  0.8917
                  C     1.7834  0.      1.7834
                  C     2.6751  0.8917  2.6751
                  C     0.      1.7834  1.7834
                  C     0.8917  2.6751  2.6751''',
        #basis = 'sto3g',
        #basis = 'ccpvdz',
        basis = 'gth-dzvp',
        #basis = 'gth-szv',
        #verbose = 5,
        #mesh = [15]*3,
        #precision=1e-6
        pseudo = 'gth-pade'
    )
    multi_grids_tasks(cell, cell.mesh, 5)

    nao = cell.nao_nr()
    numpy.random.seed(1)
    kpts = cell.make_kpts([3,1,1])

    dm = numpy.random.random((len(kpts),nao,nao)) * .2
    dm += numpy.eye(nao)
    dm = dm + dm.transpose(0,2,1)

    mf = dft.KRKS(cell)
    ref = mf.get_veff(cell, dm, kpts=kpts)
    out = multigrid(mf).get_veff(cell, dm, kpts=kpts)
    print(abs(ref-out).max())

