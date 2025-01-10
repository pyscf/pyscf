#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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

import ctypes
import numpy
import scipy.linalg

from pyscf import __config__
from pyscf import lib
from pyscf.lib import logger
from pyscf.gto import ATOM_OF, ANG_OF, NPRIM_OF, PTR_EXP, PTR_COEFF
from pyscf.dft.numint import libdft, BLKSIZE, MGGA_DENSITY_LAPL
from pyscf.pbc import tools
from pyscf.pbc import gto
from pyscf.pbc.gto import pseudo
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.pbc.dft import numint, gen_grid
from pyscf.pbc.scf.khf import KSCF
from pyscf.pbc.df.df_jk import (
    _format_dms,
    _format_kpts_band,
    _format_jks,
)
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.df import fft, ft_ao, aft
from pyscf.pbc.dft.multigrid.utils import (
    _take_4d,
    _take_5d,
    _takebak_4d,
    _takebak_5d,
)

#sys.stderr.write('WARN: multigrid is an experimental feature. It is still in '
#                 'testing\nFeatures and APIs may be changed in the future.\n')

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
    assert (all(cell._bas[:,NPRIM_OF] == 1))
    if mesh is None:
        mesh = cell.mesh
    vol = cell.vol
    weight_penalty = numpy.prod(mesh) / vol
    exp_min = numpy.hstack(cell.bas_exps()).min()
    theta_ij = exp_min / 2
    lattice_sum_fac = max(2*numpy.pi*cell.rcut/(vol*theta_ij), 1)
    precision = cell.precision / weight_penalty / lattice_sum_fac
    if xctype != 'LDA':
        precision *= .1
    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    env[PTR_EXPDROP] = min(precision*EXTRA_PREC, EXPDROP)
    ao_loc = gto.moleintor.make_loc(bas, 'cart')
    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas)
    i0, i1, j0, j1 = shls_slice
    j0 += cell.nbas
    j1 += cell.nbas
    naoi = ao_loc[i1] - ao_loc[i0]
    naoj = ao_loc[j1] - ao_loc[j0]

    Ls = gto.eval_gto.get_lattice_Ls(cell)
    nimgs = len(Ls)

    weights = numpy.asarray(weights, order='C')
    assert (weights.dtype == numpy.double)
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
    log_prec = numpy.log(precision * EXTRA_PREC)

    if abs(a-numpy.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
    else:
        lattice_type = '_nonorth'
    eval_fn = 'NUMINTeval_' + xctype.lower() + lattice_type
    drv = libdft.NUMINT_fill2c

    def make_mat(weights):
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
            mat = make_mat(wv)[0].transpose(0,2,1)
            if hermi == 1:
                for i in range(comp):
                    lib.hermi_triu(mat[i], inplace=True)
            if comp == 1:
                mat = mat[0]
        elif kpts is None or gamma_point(kpts):
            mat = make_mat(wv).sum(axis=0).transpose(0,2,1)
            if hermi == 1:
                for i in range(comp):
                    lib.hermi_triu(mat[i], inplace=True)
            if comp == 1:
                mat = mat[0]
            if getattr(kpts, 'ndim', None) == 2:
                mat = mat[None,:]
        else:
            mat = make_mat(wv)
            expkL = numpy.exp(1j*kpts.reshape(-1,3).dot(Ls.T))
            mat = lib.einsum('kr,rcij->kcij', expkL, mat)
            if hermi == 1:
                for i in range(comp):
                    for k in range(len(kpts)):
                        lib.hermi_triu(mat[k,i], inplace=True)
            mat = mat.transpose(0,1,3,2)
            if comp == 1:
                mat = mat[:,0]
        out.append(mat)

    if n_mat is None:
        out = out[0]
    return out

def eval_rho(cell, dm, shls_slice=None, hermi=0, xctype='LDA', kpts=None,
             mesh=None, offset=None, submesh=None, ignore_imag=False,
             out=None):
    '''Collocate the *real* density (opt. gradients) on the real-space grid.

    Kwargs:
        ignore_image :
            The output density is assumed to be real if ignore_imag=True.
    '''
    assert (all(cell._bas[:,NPRIM_OF] == 1))
    if mesh is None:
        mesh = cell.mesh
    vol = cell.vol
    weight_penalty = numpy.prod(mesh) / vol
    exp_min = numpy.hstack(cell.bas_exps()).min()
    theta_ij = exp_min / 2
    lattice_sum_fac = max(2*numpy.pi*cell.rcut/(vol*theta_ij), 1)
    precision = cell.precision / weight_penalty / lattice_sum_fac
    if xctype != 'LDA':
        precision *= .1
    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    env[PTR_EXPDROP] = min(precision*EXTRA_PREC, EXPDROP)
    ao_loc = gto.moleintor.make_loc(bas, 'cart')
    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas)
    i0, i1, j0, j1 = shls_slice
    if hermi == 1:
        assert (i0 == j0 and i1 == j1)
    j0 += cell.nbas
    j1 += cell.nbas
    naoi = ao_loc[i1] - ao_loc[i0]
    naoj = ao_loc[j1] - ao_loc[j0]
    dm = numpy.asarray(dm, order='C')
    assert (dm.shape[-2:] == (naoi, naoj))

    Ls = gto.eval_gto.get_lattice_Ls(cell)

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
    if offset is None:
        offset = (0, 0, 0)
    if submesh is None:
        submesh = mesh
    log_prec = numpy.log(precision * EXTRA_PREC)

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

    def make_rho_(rho, dm, hermi):
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
        if cell.dimension == 0:
            if ignore_imag:
                # basis are real. dm.imag can be dropped if ignore_imag
                dm_i = dm_i.real
            has_imag = dm_i.dtype == numpy.complex128
            if has_imag:
                dmR = numpy.asarray(dm_i.real, order='C')
                dmI = numpy.asarray(dm_i.imag, order='C')
            else:
                # make a copy because the dm may be overwritten in the
                # NUMINT_rho_drv inplace
                dmR = numpy.array(dm_i, order='C', copy=True)

        elif kpts is None or gamma_point(kpts):
            if ignore_imag:
                # basis are real. dm.imag can be dropped if ignore_imag
                dm_i = dm_i.real
            has_imag = dm_i.dtype == numpy.complex128
            if has_imag:
                dmR = numpy.repeat(dm_i.real, nimgs, axis=0)
                dmI = numpy.repeat(dm_i.imag, nimgs, axis=0)
            else:
                dmR = numpy.repeat(dm_i, nimgs, axis=0)

        else:
            dm_L = lib.dot(expkL.T, dm_i.reshape(nkpts,-1)).reshape(nimgs,naoj,naoi)
            dmR = numpy.asarray(dm_L.real, order='C')

            if ignore_imag:
                has_imag = False
            else:
                dmI = numpy.asarray(dm_L.imag, order='C')
                has_imag = (hermi == 0 and abs(dmI).max() > 1e-8)
                if (has_imag and xctype == 'LDA' and
                    naoi == naoj and
                    # For hermitian density matrices, the anti-symmetry
                    # character of the imaginary part of the density matrices
                    # can be found by rearranging the repeated images.
                    abs(dm_i - dm_i.conj().transpose(0,2,1)).max() < 1e-8):
                    has_imag = False
            dm_L = None


        if has_imag:
            # complex density cannot be updated inplace directly by
            # function NUMINT_rho_drv
            if out is None:
                rho_i = numpy.empty(shape, numpy.complex128)
                rho_i.real = make_rho_(numpy.zeros(shape), dmR, 0)
                rho_i.imag = make_rho_(numpy.zeros(shape), dmI, 0)
            else:
                assert out[i].dtype == numpy.complex128
                rho_i = out[i].reshape(shape)
                rho_i.real += make_rho_(numpy.zeros(shape), dmR, 0)
                rho_i.imag += make_rho_(numpy.zeros(shape), dmI, 0)
        else:
            if out is None:
                # rho_i needs to be initialized to 0 because rho_i is updated
                # inplace in function NUMINT_rho_drv
                rho_i = make_rho_(numpy.zeros(shape), dmR, hermi)
            else:
                assert out[i].dtype == numpy.double
                rho_i = out[i].reshape(shape)
                make_rho_(rho_i, dmR, hermi)
        dmR = dmI = None
        rho.append(rho_i)

    if n_dm == 1:
        rho = rho[0]
    return rho

def get_nuc(mydf, kpts=None):
    kpts, is_single_kpt = fft._check_kpts(mydf, kpts)
    cell = mydf.cell
    mesh = mydf.mesh
    charge = -cell.atom_charges()
    Gv = cell.get_Gv(mesh)
    SI = cell.get_SI(Gv)
    rhoG = numpy.dot(charge, SI)

    coulG = tools.get_coulG(cell, mesh=mesh, Gv=Gv)
    vneG = rhoG * coulG
    hermi = 1
    vne = _get_j_pass2(mydf, vneG, hermi, kpts)[0]

    if is_single_kpt:
        vne = vne[0]
    return numpy.asarray(vne)

def get_pp(mydf, kpts=None, max_memory=4000):
    '''Get the periodic pseudopotential nuc-el AO matrix, with G=0 removed.
    '''
    from pyscf import gto
    kpts, is_single_kpt = fft._check_kpts(mydf, kpts)
    cell = mydf.cell
    mesh = mydf.mesh
    Gv = cell.get_Gv(mesh)

    ngrids = len(Gv)
    vpplocG = numpy.empty((ngrids,), dtype=numpy.complex128)

    mem_avail = max(max_memory, mydf.max_memory-lib.current_memory()[0])
    blksize = int(mem_avail*1e6/((cell.natm*2)*16))
    blksize = min(ngrids, max(21**3, blksize))
    for ig0, ig1 in lib.prange(0, ngrids, blksize):
        vpplocG_batch = pp_int.get_gth_vlocG_part1(cell, Gv[ig0:ig1])
        SI = cell.get_SI(Gv[ig0:ig1])
        vpplocG[ig0:ig1] = -numpy.einsum('ij,ij->j', SI, vpplocG_batch)

    hermi = 1
    vpp = _get_j_pass2(mydf, vpplocG, hermi, kpts)[0]
    vpp2 = pp_int.get_pp_loc_part2(cell, kpts)
    for k, kpt in enumerate(kpts):
        vpp[k] += vpp2[k]

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

    def vppnl_by_k(kpt):
        SPG_lm_aoGs = []
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                SPG_lm_aoGs.append(None)
                continue
            pp = cell._pseudo[symb]
            p1 = 0
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                if nl > 0:
                    p1 = p1+nl*(l*2+1)
            SPG_lm_aoGs.append(numpy.zeros((p1, cell.nao), dtype=numpy.complex128))

        mem_avail = max(max_memory, mydf.max_memory-lib.current_memory()[0])
        blksize = int(mem_avail*1e6/((48+cell.nao+13+3)*16))
        blksize = min(ngrids, max(21**3, blksize))
        vppnl = 0
        for ig0, ig1 in lib.prange(0, ngrids, blksize):
            ng = ig1 - ig0
            # buf for SPG_lmi upto l=0..3 and nl=3
            buf = numpy.empty((48,ng), dtype=numpy.complex128)
            Gk = Gv[ig0:ig1] + kpt
            G_rad = numpy.linalg.norm(Gk, axis=1)
            aokG = ft_ao.ft_ao(cell, Gv[ig0:ig1], kpt=kpt) * (ngrids/cell.vol)
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
                        pYlm = numpy.ndarray((nl,l*2+1,ng), dtype=numpy.complex128, buffer=buf[p0:p1])
                        for k in range(nl):
                            qkl = pseudo.pp._qli(G_rad*rl, l, k)
                            pYlm[k] = pYlm_part.T * qkl
                        #:SPG_lmi = numpy.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                        #:SPG_lm_aoG = numpy.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                        #:tmp = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                        #:vppnl += numpy.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
                if p1 > 0:
                    SPG_lmi = buf[:p1]
                    SPG_lmi *= cell.get_SI(Gv[ig0:ig1], atmlst=[ia,]).conj()
                    SPG_lm_aoGs[ia] += lib.zdot(SPG_lmi, aokG)
            buf = None
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                continue
            pp = cell._pseudo[symb]
            p1 = 0
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                if nl > 0:
                    p0, p1 = p1, p1+nl*(l*2+1)
                    hl = numpy.asarray(hl)
                    SPG_lm_aoG = SPG_lm_aoGs[ia][p0:p1].reshape(nl,l*2+1,-1)
                    tmp = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                    vppnl += numpy.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
        SPG_lm_aoGs=None
        return vppnl * (1./ngrids**2)

    for k, kpt in enumerate(kpts):
        vppnl = vppnl_by_k(kpt)
        if gamma_point(kpt):
            vpp[k] = vpp[k].real + vppnl.real
        else:
            vpp[k] += vppnl

    if is_single_kpt:
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
        kpts_band : ``(3,)`` ndarray or ``(*,3)`` ndarray
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
    vj_kpts = _get_j_pass2(mydf, vG, hermi, kpts_band)
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

    assert (deriv < 2)
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
        #if hermi != 1 and not gamma_point(kpts):
        #    raise NotImplementedError

    elif deriv == 2:  # meta-GGA
        raise NotImplementedError

    ignore_imag = (hermi == 1)

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

            l_cell = grids_sparse.cell
            h_pcell, h_coeff = h_cell.decontract_basis(to_cart=True, aggregate=True)
            l_pcell, l_coeff = l_cell.decontract_basis(to_cart=True, aggregate=True)
            t_cell = h_pcell + l_pcell
            t_coeff = scipy.linalg.block_diag(h_coeff, l_coeff)

            nshells_h = _pgto_shells(h_cell)
            nshells_t = _pgto_shells(t_cell)

            if deriv == 0:
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
                                        'LDA', kpts, grids_dense, ignore_imag, log)
                    pgto_dms = lib.einsum('nkij,pi,qj->nkpq', dms_lh, l_coeff, h_coeff)
                    shls_slice = (nshells_h, nshells_t, 0, nshells_h)
                    #:rho += eval_rho(t_cell, pgto_dms, shls_slice, 0, 'LDA', kpts,
                    #:                offset=None, submesh=None)
                    rho += _eval_rho_ket(t_cell, pgto_dms, shls_slice, 0,
                                         'LDA', kpts, grids_dense, ignore_imag, log)

            elif deriv == 1:
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
    pcell = cell.copy(deep=False)
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
        # Update rho inplace
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
    pcell = cell.copy(deep=False)
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
        # Update rho inplace
        eval_rho(pcell, sub_dms, sub_slice, hermi, xctype, kpts,
                 mesh, ignore_imag=ignore_imag, out=rho)
    return rho.reshape((nset, rhodim, numpy.prod(mesh)))


def _get_j_pass2(mydf, vG, hermi=1, kpts=numpy.zeros((1,3)), verbose=None):
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
        ignore_vG_imag = hermi == 1 or abs(vI.sum()) < IMAG_TOL
        if ignore_vG_imag:
            v_rs = vR
        elif vj_kpts.dtype == numpy.double:
            # ensure result complex array if tddft amplitudes are complex while
            # at gamma point
            vj_kpts = vj_kpts.astype(numpy.complex128)

        idx_h = grids_dense.ao_idx
        if grids_sparse is None:
            for ao_h_etc, p0, p1 in mydf.aoR_loop(grids_dense, kpts):
                ao_h = ao_h_etc[0]
                for k in range(nkpts):
                    for i in range(nset):
                        aow = numint._scale_ao(ao_h[k], v_rs[i,p0:p1])
                        vj_sub = lib.dot(ao_h[k].conj().T, aow)
                        vj_kpts[i,k,idx_h[:,None],idx_h] += vj_sub
                ao_h = ao_h_etc = None
        else:
            idx_h = grids_dense.ao_idx
            idx_l = grids_sparse.ao_idx
            # idx_t = numpy.append(idx_h, idx_l)
            naoh = len(idx_h)

            h_cell = grids_dense.cell
            l_cell = grids_sparse.cell
            h_pcell, h_coeff = h_cell.decontract_basis(to_cart=True, aggregate=True)
            l_pcell, l_coeff = l_cell.decontract_basis(to_cart=True, aggregate=True)
            t_cell = h_pcell + l_pcell
            t_coeff = scipy.linalg.block_diag(h_coeff, l_coeff)

            nshells_h = _pgto_shells(h_cell)
            nshells_t = _pgto_shells(t_cell)
            shls_slice = (0, nshells_h, 0, nshells_t)
            vp = eval_mat(t_cell, vR, shls_slice, 1, 0, 'LDA', kpts)
            # Imaginary part may contribute
            if not ignore_vG_imag:
                vpI = eval_mat(t_cell, vI, shls_slice, 1, 0, 'LDA', kpts)
                vp = numpy.asarray(vp) + numpy.asarray(vpI) * 1j
                vpI = None

            vp = lib.einsum('nkpq,pi,qj->nkij', vp, h_coeff, t_coeff)

            vj_kpts[:,:,idx_h[:,None],idx_h] += vp[:,:,:,:naoh]
            vj_kpts[:,:,idx_h[:,None],idx_l] += vp[:,:,:,naoh:]

            if hermi == 1:
                vj_kpts[:,:,idx_l[:,None],idx_h] += \
                        vp[:,:,:,naoh:].transpose(0,1,3,2).conj()
            else:
                shls_slice = (nshells_h, nshells_t, 0, nshells_h)
                vp = eval_mat(t_cell, vR, shls_slice, 1, 0, 'LDA', kpts)
                # Imaginary part may contribute
                if not ignore_vG_imag:
                    vpI = eval_mat(t_cell, vI, shls_slice, 1, 0, 'LDA', kpts)
                    vp = numpy.asarray(vp) + numpy.asarray(vpI) * 1j
                    vpI = None
                vp = lib.einsum('nkpq,pi,qj->nkij', vp, l_coeff, h_coeff)
                vj_kpts[:,:,idx_l[:,None],idx_h] += vp

    return vj_kpts


def _get_gga_pass2(mydf, vG, hermi=1, kpts=numpy.zeros((1,3)), verbose=None):
    #if hermi != 1:
    #    raise NotImplementedError('_get_gga_pass2 assumes hermi=1')
    log = logger.new_logger(mydf, verbose)
    cell = mydf.cell
    nkpts = len(kpts)
    nao = cell.nao_nr()
    nx, ny, nz = mydf.mesh
    vG = vG.reshape(-1,4,nx,ny,nz)
    nset = vG.shape[0]

    at_gamma_point = gamma_point(kpts)
    if at_gamma_point:
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
        v_rs = tools.ifft(sub_vG, mesh).reshape(nset,4,ngrids)
        vR = numpy.asarray(v_rs.real, order='C')
        vI = numpy.asarray(v_rs.imag, order='C')
        ignore_vG_imag = hermi == 1 or abs(vI.sum()) < IMAG_TOL
        if ignore_vG_imag:
            v_rs = vR
        elif veff.dtype == numpy.double:
            # ensure result complex array if tddft amplitudes are complex while
            # at gamma point
            veff = veff.astype(numpy.complex128)

        if grids_sparse is None:
            idx_h = grids_dense.ao_idx
            naoh = len(idx_h)
            for ao_h_etc, p0, p1 in mydf.aoR_loop(grids_dense, kpts, deriv=1):
                ao_h = ao_h_etc[0]
                for k in range(nkpts):
                    for i in range(nset):
                        aow = numint._scale_ao(ao_h[k], v_rs[i])
                        v = lib.dot(ao_h[k][0].conj().T, aow)
                        veff[i,k,idx_h[:,None],idx_h] += v
                        if hermi == 1:
                            veff[i,k,idx_h[:,None],idx_h] += v.conj().T
                        else:
                            aow = numint._scale_ao(ao_h[k], v_rs[i].conj())
                            v = lib.dot(aow.conj().T, ao_h[k][0])
                            veff[i,k,idx_h[:,None],idx_h] += v
                ao_h = ao_h_etc = None
        else:
            idx_h = grids_dense.ao_idx
            idx_l = grids_sparse.ao_idx
            # idx_t = numpy.append(idx_h, idx_l)
            naoh = len(idx_h)

            h_cell = grids_dense.cell
            l_cell = grids_sparse.cell
            h_pcell, h_coeff = h_cell.decontract_basis(to_cart=True, aggregate=True)
            l_pcell, l_coeff = l_cell.decontract_basis(to_cart=True, aggregate=True)
            t_cell = h_pcell + l_pcell
            t_coeff = scipy.linalg.block_diag(h_coeff, l_coeff)

            nshells_h = _pgto_shells(h_cell)
            nshells_t = _pgto_shells(t_cell)
            shls_slice = (0, nshells_h, 0, nshells_t)
            vpR = eval_mat(t_cell, vR, shls_slice, 1, 0, 'GGA', kpts)
            vp = vpR = lib.einsum('nkpq,pi,qj->nkij', vpR, h_coeff, t_coeff)
            if not ignore_vG_imag:
                vpI = eval_mat(t_cell, vI, shls_slice, 1, 0, 'GGA', kpts)
                vpI = lib.einsum('nkpq,pi,qj->nkij', vpI, h_coeff, t_coeff)
                vp = numpy.asarray(vpR) + numpy.asarray(vpI) * 1j
            veff[:,:,idx_h[:,None],idx_h] += vp[:,:,:,:naoh]
            veff[:,:,idx_h[:,None],idx_l] += vp[:,:,:,naoh:]
            if hermi == 1:
                veff[:,:,idx_h[:,None],idx_h] += vp[:,:,:,:naoh].conj().transpose(0,1,3,2)
                veff[:,:,idx_l[:,None],idx_h] += vp[:,:,:,naoh:].conj().transpose(0,1,3,2)
            else:
                if not ignore_vG_imag:
                    # eval_mat only supports <nabla i|v|j>. Evaluate <i|v|nabla j>
                    # by conj(<nabla j|conj(v)|>)
                    vp = numpy.asarray(vpR) - numpy.asarray(vpI) * 1j
                veff[:,:,idx_h[:,None],idx_h] += vp[:,:,:,:naoh].conj().transpose(0,1,3,2)
                veff[:,:,idx_l[:,None],idx_h] += vp[:,:,:,naoh:].conj().transpose(0,1,3,2)

            shls_slice = (nshells_h, nshells_t, 0, nshells_h)
            vpR = eval_mat(t_cell, vR, shls_slice, 1, 0, 'GGA', kpts)
            vp = vpR = lib.einsum('nkpq,pi,qj->nkij', vpR, l_coeff, h_coeff)
            if not ignore_vG_imag:
                vpI = eval_mat(t_cell, vI, shls_slice, 1, 0, 'GGA', kpts)
                vpI = lib.einsum('nkpq,pi,qj->nkij', vpI, l_coeff, h_coeff)
                vp = numpy.asarray(vpR) + numpy.asarray(vpI) * 1j
            veff[:,:,idx_l[:,None],idx_h] += vp
            if hermi == 1:
                veff[:,:,idx_h[:,None],idx_l] += vp.conj().transpose(0,1,3,2)
            else:
                if not ignore_vG_imag:
                    vp = numpy.asarray(vpR) - numpy.asarray(vpI) * 1j
                veff[:,:,idx_h[:,None],idx_l] += vp.conj().transpose(0,1,3,2)

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
        kpts_band : ``(3,)`` ndarray or ``(*,3)`` ndarray
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

    ni = mydf._numint
    xctype = ni._xc_type(xc_code)

    if xctype == 'LDA':
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    elif xctype == 'MGGA':
        deriv = 2 if MGGA_DENSITY_LAPL else 1
        raise NotImplementedError
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
    nelec = rhoR[:,0].sum(axis=1) * weight

    wv_freq = []
    excsum = numpy.zeros(nset)
    for i in range(nset):
        if xctype == 'LDA':
            exc, vxc = ni.eval_xc_eff(xc_code, rhoR[i,0], deriv=1, xctype=xctype)[:2]
        else:
            exc, vxc = ni.eval_xc_eff(xc_code, rhoR[i], deriv=1, xctype=xctype)[:2]
        excsum[i] += (rhoR[i,0]*exc).sum() * weight
        wv = weight * vxc
        wv_freq.append(tools.fft(wv, mesh))
    wv_freq = numpy.asarray(wv_freq).reshape(nset,-1,*mesh)
    rhoR = rhoG = None

    if nset == 1:
        ecoul = ecoul[0]
        nelec = nelec[0]
        excsum = excsum[0]
    log.debug('Multigrid exc %s  nelec %s', excsum, nelec)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    if xctype == 'LDA':
        if with_j:
            wv_freq[:,0] += vG.reshape(nset,*mesh)
        veff = _get_j_pass2(mydf, wv_freq, hermi, kpts_band, verbose=log)
    elif xctype == 'GGA':
        if with_j:
            wv_freq[:,0] += vG.reshape(nset,*mesh)
        # *.5 because v+v.T is always called in _get_gga_pass2
        wv_freq[:,0] *= .5
        veff = _get_gga_pass2(mydf, wv_freq, hermi, kpts_band, verbose=log)
    veff = _format_jks(veff, dm_kpts, input_band, kpts)

    if return_j:
        vj = _get_j_pass2(mydf, vG, hermi, kpts_band, verbose=log)
        vj = _format_jks(veff, dm_kpts, input_band, kpts)
    else:
        vj = None

    shape = list(dm_kpts.shape)
    if len(shape) == 3 and shape[0] != kpts_band.shape[0]:
        shape[0] = kpts_band.shape[0]
    veff = veff.reshape(shape)
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
        kpts_band : ``(3,)`` ndarray or ``(*,3)`` ndarray
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
    nset //= 2
    # Do not support gks
    assert nset == 1
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band

    ni = mydf._numint
    xctype = ni._xc_type(xc_code)

    if xctype == 'LDA':
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    elif xctype == 'MGGA':
        raise NotImplementedError

    mesh = mydf.mesh
    ngrids = numpy.prod(mesh)
    rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv)
    rhoG = rhoG.reshape(nset,2,-1,ngrids)
    coulG = tools.get_coulG(cell, mesh=mesh)
    vG = numpy.einsum('nsg,g->ng', rhoG[:,:,0], coulG)
    ecoul = .5 * numpy.einsum('nsg,ng->n', rhoG[:,:,0].real, vG.real)
    ecoul+= .5 * numpy.einsum('nsg,ng->n', rhoG[:,:,0].imag, vG.imag)
    ecoul /= cell.vol
    log.debug('Multigrid Coulomb energy %s', ecoul)

    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    rhoR = rhoR.reshape(nset,2,-1,ngrids)
    nelec = numpy.einsum('nsg->n', rhoR[:,:,0]) * weight

    wv_freq = []
    excsum = numpy.zeros(nset)
    for i in range(nset):
        exc, vxc = ni.eval_xc_eff(xc_code, rhoR[i], deriv=1, xctype=xctype)[:2]
        excsum[i] = (rhoR[i,:,0]*exc).sum() * weight
        log.debug('Multigrid exc %g  nelec %s', excsum, nelec[i])
        wv = weight * vxc
        wv_freq.append(tools.fft(wv, mesh))
    wv_freq = numpy.asarray(wv_freq).reshape(nset,2,-1,*mesh)
    rhoR = rhoG = None

    if with_j:
        wv_freq[:,:,0] += vG.reshape(*mesh)

    if xctype == 'LDA':
        veff = _get_j_pass2(mydf, wv_freq, hermi, kpts_band, verbose=log)
    elif xctype == 'GGA':
        # *.5 because v+v.T is always called in _get_gga_pass2
        wv_freq[:,0] *= .5
        veff = _get_gga_pass2(mydf, wv_freq, hermi, kpts_band, verbose=log)
    veff = _format_jks(veff, dm_kpts, input_band, kpts)
    veff = veff.reshape(nset, 2, len(kpts_band), nao, nao)

    if nset == 1:
        veff = veff[0]
        ecoul = ecoul[0]
        nelec = nelec[0]
        excsum = excsum[0]

    if return_j:
        vj = _get_j_pass2(mydf, vG, hermi, kpts_band, verbose=log)
        vj = _format_jks(veff, dm_kpts, input_band, kpts)
        if nset == 1:
            vj = vj[0]
    else:
        vj = None

    shape = list(dm_kpts.shape)
    if len(shape) == 4 and shape[1] != kpts_band.shape[0]:
        shape[1] = kpts_band.shape[0]
    veff = veff.reshape(shape)
    veff = lib.tag_array(veff, ecoul=ecoul, exc=excsum, vj=vj, vk=None)
    return nelec, excsum, veff


def nr_rks_fxc(mydf, xc_code, dm0, dms, hermi=0, with_j=False,
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
    elif xctype == 'MGGA':
        deriv = 2 if MGGA_DENSITY_LAPL else 1
        raise NotImplementedError

    weight = cell.vol / ngrids
    if rho0 is None:
        rhoG = _eval_rhoG(mydf, dm0, hermi, kpts, deriv)
        rho0 = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
        if xctype == 'LDA':
            rho0 = rho0.reshape(ngrids)

    if fxc is None:
        fxc = ni.eval_xc_eff(xc_code, rho0, deriv=2, xctype=xctype)[2]

    rhoG = _eval_rhoG(mydf, dms, hermi, kpts, deriv)
    rho1 = tools.ifft(rhoG.reshape(-1,ngrids), mesh)
    if hermi == 1:
        rho1 = rho1.real
    rho1 *= (1./weight)
    rho1 = rho1.reshape(nset,-1,ngrids)
    wv = numpy.einsum('nxg,xyg->nyg', rho1, fxc)
    wv *= weight
    wv = tools.fft(wv.reshape(-1,ngrids), mesh).reshape(nset,-1,*mesh)

    if with_j:
        coulG = tools.get_coulG(cell, mesh=mesh)
        vG = rhoG[:,0] * coulG
        vG = vG.reshape(nset, *mesh)
        wv[:,0] += vG

    if xctype == 'LDA':
        veff = _get_j_pass2(mydf, wv, hermi, kpts, verbose=log)

    elif xctype == 'GGA':
        # *.5 because v+v.T is always called in _get_gga_pass2
        wv[:,0] *= .5
        veff = _get_gga_pass2(mydf, wv, hermi, kpts, verbose=log)

    return veff.reshape(dm_kpts.shape)


def nr_rks_fxc_st(mydf, xc_code, dm0, dms_alpha, singlet=True,
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
    elif xctype == 'MGGA':
        deriv = 2 if MGGA_DENSITY_LAPL else 1
        raise NotImplementedError

    weight = cell.vol / ngrids
    if rho0 is None:
        rhoG = _eval_rhoG(mydf, dm0, 1, kpts, deriv)
        # *.5 to get alpha density
        rho0 = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (.5/weight)
        if xctype == 'LDA':
            rho0 = rho0.reshape(ngrids)
        rho0 = numpy.stack((rho0, rho0))

    if gamma_point(kpts):
        # implies real orbitals and real matrix, thus K_{ia,bj} = K_{ia,jb}
        # The output matrix v = K*x_{ia} is symmetric
        hermi = 1
    else:
        hermi = 0

    if fxc is None:
        fxc = ni.eval_xc_eff(xc_code, rho0, deriv=2, xctype=xctype)[2]
    if singlet:
        fxc = fxc[0,:,0] + fxc[0,:,1]
    else:
        fxc = fxc[0,:,0] - fxc[0,:,1]
    rhoG = _eval_rhoG(mydf, dms, hermi, kpts, deriv)
    rho1 = tools.ifft(rhoG.reshape(-1,ngrids), mesh)
    if hermi == 1:
        rho1 = rho1.real
    rho1 *= (1./weight)
    rho1 = rho1.reshape(nset,-1,ngrids)
    wv = numpy.einsum('nxg,xyg->nyg', rho1, fxc)
    wv *= weight
    wv = tools.fft(wv.reshape(-1,ngrids), mesh).reshape(nset,-1,*mesh)

    if xctype == 'LDA':
        veff = _get_j_pass2(mydf, wv, hermi, kpts, verbose=log)

    elif xctype == 'GGA':
        # *.5 because v+v.T is always called in _get_gga_pass2
        wv[:,0] *= .5
        veff = _get_gga_pass2(mydf, wv, hermi, kpts, verbose=log)

    return veff.reshape(dm_kpts.shape)


def nr_uks_fxc(mydf, xc_code, dm0, dms, hermi=0, with_j=False,
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
    nstates = nset // 2

    ni = mydf._numint
    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    elif xctype == 'MGGA':
        deriv = 2 if MGGA_DENSITY_LAPL else 1
        raise NotImplementedError

    weight = cell.vol / ngrids
    if rho0 is None:
        rhoG = _eval_rhoG(mydf, dm0, hermi, kpts, deriv)
        rho0 = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
        if xctype == 'LDA':
            rho0 = rho0.reshape(2,ngrids)
        else:
            rho0 = rho0.reshape(2,-1,ngrids)

    if fxc is None:
        fxc = ni.eval_xc_eff(xc_code, rho0, deriv=2, xctype=xctype)[2]

    rhoG = _eval_rhoG(mydf, dms, hermi, kpts, deriv)
    rho1 = tools.ifft(rhoG.reshape(-1,ngrids), mesh)
    if hermi == 1:
        rho1 = rho1.real
    rho1 *= (1./weight)
    # rho1 = (rho1a, rho1b); rho1.shape = (2, nstates, nvar, ngrids)
    rho1 = rho1.reshape(2,nstates,-1,ngrids)
    wv = numpy.einsum('anxg,axbyg->nbyg', rho1, fxc)
    wv *= weight
    wv = tools.fft(wv.reshape(-1,ngrids), mesh).reshape(nset,-1,*mesh)
    if with_j:
        coulG = tools.get_coulG(cell, mesh=mesh)
        vG = (rhoG[0,0] + rhoG[1,0]) * coulG
        vG = vG.reshape(mesh)
        wv[:,0] += vG

    if xctype == 'LDA':
        veff = _get_j_pass2(mydf, wv, hermi, kpts, verbose=log)

    elif xctype == 'GGA':
        # *.5 because v+v.T is always called in _get_gga_pass2
        wv[:,0] *= .5
        veff = _get_gga_pass2(mydf, wv, hermi, kpts, verbose=log)

    return veff.reshape(dm_kpts.shape)


def cache_xc_kernel(mydf, xc_code, mo_coeff, mo_occ, spin=0, kpts=None):
    raise NotImplementedError

def cache_xc_kernel1(mydf, xc_code, dm, spin=0, kpts=None):
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
    elif xctype == 'MGGA':
        deriv = 2 if MGGA_DENSITY_LAPL else 1
        comp = 6

    hermi = 1
    weight = cell.vol / ngrids
    rhoG = _eval_rhoG(mydf, dm, hermi, kpts, deriv)
    rho = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    rho = rho.reshape(rhoG.shape)

    n_dm, comp, ngrids = rho.shape
    if n_dm == 1 and spin == 1:
        rho = numpy.repeat(rho, 2, axis=0)
        rho *= .5

    if xctype == 'LDA':
        assert comp == 1
        rho = rho[:,0]
    else:
        assert comp > 1

    if spin == 0:
        assert n_dm == 1
        rho = rho[0]

    vxc, fxc = ni.eval_xc_eff(xc_code, rho, deriv=2, xctype=xctype)[1:3]
    return rho, vxc, fxc

def _gen_rhf_response(mf, dm0, singlet=None, hermi=0):
    '''multigrid version of function pbc.scf.newton_ah._gen_rhf_response
    '''
    #assert (isinstance(mf, dft.krks.KRKS))
    if isinstance(mf, KSCF):
        kpts = mf.kpts
    else:
        kpts = mf.kpt.reshape(1,3)

    if singlet is None:  # for newton solver
        rho0, vxc, fxc = cache_xc_kernel1(mf.with_df, mf.xc, dm0, 0, kpts)
    else:
        rho0, vxc, fxc = cache_xc_kernel1(mf.with_df, mf.xc, dm0, 1, kpts)
    dm0 = None

    def vind(dm1):
        if hermi == 2:
            return numpy.zeros_like(dm1)

        if singlet is None:  # Without specify singlet, general case
            v1 = nr_rks_fxc(mf.with_df, mf.xc, dm0, dm1, hermi,
                            True, rho0, vxc, fxc, kpts)
        elif singlet:
            v1 = nr_rks_fxc_st(mf.with_df, mf.xc, dm0, dm1, singlet,
                               rho0, vxc, fxc, kpts)
        else:
            v1 = nr_rks_fxc_st(mf.with_df, mf.xc, dm0, dm1, singlet,
                               rho0, vxc, fxc, kpts)
        return v1
    return vind

def _gen_uhf_response(mf, dm0, with_j=True, hermi=0):
    '''multigrid version of function pbc.scf.newton_ah._gen_uhf_response
    '''
    #assert (isinstance(mf, dft.kuks.KUKS))
    if isinstance(mf, KSCF):
        kpts = mf.kpts
    else:
        kpts = mf.kpt.reshape(1,3)

    rho0, vxc, fxc = cache_xc_kernel1(mf.with_df, mf.xc, dm0, 1, kpts)
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
        cell_dense = cell.copy(deep=False)
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
        cell_sparse = cell.copy(deep=False)
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

    # cell that needs dense integration grids
    def make_cell_dense_exp(shls_dense, ke0, ke1):
        cell_dense = cell.copy(deep=False)
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

    # cell that needs sparse integration grids
    def make_cell_sparse_exp(shls_sparse, ke0, ke1):
        cell_sparse = cell.copy(deep=False)
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

    tasks = []
    for ke0, ke1 in zip(ke_delimeter[:-1], ke_delimeter[1:]):
        # shells which have high exps (small rcut)
        shls_dense = [ib for ib, ke in enumerate(kecuts_pgto)
                      if numpy.any((ke0 < ke) & (ke <= ke1))]
        if len(shls_dense) == 0:
            continue

        mesh = tools.cutoff_to_mesh(a, ke1)
        if TO_EVEN_GRIDS:
            mesh = int((mesh+1)//2) * 2  # to the nearest even number

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

def _primitive_gto_cutoff(cell, precision=None):
    '''Cutoff radius, above which each shell decays to a value less than the
    required precision'''
    if precision is None:
        precision = cell.precision
    vol = cell.vol
    weight_penalty = vol
    precision = cell.precision / max(weight_penalty, 1)

    omega = cell.omega
    rcut = []
    ke_cutoff = []
    for ib in range(cell.nbas):
        l = cell.bas_angular(ib)
        es = cell.bas_exp(ib)
        cs = abs(cell._libcint_ctr_coeff(ib)).max(axis=1)
        norm_ang = ((2*l+1)/(4*numpy.pi))**.5
        fac = 2*numpy.pi/vol * cs*norm_ang/es / precision
        r = cell.rcut
        r = (numpy.log(fac * r**(l+1) + 1.) / es)**.5
        r = (numpy.log(fac * r**(l+1) + 1.) / es)**.5

        ke_guess = gto.cell._estimate_ke_cutoff(es, l, cs, precision, omega)
        rcut.append(r)
        ke_cutoff.append(ke_guess)
    return rcut, ke_cutoff


class MultiGridFFTDF(fft.FFTDF):
    _keys = {'tasks'}

    def __init__(self, cell, kpts=numpy.zeros((1,3))):
        fft.FFTDF.__init__(self, cell, kpts)
        self.tasks = None

    def build(self):
        self.tasks = multi_grids_tasks(self.cell, self.mesh, self.verbose)
        return self

    def reset(self, cell=None):
        self.tasks = None
        return fft.FFTDF.reset(cell)

    get_pp = get_pp
    get_nuc = get_nuc

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv='ewald'):
        if omega is not None:  # J/K for RSH functionals
            with self.range_coulomb(omega) as rsh_df:
                return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                                     omega=None, exxdiv=exxdiv)

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

    range_coulomb = aft.AFTDF.range_coulomb

    to_gpu = lib.to_gpu


def multigrid_fftdf(mf):
    '''Use MultiGridFFTDF to replace the default FFTDF integration method in
    the DFT object.
    '''
    mf.with_df, old_df = MultiGridFFTDF(mf.cell), mf.with_df
    mf.with_df.__dict__.update(old_df.__dict__)
    return mf

multigrid = multigrid_fftdf # for backward compatibility

def _pgto_shells(cell):
    return cell._bas[:,NPRIM_OF].sum()
