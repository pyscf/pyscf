#!/usr/bin/env python
# Copyright 2018-2019 The PySCF Developers. All Rights Reserved.
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
# Author: Peng Bao <baopeng@iccas.ac.cn>
#         Qiming Sun <osirpt.sun@gmail.com>
#

'''
semi-grid Coulomb and eXchange without differential density matrix

To lower the scaling of coulomb and exchange matrix construction for large system, one
coordinate is analytical and the other is grid. The traditional two electron
integrals turn to analytical one electron integrals and numerical integration
based on grid.(see Friesner, R. A. Chem. Phys. Lett. 1985, 116, 39)

Minimizing numerical errors using overlap fitting correction.(see
Lzsak, R. et. al. J. Chem. Phys. 2011, 135, 144105)
Grid screening for weighted AO value and DktXkg.
Two SCF steps: coarse grid then fine grid. There are 5 parameters can be changed:
# threshold for Xg and Fg screening
gthrd = 1e-10
# initial and final grids level
grdlvl_i = 0
grdlvl_f = 1
# norm_ddm threshold for grids change
thrd_nddm = 0.03
# set block size to adapt memory
sblk = 200

Set mf.direct_scf = False because no traditional 2e integrals
'''


import ctypes
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.df.incore import aux_e2
from pyscf.gto import moleintor
from pyscf.scf import _vhf
from pyscf.dft import gen_grid


def get_jk_favork(sgx, dm, hermi=1, with_j=True, with_k=True,
                  direct_scf_tol=1e-13):
    t0 = logger.process_clock(), logger.perf_counter()
    mol = sgx.mol
    grids = sgx.grids
    gthrd = sgx.grids_thrd

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]

    if sgx.debug:
        batch_nuc = _gen_batch_nuc(mol)
    else:
        batch_jk = _gen_jk_direct(mol, 's2', with_j, with_k, direct_scf_tol,
                                  sgx._opt, sgx.pjs)
    t1 = logger.timer_debug1(mol, "sgX initialization", *t0)

    sn = numpy.zeros((nao,nao))
    vj = numpy.zeros_like(dms)
    vk = numpy.zeros_like(dms)

    ngrids = grids.coords.shape[0]
    max_memory = sgx.max_memory - lib.current_memory()[0]
    sblk = sgx.blockdim
    blksize = min(ngrids, max(4, int(min(sblk, max_memory*1e6/8/nao**2))))
    tnuc = 0, 0
    for i0, i1 in lib.prange(0, ngrids, blksize):
        coords = grids.coords[i0:i1]
        weights = grids.weights[i0:i1,None]
        ao = mol.eval_gto('GTOval', coords)
        wao = ao * grids.weights[i0:i1,None]
        sn += lib.dot(ao.T, wao)

        fg = lib.einsum('gi,xij->xgj', wao, dms)
        mask = numpy.zeros(i1-i0, dtype=bool)
        for i in range(nset):
            mask |= numpy.any(fg[i]>gthrd, axis=1)
            mask |= numpy.any(fg[i]<-gthrd, axis=1)
        if not numpy.all(mask):
            ao = ao[mask]
            wao = wao[mask]
            fg = fg[:,mask]
            coords = coords[mask]
            weights = weights[mask]

        if sgx.debug:
            tnuc = tnuc[0] - logger.process_clock(), tnuc[1] - logger.perf_counter()
            gbn = batch_nuc(mol, coords)
            tnuc = tnuc[0] + logger.process_clock(), tnuc[1] + logger.perf_counter()
            if with_j:
                jg = numpy.einsum('gij,xij->xg', gbn, dms)
            if with_k:
                gv = lib.einsum('gvt,xgt->xgv', gbn, fg)
            gbn = None
        else:
            tnuc = tnuc[0] - logger.process_clock(), tnuc[1] - logger.perf_counter()
            jg, gv = batch_jk(mol, coords, dms, fg.copy(), weights)
            tnuc = tnuc[0] + logger.process_clock(), tnuc[1] + logger.perf_counter()

        if with_j:
            xj = lib.einsum('gv,xg->xgv', ao, jg)
            for i in range(nset):
                vj[i] += lib.einsum('gu,gv->uv', wao, xj[i])
        if with_k:
            for i in range(nset):
                vk[i] += lib.einsum('gu,gv->uv', ao, gv[i])
        jg = gv = None

    t2 = logger.timer_debug1(mol, "sgX J/K builder", *t1)
    tdot = t2[0] - t1[0] - tnuc[0] , t2[1] - t1[1] - tnuc[1]
    logger.debug1(sgx, '(CPU, wall) time for integrals (%.2f, %.2f); '
                  'for tensor contraction (%.2f, %.2f)',
                  tnuc[0], tnuc[1], tdot[0], tdot[1])

    ovlp = mol.intor_symmetric('int1e_ovlp')
    proj = scipy.linalg.solve(sn, ovlp)

    if with_j:
        vj = lib.einsum('pi,xpj->xij', proj, vj)
        vj = (vj + vj.transpose(0,2,1))*.5
    if with_k:
        vk = lib.einsum('pi,xpj->xij', proj, vk)
        if hermi == 1:
            vk = (vk + vk.transpose(0,2,1))*.5
    logger.timer(mol, "vj and vk", *t0)
    return vj.reshape(dm_shape), vk.reshape(dm_shape)


def get_jk_favorj(sgx, dm, hermi=1, with_j=True, with_k=True,
                  direct_scf_tol=1e-13):
    t0 = logger.process_clock(), logger.perf_counter()
    mol = sgx.mol
    grids = sgx.grids
    gthrd = sgx.grids_thrd

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]

    if sgx.debug:
        batch_nuc = _gen_batch_nuc(mol)
    else:
        batch_jk = _gen_jk_direct(mol, 's2', with_j, with_k, direct_scf_tol,
                                  sgx._opt, sgx.pjs)

    sn = numpy.zeros((nao,nao))
    ngrids = grids.coords.shape[0]
    max_memory = sgx.max_memory - lib.current_memory()[0]
    sblk = sgx.blockdim
    blksize = min(ngrids, max(4, int(min(sblk, max_memory*1e6/8/nao**2))))
    for i0, i1 in lib.prange(0, ngrids, blksize):
        coords = grids.coords[i0:i1]
        ao = mol.eval_gto('GTOval', coords)
        wao = ao * grids.weights[i0:i1,None]
        sn += lib.dot(ao.T, wao)

    ovlp = mol.intor_symmetric('int1e_ovlp')
    proj = scipy.linalg.solve(sn, ovlp)
    proj_dm = lib.einsum('ki,xij->xkj', proj, dms)

    t1 = logger.timer_debug1(mol, "sgX initialization", *t0)
    vj = numpy.zeros_like(dms)
    vk = numpy.zeros_like(dms)
    tnuc = 0, 0
    for i0, i1 in lib.prange(0, ngrids, blksize):
        coords = grids.coords[i0:i1]
        weights = grids.weights[i0:i1,None]
        ao = mol.eval_gto('GTOval', coords)
        wao = ao * grids.weights[i0:i1,None]

        fg = lib.einsum('gi,xij->xgj', wao, proj_dm)
        mask = numpy.zeros(i1-i0, dtype=bool)
        for i in range(nset):
            mask |= numpy.any(fg[i]>gthrd, axis=1)
            mask |= numpy.any(fg[i]<-gthrd, axis=1)
        if not numpy.all(mask):
            ao = ao[mask]
            fg = fg[:,mask]
            coords = coords[mask]
            weights = weights[mask]

        if with_j:
            rhog = numpy.einsum('xgu,gu->xg', fg, ao)
        else:
            rhog = None

        if sgx.debug:
            tnuc = tnuc[0] - logger.process_clock(), tnuc[1] - logger.perf_counter()
            gbn = batch_nuc(mol, coords)
            tnuc = tnuc[0] + logger.process_clock(), tnuc[1] + logger.perf_counter()
            if with_j:
                jpart = numpy.einsum('guv,xg->xuv', gbn, rhog)
            if with_k:
                gv = lib.einsum('gtv,xgt->xgv', gbn, fg)
            gbn = None
        else:
            tnuc = tnuc[0] - logger.process_clock(), tnuc[1] - logger.perf_counter()
            if with_j: rhog = rhog.copy()
            jpart, gv = batch_jk(mol, coords, rhog, fg.copy(), weights)
            tnuc = tnuc[0] + logger.process_clock(), tnuc[1] + logger.perf_counter()

        if with_j:
            vj += jpart
        if with_k:
            for i in range(nset):
                vk[i] += lib.einsum('gu,gv->uv', ao, gv[i])
        jpart = gv = None

    t2 = logger.timer_debug1(mol, "sgX J/K builder", *t1)
    tdot = t2[0] - t1[0] - tnuc[0] , t2[1] - t1[1] - tnuc[1]
    logger.debug1(sgx, '(CPU, wall) time for integrals (%.2f, %.2f); '
                  'for tensor contraction (%.2f, %.2f)',
                  tnuc[0], tnuc[1], tdot[0], tdot[1])

    for i in range(nset):
        lib.hermi_triu(vj[i], inplace=True)
    if with_k and hermi == 1:
        vk = (vk + vk.transpose(0,2,1))*.5
    logger.timer(mol, "vj and vk", *t0)
    return vj.reshape(dm_shape), vk.reshape(dm_shape)

def _gen_batch_nuc(mol):
    '''Coulomb integrals of the given points and orbital pairs'''
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, 'int3c2e')
    def batch_nuc(mol, grid_coords, out=None):
        fakemol = gto.fakemol_for_charges(grid_coords)
        j3c = aux_e2(mol, fakemol, intor='int3c2e', aosym='s2ij', cintopt=cintopt)
        return lib.unpack_tril(j3c.T, out=out)
    return batch_nuc

def _gen_jk_direct(mol, aosym, with_j, with_k, direct_scf_tol, sgxopt=None, pjs=False):
    '''Contraction between sgX Coulomb integrals and density matrices

    J: einsum('guv,xg->xuv', gbn, dms) if dms == rho at grid,
    or einsum('gij,xij->xg', gbn, dms) if dms are density matrices

    K: einsum('gtv,xgt->xgv', gbn, fg)
    '''
    if sgxopt is None:
        from pyscf.sgx import sgx
        sgxopt = sgx._make_opt(mol, pjs, direct_scf_tol)
    sgxopt.direct_scf_tol = direct_scf_tol

    ncomp = 1
    nao = mol.nao
    cintor = _vhf._fpointer(sgxopt._intor)
    fdot = _vhf._fpointer('SGXdot_nrk')
    drv = _vhf.libcvhf.SGXnr_direct_drv

    def jk_part(mol, grid_coords, dms, fg, weights):
        atm, bas, env = mol._atm, mol._bas, mol._env
        ngrids = grid_coords.shape[0]
        env = numpy.append(env, grid_coords.ravel())
        env[gto.NGRIDS] = ngrids
        env[gto.PTR_GRIDS] = mol._env.size
        if pjs:
            sgxopt.set_dm(fg / numpy.sqrt(numpy.abs(weights[None,:])),
                          mol._atm, mol._bas, env)

        ao_loc = moleintor.make_loc(bas, sgxopt._intor)
        shls_slice = (0, mol.nbas, 0, mol.nbas)

        fg = numpy.ascontiguousarray(fg.transpose(0,2,1))

        vj = vk = None
        fjk = []
        dmsptr = []
        vjkptr = []
        if with_j:
            if dms[0].ndim == 1:  # the value of density at each grid
                vj = numpy.zeros((len(dms),ncomp,nao,nao))[:,0]
                for i, dm in enumerate(dms):
                    dmsptr.append(dm.ctypes.data_as(ctypes.c_void_p))
                    vjkptr.append(vj[i].ctypes.data_as(ctypes.c_void_p))
                    fjk.append(_vhf._fpointer('SGXnr'+aosym+'_ijg_g_ij'))
            else:
                vj = numpy.zeros((len(dms),ncomp,ngrids))[:,0]
                for i, dm in enumerate(dms):
                    dmsptr.append(dm.ctypes.data_as(ctypes.c_void_p))
                    vjkptr.append(vj[i].ctypes.data_as(ctypes.c_void_p))
                    fjk.append(_vhf._fpointer('SGXnr'+aosym+'_ijg_ji_g'))
        if with_k:
            vk = numpy.zeros((len(fg),ncomp,nao,ngrids))[:,0]
            for i, dm in enumerate(fg):
                dmsptr.append(dm.ctypes.data_as(ctypes.c_void_p))
                vjkptr.append(vk[i].ctypes.data_as(ctypes.c_void_p))
                fjk.append(_vhf._fpointer('SGXnr'+aosym+'_ijg_gj_gi'))

        n_dm = len(fjk)
        fjk = (ctypes.c_void_p*(n_dm))(*fjk)
        dmsptr = (ctypes.c_void_p*(n_dm))(*dmsptr)
        vjkptr = (ctypes.c_void_p*(n_dm))(*vjkptr)

        drv(cintor, fdot, fjk, dmsptr, vjkptr, n_dm, ncomp,
            (ctypes.c_int*4)(*shls_slice),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            sgxopt._cintopt, ctypes.byref(sgxopt._this),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
            env.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(env.shape[0]),
            ctypes.c_int(2 if aosym == 's2' else 1))
        if vk is not None:
            vk = vk.transpose(0,2,1)
            vk = numpy.ascontiguousarray(vk)
        return vj, vk
    return jk_part


# pre for get_k
# Use default mesh grids and weights
def get_gridss(mol, level=1, gthrd=1e-10):
    Ktime = (logger.process_clock(), logger.perf_counter())
    grids = gen_grid.Grids(mol)
    grids.level = level
    grids.build()

    ngrids = grids.weights.size
    mask = []
    for p0, p1 in lib.prange(0, ngrids, 10000):
        ao_v = mol.eval_gto('GTOval', grids.coords[p0:p1])
        ao_v *= grids.weights[p0:p1,None]
        wao_v0 = ao_v
        mask.append(numpy.any(wao_v0>gthrd, axis=1) |
                    numpy.any(wao_v0<-gthrd, axis=1))

    mask = numpy.hstack(mask)
    grids.coords = grids.coords[mask]
    grids.weights = grids.weights[mask]
    logger.debug(mol, 'threshold for grids screening %g', gthrd)
    logger.debug(mol, 'number of grids %d', grids.weights.size)
    logger.timer_debug1(mol, "Xg screening", *Ktime)
    return grids

get_jk = get_jk_favorj
