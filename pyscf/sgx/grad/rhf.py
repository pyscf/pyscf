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
# Author: Kyle Bystrom <kylebystrom@gmail.com>
#


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
from pyscf.df.grad import rhf as dfrhf_grad
from pyscf import __config__
from pyscf.sgx.sgx_jk import _gen_jk_direct
from pyscf.sgx.sgx_jk import _gen_batch_nuc, _gen_batch_nuc_grad
from pyscf.grad.rks import grids_response_cc


def get_jk_favorj(sgx, dm, hermi=1, with_j=True, with_k=True,
                  direct_scf_tol=1e-13):
    t0 = logger.process_clock(), logger.perf_counter()
    mol = sgx.mol
    grids = sgx.grids
    grids.build()
    gthrd = sgx.grids_thrd
    include_grid_response = sgx.grid_response

    if dm.ndim == 2:
        restricted = True
    else:
        restricted = False
    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]
    if include_grid_response and nset > 2:
        raise ValueError('Cannot handle multiple DMs for grid response')

    if sgx.debug:
        batch_nuc = _gen_batch_nuc(mol)
    else:
        batch_jk = _gen_jk_direct(mol, 's2', with_j, with_k, direct_scf_tol,
                                  sgx._opt, sgx.pjs)

    if include_grid_response:
        if sgx.debug:
            batch_nuc_grad = _gen_batch_nuc_grad(mol)
        else:
            batch_jk_grad = _gen_jk_direct(mol, 's1', with_j, with_k, direct_scf_tol,
                                           None, sgx.pjs, grad=True)
            if with_j:
                batch_jonly = _gen_jk_direct(mol, 's1', True, False, direct_scf_tol,
                                             None, sgx.pjs, grad=True)

    dej = numpy.zeros((mol.natm, 3)) # derivs wrt atom positions
    dek = numpy.zeros((mol.natm, 3)) # derivs wrt atom positions
    sn = numpy.zeros((nao,nao))
    ngrids = grids.coords.shape[0]
    max_memory = sgx.max_memory - lib.current_memory()[0]
    sblk = sgx.blockdim
    blksize = min(ngrids, max(4, int(min(sblk, max_memory*1e6/8/nao**2))))

    if sgx.fit_ovlp:
        for i0, i1 in lib.prange(0, ngrids, blksize):
            coords = grids.coords[i0:i1]
            ao = mol.eval_gto('GTOval', coords)
            wao = ao * grids.weights[i0:i1,None]
            sn += lib.dot(ao.T, wao)
        ovlp = mol.intor_symmetric('int1e_ovlp')
        proj = scipy.linalg.solve(sn, ovlp)
        proj_dm = lib.einsum('ki,xij->xkj', proj, dms)
    else:
        proj_dm = dms.copy()

    t1 = logger.timer_debug1(mol, "sgX initialziation", *t0)
    vj = numpy.zeros_like(dms)
    dvj = numpy.zeros((dms.shape[0], 3,) + dms.shape[1:])
    vk = numpy.zeros_like(dms)
    dvk = numpy.zeros((dms.shape[0], 3,) + dms.shape[1:])
    tnuc = 0, 0
    xed = numpy.zeros((nset, grids.weights.size)) # works if grids are not screened initially

    if include_grid_response and with_j:
        dm_sum = dms.sum(axis=0)

    for ia, (coord_ia, weight_ia, weight1_ia) in enumerate(grids_response_cc(grids)):
        ngrids = weight_ia.size
        dvj_tmp = numpy.zeros_like(dvk)
        dvk_tmp = numpy.zeros_like(dvk)
        for i0, i1 in lib.prange(0, ngrids, blksize):
            weights1 = weight1_ia[...,i0:i1]
            weights = weight_ia[i0:i1,None]
            coords = coord_ia[i0:i1]
            if mol.cart:
                _ao = mol.eval_gto('GTOval_cart_deriv1', coords)
            else:
                _ao = mol.eval_gto('GTOval_sph_deriv1', coords)
            ao = _ao[0]
            dao = _ao[1:4]
            wao = ao * weights

            fg = lib.einsum('gi,xij->xgj', wao, proj_dm)
            if include_grid_response:
                fg0_no_w = lib.einsum('gi,xij->xgj', ao, dms)
            mask = numpy.zeros(i1-i0, dtype=bool)
            for i in range(nset):
                mask |= numpy.any(fg[i]>gthrd, axis=1)
                mask |= numpy.any(fg[i]<-gthrd, axis=1)
            if not numpy.all(mask):
                ao = ao[mask]
                wao = wao[mask]
                dao = dao[:,mask]
                fg = fg[:,mask]
                if include_grid_response:
                    fg0_no_w = fg0_no_w[:,mask]
                coords = coords[mask]
                weights = weights[mask]
                weights1 = weights1[...,mask]

            if with_j:
                rhog = numpy.einsum('xgu,gu->xg', fg, ao)
                drhog_sum = numpy.einsum('xgu,ngu->ng', fg, dao)
                rhog_sum = rhog.sum(axis=0)
            else:
                rhog = None
                rhog_sum = None

            if sgx.debug:
                tnuc = tnuc[0] - logger.process_clock(), tnuc[1] - logger.perf_counter()
                gbn = batch_nuc(mol, coords)
                tnuc = tnuc[0] + logger.process_clock(), tnuc[1] + logger.perf_counter()
                if include_grid_response:
                    tnuc = tnuc[0] - logger.process_clock(), tnuc[1] - logger.perf_counter()
                    dgbn = batch_nuc_grad(mol, coords)
                    tnuc = tnuc[0] + logger.process_clock(), tnuc[1] + logger.perf_counter()
                if with_j:
                    jpart = lib.einsum('guv,xg->xuv', gbn, rhog)
                    jg = lib.einsum('guv,xuv->xg', gbn, dms)
                    if include_grid_response:
                        djpart_int = lib.einsum('nguv,xg->xnuv', dgbn, rhog)
                        djg = lib.einsum('nguv,xuv->xng', dgbn, dms)
                if with_k:
                    gv = lib.einsum('gtv,xgt->xgv', gbn, fg)
                    if include_grid_response:
                        dgv = lib.einsum('ngvt,xgt->xngv', dgbn, fg)
                gbn = None
            else:
                tnuc = tnuc[0] - logger.process_clock(), tnuc[1] - logger.perf_counter()
                if with_j: rhog = rhog.copy()
                jg, gv = batch_jk(mol, coords, dms, fg.copy(), weights)
                if include_grid_response:
                    djpart_int, dgv = batch_jk_grad(mol, coords, rhog, fg.copy(), weights)
                    if with_j:
                        djg, _ = batch_jonly(mol, coords, dms, fg.copy(), weights)
                tnuc = tnuc[0] + logger.process_clock(), tnuc[1] + logger.perf_counter()

            # TODO for j-part, orbital response is not quite right
            # because of the projected density matrix. (Used 0 times for j orb response, should be used once each)
            if with_j:
                for i in range(nset):
                    xj = lib.einsum('ngv,g->ngv', dao, jg[i])
                    if not include_grid_response:
                        dvj_tmp[i] -= lib.einsum('xgu,gv->xuv', xj, wao) # ORBITAL RESPONSE
                    else:
                        dvj_tmp[i] -= 0.5 * lib.einsum('xgu,gv->xuv', xj, wao) # ORBITAL RESPONSE
                        dvj_tmp[i] -= 0.5 * djpart_int[i] # INTEGRAL RESPONSE
                        dej[:,:] += numpy.dot(weights1, jg[i]*(rhog_sum/(weights[:,0]+1e-16))) # GRID RESPONSE PART 1
                        dej[ia] += 2 * numpy.einsum('xg,g->x', djg[i], rhog_sum)
                        dej[ia] += 2 * numpy.einsum('xg,g->x', drhog_sum, jg[i])

            if with_k:
                for i in range(nset):
                    vk[i] += lib.einsum('gu,gv->uv', ao, gv[i])
                    if not include_grid_response:
                        dvk_tmp[i] -= 1.0 * lib.einsum('xgu,gv->xuv', dao, gv[i]) # ORBITAL RESPONSE
                    else:
                        dvk_tmp[i] -= 0.5 * lib.einsum('xgu,gv->xuv', dao, gv[i]) # ORBITAL RESPONSE
                        dvk_tmp[i] -= 0.5 * lib.einsum('xgu,gv->xuv', dgv[i], ao) # INTEGRAL RESPONSE
                        xed = lib.einsum(
                            'gu,gu->g',
                            fg0_no_w[i],
                            gv[i]/(weights + 1e-200),
                        )
                        dek[:,:] += numpy.dot(weights1, xed) # GRID RESPONSE PART 1

            jpart = gv = None

        dvj += dvj_tmp
        dvk += dvk_tmp
        if include_grid_response:
            for i in range(nset):
                if with_k:
                    dek[ia] -= 4 * (dvk_tmp[i] * dms[i]).sum(axis=(1,2)) # GRID RESPONSE PART 2

    t2 = logger.timer_debug1(mol, "sgX J/K builder", *t1)
    tdot = t2[0] - t1[0] - tnuc[0] , t2[1] - t1[1] - tnuc[1]
    logger.debug1(sgx, '(CPU, wall) time for integrals (%.2f, %.2f); '
                  'for tensor contraction (%.2f, %.2f)',
                  tnuc[0], tnuc[1], tdot[0], tdot[1])

    #for i in range(nset):
    #    lib.hermi_triu(vj[i], inplace=True)
    vj, vk = dvj, dvk
    logger.timer(mol, "vj and vk", *t0)
    dm_shape = (nset, 3) + dms.shape[1:]
    vk = vk.reshape(dm_shape)
    vj = vj.reshape(dm_shape)
    if restricted:
        vj, vk = vj[0], vk[0]
    vj = lib.tag_array(vj, aux=0.5*dej)
    vk = lib.tag_array(vk, aux=0.5*dek)
    return vj, vk


def get_jk(mf_grad, mol=None, dm=None, hermi=1, vhfopt=None, with_j=True, with_k=True,
           direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13),
           omega=None):
    if omega is not None:
        raise NotImplementedError('No range-separated nuclear gradients')

    dm = mf_grad.base.make_rdm1()
    if with_j and mf_grad.base.with_df.dfj:
        vj = super(mf_grad.__class__, mf_grad).get_jk(
            mf_grad.mol, dm, hermi, with_j=True, with_k=False)[0]
        if with_k:
            mf_grad.base.with_df.grid_response = mf_grad.sgx_grid_response
            vk = get_jk_favorj(mf_grad.base.with_df, dm, hermi, False, with_k, direct_scf_tol)[1]
        else:
            vk = None
    else:
        mf_grad.base.with_df.grid_response = mf_grad.sgx_grid_response
        vj, vk = get_jk_favorj(mf_grad.base.with_df, dm, hermi, with_j, with_k, direct_scf_tol)
    return vj, vk


class Gradients(dfrhf_grad.Gradients):
    '''Restricted SGX Hartree-Fock gradients. Note: sgx_grid_response determines
    whether full grid response for SGX terms is computed.'''
    def __init__(self, mf):
        # Whether to include the response of DF auxiliary basis when computing
        # nuclear gradients of J/K matrices
        self.sgx_grid_response = True
        if mf.with_df.direct_j:
            raise ValueError("direct_j setting not supported for gradients")
        dfrhf_grad.Gradients.__init__(self, mf)

    get_jk = get_jk

Grad = Gradients
