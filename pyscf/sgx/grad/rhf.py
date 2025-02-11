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
from pyscf.sgx import sgx, sgx_jk
from pyscf.sgx.sgx_jk import _gen_jk_direct, run_k_only_setup, BLKSIZE, SGX_BLKSIZE
from pyscf.sgx.sgx_jk import _gen_batch_nuc, _gen_batch_nuc_grad
from pyscf.grad.rks import grids_response_cc, get_dw_partition_sorted


def get_jk_grad(sgx, dm, hermi=1, with_j=True, with_k=True,
                direct_scf_tol=1e-13):
    t0 = logger.process_clock(), logger.perf_counter()
    mol = sgx.mol
    grids = sgx.grids
    grids.build()
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
                                  sgx._opt)

    if include_grid_response:
        if sgx.debug:
            batch_nuc_grad = _gen_batch_nuc_grad(mol)
        else:
            batch_jk_grad = _gen_jk_direct(mol, 's1', with_j, with_k, direct_scf_tol,
                                           None, grad=True)
            if with_j:
                batch_jonly = _gen_jk_direct(mol, 's1', True, False, direct_scf_tol,
                                             None, grad=True)

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
                jg, gv = batch_jk(mol, coords, dms, fg.transpose(0, 2, 1).copy(), weights)
                if include_grid_response:
                    djpart_int, dgv = batch_jk_grad(mol, coords, rhog, fg.transpose(0, 2, 1).copy(), weights)
                    if with_j:
                        djg, _ = batch_jonly(mol, coords, dms, None, weights)
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
                    vk[i] += lib.einsum('gu,gv->uv', ao, gv[i].T)
                    if not include_grid_response:
                        dvk_tmp[i] -= 1.0 * lib.einsum('xgu,gv->xuv', dao, gv[i].T) # ORBITAL RESPONSE
                    else:
                        dvk_tmp[i] -= 0.5 * lib.einsum('xgu,gv->xuv', dao, gv[i].T) # ORBITAL RESPONSE
                        dvk_tmp[i] -= 0.5 * lib.einsum('xug,gv->xuv', dgv[i], ao) # INTEGRAL RESPONSE
                        xed = lib.einsum(
                            'gu,gu->g',
                            fg0_no_w[i],
                            gv[i].T/(weights + 1e-200),
                        )
                        dek[:,:] += numpy.dot(weights1, xed) # GRID RESPONSE PART 1

            gv = None

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

    vj, vk = dvj, dvk
    logger.timer(mol, "vj and vk", *t0)
    dm_shape = (nset, 3) + dms.shape[1:]
    vk = vk.reshape(dm_shape)
    vj = vj.reshape(dm_shape)
    if restricted:
        vj, vk = vj[0], vk[0]
    vj = lib.tag_array(vj, aux=0.5*dej[None, None])
    vk = lib.tag_array(vk, aux=0.5*dek[None, None])
    return vj, vk


def scalable_grids_response_setup(grids, mol=None):
    import time
    t0 = time.monotonic()
    if mol is None: mol = grids.mol
    if grids.verbose >= logger.WARN:
        grids.check_sanity()
    atom_grids_tab = grids.gen_atomic_grids(
        mol, grids.atom_grid, grids.radi_method, grids.level, grids.prune
    )
    coords_all = []
    ialist = []
    for ia in range(mol.natm):
        coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
        coords_all.append(coords)
        ialist.append(numpy.repeat(numpy.int32(ia), vol.size))
    coords_all = numpy.vstack(coords_all)
    ialist = numpy.hstack(ialist)
    idx = gen_grid.arg_group_grids(mol, coords)
    coords = coords[idx]
    ialist = ialist[idx]
    if grids.alignment > 1:
        padding = gen_grid._padding_size(grids.size, grids.alignment)
        logger.debug(grids, 'Padding %d grids', padding)
        if padding > 0:
            coords = numpy.vstack(
                [coords, numpy.repeat([[1e-4]*3], padding, axis=0)])
            ialist = numpy.hstack([ialist, numpy.repeat(-1, padding)])
    return coords, ialist


def scalable_grids_response_cc(grids, blksize):
    assert grids.ialist is not None
    # if grids.coords is None or grids.non0tab is None or grids.ialist is None:
    #     grids.build(with_non0tab=True, with_ialist=True)
    #     sgx._sgx_block_cond = None
    mol = grids.mol
    ngrids = grids.weights.size
    assert blksize % BLKSIZE == 0 or blksize == grids.weights.size
    for i0, i1 in lib.prange(0, ngrids, blksize):
        coords = grids.coords[i0:i1]
        weights = grids.weights[i0:i1]
        ialist = grids.ialist[i0:i1]
        mask = grids.screen_index[i0 // BLKSIZE :]
        weights1 = get_dw_partition_sorted(mol, ialist, coords, weights,
                                           grids.radii_adjust,
                                           grids.atomic_radii)
        atom_mask = (ialist == -1)
        # make sure weights are zero in the padded region
        weights[atom_mask] = 0
        weights1[..., atom_mask] = 0
        yield i0, i1, ialist, coords, weights, weights1.transpose(1, 0, 2), mask


def get_k_grad_only(sgx, dm, hermi=1, direct_scf_tol=1e-13):
    t00 = logger.process_clock(), logger.perf_counter()
    t0 = t00[1]
    if sgx.debug:
        raise NotImplementedError("debug mode for accelerated K matrix")

    mol = sgx.mol
    grids = sgx.grids
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
    shls_slice = (0, mol.nbas)
    sgx_data = run_k_only_setup(sgx, dms, hermi)
    ngrids = grids.weights.size
    blksize, screen_index, proj_dm, dm_mask, ncond, ncond_ni, ao_loc = sgx_data

    vk = numpy.zeros_like(dms)
    dvk = numpy.zeros((dms.shape[0], 3,) + dms.shape[1:])
    dek = numpy.zeros((mol.natm, 3)) # derivs wrt atom positions
    xed = numpy.zeros((nset, grids.weights.size)) # works if grids are not screened initially

    batch_k = sgx_jk._gen_k_direct(mol, 's2', direct_scf_tol, sgx._opt)
    if include_grid_response:
        batch_k_grad = sgx_jk._gen_k_direct(mol, 's1', direct_scf_tol, None, grad=True)
    t1 = logger.perf_counter()

    t2, t3 = 0, 0
    t4 = -logger.perf_counter()
    fg = None
    for i0, i1, ialist, coords, weights, weights1, mask in scalable_grids_response_cc(grids, blksize):
        # iatoms is an array shape (ngrids,)
        # coords is an array shape (ngrids, 3)
        # weights is an array shape (ngrids)
        # weights1 is an array shape (natm, 3, ngrids)
        if mol.cart:
            _ao = mol.eval_gto('GTOval_cart_deriv1', coords, non0tab=mask)
        else:
            _ao = mol.eval_gto('GTOval_sph_deriv1', coords, non0tab=mask)
        ao = _ao[0]
        dao = _ao[1:4]
        wao = ao * weights[:, None]
        wdao = dao * weights[:, None]
        ta = logger.perf_counter()
        if sgx.use_dm_screening:
            fg = sgx_jk._sgxdot_ao_dm_sparse(ao, proj_dm, mask, dm_mask, ao_loc, out=fg)
            gv = batch_k(mol, coords, fg, weights, ncond[i0 // SGX_BLKSIZE :])
            if include_grid_response:
                dgv = batch_k_grad(mol, coords, fg, weights, ncond[i0 // SGX_BLKSIZE :])
            shl_norms = sgx_jk._get_shell_norms(gv, weights, ao_loc)
            mask2 = shl_norms > ncond_ni[i0 // BLKSIZE : i0 // BLKSIZE + shl_norms.shape[0], None]
        else:
            fg = sgx_jk._sgxdot_ao_dm(ao, proj_dm, mask, shls_slice, ao_loc, out=fg)
            gv = batch_k(mol, coords, fg, weights)
            if include_grid_response:
                dgv = batch_k_grad(mol, coords, fg, weights)
            mask2 = None
        sgx_jk._sgxdot_ao_gv_sparse(ao, gv, weights, mask, mask2, ao_loc, out=vk)
        tb = logger.perf_counter()

        dxed = 0
        for i in range(nset):
            if not include_grid_response:
                dvk[i] -= lib.einsum('xgu,gv->xuv', wdao, gv[i].T) # ORBITAL RESPONSE
            else:
                dvk[i] -= 0.5 * lib.einsum('xgu,gv->xuv', wdao, gv[i].T) # ORBITAL RESPONSE
                dvk[i] -= 0.5 * lib.einsum('xug,gv->xuv', dgv[i], wao) # INTEGRAL RESPONSE
                xed = lib.einsum('ug,ug->g', fg[i], gv[i])
                dxed += lib.einsum('ug,xug->xg', fg[i], dgv[i]) # INTEGRAL RESPONSE
                tmp = lib.einsum("vg,vu->ug", gv[i], dms[i])
                dxed += lib.einsum("ug,xgu->xg", tmp, dao)
                dek[:, :] += numpy.dot(weights1, xed) # GRID RESPONSE PART 1
        dxed *= -0.5 * weights

        if include_grid_response:
            for i in range(nset):
                # dek[ia] -= 4 * (dvk_tmp[i] * dms[i]).sum(axis=(1,2)) # GRID RESPONSE PART 2
                for ia in range(mol.natm):
                    cond = (ialist == ia)
                    dek[ia] -= 4 * numpy.sum(dxed[:, cond], axis=1)  # GRID RESPONSE PART 2
        tc = logger.perf_counter()
        t2 += tb - ta
        t3 += tc - tb
    t4 += logger.perf_counter()
    print("TIMES", t1 - t0, t2, t3, t4 - t3 - t2)

    # TODO don't need to compute vk
    vk = dvk
    logger.timer(mol, "vk", *t00)
    dm_shape = (nset, 3) + dms.shape[1:]
    vk = vk.reshape(dm_shape)
    if restricted:
        vk = vk[0]
    vk = lib.tag_array(vk, aux=0.5*dek[None, None])
    return vk


def _get_jk(sgx, dm, hermi, with_j, with_k, direct_scf_tol):
    if with_j and sgx.dfj:
        vj = None
        if with_k:
            if sgx.optk and sgx.grids.becke_scheme == gen_grid.becke_lko:
                vk = get_k_grad_only(sgx, dm, hermi, direct_scf_tol)
            else:
                vk = get_jk_grad(sgx, dm, hermi, False, with_k, direct_scf_tol)[1]
        else:
            vk = None
    else:
        vj, vk = get_jk_grad(sgx, dm, hermi, with_j, with_k, direct_scf_tol)
    return vj, vk


def get_jk(mf_grad, mol=None, dm=None, hermi=1, vhfopt=None, with_j=True, with_k=True,
           direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13),
           omega=None):
    sgx = mf_grad.base.with_df
    dm = mf_grad.base.make_rdm1()
    if with_j and mf_grad.base.with_df.dfj:
        vj = super(mf_grad.__class__, mf_grad).get_jk(
            mf_grad.mol, dm, hermi, with_j=True, with_k=False)[0]
    else:
        vj = None

    if omega is not None:
        # A temporary treatment for RSH integrals
        key = '%.6f' % omega
        if key in sgx._rsh_df:
            rsh_df = sgx._rsh_df[key]
        else:
            rsh_df = sgx.copy()
            rsh_df._rsh_df = None  # to avoid circular reference
            # Not all attributes need to be reset. Resetting _vjopt
            # because it is used by get_j method of regular DF object.
            rsh_df._vjopt = None
            sgx._rsh_df[key] = rsh_df
            logger.info(sgx, 'Create RSH-SGX object %s for omega=%s', rsh_df, omega)

        with rsh_df.mol.with_range_coulomb(omega):
            rsh_df.grid_response = mf_grad.sgx_grid_response
            _vj, vk = _get_jk(rsh_df, dm, hermi, with_j, with_k, direct_scf_tol)
    else:
        sgx.grid_response = mf_grad.sgx_grid_response
        _vj, vk = _get_jk(sgx, dm, hermi, with_j, with_k, direct_scf_tol)

    if vj is None:
        vj = _vj
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

    def check_sanity(self):
        assert isinstance(self.base, sgx._SGXHF)

    get_jk = get_jk

Grad = Gradients
