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
from pyscf.dft import gen_grid
from pyscf.df.grad import rhf as dfrhf_grad
from pyscf import __config__
from pyscf.sgx import sgx, sgx_jk
from pyscf.sgx.sgx_jk import BLKSIZE, SGX_BLKSIZE
from pyscf.grad.rks import grids_response_cc, get_dw_partition_sorted
from pyscf.dft import numint as ni


def _gen_batch_nuc_grad(mol):
    '''Coulomb integrals of the given points and orbital pairs'''
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, 'int3c2e_ip1')
    def batch_nuc(mol, grid_coords, out=None):
        fakemol = gto.fakemol_for_charges(grid_coords)
        j3c = aux_e2(mol, fakemol, intor='int3c2e_ip1', aosym='s1', cintopt=cintopt)
        return j3c.transpose(0,3,1,2)
    return batch_nuc


def get_jk_grad(sgx, dm, hermi=1, with_j=True, with_k=True,
                direct_scf_tol=1e-13):
    t0 = logger.process_clock(), logger.perf_counter()
    mol = sgx.mol
    grids = sgx.grids
    grids.build()
    include_grid_response = sgx._grid_response

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
        batch_nuc = sgx_jk._gen_batch_nuc(mol)
    else:
        batch_jk = sgx_jk._gen_jk_direct(
            mol, 's2', with_j, with_k, direct_scf_tol, sgx._opt
        )

    if include_grid_response:
        if sgx.debug:
            batch_nuc_grad = _gen_batch_nuc_grad(mol)
        else:
            batch_jk_grad = sgx_jk._gen_jk_direct(
                mol, 's1', with_j, with_k, direct_scf_tol, None, grad=True
            )
            if with_j:
                batch_jonly = sgx_jk._gen_jk_direct(
                    mol, 's1', True, False, direct_scf_tol, None, grad=True
                )

    dej = numpy.zeros((mol.natm, 3)) # derivs wrt atom positions
    dek = numpy.zeros((mol.natm, 3)) # derivs wrt atom positions
    sn = numpy.zeros((nao,nao))
    ngrids = grids.coords.shape[0]
    max_memory = sgx.max_memory - lib.current_memory()[0]
    sblk = sgx.blockdim
    blksize = min(ngrids, max(4, int(min(sblk, max_memory*1e6/8/nao**2))))

    if sgx.fit_ovlp:
        if include_grid_response:
            msg = (
                "SGX-JK neglects the overlap fitting contribution to the "
                "gradient, which can cause significant errors. For exact "
                "gradients, set fit_ovlp=False OR fit_ovlp=dfj=optk=True."
            )
            logger.warn(sgx, msg)
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

    jtotal = 0
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
                rhog = lib.einsum('xgu,gu->xg', fg, ao)
                drhog_sum = lib.einsum('xgu,ngu->ng', fg, dao)
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
                    gv = lib.einsum('gtv,xgt->xvg', gbn, fg)
                    if include_grid_response:
                        dgv = lib.einsum('ngvt,xgt->xnvg', dgbn, fg)
                gbn = None
            else:
                tnuc = tnuc[0] - logger.process_clock(), tnuc[1] - logger.perf_counter()
                if with_j: rhog = rhog.copy()
                jg, gv = batch_jk(mol, coords, dms, fg.transpose(0, 2, 1).copy(), weights)
                if include_grid_response:
                    djpart_int, dgv = batch_jk_grad(mol, coords, rhog, fg.transpose(0, 2, 1).copy(), weights)
                    if with_j and include_grid_response:
                        djg, _ = batch_jonly(mol, coords, dms, None, weights)
                tnuc = tnuc[0] + logger.process_clock(), tnuc[1] + logger.perf_counter()

            # NOTE for j-part, orbital response is not quite right because the overlap
            # fitted density matrix is not used. However, the error due to this is small
            # compared to the error of neglecting the gradient of the overlap fitting matrix.
            if with_j:
                for i in range(nset):
                    xj = lib.einsum('ngv,g->ngv', dao, jg[i])
                    if not include_grid_response:
                        dvj_tmp[i] -= 1.0 * lib.einsum('xgu,gv->xuv', xj, wao) # ORBITAL RESPONSE
                    else:
                        dvj_tmp[i] -= 0.5 * lib.einsum('xgu,gv->xuv', xj, wao) # ORBITAL RESPONSE
                        dvj_tmp[i] -= 0.5 * djpart_int[i] # INTEGRAL RESPONSE
                        jtotal += jg.sum(axis=0).dot(rhog_sum)
                        dej[:,:] += numpy.dot(weights1, jg[i]*(rhog_sum/(weights[:,0]+1e-32))) # GRID RESPONSE PART 1
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
    atm_idx = []
    for ia in range(mol.natm):
        coords, vol = atom_grids_tab[mol.atom_symbol(ia)]
        coords_all.append(coords)
        atm_idx.append(numpy.repeat(numpy.int32(ia), vol.size))
    coords_all = numpy.vstack(coords_all)
    atm_idx = numpy.hstack(atm_idx)
    idx = gen_grid.arg_group_grids(mol, coords)
    coords = coords[idx]
    atm_idx = atm_idx[idx]
    if grids.alignment > 1:
        padding = gen_grid._padding_size(grids.size, grids.alignment)
        logger.debug(grids, 'Padding %d grids', padding)
        if padding > 0:
            coords = numpy.vstack(
                [coords, numpy.repeat([[1e-4]*3], padding, axis=0)])
            atm_idx = numpy.hstack([atm_idx, numpy.repeat(-1, padding)])
    return coords, atm_idx


def scalable_grids_response_cc(grids, blksize):
    assert grids.atm_idx is not None
    mol = grids.mol
    ngrids = grids.weights.size
    assert blksize % BLKSIZE == 0 or blksize == grids.weights.size
    for i0, i1 in lib.prange(0, ngrids, blksize):
        coords = grids.coords[i0:i1]
        weights = grids.weights[i0:i1]
        atm_idx = grids.atm_idx[i0:i1]
        mask = grids.screen_index[i0 // BLKSIZE :]
        weights1 = get_dw_partition_sorted(mol, atm_idx, coords, weights,
                                           grids.radii_adjust,
                                           grids.atomic_radii)
        atom_mask = (atm_idx == -1)
        # make sure weights are zero in the padded region
        weights[atom_mask] = 0
        weights1[..., atom_mask] = 0
        yield i0, i1, atm_idx, coords, weights, weights1.transpose(1, 0, 2), mask


def get_k_grad_only(sgx, dm, hermi=1, direct_scf_tol=1e-13):
    print("STARTING SGX-K GRAD")
    t00 = logger.process_clock(), logger.perf_counter()
    t0 = t00[1]
    if sgx.debug:
        raise NotImplementedError("debug mode for accelerated K matrix")

    mol = sgx.mol
    grids = sgx.grids
    include_grid_response = sgx._grid_response

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
    loop_data = sgx._pjs_data.get_loop_data(nset=nset, with_pair_mask=True)
    nbins, screen_index, pair_mask, ao_loc, blksize = loop_data
    pair_mask_ones = numpy.ones_like(pair_mask)
    proj = sgx._pjs_data._overlap_correction_matrix
    dm_mask = sgx._pjs_data.get_dm_threshold_matrix(
        dms, sgx.sgx_tol_potential, sgx.sgx_tol_energy
    )

    vk = numpy.zeros_like(dms)
    dvk = numpy.zeros((dms.shape[0], 3,) + dms.shape[1:])
    dek = numpy.zeros((mol.natm, 3)) # derivs wrt atom positions

    batch_k = sgx_jk._gen_k_direct(mol, 's2', direct_scf_tol, sgx._pjs_data, use_dm_screening=True)
    if include_grid_response:
        batch_k_grad = sgx_jk._gen_k_direct(mol, 's1', direct_scf_tol, sgx._pjs_data,
                                            use_dm_screening=True, grad=True)
    t1 = logger.perf_counter()

    t2, t3 = 0, 0
    t4 = -logger.perf_counter()
    fg = None
    if sgx.fit_ovlp and include_grid_response:
        sn = numpy.zeros((nao, nao))
        proj_dm = lib.einsum('ki,xij->xkj', proj, dms)
        if sgx._symm_ovlp_fit:
            other_dm = proj_dm
            dvk1 = numpy.zeros((3, nao))
        else:
            other_dm = dms
            dvk1 = numpy.zeros((3, nao))
        other_dm = numpy.ascontiguousarray(other_dm.transpose(0, 2, 1))
    else:
        sn = None
        proj_dm = lib.einsum('ki,xij->xkj', proj, dms)
        other_dm = dms
        dvk1 = numpy.zeros((3, nao))
    for items in scalable_grids_response_cc(grids, blksize):
        # atm_idx is an array shape (ngrids,)
        # coords is an array shape (ngrids, 3)
        # weights is an array shape (ngrids)
        # weights1 is an array shape (natm, 3, ngrids)
        ta = logger.perf_counter()
        i0, i1, atm_idx, coords, weights, weights1, mask = items
        print("LOOP", i0)
        if mol.cart:
            _ao = mol.eval_gto('GTOval_cart_deriv1', coords, non0tab=mask)
        else:
            _ao = mol.eval_gto('GTOval_sph_deriv1', coords, non0tab=mask)
        ao = _ao[0]
        dao = _ao[1:4]
        b0 = i0 // SGX_BLKSIZE
        if sgx.use_dm_screening:
            fg = sgx_jk._sgxdot_ao_dm_sparse(ao, proj_dm, mask, dm_mask, ao_loc, out=fg)
            gv = batch_k(mol, coords, fg, weights, b0)
            if include_grid_response:
                dgv = batch_k_grad(mol, coords, fg, weights, b0)
            mask2 = sgx._pjs_data.get_g_threshold(
                fg, gv, i0, sgx.sgx_tol_potential, sgx.sgx_tol_energy, weights
            )
        else:
            fg = sgx_jk._sgxdot_ao_dm(ao, proj_dm, mask, shls_slice, ao_loc, out=fg)
            gv = batch_k(mol, coords, fg, weights, b0)
            if include_grid_response:
                dgv = batch_k_grad(mol, coords, fg, weights, b0)
            mask2 = None
        if sgx.fit_ovlp and include_grid_response and (not sgx._symm_ovlp_fit):
            fg = lib.einsum("gu,xvu->xvg", ao, other_dm)
        sgx_jk._sgxdot_ao_gv_sparse(ao, gv, weights, mask, mask2, ao_loc, out=vk)

        if sgx.fit_ovlp and include_grid_response:
            sn += sgx_jk._dot_ao_ao_sparse(
                ao, ao, weights, nbins, mask, pair_mask, ao_loc, hermi=0
            )

        tb = logger.perf_counter()
        no_mask2 = mask2 is None
        if no_mask2:
            dxed = 0
        else:
            dxed = numpy.zeros((3, weights.size))
        for i in range(nset):
            if no_mask2:
                tmp = lib.einsum("vg,vu->ug", gv[i], other_dm[i])
                tmp[:] *= 0.5 * weights
            else:
                # If gv or ao is small, ignore shell-block. If EITHER is
                # signficant at all, don't ignore. TODO this is overly
                # conservative but matches the approach for energy/potential
                mask3 = numpy.uint8(127) * numpy.logical_or(mask, mask2)
                print(numpy.count_nonzero(mask3) / mask3.size)
                tmp = sgx_jk._dot_ao_dm_sparse(
                    gv[i].T, other_dm[i], nbins, mask3, pair_mask_ones, ao_loc
                ).T
                tmp = ni._scale_ao_sparse(tmp.T, 0.5 * weights, mask3, ao_loc).T
            dvk1[:] -= lib.einsum('xgu,ug->xu', dao, tmp)
            if include_grid_response:
                if no_mask2:
                    wfg = lib.einsum("ug,g->ug", fg[i], 0.5 * weights)
                    dvk1[:] -= lib.einsum('xug,ug->xu', dgv[i], wfg) # INTEGRAL RESPONSE
                    xed = lib.einsum('ug,ug->g', fg[i], gv[i])
                    dxed -= lib.einsum('ug,xug->xg', wfg, dgv[i]) # INTEGRAL RESPONSE
                    dxed -= lib.einsum("ug,xgu->xg", tmp, dao)
                else:
                    wfg = ni._scale_ao_sparse(fg[i].T, 0.5 * weights, mask3, ao_loc)
                    xed = ni._contract_rho_sparse(fg[i].T, gv[i].T, mask3, ao_loc)
                    for x in range(3):
                        dvk1[x] -= lib.einsum('ug,ug->u', dgv[i, x], wfg.T)
                        dxed[x] -= ni._contract_rho_sparse(wfg, dgv[i, x].T, mask3, ao_loc)
                        dxed[x] -= ni._contract_rho_sparse(tmp.T, dao[x], mask3, ao_loc)
                dek[:, :] += lib.einsum("axg,g->ax", weights1, xed) # GRID RESPONSE PART 1
        if include_grid_response:
            for ia in range(mol.natm):
                cond = (atm_idx == ia)
                dek[ia] -= 4 * numpy.sum(dxed[:, cond], axis=1)  # GRID RESPONSE PART 2
        else:
            dvk1[:] *= 2
        tc = logger.perf_counter()

        t2 += tb - ta
        t3 += tc - tb
    tmid = logger.perf_counter()
    t4 += tmid
    print("TIMES", t1 - t0, t2, t3, t4 - t3 - t2)

    aoslices = mol.aoslice_by_atom()
    if sgx.fit_ovlp and include_grid_response:
        if sgx._symm_ovlp_fit:
            proj = 2 * proj - numpy.identity(proj.shape[0])
        else:
            msg = (
                "SGX-K gradients with fit_ovlp=True are only exact if "
                "_symm_ovlp_fit=True, and it is currently False. "
                "There might be small errors in the analytical gradients. "
                "Set _symm_ovlp_fit=True for exact analytical gradients."
            )
            logger.warn(sgx, msg)
        grad_ovlp = mol.intor("int1e_ipovlp", comp=3)
        Cmat = 0
        for i in range(nset):
            Cmat += lib.dot(vk[i], dms[i])
        Cmat = numpy.linalg.solve(sn, Cmat)
        for x in range(3):
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia, 2:]
                dek[ia, x] -= lib.einsum("uv,vu->", grad_ovlp[x, p0 : p1], Cmat[:, p0 : p1])
                dek[ia, x] -= lib.einsum("uv,uv->", grad_ovlp[x, p0 : p1], Cmat[p0 : p1, :])
        Cmat = lib.dot(Cmat, proj.T)
        Cmat = Cmat + Cmat.T
        mat = numpy.zeros_like(grad_ovlp)
        for items in scalable_grids_response_cc(grids, blksize):
            i0, i1, atm_idx, coords, weights, weights1, mask = items
            if mol.cart:
                _ao = mol.eval_gto('GTOval_cart_deriv1', coords, non0tab=mask)
            else:
                _ao = mol.eval_gto('GTOval_sph_deriv1', coords, non0tab=mask)
            ao = _ao[0]
            dao = _ao[1:4]
            ca = sgx_jk._dot_ao_dm_sparse(ao, Cmat, nbins, mask, pair_mask_ones, ao_loc)
            #xed = lib.einsum("gv,gv->g", ca, ao)
            #dxed = lib.einsum("gv,xgv->xg", ca, dao)
            xed = ni._contract_rho_sparse(ca, ao, mask, ao_loc)
            dxed = numpy.empty((3, weights.size))
            for x in range(3):
                dxed[x] = ni._contract_rho_sparse(ca, dao[x], mask, ao_loc)
            dxed[:] *= weights
            for x in range(3):
                mat[x] += sgx_jk._dot_ao_ao_sparse(
                    dao[x], ao, weights, nbins, mask, pair_mask, ao_loc, hermi=0
                )
            # negative because d sn^-1 / dR = -sn^-1 (dsn/dR) sn^-1
            dek[:, :] -= 0.5 * lib.einsum("axg,g->ax", weights1, xed)  # GRID RESPONSE PART 1
            for ia in range(mol.natm):
                cond = (atm_idx == ia)
                dek[ia] -= numpy.sum(dxed[:, cond], axis=1)  # GRID RESPONSE PART 2
        for x in range(3):
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia, 2:]
                dek[ia, x] += lib.einsum("uv,uv->", mat[x, p0 : p1], Cmat[p0 : p1])

    for x in range(3):
        for ia in range(mol.natm):
            p0, p1 = aoslices[ia, 2:]
            dek[ia, x] += numpy.sum(4 * dvk1[x, p0 : p1])

    # TODO don't need to compute vk
    vk = dvk
    logger.timer(mol, "vk", *t00)
    dm_shape = (nset, 3) + dms.shape[1:]
    vk = vk.reshape(dm_shape)
    if restricted:
        vk = vk[0]
    vk = lib.tag_array(vk, aux=0.5*dek[None, None])
    tfin = logger.perf_counter()
    print("TOTAL SGX-K TIME", tfin - t0, tfin - tmid)
    return vk


def _get_jk(sgx, dm, hermi, with_j, with_k, direct_scf_tol):
    is_opt = sgx.optk and (sgx.grids.becke_scheme == gen_grid.becke_lko)
    if with_j and sgx.dfj:
        vj = None
        if with_k:
            if is_opt:
                vk = get_k_grad_only(sgx, dm, hermi, direct_scf_tol)
            else:
                vk = get_jk_grad(sgx, dm, hermi, False, with_k, direct_scf_tol)[1]
        else:
            vk = None
    else:
        if (not with_j) and is_opt:
            vj = None
            vk = get_k_grad_only(sgx, dm, hermi, direct_scf_tol)
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
        if not with_k:
            return vj, None
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
            rsh_df._overlap_correction_matrix = None
            rsh_df.build()
            sgx._rsh_df[key] = rsh_df
            logger.info(sgx, 'Create RSH-SGX object %s for omega=%s', rsh_df, omega)

        with rsh_df.mol.with_range_coulomb(omega):
            rsh_df._grid_response = mf_grad.sgx_grid_response
            _vj, vk = _get_jk(rsh_df, dm, hermi, with_j, with_k, direct_scf_tol)
    else:
        sgx._grid_response = mf_grad.sgx_grid_response
        _vj, vk = _get_jk(sgx, dm, hermi, with_j, with_k, direct_scf_tol)

    if vj is None:
        vj = _vj
    return vj, vk


class _GradientsMixin:
    _keys = {'sgx_grid_response'}

    def __init__(self, mf):
        # Whether to include the response of DF auxiliary basis when computing
        # nuclear gradients of J/K matrices
        self.sgx_grid_response = True
        if mf.with_df.direct_j:
            raise ValueError("direct_j setting not supported for gradients")
        super().__init__(mf)

    def check_sanity(self):
        assert isinstance(self.base, sgx._SGXHF)

    def get_j(self, mol=None, dm=None, hermi=0, omega=None):
        return self.get_jk(mol=mol, dm=dm, hermi=hermi, vhfopt=None,
                           with_j=True, with_k=False, omega=omega)[0]

class Gradients(_GradientsMixin, dfrhf_grad.Gradients):
    '''Restricted SGX Hartree-Fock gradients. Note: sgx_grid_response determines
    whether full grid response for SGX terms is computed.'''
    get_jk = get_jk


Grad = Gradients
