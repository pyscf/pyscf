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
from pyscf.dft.gen_grid import LEBEDEV_ORDER, SGX_ANG_MAPPING, \
                               sgx_prune, becke_lko
from pyscf.dft.numint import eval_ao, BLKSIZE, NBINS, libdft, \
    SWITCH_SIZE, _scale_ao, _dot_ao_ao_sparse, _dot_ao_dm_sparse, \
    _scale_ao_sparse


libdft.SGXreturn_blksize.restype = int
SGX_BLKSIZE = libdft.SGXreturn_blksize()
assert SGX_BLKSIZE % BLKSIZE == 0

SGX_DELTA_1 = 0.1
SGX_DELTA_2 = 0.9
SGX_DELTA_3 = 0.5


def _sgxdot_ao_dm(ao, dms, non0tab, shls_slice, ao_loc, out=None):
    '''return numpy.dot(ao, dms)'''
    ngrids, nao = ao.shape
    nbas = len(ao_loc) - 1
    vms = numpy.ndarray((dms.shape[0], dms.shape[2], ngrids), dtype=ao.dtype, order='C', buffer=out)
    # TODO don't need sparsity for small systems
    #if (nao < SWITCH_SIZE or
    #    non0tab is None or shls_slice is None or ao_loc is None):
    #    out[:] = lib.dot(dm.T, ao.T)
    if (nao < SWITCH_SIZE or non0tab is None or shls_slice is None or ao_loc is None):
        for i in range(len(dms)):
            lib.dot(dms[i].T, ao.T, c=vms[i])
        return vms

    if not ao.flags.f_contiguous:
        ao = lib.transpose(ao)
    if ao.dtype == dms.dtype == numpy.double:
        fn = libdft.VXCsgx_ao_dm
        # fn = libdft.VXCdot_ao_dm
    else:
        # TODO implement this
        raise NotImplementedError("Complex sgxdot")
        fn = libdft.VXCzdot_ao_dm
        ao = numpy.asarray(ao, numpy.complex128)
        dms = numpy.asarray(dms, numpy.complex128)

    dms = numpy.asarray(dms, order='C')
    for i in range(len(dms)):
        fn(
            vms[i].ctypes.data_as(ctypes.c_void_p),
            ao.ctypes.data_as(ctypes.c_void_p),
            dms[i].ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao), ctypes.c_int(dms.shape[2]),
            ctypes.c_int(ngrids), ctypes.c_int(nbas),
            non0tab.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*2)(*shls_slice),
            ao_loc.ctypes.data_as(ctypes.c_void_p)
        )
    return vms


def _sgxdot_ao_dm_sparse(ao, dms, mask, pair_mask, ao_loc, out=None):
    '''return numpy.dot(ao, dms)'''
    ngrids, nao = ao.shape
    nbas = len(ao_loc) - 1
    vms = numpy.ndarray((dms.shape[0], dms.shape[2], ngrids), dtype=ao.dtype, order='C', buffer=out)
    if (nao < SWITCH_SIZE or mask is None or ao_loc is None):
        for i in range(len(dms)):
            lib.dot(dms[i].T, ao.T, c=vms[i])
        return vms
    if pair_mask is None:
        return _sgxdot_ao_dm(ao, dms, mask, (0, nbas), ao_loc, out=out)

    if not ao.flags.f_contiguous:
        ao = lib.transpose(ao)
    if ao.dtype == dms.dtype == numpy.double:
        fn = _vhf.libcvhf.SGXdot_ao_dm_sparse
        # fn = libdft.VXCdot_ao_dm
    else:
        # TODO implement this
        raise NotImplementedError("Complex sgxdot")
        fn = libdft.VXCzdot_ao_dm
        ao = numpy.asarray(ao, numpy.complex128)
        dms = numpy.asarray(dms, numpy.complex128)

    dms = numpy.asarray(dms, order='C')
    for i in range(len(dms)):
        fn(
            vms[i].ctypes.data_as(ctypes.c_void_p),
            ao.ctypes.data_as(ctypes.c_void_p),
            dms[i].ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao),
            ctypes.c_int(ngrids),
            ctypes.c_int(nbas),
            mask.ctypes.data_as(ctypes.c_void_p),
            pair_mask.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p)
        )
    return vms


def _sgxdot_ao_gv(ao, gv, non0tab, shls_slice, ao_loc, out=None):
    '''return numpy.dot(ao.T, gv.T)'''
    ngrids, nao = ao.shape
    # TODO don't need sparsity for small systems
    #if (nao < SWITCH_SIZE or
    #    for i in range(len(gv)):
    #        lib.dot(ao1.conj().T, gv[i].T, c=vms[i])
    #    return vms
    outshape = (gv.shape[0], nao, nao)
    if (nao < SWITCH_SIZE or non0tab is None
            or shls_slice is None or ao_loc is None):
        if out is None:
            vv = numpy.zeros(outshape, dtype=ao.dtype, order='C')
        else:
            vv = out
        for i in range(len(gv)):
            lib.dot(ao.conj().T, gv[i].T, c=vv[i], beta=1.0)
        return vv
    nbas = non0tab.shape[1]

    if not ao.flags.f_contiguous:
        ao = lib.transpose(ao)
    if not gv.flags.c_contiguous:
        gv = numpy.ascontiguousarray(gv)
    if ao.dtype == gv.dtype == numpy.double:
        fn = libdft.VXCsgx_ao_ao
    else:
        # TODO implement this
        raise NotImplementedError("Complex sgxdot")
        fn = libdft.VXCzdot_ao_ao
        ao = numpy.asarray(ao, numpy.complex128)
        gv = numpy.asarray(gv, numpy.complex128)

    vv = numpy.empty(outshape, dtype=ao.dtype, order='C')
    for i in range(len(gv)):
        print(shls_slice, ao_loc)
        fn(
            vv[i].ctypes.data_as(ctypes.c_void_p),
            ao.ctypes.data_as(ctypes.c_void_p),
            gv[i].ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao), ctypes.c_int(ngrids),
            ctypes.c_int(nbas),
            non0tab.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*2)(*shls_slice),
            ao_loc.ctypes.data_as(ctypes.c_void_p)
        )
    vv = vv.transpose(0, 2, 1)
    if out is None:
        return vv
    else:
        out[:] += vv
        return out


def _sgxdot_ao_gv_sparse(ao, gv, wt, mask1, mask2, ao_loc, out=None):
    ngrids, nao = ao.shape
    nbas = mask1.shape[1]
    assert gv.shape[1:] == ao.shape[::-1]
    assert wt is not None
    if out is None:
        vv = numpy.zeros((gv.shape[0], nao, nao), dtype=ao.dtype, order='C')
    else:
        vv = out
    if (nao < SWITCH_SIZE or mask1 is None
            or mask2 is None or ao_loc is None):
        return _sgxdot_ao_gv(ao * wt[:, None], gv, mask1, (0, nbas), ao_loc, out)

    if not ao.flags.f_contiguous:
        ao = lib.transpose(ao)
    if not gv.flags.c_contiguous:
        gv = numpy.ascontiguousarray(gv)

    fn = _vhf.libcvhf.SGXdot_ao_ao_sparse
    for i in range(len(gv)):
        fn(
            vv[i].ctypes.data_as(ctypes.c_void_p),
            ao.ctypes.data_as(ctypes.c_void_p),
            gv[i].ctypes.data_as(ctypes.c_void_p),
            wt.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao),
            ctypes.c_int(ngrids),
            ctypes.c_int(nbas),
            ctypes.c_int(0),
            mask1.ctypes.data_as(ctypes.c_void_p),
            mask2.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
        )
    return vv


def _get_shell_norms(dms, weights, ao_loc, buffer=None):
    nao, ngrids = dms[0].shape
    ndm = len(dms)
    nbas = len(ao_loc) - 1
    _dms = [dm.ctypes.data_as(ctypes.c_void_p) for dm in dms]
    nblk = (ngrids + BLKSIZE - 1) // BLKSIZE
    shell_norms = numpy.ndarray((nblk, nbas), buffer=buffer)
    _vhf.libcvhf.SGXget_shell_norms(
        (ctypes.c_void_p * ndm)(*_dms),
        shell_norms.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ndm),
        ao_loc.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nbas),
        ctypes.c_int(ngrids),
        weights.ctypes.data_as(ctypes.c_void_p),
    )
    return shell_norms


def _reduce_dft_mask(dft_mask, ngrids, buffer=None):
    dft_nblk, nbas = dft_mask.shape
    dft_nblk = (ngrids + BLKSIZE - 1) // BLKSIZE
    ratio = SGX_BLKSIZE // BLKSIZE
    sgx_nblk = (dft_nblk + ratio - 1) // ratio
    assert dft_mask.dtype == numpy.uint8
    assert dft_mask.flags.c_contiguous
    sgx_mask = numpy.ndarray(
        (sgx_nblk, nbas), dtype=numpy.uint8, buffer=buffer
    )
    _vhf.libcvhf.SGXreduce_dft_mask(
        sgx_mask.ctypes.data_as(ctypes.c_void_p),
        dft_mask.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(dft_nblk),
        ctypes.c_int(nbas),
        ctypes.c_int(ratio),
    )
    return sgx_mask


def _sgx_update_masks_(asn, basmax, ao_cond_sgx, ao_cond_ni,
                       ao, weights, nbins, mask, pair_mask,
                       ao_loc, hermi=0):
    wt = numpy.sqrt(numpy.abs(weights))
    ngrids = wt.size
    nbas = len(ao_loc) - 1
    assert ao_cond_sgx.flags.c_contiguous
    assert ao_cond_ni.flags.c_contiguous
    assert basmax.flags.c_contiguous
    assert ao.flags.f_contiguous
    assert ao.shape == (ngrids, ao_loc[-1])
    libdft.SGXmake_screen_q_cond(
        basmax.ctypes.data_as(ctypes.c_void_p),
        ao.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(nbas),
        ao_loc.ctypes.data_as(ctypes.c_void_p),
    )
    wao = _scale_ao_sparse(ao, wt, mask, ao_loc, out=ao)
    libdft.SGXmake_screen_norm(
        ao_cond_sgx.ctypes.data_as(ctypes.c_void_p),
        wao.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(nbas),
        ao_loc.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(SGX_BLKSIZE),
    )
    libdft.SGXmake_screen_norm(
        ao_cond_ni.ctypes.data_as(ctypes.c_void_p),
        wao.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(nbas),
        ao_loc.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(BLKSIZE),
    )
    _dot_ao_ao_sparse(wao, wao, None, nbins, mask,
                      pair_mask, ao_loc, hermi=hermi, out=asn)
    

def _sgx_create_masks(asn, basmax, ao_loc):
    nao = ao_loc[-1]
    nblk, nbas = basmax.shape
    rinv_bound = numpy.empty((nbas, nbas))
    libdft.SGXmake_rinv_ubound(
        rinv_bound.ctypes.data_as(ctypes.c_void_p),
        asn.ctypes.data_as(ctypes.c_void_p),
        basmax.ctypes.data_as(ctypes.c_void_p),
        ao_loc.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nbas),
        ctypes.c_int(nao),
        ctypes.c_int(nblk),
    )
    shl_sn = numpy.empty((nbas, nbas))
    _vhf.libcvhf.SGXmake_shl_op(
        asn.ctypes.data_as(ctypes.c_void_p),
        shl_sn.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao),
        ctypes.c_int(nbas),
        ao_loc.ctypes.data_as(ctypes.c_void_p),
    )
    delta = 0.1
    norm = shl_sn**delta * numpy.sum(shl_sn**(1 - delta), axis=1)
    norm_mask = numpy.max(norm, axis=1)
    return rinv_bound, norm_mask


def _get_ao_block_cond(sgx, ao_cond):
    nblk, nbas = ao_cond.shape
    ncond = numpy.empty(nblk)
    libdft.SGXreduce_screen(
        ncond.ctypes.data_as(ctypes.c_void_p),
        ao_cond.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nblk),
        ctypes.c_int(nbas),
        ctypes.c_double(SGX_DELTA_2),
    )
    return sgx.sgx_tol_potential / ncond


def _get_sgx_dm_mask(sgx, dms, ao_loc):
    if sgx.use_dm_screening:
        mol = sgx.mol
        nao = mol.nao_nr()
        nset = dms.shape[0]
        ta = logger.perf_counter()
        shl_dm = numpy.empty((nset, mol.nbas, mol.nbas))
        for i in range(nset):
            _vhf.libcvhf.SGXmake_shl_dm(
                dms[i].ctypes.data_as(ctypes.c_void_p),
                shl_dm[i].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao),
                ctypes.c_int(mol.nbas),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
            )
        shl_dm = shl_dm.sum(axis=0)
        dm_mask = shl_dm * sgx._norm_mask[:, None] * sgx._rinv_mask > sgx.sgx_tol_potential
        tb = logger.perf_counter()
        print("OTHER SETUP", tb - ta)
        return dm_mask
    else:
        return None


def get_jk_favork(sgx, dm, hermi=1, with_j=True, with_k=True,
                  direct_scf_tol=1e-13):
    t0 = logger.process_clock(), logger.perf_counter()
    mol = sgx.mol
    grids = sgx.grids

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]

    if sgx.debug:
        batch_nuc = _gen_batch_nuc(mol)
    else:
        batch_jk = _gen_jk_direct(mol, 's2', with_j, with_k, direct_scf_tol,
                                  sgx._opt)
    t1 = logger.timer_debug1(mol, "sgX initialization", *t0)

    sn = numpy.zeros((nao,nao))
    vj = numpy.zeros_like(dms)
    vk = numpy.zeros_like(dms)

    ngrids = grids.coords.shape[0]
    max_memory = sgx.max_memory - lib.current_memory()[0]
    # We need to store ao, wao, and fg -> 3 sets of size nao
    blksize = min(ngrids, max(112, int(max_memory*1e6/8/(3 * nao))))
    tnuc = 0, 0
    for i0, i1 in lib.prange(0, ngrids, blksize):
        coords = grids.coords[i0:i1]
        weights = grids.weights[i0:i1,None]
        ao = mol.eval_gto('GTOval', coords)
        wao = ao * grids.weights[i0:i1,None]
        sn += lib.dot(ao.T, wao)

        fg = lib.einsum('xij,ig->xjg', dms, wao.T)

        if sgx.debug:
            tnuc = tnuc[0] - logger.process_clock(), tnuc[1] - logger.perf_counter()
            gbn = batch_nuc(mol, coords)
            tnuc = tnuc[0] + logger.process_clock(), tnuc[1] + logger.perf_counter()
            if with_j:
                jg = numpy.einsum('gij,xij->xg', gbn, dms)
            if with_k:
                gv = lib.einsum('gvt,xtg->xvg', gbn, fg)
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
                vk[i] += lib.einsum('gu,vg->uv', ao, gv[i])
        jg = gv = None

    t2 = logger.timer_debug1(mol, "sgX J/K builder", *t1)
    tdot = t2[0] - t1[0] - tnuc[0], t2[1] - t1[1] - tnuc[1]
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


def run_k_only_setup(sgx, dms, hermi):
    mol = sgx.mol
    grids = sgx.grids
    nao = dms.shape[-1]
    nset = dms.shape[0]
    ao_loc = mol.ao_loc_nr()
    t0 = logger.perf_counter()
    print("START SETUP")

    if grids.coords is None or grids.non0tab is None or grids.ialist is None:
        grids.build(with_non0tab=True, with_ialist=True)
        sgx._sgx_block_cond = None

    if mol is grids.mol:
        non0tab = grids.non0tab
    if non0tab is None:
        non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                                dtype=numpy.uint8)
        non0tab[:] = NBINS + 1  # Corresponding to AO value ~= 1
    screen_index = non0tab

    ngrids = grids.coords.shape[0]
    max_memory = sgx.max_memory - lib.current_memory()[0]
    # We need to store ao, wao, and fg -> 2 + nset sets of size nao
    blksize = max(112, int(max_memory*1e6/8/((2+nset)*nao)))
    blksize = min(ngrids, max(1, blksize // SGX_BLKSIZE) * SGX_BLKSIZE)

    if sgx._overlap_correction_matrix is None:
        sn = numpy.zeros((nao,nao))
        cutoff = grids.cutoff * 1e2
        nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))
        pair_mask = mol.get_overlap_cond() < -numpy.log(cutoff)
        if sgx.fit_ovlp:
            sn = numpy.zeros((nao, nao))
            for i0, i1 in lib.prange(0, ngrids, blksize):
                assert i0 % SGX_BLKSIZE == 0
                coords = grids.coords[i0:i1]
                mask = screen_index[i0 // BLKSIZE:]
                ao = mol.eval_gto('GTOval', coords, non0tab=mask)
                _dot_ao_ao_sparse(ao, ao, grids.weights[i0:i1], nbins, mask,
                                  pair_mask, ao_loc, hermi=hermi, out=sn)
                # wao = _scale_ao(ao, grids.weights[i0:i1])
                # sn[:] += _dot_ao_ao(mol, ao, wao, mask, (0, mol.nbas),
                #                     ao_loc, hermi=hermi)
            ovlp = mol.intor_symmetric('int1e_ovlp')
            proj = scipy.linalg.solve(sn, ovlp)
            sgx._overlap_correction_matrix = proj
            print("PROJ", numpy.max(numpy.abs(proj - proj.T)))
        else:
            sgx._overlap_correction_matrix = numpy.identity(nao)
    proj = sgx._overlap_correction_matrix
    proj_dm = lib.einsum('ki,xij->xkj', proj, dms)

    if sgx.use_dm_screening and sgx._sgx_block_cond is None:
        ta = logger.perf_counter()
        nblk = (grids.weights.size + SGX_BLKSIZE - 1) // SGX_BLKSIZE
        nblk_ni = (grids.weights.size + BLKSIZE - 1) // BLKSIZE
        ao_cond_sgx = numpy.empty((nblk, mol.nbas), dtype=numpy.float64)
        ao_cond_ni = numpy.empty((nblk_ni, mol.nbas), dtype=numpy.float64)
        basmax = numpy.empty((nblk, mol.nbas), dtype=numpy.float64)
        asn = numpy.zeros((nao, nao))
        ao = None
        for i0, i1 in lib.prange(0, ngrids, blksize):
            assert i0 % SGX_BLKSIZE == 0
            coords = grids.coords[i0:i1]
            mask = screen_index[i0 // BLKSIZE:]
            ao = mol.eval_gto('GTOval', coords, non0tab=mask, out=ao)
            _sgx_update_masks_(asn, basmax[i0 // SGX_BLKSIZE:],
                               ao_cond_sgx[i0 // SGX_BLKSIZE:],
                               ao_cond_ni[i0 // BLKSIZE:],
                               ao, grids.weights[i0:i1], nbins,
                               mask, pair_mask, ao_loc, hermi=hermi)
        tb = logger.perf_counter()
        ncond = _get_ao_block_cond(sgx, ao_cond_sgx)
        ncond_ni = _get_ao_block_cond(sgx, ao_cond_ni)
        rinv_bound, norm_mask = _sgx_create_masks(asn, basmax, ao_loc)
        sgx._sgx_block_cond = ncond
        sgx._ni_block_cond = ncond_ni
        sgx._rinv_bound = rinv_bound
        rinv = rinv_bound**(1 - SGX_DELTA_3)
        sgx._rinv_mask = numpy.max(
            rinv_bound**SGX_DELTA_3 * numpy.sum(rinv, axis=1),
            axis=1
        )
        sgx._ao_block_norm = ao_cond_sgx
        sgx._norm_mask = norm_mask
        tc = logger.perf_counter()
        print("SETUP TOOK", ta - t0, tb - ta, tc - tb)
    elif sgx.use_dm_screening:
        ncond = sgx._sgx_block_cond
        ncond_ni = sgx._ni_block_cond
    else:
        ncond = None
        ncond_ni = None

    if sgx.use_dm_screening:
        dm_mask = _get_sgx_dm_mask(sgx, dms, ao_loc)
    else:
        dm_mask = None

    return blksize, screen_index, proj_dm, dm_mask, ncond, ncond_ni, ao_loc


def get_k_only(sgx, dm, hermi=1, direct_scf_tol=1e-13):
    if sgx.debug:
        raise NotImplementedError("debug mode for accelerated K matrix")

    t0 = logger.process_clock(), logger.perf_counter()
    mol = sgx.mol
    grids = sgx.grids

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)

    # batch_jk = _gen_jk_direct(mol, 's2', False, True, direct_scf_tol, sgx._opt)
    t1 = logger.timer_debug1(mol, "sgX initialization", *t0)
    vk = numpy.zeros_like(dms)
    tnuc = 0, 0
    shls_slice = (0, mol.nbas)
    sgx_data = run_k_only_setup(sgx, dms, hermi)
    ngrids = grids.weights.size
    blksize, screen_index, proj_dm, dm_mask, ncond, ncond_ni, ao_loc = sgx_data

    batch_k = _gen_k_direct(mol, 's2', direct_scf_tol, sgx._opt)

    t_setup = logger.perf_counter()
    print("AFTER_SETUP", t_setup - t0[1])

    ao = None
    fg = None
    t_ao = 0
    t_scale = 0
    t_ao_dm = 0
    t_ao_ao = 0
    t_int = 0
    print("LEFTOVER", ngrids % SGX_BLKSIZE)
    for i0, i1 in lib.prange(0, ngrids, blksize):
        assert i0 % SGX_BLKSIZE == 0
        coords = grids.coords[i0:i1]
        weights = grids.weights[i0:i1]
        mask = screen_index[i0 // BLKSIZE :]
        ta = logger.perf_counter()
        ao = mol.eval_gto('GTOval', coords, non0tab=mask, out=ao)
        tb = logger.perf_counter()
        print("DOT 1")
        tnuc = tnuc[0] - logger.process_clock(), tnuc[1] - logger.perf_counter()
        tc = logger.perf_counter()
        if sgx.use_dm_screening:
            fg = _sgxdot_ao_dm_sparse(ao, proj_dm, mask, dm_mask, ao_loc, out=fg)
        else:
            # fg = lib.einsum('xij,ig->xjg', proj_dm, wao.T)
            fg = _sgxdot_ao_dm(ao, proj_dm, mask, shls_slice, ao_loc, out=fg)
        te = logger.perf_counter()
        print("INTEGRALS")
        
        if ncond is None:
            gv = batch_k(mol, coords, fg, weights)
        else:
            gv = batch_k(mol, coords, fg, weights, ncond[i0 // SGX_BLKSIZE :])
        tf = logger.perf_counter()
        tnuc = tnuc[0] + logger.process_clock(), tnuc[1] + logger.perf_counter()
        print("DOT 2")
        if sgx.use_dm_screening:
            shl_norms = _get_shell_norms(gv, weights, ao_loc)
            mask2 = shl_norms > ncond_ni[i0 // BLKSIZE : i0 // BLKSIZE + shl_norms.shape[0], None]
            _sgxdot_ao_gv_sparse(ao, gv, weights, mask, mask2, ao_loc, out=vk)
        else:
            _sgxdot_ao_gv_sparse(ao, gv, weights, mask, None, ao_loc, out=vk)
        tg = logger.perf_counter()
        gv = None
        print("FINISHED 1 ITERATION")
        t_ao += tb - ta
        t_scale += tc - tb
        t_ao_dm += te - tc
        t_int += tf - te
        t_ao_ao += tg - tf
    print("Times", t_ao, t_scale, t_ao_dm, t_int, t_ao_ao)

    t2 = logger.timer_debug1(mol, "sgX J/K builder", *t1)
    tdot = t2[0] - t1[0] - tnuc[0] , t2[1] - t1[1] - tnuc[1]
    logger.debug1(sgx, '(CPU, wall) time for integrals (%.2f, %.2f); '
                  'for tensor contraction (%.2f, %.2f)',
                  tnuc[0], tnuc[1], tdot[0], tdot[1])
    print(sgx, '(CPU, wall) time for integrals (%.2f, %.2f); '
          'for tensor contraction (%.2f, %.2f)',
          tnuc[0], tnuc[1], tdot[0], tdot[1])

    if hermi == 1:
        vk = (vk + vk.transpose(0,2,1))*.5
    logger.timer(mol, "vk", *t0)
    return vk.reshape(dm_shape)


def get_jk_favorj(sgx, dm, hermi=1, with_j=True, with_k=True,
                  direct_scf_tol=1e-13):
    t0 = logger.process_clock(), logger.perf_counter()
    mol = sgx.mol
    grids = sgx.grids

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]

    if sgx.debug:
        batch_nuc = _gen_batch_nuc(mol)
    else:
        batch_jk = _gen_jk_direct(mol, 's2', with_j, with_k, direct_scf_tol,
                                  sgx._opt)

    if mol is grids.mol:
        non0tab = grids.non0tab
    if non0tab is None:
        ngrids = grids.weights.size
        non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                                dtype=numpy.uint8)
        non0tab[:] = NBINS + 1  # Corresponding to AO value ~= 1
    screen_index = non0tab
    # print("SPARSITY", (screen_index > 0).sum() / screen_index.size)

    sn = numpy.zeros((nao,nao))
    ngrids = grids.coords.shape[0]
    max_memory = sgx.max_memory - lib.current_memory()[0]
    # We need to store ao, wao, and fg -> 2 + nset sets of size nao
    blksize = min(ngrids, max(112, int(max_memory*1e6/8/((2 + nset) * nao))))
    blksize = max(4, blksize // BLKSIZE) * BLKSIZE
    if sgx.fit_ovlp:
        for i0, i1 in lib.prange(0, ngrids, blksize):
            assert i0 % BLKSIZE == 0
            coords = grids.coords[i0:i1]
            mask = screen_index[i0 // BLKSIZE:]
            ao = mol.eval_gto('GTOval', coords, non0tab=mask)
            wao = ao * grids.weights[i0:i1,None]
            sn += lib.dot(ao.T, wao)
            # TODO make sparse
            # sn += _dot_ao_ao_sparse(ao, ao, grids.weights[i0:i1], )
        ovlp = mol.intor_symmetric('int1e_ovlp')
        proj = scipy.linalg.solve(sn, ovlp)
        proj_dm = lib.einsum('ki,xij->xkj', proj, dms)
    else:
        proj_dm = dms.copy()

    t1 = logger.timer_debug1(mol, "sgX initialization", *t0)
    vj = numpy.zeros_like(dms)
    vk = numpy.zeros_like(dms)
    tnuc = 0, 0
    wao = None
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    fg = None
    for i0, i1 in lib.prange(0, ngrids, blksize):
        assert i0 % BLKSIZE == 0
        coords = grids.coords[i0:i1]
        weights = grids.weights[i0:i1]
        mask = screen_index[i0 // BLKSIZE:]
        ao = mol.eval_gto('GTOval', coords, non0tab=mask)
        # wao = ao * weights[:, None]
        # fg = lib.einsum('xij,ig->xjg', proj_dm, wao.T)
        wao = _scale_ao(ao, weights, out=wao)
        fg = _sgxdot_ao_dm(wao, proj_dm, mask, shls_slice, ao_loc, out=fg)

        if with_j:
            rhog = numpy.einsum('xug,gu->xg', fg, ao)
        else:
            rhog = None

        if sgx.debug:
            tnuc = tnuc[0] - logger.process_clock(), tnuc[1] - logger.perf_counter()
            gbn = batch_nuc(mol, coords)
            tnuc = tnuc[0] + logger.process_clock(), tnuc[1] + logger.perf_counter()
            if with_j:
                jpart = numpy.einsum('guv,xg->xuv', gbn, rhog)
            if with_k:
                gv = lib.einsum('gtv,xtg->xvg', gbn, fg)
            gbn = None
        else:
            tnuc = tnuc[0] - logger.process_clock(), tnuc[1] - logger.perf_counter()
            if with_j: rhog = rhog.copy()
            jpart, gv = batch_jk(mol, coords, rhog, fg, weights)
            tnuc = tnuc[0] + logger.process_clock(), tnuc[1] + logger.perf_counter()

        if with_j:
            vj += jpart
        if with_k:
            # vk[:] += lib.einsum('gu,xvg->xuv', ao, gv)
            # vk[:] += _sgxdot_ao_gv(ao, gv, mask, shls_slice, ao_loc)
            _sgxdot_ao_gv(ao, gv, mask, shls_slice, ao_loc, out=vk)
        jpart = gv = None

    t2 = logger.timer_debug1(mol, "sgX J/K builder", *t1)
    tdot = t2[0] - t1[0] - tnuc[0] , t2[1] - t1[1] - tnuc[1]
    print(sgx, '(CPU, wall) time for integrals (%.2f, %.2f); '
          'for tensor contraction (%.2f, %.2f)',
          tnuc[0], tnuc[1], tdot[0], tdot[1])

    for i in range(nset):
        lib.hermi_triu(vj[i], inplace=True)
    if with_k and hermi == 1:
        vk = (vk + vk.transpose(0, 2, 1))*.5
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


def _gen_batch_nuc_grad(mol):
    '''Coulomb integrals of the given points and orbital pairs'''
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, 'int3c2e_ip1')
    def batch_nuc(mol, grid_coords, out=None):
        fakemol = gto.fakemol_for_charges(grid_coords)
        j3c = aux_e2(mol, fakemol, intor='int3c2e_ip1', aosym='s1', cintopt=cintopt)
        return j3c.transpose(0,3,1,2)
    return batch_nuc


def _gen_jk_direct(mol, aosym, with_j, with_k, direct_scf_tol,
                   sgxopt=None, grad=False):
    '''Contraction between sgX Coulomb integrals and density matrices

    J: einsum('guv,xg->xuv', gbn, dms) if dms == rho at grid,
    or einsum('gij,xij->xg', gbn, dms) if dms are density matrices

    K: einsum('gtv,xgt->xgv', gbn, fg)
    '''
    if sgxopt is None:
        from pyscf.sgx import sgx
        sgxopt = sgx._make_opt(mol, False, grad, direct_scf_tol)
    sgxopt.direct_scf_tol = direct_scf_tol

    ao_loc = moleintor.make_loc(mol._bas, sgxopt._intor)

    if grad:
        ncomp = 3
    else:
        ncomp = 1
    nao = mol.nao
    cintor = _vhf._fpointer(sgxopt._intor)
    drv = _vhf.libcvhf.SGXnr_direct_drv
    ncond = lib.c_null_ptr()

    def jk_part(mol, grid_coords, dms, fg, weights):
        atm, bas, env = mol._atm, mol._bas, mol._env
        ngrids = grid_coords.shape[0]
        env = numpy.append(env, grid_coords.ravel())
        env[gto.NGRIDS] = ngrids
        env[gto.PTR_GRIDS] = mol._env.size
        shls_slice = (0, mol.nbas, 0, mol.nbas)

        vj = vk = None
        fjk = []
        dmsptr = []
        vjkptr = []
        if with_j:
            if dms[0].ndim == 1:  # the value of density at each grid
                if grad:
                    vj = numpy.zeros((len(dms),ncomp,nao,nao))
                else:
                    vj = numpy.zeros((len(dms),ncomp,nao,nao))[:,0]
                for i, dm in enumerate(dms):
                    dmsptr.append(dm.ctypes.data_as(ctypes.c_void_p))
                    vjkptr.append(vj[i].ctypes.data_as(ctypes.c_void_p))
                    fjk.append(_vhf._fpointer('SGXnr'+aosym+'_ijg_g_ij'))
            else:
                if grad:
                    vj = numpy.zeros((len(dms),ncomp,ngrids))
                else:
                    vj = numpy.zeros((len(dms),ncomp,ngrids))[:,0]
                for i, dm in enumerate(dms):
                    dmsptr.append(dm.ctypes.data_as(ctypes.c_void_p))
                    vjkptr.append(vj[i].ctypes.data_as(ctypes.c_void_p))
                    fjk.append(_vhf._fpointer('SGXnr'+aosym+'_ijg_ji_g'))
        if with_k:
            assert fg.flags.c_contiguous
            assert fg.shape[-2:] == (ao_loc[-1], weights.size)
            if grad:
                vk = numpy.zeros((len(fg),ncomp,nao,ngrids))
            else:
                vk = numpy.zeros((len(fg),ncomp,nao,ngrids))[:,0]
            for i, dm in enumerate(fg):
                dmsptr.append(dm.ctypes.data_as(ctypes.c_void_p))
                vjkptr.append(vk[i].ctypes.data_as(ctypes.c_void_p))
                fjk.append(_vhf._fpointer('SGXnr'+aosym+'_ijg_gj_gi'))

        n_dm = len(fjk)
        fjk = (ctypes.c_void_p*(n_dm))(*fjk)
        dmsptr = (ctypes.c_void_p*(n_dm))(*dmsptr)
        vjkptr = (ctypes.c_void_p*(n_dm))(*vjkptr)

        drv(cintor, fjk, dmsptr, vjkptr, n_dm, ncomp,
            (ctypes.c_int*4)(*shls_slice),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            sgxopt._cintopt, ctypes.byref(sgxopt._this),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
            env.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(env.shape[0]),
            ctypes.c_int(2 if aosym == 's2' else 1),
            ncond,
            weights.ctypes.data_as(ctypes.c_void_p))
        return vj, vk
    return jk_part


def _gen_k_direct(mol, aosym, direct_scf_tol,
                  sgxopt=None, grad=False):
    '''Contraction between sgX Coulomb integrals and density matrices

    J: einsum('guv,xg->xuv', gbn, dms) if dms == rho at grid,
    or einsum('gij,xij->xg', gbn, dms) if dms are density matrices

    K: einsum('gtv,xgt->xgv', gbn, fg)
    '''
    if sgxopt is None:
        from pyscf.sgx import sgx
        sgxopt = sgx._make_opt(mol, False, grad, direct_scf_tol)
    sgxopt.direct_scf_tol = direct_scf_tol

    ao_loc = moleintor.make_loc(mol._bas, sgxopt._intor)

    if grad:
        ncomp = 3
    else:
        ncomp = 1
    nao = mol.nao
    cintor = _vhf._fpointer(sgxopt._intor)
    drv = _vhf.libcvhf.SGXnr_direct_drv

    def k_part(mol, grid_coords, fg, weights, ncond=None):
        if ncond is None:
            ncond = lib.c_null_ptr()
        else:
            ncond = ncond.ctypes.data_as(ctypes.c_void_p)
        atm, bas, env = mol._atm, mol._bas, mol._env
        ngrids = grid_coords.shape[0]
        env = numpy.append(env, grid_coords.ravel())
        env[gto.NGRIDS] = ngrids
        env[gto.PTR_GRIDS] = mol._env.size
        shls_slice = (0, mol.nbas, 0, mol.nbas)

        vk = None
        fjk = []
        dmsptr = []
        vjkptr = []
        if grad:
            vk = numpy.zeros((len(fg),ncomp,nao,ngrids))
        else:
            vk = numpy.zeros((len(fg),ncomp,nao,ngrids))[:,0]
        for i, dm in enumerate(fg):
            assert fg[i].flags.c_contiguous
            assert fg[i].shape == (ao_loc[-1], weights.size)
            dmsptr.append(dm.ctypes.data_as(ctypes.c_void_p))
            vjkptr.append(vk[i].ctypes.data_as(ctypes.c_void_p))
            fjk.append(_vhf._fpointer('SGXnr'+aosym+'_ijg_gj_gi'))

        n_dm = len(fjk)
        fjk = (ctypes.c_void_p*(n_dm))(*fjk)
        dmsptr = (ctypes.c_void_p*(n_dm))(*dmsptr)
        vjkptr = (ctypes.c_void_p*(n_dm))(*vjkptr)

        drv(cintor, fjk, dmsptr, vjkptr, n_dm, ncomp,
            (ctypes.c_int*4)(*shls_slice),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            sgxopt._cintopt, ctypes.byref(sgxopt._this),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
            env.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(env.shape[0]),
            ctypes.c_int(2 if aosym == 's2' else 1),
            ncond,
            weights.ctypes.data_as(ctypes.c_void_p))
        return vk
    return k_part


SGX_RAD_GRIDS = []
for eps in [3.816, 4.020, 4.338, 4.871, 5.3, 5.8, 6.3, 9.0]:
    nums = []
    for row in range(1, 8):
        nums.append(int(eps * 15 - 40 + 5 * row))
    SGX_RAD_GRIDS.append(nums)
SGX_RAD_GRIDS = numpy.array(SGX_RAD_GRIDS)


def get_gridss(mol, level=1, gthrd=1e-10, use_opt_grids=False):
    Ktime = (logger.process_clock(), logger.perf_counter())
    grids = gen_grid.Grids(mol)
    grids.level = level
    if use_opt_grids:
        grids.becke_scheme = becke_lko
        grids.prune = sgx_prune
        grids.atom_grid = {}
        tab   = numpy.array( (2 , 10, 18, 36, 54, 86, 118))
        for ia in range(mol.natm):
            symb = mol.atom_symbol(ia)
            if symb not in grids.atom_grid:
                chg = gto.charge(symb)
                period = (chg > tab).sum()
                grids.atom_grid[symb] = (SGX_RAD_GRIDS[level, period],
                                         LEBEDEV_ORDER[SGX_ANG_MAPPING[level, 3]])
    grids.build(with_non0tab=True)

    ngrids = grids.weights.size
    mask = []
    for p0, p1 in lib.prange(0, ngrids, 10000):
        ao_v = mol.eval_gto('GTOval', grids.coords[p0:p1])
        ao_v *= grids.weights[p0:p1,None]
        wao_v0 = ao_v
        mask.append(numpy.any(wao_v0>gthrd, axis=1) |
                    numpy.any(wao_v0<-gthrd, axis=1))

    if gthrd > 0:
        mask = numpy.hstack(mask)
        grids.coords = grids.coords[mask]
        grids.weights = grids.weights[mask]
    grids.non0tab = grids.make_mask(mol, grids.coords)
    grids.screen_index = grids.non0tab
    logger.debug(mol, 'threshold for grids screening %g', gthrd)
    logger.debug(mol, 'number of grids %d', grids.weights.size)
    logger.timer_debug1(mol, "Xg screening", *Ktime)
    return grids

get_jk = get_jk_favorj
