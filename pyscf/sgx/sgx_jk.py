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
from pyscf.data.elements import _std_symbol_without_ghost, NUC
from pyscf.data.radii import BRAGG
from pyscf.dft.radi import treutler_ahlrichs
from pyscf.dft.gen_grid import LEBEDEV_ORDER, SGX_ANG_MAPPING, \
                               sgx_prune, becke_lko
from pyscf.dft.numint import eval_ao, BLKSIZE, NBINS, libdft, \
    SWITCH_SIZE, _scale_ao, _dot_ao_ao_sparse, _dot_ao_dm_sparse, \
    _scale_ao_sparse
import time


libdft.SGXreturn_blksize.restype = int
SGX_BLKSIZE = libdft.SGXreturn_blksize()
assert SGX_BLKSIZE % BLKSIZE == 0

SGX_DELTA_1 = 0.1
SGX_DELTA_2 = 0.9
SGX_DELTA_3 = 0.5

_SGX_DELTA_1 = 0.99
_SGX_DELTA_2 = 0.1
_SGX_DELTA_3 = 0.1


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
    # ao is F-order, gv is C-order
    # gu,ivg->iuv
    # in C-order, the operation is ug,ivg->iuv
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


def _get_block_fg(sgx, dms, ao_loc, ao_block_norm):
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
    block_fg = lib.dot(ao_block_norm, shl_dm)
    return block_fg


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


class SGXData:
    def __init__(self, mol, grids, k_only=False,
                 fit_ovlp=True, sym_ovlp=False,
                 max_memory=2000, hermi=0,
                 integral_bound="sample",
                 sgxopt=None):
        # integral bound can be "ovlp", "strict", "sample"
        self.mol = mol
        self.grids = grids
        self.k_only = k_only
        self.fit_ovlp = fit_ovlp
        self.sym_ovlp = sym_ovlp
        self.max_memory = max_memory
        self.hermi = hermi
        self._loop_data = None
        self._nquad = 200
        self._nrad = 32
        self._rcmul = 2.0
        self.integral_bound = integral_bound
        self._opt = sgxopt
        self.use_dm_screening = True  # TODO set
        self.reset()

    def _get_loop_data(self, nset=1):
        mol = self.mol
        grids = self.grids
        nao = mol.nao_nr()
        ao_loc = mol.ao_loc_nr()
        if (grids.coords is None or grids.non0tab is None
                or grids.atm_idx is None):
            grids.build(with_non0tab=True, with_ialist=True)
        self._sgx_block_cond = None
        if mol is grids.mol:
            non0tab = grids.non0tab
        if non0tab is None:
            non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                                    dtype=numpy.uint8)
            non0tab[:] = NBINS + 1  # Corresponding to AO value ~= 1
        screen_index = non0tab
        ngrids = grids.coords.shape[0]
        max_memory = self.max_memory - lib.current_memory()[0]
        # We need to store ao, wao, and fg -> 2 + nset sets of size nao
        blksize = max(112, int(max_memory*1e6/8/((2+nset)*nao)))
        blksize = min(ngrids, max(1, blksize // SGX_BLKSIZE) * SGX_BLKSIZE)
        cutoff = grids.cutoff * 1e2
        nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))
        pair_mask = mol.get_overlap_cond() < -numpy.log(cutoff)
        self._loop_data = [
            nbins, screen_index, pair_mask, ao_loc, blksize
        ]

    @property
    def loop_data(self):
        if self._loop_data is None:
            self._get_loop_data()
        return self._loop_data

    def _build_integral_bounds(self):
        mol = self.mol
        mbar_ij = numpy.zeros((mol.nbas, mol.nbas))
        drv = _vhf.libcvhf.SGXsample_ints
        atm_coords = numpy.ascontiguousarray(
            mol.atom_coords(unit='Bohr')
        )
        ao_loc = mol.ao_loc_nr().astype(numpy.int32)
        if self._opt is None:
            from pyscf.sgx.sgx import _make_opt
            sgxopt = _make_opt(mol, False, 1e-16)
        else:
            sgxopt = self._opt
        cintor = _vhf._fpointer(sgxopt._intor)
        atm, bas, env = mol._atm, mol._bas, mol._env
        ngmax = self._nquad + self._nrad + 2
        env = numpy.append(env, numpy.zeros(3 * ngmax))
        env[gto.NGRIDS] = ngmax
        env[gto.PTR_GRIDS] = mol._env.size
        rs = {}
        radii = {}
        elements = [x for x, _ in mol._atom]
        for el in set(elements):
            a = _std_symbol_without_ghost(el)
            z = NUC[a]
            radii[el] = BRAGG[z]
            rs[el] = treutler_ahlrichs(self._nrad, z)[0]
        radii = numpy.asarray(
            [radii[el] for el in elements], order="C", dtype=numpy.float64
        )
        rs = numpy.asarray(
            [rs[el] for el in elements], order="C", dtype=numpy.float64
        )
        self._check_arr(mbar_ij, dtype=numpy.float64)
        self._check_arr(radii, dtype=numpy.float64)
        self._check_arr(atm_coords, dtype=numpy.float64)
        self._check_arr(rs, dtype=numpy.float64)
        self._check_arr(ao_loc, dtype=numpy.int32)
        assert sgxopt._cintopt is not None
        self._check_arr(atm, dtype=numpy.int32)
        self._check_arr(bas, dtype=numpy.int32)
        self._check_arr(env, dtype=numpy.float64)
        #print(sgxopt._intor)
        #vals = locals()
        #for term in ["mbar_ij", "radii", "atm_coords", "rs", "ao_loc", "atm", "bas", "env"]:
        #    numpy.save(term + ".npy", vals[term])
        drv(
            cintor, mbar_ij.ctypes, None, radii.ctypes,
            ctypes.c_int(self._nquad), ctypes.c_int(self._nrad),
            ctypes.c_double(self._rcmul), atm_coords.ctypes,
            rs.ctypes, ao_loc.ctypes, sgxopt._cintopt,
            atm.ctypes, ctypes.c_int(mol.natm), bas.ctypes,
            ctypes.c_int(mol.nbas), env.ctypes, ctypes.c_int(env.size)
        )
        self._mbar_ij = mbar_ij
        self._opt.q_cond = mbar_ij

    def _build_ovlp_fit(self):
        mol = self.mol
        grids = self.grids
        nao = mol.nao_nr()
        sn = numpy.zeros((nao,nao))
        nbins, screen_index, pair_mask, ao_loc, blksize = self.loop_data
        ngrids = grids.weights.size
        if self.fit_ovlp:
            sn = numpy.zeros((nao, nao))
            for i0, i1 in lib.prange(0, ngrids, blksize):
                assert i0 % SGX_BLKSIZE == 0
                coords = grids.coords[i0:i1]
                mask = screen_index[i0 // BLKSIZE:]
                ao = mol.eval_gto('GTOval', coords, non0tab=mask)
                _dot_ao_ao_sparse(ao, ao, grids.weights[i0:i1], nbins, mask,
                                  pair_mask, ao_loc, hermi=self.hermi, out=sn)
                # wao = _scale_ao(ao, grids.weights[i0:i1])
                # sn[:] += _dot_ao_ao(mol, ao, wao, mask, (0, mol.nbas),
                #                     ao_loc, hermi=hermi)
            ovlp = mol.intor_symmetric('int1e_ovlp')
            proj = scipy.linalg.solve(sn, ovlp)
            if self.fit_ovlp:
                proj = 0.5 * (proj + numpy.identity(nao))
            self._overlap_correction_matrix = proj
        else:
            self._overlap_correction_matrix = numpy.identity(nao)

    def _check_arr(self, x, order="C", dtype=None):
        if dtype is not None:
            assert x.dtype == dtype
        if order == "C":
            assert x.flags.c_contiguous
        elif order == "F":
            assert x.flags.f_contiguous
        else:
            raise ValueError

    def _make_shl_mat(self, x_gu, x_bi, rlocs, clocs, wt=None):
        self._check_arr(x_gu, dtype=numpy.float64)
        self._check_arr(x_bi, dtype=numpy.float64)
        self._check_arr(rlocs, dtype=numpy.int32)
        self._check_arr(clocs, dtype=numpy.int32)
        assert x_gu.shape[-2:] == (rlocs[-1], clocs[-1])
        assert x_bi.shape == (rlocs.size - 1, clocs.size - 1)
        args = [
            x_gu.ctypes,
            x_bi.ctypes,
            ctypes.c_int(rlocs.size - 1),
            ctypes.c_int(clocs.size - 1),
            rlocs.ctypes,
            clocs.ctypes,
        ]
        if wt is None:
            fn = _vhf.libcvhf.SGXmake_shl_mat
            assert x_gu.ndim == 2
        else:
            if x_gu.ndim == 2:
                ncomp = 1
            else:
                assert x_gu.ndim == 3
                ncomp = x_gu.shape[0]
            self._check_arr(wt, dtype=numpy.float64)
            args.append(wt.ctypes)
            args.append(ncomp)
            fn = _vhf.libcvhf.SGXmake_shl_mat_wt
        fn(*args)

    def _pow_screen(self, xbar_ij, delta):
        # sum is over outer index here
        # This is done in-place
        self._check_arr(xbar_ij, dtype=numpy.float64)
        n, m = xbar_ij.shape
        _vhf.libcvhf.SGXpow_screen(
            xbar_ij.ctypes,
            ctypes.c_int(n),
            ctypes.c_int(m),
            ctypes.c_double(delta),
        )

    def _sqrt_screen(self, xbar_ij, eps):
        # sum is over outer index here
        # This is done in-place
        self._check_arr(xbar_ij, dtype=numpy.float64)
        n, m = xbar_ij.shape
        _vhf.libcvhf.switch_screen(
            xbar_ij.ctypes,
            ctypes.c_int(n),
            ctypes.c_int(m),
            ctypes.c_double(eps),
        )

    def _einmin(self, out, mat0, mat1):
        # double *res, double *mat0, double *mat1, int m, int n, int k
        m, n = out.shape
        k = mat0.shape[1]
        assert mat0.shape == (m, k)
        assert mat1.shape == (n, k)
        self._check_arr(out, dtype=numpy.float64)
        self._check_arr(mat0, dtype=numpy.float64)
        self._check_arr(mat1, dtype=numpy.float64)
        _vhf.libcvhf.SGXeinmin(
            out.ctypes,
            mat0.ctypes,
            mat1.ctypes,
            ctypes.c_int(m),
            ctypes.c_int(n),
            ctypes.c_int(k),
        )

    def _build_dm_screen(self):
        ta = logger.perf_counter()
        mol = self.mol
        grids = self.grids
        ngrids = grids.weights.size
        nao = mol.nao_nr()
        nbins, screen_index, pair_mask, ao_loc, blksize = self.loop_data
        nblk = (grids.weights.size + SGX_BLKSIZE - 1) // SGX_BLKSIZE
        nblk_ni = (grids.weights.size + BLKSIZE - 1) // BLKSIZE
        xsgx_bi = numpy.empty((nblk, mol.nbas), dtype=numpy.float64)
        xni_bi = numpy.empty((nblk_ni, mol.nbas), dtype=numpy.float64)
        ao = None
        dft_locs = BLKSIZE * numpy.arange(nblk_ni, dtype=numpy.int32)
        if grids.weights.size >= 2**31:
            raise ValueError("Too many grids for signed int32")
        dft_locs[-1] = grids.weights.size
        dft_per_sgx = SGX_BLKSIZE // BLKSIZE
        ovlp = mol.intor_symmetric("int1e_ovlp")
        if not ovlp.flags.c_contiguous:
            ovlp = ovlp.T
        stilde_ij = numpy.zeros((mol.nbas, mol.nbas))
        self._make_shl_mat(
            ovlp, stilde_ij, ao_loc, ao_loc,
        )
        self._sbar_ij = stilde_ij.copy()
        self._pow_screen(stilde_ij, _SGX_DELTA_2)
        self._stilde_ij = stilde_ij
        self._mtilde_ij = self._mbar_ij.copy()
        self._pow_screen(self._mtilde_ij, _SGX_DELTA_2)
        for i0, i1 in lib.prange(0, ngrids, blksize):
            assert i0 % SGX_BLKSIZE == 0
            nblk_ni_curr = (i1 - i0 + BLKSIZE - 1) // BLKSIZE
            ni_locs = BLKSIZE * numpy.arange(
                nblk_ni_curr + 1, dtype=numpy.int32
            )
            ni_locs[-1] = i1 - i0
            coords = grids.coords[i0:i1]
            mask = screen_index[i0 // BLKSIZE:]
            ao = mol.eval_gto('GTOval', coords, non0tab=mask, out=ao)
            rtwt = numpy.sqrt(numpy.abs(grids.weights[i0:i1]))
            x_gu = _scale_ao_sparse(ao, rtwt, mask, ao_loc)   # ao * rtwt[:, None]  # TODO scale ao
            print(x_gu.shape, ao.shape, rtwt.shape)
            print(x_gu.flags.c_contiguous, ao.flags.c_contiguous)
            x_gu = numpy.ascontiguousarray(x_gu)
            print(x_gu.shape, xni_bi[i0 // BLKSIZE:].shape, ni_locs.shape,
                  ao_loc.shape, ni_locs[-1], ao_loc[-1])
            ao_loc = ao_loc.astype(numpy.int32)
            xtmp_bi = numpy.empty((nblk_ni_curr, mol.nbas))
            self._make_shl_mat(x_gu, xtmp_bi, ni_locs, ao_loc)
            b0 = i0 // BLKSIZE
            xni_bi[b0:b0 + nblk_ni_curr] = xtmp_bi
        locs = dft_per_sgx * numpy.arange(nblk + 1, dtype=numpy.int32)
        locs[-1] = xni_bi.shape[0]
        ones = numpy.arange(mol.nbas + 1, dtype=numpy.int32)
        print(xsgx_bi.shape, locs.size, ones.size)
        self._make_shl_mat(xni_bi, xsgx_bi, locs, ones)
        self._pow_screen(xni_bi, _SGX_DELTA_3)
        self._pow_screen(xsgx_bi, _SGX_DELTA_3)
        self._buf = xni_bi
        self._xni_b = numpy.min(xni_bi, axis=1)
        self._xsgx_b = numpy.min(xsgx_bi, axis=1)
        tc = logger.perf_counter()
        print("SETUP TOOK", tc - ta)

    def reset(self):
        self._ovlp_proj = numpy.identity(self.mol.nao_nr())
        self._mbar_ij = None
        self._stilde_ij = None
        self._mtilde_ij = None
        self._mbar_ij = None
        self._xni_b = None
        self._xsgx_b = None

    def build(self):
        self._build_integral_bounds()
        if self.fit_ovlp:
            self._ovlp_proj = self._build_ovlp_fit()
        if self.use_dm_screening:
            self._build_dm_screen()

    def get_dm_threshold_matrix(self, dm, dv, de):
        mol = self.mol
        dm_ij = numpy.zeros((mol.nbas, mol.nbas))
        if dm.ndim == 3:
            dm = dm.sum(axis=0)
        ao_loc = mol.ao_loc_nr()
        self._make_shl_mat(dm, dm_ij, ao_loc, ao_loc)
        tmp = lib.einsum("lk,jk->lj", self._stilde_ij, dm_ij)
        Z_il = lib.einsum("ij,lj->il", self._mtilde_ij, tmp) + 1e-200
        thresh_ij = de / Z_il
        Y_il = numpy.max(self._mtilde_ij, axis=1)
        Y_il = Y_il * numpy.max(self._stilde_ij, axis=1)[:, None]
        Y_il = numpy.maximum(Y_il, Y_il.T) + 1e-200
        thresh_ij = numpy.minimum(dv / Y_il, thresh_ij)
        return dm_ij > thresh_ij
    
    def get_dm_threshold_matrix2(self, dm, dv, de):
        mol = self.mol
        dm_ij = numpy.zeros((mol.nbas, mol.nbas))
        if dm.ndim == 3:
            dm = dm.sum(axis=0)
        ao_loc = mol.ao_loc_nr()
        self._make_shl_mat(dm, dm_ij, ao_loc, ao_loc)
        inds = self._argsort(dm_ij)
        sdm_ij = self._cumsum(dm_ij, inds)
        sdm_ij = 0.5 * (sdm_ij + sdm_ij.T)
        tmp = lib.einsum("lk,jk->lj", self._sbar_ij, dm_ij)
        Z_il = lib.einsum("ij,lj->il", self._mbar_ij, tmp) + 1e-200
        PZ = sdm_ij * Z_il
        PY = sdm_ij * numpy.max(self._mbar_ij, axis=1) * numpy.max(self._sbar_ij, axis=1)[:, None]
        SPZ = self._cumsum(PZ, self._argsort(PZ))
        SPY = self._cumsum(PY, self._argsort(PY))
        SPZ = 0.5 * (SPZ + SPZ.T)
        SPY = 0.5 * (SPY + SPY.T)
        return numpy.logical_or(SPZ > de / mol.nbas, SPY > dv / mol.nbas)
        thresh_ij = de / Z_il
        Y_il = numpy.max(self._mbar_ij, axis=1)
        Y_il = Y_il * numpy.max(self._sbar_ij, axis=1)[:, None]
        Y_il = numpy.maximum(Y_il, Y_il.T) + 1e-200
        thresh_ij = numpy.minimum(dv / Y_il, thresh_ij)
        return sdm_ij > thresh_ij

    def _argsort(self, f):
        assert f.ndim == 2
        self._check_arr(f, dtype=numpy.float64)
        _inds = numpy.empty_like(f, dtype=numpy.int32)
        _vhf.libcvhf.SGXargsort_lists(
            _inds.ctypes,
            f.ctypes,
            ctypes.c_int(f.shape[0]),
            ctypes.c_int(f.shape[1]),
        )
        return _inds

    def _cumsum(self, f, inds):
        assert f.ndim == 2
        self._check_arr(f, dtype=numpy.float64)
        out = numpy.empty_like(f)
        _vhf.libcvhf.SGXcumsum_lists(
            out.ctypes,
            f.ctypes,
            inds.ctypes,
            ctypes.c_int(f.shape[0]),
            ctypes.c_int(f.shape[1]),
        )
        return out

    """
    def _get_g_thresh(self, f_ug, b0, dv, de, wt, blksize, x_b):
        t0 = time.monotonic()
        assert f_ug.flags.c_contiguous
        nao, ng = f_ug.shape[-2:]
        b1 = (ng + blksize - 1) // blksize + b0
        thresh = dv * x_b[b0:b1]
        fbar_ib = numpy.ndarray((self.mol.nbas, b1 - b0), buffer=self._buf)
        # ao_loc = self.mol.ao_loc_nr()
        ao_loc = self._loop_data[3]
        assert nao == ao_loc[-1]
        blk_loc = blksize * numpy.arange(b1 - b0 + 1, dtype=numpy.int32)
        blk_loc[-1] = ng
        t1 = time.monotonic()
        self._make_shl_mat(f_ug, fbar_ib, ao_loc, blk_loc, wt=wt)
        t2 = time.monotonic()
        _ftmp_bi = fbar_ib.T.copy()
        self._pow_screen(fbar_ib, _SGX_DELTA_1)
        _inds1_bi = self._argsort(_ftmp_bi * numpy.max(self._mbar_ij, axis=1))
        _fm_bi = _ftmp_bi.dot(self._mbar_ij)
        _fmf_bi = _fm_bi * _ftmp_bi
        _inds3_bi = self._argsort(_fmf_bi)
        _sum1_bi = self._cumsum(_ftmp_bi, _inds1_bi)
        _sum3_bi = self._cumsum(_fmf_bi, _inds3_bi)
        cond1 = _sum1_bi > thresh[:, None]
        cond2 = _fm_bi > thresh[:, None]
        cond3 = _sum3_bi > de / x_b.size
        #cond1 = _ftmp_bi > thresh[:, None] / numpy.max(self._mbar_ij, axis=1) / self.mol.nbas
        #cond2 = _fm_bi > thresh[:, None]
        #cond3 = _fmf_bi > de / x_b.size / self.mol.nbas
        cond4 = numpy.logical_and(cond1, cond2)
        cond5 = numpy.logical_or(cond1, cond3)
        cond6 = (self._mbar_ij.dot(1.0 / fbar_ib) > de * fbar_ib / x_b.size).T
        print("RESULTS", cond1.shape)
        print([cond.sum(1).mean() for cond in [cond1, cond2, cond3, cond4, cond5, cond6]])
        t3 = time.monotonic()
        fbar_bi = numpy.ascontiguousarray(fbar_ib.T)
        # bni_bi = numpy.minimum(thresh[:, None], (de / x_b.size) * fbar_bi)
        bni_bi = numpy.minimum(thresh[:, None], (de / cond3.sum(1))[:, None] * fbar_bi)
        bni_bi = numpy.ascontiguousarray(bni_bi)
        t4 = time.monotonic()
        print("GTIMES", t1 - t0, t2 - t1, t3 - t2, t4 - t3)
        # return fbar_bi, bni_bi
        print("TYPE", cond5.flags.c_contiguous, cond5.shape, cond5.dtype)
        return _ftmp_bi, bni_bi, cond5

    def get_g_threshold(self, f_ug, b0, dv, de, wt):
        ftilde_bi, bsgx_bi, shl_screen = self._get_g_thresh(
            f_ug, b0, dv, de, wt, SGX_BLKSIZE, self._xsgx_b
        )
        _, bni_bi, _ = self._get_g_thresh(
            f_ug, b0 * SGX_BLKSIZE // BLKSIZE, dv, de, wt, BLKSIZE, self._xni_b
        )
        return bni_bi, bsgx_bi, ftilde_bi, shl_screen
    """
    
    def _sgx_thresh(self, f_ug, b0, dv, de, wt, blksize, x_b):
        t0 = time.monotonic()
        assert f_ug.flags.c_contiguous
        nao, ng = f_ug.shape[-2:]
        b1 = (ng + blksize - 1) // blksize + b0
        thresh = dv * x_b[b0:b1]
        fbar_ib = numpy.ndarray((self.mol.nbas, b1 - b0), buffer=self._buf)
        ao_loc = self._loop_data[3]
        assert nao == ao_loc[-1]
        blk_loc = blksize * numpy.arange(b1 - b0 + 1, dtype=numpy.int32)
        blk_loc[-1] = ng
        t1 = time.monotonic()
        self._make_shl_mat(f_ug, fbar_ib, ao_loc, blk_loc, wt=wt)
        t2 = time.monotonic()
        fbar_bi = numpy.ascontiguousarray(fbar_ib.T)
        inds1_bi = self._argsort(fbar_bi * numpy.max(self._mbar_ij, axis=1))
        fmf_bi = lib.dot(fbar_bi, self._mbar_ij)
        fmf_bi[:] *= fbar_bi
        inds2_bi = self._argsort(fmf_bi)
        sum1_bi = self._cumsum(fbar_bi, inds1_bi)
        sum2_bi = self._cumsum(fmf_bi, inds2_bi)
        cond = sum1_bi > thresh[:, None]
        cond = numpy.logical_or(sum2_bi > de / x_b.size, cond)
        print("RESULTS", cond.shape)
        print(cond.sum(1).mean())
        t3 = time.monotonic()
        denom = x_b.size * cond.sum(1)
        b_bi = numpy.minimum(thresh[:, None], (de / denom)[:, None] / (fbar_bi + 1e-200))
        b_bi = numpy.ascontiguousarray(b_bi)
        t4 = time.monotonic()
        print("GTIMES", t1 - t0, t2 - t1, t3 - t2, t4 - t3)
        return fbar_bi, b_bi, cond
    
    def _ni_thresh(self, f_ug, b0, dv, de, wt, blksize, x_b):
        assert f_ug.flags.c_contiguous
        nao, ng = f_ug.shape[-2:]
        b1 = (ng + blksize - 1) // blksize + b0
        thresh = dv * x_b[b0:b1]
        fbar_ib = numpy.ndarray((self.mol.nbas, b1 - b0), buffer=self._buf)
        ao_loc = self._loop_data[3]
        assert nao == ao_loc[-1]
        blk_loc = blksize * numpy.arange(b1 - b0 + 1, dtype=numpy.int32)
        blk_loc[-1] = ng
        self._make_shl_mat(f_ug, fbar_ib, ao_loc, blk_loc, wt=wt)
        fbar_bi = numpy.ascontiguousarray(fbar_ib.T)
        inds_bi = self._argsort(fbar_bi)
        sum_bi = self._cumsum(fbar_bi, inds_bi)
        b_bi = numpy.minimum(thresh[:, None], de / (x_b.size * sum_bi + 1e-64))
        return b_bi

    def get_g_threshold(self, f_ug, b0, dv, de, wt):
        t0 = time.monotonic()
        fbar_bi, bsgx_bi, shl_screen = self._sgx_thresh(
            f_ug, b0, dv, de, wt, SGX_BLKSIZE, self._xsgx_b
        )
        bni_bi = self._ni_thresh(
            f_ug, b0 * (SGX_BLKSIZE // BLKSIZE), dv, de, wt, BLKSIZE, self._xni_b
        )
        t1 = time.monotonic()
        print("TOTAL GTIME", t1 - t0)
        return bni_bi, bsgx_bi, fbar_bi, shl_screen

def run_k_only_setup(sgx, dms, hermi):
    mol = sgx.mol
    grids = sgx.grids
    nao = dms.shape[-1]
    nset = dms.shape[0]
    ao_loc = mol.ao_loc_nr()
    t0 = logger.perf_counter()

    if grids.coords is None or grids.non0tab is None or grids.atm_idx is None:
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
            if sgx._symm_ovlp_fit:
                proj = 0.5 * (proj + numpy.identity(nao))
            sgx._overlap_correction_matrix = proj
        else:
            sgx._overlap_correction_matrix = numpy.identity(nao)
    proj = sgx._overlap_correction_matrix

    if sgx.use_dm_screening and sgx._sgx_block_cond is None:
        ta = logger.perf_counter()
        cutoff = grids.cutoff * 1e2
        nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))
        pair_mask = mol.get_overlap_cond() < -numpy.log(cutoff)
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

    return blksize, screen_index, proj, dm_mask, ncond, ncond_ni, ao_loc


def get_k_only(sgx, dm, hermi=1, direct_scf_tol=1e-13, full_dm=None):
    if sgx.debug:
        raise NotImplementedError("debug mode for accelerated K matrix")

    t0 = logger.process_clock(), logger.perf_counter()
    mol = sgx.mol
    grids = sgx.grids

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    t1 = logger.timer_debug1(mol, "sgX initialization", *t0)
    vk = numpy.zeros_like(dms)
    tnuc = 0, 0
    shls_slice = (0, mol.nbas)
    sgx_data = run_k_only_setup(sgx, dms, hermi)
    ngrids = grids.weights.size
    blksize, screen_index, proj, dm_mask, ncond, ncond_ni, ao_loc = sgx_data
    proj_dm = lib.einsum('ki,xij->xkj', proj, dms)

    batch_k = _gen_k_direct(mol, 's2', direct_scf_tol, sgx._opt)

    if True:
        assert sgx._opt is not None
        if not hasattr(sgx, "_pjs_data") or sgx._pjs_data is None:
            sgx._pjs_data = SGXData(
                mol,
                grids,
                k_only=True,
                fit_ovlp=True,
                sym_ovlp=False,
                max_memory=sgx.max_memory,
                hermi=1,
                integral_bound="sample",
                sgxopt=sgx._opt,
            )
            sgx._pjs_data.build()
        dm_mask = sgx._pjs_data.get_dm_threshold_matrix2(
            dms, sgx.sgx_tol_potential, sgx.sgx_tol_energy
        )

    if full_dm is None:
        full_dm = dms
    else:
        full_dm = numpy.asarray(full_dm)
        full_dm = full_dm.reshape(-1,nao,nao)
    sgx._block_fg = _get_block_fg(sgx, full_dm, ao_loc, sgx._ao_block_norm)
    # divide by the number of blocks to get effective energy tolerance,
    # since we are going to sum over blocks
    sgx._etol = sgx.sgx_tol_energy / sgx._block_fg.shape[0]

    t_setup = logger.perf_counter()
    print("AFTER_SETUP", t_setup - t0[1])

    v2 = True

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
        tnuc = tnuc[0] - logger.process_clock(), tnuc[1] - logger.perf_counter()
        tc = logger.perf_counter()
        if sgx.use_dm_screening:
            fg = _sgxdot_ao_dm_sparse(ao, proj_dm, mask, dm_mask, ao_loc, out=fg)
            bni_bi, bsgx_bi, fbar_bi, shl_screen = sgx._pjs_data.get_g_threshold(
                fg, i0 // SGX_BLKSIZE, sgx.sgx_tol_potential,
                sgx.sgx_tol_energy, weights
            )
        else:
            # fg = lib.einsum('xij,ig->xjg', proj_dm, wao.T)
            fg = _sgxdot_ao_dm(ao, proj_dm, mask, shls_slice, ao_loc, out=fg)
        te = logger.perf_counter()

        if sgx.use_dm_screening:
            gv = batch_k(mol, coords, fg, weights=(fbar_bi, bsgx_bi, shl_screen), v2=True)
        else:
            gv = batch_k(mol, coords, fg, weights)
        tnuc = tnuc[0] + logger.process_clock(), tnuc[1] + logger.perf_counter()
        if sgx.use_dm_screening:
            assert gv.shape[2] == weights.size
            b0 = i0 // BLKSIZE
            b1 = b0 + (weights.size + BLKSIZE - 1) // BLKSIZE
            assert sgx._pjs_data._xni_b.shape[0] >= b1
            clocs = BLKSIZE * numpy.arange(b1 - b0 + 1, dtype=numpy.int32)
            clocs[-1] = gv.shape[2]
            g_ib = numpy.empty((mol.nbas, b1 - b0))
            sgx._pjs_data._make_shl_mat(gv, g_ib, ao_loc, clocs, wt=weights)
            mask2 = g_ib.T > bni_bi
        else:
            mask2 = None
        tf = logger.perf_counter()
        _sgxdot_ao_gv_sparse(ao, gv, weights, mask, mask2, ao_loc, out=vk)
        tg = logger.perf_counter()
        gv = None
        t_ao += tb - ta
        t_scale += tc - tb
        t_ao_dm += te - tc
        t_int += tf - te
        t_ao_ao += tg - tf
    print("Times", t_ao, t_scale, t_ao_dm, t_int, t_ao_ao)
    if sgx._symm_ovlp_fit:
        vk = lib.einsum('ik,xij->xkj', proj, vk)

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
        sgxopt = sgx._make_opt(mol, grad, direct_scf_tol)
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
        sgxopt = sgx._make_opt(mol, grad, direct_scf_tol)
    sgxopt.direct_scf_tol = direct_scf_tol

    ao_loc = moleintor.make_loc(mol._bas, sgxopt._intor)

    if grad:
        ncomp = 3
    else:
        ncomp = 1
    nao = mol.nao
    cintor = _vhf._fpointer(sgxopt._intor)
    drv = _vhf.libcvhf.SGXnr_direct_drv

    def k_part(mol, grid_coords, fg, weights, ncond=None,
               econd=None, etol=None, v2=False):
        if ncond is None:
            ncond = lib.c_null_ptr()
            etol = 0
            econd = numpy.empty((1,1))
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
            assert fg[i].shape == (ao_loc[-1], ngrids)
            dmsptr.append(dm.ctypes.data_as(ctypes.c_void_p))
            vjkptr.append(vk[i].ctypes.data_as(ctypes.c_void_p))
            fjk.append(_vhf._fpointer('SGXnr'+aosym+'_ijg_gj_gi'))

        n_dm = len(fjk)
        fjk = (ctypes.c_void_p*(n_dm))(*fjk)
        dmsptr = (ctypes.c_void_p*(n_dm))(*dmsptr)
        vjkptr = (ctypes.c_void_p*(n_dm))(*vjkptr)

        if v2:
            ftilde_bi = weights[0]
            bscreen_bi = weights[1]
            args = [
                cintor, fjk, dmsptr, vjkptr, n_dm, ncomp,
                (ctypes.c_int*4)(*shls_slice),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                sgxopt._cintopt, ctypes.byref(sgxopt._this),
                atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
                env.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(2 if aosym == 's2' else 1),
                ftilde_bi.ctypes.data_as(ctypes.c_void_p),
                bscreen_bi.ctypes.data_as(ctypes.c_void_p),
            ]
            if len(weights) == 2:
                fn = _vhf.libcvhf.SGXnr_direct_drv2
            else:
                fn = _vhf.libcvhf.SGXnr_direct_drv3
                args.append(weights[2].ctypes)
            fn(*args)
        else:
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
                weights.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_double(etol),
                econd.ctypes.data_as(ctypes.c_void_p))
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
