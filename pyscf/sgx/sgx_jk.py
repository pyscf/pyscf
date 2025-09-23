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
_SGX_DELTA_3 = 0.01


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


class _CSGXOpt(ctypes.Structure):
    __slots__ = []
    _fields_ = [('mode', ctypes.c_int),
                ('nbas', ctypes.c_int),
                ('ngrids', ctypes.c_int),
                ('ao_loc', ctypes.c_void_p),
                ('etol', ctypes.c_double),
                ('vtol', ctypes.c_void_p),
                ('wt', ctypes.c_void_p),
                #('fbar_i', ctypes.c_void_p),
                #('bscreen_i', ctypes.c_void_p),
                #('shlscreen_i', ctypes.c_void_p),
                ('mbar_ij', ctypes.c_void_p),
                ('mbar_bi', ctypes.c_void_p),
                ('rbar_ij', ctypes.c_void_p),
                ('mmax_i', ctypes.c_void_p),
                ('msum_i', ctypes.c_void_p),
                ('buf_size_bytes', ctypes.c_size_t),
                ('shl_info_size_bytes', ctypes.c_size_t),
                ('direct_scf_tol', ctypes.c_double),
                ('fscreen_grid', ctypes.c_void_p),
                ('fscreen_shl', ctypes.c_void_p)]


def _arr2c(arr):
    return None if arr is None else arr.ctypes.data_as(ctypes.c_void_p)


class SGXData:
    def __init__(self, mol, grids, k_only=False,
                 fit_ovlp=True, sym_ovlp=False,
                 max_memory=2000, hermi=0,
                 integral_bound="sample",
                 sgxopt=None, etol=1e-10,
                 vtol=1e-5, upper_bound_algo="sort1"):
        """
        integral bound can be "ovlp", "strict", "sample"
        upper_bound_algo can be
            "norm": Set the upper bound based on the
                power-screened norm of the density matrix
            "sort1": Set the upper bound based on the
                cumulative sum of sorted density matrix
                contributions.
            "sort2": Same as "sort1" but the outer
                (batchwise) loop of screening is faster
            "fullsort": Same as "sort1" but the outer screening
                loop sorts all shell pair contributions, which
                is more costly and memory intensive but leads
                to stricter thresholding of the energy.
        For each upper bound algo, screening can be done based
        on energy, potential, or both
        """
        self.mol = mol
        self.grids = grids
        self.k_only = k_only
        self.fit_ovlp = fit_ovlp
        self.sym_ovlp = sym_ovlp
        self.max_memory = max_memory
        self.hermi = hermi
        # Use position-dependent screening of integrals
        # (speeds up short-range exchange)
        self._build_rdpt_screen = False
        self._etol = etol
        self._screen_energy = True
        self._vtol = vtol
        self._screen_potential = True
        self._nquad = 200
        self._nrad = 32
        self._rcmul = 2.0
        self.integral_bound = integral_bound
        self.upper_bound_algo = upper_bound_algo
        self._opt = sgxopt
        self.use_dm_screening = True  # TODO set
        self._this = _CSGXOpt()

    def reset(self):
        self._ovlp_proj = numpy.identity(self.mol.nao_nr())
        self._stilde_ij = None
        self._mtilde_ij = None
        self._xni_b = None
        self._xsgx_b = None

        # _this properties
        self._mbar_ij = None
        self._etol_per_sgx_block = None
        self._vtol_arr = None
        self._mbar_ij = None
        self._mmax_i = None
        self._msum_i = None
        self._rbar_ij = None
        self._mbar_bi = None

    @property
    def mol(self):
        return self._mol
    @mol.setter
    def mol(self, mol):
        self._mol = mol
        self._ao_loc = self.mol.ao_loc_nr()
        self.reset()

    @property
    def grids(self):
        return self._grids
    @grids.setter
    def grids(self, grids):
        self._grids = grids
        self.reset()

    def _setup_opt(self):
        self._this.mode = 0
        self._this.nbas = self.mol.nbas
        self._this.ngrids = self.grids.weights.size
        self._this.ao_loc = _arr2c(self._ao_loc)
        self._this.etol = self._etol_per_sgx_block
        self._this.vtol = _arr2c(self._vtol_arr)
        self._this.wt = _arr2c(self.grids.weights)
        self._this.mbar_ij = _arr2c(self._mbar_ij)
        self._this.mmax_i = _arr2c(self._mmax_i)
        self._this.msum_i = _arr2c(self._msum_i)
        if self._build_rdpt_screen:
            self._this.rbar_ij = _arr2c(self._rbar_ij)
            self._this.mbar_bi = _arr2c(self._mbar_bi)
        # buf can contain the following:
        # For batch screening step
        #   Nothing for no dm screening
        #   3 * nbas * double, 2 * nbas * int for sort1
        #   5 * nbas * double, 2 * nbas * int for sort2
        # For the shell screening step
        #   Nothing for no dm screening
        #   3 * nbas * double, 3 nbas * int for sort1 & sort2
        dsize = self.mol.nbas * ctypes.sizeof(ctypes.c_double)
        isize = self.mol.nbas * ctypes.sizeof(ctypes.c_int)
        bsize = self.mol.nbas * ctypes.sizeof(ctypes.c_uint8)
        self._this.buf_size_bytes = 5 * dsize + 3 * isize
        # shl_info can contain the following:
        #   shlscreen_i (nbas * uint8_t)
        #   bscreen_i (nbas * double)
        #   fbar_i (nbas * double)
        #   fmf_ij (nbas^2 * double)
        sisb = bsize + 2 * dsize
        if self.upper_bound_algo == "fullsort":
            sisb += dsize * self.mol.nbas
        self._this.shl_info_size_bytes = sisb
        self._this.fscreen_grid = None  # TODO _vhf._fpointer(...)
        self._this.fscreen_shl = None  # TODO _vhf._fpointer(...)

    def get_loop_data(self, nset=1, with_pair_mask=True):
        mol = self.mol
        grids = self.grids
        nao = mol.nao_nr()
        ao_loc = mol.ao_loc_nr()
        if (grids.coords is None or grids.non0tab is None
                or grids.atm_idx is None):
            grids.build(with_non0tab=True)
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
        blksize = max(SGX_BLKSIZE, int(max_memory*1e6/8/((2+nset)*nao)))
        blksize = min(ngrids, max(1, blksize // SGX_BLKSIZE) * SGX_BLKSIZE)
        cutoff = grids.cutoff * 1e2
        nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))
        if with_pair_mask:
            pair_mask = mol.get_overlap_cond() < -numpy.log(cutoff)
        else:
            pair_mask = None
        self._ao_loc = ao_loc
        return nbins, screen_index, pair_mask, ao_loc, blksize

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
        drv(
            cintor, mbar_ij.ctypes, None, radii.ctypes,
            ctypes.c_int(self._nquad), ctypes.c_int(self._nrad),
            ctypes.c_double(self._rcmul), atm_coords.ctypes,
            rs.ctypes, ao_loc.ctypes, sgxopt._cintopt,
            atm.ctypes, ctypes.c_int(mol.natm), bas.ctypes,
            ctypes.c_int(mol.nbas), env.ctypes, ctypes.c_int(env.size)
        )
        self._mbar_ij = mbar_ij
        self._mmax_i = numpy.max(mbar_ij, axis=1)
        self._msum_i = numpy.sum(mbar_ij, axis=1)
        self._opt.q_cond = mbar_ij
        if self._build_rdpt_screen:
            raise NotImplementedError
        else:
            self._rbar_ij = None
            self._mbar_bi = None

    def _build_ovlp_fit(self):
        mol = self.mol
        grids = self.grids
        nao = mol.nao_nr()
        sn = numpy.zeros((nao,nao))
        nbins, screen_index, pair_mask, ao_loc, blksize = self.get_loop_data()
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
            if self.sym_ovlp:
                proj = scipy.linalg.solve(sn, ovlp)
                proj = 0.5 * (proj + numpy.identity(nao))
            elif self.sym_ovlp:
                vals, vecs = scipy.linalg.eigh(ovlp, sn)
                vals = numpy.sqrt(vals)
                proj = numpy.ascontiguousarray(
                    scipy.linalg.solve(vecs.T, (vecs * vals).T).T
                )
                # print("NORMS", numpy.linalg.norm(proj.T.dot(sn).dot(proj) - ovlp))
            else:
                proj = scipy.linalg.solve(sn, ovlp)
            self._overlap_correction_matrix = proj
        else:
            proj = scipy.linalg.solve(sn, ovlp)
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

    def _make_shl_mat(self, x_gu, x_bi, rlocs, clocs, wt=None, tr=False):
        self._check_arr(x_gu, dtype=numpy.float64)
        self._check_arr(x_bi, dtype=numpy.float64)
        self._check_arr(rlocs, dtype=numpy.int32)
        self._check_arr(clocs, dtype=numpy.int32)
        assert x_gu.shape[-2:] == (rlocs[-1], clocs[-1])
        if tr:
            assert wt is not None
            assert x_bi.shape == (clocs.size - 1, rlocs.size - 1)
        else:
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
            if tr:
                fn = _vhf.libcvhf.SGXmake_shl_mat_wt_tr
            else:
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

    def _build_dm_screen(self):
        ta = logger.perf_counter()
        mol = self.mol
        grids = self.grids
        ngrids = grids.weights.size
        nbins, screen_index, pair_mask, ao_loc, blksize = self.get_loop_data()
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
        t1 = logger.perf_counter()
        if self.upper_bound_algo == "norm":
            self._pow_screen(stilde_ij, _SGX_DELTA_2)
            self._stilde_ij = stilde_ij
            self._mtilde_ij = self._mbar_ij.copy()
            self._pow_screen(self._mtilde_ij, _SGX_DELTA_2)
        else:
            self._mmax_i = numpy.max(self._mbar_ij, axis=1)
            self._msum_i = numpy.sum(self._mbar_ij, axis=1)
        t2 = logger.perf_counter()
        if self._screen_potential:
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
                #rtwt = numpy.sqrt(numpy.abs(grids.weights[i0:i1]))
                #x_gu = _scale_ao_sparse(ao, rtwt, mask, ao_loc)
                #x_gu = numpy.ascontiguousarray(x_gu)
                xtmp_bi = numpy.empty((nblk_ni_curr, mol.nbas))
                # self._make_shl_mat(ao.T, xtmp_bi, ni_locs, ao_loc, wt=grids.weights[i0:i1], tr=True)
                self._make_shl_mat(ao.T, xtmp_bi, ao_loc, ni_locs, wt=grids.weights[i0:i1], tr=True)
                b0 = i0 // BLKSIZE
                xni_bi[b0:b0 + nblk_ni_curr] = xtmp_bi
            t3 = logger.perf_counter()
            locs = dft_per_sgx * numpy.arange(nblk + 1, dtype=numpy.int32)
            locs[-1] = xni_bi.shape[0]
            ones = numpy.arange(mol.nbas + 1, dtype=numpy.int32)
            self._make_shl_mat(xni_bi, xsgx_bi, locs, ones)
            self._pow_screen(xni_bi, _SGX_DELTA_3)
            self._pow_screen(xsgx_bi, _SGX_DELTA_3)
            self._xni_b = numpy.min(xni_bi, axis=1)
            self._xsgx_b = numpy.min(xsgx_bi, axis=1)
            self._vtol_arr = self._vtol * self._xsgx_b
            print(t3 - t2)
        else:
            self._vtol_arr = numpy.empty_like(self._xsgx_b)
            self._vtol_arr[:] = numpy.inf
        self._buf = xni_bi
        self._etol_per_sgx_block = self._etol / nblk
        self._etol_per_ni_block = self._etol / nblk_ni
        tc = logger.perf_counter()
        print("SETUP TOOK", tc - ta, t1 - ta, t2 - t1, tc - t2)

    def build(self):
        t0 = logger.perf_counter()
        self._build_integral_bounds()
        t1 = logger.perf_counter()
        self._ovlp_proj = self._build_ovlp_fit()
        t2 = logger.perf_counter()
        print("SETUP", t2 - t1, t1 - t0)
        if self.use_dm_screening:
            self._build_dm_screen()
        self._setup_opt()

    def get_dm_threshold_matrix(self, dm, dv, de):
        mol = self.mol
        dm_ij = numpy.zeros((mol.nbas, mol.nbas))
        if dm.ndim == 3:
            dm = dm.sum(axis=0)
        ao_loc = mol.ao_loc_nr()
        self._make_shl_mat(dm, dm_ij, ao_loc, ao_loc)
        # TODO do this in C?
        if self.upper_bound_algo == "norm":
            # TODO might be wrong
            tmp = lib.einsum("lk,jk->lj", self._stilde_ij, dm_ij)
            Z_il = lib.einsum("ij,lj->il", self._mtilde_ij, tmp) + 1e-200
            thresh_ij = de / Z_il
            Y_il = numpy.max(self._mtilde_ij, axis=1)
            Y_il = Y_il * numpy.max(self._stilde_ij, axis=1)[:, None]
            Y_il = numpy.maximum(Y_il, Y_il.T) + 1e-200
            thresh_ij = numpy.minimum(dv / Y_il, thresh_ij)
            return dm_ij > thresh_ij
        else:
            if self._screen_energy:
                tmp = lib.einsum("lk,jk->lj", self._sbar_ij, dm_ij)
                Z_il = lib.einsum("ij,lj->il", self._mbar_ij, tmp) + 1e-200
                PZ = dm_ij * Z_il
                SPZ = self._cumsum(PZ, self._argsort(PZ))
                SPZ = 0.5 * (SPZ + SPZ.T)
                econd = SPZ > de / mol.nbas
            else:
                econd = None
            if self._screen_potential:
                PY = (dm_ij * numpy.max(self._mbar_ij, axis=1)
                    * numpy.max(self._sbar_ij, axis=1)[:, None])
                SPY = self._cumsum(PY, self._argsort(PY))
                SPY = 0.5 * (SPY + SPY.T)
                vcond = SPY > dv / mol.nbas
            if self._screen_energy and self._screen_potential:
                return numpy.logical_or(econd, vcond)
            elif self._screen_energy:
                return econd
            elif self._screen_potential:
                return vcond
            else:
                return None

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

    def get_g_threshold(self, f_ug, g_ug, i0, dv, de, wt):
        t0 = time.monotonic()
        b0 = i0 // BLKSIZE
        ao_loc = self._ao_loc
        assert f_ug.flags.c_contiguous
        ndm, nao, ngrids = f_ug.shape
        assert g_ug.flags.c_contiguous
        assert g_ug.shape == f_ug.shape
        assert nao == ao_loc[-1]
        b1 = (ngrids + BLKSIZE - 1) // BLKSIZE + b0
        thresh = dv * self._xni_b[b0:b1]
        if self._screen_energy:
            if not self._screen_potential:
                # no screening on potential
                thresh[:] = numpy.inf
            cond = numpy.empty((b1 - b0, self.mol.nbas), dtype=numpy.uint8)
            _vhf.libcvhf.SGXbuild_gv_threshold(
                f_ug.ctypes,
                g_ug.ctypes,
                ao_loc.ctypes,
                thresh.ctypes,
                ctypes.c_double(de / self._xni_b.size),
                wt.ctypes,
                ctypes.c_int(ndm),
                ctypes.c_int(self.mol.nbas),
                ctypes.c_int(ngrids),
                cond.ctypes,
            )
        elif self._screen_potential:
            # TODO test
            g_ib = numpy.empty_like((self.mol.nbas, b1 - b0))
            cloc = numpy.arange(b1 - b0 + 1, dtype=numpy.int32) * BLKSIZE
            cloc[-1] = ngrids
            _vhf.libcvhf.SGXmake_shl_mat_wt(
                g_ug.ctypes, g_ib.ctypes, ctypes.c_int(self.mol.nbas),
                ctypes.c_int(g_ib.shape[1]), ao_loc.ctypes, cloc.ctypes,
                wt.ctypes, ctypes.c_int(ndm)
            )
            cond = g_ib.T > thresh[:, None]
        else:
            cond = None
        t1 = time.monotonic()
        print("TOTAL GTIME", t1 - t0)
        return cond


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
    nset = dms.shape[0]
    t1 = logger.timer_debug1(mol, "sgX initialization", *t0)
    vk = numpy.zeros_like(dms)
    tnuc = 0, 0
    shls_slice = (0, mol.nbas)
    ngrids = grids.weights.size

    assert sgx._opt is not None
    if (
        not hasattr(sgx, "_pjs_data")
        or sgx._pjs_data is None
        or sgx._pjs_data.mol is not mol
        or sgx._pjs_data.grids is not grids
    ):
        sgx._pjs_data = SGXData(
            mol,
            grids,
            k_only=True,
            fit_ovlp=sgx.fit_ovlp,
            sym_ovlp=sgx._symm_ovlp_fit,
            max_memory=sgx.max_memory,
            vtol=sgx.sgx_tol_potential,
            etol=sgx.sgx_tol_energy,
            hermi=1,
            integral_bound="sample",
            sgxopt=sgx._opt,
        )
        sgx._pjs_data.build()
    loop_data = sgx._pjs_data.get_loop_data(nset=nset, with_pair_mask=True)
    nbins, screen_index, pair_mask, ao_loc, blksize = loop_data
    proj = sgx._pjs_data._overlap_correction_matrix
    proj_dm = lib.einsum('ki,xij->xkj', proj, dms)
    dm_mask = sgx._pjs_data.get_dm_threshold_matrix(
        dms, sgx.sgx_tol_potential, sgx.sgx_tol_energy
    )

    batch_k = _gen_k_direct(mol, 's2', direct_scf_tol, sgx._opt)

    if full_dm is None:
        full_dm = dms
    else:
        full_dm = numpy.asarray(full_dm)
        full_dm = full_dm.reshape(-1,nao,nao)

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
        tnuc = tnuc[0] - logger.process_clock(), tnuc[1] - logger.perf_counter()
        tc = logger.perf_counter()
        if sgx.use_dm_screening:
            fg = _sgxdot_ao_dm_sparse(ao, proj_dm, mask, dm_mask, ao_loc, out=fg)
        else:
            # fg = lib.einsum('xij,ig->xjg', proj_dm, wao.T)
            fg = _sgxdot_ao_dm(ao, proj_dm, mask, shls_slice, ao_loc, out=fg)
        te = logger.perf_counter()

        if sgx.use_dm_screening:
            #wt_input = (fbar_bi, bsgx_bi, shl_screen)
            wt_input = (weights, sgx._pjs_data._mbar_ij,
                        sgx.sgx_tol_potential * sgx._pjs_data._xsgx_b[i0 // SGX_BLKSIZE:],
                        sgx.sgx_tol_energy / sgx._pjs_data._xsgx_b.size)
            gv = batch_k(mol, coords, fg, weights=wt_input, v2=True)
        else:
            gv = batch_k(mol, coords, fg, weights)
        tnuc = tnuc[0] + logger.process_clock(), tnuc[1] + logger.perf_counter()
        if sgx.use_dm_screening:
            mask2 = sgx._pjs_data.get_g_threshold(
                fg, gv, i0, sgx.sgx_tol_potential, sgx.sgx_tol_energy, weights
            )
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
    if sgx._pjs_data.sym_ovlp:
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
        assert fg.flags.c_contiguous
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
            args = [
                cintor, fjk, dmsptr, vjkptr, n_dm, ncomp,
                (ctypes.c_int*4)(*shls_slice),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                sgxopt._cintopt, ctypes.byref(sgxopt._this),
                atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
                env.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(2 if aosym == 's2' else 1),
            ]
            if len(weights) == 2:
                args = args + [weights[0].ctypes, weights[1].ctypes]
                fn = _vhf.libcvhf.SGXnr_direct_drv2
            elif len(weights) == 3:
                args = args + [weights[0].ctypes, weights[1].ctypes, weights[2].ctypes]
                fn = _vhf.libcvhf.SGXnr_direct_drv3
            else:
                assert len(weights) == 4
                args = args + [weights[0].ctypes, weights[1].ctypes,
                               weights[2].ctypes, ctypes.c_double(weights[3])]
                fn = _vhf.libcvhf.SGXnr_direct_drv4
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
