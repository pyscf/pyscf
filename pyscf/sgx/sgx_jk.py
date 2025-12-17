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
Two SCF steps: coarse grid then fine grid.

See the docs of pyscf.sgx.sgx for details on adjustable parameters.
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
from pyscf.dft.numint import BLKSIZE, NBINS, libdft, \
    SWITCH_SIZE, _scale_ao, _dot_ao_ao_sparse
import time


libdft.SGXreturn_blksize.restype = int
SGX_BLKSIZE = libdft.SGXreturn_blksize()
assert SGX_BLKSIZE % BLKSIZE == 0
# number of DFT numint blocks per SGX block
DFT_PER_SGX = SGX_BLKSIZE // BLKSIZE
SGX_DELTA = 0.01


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
    else:
        # TODO implement complex dot
        raise NotImplementedError("Complex sgxdot")

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
    if pair_mask is None:
        return _sgxdot_ao_dm(ao, dms, mask, (0, nbas), ao_loc, out=out)
    vms = numpy.ndarray((dms.shape[0], dms.shape[2], ngrids), dtype=ao.dtype, order='C', buffer=out)
    if (nao < SWITCH_SIZE or mask is None or ao_loc is None):
        for i in range(len(dms)):
            lib.dot(dms[i].T, ao.T, c=vms[i])
        return vms

    if not ao.flags.f_contiguous:
        ao = lib.transpose(ao)
    if ao.dtype == dms.dtype == numpy.double:
        fn = _vhf.libcvhf.SGXdot_ao_dm_sparse
    else:
        # TODO implement complex dot
        raise NotImplementedError("Complex sgxdot")

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
        # TODO implement complex dot
        raise NotImplementedError("Complex sgxdot")

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
                ('mbar_ij', ctypes.c_void_p),
                ('mbar_bi', ctypes.c_void_p),
                ('rbar_ij', ctypes.c_void_p),
                ('mmax_i', ctypes.c_void_p),
                ('msum_i', ctypes.c_void_p),
                ('buf_size_bytes', ctypes.c_size_t),
                ('shl_info_size_bytes', ctypes.c_size_t),
                ('direct_scf_tol', ctypes.c_double),
                ('fscreen_grid', ctypes.c_void_p),
                ('fscreen_shl', ctypes.c_void_p),
                ('full_f_bi', ctypes.c_void_p)]


def _arr2c(arr):
    return None if arr is None else arr.ctypes.data_as(ctypes.c_void_p)


class SGXData:
    def __init__(self, mol, grids,
                 fit_ovlp=True, sym_ovlp=False,
                 max_memory=2000, hermi=0,
                 bound_algo="sample_pos",
                 sgxopt=None, etol=None, vtol=None,
                 direct_scf_tol=0.0):
        """
        bound_algo can be
            "ovlp": Screen integrals based on overlap of
                orbital pairs. Overlap serves as a rough
                approximation of the maximum ESP integral.
            "sample": Provide an approximate but accurate
                upper bound for the ESP integrals by sampling
                _nquad points for each shell pair.
            "sample_pos": Same as sample, but the ESP
                bounds are position-dependent, which gives
                a slight speed increase for large systems
                and a significant speed increase for
                short-range hybrids.
        For each bound algo, screening can be done based
        on energy, potential, or both
        """
        self.mol = mol
        self.grids = grids
        self.fit_ovlp = fit_ovlp
        self.sym_ovlp = sym_ovlp
        self.max_memory = max_memory
        self.hermi = hermi
        self._itol = direct_scf_tol
        if etol == "auto":
            etol = direct_scf_tol
        if vtol == "auto":
            if etol is None:
                vtol = direct_scf_tol**0.5
            else:
                vtol = etol**0.5
        self._etol = etol
        self._screen_energy = etol is not None
        self._vtol = vtol
        self._screen_potential = vtol is not None
        self._nquad = 200
        self._nrad = 32
        self._rcmul = 2.0
        self.bound_algo = bound_algo
        if self.bound_algo not in ["ovlp", "sample", "sample_pos"]:
            raise ValueError("Unsupported integral bound algorithm")
        self._opt = sgxopt
        self._this = _CSGXOpt()

    @property
    def _cintopt(self):
        return self._opt._cintopt

    @property
    def use_dm_screening(self):
        return self._screen_energy or self._screen_potential

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
        self._ao_loc = numpy.ascontiguousarray(
            self.mol.ao_loc_nr().astype(numpy.int32)
        )
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
        if self._screen_energy:
            self._this.etol = self._etol_per_sgx_block
        else:
            self._this.etol = 1e10
        self._this.vtol = None
        self._this.wt = None  # set in each cycle
        self._this.mbar_ij = _arr2c(self._mbar_ij)
        self._this.mmax_i = _arr2c(self._mmax_i)
        self._this.msum_i = _arr2c(self._msum_i)
        if self.bound_algo == "sample_pos":
            self._this.rbar_ij = _arr2c(self._rbar_ij)
        else:
            self._this.rbar_ij = None
        self._this.mbar_bi = None
        if self._screen_energy or self._screen_potential:
            # buf can contain the following:
            # For batch screening step
            #   Nothing for no dm screening
            #   3 * nbas * double, 2 * nbas * int for sort1
            #   5 * nbas * double, 2 * nbas * int for sort2
            # For the shell screening step
            #   Nothing for no dm screening
            #   3 * nbas * double, 3 nbas * int
            dsize = self.mol.nbas * ctypes.sizeof(ctypes.c_double)
            isize = self.mol.nbas * ctypes.sizeof(ctypes.c_int)
            bsize = self.mol.nbas * ctypes.sizeof(ctypes.c_uint8)
            # shl_info can contain the following:
            #   shlscreen_i (nbas * uint8_t)
            #   bscreen_i (nbas * double)
            #   fbar_i (nbas * double)
            #   ffbar_i (nbas * double)
            bsb = 5 * dsize + 3 * isize
            sisb = bsize + 3 * dsize
            grid_screen = "SGXscreen_grid"
            shl_screen = "SGXscreen_shells_sorted"
        else:
            bsb = 0
            sisb = 0
            grid_screen = "SGXnoscreen_grid"
            shl_screen = "SGXscreen_shells_simple"
        self._this.fscreen_grid = _vhf._fpointer(grid_screen)
        self._this.fscreen_shl = _vhf._fpointer(shl_screen)
        self._this.direct_scf_tol = self._itol
        self._this.buf_size_bytes = bsb
        self._this.shl_info_size_bytes = sisb
        # TODO need to make this modifiable
        if self.mol.cart:
            self._intor = "int1e_grids_cart"
        else:
            self._intor = "int1e_grids_sph"

    def set_block(self, b0, weights, full_f_bi=None):
        self._wt = weights
        self._this.ngrids = weights.size
        self._this.wt = _arr2c(weights)
        if self._screen_potential or self._screen_energy:
            self._this.vtol = _arr2c(self._vtol_arr[b0:])
        else:
            self._this.vtol = None
        if self.bound_algo == "sample_pos":
            self._this.mbar_bi = _arr2c(self._mbar_bi[b0:])
        else:
            self._this.mbar_bi = None
        if full_f_bi is None:
            self._this.full_f_bi = None
        else:
            self._full_f_bi = full_f_bi
            self._this.full_f_bi = _arr2c(full_f_bi)

    def unset_block(self):
        self._wt = None
        self._full_f_bi = None
        self._this.ngrids = 0
        self._this.wt = None
        self._this.vtol = None
        self._this.mbar_bi = None
        self._this.full_f_bi = None

    def get_loop_data(self, nset=1, with_pair_mask=True, grad=False):
        mol = self.mol
        grids = self.grids
        nao = mol.nao_nr()
        ao_loc = mol.ao_loc_nr()
        if (grids.coords is None or grids.non0tab is None
                or grids.atm_idx is None):
            grids.build(with_non0tab=True)
        ngrids = grids.coords.shape[0]
        if mol is grids.mol:
            non0tab = grids.non0tab
        if non0tab is None:
            non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                                    dtype=numpy.uint8)
            non0tab[:] = NBINS + 1  # Corresponding to AO value ~= 1
        screen_index = non0tab
        max_memory = self.max_memory - lib.current_memory()[0]
        if grad:
            # We need ~4 ao copes (ao, dx, dy, dz), plus 2 temporary arrays
            # along with gv, dgv, and fg
            data_dim = 6 + 5 * nset
        else:
            # We need to store ao, wao, fg, and gv -> 2 + nset sets of size nao
            data_dim = 2 + 2 * nset
        blksize = max(SGX_BLKSIZE, int(max_memory*1e6/8/(data_dim*nao)))
        blksize = min(ngrids, max(1, blksize // SGX_BLKSIZE) * SGX_BLKSIZE)
        cutoff = grids.cutoff * 1e2
        nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))
        if with_pair_mask:
            pair_mask = mol.get_overlap_cond() < -numpy.log(cutoff)
        else:
            pair_mask = None
        return nbins, screen_index, pair_mask, ao_loc, blksize

    def _get_diagonal_ints(self, sgxopt):
        """
        Compute the maximum of the ESP integrals within each block
        for the diagonal shell pairs (i.e. ish == jsh).
        """
        mol = self.mol
        drv = _vhf.libcvhf.SGXfit_diagonal_ints
        atm_coords = numpy.ascontiguousarray(
            mol.atom_coords(unit='Bohr')
        )
        ao_loc = mol.ao_loc_nr().astype(numpy.int32)
        cintor = _vhf._fpointer(sgxopt._intor)
        atm, bas, env = mol._atm, mol._bas, mol._env
        nrad = self._nrad
        env = numpy.append(env, numpy.zeros(3 * mol.nbas * nrad))
        env[gto.NGRIDS] = 3 * mol.nbas * nrad
        env[gto.PTR_GRIDS] = mol._env.size
        ovlp = mol.intor_symmetric("int1e_ovlp")
        if not ovlp.flags.c_contiguous:
            ovlp = ovlp.T
        sbar_ij = numpy.zeros((mol.nbas, mol.nbas))
        self._make_shl_mat(ovlp, sbar_ij, ao_loc, ao_loc)
        norms = numpy.ascontiguousarray(numpy.diag(sbar_ij))
        drv(
            cintor, ctypes.c_int(nrad), atm_coords.ctypes,
            ao_loc.ctypes, sgxopt._cintopt,
            atm.ctypes, ctypes.c_int(mol.natm), bas.ctypes,
            ctypes.c_int(mol.nbas), env.ctypes,
            norms.ctypes,
        )
        res = env[mol._env.size:].reshape(mol.nbas, 3, nrad)
        radii = numpy.ascontiguousarray(res[:, 1, 0])
        res = numpy.ascontiguousarray(res[:, 0, :])
        res[:, -2:] = 0  # avoid interpolating in crazy directions
        return res, radii, norms

    def _build_integral_bounds(self):
        """
        Compute and save bounds on the ESP integrals
        """
        mol = self.mol
        mbar_ij = numpy.zeros((mol.nbas, mol.nbas))
        if self.bound_algo == "sample_pos":
            rbar_ij = numpy.zeros((mol.nbas, mol.nbas))
        else:
            rbar_ij = None
        atm_coords = numpy.ascontiguousarray(
            mol.atom_coords(unit='Bohr')
        )
        ao_loc = mol.ao_loc_nr().astype(numpy.int32)
        if self._opt is None:
            from pyscf.sgx.sgx import _make_opt
            sgxopt = _make_opt(mol, False, self._itol)
        else:
            sgxopt = self._opt
        if self.bound_algo == "ovlp":
            atm, bas, env = mol._atm, mol._bas, mol._env
            if self.mol.cart:
                intor = "int1e_ovlp_cart"
            else:
                intor = "int1e_ovlp_sph"
            cintor = _vhf._fpointer(intor)
            _vhf.libcvhf.SGXnr_q_cond(
                cintor, lib.c_null_ptr(),
                mbar_ij.ctypes, ao_loc.ctypes,
                atm.ctypes, ctypes.c_int(mol.natm), bas.ctypes,
                ctypes.c_int(mol.nbas), env.ctypes,
            )
        else:
            vals, widths, norms = self._get_diagonal_ints(sgxopt)
            drv = _vhf.libcvhf.SGXsample_ints
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
            for _arr in [mbar_ij, radii, atm_coords, rs, env]:
                self._check_arr(_arr, dtype=numpy.float64)
            for _arr in [ao_loc, atm, bas]:
                self._check_arr(_arr, dtype=numpy.int32)
            assert sgxopt._cintopt is not None
            crbar = None if rbar_ij is None else rbar_ij.ctypes
            drv(
                cintor, mbar_ij.ctypes, crbar, radii.ctypes,
                ctypes.c_int(self._nquad), ctypes.c_int(self._nrad),
                ctypes.c_double(self._rcmul), atm_coords.ctypes,
                rs.ctypes, ao_loc.ctypes, sgxopt._cintopt,
                atm.ctypes, ctypes.c_int(mol.natm), bas.ctypes,
                ctypes.c_int(mol.nbas), env.ctypes, ctypes.c_int(env.size),
                widths.ctypes, norms.ctypes, vals.ctypes,
                ctypes.c_double(self._itol)
            )
        self._mbar_ij = numpy.maximum(mbar_ij, mbar_ij.T)
        self._mmax_i = numpy.max(mbar_ij, axis=1)
        self._msum_i = numpy.sum(mbar_ij, axis=1)
        self._opt.q_cond = mbar_ij
        if self.bound_algo == "sample_pos":
            self._rbar_ij = numpy.maximum(rbar_ij, rbar_ij.T)
            self._get_pos_dpt_ints(sgxopt, widths, norms, vals, self._nrad, atm_coords)
        else:
            self._rbar_ij = None
            self._mbar_bi = None

    def _get_pos_dpt_ints(self, sgxopt, widths, norms, vals, nrad, atm_coords):
        """
        Calculate position-dependent bounds on the ESP integrals.
        Gives a slight speedup to full exchange and a large
        speedup to short-range exchange.
        """
        mol = self.mol
        grid_coords = self.grids.coords
        atm, bas, env = mol._atm, mol._bas, mol._env
        cintor = _vhf._fpointer(sgxopt._intor)
        ngrids = grid_coords.shape[0]
        nblk = (ngrids + SGX_BLKSIZE - 1) // SGX_BLKSIZE
        env = numpy.append(env, grid_coords.ravel())
        env[gto.NGRIDS] = ngrids
        env[gto.PTR_GRIDS] = mol._env.size
        m_bi = numpy.empty((nblk, mol.nbas))
        _vhf.libcvhf.SGXdiagonal_ints(
            cintor,
            m_bi.ctypes,
            self._ao_loc.ctypes,
            sgxopt._cintopt,
            atm.ctypes,
            ctypes.c_int(mol.natm),
            bas.ctypes,
            ctypes.c_int(mol.nbas),
            env.ctypes,
            widths.ctypes,
            norms.ctypes,
            vals.ctypes,
            ctypes.c_int(nrad),
            atm_coords.ctypes,
        )
        m_bi[:] = numpy.sqrt(m_bi)
        self._mbar_bi = m_bi

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
            ovlp = mol.intor_symmetric('int1e_ovlp')
            if self.sym_ovlp:
                proj = scipy.linalg.solve(sn, ovlp)
                proj = 0.5 * (proj + numpy.identity(nao))
            else:
                proj = scipy.linalg.solve(sn, ovlp)
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
            args.append(ctypes.c_int(ncomp))
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

    def reduce_mask_ni2sgx(self, xni_bi):
        nblk_ni, nbas = xni_bi.shape
        nblk = (nblk_ni + DFT_PER_SGX - 1) // DFT_PER_SGX
        xsgx_bi = numpy.empty((nblk, nbas), dtype=numpy.float64)
        locs = DFT_PER_SGX * numpy.arange(nblk + 1, dtype=numpy.int32)
        locs[-1] = nblk_ni
        ones = numpy.arange(nbas + 1, dtype=numpy.int32)
        self._make_shl_mat(xni_bi, xsgx_bi, locs, ones)
        return xsgx_bi

    def reduce_ao_to_shl_mat(self, ao, ao_loc, wt):
        assert ao.flags.f_contiguous
        ngrids, nao = ao.shape
        nblk_ni_curr = (ngrids + BLKSIZE - 1) // BLKSIZE
        ni_locs = BLKSIZE * numpy.arange(
            nblk_ni_curr + 1, dtype=numpy.int32
        )
        ni_locs[-1] = ngrids
        x_bi = numpy.empty((nblk_ni_curr, ao_loc.size - 1))
        self._make_shl_mat(ao.T, x_bi, ao_loc, ni_locs, wt=wt, tr=True)
        return x_bi

    def _build_dm_screen(self):
        mol = self.mol
        grids = self.grids
        ngrids = grids.weights.size
        nbins, screen_index, pair_mask, ao_loc, blksize = self.get_loop_data()
        nblk = (grids.weights.size + SGX_BLKSIZE - 1) // SGX_BLKSIZE
        nblk_ni = (grids.weights.size + BLKSIZE - 1) // BLKSIZE
        xni_bi = numpy.empty((nblk_ni, mol.nbas), dtype=numpy.float64)
        dft_locs = BLKSIZE * numpy.arange(nblk_ni, dtype=numpy.int32)
        if grids.weights.size >= 2**31:
            raise ValueError("Too many grids for signed int32")
        dft_locs[-1] = grids.weights.size
        ovlp = mol.intor_symmetric("int1e_ovlp")
        if not ovlp.flags.c_contiguous:
            ovlp = ovlp.T
        stilde_ij = numpy.zeros((mol.nbas, mol.nbas))
        self._make_shl_mat(ovlp, stilde_ij, ao_loc, ao_loc)
        self._sbar_ij = stilde_ij.copy()
        if self._screen_potential:
            ao = None
            for i0, i1 in lib.prange(0, ngrids, blksize):
                assert i0 % SGX_BLKSIZE == 0
                b0 = i0 // BLKSIZE
                b1 = b0 + (i1 - i0 + BLKSIZE - 1) // BLKSIZE
                coords = grids.coords[i0:i1]
                mask = screen_index[b0:b1]
                ao = mol.eval_gto('GTOval', coords, non0tab=mask, out=ao)
                xtmp_bi = self.reduce_ao_to_shl_mat(ao, ao_loc, grids.weights[i0:i1])
                xni_bi[b0:b1] = xtmp_bi
            xsgx_bi = self.reduce_mask_ni2sgx(xni_bi)
            assert xsgx_bi.shape == (nblk, mol.nbas)
            self._pow_screen(xni_bi, SGX_DELTA)
            self._pow_screen(xsgx_bi, SGX_DELTA)
            self._xni_b = numpy.min(xni_bi, axis=1)
            self._xsgx_b = numpy.min(xsgx_bi, axis=1)
            self._vtol_arr = self._vtol * self._xsgx_b
        else:
            self._xni_b = numpy.empty(nblk_ni)
            self._xni_b[:] = numpy.inf
            self._vtol_arr = numpy.empty(nblk)
            self._vtol_arr[:] = numpy.inf
        self._buf = xni_bi
        if self._screen_energy:
            self._etol_per_sgx_block = self._etol / nblk
            self._etol_per_ni_block = self._etol / nblk_ni
        else:
            self._etol_per_sgx_block = 1e10
            self._etol_per_ni_block = 1e10

    def build(self):
        self._build_integral_bounds()
        self._ovlp_proj = self._build_ovlp_fit()
        if self.use_dm_screening:
            self._build_dm_screen()
        self._setup_opt()

    def get_dm_threshold_matrix(self, dm, full_dm_ij=None):
        de, dv = self._etol, self._vtol
        mol = self.mol
        ao_loc = mol.ao_loc_nr()
        if dm.ndim == 3:
            dm = dm.sum(axis=0)
        dm_ij = numpy.zeros((mol.nbas, mol.nbas))
        self._make_shl_mat(dm, dm_ij, ao_loc, ao_loc)
        if full_dm_ij is None:
            full_dm_ij = dm_ij
        if self._screen_energy:
            tmp = lib.einsum("lk,jk->lj", self._sbar_ij, full_dm_ij)
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

    def get_g_threshold(self, f_ug, g_ug, i0, wt, full_f_bi=None):
        de, dv = self._etol, self._vtol
        b0 = i0 // BLKSIZE
        ao_loc = self._ao_loc
        assert f_ug.flags.c_contiguous
        ndm, nao, ngrids = f_ug.shape
        assert g_ug.flags.c_contiguous
        assert g_ug.shape == f_ug.shape
        assert nao == ao_loc[-1]
        b1 = (ngrids + BLKSIZE - 1) // BLKSIZE + b0
        if self._screen_potential:
            thresh = dv * self._xni_b[b0:b1]
        else:
            thresh = numpy.empty(b1-b0, dtype=numpy.float64)
            thresh[:] = numpy.inf
        fullf_ptr = None if full_f_bi is None else full_f_bi.ctypes
        if self._screen_energy:
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
                fullf_ptr,
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
    nset = dms.shape[0]
    t1 = logger.timer_debug1(mol, "sgX initialization", *t0)
    vk = numpy.zeros_like(dms)
    tnuc = 0, 0
    ngrids = grids.weights.size

    if (
        sgx._pjs_data is None
        or sgx._pjs_data.mol is not mol
        or sgx._pjs_data.grids is not grids
        or sgx._pjs_data._itol != direct_scf_tol
    ):
        sgx._build_pjs(direct_scf_tol)
    sgxdat = sgx._pjs_data
    loop_data = sgx._pjs_data.get_loop_data(nset=nset, with_pair_mask=True)
    nbins, screen_index, pair_mask, ao_loc, blksize = loop_data
    proj = sgx._pjs_data._overlap_correction_matrix
    proj_dm = lib.einsum('ki,xij->xkj', proj, dms)
    if sgxdat.use_dm_screening:
        dm_mask = sgx._pjs_data.get_dm_threshold_matrix(dms)
        if sgx._full_dm is None:
            full_dm_ij = None
        else:
            full_dm = sgx._full_dm
            if full_dm.ndim == 3:
                full_dm = full_dm.sum(axis=0)
            full_dm_ij = numpy.zeros((mol.nbas, mol.nbas))
            sgx._pjs_data._make_shl_mat(
                full_dm, full_dm_ij, ao_loc, ao_loc
            )
    else:
        dm_mask = None
        full_dm_ij = None

    batch_k = _gen_k_direct(mol, 's2', direct_scf_tol, sgx._pjs_data)

    ao = None
    fg = None
    for i0, i1 in lib.prange(0, ngrids, blksize):
        assert i0 % SGX_BLKSIZE == 0
        coords = grids.coords[i0:i1]
        weights = grids.weights[i0:i1]
        mask = screen_index[i0 // BLKSIZE :]
        ao = mol.eval_gto('GTOval', coords, non0tab=mask, out=ao)
        if full_dm_ij is None:
            full_fni_bi = None
            full_fsgx_bi = None
        else:
            full_fni_bi = sgx._pjs_data.reduce_ao_to_shl_mat(ao, ao_loc, weights)
            full_fni_bi = lib.dot(full_fni_bi, full_dm_ij)
            assert full_fni_bi.flags.c_contiguous
            full_fsgx_bi = sgx._pjs_data.reduce_mask_ni2sgx(full_fni_bi)
        fg = _sgxdot_ao_dm_sparse(ao, proj_dm, mask, dm_mask, ao_loc, out=fg)
        tnuc = tnuc[0] - logger.process_clock(), tnuc[1] - logger.perf_counter()
        gv = batch_k(mol, coords, fg, weights, i0 // SGX_BLKSIZE, full_fsgx_bi)
        tnuc = tnuc[0] + logger.process_clock(), tnuc[1] + logger.perf_counter()
        if sgxdat.use_dm_screening:
            mask2 = sgx._pjs_data.get_g_threshold(
                fg, gv, i0, weights, full_f_bi=full_fni_bi
            )
        else:
            mask2 = None
        _sgxdot_ao_gv_sparse(ao, gv, weights, mask, mask2, ao_loc, out=vk)
        gv = None
    if sgx._pjs_data.sym_ovlp:
        vk = lib.einsum('ik,xij->xkj', proj, vk)

    t2 = logger.timer_debug1(mol, "sgX J/K builder", *t1)
    tdot = t2[0] - t1[0] - tnuc[0] , t2[1] - t1[1] - tnuc[1]
    logger.debug1(sgx, '(CPU, wall) time for integrals (%.2f, %.2f); '
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
    tdot = t2[0] - t1[0] - tnuc[0], t2[1] - t1[1] - tnuc[1]
    logger.debug1(sgx, '(CPU, wall) time for integrals (%.2f, %.2f); '
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
            ctypes.c_int(2 if aosym == 's2' else 1))
        return vj, vk
    return jk_part


def _gen_k_direct(mol, aosym, direct_scf_tol,
                  sgxopt=None, grad=False):
    '''K-only ontraction between sgX Coulomb integrals and density matrices
    K: einsum('gtv,xgt->xgv', gbn, fg)
    '''
    if sgxopt is None:
        from pyscf.sgx import sgx
        sgxopt = sgx._make_opt(mol, grad, direct_scf_tol)
    if isinstance(sgxopt, SGXData):
        if grad:
            from pyscf.sgx import sgx
            tmpopt = sgx._make_opt(mol, grad, direct_scf_tol)
            _cintopt = tmpopt._cintopt
            intor_name = tmpopt._intor
        else:
            _cintopt = sgxopt._cintopt
            intor_name = sgxopt._intor
    else:
        intor_name = sgxopt._intor
    sgxopt.direct_scf_tol = direct_scf_tol

    ao_loc = moleintor.make_loc(mol._bas, sgxopt._intor)

    if grad:
        ncomp = 3
    else:
        ncomp = 1
    nao = mol.nao
    cintor = _vhf._fpointer(intor_name)
    drv = _vhf.libcvhf.SGXnr_direct_k_drv

    def k_part(mol, grid_coords, fg, weights=None, b0=None,
               full_f_bi=None):
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

        assert weights is not None
        assert b0 is not None
        sgxopt.set_block(b0, weights, full_f_bi)
        drv(
            cintor, fjk, dmsptr, vjkptr, n_dm, ncomp,
            (ctypes.c_int*4)(*shls_slice),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            _cintopt, ctypes.byref(sgxopt._this),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
            env.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(2 if aosym == 's2' else 1),
        )
        sgxopt.unset_block()
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

    if gthrd > 0:
        ngrids = grids.weights.size
        mask = []
        for p0, p1 in lib.prange(0, ngrids, 10000):
            ao_v = mol.eval_gto('GTOval', grids.coords[p0:p1])
            ao_v *= grids.weights[p0:p1,None]
            wao_v0 = ao_v
            mask.append(numpy.any(wao_v0>gthrd, axis=1) |
                        numpy.any(wao_v0<-gthrd, axis=1))
        mask = numpy.hstack(mask)
        grids._select_grids(mask)
        grids._add_padding()
    grids.non0tab = grids.make_mask(mol, grids.coords)
    grids.screen_index = grids.non0tab
    logger.debug(mol, 'threshold for grids screening %g', gthrd)
    logger.debug(mol, 'number of grids %d', grids.weights.size)
    logger.timer_debug1(mol, "Xg screening", *Ktime)
    return grids

get_jk = get_jk_favorj
