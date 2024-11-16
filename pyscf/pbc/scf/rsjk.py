#!/usr/bin/env python
# Copyright 2020-2021 The PySCF Developers. All Rights Reserved.
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
# Authors: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Range separation JK builder

Ref:
    Q. Sun, arXiv:2012.07929
    Q. Sun, arXiv:2302.11307
'''

import ctypes
import numpy as np
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.tools import pbc as pbctools
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.scf import addons
from pyscf.pbc.df import aft, rsdf_builder, aft_jk
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df.df_jk import (zdotNN, zdotCN, zdotNC, _ewald_exxdiv_for_G0,
                                _format_dms, _format_kpts_band, _format_jks)
from pyscf.pbc.df.incore import libpbc, _get_cache_size, LOG_ADJUST
from pyscf.pbc.lib.kpts_helper import (is_zero, kk_adapted_iter,
                                       group_by_conj_pairs, intersection)
from pyscf import __config__

# Threshold of steep bases and local bases
RCUT_THRESHOLD = getattr(__config__, 'pbc_scf_rsjk_rcut_threshold', 1.0)
OMEGA_MIN = rsdf_builder.OMEGA_MIN
INDEX_MIN = rsdf_builder.INDEX_MIN

class RangeSeparatedJKBuilder(lib.StreamObject):
    _keys = {
        'cell', 'mesh', 'kpts', 'purify', 'omega', 'rs_cell', 'cell_d',
        'bvk_kmesh', 'supmol_sr', 'supmol_ft', 'supmol_d', 'cell0_basis_mask',
        'ke_cutoff', 'direct_scf_tol', 'time_reversal_symmetry',
        'exclude_dd_block', 'allow_drv_nodddd', 'approx_vk_lr_missing_mo',
    }

    def __init__(self, cell, kpts=np.zeros((1,3))):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory
        self.mesh = None
        self.kpts = np.reshape(kpts, (-1, 3))
        self.purify = True

        if cell.omega == 0:
            self.omega = None
        elif cell.omega < 0:
            # Initialize omega to cell.omega for HF exchange of short range
            # int2e in RSH functionals
            self.omega = -cell.omega
        else:
            raise RuntimeError('RSJK does not support LR integrals')
        self.rs_cell = None
        self.cell_d = None
        # Born-von Karman supercell
        self.bvk_kmesh = None
        self.supmol_sr = None
        self.supmol_ft = None
        self.supmol_d = None
        # which shells are located in the first primitive cell
        self.cell0_basis_mask = None
        self.ke_cutoff = None
        self.direct_scf_tol = None
        # Use the k-point conjugation symmetry between k and -k
        self.time_reversal_symmetry = True
        self.exclude_dd_block = True
        self.allow_drv_nodddd = True
        self._sr_without_dddd = None
        # Allow to approximate vk_lr with less mesh if mo_coeff not attached to dm.
        self.approx_vk_lr_missing_mo = False
        # TODO: incrementally build SR part
        self._last_vs = (0, 0)
        self._qindex = None

    __getstate__, __setstate__ = lib.generate_pickle_methods(
            excludes=('rs_cell', 'cell_d', 'supmol_sr', 'supmol_ft', 'supmol_d',
                      '_sr_without_dddd', '_last_vs', '_qindex'),
            reset_state=True)

    def has_long_range(self):
        '''Whether to add the long-range part computed with AFT/FFT integrals'''
        return self.omega is None or abs(self.cell.omega) < self.omega

    def dump_flags(self, verbose=None):
        logger.info(self, '\n')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'mesh = %s (%d PWs)', self.mesh, np.prod(self.mesh))
        logger.info(self, 'omega = %s', self.omega)
        logger.info(self, 'purify = %s', self.purify)
        logger.info(self, 'bvk_kmesh = %s', self.bvk_kmesh)
        logger.info(self, 'ke_cutoff = %s', self.ke_cutoff)
        logger.info(self, 'time_reversal_symmetry = %s', self.time_reversal_symmetry)
        logger.info(self, 'has_long_range = %s', self.has_long_range())
        logger.info(self, 'exclude_dd_block = %s', self.exclude_dd_block)
        logger.info(self, 'allow_drv_nodddd = %s', self.allow_drv_nodddd)
        logger.debug(self, 'approx_vk_lr_missing_mo = %s', self.approx_vk_lr_missing_mo)
        return self

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self.rs_cell = None
        self.cell_d = None
        self.supmol_sr = None
        self.supmol_ft = None
        self.supmol_d = None
        self._sr_without_dddd = None
        self._last_vs = (0, 0)
        self._qindex = None
        return self

    def build(self, omega=None, intor='int2e'):
        cpu0 = logger.process_clock(), logger.perf_counter()
        log = logger.new_logger(self)
        cell = self.cell
        kpts = self.kpts

        if omega is not None:
            self.omega = omega

        if self.omega is None or self.omega == 0:
            # Search a proper range-separation parameter omega that can balance the
            # computational cost between the real space integrals and moment space
            # integrals
            self.omega, self.mesh, self.ke_cutoff = _guess_omega(cell, kpts, self.mesh)
        else:
            self.ke_cutoff = estimate_ke_cutoff_for_omega(cell, self.omega)
            self.mesh = cell.cutoff_to_mesh(self.ke_cutoff)

        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            # To ensure trunc-coulG converged for all basis
            self.mesh[2] = rsdf_builder._estimate_meshz(cell)

        self.check_sanity()

        log.info('omega = %.15g  ke_cutoff = %s  mesh = %s',
                 self.omega, self.ke_cutoff, self.mesh)

        # AFT part exchange cost may be dominant
        if cell.nao * len(kpts) > 5000:
            log.debug1('Disable exclude_dd_block')
            self.exclude_dd_block = False

        # basis with cutoff under ~150 eV are handled by AFTDF
        ke_cutoff = max(6., self.ke_cutoff)
        rs_cell = ft_ao._RangeSeparatedCell.from_cell(
            cell, ke_cutoff, RCUT_THRESHOLD, in_rsjk=True, verbose=log)
        self.rs_cell = rs_cell

        self.bvk_kmesh = kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        log.debug('kmesh for bvk-cell = %s', kmesh)

        exp_min = np.hstack(cell.bas_exps()).min()
        lattice_sum_factor = max((2*cell.rcut)**3/cell.vol * 1/exp_min, 1)
        cutoff = cell.precision / lattice_sum_factor * .1
        self.direct_scf_tol = cutoff
        log.debug('Set RangeSeparationJKBuilder.direct_scf_tol to %g', cutoff)

        rcut_sr = estimate_rcut(rs_cell, self.omega,
                                exclude_dd_block=self.exclude_dd_block)
        supmol_sr = ft_ao.ExtendedMole.from_cell(rs_cell, kmesh, rcut_sr.max(), log)
        supmol_sr.omega = -self.omega
        self.supmol_sr = supmol_sr.strip_basis(rcut_sr)
        log.debug('supmol nbas = %d cGTO = %d pGTO = %d',
                  supmol_sr.nbas, supmol_sr.nao, supmol_sr.npgto_nr())

        if self.has_long_range():
            rcut = rsdf_builder.estimate_ft_rcut(
                rs_cell, exclude_dd_block=self.exclude_dd_block)
            supmol_ft = rsdf_builder._ExtendedMoleFT.from_cell(
                rs_cell, kmesh, rcut.max(), log)
            supmol_ft.exclude_dd_block = self.exclude_dd_block
            self.supmol_ft = supmol_ft.strip_basis(rcut)
            log.debug('sup-mol-ft nbas = %d cGTO = %d pGTO = %d',
                      supmol_ft.nbas, supmol_ft.nao, supmol_ft.npgto_nr())

            self.cell_d = rs_cell.smooth_basis_cell()
            if self.cell_d.nbas > 0:
                rcut = ft_ao.estimate_rcut(self.cell_d)
                supmol_d = ft_ao.ExtendedMole.from_cell(
                    self.cell_d, self.bvk_kmesh, rcut.max(), log)
                self.supmol_d = supmol_d.strip_basis(rcut)
                log.debug('supmol_d nbas = %d cGTO = %d', supmol_d.nbas, supmol_d.nao)
        log.timer_debug1('initializing supmol', *cpu0)

        self._cintopt = _vhf.make_cintopt(supmol_sr._atm, supmol_sr._bas,
                                          supmol_sr._env, 'int2e')
        nbas = supmol_sr.nbas
        qindex = np.empty((3,nbas,nbas), dtype=np.int16)
        ao_loc = supmol_sr.ao_loc
        with supmol_sr.with_integral_screen(self.direct_scf_tol**2):
            libpbc.PBCVHFnr_int2e_q_cond(
                libpbc.int2e_sph, self._cintopt,
                qindex.ctypes.data_as(ctypes.c_void_p),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                supmol_sr._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol_sr.natm),
                supmol_sr._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol_sr.nbas),
                supmol_sr._env.ctypes.data_as(ctypes.c_void_p))
        libpbc.PBCVHFnr_sindex(
            qindex[2:].ctypes.data_as(ctypes.c_void_p),
            supmol_sr._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol_sr.natm),
            supmol_sr._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol_sr.nbas),
            supmol_sr._env.ctypes.data_as(ctypes.c_void_p))

        if self.exclude_dd_block:
            # Remove the smooth-smooth basis block.
            smooth_idx = supmol_sr.bas_type_to_indices(ft_ao.SMOOTH_BASIS)
            qindex[0,smooth_idx[:,None], smooth_idx] = INDEX_MIN

        self.qindex = qindex

        log.timer('initializing qindex', *cpu0)
        return self

    @property
    def qindex(self):
        if self._qindex is None or isinstance(self._qindex, np.ndarray):
            return self._qindex
        return self._qindex['qindex'][:]

    @qindex.setter
    def qindex(self, x):
        if x.size < self.max_memory*.2e6:
            self._qindex = x
        else:
            self._qindex = lib.H5TmpFile()
            self._qindex['qindex'] = x
            self._qindex.flush()

    def _sort_qcond_cell0(self, seg_loc, seg2sh, nbasp, qindex):
        '''Sort qcond for ij-pair in cell0 so that loops in PBCVHF_direct_drv
        can be "break" early
        '''
        qcell0_ijij = _qcond_cell0_abstract(qindex[0], seg_loc, seg2sh, nbasp)
        qcell0_iijj = _qcond_cell0_abstract(qindex[1], seg_loc, seg2sh, nbasp)
        idx = np.asarray(qcell0_ijij.ravel().argsort(kind='stable'), np.int32)[::-1]
        jsh_idx, ish_idx = divmod(idx, nbasp)
        ish_idx = np.asarray(ish_idx, dtype=np.int32, order='C')
        jsh_idx = np.asarray(jsh_idx, dtype=np.int32, order='C')
        return qcell0_ijij, qcell0_iijj, ish_idx, jsh_idx

    def _get_jk_sr(self, dm_kpts, hermi=1, kpts=None, kpts_band=None,
                   with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega is not None:  # J/K for RSH functionals
            raise NotImplementedError

        cpu0 = logger.process_clock(), logger.perf_counter()
        log = logger.new_logger(self)

        comp = 1
        nkpts = kpts.shape[0]
        supmol = self.supmol_sr
        cell = self.cell
        rs_cell = self.rs_cell
        nao = cell.nao
        bvk_ncells, rs_nbas, nimgs = supmol.bas_mask.shape

        if dm_kpts.ndim != 4:
            dm = dm_kpts.reshape(-1, nkpts, nao, nao)
        else:
            dm = dm_kpts
        n_dm = dm.shape[0]

        cutoff = int(np.log(self.direct_scf_tol) * LOG_ADJUST)

        self._sr_without_dddd = (
            self.allow_drv_nodddd and
            not self.exclude_dd_block and
            self.cell_d.nbas > 0 and
            self.has_long_range() and
            (cell.dimension == 3 or cell.low_dim_ft_type != 'inf_vacuum'))

        if self._sr_without_dddd:
            # To exclude (dd|dd) block, diffused shell needs to be independent
            # segments in PBCVHF_direct_drv1. Decontract the segments in rs-shell.
            # map ao indices in cell to ao indices in rs_cell
            log.debug1('get_jk_sr with PBCVHF_direct_drv_nodddd')
            drv = libpbc.PBCVHF_direct_drv_nodddd
            ao_map = lib.locs_to_indices(cell.ao_loc, rs_cell.bas_map)
            rs_nao = ao_map.size
            rs_dm = np.empty((n_dm,nkpts*rs_nao,rs_nao), dtype=dm.dtype)
            idx = np.arange(nkpts)[:,None] * nao + ao_map
            idx = np.asarray(idx, dtype=np.int32).ravel()
            for i in range(n_dm):
                lib.take_2d(dm[i].reshape(nkpts*nao,nao), idx, ao_map, out=rs_dm[i])
            dm = rs_dm.reshape(n_dm,nkpts,rs_nao,rs_nao)
            nbasp = rs_cell.nbas
            cell0_ao_loc = rs_cell.ao_loc

            # uncontracted (local, diffused) basis segments in bvk-cell
            seg_loc = lib.locs_to_indices(
                supmol.seg_loc, np.arange(supmol.seg_loc.size-1))
            seg_loc = np.append(seg_loc, supmol.seg_loc[-1]).astype(np.int32)
        else:
            drv = libpbc.PBCVHF_direct_drv
            nbasp = cell.nbas  # The number of shells in the primitive cell
            cell0_ao_loc = cell.ao_loc
            seg_loc = supmol.seg_loc

        qindex = self.qindex
        qcell0_ijij, qcell0_iijj, ish_idx, jsh_idx = \
                self._sort_qcond_cell0(seg_loc, supmol.seg2sh, nbasp, qindex)

        weight = 1. / nkpts
        expLk = np.exp(1j*np.dot(supmol.bvkmesh_Ls, kpts.T))
        # Utilized symmetry sc_dm[R,S] = sc_dm[S-R] = sc_dm[(S-R)%N]
        #:phase = expLk / nkpts**.5
        #:sc_dm = lib.einsum('Rk,nkuv,Sk->nRuSv', phase, sc_dm, phase.conj())
        sc_dm = lib.einsum('k,Sk,nkuv->nSuv', expLk[0]*weight, expLk.conj(), dm)
        dm_translation = k2gamma.double_translation_indices(self.bvk_kmesh).astype(np.int32)
        dm_imag_max = abs(sc_dm.imag).max()
        is_complex_dm = dm_imag_max > 1e-6
        if is_complex_dm:
            if dm_imag_max < 1e-2:
                log.warn('DM in (BvK) cell has small imaginary part.  '
                         'It may be a signal of symmetry broken in k-point symmetry')
            sc_dm = np.vstack([sc_dm.real, sc_dm.imag])
            self.purify = False
        else:
            sc_dm = sc_dm.real
        nao1 = dm.shape[-1]
        sc_dm = np.asarray(sc_dm.reshape(-1, bvk_ncells, nao1, nao1), order='C')
        n_sc_dm = sc_dm.shape[0]

        # * sparse_ao_loc has dimension (Nk,nbas), corresponding to the
        # bvkcell with all basis
        sparse_ao_loc = nao1 * np.arange(bvk_ncells)[:,None] + cell0_ao_loc[:-1]
        sparse_ao_loc = np.append(sparse_ao_loc.ravel(), nao1 * bvk_ncells)
        dm_cond = [lib.condense('NP_absmax', d, sparse_ao_loc, cell0_ao_loc)
                   for d in sc_dm.reshape(n_sc_dm, bvk_ncells*nao1, nao1)]
        dm_cond = np.max(dm_cond, axis=0)
        dm_cond[dm_cond < 1e-100] = 1e-100
        dmindex = np.log(dm_cond)
        dmindex *= LOG_ADJUST
        dmindex = np.asarray(dmindex, order='C', dtype=np.int16)
        dmindex = dmindex.reshape(bvk_ncells, nbasp, nbasp)
        dm_cond = None

        bvk_nbas = nbasp * bvk_ncells
        shls_slice = (0, nbasp, 0, bvk_nbas, 0, bvk_nbas, 0, bvk_nbas)

        cache_size = _get_cache_size(cell, 'int2e_sph')
        cell0_dims = cell0_ao_loc[1:] - cell0_ao_loc[:-1]
        cache_size += int(cell0_dims.max())**4 * comp * 2

        if hermi:
            fdot_suffix = 's2kl'
        else:
            fdot_suffix = 's1'
        nvs = 1
        if with_j and with_k:
            fdot = 'PBCVHF_contract_jk_' + fdot_suffix
            nvs = 2
        elif with_j:
            fdot = 'PBCVHF_contract_j_' + fdot_suffix
        else:  # with_k
            fdot = 'PBCVHF_contract_k_' + fdot_suffix
        vs = np.zeros((nvs, n_sc_dm, nao1, bvk_ncells, nao1))

        if supmol.cart:
            intor = 'PBCint2e_cart'
        else:
            intor = 'PBCint2e_sph'

        drv(getattr(libpbc, fdot), getattr(libpbc, intor),
            vs.ctypes.data_as(ctypes.c_void_p),
            sc_dm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(vs.size), ctypes.c_int(n_sc_dm),
            ctypes.c_int(bvk_ncells), ctypes.c_int(nimgs), ctypes.c_int(nkpts),
            ctypes.c_int(nbasp), ctypes.c_int(comp),
            seg_loc.ctypes.data_as(ctypes.c_void_p),
            supmol.seg2sh.ctypes.data_as(ctypes.c_void_p),
            cell0_ao_loc.ctypes.data_as(ctypes.c_void_p),
            rs_cell.bas_type.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*8)(*shls_slice),
            dm_translation.ctypes.data_as(ctypes.c_void_p),
            qindex.ctypes.data_as(ctypes.c_void_p),
            dmindex.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cutoff),
            qcell0_ijij.ctypes.data_as(ctypes.c_void_p),
            qcell0_iijj.ctypes.data_as(ctypes.c_void_p),
            ish_idx.ctypes.data_as(ctypes.c_void_p),
            jsh_idx.ctypes.data_as(ctypes.c_void_p),
            self._cintopt, ctypes.c_int(cache_size),
            supmol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.natm),
            supmol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.nbas),
            supmol._env.ctypes.data_as(ctypes.c_void_p))

        if is_complex_dm:
            vs = vs[:,:n_dm] + vs[:,n_dm:] * 1j

        if self._sr_without_dddd:
            jdx = np.arange(bvk_ncells)[:,None] * nao + ao_map
            jdx = np.asarray(jdx, dtype=np.int32).ravel()
            rs_vs = vs.reshape(nvs*n_dm, nao1, bvk_ncells*nao1)
            vs = np.zeros((nvs*n_dm, nao, bvk_ncells*nao), dtype=rs_vs.dtype)
            for i in range(nvs*n_dm):
                lib.takebak_2d(vs[i], rs_vs[i], ao_map, jdx, thread_safe=False)
            vs = vs.reshape(nvs,n_dm,nao,bvk_ncells,nao)

        if kpts_band is not None:
            kpts_band = np.reshape(kpts_band, (-1, 3))
            subset_only = intersection(kpts, kpts_band).size == len(kpts_band)
            if not subset_only:
                log.warn('Approximate J/K matrices at kpts_band '
                         'with the bvk-cell derived from kpts')
                expLk = np.exp(1j*np.dot(supmol.bvkmesh_Ls, kpts_band.T))
        vs = lib.einsum('snpRq,Rk->snkpq', vs, expLk)
        vs = np.asarray(vs, order='C')
        log.timer_debug1('short range part vj and vk', *cpu0)
        return vs

    def get_jk(self, dm_kpts, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        cell = self.cell
        if omega is not None:  # J/K for RSH functionals
            if omega > 0:  # Long-range part only, call AFTDF
                dfobj = aft.AFTDF(cell, self.kpts)
                ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
                dfobj.mesh = cell.cutoff_to_mesh(ke_cutoff)
                return dfobj.get_jk(dm_kpts, hermi, kpts, kpts_band,
                                    with_j, with_k, omega, exxdiv)
            elif omega < 0:  # Short-range part only
                if self.omega is not None and self.omega != omega:
                    raise RuntimeError(f'omega = {omega}, self.omega = {self.omega}')
                raise NotImplementedError

        if self.supmol_sr is None:
            self.build()

        # Does not support to specify arbitrary kpts
        if kpts is not None and abs(kpts-self.kpts).max() > 1e-7:
            raise RuntimeError('kpts error. kpts cannot be modified in RSJK')
        kpts = np.asarray(self.kpts.reshape(-1, 3), order='C')

        mo_coeff = getattr(dm_kpts, 'mo_coeff', None)
        mo_occ = getattr(dm_kpts, 'mo_occ', None)
        dm_kpts = np.asarray(dm_kpts)
        dms = _format_dms(dm_kpts, kpts)

        # compute delta vs if dm is obtained from SCF make_rdm1
        if mo_coeff is not None:
            last_dm, last_vs = self._last_vs
            vs = self._get_jk_sr(dms-last_dm, hermi, kpts, kpts_band,
                                 with_j, with_k, omega, exxdiv)
            vs += last_vs
            self._last_vs = (dms, vs.copy())
            last_dm = last_vs = None

            # dm ~= dm_factor * dm_factor.T
            n_dm, nkpts, nao = dms.shape[:3]
            # mo_coeff, mo_occ are not a list of aligned array if
            # remove_lin_dep was applied to scf object
            if dm_kpts.ndim == 4:  # KUHF
                nocc = max(max(np.count_nonzero(x > 0) for x in z) for z in mo_occ)
                dm_factor = [[x[:,:nocc] for x in mo] for mo in mo_coeff]
                occs = [[x[:nocc] for x in z] for z in mo_occ]
            else:  # KRHF
                nocc = max(np.count_nonzero(x > 0) for x in mo_occ)
                dm_factor = [[mo[:,:nocc] for mo in mo_coeff]]
                occs = [[x[:nocc] for x in mo_occ]]
            dm_factor = np.array(dm_factor, dtype=np.complex128, order='C')
            dm_factor *= np.sqrt(np.array(occs, dtype=np.double))[:,:,None]
        else:
            vs = self._get_jk_sr(dms, hermi, kpts, kpts_band,
                                 with_j, with_k, omega, exxdiv)
            dm_factor = None
        dms = lib.tag_array(dms, dm_factor=dm_factor)

        if with_j and with_k:
            vj, vk = vs
        elif with_j:
            vj, vk = vs[0], None
        else:
            vj, vk = None, vs[0]

        if self.purify and kpts_band is None:
            phase = np.exp(1j*np.dot(self.supmol_sr.bvkmesh_Ls, kpts.T))
            phase /= np.sqrt(len(kpts))
        else:
            phase = None

        if with_j:
            if self.has_long_range():
                vj += self._get_vj_lr(dms, hermi, kpts, kpts_band)
            if hermi:
                vj = (vj + vj.conj().transpose(0,1,3,2)) * .5
            vj = _purify(vj, phase)
            vj = _format_jks(vj, dm_kpts, kpts_band, kpts)
            if is_zero(kpts) and dm_kpts.dtype == np.double:
                vj = vj.real.copy()

        if with_k:
            if self.has_long_range():
                approx_vk_lr = dm_factor is None and self.approx_vk_lr_missing_mo
                if not approx_vk_lr:
                    vk += self._get_vk_lr(dms, hermi, kpts, kpts_band, exxdiv)
                else:
                    mesh1 = np.array(self.mesh)//3*2 + 1
                    logger.debug(self, 'Approximate lr_k with mesh %s', mesh1)
                    with lib.temporary_env(self, mesh=mesh1):
                        vk += self._get_vk_lr(dms, hermi, kpts, kpts_band, exxdiv)
                    self.approx_vk_lr_missing_mo = False
            if hermi:
                vk = (vk + vk.conj().transpose(0,1,3,2)) * .5
            vk = _purify(vk, phase)
            vk = _format_jks(vk, dm_kpts, kpts_band, kpts)
            if is_zero(kpts) and dm_kpts.dtype == np.double:
                vk = vk.real.copy()

        return vj, vk

    weighted_coulG = aft.weighted_coulG

    def weighted_coulG_LR(self, kpt=np.zeros(3), exx=False, mesh=None):
        # The long range part Coulomb kernel has to be computed as the
        # difference between coulG(cell.omega) - coulG(self.omega). It allows this
        # module to handle the SR- and regular integrals in the same framework
        return (self.weighted_coulG(kpt, exx, mesh) -
                self.weighted_coulG_SR(kpt, exx, mesh))

    def weighted_coulG_SR(self, kpt=np.zeros(3), exx=False, mesh=None):
        return self.weighted_coulG(kpt, False, mesh, -self.omega)

    def _get_vj_lr(self, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None):
        '''
        Long-range part of J matrix
        '''
        if kpts_band is None:
            return self._get_lr_j_kpts(dm_kpts, hermi, kpts)

        logger.warn(self, 'Approximate kpts_band for vj with k-point projection')
        vj = self._get_lr_j_kpts(dm_kpts, hermi, kpts)
        pk2k = addons._k2k_projection(kpts, kpts_band, self.supmol_ft.bvkmesh_Ls)
        return lib.einsum('nkpq,kh->nhpq', vj, pk2k)

    def _get_lr_j_kpts(self, dm_kpts, hermi=1, kpts=np.zeros((1,3))):
        '''
        Long-range part of J matrix
        '''
        if len(kpts) == 1 and not is_zero(kpts):
            raise NotImplementedError('Single k-point get-j')

        cpu0 = logger.process_clock(), logger.perf_counter()
        log = logger.new_logger(self)
        cell = self.cell
        rs_cell = self.rs_cell
        if self.exclude_dd_block or self._sr_without_dddd:
            cell_d = self.cell_d
            naod = cell_d.nao
            ngrids_d = np.prod(cell_d.mesh)
        else:
            cell_d = None
            naod = ngrids_d = 0
        kpts = np.asarray(kpts.reshape(-1, 3), order='C')
        dms = dm_kpts
        n_dm, nkpts, nao = dms.shape[:3]
        mesh = self.mesh
        ngrids = np.prod(mesh)

        is_real = is_zero(kpts) and dms.dtype == np.double
        if is_real:
            vj_kpts = np.zeros((n_dm,nkpts,nao,nao))
        else:
            vj_kpts = np.zeros((n_dm,nkpts,nao,nao), dtype=np.complex128)

        # TODO: aosym == 's2'
        aosym = 's1'
        ft_kern = self.supmol_ft.gen_ft_kernel(
            aosym, return_complex=False, kpts=kpts, verbose=log)
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])

        kpt_allow = np.zeros(3)
        coulG = self.weighted_coulG(kpt_allow, False, mesh)

        if (cell.dimension == 3 or
            (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum')):
            G0_idx = 0  # due to np.fft.fftfreq convention
            # G=0 associated to 2e integrals in real-space
            coulG_SR_at_G0 = np.pi/self.omega**2
            # For cell.dimension = 2, coulG is computed with truncated Coulomb
            # interactions. The 3d coulG_SR below is to remove the analytical
            # SR from get_jk_sr (which was computed with full Coulomb) then to
            # add truncated Coulomb for AFT part.
            with lib.temporary_env(cell, dimension=3):
                coulG_SR = self.weighted_coulG_SR(kpt_allow, False, mesh)
            coulG_SR[G0_idx] += coulG_SR_at_G0 * kws
        else:
            coulG_SR = self.weighted_coulG_SR(kpt_allow, False, mesh)
            coulG_SR_at_G0 = None

        if naod > 0:
            smooth_bas_mask = rs_cell.bas_type == ft_ao.SMOOTH_BASIS
            smooth_bas_idx = rs_cell.bas_map[smooth_bas_mask]
            ao_d_idx = rs_cell.get_ao_indices(smooth_bas_idx, cell.ao_loc)

        mem_now = lib.current_memory()[0]
        max_memory = self.max_memory - mem_now
        log.debug1('max_memory = %d MB (%d in use)', max_memory+mem_now, mem_now)
        Gblksize = max(24, int(max_memory*.8e6/16/(nao**2+naod**2)/(nkpts+1))//8*8)
        Gblksize = min(Gblksize, ngrids)
        log.debug1('Gblksize = %d', Gblksize)

        if not self.exclude_dd_block or naod == 0:
            log.debug1('get_lr_j_kpts with aft_aopair')
            # Long-range part is calculated as the difference
            # coulG(cell.omega) - coulG(self.omega) . It can support both regular
            # integrals and LR integrals.
            coulG_LR = coulG - coulG_SR
            buf = np.empty(nkpts*Gblksize*nao**2*2)
            for p0, p1 in lib.prange(0, ngrids, Gblksize):
                Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt_allow, out=buf)
                aft_jk._update_vj_(vj_kpts, Gpq, dms, coulG_LR[p0:p1])
            Gpq = buf = None

            if self._sr_without_dddd and naod > 0:
                log.debug1('get_lr_j_kpts dd block with mesh %s', cell_d.mesh)
                if cell.dimension < 2:
                    raise NotImplementedError
                if cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum':
                    raise NotImplementedError
                aoR_ks, aoI_ks = rsdf_builder._eval_gto(cell_d, cell_d.mesh, kpts)

                # rho = einsum('nkji,kig,kjg->ng', dm, ao.conj(), ao)
                rho = np.zeros((n_dm, ngrids_d))
                tmpR = np.empty((naod, ngrids_d))
                tmpI = np.empty((naod, ngrids_d))
                dm_dd = dms[:,:,ao_d_idx[:,None],ao_d_idx]
                dmR_dd = np.asarray(dm_dd.real, order='C')
                dmI_dd = np.asarray(dm_dd.imag, order='C')
                # vG = einsum('ij,gji->g', dm_dd[k], aoao[k]) * coulG
                for i in range(n_dm):
                    for k in range(nkpts):
                        zdotNN(dmR_dd[i,k].T, dmI_dd[i,k].T,
                               aoR_ks[k], aoI_ks[k], 1, tmpR, tmpI)
                        rho[i] += np.einsum('ig,ig->g', aoR_ks[k], tmpR)
                        rho[i] += np.einsum('ig,ig->g', aoI_ks[k], tmpI)

                if coulG_SR_at_G0 is not None:
                    with lib.temporary_env(cell_d, dimension=3):
                        coulG_SR = pbctools.get_coulG(cell_d, omega=-self.omega)
                    coulG_SR[G0_idx] += coulG_SR_at_G0
                else:
                    coulG_SR = pbctools.get_coulG(cell_d, omega=-self.omega)
                coulG_SR *= cell.vol / ngrids_d
                vG = pbctools.fft(rho, cell_d.mesh) * coulG_SR
                vR = pbctools.ifft(vG, cell_d.mesh).real

                nband = nkpts
                vjR_dd = np.empty((naod,naod))
                vjI_dd = np.empty((naod,naod))
                for i in range(n_dm):
                    for k in range(nband):
                        aowR = np.einsum('xi,x->xi', aoR_ks[k].T, vR[i])
                        aowI = np.einsum('xi,x->xi', aoI_ks[k].T, vR[i])
                        zdotCN(aoR_ks[k], aoI_ks[k], aowR, aowI, 1, vjR_dd, vjI_dd)
                        if is_real:
                            vj = vjR_dd
                        else:
                            vj = vjR_dd + vjI_dd*1j
                        lib.takebak_2d(vj_kpts[i,k], vj, ao_d_idx, ao_d_idx,
                                       thread_safe=False)

        elif naod > 0 and ngrids < ngrids_d:
            # Prefer AFTDF for everything otherwise cell_d.mesh have to be used
            # for AFTDF
            log.debug1('get_lr_j_kpts dd block cached aft_aopair_dd')
            ft_kern_dd = self.supmol_d.gen_ft_kernel(
                aosym, return_complex=False, kpts=kpts, verbose=log)

            def merge_dd(Gpq, p0, p1):
                '''Merge diffused basis block into ao-pair tensor inplace'''
                GpqR, GpqI = Gpq
                pqG_ddR, pqG_ddI = ft_kern_dd(Gv[p0:p1], gxyz[p0:p1], Gvbase,
                                              kpt_allow, kpts)
                # Gpq should be an array of (nkpts,ni,nj,ngrids) in C order
                if not GpqR[0].flags.c_contiguous:
                    assert GpqR[0].strides[0] == 8  # stride for grids
                for k in range(nkpts):
                    libpbc.PBC_ft_fuse_dd_s1(
                        GpqR[k].ctypes.data_as(ctypes.c_void_p),
                        GpqI[k].ctypes.data_as(ctypes.c_void_p),
                        pqG_ddR[k].ctypes.data_as(ctypes.c_void_p),
                        pqG_ddI[k].ctypes.data_as(ctypes.c_void_p),
                        ao_d_idx.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(nao), ctypes.c_int(naod), ctypes.c_int(p1-p0))
                return (GpqR, GpqI)

            buf = np.empty(nkpts*Gblksize*nao**2*2)
            for p0, p1 in lib.prange(0, ngrids, Gblksize):
                Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt_allow, out=buf)
                aft_jk._update_vj_(vj_kpts, Gpq, dms, -coulG_SR[p0:p1])
                Gpq = merge_dd(Gpq, p0, p1)
                aft_jk._update_vj_(vj_kpts, Gpq, dms, coulG[p0:p1])
            Gpq = buf = None

        elif naod > 0:
            log.debug1('get_lr_j_kpts dd block cached fft_aopair_dd')
            aoR_ks, aoI_ks = rsdf_builder._eval_gto(cell_d, mesh, kpts)

            # rho = einsum('nkji,kig,kjg->ng', dm, ao.conj(), ao)
            rho = np.zeros((n_dm, ngrids))
            tmpR = np.empty((naod, ngrids))
            tmpI = np.empty((naod, ngrids))
            dm_dd = dms[:,:,ao_d_idx[:,None],ao_d_idx]
            dmR_dd = np.asarray(dm_dd.real, order='C')
            dmI_dd = np.asarray(dm_dd.imag, order='C')
            # vG = einsum('ij,gji->g', dm_dd[k], aoao[k]) * coulG
            for i in range(n_dm):
                for k in range(nkpts):
                    zdotNN(dmR_dd[i,k].T, dmI_dd[i,k].T, aoR_ks[k], aoI_ks[k], 1, tmpR, tmpI)
                    rho[i] += np.einsum('ig,ig->g', aoR_ks[k], tmpR)
                    rho[i] += np.einsum('ig,ig->g', aoI_ks[k], tmpI)
            vG_dd = pbctools.ifft(rho, mesh) * cell.vol * coulG
            tmpR = tmpI = dmR_dd = dmI_dd = None
            cpu1 = log.timer_debug1('get_lr_j_kpts dd block', *cpu0)

            mem_now = lib.current_memory()[0]
            max_memory = self.max_memory - mem_now
            log.debug1('max_memory = %d MB (%d in use)', max_memory+mem_now, mem_now)
            Gblksize = min(Gblksize, int(max_memory*.8e6/16/(nao**2+naod**2)/(nkpts+1))//8*8)
            log.debug1('Gblksize = %d', Gblksize)
            buf = np.empty(nkpts*Gblksize*nao**2*2)
            for p0, p1 in lib.prange(0, ngrids, Gblksize):
                Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt_allow, out=buf)
                #: aft_jk._update_vj_(vj_kpts, aoaoks, dms, coulG[p0:p1], 1)
                #: aft_jk._update_vj_(vj_kpts, aoaoks, dms, coulG_SR[p0:p1], -1)
                GpqR, GpqI = Gpq
                for i in range(n_dm):
                    rhoR = np.einsum('kij,kgij->g', dms[i].real, GpqR)
                    rhoI = -np.einsum('kij,kgij->g', dms[i].real, GpqI)
                    if not is_real:
                        rhoR += np.einsum('kij,kgij->g', dms[i].imag, GpqI)
                        rhoI += np.einsum('kij,kgij->g', dms[i].imag, GpqR)
                    rho = rhoR + rhoI * 1j
                    vG = vG_dd[i,p0:p1]
                    # Update vG_dd inplace to include rho contributions
                    vG += coulG[p0:p1] * rho
                    vG_SR = coulG_SR[p0:p1] * rho
                    # vG_LR contains full vG of dd-block and vG_LR of rest blocks
                    vG_LR = vG - vG_SR
                    vj_kpts[i].real += np.einsum('g,kgij->kij', vG_LR.real, GpqR)
                    vj_kpts[i].real -= np.einsum('g,kgij->kij', vG_LR.imag, GpqI)
                    if not is_real:
                        vj_kpts[i].imag += np.einsum('g,kgij->kij', vG_LR.real, GpqI)
                        vj_kpts[i].imag += np.einsum('g,kgij->kij', vG_LR.imag, GpqR)
                Gpq = None
            log.timer_debug1('get_lr_j_kpts ft_aopair', *cpu1)

            vR = pbctools.fft(vG_dd, mesh).real * (cell.vol/ngrids)
            vjR_dd = np.empty((naod, naod))
            vjI_dd = np.empty((naod, naod))
            for i in range(n_dm):
                for k in range(nkpts):
                    tmpR = aoR_ks[k] * vR[i]
                    tmpI = aoI_ks[k] * vR[i]
                    zdotCN(aoR_ks[k], aoI_ks[k], tmpR.T, tmpI.T, 1, vjR_dd, vjI_dd)
                    if is_real:
                        vj_dd = vjR_dd
                    else:
                        vj_dd = vjR_dd + vjI_dd * 1j
                    lib.takebak_2d(vj_kpts[i,k], vj_dd, ao_d_idx, ao_d_idx,
                                   thread_safe=False)

        else:
            raise RuntimeError(f'exclude_dd_block={self.exclude_dd_block} '
                               f'sr_without_dddd={self._sr_without_dddd} naod={naod}')

        if nkpts > 1:
            vj_kpts *= 1./nkpts
        log.timer_debug1('get_lr_j_kpts', *cpu0)
        return vj_kpts

    def _get_vk_lr(self, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None,
                   exxdiv=None):
        '''
        Long-range part of K matrix
        '''
        if kpts_band is None:
            return self._get_lr_k_kpts(dm_kpts, hermi, kpts, exxdiv)

        # Note: Errors in k2k-projection for vj is relatively small.
        # k2k-projection for vk has significant finite-size errors.
        logger.warn(self, 'Approximate kpts_band for vk with k-point projection')
        vk = self._get_lr_k_kpts(dm_kpts, hermi, kpts, exxdiv=None)
        pk2k = addons._k2k_projection(kpts, kpts_band, self.supmol_ft.bvkmesh_Ls)
        vk = lib.einsum('nkpq,kh->nhpq', vk, pk2k)
        if exxdiv == 'ewald':
            _ewald_exxdiv_for_G0(self.cell, kpts, dm_kpts, vk, kpts_band)
        return vk

    def _get_lr_k_kpts(self, dm_kpts, hermi=1, kpts=np.zeros((1,3)), exxdiv=None):
        '''
        Long-range part of K matrix
        '''
        cpu0 = cpu1 = logger.process_clock(), logger.perf_counter()
        log = logger.new_logger(self)
        cell = self.cell
        rs_cell = self.rs_cell
        if self.exclude_dd_block or self._sr_without_dddd:
            cell_d = self.cell_d
            naod = cell_d.nao
            ngrids_d = np.prod(cell_d.mesh)
        else:
            cell_d = None
            naod = ngrids_d = 0

        mesh = self.mesh
        ngrids = np.prod(mesh)
        dms = dm_kpts
        n_dm, nkpts, nao = dms.shape[:3]
        vkR = np.zeros((n_dm,nkpts,nao,nao))
        vkI = np.zeros((n_dm,nkpts,nao,nao))
        vk = [vkR, vkI]
        weight = 1. / nkpts

        # Test if vk[k] == vk[k_conj].conj()
        t_rev_pairs = group_by_conj_pairs(cell, kpts, return_kpts_pairs=False)
        try:
            t_rev_pairs = np.asarray(t_rev_pairs, dtype=np.int32, order='F')
        except TypeError:
            t_rev_pairs = [[k, k] if k_conj is None else [k, k_conj]
                           for k, k_conj in t_rev_pairs]
            t_rev_pairs = np.asarray(t_rev_pairs, dtype=np.int32, order='F')
        log.debug1('Num kpts conj_pairs %d', len(t_rev_pairs))
        time_reversal_symmetry = self.time_reversal_symmetry
        if time_reversal_symmetry:
            for k, k_conj in t_rev_pairs:
                if k != k_conj and abs(dms[:,k_conj] - dms[:,k].conj()).max() > 1e-6:
                    time_reversal_symmetry = False
                    log.debug2('Disable time_reversal_symmetry')
                    break

        # TODO: aosym == 's2'
        aosym = 's1'
        if time_reversal_symmetry:
            k_to_compute = np.zeros(nkpts, dtype=np.int8)
            k_to_compute[t_rev_pairs[:,0]] = 1
        else:
            k_to_compute = np.ones(nkpts, dtype=np.int8)
            t_rev_pairs = None

        dm_factor = getattr(dm_kpts, 'dm_factor', None)
        contract_mo_early = False
        if dm_factor is None:
            dmsR = np.asarray(dms.real, order='C')
            dmsI = np.asarray(dms.imag, order='C')
            dm = [dmsR, dmsI]
            dm_factor = None
            if np.count_nonzero(k_to_compute) >= 2 * lib.num_threads():
                update_vk = aft_jk._update_vk1_
            else:
                update_vk = aft_jk._update_vk_
        else:
            # dm ~= dm_factor * dm_factor.T
            nocc = dm_factor.shape[-1]
            if nocc == 0:
                return vkR

            bvk_ncells, rs_nbas, nimgs = self.supmol_ft.bas_mask.shape
            s_nao = self.supmol_ft.nao
            contract_mo_early = (time_reversal_symmetry and naod == 0 and
                                 bvk_ncells*nao*6 > s_nao*nocc*n_dm)
            log.debug2('time_reversal_symmetry = %s bvk_ncells = %d '
                       's_nao = %d nocc = %d n_dm = %d',
                       time_reversal_symmetry, bvk_ncells, s_nao, nocc, n_dm)
            log.debug2('Use algorithm contract_mo_early = %s', contract_mo_early)
            if contract_mo_early:
                bvk_kmesh = self.bvk_kmesh
                rcut = ft_ao.estimate_rcut(cell)
                supmol = ft_ao.ExtendedMole.from_cell(cell, bvk_kmesh, rcut.max())
                supmol = supmol.strip_basis(rcut)
                s_nao = supmol.nao
                moR, moI = aft_jk._mo_k2gamma(supmol, dm_factor, kpts, t_rev_pairs)
                if abs(moI).max() < 1e-5:
                    dm = [moR, None]
                    ft_kern = aft_jk._gen_ft_kernel_fake_gamma(cell, supmol, aosym)
                    update_vk = aft_jk._update_vk_fake_gamma
                else:
                    contract_mo_early = False
                moR = moI = None

            if not contract_mo_early:
                dm = [np.asarray(dm_factor.real, order='C'),
                      np.asarray(dm_factor.imag, order='C')]
                if np.count_nonzero(k_to_compute) >= 2 * lib.num_threads():
                    update_vk = aft_jk._update_vk1_dmf
                else:
                    update_vk = aft_jk._update_vk_dmf
        log.debug2('set update_vk to %s', update_vk)

        if not contract_mo_early:
            ft_kern = self.supmol_ft.gen_ft_kernel(
                aosym, return_complex=False, kpts=kpts, verbose=log)

        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        Gv = np.asarray(Gv, order='F')
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        G0_idx = 0
        if (cell.dimension == 3 or
            (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum')):
            coulG_SR_at_G0 = np.pi/self.omega**2 * kws
        else:
            coulG_SR_at_G0 = None

        if naod > 0:
            smooth_bas_mask = rs_cell.bas_type == ft_ao.SMOOTH_BASIS
            smooth_bas_idx = rs_cell.bas_map[smooth_bas_mask]
            ao_d_idx = rs_cell.get_ao_indices(smooth_bas_idx, cell.ao_loc)

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, (self.max_memory - mem_now))
        log.debug1('max_memory = %d MB (%d in use)', max_memory+mem_now, mem_now)

        if self.exclude_dd_block and not self._sr_without_dddd and naod > 0:
            cache_size = (naod*(naod+1)*ngrids*(nkpts+1))*16e-6
            log.debug1('naod = %d cache_size = %d', naod, cache_size)

            # fft_aopair_dd seems less efficient than aft_aopair_dd
            if 0 and max_memory * .5 > cache_size and cell.dimension >= 2:
                from pyscf.pbc.dft.multigrid import _take_5d
                mesh_d = cell_d.mesh
                log.debug1('merge_dd with cached fft_aopair_dd')
                aoR_ks, aoI_ks = rsdf_builder._eval_gto(cell_d, mesh_d, kpts)
                coords = cell_d.get_uniform_grids(mesh_d)
                max_memory -= cache_size
                gx = np.fft.fftfreq(mesh[0], 1./mesh[0]).astype(np.int32)
                gy = np.fft.fftfreq(mesh[1], 1./mesh[1]).astype(np.int32)
                gz = np.fft.fftfreq(mesh[2], 1./mesh[2]).astype(np.int32)

                def fft_aopair_dd(ki, kj, expmikr):
                    # einsum('g,ig,jg->ijg', expmikr, ao_ki.conj(), ao_kj)
                    pqG_ddR = np.empty((naod**2, ngrids_d))
                    pqG_ddI = np.empty((naod**2, ngrids_d))
                    expmikrR, expmikrI = expmikr
                    libpbc.PBC_zjoin_fCN_s1(
                        pqG_ddR.ctypes.data_as(ctypes.c_void_p),
                        pqG_ddI.ctypes.data_as(ctypes.c_void_p),
                        expmikrR.ctypes.data_as(ctypes.c_void_p),
                        expmikrI.ctypes.data_as(ctypes.c_void_p),
                        aoR_ks[ki].ctypes.data_as(ctypes.c_void_p),
                        aoI_ks[ki].ctypes.data_as(ctypes.c_void_p),
                        aoR_ks[kj].ctypes.data_as(ctypes.c_void_p),
                        aoI_ks[kj].ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(naod), ctypes.c_int(naod), ctypes.c_int(ngrids_d))
                    pqG_dd = np.empty((naod**2, ngrids_d), dtype=np.complex128)
                    pqG_dd.real = pqG_ddR
                    pqG_dd.imag = pqG_ddI
                    pqG_ddR = pqG_ddI = None
                    pqG_dd *= cell.vol / ngrids_d
                    pqG_dd = pbctools.fft(pqG_dd, mesh_d).reshape(naod, naod, *mesh_d)
                    pqG_dd = _take_5d(pqG_dd, (None, None, gx, gy, gz))
                    return np.asarray(pqG_dd.reshape(naod, naod, ngrids), order='C')

                cache = {}
                def merge_dd(Gpq, p0, p1, ki_lst, kj_lst):
                    '''Merge diffused basis block into ao-pair tensor inplace'''
                    expmikr = np.exp(-1j * np.dot(coords, kpts[kj_lst[0]]-kpts[ki_lst[0]]))
                    expmikrR = np.asarray(expmikr.real, order='C')
                    expmikrI = np.asarray(expmikr.imag, order='C')
                    GpqR, GpqI = Gpq
                    # Gpq should be an array of (nkpts,ni,nj,ngrids) in C order
                    if not GpqR[0].flags.c_contiguous:
                        assert GpqR[0].strides[0] == 8  # stride for grids
                    cpu0 = logger.process_clock(), logger.perf_counter()
                    for k, (ki, kj) in enumerate(zip(ki_lst, kj_lst)):
                        if (ki, kj) not in cache:
                            log.debug3('cache dd block (%d, %d)', ki, kj)
                            cache[ki, kj] = fft_aopair_dd(ki, kj, (expmikrR, expmikrI))

                        pqG_dd = cache[ki, kj]
                        libpbc.PBC_ft_zfuse_dd_s1(
                            GpqR[kj].ctypes.data_as(ctypes.c_void_p),
                            GpqI[kj].ctypes.data_as(ctypes.c_void_p),
                            pqG_dd.ctypes.data_as(ctypes.c_void_p),
                            ao_d_idx.ctypes.data_as(ctypes.c_void_p),
                            (ctypes.c_int*2)(p0, p1), ctypes.c_int(nao),
                            ctypes.c_int(naod), ctypes.c_int(ngrids))
                    log.timer_debug1('merge_dd', *cpu0)
                    return (GpqR, GpqI)

                cpu1 = log.timer_debug1('get_lr_k_kpts initializing dd block', *cpu1)

            else:
                log.debug1('merge_dd with aft_aopair_dd')
                ft_kern_dd = self.supmol_d.gen_ft_kernel(
                    aosym, return_complex=False, kpts=kpts, verbose=log)

                buf1 = None
                def merge_dd(Gpq, p0, p1, ki_lst, kj_lst):
                    '''Merge diffused basis block into ao-pair tensor inplace'''
                    kpt = kpts[kj_lst[0]] - kpts[ki_lst[0]]
                    GpqR, GpqI = Gpq
                    pqG_ddR, pqG_ddI = ft_kern_dd(Gv[p0:p1], gxyz[p0:p1], Gvbase,
                                                  kpt, out=buf1)
                    # Gpq should be an array of (nkpts,ni,nj,ngrids) in C order
                    if not GpqR[0].flags.c_contiguous:
                        assert GpqR[0].strides[0] == 8  # stride for grids
                    for k in range(nkpts):
                        libpbc.PBC_ft_fuse_dd_s1(
                            GpqR[k].ctypes.data_as(ctypes.c_void_p),
                            GpqI[k].ctypes.data_as(ctypes.c_void_p),
                            pqG_ddR[k].ctypes.data_as(ctypes.c_void_p),
                            pqG_ddI[k].ctypes.data_as(ctypes.c_void_p),
                            ao_d_idx.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(nao), ctypes.c_int(naod), ctypes.c_int(p1-p0))
                    return (GpqR, GpqI)

        if contract_mo_early:
            Gblksize = max(24, int((max_memory*1e6/16-nkpts*nao**2*3)/
                                   (nao*s_nao+nao*nkpts*nocc))//8*8)
            Gblksize = min(Gblksize, ngrids, 200000)
            log.debug1('Gblksize = %d', Gblksize)
            buf = np.empty(Gblksize*s_nao*nao*2)
        else:
            Gblksize = max(24, int(max_memory*.8e6/16/(nao**2+naod**2)/(nkpts+3))//8*8)
            Gblksize = min(Gblksize, ngrids, 200000)
            log.debug1('Gblksize = %d', Gblksize)
            buf = np.empty(nkpts*Gblksize*nao**2*2)
            if naod > 0:
                buf1 = np.empty(nkpts*Gblksize*naod**2*2)

        for group_id, (kpt, ki_idx, kj_idx, self_conj) \
                in enumerate(kk_adapted_iter(cell, kpts)):
            coulG = self.weighted_coulG(kpt, exxdiv, mesh)
            if coulG_SR_at_G0 is not None:
                # For cell.dimension = 2, coulG is computed with truncated coulomb
                # interactions. The 3d coulG_SR below is to remove the analytical
                # SR from get_jk_sr (which was computed with full Coulomb) then
                # add the truncated Coulomb for AFT part.
                with lib.temporary_env(cell, dimension=3):
                    coulG_SR = self.weighted_coulG_SR(kpt, False, mesh)
                if is_zero(kpt):
                    coulG_SR[G0_idx] += coulG_SR_at_G0
            else:
                coulG_SR = self.weighted_coulG_SR(kpt, False, mesh)

            if self.exclude_dd_block and not self._sr_without_dddd and naod > 0:
                for p0, p1 in lib.prange(0, ngrids, Gblksize):
                    # C ~ compact basis, D ~ diffused basis
                    # K matrix with coulG_LR:
                    # (CC|CC) (CC|CD) (CC|DC) (CD|CC) (CD|CD) (CD|DC) (DC|CC) (DC|CD) (DC|DC)
                    # K matrix with full coulG:
                    # (CC|DD) (CD|DD) (DC|DD) (DD|CC) (DD|CD) (DD|DC) (DD|DD)
                    log.debug3('update_vk [%s:%s]', p0, p1)
                    Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt, out=buf)
                    update_vk(vk, Gpq, dm, coulG_SR[p0:p1] * -weight, ki_idx,
                              kj_idx, not self_conj, k_to_compute, t_rev_pairs)
                    Gpq = merge_dd(Gpq, p0, p1, ki_idx, kj_idx)
                    update_vk(vk, Gpq, dm, coulG[p0:p1] * weight, ki_idx,
                              kj_idx, not self_conj, k_to_compute, t_rev_pairs)
                    Gpq = None
                # clear cache to release memory for merge_dd function
                cache = {}
            else:
                coulG_LR = coulG - coulG_SR
                for p0, p1 in lib.prange(0, ngrids, Gblksize):
                    log.debug3('update_vk [%s:%s]', p0, p1)
                    Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt, out=buf)
                    update_vk(vk, Gpq, dm, coulG_LR[p0:p1] * weight, ki_idx,
                              kj_idx, not self_conj, k_to_compute, t_rev_pairs)
                    Gpq = None
            cpu1 = log.timer_debug1(f'ft_aopair group {group_id}', *cpu1)

        if self._sr_without_dddd and naod > 0:
            # (DD|DD) with full coulG, rest terms with coulG_LR
            log.debug1('ft_aopair dd-block for dddd-block ERI')
            if cell.dimension < 2:
                raise NotImplementedError
            if cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum':
                raise NotImplementedError
            vkR_dd = np.zeros((n_dm,nkpts,naod,naod))
            vkI_dd = np.zeros((n_dm,nkpts,naod,naod))
            vk_dd = [vkR_dd, vkI_dd]
            if dm_factor is None:
                dmsR, dmsI = dm
                dmR_dd = np.asarray(dmsR[:,:,ao_d_idx[:,None],ao_d_idx], order='C')
                dmI_dd = np.asarray(dmsI[:,:,ao_d_idx[:,None],ao_d_idx], order='C')
                dm_dd = [dmR_dd, dmI_dd]
            else:
                assert update_vk is not aft_jk._update_vk_fake_gamma
                dmfR, dmfI = dm
                dmfR_dd = np.asarray(dmfR[:,:,ao_d_idx], order='C')
                dmfI_dd = np.asarray(dmfI[:,:,ao_d_idx], order='C')
                dm_dd = [dmfR_dd, dmfI_dd]

            # AFT mesh for ERI is usually smaller than cell_d.mesh
            ke_cutoff = aft.estimate_ke_cutoff(cell_d)
            mesh_d = cell_d.cutoff_to_mesh(ke_cutoff)
            log.debug1('lr_k dd-block ke_cutoff = %s mesh = %s', ke_cutoff, mesh_d)

            ft_kern_dd = self.supmol_d.gen_ft_kernel(
                aosym, return_complex=False, kpts=kpts, verbose=log)
            Gv, Gvbase, kws = cell_d.get_Gv_weights(mesh_d)
            gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
            ngrids_d = len(Gv)

            Gblksize = max(24, int(max_memory*.8e6/16/naod**2/(nkpts+max(nkpts,3)))//8*8)
            Gblksize = min(Gblksize, ngrids_d)
            log.debug1('Gblksize = %d', Gblksize)
            buf = np.empty(nkpts*Gblksize*naod**2*2)
            for group_id, (kpt, ki_idx, kj_idx, self_conj) \
                    in enumerate(kk_adapted_iter(cell, kpts)):
                with lib.temporary_env(cell, dimension=3):
                    coulG_SR = self.weighted_coulG_SR(kpt, False, mesh_d)
                if is_zero(kpt) and coulG_SR_at_G0 is not None:
                    coulG_SR[G0_idx] += coulG_SR_at_G0

                for p0, p1 in lib.prange(0, ngrids_d, Gblksize):
                    Gpq = ft_kern_dd(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt, out=buf)
                    update_vk(vk_dd, Gpq, dm_dd, coulG_SR[p0:p1] * weight,
                              ki_idx, kj_idx, not self_conj, k_to_compute, t_rev_pairs)
                    Gpq = None
                cpu1 = log.timer_debug1(f'ft_aopair dd-block group {group_id}', *cpu1)

            for i in range(n_dm):
                for k in range(nkpts):
                    lib.takebak_2d(vkR[i,k], vkR_dd[i,k], ao_d_idx, ao_d_idx,
                                   thread_safe=False)
                    lib.takebak_2d(vkI[i,k], vkI_dd[i,k], ao_d_idx, ao_d_idx,
                                   thread_safe=False)

        buf = buf1 = None
        if is_zero(kpts) and not np.iscomplexobj(dm_kpts):
            vk_kpts = vkR
        else:
            vk_kpts = vkR + vkI * 1j

        # Add ewald_exxdiv contribution because G=0 was not included in the
        # non-uniform grids
        if (exxdiv == 'ewald' and
            (cell.dimension < 2 or  # 0D and 1D are computed with inf_vacuum
             (cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum'))):
            _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts)

        if time_reversal_symmetry:
            for k, k_conj in t_rev_pairs:
                if k != k_conj:
                    vk_kpts[:,k_conj] = vk_kpts[:,k].conj()
        log.timer_debug1('get_lr_k_kpts', *cpu0)
        return vk_kpts

    to_gpu = lib.to_gpu

RangeSeparationJKBuilder = RangeSeparatedJKBuilder

def _purify(mat_kpts, phase):
    if phase is None:
        return mat_kpts
    #:mat_bvk = np.einsum('Rk,nkij,Sk->nRSij', phase, mat_kpts, phase.conj())
    #:return np.einsum('Rk,nRSij,Sk->nkij', phase.conj(), mat_bvk.real, phase)
    nkpts = phase.shape[1]
    mat_bvk = lib.einsum('k,Sk,nkuv->nSuv', phase[0], phase.conj(), mat_kpts)
    return lib.einsum('S,Sk,nSuv->nkuv', nkpts*phase[:,0].conj(), phase, mat_bvk.real)


def estimate_rcut(rs_cell, omega, precision=None,
                  exclude_dd_block=True):
    '''Estimate rcut for 2e SR-integrals'''
    if precision is None:
        # Adjust precision a little bit as errors are found slightly larger than cell.precision.
        precision = rs_cell.precision * 1e-1

    rs_cell = rs_cell
    exps, cs = pbcgto.cell._extract_pgto_params(rs_cell, 'min')
    ls = rs_cell._bas[:,gto.ANG_OF]

    exp_min_idx = exps.argmin()
    cost = cs * (.5*abs(omega)*rs_cell.rcut)**ls / (2*exps)**(ls/2+.75)
    ai_idx = ak_idx = cost.argmax()
    compact_mask = rs_cell.bas_type != ft_ao.SMOOTH_BASIS
    compact_idx = np.where(compact_mask)[0]
    if exclude_dd_block and compact_idx.size > 0:
        ak_idx = compact_idx[cost[compact_idx].argmax()]
    logger.debug2(rs_cell, 'ai_idx=%d ak_idx=%d', ai_idx, ak_idx)
    # Case 1: l in cell0, product kl ~ dc, product ij ~ dd and dc
    # This includes interactions (dc|dd)
    ak = exps[ak_idx]
    lk = rs_cell._bas[ak_idx,gto.ANG_OF]
    ck = cs[ak_idx]
    aj = exps
    lj = ls
    cj = cs
    ai = exps[ai_idx]
    li = rs_cell._bas[ai_idx,gto.ANG_OF]
    ci = cs[ai_idx]
    al = exps[exp_min_idx]
    ll = rs_cell._bas[exp_min_idx,gto.ANG_OF]
    cl = cs[exp_min_idx]

    aij = ai + aj
    akl = ak + al
    lij = li + lj
    lkl = lk + ll
    l4 = lij + lkl
    norm_ang = ((2*li+1)*(2*lj+1)*(2*lk+1)*(2*ll+1)/(4*np.pi)**4)**.5
    c1 = ci * cj * ck * cl * norm_ang
    theta = omega**2*aij*akl/(aij*akl + (aij+akl)*omega**2)
    sfac = omega**2*aj*al/(aj*al + (aj+al)*omega**2) / theta
    fl = 2
    fac = 2**(li+lk)*np.pi**2.5*c1 * theta**(l4-.5)
    fac *= 2*np.pi/rs_cell.vol/theta
    fac /= aij**(li+1.5) * akl**(lk+1.5) * aj**lj * al**ll
    fac *= fl / precision

    r0 = rs_cell.rcut
    r0 = (np.log(fac * r0 * (sfac*r0)**(l4-1) + 1.) / (sfac*theta))**.5
    r0 = (np.log(fac * r0 * (sfac*r0)**(l4-1) + 1.) / (sfac*theta))**.5
    rcut = r0

    if exclude_dd_block and 0 < compact_idx.size < rs_cell.nbas:
        # Case 2: l in cell0, product kl ~ dc, product ij ~ cd
        # so as to exclude interaction (dc|dd)
        smooth_mask = ~compact_mask
        ai, li, ci = ak, lk, ck
        aj = exps[smooth_mask]
        lj = ls[smooth_mask]
        cj = cs[smooth_mask]
        aij = ai + aj
        lij = li + lj
        l4 = lij + lkl
        norm_ang = ((2*li+1)*(2*lj+1)*(2*lk+1)*(2*ll+1)/(4*np.pi)**4)**.5
        c1 = ci * cj * ck * cl * norm_ang
        theta = omega**2*aij*akl/(aij*akl + (aij+akl)*omega**2)
        sfac = omega**2*aj*al/(aj*al + (aj+al)*omega**2) / theta
        fl = 2
        fac = 2**(li+lk)*np.pi**2.5*c1 * theta**(l4-.5)
        fac *= 2*np.pi/rs_cell.vol/theta
        fac /= aij**(li+1.5) * akl**(lk+1.5) * aj**lj * al**ll
        fac *= fl / precision

        r0 = rcut[smooth_mask]
        r0 = (np.log(fac * r0 * (sfac*r0)**(l4-1) + 1.) / (sfac*theta))**.5
        r0 = (np.log(fac * r0 * (sfac*r0)**(l4-1) + 1.) / (sfac*theta))**.5
        rcut[smooth_mask] = r0
    return rcut

def _guess_omega(cell, kpts, mesh=None):
    a = cell.lattice_vectors()
    if cell.dimension == 0:
        if mesh is None:
            mesh = cell.mesh
        ke_cutoff = pbctools.mesh_to_cutoff(a, mesh).min()
        return 0, mesh, ke_cutoff

    precision = cell.precision
    nkpts = len(kpts)
    if mesh is None:
        omega_min = OMEGA_MIN
        ke_min = estimate_ke_cutoff_for_omega(cell, omega_min)
        nk = (cell.nao/25 * nkpts)**(1./3)
        ke_cutoff = 50 / (.7+.25*nk+.05*nk**3)
        ke_cutoff = max(ke_cutoff, ke_min)
        # avoid large omega since numerical issues were found in Rys
        # polynomials when computing SR integrals with nroots > 3
        exps = [e for l, e in zip(cell._bas[:,gto.ANG_OF], cell.bas_exps()) if l != 0]
        if exps:
            omega_max = np.hstack(exps).min()**.5 * 2
            ke_max = estimate_ke_cutoff_for_omega(cell, omega_max)
            ke_cutoff = min(ke_cutoff, ke_max)
        mesh = cell.cutoff_to_mesh(ke_cutoff)
    else:
        mesh = np.asarray(mesh)
    ke_cutoff = min(pbctools.mesh_to_cutoff(a, mesh)[:cell.dimension])
    omega = estimate_omega_for_ke_cutoff(cell, ke_cutoff, precision)
    return omega, mesh, ke_cutoff

def estimate_ke_cutoff_for_omega(cell, omega, precision=None):
    '''Energy cutoff for FFTDF to converge attenuated Coulomb in moment space
    '''
    if precision is None:
        precision = cell.precision
    ai = np.hstack(cell.bas_exps()).max()
    theta = 1./(1./ai + omega**-2)
    fac = 32*np.pi**2 * theta / precision
    Ecut = 20.
    Ecut = np.log(fac / (2*Ecut) + 1.) * 2*theta
    Ecut = np.log(fac / (2*Ecut) + 1.) * 2*theta
    return Ecut

def estimate_omega_for_ke_cutoff(cell, ke_cutoff, precision=None):
    '''The minimal omega in attenuated Coulomb given energy cutoff
    '''
    if precision is None:
        precision = cell.precision
#    # estimation based on \int dk 4pi/k^2 exp(-k^2/4omega) sometimes is not
#    # enough to converge the 2-electron integrals. A penalty term here is to
#    # reduce the error in integrals
#    precision *= 1e-2
#    kmax = (ke_cutoff*2)**.5
#    log_rest = np.log(precision / (16*np.pi**2 * kmax**lmax))
#    omega = (-.5 * ke_cutoff / log_rest)**.5
#    return omega

    ai = np.hstack(cell.bas_exps()).max()
    aij = ai * 2
    fac = 32*np.pi**2 / precision
    omega = .3
    theta = 1./(1./ai + omega**-2)
    omega2 = 1./(np.log(fac * theta/ (2*ke_cutoff) + 1.)*2/ke_cutoff - 1./aij)
    if omega2 > 0:
        theta = 1./(1./ai + 1./omega2)
        omega2 = 1./(np.log(fac * theta/ (2*ke_cutoff) + 1.)*2/ke_cutoff - 1./aij)
    omega = max(omega2, 0)**.5
    if omega < OMEGA_MIN:
        logger.warn(cell, 'omega=%g smaller than the required minimal value %g. '
                    'Set omega to %g', omega2, OMEGA_MIN, OMEGA_MIN)
        omega = OMEGA_MIN
    return omega

def _qcond_cell0_abstract(qcond, seg_loc, seg2sh, nbasp):
    '''Find the max qcond for ij pair'''
    # The first shell in each seg_loc[i]:seg_loc[i+1] is inside cell0
    cell0_prim_idx = seg2sh[np.arange(seg_loc[nbasp])]
    qcond_sub = qcond[:,cell0_prim_idx]
    sh_loc = seg2sh[seg_loc]
    nbas_bvk = sh_loc.size - 1
    qtmp = np.empty((nbas_bvk, cell0_prim_idx.size), dtype=qcond.dtype)
    for i, (i0, i1) in enumerate(zip(sh_loc[:-1], sh_loc[1:])):
        if i0 != i1:
            qtmp[i] = qcond_sub[i0:i1].max(axis=0)
        else:
            qtmp[i] = INDEX_MIN
    qcond_cell0 = np.empty((nbas_bvk, nbasp), dtype=qcond.dtype)
    for j, (j0, j1) in enumerate(zip(seg_loc[:nbasp], seg_loc[1:nbasp+1])):
        if j0 != j1:
            qcond_cell0[:,j] = qtmp[:,j0:j1].max(axis=1)
        else:
            qcond_cell0[:,j] = INDEX_MIN
    return qcond_cell0
