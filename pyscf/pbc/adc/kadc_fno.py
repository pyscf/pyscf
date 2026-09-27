# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
# Author: Ning-Yuan Chen <cny003@outlook.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

import time
import numpy as np
from pyscf import ao2mo
import pyscf.adc
import pyscf.adc.radc
from pyscf.adc import radc_ao2mo
import itertools

from itertools import product
from pyscf import lib
from pyscf.pbc import scf
from pyscf.pbc import df
from pyscf.pbc import mp
from pyscf.lib import logger
from pyscf.pbc.adc import kadc_rhf
from pyscf.pbc.adc import kadc_ao2mo
from pyscf.pbc.adc import dfadc
from pyscf import __config__
from pyscf.pbc.mp.kmp2 import (get_nocc, get_nmo, padding_k_idx, _padding_k_idx,
                               padded_mo_coeff, get_frozen_mask, _add_padding)
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.pbc.cc.kccsd_t_rhf import _get_epqr
from pyscf.pbc.lib import kpts_helper
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM  # noqa
from pyscf.data.nist import HARTREE2EV

from pyscf.pbc import tools
import h5py
import tempfile


def _count_thresholds(thresh, pct_occ, nvir_act):
    """Return the number of thresholds implied by the parameters."""
    n = 1
    for val in (thresh, pct_occ, nvir_act):
        if val is not None and isinstance(val, (list, tuple, np.ndarray)):
            n = max(n, len(val))
    return n


def _pick(param, i):
    """Extract the i-th element if *param* is a sequence, else return as-is."""
    if param is not None and isinstance(param, (list, tuple, np.ndarray)):
        return param[i]
    return param


class RADC2FNO(kadc_rhf.RADC):
    # J. Chem. Phys. 159, 084113 (2023)
    _keys = kadc_rhf.RADC._keys | {'delta_e', 'e_can', 'v_can', 'e_corr_can',
                                   'rdm1_ss', 'trans_guess', 'mode', 'ref_state',
                                   'if_adc2_guess', 'delta_e_corr', 'p_can',
                                   'if_ref_qp', 'delta_e_qp', 'is_qp',
                                   'e_corr_fno'}

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        super().__init__(mf, frozen, mo_coeff, mo_occ)
        self.delta_e = None
        self.delta_e_corr = None
        self.delta_e_qp = None
        self.is_qp = 0.5
        self.e_can = None
        self.v_can = None
        self.p_can = None
        self.e_corr_can = None
        self.rdm1_ss = None
        self.trans_guess = False
        self.mode = "min"
        self.ref_state = None
        self.if_ref_qp = True
        self.if_adc2_guess = False
        self.e_corr_fno = None

    def _reset_adc_state(self):
        """Clear cached amplitudes / intermediates so a fresh FNO-ADC run can start."""
        self.t1 = None
        self.t2 = None
        self._adc_es = None
        if hasattr(self.imds, 't2_1_vvvv'):
            self.imds.t2_1_vvvv = None

    def kernel_gs(self, eris=None, thresh=1e-4, pct_occ=None, nvir_act=None):
        """Ground-state FNO driver: generate the FNO virtual space from the MP2 density
        and the additive correlation-energy correction."""
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        logger.info(self, "generate fno with correction for the ground state")
        self.ref_state = None

        self.make_ss_rdm1(log, cput0, if_gs=True)
        log.timer('make gs rdm1', *cput0)

        # Snapshot canonical orbital layout needed by make_fno
        _can_frozen = self.frozen
        _can_mo_coeff = self.mo_coeff
        _can_mo_energy = self.mo_energy
        _can_mo_occ = [np.copy(oc) for oc in self.mo_occ]

        n_thresh = _count_thresholds(thresh, pct_occ, nvir_act)

        all_mo_coeff, all_mo_energy, all_frozen = [], [], []
        all_delta_e_corr, all_e_corr_fno = [], []

        for i in range(n_thresh):
            # Restore canonical orbitals so make_fno always starts from the full space
            self.frozen = _can_frozen
            self.mo_coeff = _can_mo_coeff
            self.mo_energy = _can_mo_energy

            t = _pick(thresh, i)
            p = _pick(pct_occ, i)
            v = _pick(nvir_act, i)

            self.make_fno(self.rdm1_ss, self._scf, log, thresh=t, pct_occ=p, nvir_act=v)

            # Snapshot FNO orbitals before solver state is overwritten
            fno_mo_coeff = list(self.mo_coeff)
            fno_mo_energy = list(self.mo_energy)
            fno_frozen = list(self.frozen)

            self._reset_adc_state()
            self.compute_correction(if_gs=True)

            all_mo_coeff.append(fno_mo_coeff)
            all_mo_energy.append(fno_mo_energy)
            all_frozen.append(fno_frozen)
            all_delta_e_corr.append(self.delta_e_corr)
            all_e_corr_fno.append(self.e_corr)

            log.timer('get frozen info (%d/%d)' % (i + 1, n_thresh), *cput0)

        log.timer('gs FNO', *cput0)

        # Expose results — single threshold keeps scalar attrs (backward compatible)
        if n_thresh == 1:
            self.mo_coeff = all_mo_coeff[0]
            self.mo_energy = all_mo_energy[0]
            self.frozen = all_frozen[0]
            self.delta_e_corr = all_delta_e_corr[0]
            self.e_corr_fno = all_e_corr_fno[0]
        else:
            self.mo_coeff = all_mo_coeff
            self.mo_energy = all_mo_energy
            self.frozen = all_frozen
            self.mo_occ = [[np.copy(oc) for oc in _can_mo_occ]
                           for _ in range(n_thresh)]
            self.delta_e_corr = all_delta_e_corr
            self.e_corr_fno = all_e_corr_fno

    def kernel(self, nroots=1, guess=None, eris=None, thresh=1e-4, pct_occ=None, nvir_act=None, kptlist=None):
        """Excited-state FNO driver: generate the FNO (SS/SA-FNO when ref_state is set) virtual space
        and the excitation-energy corrections."""
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        if self.ref_state is None:
            logger.info(self, "generate fno with correction for the excited state")
        elif (isinstance(self.ref_state, int) and 0 < self.ref_state <= nroots) or \
                (hasattr(self.ref_state, '__len__') and len(self.ref_state) == 2):
            logger.info(self, "generate ss-fno with correction for the excited state")
        else:
            raise ValueError("ref_state should be an int type or or a array-like object with two elements")

        if kptlist is None:
            kptlist = range(self.nkpts)

        self.make_ss_rdm1(log, cput0, kptlist, nroots, guess)
        log.timer('make ss rdm1', *cput0)

        # Snapshot canonical orbital layout
        _can_frozen = self.frozen
        _can_mo_coeff = self.mo_coeff
        _can_mo_energy = self.mo_energy
        _can_mo_occ = [np.copy(oc) for oc in self.mo_occ]

        n_thresh = _count_thresholds(thresh, pct_occ, nvir_act)

        all_mo_coeff, all_mo_energy, all_frozen = [], [], []
        all_delta_e_corr, all_e_corr_fno = [], []
        all_delta_e, all_delta_e_qp = [], []
        all_e_ssfno, all_v_ssfno, all_p_ssfno = [], [], []

        for i in range(n_thresh):
            self.frozen = _can_frozen
            self.mo_coeff = _can_mo_coeff
            self.mo_energy = _can_mo_energy

            t = _pick(thresh, i)
            p = _pick(pct_occ, i)
            v = _pick(nvir_act, i)

            self.make_fno(self.rdm1_ss, self._scf, log, thresh=t, pct_occ=p, nvir_act=v)

            fno_mo_coeff = list(self.mo_coeff)
            fno_mo_energy = list(self.mo_energy)
            fno_frozen = list(self.frozen)

            self._reset_adc_state()
            self.compute_correction(kptlist, nroots, guess)

            all_mo_coeff.append(fno_mo_coeff)
            all_mo_energy.append(fno_mo_energy)
            all_frozen.append(fno_frozen)
            all_delta_e_corr.append(self.delta_e_corr)
            all_e_corr_fno.append(self.e_corr)
            all_delta_e.append(self.delta_e)
            all_delta_e_qp.append(self.delta_e_qp)
            all_e_ssfno.append(self.e_ssfno)
            all_v_ssfno.append(self.v_ssfno)
            all_p_ssfno.append(self.p_ssfno)

            log.timer('es FNO (%d/%d)' % (i + 1, n_thresh), *cput0)

        log.timer('es FNO', *cput0)

        if n_thresh == 1:
            self.mo_coeff = all_mo_coeff[0]
            self.mo_energy = all_mo_energy[0]
            self.frozen = all_frozen[0]
            self.delta_e_corr = all_delta_e_corr[0]
            self.e_corr_fno = all_e_corr_fno[0]
            self.delta_e = all_delta_e[0]
            self.delta_e_qp = all_delta_e_qp[0]
        else:
            self.mo_coeff = all_mo_coeff
            self.mo_energy = all_mo_energy
            self.frozen = all_frozen
            self.mo_occ = [[np.copy(oc) for oc in _can_mo_occ]
                           for _ in range(n_thresh)]
            self.delta_e_corr = all_delta_e_corr
            self.e_corr_fno = all_e_corr_fno
            self.delta_e = all_delta_e
            self.delta_e_qp = all_delta_e_qp
            self.e_ssfno = all_e_ssfno
            self.v_ssfno = all_v_ssfno
            self.p_ssfno = all_p_ssfno

    def compute_correction(self, kptlist=None, nroots=None, guess=None, if_gs=False):
        """Compute the additive FNO corrections by running the reference MP2/ADC(2) in the FNO space."""
        if if_gs:
            _, _, _ = kadc_rhf.RADC.kernel_gs(self)
        else:
            self.e_ssfno, self.v_ssfno, self.p_ssfno, _ = kadc_rhf.RADC.kernel(
                self, nroots, guess=guess, kptlist=kptlist)
            self.delta_e = self.e_can - self.e_ssfno
            self.delta_e_qp = []
            mask_fno = self.p_ssfno > self.is_qp
            mask_can = self.p_can > self.is_qp
            for kpt in kptlist:
                e_can_qp_k = self.e_can[kpt][mask_can[kpt]]
                e_ssfno_qp_k = self.e_ssfno[kpt][mask_fno[kpt]]
                n_qp = min(len(e_can_qp_k), len(e_ssfno_qp_k))
                self.delta_e_qp.append(e_can_qp_k[:n_qp] - e_ssfno_qp_k[:n_qp])
        self.delta_e_corr = self.e_corr_can - self.e_corr

    def correct(self, e, i=None):
        """Additively-corrected excitation energies e + delta_e (use i for multi-threshold)."""
        return e + (self.delta_e[i] if i is not None else self.delta_e)

    def correct_corr(self, e, i=None):
        """Additively-corrected correlation energy e + delta_e_corr (use i for multi-threshold)."""
        return e + (self.delta_e_corr[i] if i is not None else self.delta_e_corr)

    def make_ss_rdm1(self, log, cput0, kptlist=None, nroots=None, guess=None, if_gs=False):
        """Run the canonical reference and build the 1-RDM used to construct the FNOs."""
        if if_gs:
            _, _, _ = kadc_rhf.RADC.kernel_gs(self)
        else:
            self.e_can, self.v_can, self.p_can, _ = kadc_rhf.RADC.kernel(self, nroots, guess=guess, kptlist=kptlist)
        log.info('current use %d MB', lib.current_memory()[0])
        self.e_corr_can = self.e_corr
        if self.ref_state is not None:
            if isinstance(self.ref_state, (int, np.integer)):
                idx = np.argsort(self.e_can.ravel()).tolist()
                sidx = [[idx[self.ref_state - 1] % self.nkpts]]
                kidx = [idx[self.ref_state - 1] // self.nkpts]
            elif hasattr(self.ref_state, '__len__'):
                if len(self.ref_state) != 2:
                    raise ValueError
                if not isinstance(self.ref_state[0], list) or not isinstance(self.ref_state[1], list):
                    raise ValueError("when ref_state is a array-like object, both elements should be list type")
                if not isinstance(self.ref_state[1][0], (int, np.integer)):
                    raise ValueError("elements in the second list of ref_state should be int type")
                if isinstance(self.ref_state[0][0], (int, np.integer)):
                    sidx = [self.ref_state[0] for _ in range(len(self.ref_state[1]))]
                else:
                    if len(self.ref_state[0]) != len(self.ref_state[1]):
                        raise ValueError("when the first element of ref_state is a "
                                         "array-like object, its length should be "
                                         "the same as the second element")
                    sidx = self.ref_state[0]
                kidx = self.ref_state[1]
                if self.if_ref_qp:
                    state_list = []
                    mask_can = self.p_can > self.is_qp
                    for kpt, kshift in enumerate(kidx):
                        k = kptlist.index(kshift)
                        state_list_k = np.arange(nroots)
                        state_list_k = state_list_k[mask_can[k]].tolist()
                        state_list.append([state_list_k[s] for s in sidx[kpt]])
                    sidx = state_list

            log.info(f"the specific state is {sidx} with kidx {kidx}")
            es_DM = self.make_rdm1(kptlist, root=sidx, K_idx=kidx, if_ss=True)
            self.rdm1_ss = np.zeros_like(es_DM[0][0])
            n_state = sum([len(s_k) for s_k in sidx])
            for k in range(len(kidx)):
                for i in range(len(sidx[k])):
                    self.rdm1_ss += es_DM[k][i] / n_state
            log.info('current use %d MB', lib.current_memory()[0])
            log.timer('make ss rdm1', *cput0)
        else:
            self.rdm1_ss = self.make_ref_rdm1()
            log.info('current use %d MB', lib.current_memory()[0])
            log.timer('make ref rdm1', *cput0)

        def incore_transform():
            return kadc_ao2mo.transform_integrals_incore(self)
        self.transform_integrals = incore_transform
        self.t1 = None
        self.t2 = None
        self._adc_es = None
        self.imds.t2_1_vvvv = None

    def make_fno(self, rdm1_ss, mf, log, thresh=None, pct_occ=None, nvir_act=None):
        """Build the FNO virtual space: diagonalize the virtual 1-RDM, truncate, and semicanonicalize."""
        nocc = mf.mol.nelectron // 2
        masks = kadc_rhf.mo_splitter(self)
        no_coeff = []
        no_frozen = []
        no_energy = []
        V_trunc = []
        padding_convention = padding_k_idx(self, kind="joint")

        T = []
        for kpt in range(self.nkpts):
            rdm1_ss_comp = rdm1_ss[kpt][np.ix_(padding_convention[kpt], padding_convention[kpt])]
            n, V_k = np.linalg.eigh(rdm1_ss_comp[nocc:, nocc:])
            idx = np.argsort(n)[::-1]
            n, V_k = n[idx], V_k[:, idx]
            V_trunc.append(V_k)
            if nvir_act is not None:
                T.append(np.arange(len(n)) < nvir_act)
            elif pct_occ is not None:
                cumsum = np.cumsum(n / np.sum(n))
                T.append(np.array([c <= pct_occ or np.isclose(c, pct_occ) for c in cumsum]))
            else:
                T.append(n > thresh)

        # "min": union the per-kpt masks so every k-point keeps the same count
        if self.mode.lower() == "min":
            T_min = np.logical_or.reduce(np.stack(T), axis=0)
            n_keep = int(np.sum(T_min))
            if n_keep == 0:
                log.warn("All virtual orbitals were requested to be frozen.\n"
                         "At least one virtual orbital must be retained for ADC calculations.\n"
                         "Keeping one virtual orbital automatically.")
                n_keep += 1
                T_min[0] = True

        for kpt in range(self.nkpts):
            V_trunc_k = V_trunc[kpt]
            if self.mode.lower() != "min":
                n_keep = int(np.sum(T[kpt]))
                if n_keep == 0:
                    log.warn("All virtual orbitals frozen at kpt %d; keeping one." % kpt)
                    n_keep += 1
                    T[kpt][0] = True

            moeoccfrz0, moeocc, moevir, moevirfrz0 = [mf.mo_energy[kpt][m] for m in masks[kpt]]
            orboccfrz0, orbocc, orbvir, orbvirfrz0 = [mf.mo_coeff[kpt][:, m] for m in masks[kpt]]
            F_can = np.diag(moevir)
            F_trunc = V_trunc_k.T.conj().dot(F_can).dot(V_trunc_k)
            e_trunc, Z_trunc = np.linalg.eigh(F_trunc[:n_keep, :n_keep])
            e_fro = np.diagonal(F_trunc[n_keep:, n_keep:]).copy()
            U_vir_act = orbvir.dot(V_trunc_k[:, :n_keep]).dot(Z_trunc)
            U_vir_fro = orbvir.dot(V_trunc_k[:, n_keep:])
            no_comp = (orboccfrz0, orbocc, U_vir_act, U_vir_fro, orbvirfrz0)
            no_e_comp = (moeoccfrz0, moeocc, e_trunc, e_fro, moevirfrz0)
            no_coeff_k = np.hstack(no_comp)
            no_energy_k = np.hstack(no_e_comp)
            nocc_loc = np.cumsum([0] + [x.shape[1] for x in no_comp]).astype(int)
            no_frozen_k = np.hstack((np.arange(nocc_loc[0], nocc_loc[1]),
                                    np.arange(nocc_loc[3], nocc_loc[5]))).astype(int)
            no_coeff.append(no_coeff_k)
            no_energy.append(no_energy_k)
            no_frozen.append(no_frozen_k)

        self.mo_coeff, self.mo_energy, self.frozen = no_coeff, no_energy, no_frozen
