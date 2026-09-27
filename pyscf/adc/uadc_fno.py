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

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.adc import uadc_ao2mo
from pyscf.adc import uadc_amplitudes
from pyscf.adc import uadc
from pyscf import __config__
from pyscf import df
from pyscf import scf
from pyscf.data.nist import HARTREE2EV

class UADC2FNO(uadc.UADC):
    #J. Chem. Phys. 159, 084113 (2023)
    _keys = uadc.UADC._keys | {'delta_e', 'delta_e_corr', 'e_can', 'v_can', 'e_corr_can',
                               'mo_energy', 'rdm1_ss', 'ref_state', 'trans_guess',
                               'p_can', 'p_ssfno', 'delta_e_qp', 'is_qp', 'if_osfno'}

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, mo_energy=None, f_ov=None):
        super().__init__(mf, frozen, mo_coeff, mo_occ, mo_energy, f_ov)
        self.delta_e = None
        self.delta_e_corr = None
        self.delta_e_qp = None
        self.is_qp = 0.5
        self.e_can = None
        self.v_can = None
        self.p_can = None
        self.p_ssfno = None
        self.e_corr_can = None
        self.rdm1_ss = None
        self.ref_state = None
        self.if_naf = False
        self.trans_guess = False
        self.if_osfno = False

    def kernel_gs(self, eris=None, thresh = 1e-4, pct_occ=None, nvir_act=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        logger.info(self, "generate fno with correction for the ground state")
        self.ref_state = None

        if not getattr(self, 'with_df', None) and not getattr(self._scf, 'with_df', None):
            self.if_naf = False

        self.make_ss_rdm1(if_gs=True)
        log.timer('make gs rdm1', *cput0)
        self.make_fno(self.rdm1_ss, thresh, pct_occ, nvir_act)
        log.timer('get frozen info', *cput0)
        self.compute_correction(self._scf, eris=eris, if_gs=True)

        log.timer('gs FNO', *cput0)

    def kernel(self, nroots=1, guess=None, eris=None, thresh = 1e-4, pct_occ=None, nvir_act=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        if self.ref_state is None or self.ref_state == 0:
            logger.info(self,"Do fno adc calculation")
        elif isinstance(self.ref_state, int) and 0<self.ref_state<=nroots:
            logger.info(self,f"Do ss-fno adc calculation, the specic state is {self.ref_state}")
        else:
            raise ValueError("ref_state should be an int type and in [0,nroots]")

        if not getattr(self, 'with_df', None) and not getattr(self._scf, 'with_df', None):
            self.if_naf = False

        self.make_ss_rdm1(nroots, guess)
        log.timer('make ss rdm1', *cput0)
        self.make_fno(self.rdm1_ss, thresh, pct_occ, nvir_act)
        log.timer('get frozen info', *cput0)

        if self.trans_guess and self.method_type == 'ip' and self.ncvs == 0:
            self.compute_correction(self._scf, nroots, eris, guess=self.v_can)
        else:
            self.compute_correction(self._scf, nroots, eris, guess=guess)

        log.timer('es FNO', *cput0)

    def compute_correction(self, mf, nroots=None, eris=None, guess=None, if_gs=False):
        adc_ssfno = uadc.UADC(mf, self.frozen, self.mo_coeff, mo_energy = self.mo_energy, f_ov = self.f_ov).set(
                                                        verbose = self.verbose,
                                                        method_type = self.method_type,method=self.method,
                                                        with_df = self.with_df,if_naf = self.if_naf,
                                                        thresh_naf = self.thresh_naf,naux = self.naux,
                                                        if_heri_eris = self.if_heri_eris,ncvs = self.ncvs,
                                                        approx_trans_moments = self.approx_trans_moments,
                                                        conv_tol = self.conv_tol,tol_residual = self.tol_residual,
                                                        max_space = self.max_space, max_cycle = self.max_cycle)
        if if_gs:
            _,_,_ = adc_ssfno.kernel_gs(eris)
        else:
            self.e_ssfno,self.v_ssfno,self.p_ssfno,_ = adc_ssfno.kernel(nroots,guess,eris)
            self.delta_e = self.e_can - self.e_ssfno
            mask_fno = self.p_ssfno > self.is_qp
            mask_can = self.p_can > self.is_qp
            e_can_qp = self.e_can[mask_can]
            e_ssfno_qp = self.e_ssfno[mask_fno]
            n_qp = min(len(e_can_qp), len(e_ssfno_qp))
            self.delta_e_qp = e_can_qp[:n_qp] - e_ssfno_qp[:n_qp]
        self.naux = adc_ssfno.naux
        self.eris = adc_ssfno.eris
        self.delta_e_corr = self.e_corr_can - adc_ssfno.e_corr

    def correct(self, e):
        """Additively-corrected excitation energies e + delta_e."""
        return e + self.delta_e

    def correct_corr(self, e):
        """Additively-corrected correlation energy e + delta_e_corr."""
        return e + self.delta_e_corr

    def make_ss_rdm1(self,nroots=None,guess=None,if_gs=False):
        heri_tmp = self.if_heri_eris
        self.if_heri_eris = False
        if if_gs:
            _,_,_ = uadc.UADC.kernel_gs(self)
        else:
            self.e_can,self.v_can,self.p_can,_ = uadc.UADC.kernel(self,nroots,guess)
        self.if_heri_eris = heri_tmp
        rdm1_gs = self.make_ref_rdm1(ao_repr=self.if_osfno)
        self.e_corr_can = self.e_corr
        if self.ref_state is not None and self.ref_state > 0:
            rdm1_gs_a = rdm1_gs[0]
            rdm1_gs_b = rdm1_gs[1]
            rdm1_es = self.make_rdm1(ao_repr=self.if_osfno)
            rdm1_es_a = rdm1_es[0][self.ref_state - 1]
            rdm1_es_b = rdm1_es[1][self.ref_state - 1]
            self.rdm1_ss = (rdm1_es_a + rdm1_gs_a, rdm1_es_b + rdm1_gs_b)
        else:
            self.rdm1_ss = rdm1_gs

    def make_fno(self, rdm1_ss, thresh, pct_occ, nvir_act):
        nocc_a = self._scf.nelec[0]
        nocc_b = self._scf.nelec[1]
        mo_energy_a = self.mo_energy_hf[0]
        mo_energy_b = self.mo_energy_hf[1]
        mo_a_coeff = self.mo_coeff_hf[0]
        mo_b_coeff = self.mo_coeff_hf[1]
        masks = self._mo_splitter()
        mask_a = masks[0]
        mask_b = masks[1]
        rdm1_ss_a = rdm1_ss[0]
        rdm1_ss_b = rdm1_ss[1]
        moeoccfrz0_a, moeocc_a, moevir_a, moevirfrz0_a = [mo_energy_a[m] for m in mask_a]
        orboccfrz0_a, orbocc_a, orbvir_a, orbvirfrz0_a = [mo_a_coeff[:,m] for m in mask_a]
        moeoccfrz0_b, moeocc_b, moevir_b, moevirfrz0_b = [mo_energy_b[m] for m in mask_b]
        orboccfrz0_b, orbocc_b, orbvir_b, orbvirfrz0_b = [mo_b_coeff[:,m] for m in mask_b]

        if self.if_osfno:
            # Open-shell FNO, J. Chem. Phys. 152, 034105 (2020)
            ovlp_ao = self._scf.get_ovlp()

            if nocc_a >= nocc_b:
                ovlp_oV = orbocc_a.T.dot(ovlp_ao).dot(orbvir_b)
                _, s_sv, Vh_sv = np.linalg.svd(ovlp_oV)
                B_a = np.eye(orbvir_a.shape[1])
                B_b = Vh_sv.T
            else:
                ovlp_Ov = orbocc_b.T.dot(ovlp_ao).dot(orbvir_a)
                _, s_sv, Vh_sv = np.linalg.svd(ovlp_Ov)
                B_a = Vh_sv.T
                B_b = np.eye(orbvir_b.shape[1])

            k_os = int(np.count_nonzero(s_sv > 0.8))
            logger.debug(self, "OSFNO overlap singular values: %s", s_sv)
            logger.info(self, "OSFNO: %d open-shell partner virtuals detected "
                        "(nocc_a - nocc_b = %d)", k_os, nocc_a - nocc_b)
            if k_os != abs(nocc_a - nocc_b):
                logger.warn(self, "OSFNO: number of near-unity overlap singular "
                            "values (%d) differs from nocc_a - nocc_b (%d)",
                            k_os, nocc_a - nocc_b)

            # singlet part of the state density in the AO basis
            rdm1_s_ao = 0.5 * (rdm1_ss_a + rdm1_ss_b)

            if nocc_a >= nocc_b:
                tilde_vir_a = orbvir_a
                tilde_vir_b = orbvir_b.dot(B_b[:, k_os:])
            else:
                tilde_vir_a = orbvir_a.dot(B_a[:, k_os:])
                tilde_vir_b = orbvir_b

            rdm1_vV = tilde_vir_a.T.dot(ovlp_ao).dot(rdm1_s_ao).dot(ovlp_ao).dot(tilde_vir_b)
            V_a, n, Vh_b = np.linalg.svd(rdm1_vV)
            V_b = Vh_b.T

            if nvir_act is None:
                if pct_occ is None:
                    T = n > thresh
                else:
                    cumsum = np.cumsum(n/np.sum(n))
                    T = np.array([c <= pct_occ or np.isclose(c, pct_occ) for c in cumsum])
            else:
                T = np.array([i < nvir_act for i in range(len(n))])
            if not T.any():
                logger.warn(self, "All virtual NO pairs were requested to be frozen.\n"
                            "At least one pair must be retained for ADC calculations.\n"
                            "Keeping one pair automatically.")
                T[0] = True

            idx = np.argsort(~T, kind='stable')
            n_keep = int(np.sum(T))

            if nocc_a >= nocc_b:
                V_trunc_a = V_a[:, idx]
                n_keep_a = n_keep
                V_trunc_b = np.hstack((B_b[:, :k_os], B_b[:, k_os:].dot(V_b[:, idx])))
                n_keep_b = k_os + n_keep
            else:
                V_trunc_b = V_b[:, idx]
                n_keep_b = n_keep
                V_trunc_a = np.hstack((B_a[:, :k_os], B_a[:, k_os:].dot(V_a[:, idx])))
                n_keep_a = k_os + n_keep

        else:
            n_a,V_a = np.linalg.eigh(rdm1_ss_a[nocc_a:,nocc_a:])
            idx = np.argsort(n_a)[::-1]
            n_a,V_trunc_a = n_a[idx], V_a[:,idx]

            n_b,V_b = np.linalg.eigh(rdm1_ss_b[nocc_b:,nocc_b:])
            idx = np.argsort(n_b)[::-1]
            n_b,V_trunc_b = n_b[idx], V_b[:,idx]

            if nvir_act is None:
                if pct_occ is None:
                    T_a = n_a > thresh
                    T_b = n_b > thresh
                else:
                    cumsum_a = np.cumsum(n_a/np.sum(n_a))
                    cumsum_b = np.cumsum(n_b/np.sum(n_b))
                    T_a = np.array([c <= pct_occ or np.isclose(c, pct_occ) for c in cumsum_a])
                    T_b = np.array([c <= pct_occ or np.isclose(c, pct_occ) for c in cumsum_b])
            else:
                T_a = np.array([i < nvir_act for i in range(len(n_a))])
                T_b = np.array([i < nvir_act for i in range(len(n_b))])

            n_keep_a = int(np.sum(T_a))
            n_keep_b = int(np.sum(T_b))

        F_can_a = np.diag(moevir_a)
        F_can_b = np.diag(moevir_b)
        F_trunc_a = V_trunc_a.T.dot(F_can_a).dot(V_trunc_a)
        F_trunc_b = V_trunc_b.T.dot(F_can_b).dot(V_trunc_b)
        e_trunc_a, Z_trunc_a = np.linalg.eigh(F_trunc_a[:n_keep_a, :n_keep_a])
        e_trunc_b, Z_trunc_b = np.linalg.eigh(F_trunc_b[:n_keep_b, :n_keep_b])
        e_fro_a = np.diagonal(F_trunc_a[n_keep_a:, n_keep_a:]).copy()
        e_fro_b = np.diagonal(F_trunc_b[n_keep_b:, n_keep_b:]).copy()

        U_vir_act_a = orbvir_a.dot(V_trunc_a[:, :n_keep_a]).dot(Z_trunc_a)
        U_vir_act_b = orbvir_b.dot(V_trunc_b[:, :n_keep_b]).dot(Z_trunc_b)
        U_vir_fro_a = orbvir_a.dot(V_trunc_a[:, n_keep_a:])
        U_vir_fro_b = orbvir_b.dot(V_trunc_b[:, n_keep_b:])

        no_comp_a = (orboccfrz0_a,orbocc_a,U_vir_act_a,U_vir_fro_a,orbvirfrz0_a)
        no_e_comp_a = (moeoccfrz0_a, moeocc_a, e_trunc_a, e_fro_a, moevirfrz0_a)
        no_coeff_a = np.hstack(no_comp_a)
        no_energy_a = np.hstack(no_e_comp_a)
        nocc_loc_a = np.cumsum([0]+[x.shape[1] for x in no_comp_a]).astype(int)
        no_frozen_a = np.hstack((np.arange(nocc_loc_a[0], nocc_loc_a[1]),
                                np.arange(nocc_loc_a[3], nocc_loc_a[5]))).astype(int)

        no_comp_b = (orboccfrz0_b,orbocc_b,U_vir_act_b,U_vir_fro_b,orbvirfrz0_b)
        no_e_comp_b = (moeoccfrz0_b, moeocc_b, e_trunc_b, e_fro_b, moevirfrz0_b)
        no_coeff_b = np.hstack(no_comp_b)
        no_energy_b = np.hstack(no_e_comp_b)
        nocc_loc_b = np.cumsum([0]+[x.shape[1] for x in no_comp_b]).astype(int)
        no_frozen_b = np.hstack((np.arange(nocc_loc_b[0], nocc_loc_b[1]),
                                np.arange(nocc_loc_b[3], nocc_loc_b[5]))).astype(int)

        no_coeff = (no_coeff_a,no_coeff_b)
        no_energy = (no_energy_a,no_energy_b)
        no_frozen = (no_frozen_a,no_frozen_b)

        if isinstance(self._scf, scf.rohf.ROHF):
            f_ov_a, f_ov_b = self.f_ov
            f_ov_a = f_ov_a.dot(V_trunc_a[:,:n_keep_a]).dot(Z_trunc_a)
            f_ov_b = f_ov_b.dot(V_trunc_b[:,:n_keep_b]).dot(Z_trunc_b)
            f_ov = (f_ov_a,f_ov_b)
            self.mo_coeff,self.mo_energy,self.frozen,self.f_ov = no_coeff,no_energy,no_frozen,f_ov
        else:
            self.mo_coeff,self.mo_energy,self.frozen = no_coeff,no_energy,no_frozen
