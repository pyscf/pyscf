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
from pyscf.lib import logger
from pyscf.adc import radc
from pyscf import __config__

class RADC2FNO(radc.RADC):
    #J. Chem. Phys. 159, 084113 (2023)
    _keys = radc.RADC._keys | {'delta_e', 'delta_e_corr', 'e_can', 'v_can', 'e_corr_can',
                               'mo_energy', 'rdm1_ss', 'ref_state', 'trans_guess',
                               'p_can', 'p_ssfno', 'delta_e_qp', 'is_qp'}

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, mo_energy=None):
        super().__init__(mf, frozen, mo_coeff, mo_occ, mo_energy)
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

    def kernel_gs(self, eris=None, thresh = 1e-4, pct_occ=None, nvir_act=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        logger.info(self, "generate fno with correction for the ground state")
        self.ref_state = None

        if not getattr(self, 'with_df', None) and not getattr(self._scf, 'with_df', None):
            self.if_naf = False

        self.make_ss_rdm1(log, cput0, if_gs=True)
        log.timer('make gs rdm1', *cput0)
        self.make_fno(self.rdm1_ss, self._scf, thresh, pct_occ, nvir_act)
        log.timer('get frozen info', *cput0)
        self.compute_correction(self._scf, self.frozen, eris=eris, if_gs=True)

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
        self.make_fno(self.rdm1_ss, self._scf, thresh, pct_occ, nvir_act)
        log.timer('get frozen info', *cput0)

        if self.trans_guess and self.method_type == 'ip' and self.ncvs == 0:
            self.compute_correction(self._scf, nroots, eris, guess=self.v_can)
        else:
            self.compute_correction(self._scf, nroots, eris, guess)

        log.timer('es FNO', *cput0)

    def compute_correction(self, mf, nroots=None, eris=None, guess=None, if_gs=False):
        adc_ssfno = radc.RADC(mf, self.frozen, self.mo_coeff, mo_energy = self.mo_energy).set(verbose = self.verbose,
                                                        method = self.method,method_type = self.method_type,
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

    def make_ss_rdm1(self,nroots,guess,if_gs=False):
        heri_tmp = self.if_heri_eris
        self.if_heri_eris = False
        if if_gs:
            _,_,_ = radc.RADC.kernel_gs(self)
        else:
            self.e_can,self.v_can,self.p_can,_ = radc.RADC.kernel(self,nroots,guess)
        self.if_heri_eris = heri_tmp
        rdm1_gs = self.make_ref_rdm1()
        self.e_corr_can = self.e_corr
        if self.ref_state is not None and self.ref_state > 0:
            rdm1_es = self.make_rdm1()[self.ref_state - 1]
            self.rdm1_ss = rdm1_es + rdm1_gs
        else:
            self.rdm1_ss = rdm1_gs

    def make_fno(self, rdm1_ss, mf, thresh, pct_occ, nvir_act):
        from pyscf.mp import mp2
        nocc = mf.mol.nelectron//2
        nmo = self._nmo
        self._nmo = None
        masks = mp2._mo_splitter(self)
        self._nmo = nmo

        n,V = np.linalg.eigh(rdm1_ss[nocc:,nocc:])
        idx = np.argsort(n)[::-1]
        n,V_trunc = n[idx], V[:,idx]
        if nvir_act is None:
            if pct_occ is None:
                T = n > thresh
            else:
                cumsum = np.cumsum(n/np.sum(n))
                T = np.array([c <= pct_occ or np.isclose(c, pct_occ) for c in cumsum])
        else:
            T = np.array([i < nvir_act for i in range(len(n))])

        n_keep = int(np.sum(T))

        moeoccfrz0, moeocc, moevir, moevirfrz0 = [mf.mo_energy[m] for m in masks]
        orboccfrz0, orbocc, orbvir, orbvirfrz0 = [mf.mo_coeff[:,m] for m in masks]
        F_can =  np.diag(moevir)
        F_trunc = V_trunc.T.dot(F_can).dot(V_trunc)
        e_trunc,Z_trunc = np.linalg.eigh(F_trunc[:n_keep,:n_keep])
        e_fro = np.diagonal(F_trunc[n_keep:, n_keep:]).copy()

        U_vir_act = orbvir.dot(V_trunc[:,:n_keep]).dot(Z_trunc)
        U_vir_fro = orbvir.dot(V_trunc[:,n_keep:])

        no_comp = (orboccfrz0,orbocc,U_vir_act,U_vir_fro,orbvirfrz0)
        no_e_comp = (moeoccfrz0,moeocc,e_trunc,e_fro,moevirfrz0)
        no_coeff = np.hstack(no_comp)
        no_energy = np.hstack(no_e_comp)
        nocc_loc = np.cumsum([0]+[x.shape[1] for x in no_comp]).astype(int)
        no_frozen = np.hstack((np.arange(nocc_loc[0], nocc_loc[1]),
                                np.arange(nocc_loc[3], nocc_loc[5]))).astype(int)

        self.mo_coeff,self.mo_energy,self.frozen = no_coeff,no_energy,no_frozen
