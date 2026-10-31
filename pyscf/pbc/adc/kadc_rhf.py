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
# Author: Samragni Banerjee <samragnibanerjee4@gmail.com>
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
from pyscf.pbc.adc import kadc_rhf_amplitudes
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

import h5py
import tempfile

# Note : All integrals are in Chemist's notation except for vvvv
#        Eg.of momentum conservation :
#        Chemist's  oovv(ijab) : ki - kj + ka - kb
#        Amplitudes t2(ijab)  : ki + kj - ka - kba


def kernel(adc, nroots=1, guess=None, eris=None, kptlist=None):

    adc.method = adc.method.lower()
    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)
    if adc.verbose >= logger.WARN:
        adc.check_sanity()
    adc.dump_flags()

    if eris is None:
        eris = adc.transform_integrals()

    size = adc.vector_size()
    nroots = min(nroots, size)
    nkpts = adc.nkpts
    nmo = adc.nmo

    if kptlist is None:
        kptlist = range(nkpts)

    dtype = np.result_type(adc.t2[0])

    evals = np.zeros((len(kptlist), nroots), np.float64)
    evecs = np.zeros((len(kptlist), nroots, size), dtype)
    conv = np.zeros((len(kptlist), nroots), np.bool_)
    P = np.zeros((len(kptlist), nroots), np.float64)
    X = np.zeros((len(kptlist), nmo, nroots), dtype)

    imds = adc.get_imds(eris)
    guess_type = None
    if guess is None:
        pass
    elif hasattr(guess, '__len__'):
        guess_type = "read"
        if isinstance(guess, list):
            guess_k = np.array(guess)
        else:
            guess_k = guess.copy()

    for k, kshift in enumerate(kptlist):
        matvec, diag = adc.gen_matvec(kshift, imds, eris)
        if guess_type == "read":
            guess = adc.get_init_guess(nroots, diag, ascending=True, type=guess_type,
                                       ini=guess_k[k], kshift=kshift)
        elif guess_type is None:
            guess = adc.get_init_guess(nroots, diag, ascending=True, kshift=kshift)
        else:
            raise NotImplementedError("Guess type not implemented")
        conv_k, evals_k, evecs_k = lib.linalg_helper.davidson_nosym1(
            lambda xs: [matvec(x) for x in xs], guess, diag,
            nroots=nroots, verbose=log, tol=adc.conv_tol,
            max_cycle=adc.max_cycle, max_space=adc.max_space,
            tol_residual=adc.tol_residual)

        evals_k = evals_k.real
        evals[k] = evals_k
        evecs[k, :, :size] += evecs_k
        conv[k] = conv_k.real

        U = np.array(evecs[k]).T.copy()

        if adc.compute_properties:
            spec_fac, spec_amp = adc.get_properties(kshift, U, nroots)
            P[k] = spec_fac
            X[k] = spec_amp

    nfalse = np.shape(conv)[0] - np.sum(conv)

    msg = ("\n*************************************************************"
           "\n            ADC calculation summary"
           "\n*************************************************************")
    logger.info(adc, msg)
    if nfalse >= 1:
        logger.warn(adc, "Davidson iterations for %s root(s) not converged\n", nfalse)

    for k, kshift in enumerate(kptlist):
        for n in range(nroots):
            print_string = ('%s k-point %d | root %d  |  Energy (Eh) = %14.10f  |  Energy (eV) = %12.8f  ' %
                            (adc.method, kshift, n, evals[k][n], evals[k][n] * HARTREE2EV))
            if adc.compute_properties:
                print_string += ("|  Spec factors = %10.8f  " % P[k][n])
            print_string += ("|  conv = %s" % conv[k][n].real)
            logger.info(adc, print_string)

    log.timer('ADC', *cput0)

    return evals, evecs, P, X


def make_ref_rdm1(adc, with_frozen=True, ao_repr=False):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    t1 = adc.t1
    t2 = adc.t2
    t2_ce = t1[0]
    t1_ccee = t2[0]

    ######################
    einsum_type = True
    nocc = adc.nocc
    nmo = adc.nmo
    nvir = nmo - nocc
    nkpts = adc.nkpts

    OPDM = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
    OPDM[:, :nocc, :nocc] += np.identity(nocc)

    ####### ADC(2) SPIN ADAPTED REF OPDM with SQA ################
    kj, ka = np.indices((nkpts, nkpts))
    idx0 = np.arange(nkpts)[:, None]
    oo = np.zeros((nkpts, nocc, nocc), dtype=np.complex128)
    vv = np.zeros((nkpts, nvir, nvir), dtype=np.complex128)
    for ki in range(nkpts):
        kb = adc.khelper.kconserv[ki, ka, kj]
        t1_ccee_np = np.array(t1_ccee[ki])
        t1_ccee_ijb = t1_ccee_np[idx0, kb]

        oo[ki] -= 2 * lib.einsum('KkIiab,KkJiab->IJ', t1_ccee_np, t1_ccee_np.conj(), optimize=einsum_type)
        oo[ki] += lib.einsum('KkIiab,KkJiba->IJ', t1_ccee_np, t1_ccee_ijb.conj(), optimize=einsum_type)
        vv += 2 * lib.einsum('kKijBa,kKijAa->KAB', t1_ccee_np, t1_ccee_np.conj(), optimize=einsum_type)
        vv -= lib.einsum('kKijBa,kKijaA->KAB', t1_ccee_np, t1_ccee_ijb.conj(), optimize=einsum_type)
        del (t1_ccee_np)
        del (t1_ccee_ijb)

    OPDM[:, :nocc, :nocc] += oo
    OPDM[:, nocc:, nocc:] += vv
    del (oo)
    del (vv)

    if adc.approx_trans_moments is False or adc.method == "adc(3)":
        ### OCC-VIR ###
        OPDM[:, :nocc, nocc:] += t2_ce
        ### VIR-OCC ###
        OPDM[:, nocc:, :nocc] = OPDM[:, :nocc, nocc:].conj().transpose(0, 2, 1)

    ####### ADC(3) SPIN ADAPTED REF OPDM WITH SQA ################
    if adc.method == "adc(3)":
        t2_ccee = t2[1]

        for ki in range(nkpts):
            for kj in range(nkpts):
                for ka in range(nkpts):
                    kb = adc.khelper.kconserv[ki, ka, kj]
                    #### OCC-OCC ###
                    OPDM[ki][:nocc, :nocc] -= 2 * lib.einsum('Iiab,Jiab->IJ', t1_ccee[ki][kj][ka],
                                                             t2_ccee[ki][kj][ka].conj(), optimize=einsum_type)
                    OPDM[ki][:nocc, :nocc] += lib.einsum('Iiab,Jiba->IJ', t1_ccee[ki][kj][ka],
                                                         t2_ccee[ki][kj][kb].conj(), optimize=einsum_type)
                    OPDM[ki][:nocc, :nocc] -= 2 * lib.einsum('Jiab,Iiab->IJ', t1_ccee[ki][kj][ka].conj(),
                                                             t2_ccee[ki][kj][ka], optimize=einsum_type)
                    OPDM[ki][:nocc, :nocc] += lib.einsum('Jiab,Iiba->IJ', t1_ccee[ki][kj][ka].conj(),
                                                         t2_ccee[ki][kj][kb], optimize=einsum_type)
                    ##### VIR-VIR ###
                    OPDM[ka][nocc:, nocc:] += 2 * lib.einsum('ijBa,ijAa->AB', t1_ccee[ki][kj][ka],
                                                             t2_ccee[ki][kj][ka].conj(), optimize=einsum_type)
                    OPDM[ka][nocc:, nocc:] -= lib.einsum('ijBa,ijaA->AB', t1_ccee[ki][kj][ka],
                                                         t2_ccee[ki][kj][kb].conj(), optimize=einsum_type)
                    OPDM[ka][nocc:, nocc:] += 2 * lib.einsum('ijAa,ijBa->AB', t1_ccee[ki][kj][ka].conj(),
                                                             t2_ccee[ki][kj][ka], optimize=einsum_type)
                    OPDM[ka][nocc:, nocc:] -= lib.einsum('ijAa,ijaB->AB', t1_ccee[ki][kj][ka].conj(),
                                                         t2_ccee[ki][kj][kb], optimize=einsum_type)

                ka = ki
                kb = kj
                ##### OCC-VIR ### ####
                OPDM[ki][:nocc, nocc:] += lib.einsum('IiAa,ia->IA', t1_ccee[ki][kj]
                                                     [ka], t2_ce[kj].conj(), optimize=einsum_type)
                OPDM[ki][:nocc, nocc:] -= 1 / 2 * \
                    lib.einsum('IiaA,ia->IA', t1_ccee[ki][kj][kb], t2_ce[kj].conj(), optimize=einsum_type)
            ###### VIR-OCC ###
            OPDM[ki][nocc:, :nocc] = OPDM[ki][:nocc, nocc:].conj().T

    for ki in range(nkpts):
        OPDM[ki] += OPDM[ki].conj().T

    if with_frozen and adc.frozen is not None:
        nmo = adc.mo_occ[0].size
        nocc = np.count_nonzero(adc.mo_occ[0] > 0)
        mask = get_frozen_mask(adc)
        dm = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
        occ_idx = np.arange(nocc)
        k_idx = np.arange(nkpts)
        p_k_idx = padding_k_idx(adc, kind="joint")
        dm[k_idx[:, None], occ_idx, occ_idx] = 2
        for k in k_idx:
            moidx = np.where(mask[k])[0]
            dm[k][moidx[:, None], moidx] = OPDM[k][p_k_idx[k][:, None], p_k_idx[k]]
        OPDM = dm
        if ao_repr:
            mo = adc.mo_coeff
            for k in k_idx:
                OPDM[k] = lib.einsum('pI,IJ,qJ->pq', mo[k], OPDM[k], mo[k].conj())

    elif ao_repr:
        mo = padded_mo_coeff(adc, adc.mo_coeff)
        for k in k_idx:
            OPDM[k] = lib.einsum('pI,IJ,qJ->pq', mo[k], OPDM[k], mo[k].conj())

    return OPDM


def mo_splitter(myadc):
    masks = []
    maskact = get_frozen_mask(myadc)
    for kpt in range(myadc.nkpts):
        maskact_k = maskact[kpt]
        maskocc_k = myadc.mo_occ[kpt] > 1e-6
        masks_k = [
            maskocc_k & ~maskact_k,    # frz occ
            maskocc_k & maskact_k,    # act occ
            ~maskocc_k & maskact_k,    # act vir
            ~maskocc_k & ~maskact_k,    # frz vir
        ]
        masks.append(masks_k)
    return masks


class RADC(pyscf.adc.radc.RADC):
    _keys = pyscf.adc.radc.RADC._keys | {
        'kpts', 'khelper', 'exxdiv', 'cell',
        'nkop_chk', 'kop_npick', 'chnk_size', 'keep_exxdiv',
        'naux', 'if_heri_eris', 'eris', 'if_naf', 'thresh_naf', 'if_corr'
    }

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, mo_energy=None):

        from pyscf.pbc.cc.ccsd import _adjust_occ
        assert (isinstance(mf, scf.khf.KSCF))

        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ
        if mo_energy is None:
            mo_energy = mf.mo_energy

        self._scf = mf
        self.kpts = self._scf.kpts
        self.exxdiv = self._scf.exxdiv
        self.verbose = mf.verbose
        self.max_memory = mf.max_memory
        self.method = "adc(2)"
        self.method_type = "ip"

        self.max_space = getattr(__config__, 'adc_kadc_RADC_max_space', 12)
        self.max_cycle = getattr(__config__, 'adc_kadc_RADC_max_cycle', 50)
        self.conv_tol = getattr(__config__, 'adc_kadc_RADC_conv_tol', 1e-7)
        self.tol_residual = getattr(__config__, 'adc_kadc_RADC_tol_res', 1e-4)
        self.scf_energy = mf.e_tot

        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.cell = self._scf.cell
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.mo_energy = mo_energy
        self.frozen = frozen
        self.compute_properties = True
        self.approx_trans_moments = True

        self._nocc = None
        self._nmo = None
        self._nvir = None
        self.nkop_chk = False
        self.kop_npick = False

        self.t1 = None
        self.t2 = None
        self.e_corr = None
        self.chnk_size = None
        self.imds = lambda: None

        self.keep_exxdiv = False
        self.U = None
        self.naux = None
        self.if_heri_eris = False
        self.if_naf = False
        self.if_corr = True
        self.thresh_naf = 1e-2
        self.eris = None

    make_ref_rdm1 = make_ref_rdm1
    transform_integrals = kadc_ao2mo.transform_integrals_incore
    compute_amplitudes = kadc_rhf_amplitudes.compute_amplitudes
    compute_energy = kadc_rhf_amplitudes.compute_energy
    compute_amplitudes_energy = kadc_rhf_amplitudes.compute_amplitudes_energy
    get_chnk_size = kadc_ao2mo.calculate_chunk_size

    @property
    def nkpts(self):
        return len(self.kpts)

    @property
    def nocc(self):
        return self.get_nocc()

    @property
    def nmo(self):
        return self.get_nmo()

    get_nocc = get_nocc
    get_nmo = get_nmo

    def kernel_gs(self, eris=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        assert (self.mo_coeff is not None)
        assert (self.mo_occ is not None)

        self.method = self.method.lower()
        if self.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
            raise NotImplementedError(self.method)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags_gs()

        nmo = self.nmo
        nocc = self.nocc
        nvir = nmo - nocc
        nkpts = self.nkpts
        mem_incore = nkpts ** 3 * (nocc + nvir) ** 4
        mem_incore *= 4
        mem_incore *= 16 / 1e6
        mem_now = lib.current_memory()[0]
        nocc_arr = np.array(self.get_nocc(per_kpoint=True))
        nmo_arr = np.array(self.get_nmo(per_kpoint=True))
        nvir_arr = nmo_arr - nocc_arr
        nocc_fr = np.array(self._scf.mol.nelectron // 2) - nocc_arr
        nvir_fr = np.array([len(self.mo_occ[ikpt]) for ikpt in range(nkpts)]) - nmo_arr - nocc_fr

        logger.info(self, '******** ADC Orbital Information ********')
        logger.info(self, 'Number list of Frozen Occupied Orbitals: %s', nocc_fr)
        logger.info(self, 'Number list of Frozen Virtual Orbitals: %s', nvir_fr)
        logger.info(self, 'Number list of Active Occupied Orbitals: %s', nocc_arr)
        logger.info(self, 'Number list of Active Virtual Orbitals: %s', nvir_arr)
        if hasattr(self.frozen, '__len__') and isinstance(self.frozen[0], (list, np.ndarray)):
            logger.info(self, 'Frozen Orbital List: %s', self.frozen)
        logger.info(self, '*****************************************')

        if eris is None:
            if isinstance(self._scf.with_df, df.GDF):
                self.chnk_size = self.get_chnk_size()
                self.with_df = self._scf.with_df

                def df_transform():
                    return kadc_ao2mo.transform_integrals_df(self)
                self.transform_integrals = df_transform
            elif (mem_incore + mem_now >= self.max_memory and not self.incore_complete):
                def outcore_transform():
                    return kadc_ao2mo.transform_integrals_outcore(self)
                self.transform_integrals = outcore_transform

            eris = self.transform_integrals()

        self.e_corr, self.t1, self.t2 = kadc_rhf_amplitudes.compute_amplitudes_energy(
            self, eris=eris, verbose=self.verbose, if_corr=self.if_corr)
        self._finalize()
        log.timer('complete kernel', *cput0)
        if self.if_heri_eris:
            self.eris = eris
        return self.e_corr, self.t1, self.t2

    def kernel(self, nroots=1, guess=None, eris=None, kptlist=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        assert (self.mo_coeff is not None)
        assert (self.mo_occ is not None)

        self.method = self.method.lower()
        if self.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
            raise NotImplementedError(self.method)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags_gs()

        nmo = self.nmo
        nocc = self.nocc
        nvir = nmo - nocc
        nkpts = self.nkpts
        mem_incore = nkpts ** 3 * (nocc + nvir) ** 4
        mem_incore *= 4
        mem_incore *= 16 / 1e6
        mem_now = lib.current_memory()[0]
        nocc_arr = np.array(self.get_nocc(per_kpoint=True))
        nmo_arr = np.array(self.get_nmo(per_kpoint=True))
        nvir_arr = nmo_arr - nocc_arr
        nocc_fr = np.array(self._scf.mol.nelectron // 2) - nocc_arr
        nvir_fr = np.array([len(self.mo_occ[ikpt]) for ikpt in range(nkpts)]) - nmo_arr - nocc_fr

        logger.info(self, '******** ADC Orbital Information ********')
        logger.info(self, 'Number list of Frozen Occupied Orbitals: %s', nocc_fr)
        logger.info(self, 'Number list of Frozen Virtual Orbitals: %s', nvir_fr)
        logger.info(self, 'Number list of Active Occupied Orbitals: %s', nocc_arr)
        logger.info(self, 'Number list of Active Virtual Orbitals: %s', nvir_arr)
        if hasattr(self.frozen, '__len__') and isinstance(self.frozen[0], (list, np.ndarray)):
            logger.info(self, 'Frozen Orbital List: %s', self.frozen)
        logger.info(self, '*****************************************')

        if eris is None:
            if isinstance(self._scf.with_df, df.GDF):
                self.chnk_size = self.get_chnk_size()
                self.with_df = self._scf.with_df

                def df_transform():
                    return kadc_ao2mo.transform_integrals_df(self)
                self.transform_integrals = df_transform
            elif (mem_incore + mem_now >= self.max_memory and not self.incore_complete):
                def outcore_transform():
                    return kadc_ao2mo.transform_integrals_outcore(self)
                self.transform_integrals = outcore_transform

            eris = self.transform_integrals()

        self.e_corr, self.t1, self.t2 = kadc_rhf_amplitudes.compute_amplitudes_energy(
            self, eris=eris, verbose=self.verbose, if_corr=self.if_corr)
        self._finalize()

        self.method_type = self.method_type.lower()
        if (self.method_type == "ea"):
            e_exc, v_exc, spec_fac, x, adc_es = self.ea_adc(
                nroots=nroots, guess=guess, eris=eris, kptlist=kptlist)

        elif (self.method_type == "ip"):
            e_exc, v_exc, spec_fac, x, adc_es = self.ip_adc(
                nroots=nroots, guess=guess, eris=eris, kptlist=kptlist)

        else:
            raise NotImplementedError(self.method_type)
        self._adc_es = adc_es
        log.timer('complete kernel', *cput0)
        if self.if_heri_eris:
            self.eris = eris
        return e_exc, v_exc, spec_fac, x

    def _close_amp_scratch(self):
        # close the HDF5 scratch file backing the on-disk amplitudes
        for attr in ('_amp_h5file', '_amp_tmpfile'):
            fh = getattr(self, attr, None)
            if fh is not None:
                try:
                    fh.close()
                except (AttributeError, OSError, ValueError):
                    pass

    def __del__(self):
        self._close_amp_scratch()

    def ip_adc(self, nroots=1, guess=None, eris=None, kptlist=None):
        from pyscf.pbc.adc import kadc_rhf_ip
        adc_es = kadc_rhf_ip.RADCIP(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris, kptlist)
        adc_es.U = v_exc
        return e_exc, v_exc, spec_fac, x, adc_es

    def ea_adc(self, nroots=1, guess=None, eris=None, kptlist=None):
        from pyscf.pbc.adc import kadc_rhf_ea
        adc_es = kadc_rhf_ea.RADCEA(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris, kptlist)
        adc_es.U = v_exc
        return e_exc, v_exc, spec_fac, x, adc_es

    def density_fit(self, auxbasis=None, with_df=None):
        if with_df is None:
            self.with_df = df.DF(self._scf.mol)
            self.with_df.max_memory = self.max_memory
            self.with_df.stdout = self.stdout
            self.with_df.verbose = self.verbose
            if auxbasis is None:
                self.with_df.auxbasis = self._scf.with_df.auxbasis
            else:
                self.with_df.auxbasis = auxbasis
        else:
            self.with_df = with_df
        return self

    def make_rdm1(self, kptlist, root=None, K_idx=None, with_frozen=True, ao_repr=False, if_ss=False):
        if K_idx is None:
            K_idx = kptlist

        if root is None:
            nroots = [range(self._adc_es.U.shape[1]) for _ in K_idx]
        else:
            nroots = root

        rdm1 = self._adc_es.make_rdm1(nroots, kptlist, K_idx, if_ss)

        if with_frozen and self.frozen is not None:
            nmo = self.mo_occ[0].size
            nocc = np.count_nonzero(self.mo_occ[0] > 0)
            mask = get_frozen_mask(self)
            dm = np.zeros((len(K_idx), len(nroots), self.nkpts, nmo, nmo), dtype=np.complex128)
            occ_idx = np.arange(nocc)
            k0_idx = np.arange(len(K_idx))
            k1_idx = np.arange(self.nkpts)
            p_k_idx = padding_k_idx(self, kind="joint")
            s_idx = np.arange(len(nroots))
            K0, S, K1, O = np.meshgrid(k0_idx, s_idx, k1_idx, occ_idx, indexing='ij')
            if if_ss:
                dm[K0, S, K1, O, O] = 4
            else:
                dm[K0, S, K1, O, O] = 2
            for ki in k1_idx:
                moidx = np.where(mask[ki])[0]
                for i in range(len(nroots)):
                    for kj in k0_idx:
                        dm[kj][i][ki][moidx[:, None], moidx] = rdm1[kj][i][ki][p_k_idx[ki][:, None], p_k_idx[ki]]
            rdm1 = dm
            if ao_repr:
                mo = self.mo_coeff
                mo_H = [c.conj() for c in mo]
                rdm1 = lib.einsum('kpI,KSkIJ,kqJ->KSkpq', mo, rdm1, mo_H)
        elif ao_repr:
            mo = padded_mo_coeff(self, self.mo_coeff)
            mo_H = [c.conj() for c in mo]
            rdm1 = lib.einsum('kpI,KSkIJ,kqJ->KSkpq', mo, rdm1, mo_H)

        return rdm1
