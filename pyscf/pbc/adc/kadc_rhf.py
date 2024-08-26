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
import pyscf.ao2mo as ao2mo
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
from pyscf.pbc.mp.kmp2 import (get_nocc, get_nmo, padding_k_idx,_padding_k_idx,
                               padded_mo_coeff, get_frozen_mask, _add_padding)
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.pbc.cc.kccsd_t_rhf import _get_epqr
from pyscf.pbc.lib import kpts_helper
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM  # noqa

import h5py
import tempfile

# Note : All integrals are in Chemist's notation except for vvvv
#        Eg.of momentum conservation :
#        Chemist's  oovv(ijab) : ki - kj + ka - kb
#        Amplitudes t2(ijab)  : ki + kj - ka - kba

def kernel(adc, nroots=1, guess=None, eris=None, kptlist=None, verbose=None):

    adc.method = adc.method.lower()
    if adc.method not in ("adc(2)", "adc(2)-x","adc(3)"):
        raise NotImplementedError(adc.method)

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)
    if adc.verbose >= logger.WARN:
        adc.check_sanity()
    adc.dump_flags()

    if eris is None:
        eris = adc.transform_integrals()

    size = adc.vector_size()
    nroots = min(nroots,size)
    nkpts = adc.nkpts
    nmo = adc.nmo

    if kptlist is None:
        kptlist = range(nkpts)

    dtype = np.result_type(adc.t2[0])

    evals = np.zeros((len(kptlist),nroots), np.float64)
    evecs = np.zeros((len(kptlist),nroots,size), dtype)
    conv = np.zeros((len(kptlist),nroots), np.bool_)
    P = np.zeros((len(kptlist),nroots), np.float64)
    X = np.zeros((len(kptlist),nmo,nroots), dtype)

    imds = adc.get_imds(eris)

    for k, kshift in enumerate(kptlist):
        matvec, diag = adc.gen_matvec(kshift, imds, eris)

        guess = adc.get_init_guess(nroots, diag, ascending=True)

        conv_k,evals_k, evecs_k = lib.linalg_helper.davidson_nosym1(
                lambda xs : [matvec(x) for x in xs], guess, diag,
                nroots=nroots, verbose=log, tol=adc.conv_tol,
                max_cycle=adc.max_cycle, max_space=adc.max_space,
                tol_residual=adc.tol_residual)

        evals_k = evals_k.real
        evals[k] = evals_k
        evecs[k] = evecs_k
        conv[k] = conv_k.real

        U = np.array(evecs[k]).T.copy()

        if adc.compute_properties:
            spec_fac,spec_amp = adc.get_properties(kshift,U,nroots)
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
                            (adc.method, kshift, n, evals[k][n], evals[k][n]*27.2114))
            if adc.compute_properties:
                print_string += ("|  Spec factors = %10.8f  " % P[k][n])
            print_string += ("|  conv = %s" % conv[k][n].real)
            logger.info(adc, print_string)

    log.timer('ADC', *cput0)

    return evals, evecs, P, X


class RADC(pyscf.adc.radc.RADC):
    _keys = {
        'tol_residual','conv_tol', 'e_corr', 'method', 'mo_coeff',
        'mol', 'mo_energy', 'incore_complete',
        'scf_energy', 'e_tot', 't1', 'frozen', 'chkfile',
        'max_space', 't2', 'mo_occ', 'max_cycle','kpts', 'khelper',
        'exxdiv', 'cell', 'nkop_chk', 'kop_npick', 'chnk_size', 'keep_exxdiv',
    }

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):

        from pyscf.pbc.cc.ccsd import _adjust_occ
        assert (isinstance(mf, scf.khf.KSCF))

        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

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
        self.imds = lambda:None

        self.keep_exxdiv = False
        self.mo_energy = mf.mo_energy

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


    def kernel_gs(self):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)

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
        mem_incore *= 16 /1e6
        mem_now = lib.current_memory()[0]

        if isinstance(self._scf.with_df, df.GDF):
            self.chnk_size = self.get_chnk_size()
            self.with_df = self._scf.with_df
            def df_transform():
                return kadc_ao2mo.transform_integrals_df(self)
            self.transform_integrals = df_transform
        elif (mem_incore+mem_now >= self.max_memory and not self.incore_complete):
            def outcore_transform():
                return kadc_ao2mo.transform_integrals_outcore(self)
            self.transform_integrals = outcore_transform

        eris = self.transform_integrals()
        self.e_corr,self.t1,self.t2 = kadc_rhf_amplitudes.compute_amplitudes_energy(
            self, eris=eris, verbose=self.verbose)
        print ("MPn:",self.e_corr)
        self._finalize()
        return self.e_corr, self.t1,self.t2

    def kernel(self, nroots=1, guess=None, eris=None, kptlist=None):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)

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
        mem_incore *= 16 /1e6
        mem_now = lib.current_memory()[0]

        if isinstance(self._scf.with_df, df.GDF):
            self.chnk_size = self.get_chnk_size()
            self.with_df = self._scf.with_df
            def df_transform():
                return kadc_ao2mo.transform_integrals_df(self)
            self.transform_integrals = df_transform
        elif (mem_incore+mem_now >= self.max_memory and not self.incore_complete):
            def outcore_transform():
                return kadc_ao2mo.transform_integrals_outcore(self)
            self.transform_integrals = outcore_transform

        eris = self.transform_integrals()

        self.e_corr, self.t1, self.t2 = kadc_rhf_amplitudes.compute_amplitudes_energy(
            self, eris=eris, verbose=self.verbose)
        print ("MPn:",self.e_corr)
        self._finalize()

        self.method_type = self.method_type.lower()
        if(self.method_type == "ea"):
            e_exc, v_exc, spec_fac, x, adc_es = self.ea_adc(
                nroots=nroots, guess=guess, eris=eris, kptlist=kptlist)

        elif(self.method_type == "ip"):
            e_exc, v_exc, spec_fac, x, adc_es = self.ip_adc(
                nroots=nroots, guess=guess, eris=eris, kptlist=kptlist)

        else:
            raise NotImplementedError(self.method_type)
        self._adc_es = adc_es
        return e_exc, v_exc, spec_fac, x

    def ip_adc(self, nroots=1, guess=None, eris=None, kptlist=None):
        from pyscf.pbc.adc import kadc_rhf_ip
        adc_es = kadc_rhf_ip.RADCIP(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris, kptlist)
        return e_exc, v_exc, spec_fac, x, adc_es

    def ea_adc(self, nroots=1, guess=None, eris=None, kptlist=None):
        from pyscf.pbc.adc import kadc_rhf_ea
        adc_es = kadc_rhf_ea.RADCEA(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris, kptlist)
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
