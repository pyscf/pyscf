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
from pyscf.pbc.adc import kadc_rhf
from pyscf.pbc.adc import kadc_ao2mo
from pyscf.pbc.adc import dfadc
from pyscf import __config__
from pyscf.pbc.mp.kmp2 import (get_nocc, get_nmo, padding_k_idx, _padding_k_idx,
                               padded_mo_coeff, get_frozen_mask, _add_padding)
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.pbc.cc.kccsd_t_rhf import _get_epqr
from pyscf.pbc.lib import kpts_helper
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM

from pyscf.pbc import tools
import h5py
import tempfile


def vector_size(adc):

    nkpts = adc.nkpts
    nocc = adc.nocc
    nvir = adc.nmo - adc.nocc

    n_singles = nvir
    n_doubles = nkpts * nkpts * nocc * nvir * nvir
    size = n_singles + n_doubles

    return size


def get_imds(adc, eris=None):

    cput0 = (time.process_time(), time.time())
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    nkpts = adc.nkpts

    mo_energy = adc.mo_energy
    mo_coeff = adc.mo_coeff
    nocc = adc.nocc
    nvir = adc.nmo - adc.nocc
    kconserv = adc.khelper.kconserv

    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_occ = np.array(e_occ)
    e_vir = np.array(e_vir)

    idn_vir = np.identity(nvir)
    if eris is None:
        eris = adc.transform_integrals()

    eris_ovov = eris.ovov

    # a-b block
    # Zeroth-order terms
    M_ab = np.empty((nkpts, nvir, nvir), dtype=mo_coeff[0].dtype)

    for ka in range(nkpts):
        kb = ka
        M_ab[ka] = lib.einsum('ab,a->ab', idn_vir, e_vir[ka])
        for kl in range(nkpts):
            for km in range(nkpts):

                kd = kconserv[kl, ka, km]
                # Second-order terms
                t2_1_mla = adc.t2[0][km, kl, ka]
                M_ab[ka] += 0.5 * 0.5 * \
                    lib.einsum('mlad,lbmd->ab', t2_1_mla, eris_ovov[kl, kb, km], optimize=True)
                M_ab[ka] -= 0.5 * 0.5 * \
                    lib.einsum('mlad,ldmb->ab', t2_1_mla, eris_ovov[kl, kd, km], optimize=True)

                t2_1_lma = adc.t2[0][kl, km, ka]
                M_ab[ka] -= 0.5 * 0.5 * \
                    lib.einsum('lmad,lbmd->ab', t2_1_lma, eris_ovov[kl, kb, km], optimize=True)
                M_ab[ka] -= 0.5 * \
                    lib.einsum('lmad,lbmd->ab', t2_1_lma, eris_ovov[kl, kb, km], optimize=True)
                M_ab[ka] += 0.5 * 0.5 * \
                    lib.einsum('lmad,ldmb->ab', t2_1_lma, eris_ovov[kl, kd, km], optimize=True)
                del t2_1_lma

                t2_1_lmb = adc.t2[0][kl, km, kb]
                M_ab[ka] -= 0.5 * 0.5 * \
                    lib.einsum('lmbd,lamd->ab', t2_1_lmb.conj(),
                               eris_ovov[kl, ka, km].conj(), optimize=True)
                M_ab[ka] -= 0.5 * \
                    lib.einsum('lmbd,lamd->ab', t2_1_lmb.conj(),
                               eris_ovov[kl, ka, km].conj(), optimize=True)
                M_ab[ka] += 0.5 * 0.5 * \
                    lib.einsum('lmbd,ldma->ab', t2_1_lmb.conj(),
                               eris_ovov[kl, kd, km].conj(), optimize=True)
                del t2_1_lmb

                t2_1_mlb = adc.t2[0][km, kl, kb]
                M_ab[ka] += 0.5 * 0.5 * \
                    lib.einsum('mlbd,lamd->ab', t2_1_mlb.conj(),
                               eris_ovov[kl, ka, km].conj(), optimize=True)
                M_ab[ka] -= 0.5 * 0.5 * \
                    lib.einsum('mlbd,ldma->ab', t2_1_mlb.conj(),
                               eris_ovov[kl, kd, km].conj(), optimize=True)
                del t2_1_mlb

    cput0 = log.timer_debug1("Completed M_ab second-order terms ADC(2) calculation", *cput0)

    if (method == 'adc(3)'):

        eris_oovv = eris.oovv
        eris_ovvo = eris.ovvo

        t1_2 = adc.t1[0]

        for ka in range(nkpts):
            kb = ka
            for kl in range(nkpts):

                if eris.ovvv is None:
                    chnk_size = adc.chnk_size
                    chnk_size = min(chnk_size, nocc)
                    a = 0
                    for p in range(0, nocc, chnk_size):
                        kd = kconserv[ka, kb, kl]
                        eris_ovvv = dfadc.get_ovvv_df(
                            adc, eris.Lov[kl, kd], eris.Lvv[ka, kb], p, chnk_size).reshape(-1, nvir, nvir, nvir) / nkpts
                        k = eris_ovvv.shape[0]
                        M_ab[ka] += 2. * lib.einsum('ld,ldab->ab', t1_2[kl]
                                                    [a:a + k], eris_ovvv, optimize=True)
                        del eris_ovvv
                        kd = kconserv[kb, ka, kl]
                        eris_ovvv = dfadc.get_ovvv_df(
                            adc, eris.Lov[kl, kd], eris.Lvv[kb, ka], p, chnk_size).reshape(-1, nvir, nvir, nvir) / nkpts
                        k = eris_ovvv.shape[0]
                        M_ab[ka] += 2. * lib.einsum('ld,ldba->ab', t1_2[kl]
                                                    [a:a + k].conj(), eris_ovvv.conj(), optimize=True)
                        del eris_ovvv
                        kd = kconserv[ka, kb, kl]
                        eris_ovvv = dfadc.get_ovvv_df(
                            adc, eris.Lov[kl, kb], eris.Lvv[ka, kd], p, chnk_size).reshape(-1, nvir, nvir, nvir) / nkpts
                        k = eris_ovvv.shape[0]
                        M_ab[ka] -= lib.einsum('ld,lbad->ab', t1_2[kl][a:a + k],
                                               eris_ovvv, optimize=True)
                        del eris_ovvv
                        kd = kconserv[kb, ka, kl]
                        eris_ovvv = dfadc.get_ovvv_df(
                            adc, eris.Lov[kl, ka], eris.Lvv[kb, kd], p, chnk_size).reshape(-1, nvir, nvir, nvir) / nkpts
                        k = eris_ovvv.shape[0]
                        M_ab[ka] -= lib.einsum('ld,labd->ab', t1_2[kl, a:a + k].conj(),
                                               eris_ovvv.conj(), optimize=True)
                        del eris_ovvv
                        a += k

                else:
                    eris_ovvv = eris.ovvv
                    kd = kconserv[ka, kb, kl]
                    M_ab[ka] += 2. * lib.einsum('ld,ldab->ab', t1_2[kl],
                                                eris_ovvv[kl, kd, ka], optimize=True)
                    M_ab[ka] -= lib.einsum('ld,lbad->ab', t1_2[kl],
                                           eris_ovvv[kl, kb, ka], optimize=True)
                    kd = kconserv[kb, ka, kl]
                    M_ab[ka] += 2. * lib.einsum('ld,ldba->ab', t1_2[kl].conj(),
                                                eris_ovvv[kl, kd, kb].conj(), optimize=True)
                    M_ab[ka] -= lib.einsum('ld,labd->ab', t1_2[kl].conj(),
                                           eris_ovvv[kl, ka, kb].conj(), optimize=True)

        cput0 = log.timer_debug1("Completed M_ab ovvv ADC(3) calculation", *cput0)

        for ka in range(nkpts):
            kb = ka
            log.timer_debug1("start:t2_ijbc*v_acij||O(No^2*Nv^3*Nk^3)")
            for kl in range(nkpts):
                for km in range(nkpts):
                    kd = kconserv[km, ka, kl]

                    t2_1 = adc.t2[0]
                    t2_2_lma = adc.t2[1][kl, km, ka]
                    M_ab[ka] -= 0.5 * 0.5 * \
                        lib.einsum('lmad,lbmd->ab', t2_2_lma, eris_ovov[kl, kb, km], optimize=True)
                    M_ab[ka] += 0.5 * 0.5 * \
                        lib.einsum('lmad,ldmb->ab', t2_2_lma, eris_ovov[kl, kd, km], optimize=True)
                    M_ab[ka] -= 0.5 * lib.einsum('lmad,lbmd->ab',
                                                 t2_2_lma, eris_ovov[kl, kb, km], optimize=True)

                    t2_2_mla = adc.t2[1][km, kl, ka]
                    M_ab[ka] += 0.5 * 0.5 * \
                        lib.einsum('mlad,lbmd->ab', t2_2_mla, eris_ovov[kl, kb, km], optimize=True)
                    M_ab[ka] -= 0.5 * 0.5 * \
                        lib.einsum('mlad,ldmb->ab', t2_2_mla, eris_ovov[kl, kd, km], optimize=True)

                    t2_2_lmb = adc.t2[1][kl, km, kb]
                    M_ab[ka] -= 0.5 * 0.5 * \
                        lib.einsum('lmbd,lamd->ab', t2_2_lmb.conj(),
                                   eris_ovov[kl, ka, km].conj(), optimize=True)
                    M_ab[ka] += 0.5 * 0.5 * \
                        lib.einsum('lmbd,ldma->ab', t2_2_lmb.conj(),
                                   eris_ovov[kl, kd, km].conj(), optimize=True)
                    M_ab[ka] -= 0.5 * lib.einsum('lmbd,lamd->ab', t2_2_lmb.conj(),
                                                 eris_ovov[kl, ka, km].conj(), optimize=True)

                    t2_2_mlb = adc.t2[1][km, kl, kb]
                    M_ab[ka] += 0.5 * 0.5 * \
                        lib.einsum('mlbd,mdla->ab', t2_2_mlb.conj(),
                                   eris_ovov[km, kd, kl].conj(), optimize=True)
                    M_ab[ka] -= 0.5 * 0.5 * \
                        lib.einsum('mlbd,ldma->ab', t2_2_mlb.conj(),
                                   eris_ovov[kl, kd, km].conj(), optimize=True)
                    del t2_2_mlb

            log.timer_debug1("start:temp1_ovov*v_ovov||O(No^3*Nv^3*Nk^3)")
            for kn, ke, kd in kpts_helper.loop_kkk(nkpts):

                kl = kconserv[ke, kn, kd]
                km = kconserv[kb, kl, kd]
                temp_t2_v_1 = lib.einsum(
                    'lned,mlbd->nemb', t2_1[kl, kn, ke], t2_1[km, kl, kb].conj(), optimize=True)
                M_ab[ka] -= 0.5 * lib.einsum('nemb,nmae->ab',
                                             temp_t2_v_1, eris_oovv[kn, km, ka], optimize=True)
                M_ab[ka] += 0.5 * 2. * lib.einsum('nemb,neam->ab',
                                                  temp_t2_v_1, eris_ovvo[kn, ke, ka], optimize=True)
                M_ab[ka] += 0.5 * 2. * lib.einsum('nema,nebm->ab',
                                                  temp_t2_v_1, eris_ovvo[kn, ke, kb], optimize=True)
                M_ab[ka] -= 0.5 * lib.einsum('nema,nmbe->ab',
                                             temp_t2_v_1, eris_oovv[kn, km, kb], optimize=True)
                del temp_t2_v_1

                temp_t2_v_1 = lib.einsum(
                    'nled,lmbd->mbne', t2_1[kn, kl, ke], t2_1[kl, km, kb].conj(), optimize=True)
                M_ab[ka] -= 0.5 * lib.einsum('mbne,nmae->ab',
                                             temp_t2_v_1, eris_oovv[kn, km, ka], optimize=True)
                M_ab[ka] += 0.5 * 2. * lib.einsum('mbne,neam->ab',
                                                  temp_t2_v_1, eris_ovvo[kn, ke, ka], optimize=True)
                M_ab[ka] -= 0.5 * lib.einsum('mane,nmbe->ab',
                                             temp_t2_v_1, eris_oovv[kn, km, kb], optimize=True)
                M_ab[ka] += 0.5 * 2. * lib.einsum('mane,nebm->ab',
                                                  temp_t2_v_1, eris_ovvo[kn, ke, kb], optimize=True)
                del temp_t2_v_1

                temp_t2_v_2 = lib.einsum(
                    'nled,mlbd->nemb', t2_1[kn, kl, ke], t2_1[km, kl, kb].conj(), optimize=True)
                M_ab[ka] += 0.5 * 2. * lib.einsum('nemb,nmae->ab',
                                                  temp_t2_v_2, eris_oovv[kn, km, ka], optimize=True)
                M_ab[ka] -= 0.5 * 4. * lib.einsum('nemb,neam->ab',
                                                  temp_t2_v_2, eris_ovvo[kn, ke, ka], optimize=True)
                M_ab[ka] += 0.5 * 2. * lib.einsum('nema,nmbe->ab',
                                                  temp_t2_v_2, eris_oovv[kn, km, kb], optimize=True)
                M_ab[ka] -= 0.5 * 4. * lib.einsum('nema,nebm->ab',
                                                  temp_t2_v_2, eris_ovvo[kn, ke, kb], optimize=True)
                del temp_t2_v_2

                temp_t2_v_3 = lib.einsum(
                    'lned,lmbd->nemb', t2_1[kl, kn, ke], t2_1[kl, km, kb].conj(), optimize=True)
                M_ab[ka] -= 0.5 * lib.einsum('nemb,neam->ab',
                                             temp_t2_v_3, eris_ovvo[kn, ke, ka], optimize=True)
                M_ab[ka] += 0.5 * 2. * lib.einsum('nemb,nmae->ab',
                                                  temp_t2_v_3, eris_oovv[kn, km, ka], optimize=True)
                M_ab[ka] += 0.5 * 2. * lib.einsum('nema,nmbe->ab',
                                                  temp_t2_v_3, eris_oovv[kn, km, kb], optimize=True)
                M_ab[ka] -= 0.5 * lib.einsum('nema,nebm->ab',
                                             temp_t2_v_3, eris_ovvo[kn, ke, kb], optimize=True)
                del temp_t2_v_3

            log.timer_debug1("start:temp2_oo*v_ovov||O(No^3*Nv^2*Nk^3)")
            for kn, ke, kd in kpts_helper.loop_kkk(nkpts):
                kl = kconserv[ke, kn, kd]
                km = kconserv[ke, kl, kd]
                temp_t2_v_8 = lib.einsum(
                    'lned,mled->mn', t2_1[kl, kn, ke], t2_1[km, kl, ke].conj(), optimize=True)
                M_ab[ka] += 2. * lib.einsum('mn,nmab->ab', temp_t2_v_8,
                                            eris_oovv[kn, km, ka], optimize=True)
                M_ab[ka] -= lib.einsum('mn,nbam->ab', temp_t2_v_8,
                                       eris_ovvo[kn, kb, ka], optimize=True)
                del temp_t2_v_8

                temp_t2_v_9 = lib.einsum(
                    'nled,mled->mn', t2_1[kn, kl, ke], t2_1[km, kl, ke].conj(), optimize=True)
                M_ab[ka] -= 4. * lib.einsum('mn,nmab->ab', temp_t2_v_9,
                                            eris_oovv[kn, km, ka], optimize=True)
                M_ab[ka] += 2. * lib.einsum('mn,nbam->ab', temp_t2_v_9,
                                            eris_ovvo[kn, kb, ka], optimize=True)
                del temp_t2_v_9

            log.timer_debug1("start:Mab vvvv||O(No^2*Nv^3*Nk^3)")
            for km in range(nkpts):
                for kl in range(nkpts):
                    kf = kconserv[km, kl, ka]
                    temp_t2 = adc.imds.t2_1_vvvv[:]
                    t2_1_mla = adc.t2[0][km, kl, ka]
                    M_ab[ka] -= 0.5 * 0.25 * \
                        lib.einsum('mlaf,mlbf->ab', t2_1_mla,
                                   temp_t2[km, kl, kb].conj(), optimize=True)
                    M_ab[ka] += 0.5 * 0.25 * \
                        lib.einsum('mlaf,lmbf->ab', t2_1_mla,
                                   temp_t2[kl, km, kb].conj(), optimize=True)
                    M_ab[ka] += 0.5 * 0.25 * \
                        lib.einsum('mlaf,mlfb->ab', t2_1_mla,
                                   temp_t2[km, kl, kf].conj(), optimize=True)
                    M_ab[ka] -= 0.5 * 0.25 * \
                        lib.einsum('mlaf,lmfb->ab', t2_1_mla,
                                   temp_t2[kl, km, kf].conj(), optimize=True)
                    M_ab[ka] -= 0.5 * lib.einsum('mlaf,mlbf->ab',
                                                 t2_1_mla, temp_t2[km, kl, kb].conj(), optimize=True)
                    del t2_1_mla

                    t2_1_lma = adc.t2[0][kl, km, ka]
                    M_ab[ka] += 0.5 * 0.25 * \
                        lib.einsum('lmaf,mlbf->ab', t2_1_lma,
                                   temp_t2[km, kl, kb].conj(), optimize=True)
                    M_ab[ka] -= 0.5 * 0.25 * \
                        lib.einsum('lmaf,lmbf->ab', t2_1_lma,
                                   temp_t2[kl, km, kb].conj(), optimize=True)
                    M_ab[ka] -= 0.5 * 0.25 * \
                        lib.einsum('lmaf,mlfb->ab', t2_1_lma,
                                   temp_t2[km, kl, kf].conj(), optimize=True)
                    M_ab[ka] += 0.5 * 0.25 * \
                        lib.einsum('lmaf,lmfb->ab', t2_1_lma,
                                   temp_t2[kl, km, kf].conj(), optimize=True)
                    del t2_1_lma

                    kd = kconserv[km, ka, kl]
                    t2_1_mlb = adc.t2[0][km, kl, kb]
                    M_ab[ka] -= 0.5 * 0.25 * \
                        lib.einsum('mlad,mlbd->ab', temp_t2[km,
                                   kl, ka].conj(), t2_1_mlb, optimize=True)
                    M_ab[ka] += 0.5 * 0.25 * \
                        lib.einsum('lmad,mlbd->ab', temp_t2[kl,
                                   km, ka].conj(), t2_1_mlb, optimize=True)
                    M_ab[ka] -= 0.5 * lib.einsum('mlad,mlbd->ab',
                                                 temp_t2[km, kl, ka].conj(), t2_1_mlb, optimize=True)

                    M_ab[ka] += 0.5 * 0.25 * \
                        lib.einsum('lmad,mlbd->ab', temp_t2[kl,
                                   km, ka].conj(), t2_1_mlb, optimize=True)
                    M_ab[ka] -= 0.5 * 0.25 * \
                        lib.einsum('mlad,mlbd->ab', temp_t2[km,
                                   kl, ka].conj(), t2_1_mlb, optimize=True)
                    del t2_1_mlb

                    t2_1_lmb = adc.t2[0][kl, km, kb]
                    M_ab[ka] += 0.5 * 0.25 * \
                        lib.einsum('mlad,lmbd->ab', temp_t2[km,
                                   kl, ka].conj(), t2_1_lmb, optimize=True)
                    M_ab[ka] -= 0.5 * 0.25 * \
                        lib.einsum('lmad,lmbd->ab', temp_t2[kl,
                                   km, ka].conj(), t2_1_lmb, optimize=True)

                    M_ab[ka] -= 0.5 * 0.25 * \
                        lib.einsum('lmad,lmbd->ab', temp_t2[kl,
                                   km, ka].conj(), t2_1_lmb, optimize=True)
                    M_ab[ka] += 0.5 * 0.25 * \
                        lib.einsum('mlad,lmbd->ab', temp_t2[km,
                                   kl, ka].conj(), t2_1_lmb, optimize=True)
                    del temp_t2
                    del t2_1_lmb

                    for kd in range(nkpts):
                        kf = kconserv[km, kd, kl]
                        ke = kconserv[kb, ka, kf]
                        temp_t2_vv_1 = lib.einsum(
                            'mldf,mled->ef', t2_1[km, kl, kd], t2_1[km, kl, ke].conj(), optimize=True)
                        temp_t2_vv_2 = lib.einsum(
                            'mldf,lmed->ef', t2_1[km, kl, kd], t2_1[kl, km, ke].conj(), optimize=True)
                        temp_t2_vv_3 = lib.einsum(
                            'lmdf,mled->ef', t2_1[kl, km, kd], t2_1[km, kl, ke].conj(), optimize=True)
                        temp_t2_vv_4 = lib.einsum(
                            'lmdf,lmed->ef', t2_1[kl, km, kd], t2_1[kl, km, ke].conj(), optimize=True)
                        temp_t2_vv_5 = lib.einsum(
                            'mlfd,mled->ef', t2_1[km, kl, kf], t2_1[km, kl, ke].conj(), optimize=True)
                        if eris.vvvv is None:
                            chnk_size = adc.chnk_size
                            chnk_size = min(chnk_size, nvir)
                            a = 0
                            for p in range(0, nvir, chnk_size):
                                eris_vvvv = dfadc.get_vvvv_df(
                                    adc, eris.Lvv[kb, ka], eris.Lvv[ke, kf], p, chnk_size) / nkpts
                                k = eris_vvvv.shape[0]
                                M_ab[ka, a:a + k] -= lib.einsum('ef,aebf->ab', temp_t2_vv_1, eris_vvvv, optimize=True)
                                M_ab[ka, a:a + k] += lib.einsum('ef,aebf->ab', temp_t2_vv_2, eris_vvvv, optimize=True)
                                M_ab[ka, a:a + k] += lib.einsum('ef,aebf->ab', temp_t2_vv_3, eris_vvvv, optimize=True)
                                M_ab[ka, a:a + k] -= lib.einsum('ef,aebf->ab', temp_t2_vv_4, eris_vvvv, optimize=True)
                                M_ab[ka, a:a + k] += 2. * \
                                    lib.einsum('ef,aebf->ab', temp_t2_vv_5, eris_vvvv, optimize=True)
                                del eris_vvvv

                                eris_vvvv = dfadc.get_vvvv_df(
                                    adc, eris.Lvv[kb, kf], eris.Lvv[ke, ka], p, chnk_size) / nkpts
                                M_ab[ka, a:a + k] += 0.5 * \
                                    lib.einsum('ef,aefb->ab', temp_t2_vv_1, eris_vvvv, optimize=True)
                                M_ab[ka, a:a + k] -= 0.5 * \
                                    lib.einsum('ef,aefb->ab', temp_t2_vv_2, eris_vvvv, optimize=True)
                                M_ab[ka, a:a + k] -= 0.5 * \
                                    lib.einsum('ef,aefb->ab', temp_t2_vv_3, eris_vvvv, optimize=True)
                                M_ab[ka, a:a + k] += 0.5 * \
                                    lib.einsum('ef,aefb->ab', temp_t2_vv_4, eris_vvvv, optimize=True)
                                M_ab[ka, a:a + k] -= lib.einsum('ef,aefb->ab', temp_t2_vv_5, eris_vvvv, optimize=True)
                                del eris_vvvv
                                a += k

                        elif isinstance(eris.vvvv, np.ndarray):
                            eris_vvvv = eris.vvvv
                            M_ab[ka] -= lib.einsum('ef,aebf->ab', temp_t2_vv_1, eris_vvvv[ka, ke, kb], optimize=True)
                            M_ab[ka] += lib.einsum('ef,aebf->ab', temp_t2_vv_2, eris_vvvv[ka, ke, kb], optimize=True)
                            M_ab[ka] += lib.einsum('ef,aebf->ab', temp_t2_vv_3, eris_vvvv[ka, ke, kb], optimize=True)
                            M_ab[ka] -= lib.einsum('ef,aebf->ab', temp_t2_vv_4, eris_vvvv[ka, ke, kb], optimize=True)
                            M_ab[ka] += 2. * lib.einsum('ef,aebf->ab', temp_t2_vv_5,
                                                        eris_vvvv[ka, ke, kb], optimize=True)
                            M_ab[ka] += 0.5 * lib.einsum('ef,aefb->ab', temp_t2_vv_1,
                                                         eris_vvvv[ka, ke, kf], optimize=True)
                            M_ab[ka] -= 0.5 * lib.einsum('ef,aefb->ab', temp_t2_vv_2,
                                                         eris_vvvv[ka, ke, kf], optimize=True)
                            M_ab[ka] -= 0.5 * lib.einsum('ef,aefb->ab', temp_t2_vv_3,
                                                         eris_vvvv[ka, ke, kf], optimize=True)
                            M_ab[ka] += 0.5 * lib.einsum('ef,aefb->ab', temp_t2_vv_4,
                                                         eris_vvvv[ka, ke, kf], optimize=True)
                            M_ab[ka] -= lib.einsum('ef,aefb->ab', temp_t2_vv_5, eris_vvvv[ka, ke, kf], optimize=True)
                        else:
                            chnk_size = adc.chnk_size
                            chnk_size = min(chnk_size, nvir)
                            a = 0
                            for p in range(0, nvir, chnk_size):
                                eris_vvvv = eris.vvvv[kb, ke, ka, p:p + chnk_size]
                                k = eris_vvvv.shape[0]
                                M_ab[ka, a:a + k] -= lib.einsum('ef,aebf->ab', temp_t2_vv_1, eris_vvvv, optimize=True)
                                M_ab[ka, a:a + k] += lib.einsum('ef,aebf->ab', temp_t2_vv_2, eris_vvvv, optimize=True)
                                M_ab[ka, a:a + k] += lib.einsum('ef,aebf->ab', temp_t2_vv_3, eris_vvvv, optimize=True)
                                M_ab[ka, a:a + k] -= lib.einsum('ef,aebf->ab', temp_t2_vv_4, eris_vvvv, optimize=True)
                                M_ab[ka, a:a + k] += 2. * \
                                    lib.einsum('ef,aebf->ab', temp_t2_vv_5, eris_vvvv, optimize=True)
                                del eris_vvvv
                                eris_vvvv = eris.vvvv[ka, ke, kf, p:p + chnk_size]
                                M_ab[ka, a:a + k] += 0.5 * \
                                    lib.einsum('ef,aefb->ab', temp_t2_vv_1, eris_vvvv, optimize=True)
                                M_ab[ka, a:a + k] -= 0.5 * \
                                    lib.einsum('ef,aefb->ab', temp_t2_vv_2, eris_vvvv, optimize=True)
                                M_ab[ka, a:a + k] -= 0.5 * \
                                    lib.einsum('ef,aefb->ab', temp_t2_vv_3, eris_vvvv, optimize=True)
                                M_ab[ka, a:a + k] += 0.5 * \
                                    lib.einsum('ef,aefb->ab', temp_t2_vv_4, eris_vvvv, optimize=True)
                                M_ab[ka, a:a + k] -= lib.einsum('ef,aefb->ab', temp_t2_vv_5, eris_vvvv, optimize=True)
                                del eris_vvvv
                                a += k
                        del temp_t2_vv_1
                        del temp_t2_vv_2
                        del temp_t2_vv_3
                        del temp_t2_vv_4
                        del temp_t2_vv_5

    cput0 = log.timer_debug1("Completed M_ab ADC(3) calculation", *cput0)

    return M_ab


def get_diag(adc, kshift, M_ab=None, eris=None):

    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    if M_ab is None:
        M_ab = adc.get_imds()

    nkpts = adc.nkpts
    kconserv = adc.khelper.kconserv

    nocc = adc.nocc
    nvir = adc.nmo - adc.nocc
    n_singles = nvir
    n_doubles = nkpts * nkpts * nocc * nvir * nvir

    dim = n_singles + n_doubles

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    mo_energy = adc.mo_energy
    mo_coeff = adc.mo_coeff
    nocc = adc.nocc
    nmo = adc.nmo
    nvir = nmo - nocc
    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_vir = np.array(e_vir)
    e_occ = np.array(e_occ)

    diag = np.zeros((dim), dtype=np.complex128)
    doubles = np.zeros((nkpts, nkpts, nocc * nvir * nvir), dtype=np.complex128)

    # Compute precond in h1-h1 block
    M_ab_diag = np.diagonal(M_ab[kshift])

    # Compute precond in 2p1h-2p1h block
    for kj in range(nkpts):
        for ka in range(nkpts):
            kb = kconserv[kshift, ka, kj]
            d_ab = e_vir[ka][:nvir, None] + e_vir[kb][:nvir]
            d_i = e_occ[kj][:, None]
            D_n = -d_i + d_ab.reshape(-1)
            doubles[kj, ka] += D_n.reshape(-1)
    if (adc.frozen is not None) or (not np.all([x.shape[1] == adc.nmo for x in mo_coeff])):
        doubles = doubles.reshape(nkpts, nkpts, nocc, nvir, nvir)
        (M_ab_diag, doubles) = mask_frozen_ea(adc, M_ab_diag, doubles, kshift)

    diag[s1:f1] = M_ab_diag.copy()
    diag[s2:f2] = doubles.reshape(-1)
    log.timer_debug1("Completed ea_diag calculation")

    return diag


def matvec(adc, kshift, M_ab=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nkpts = adc.nkpts
    nocc = adc.nocc
    kconserv = adc.khelper.kconserv
    nvir = adc.nmo - adc.nocc
    n_singles = adc.nmo - adc.nocc
    n_doubles = nkpts * nkpts * nocc * nvir * nvir

    s_singles = 0
    f_singles = n_singles
    s_doubles = f_singles
    f_doubles = s_doubles + n_doubles

    mo_energy = adc.mo_energy
    mo_coeff = adc.mo_coeff
    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:nvir + nocc] for k in range(nkpts)]

    e_vir = np.array(e_vir)
    e_occ = np.array(e_occ)

    if M_ab is None:
        M_ab = adc.get_imds()

    # Calculate sigma vector
    def sigma_(r):
        cput0 = (time.process_time(), time.time())
        log = logger.Logger(adc.stdout, adc.verbose)

        r1 = r[s_singles:f_singles]
        r2 = r[s_doubles:f_doubles]

        r2 = r2.reshape(nkpts, nkpts, nocc, nvir, nvir)
        s2 = np.zeros((nkpts, nkpts, nocc, nvir, nvir), dtype=np.complex128)
        cell = adc.cell
        kpts = adc.kpts
        madelung = tools.madelung(cell, kpts)

############ ADC(2) ij block ############################

        s1 = lib.einsum('ab,b->a', M_ab[kshift], r1)

########### ADC(2) coupling blocks #########################

        for kb in range(nkpts):
            for kc in range(nkpts):
                ki = kconserv[kb, kshift, kc]
                if eris.ovvv is None:
                    chnk_size = adc.chnk_size
                    chnk_size = min(chnk_size, nocc)
                    a = 0
                    for p in range(0, nocc, chnk_size):
                        eris_ovvv = dfadc.get_ovvv_df(
                            adc, eris.Lov[ki, kc], eris.Lvv[kshift, kb], p, chnk_size).reshape(
                                -1, nvir, nvir, nvir) / nkpts
                        k = eris_ovvv.shape[0]
                        s1 += 2. * lib.einsum('icab,ibc->a', eris_ovvv.conj(),
                                              r2[ki, kb, a:a + k], optimize=True)
                        s2[ki, kb, a:a + k] += lib.einsum('icab,a->ibc', eris_ovvv, r1, optimize=True)
                        del eris_ovvv

                        eris_ovvv = dfadc.get_ovvv_df(
                            adc, eris.Lov[ki, kb], eris.Lvv[kshift, kc], p, chnk_size).reshape(
                                -1, nvir, nvir, nvir) / nkpts
                        s1 -= lib.einsum('ibac,ibc->a', eris_ovvv.conj(),
                                         r2[ki, kb, a:a + k], optimize=True)
                        del eris_ovvv
                        a += k
                else:
                    eris_ovvv = eris.ovvv[:]
                    s1 += 2. * lib.einsum('icab,ibc->a',
                                          eris_ovvv[ki, kc, kshift].conj(),
                                          r2[ki, kb], optimize=True)
                    s2[ki, kb] += lib.einsum('icab,a->ibc',
                                             eris_ovvv[ki, kc, kshift],
                                             r1, optimize=True)
                    del eris_ovvv


#########    ########## ADC(2) ajk - bil block ############################

                s2[ki, kb] -= lib.einsum('i,ibc->ibc', e_occ[ki], r2[ki, kb])
                s2[ki, kb] += lib.einsum('b,ibc->ibc', e_vir[kb], r2[ki, kb])
                s2[ki, kb] += lib.einsum('c,ibc->ibc', e_vir[kc], r2[ki, kb])

################ ADC(3) ajk - bil block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):
            eris_oovv = eris.oovv
            eris_ovvo = eris.ovvo

            for kx in range(nkpts):
                for ky in range(nkpts):
                    ki = kconserv[ky, kshift, kx]
                    for kz in range(nkpts):
                        kw = kconserv[kx, kz, ky]

                        if isinstance(eris.vvvv, np.ndarray):
                            eris_vvvv = eris.vvvv.reshape(nkpts, nkpts, nkpts, nvir * nvir, nvir * nvir)
                            r2_1 = r2.reshape(nkpts, nkpts, nocc, nvir * nvir)
                            s2[ki, kx] += np.dot(r2_1[ki, kw], eris_vvvv[kx, ky,
                                                 kw].T.conj()).reshape(nocc, nvir, nvir)
                        elif eris.vvvv is None:
                            s2[ki, kx] += ea_contract_r_vvvv(adc, r2[ki, kw], eris.Lvv, kx, ky, kw)
                        else:
                            s2[ki, kx] += ea_contract_r_vvvv(adc, r2[ki, kw], eris.vvvv, kx, ky, kw)
                    for kj in range(nkpts):
                        kz = kconserv[ky, ki, kj]
                        s2[ki, kx] -= 0.5 * lib.einsum('iyzj,jzx->ixy',
                                                       eris_ovvo[ki, ky, kz], r2[kj, kz], optimize=True)
                        s2[ki, kx] += lib.einsum('iyzj,jxz->ixy', eris_ovvo[ki,
                                                                            ky, kz], r2[kj, kx], optimize=True)
                        s2[ki, kx] -= 0.5 * lib.einsum('ijzy,jxz->ixy',
                                                       eris_oovv[ki, kj, kz], r2[kj, kx], optimize=True)

                        kz = kconserv[kj, ki, kx]
                        s2[ki, kx] -= 0.5 * lib.einsum('ijzx,jzy->ixy',
                                                       eris_oovv[ki, kj, kz], r2[kj, kz], optimize=True)

                        kw = kconserv[kj, ki, kx]
                        s2[ki, kx] -= 0.5 * lib.einsum('ijwx,jwy->ixy',
                                                       eris_oovv[ki, kj, kw], r2[kj, kw], optimize=True)

                        kw = kconserv[kj, ki, ky]
                        s2[ki, kx] -= 0.5 * lib.einsum('ijwy,jxw->ixy',
                                                       eris_oovv[ki, kj, kw], r2[kj, kx], optimize=True)
                        s2[ki, kx] += lib.einsum('iywj,jxw->ixy', eris_ovvo[ki,
                                                                            ky, kw], r2[kj, kx], optimize=True)
                        s2[ki, kx] -= 0.5 * lib.einsum('iywj,jwx->ixy',
                                                       eris_ovvo[ki, ky, kw], r2[kj, kw], optimize=True)

            if adc.exxdiv is not None:
                s2 += -madelung * r2
        if (method == "adc(3)"):

            eris_ovoo = eris.ovoo

################ ADC(3) a - ibc block and ibc-a coupling blocks ########################

            t2_1 = adc.t2[0]

            for kz in range(nkpts):
                for kw in range(nkpts):
                    kj = kconserv[kw, kshift, kz]

                    for kl in range(nkpts):
                        km = kconserv[kw, kl, kz]
                        t2_1_lmz = adc.t2[0][kl, km, kz]
                        ka = kconserv[km, kj, kl]
                        temp_1 = lib.einsum('lmzw,jzw->jlm', t2_1_lmz, r2[kj, kz])
                        temp = 0.25 * lib.einsum('lmzw,jzw->jlm', t2_1_lmz, r2[kj, kz])
                        temp -= 0.25 * lib.einsum('lmzw,jwz->jlm', t2_1_lmz, r2[kj, kw])
                        del t2_1_lmz

                        t2_1_mlz = adc.t2[0][km, kl, kz]
                        temp -= 0.25 * lib.einsum('mlzw,jzw->jlm', t2_1_mlz, r2[kj, kz])
                        temp += 0.25 * lib.einsum('mlzw,jwz->jlm', t2_1_mlz, r2[kj, kw])
                        del t2_1_mlz

                        s1 += lib.einsum('jlm,lamj->a', temp, eris_ovoo[kl, ka, km], optimize=True)
                        s1 -= lib.einsum('jlm,malj->a', temp, eris_ovoo[km, ka, kl], optimize=True)
                        s1 += lib.einsum('jlm,lamj->a', temp_1, eris_ovoo[kl, ka, km], optimize=True)

                        del temp
                        del temp_1

            for kx in range(nkpts):
                for ky in range(nkpts):
                    ki = kconserv[ky, kshift, kx]
                    for kl in range(nkpts):
                        km = kconserv[kx, kl, ky]
                        kb = kconserv[km, ki, kl]
                        temp = lib.einsum(
                            'b,lbmi->lmi', r1, eris_ovoo[kl, kb, km].conj(), optimize=True)
                        s2[ki, kx] += lib.einsum('lmi,lmxy->ixy', temp,
                                                 t2_1[kl, km, kx].conj(), optimize=True)
                        del temp

            for kz in range(nkpts):
                for kw in range(nkpts):
                    kj = kconserv[kz, kshift, kw]
                    for kl in range(nkpts):
                        kd = kconserv[kj, kw, kl]
                        t2_1_jlw = adc.t2[0][kj, kl, kw]

                        temp_s_a = lib.einsum('jlwd,jzw->lzd', t2_1_jlw, r2[kj, kz], optimize=True)
                        temp_s_a -= lib.einsum('jlwd,jwz->lzd', t2_1_jlw, r2[kj, kw], optimize=True)

                        temp_t2_r2_1 = lib.einsum('jlwd,jzw->lzd', t2_1_jlw, r2[kj, kz], optimize=True)
                        temp_t2_r2_1 -= lib.einsum('jlwd,jwz->lzd', t2_1_jlw, r2[kj, kw], optimize=True)
                        temp_t2_r2_1 += lib.einsum('jlwd,jzw->lzd', t2_1_jlw, r2[kj, kz], optimize=True)
                        del t2_1_jlw

                        t2_1_ljw = adc.t2[0][kl, kj, kw]
                        temp_s_a -= lib.einsum('ljwd,jzw->lzd', t2_1_ljw, r2[kj, kz], optimize=True)
                        temp_s_a += lib.einsum('ljwd,jwz->lzd', t2_1_ljw, r2[kj, kw], optimize=True)
                        temp_s_a += lib.einsum('ljdw,jzw->lzd',
                                               t2_1[kl, kj, kd], r2[kj, kz], optimize=True)

                        temp_t2_r2_1 -= lib.einsum('ljwd,jzw->lzd', t2_1_ljw, r2[kj, kz], optimize=True)
                        del t2_1_ljw
                        temp_t2_r2_4 = lib.einsum(
                            'jldw,jwz->lzd', t2_1[kj, kl, kd], r2[kj, kw], optimize=True)

                        if eris.ovvv is None:
                            chnk_size = adc.chnk_size
                            chnk_size = min(chnk_size, nocc)
                            a = 0
                            for p in range(0, nocc, chnk_size):
                                ka = kconserv[kz, kd, kl]
                                eris_ovvv = dfadc.get_ovvv_df(
                                    adc, eris.Lov[kl, kd], eris.Lvv[kz, ka], p,
                                    chnk_size).reshape(-1, nvir, nvir, nvir) / nkpts
                                k = eris_ovvv.shape[0]

                                s1 += 0.5 * lib.einsum('lzd,ldza->a',
                                                       temp_s_a[a:a + k], eris_ovvv, optimize=True)
                                s1 += 0.5 * lib.einsum('lzd,ldza->a',
                                                       temp_t2_r2_1[a:a + k], eris_ovvv, optimize=True)
                                del eris_ovvv

                                eris_ovvv = dfadc.get_ovvv_df(
                                    adc, eris.Lov[kl, ka], eris.Lvv[kz, kd], p,
                                    chnk_size).reshape(-1, nvir, nvir, nvir) / nkpts
                                s1 -= 0.5 * lib.einsum('lzd,lazd->a',
                                                       temp_s_a[a:a + k], eris_ovvv, optimize=True)
                                s1 -= 0.5 * lib.einsum('lzd,lazd->a',
                                                       temp_t2_r2_4[a:a + k], eris_ovvv, optimize=True)
                                del eris_ovvv
                                a += k
                        else:
                            ka = kconserv[kz, kd, kl]
                            eris_ovvv = eris.ovvv[:]
                            s1 += 0.5 * lib.einsum('lzd,ldza->a', temp_s_a,
                                                   eris_ovvv[kl, kd, kz], optimize=True)
                            s1 += 0.5 * lib.einsum('lzd,ldza->a', temp_t2_r2_1,
                                                   eris_ovvv[kl, kd, kz], optimize=True)

                            s1 -= 0.5 * lib.einsum('lzd,lazd->a', temp_s_a,
                                                   eris_ovvv[kl, ka, kz], optimize=True)
                            s1 -= 0.5 * lib.einsum('lzd,lazd->a', temp_t2_r2_4,
                                                   eris_ovvv[kl, ka, kz], optimize=True)
                            del eris_ovvv
                        del temp_s_a
                        del temp_t2_r2_1
                        del temp_t2_r2_4

            for kz in range(nkpts):
                for kw in range(nkpts):
                    kj = kconserv[kz, kshift, kw]
                    for kl in range(nkpts):
                        kd = kconserv[kj, kz, kl]
                        t2_1_jlz = adc.t2[0][kj, kl, kz]

                        temp_s_a_1 = -lib.einsum('jlzd,jwz->lwd',
                                                 t2_1_jlz, r2[kj, kw], optimize=True)
                        temp_s_a_1 += lib.einsum('jlzd,jzw->lwd',
                                                 t2_1_jlz, r2[kj, kz], optimize=True)
                        temp_t2_r2_2 = -lib.einsum('jlzd,jwz->lwd',
                                                   t2_1_jlz, r2[kj, kw], optimize=True)
                        temp_t2_r2_2 += lib.einsum('jlzd,jzw->lwd',
                                                   t2_1_jlz, r2[kj, kz], optimize=True)
                        temp_t2_r2_2 -= lib.einsum('jlzd,jwz->lwd',
                                                   t2_1_jlz, r2[kj, kw], optimize=True)
                        del t2_1_jlz

                        t2_1_ljz = adc.t2[0][kl, kj, kz]
                        temp_s_a_1 += lib.einsum('ljzd,jwz->lwd', t2_1_ljz, r2[kj, kw], optimize=True)
                        temp_s_a_1 -= lib.einsum('ljzd,jzw->lwd', t2_1_ljz, r2[kj, kz], optimize=True)
                        temp_s_a_1 -= lib.einsum('ljdz,jwz->lwd',
                                                 t2_1[kl, kj, kd], r2[kj, kw], optimize=True)

                        temp_t2_r2_2 += lib.einsum('ljzd,jwz->lwd', t2_1_ljz, r2[kj, kw], optimize=True)

                        temp_t2_r2_3 = -lib.einsum('ljzd,jzw->lwd', t2_1_ljz, r2[kj, kz], optimize=True)
                        del t2_1_ljz

                        if eris.ovvv is None:
                            chnk_size = adc.chnk_size
                            chnk_size = min(chnk_size, nocc)
                            a = 0
                            for p in range(0, nocc, chnk_size):

                                ka = kconserv[kw, kd, kl]
                                eris_ovvv = dfadc.get_ovvv_df(
                                    adc, eris.Lov[kl, kd], eris.Lvv[kw, ka], p,
                                    chnk_size).reshape(-1, nvir, nvir, nvir) / nkpts
                                k = eris_ovvv.shape[0]
                                s1 -= 0.5 * lib.einsum('lwd,ldwa->a',
                                                       temp_s_a_1[a:a + k], eris_ovvv, optimize=True)
                                s1 -= 0.5 * lib.einsum('lwd,ldwa->a',
                                                       temp_t2_r2_2[a:a + k], eris_ovvv, optimize=True)
                                del eris_ovvv

                                eris_ovvv = dfadc.get_ovvv_df(
                                    adc, eris.Lov[kl, ka], eris.Lvv[kw, kd], p,
                                    chnk_size).reshape(-1, nvir, nvir, nvir) / nkpts
                                s1 += 0.5 * lib.einsum('lwd,lawd->a',
                                                       temp_s_a_1[a:a + k], eris_ovvv, optimize=True)
                                s1 += 0.5 * lib.einsum('lwd,lawd->a',
                                                       temp_t2_r2_3[a:a + k], eris_ovvv, optimize=True)
                                del eris_ovvv
                                a += k

                        else:
                            ka = kconserv[kw, kd, kl]
                            eris_ovvv = eris.ovvv[:]
                            s1 -= 0.5 * lib.einsum('lwd,ldwa->a', temp_s_a_1,
                                                   eris_ovvv[kl, kd, kw], optimize=True)
                            s1 += 0.5 * lib.einsum('lwd,lawd->a', temp_s_a_1,
                                                   eris_ovvv[kl, ka, kw], optimize=True)

                            s1 -= 0.5 * lib.einsum('lwd,ldwa->a', temp_t2_r2_2,
                                                   eris_ovvv[kl, kd, kw], optimize=True)
                            s1 += 0.5 * lib.einsum('lwd,lawd->a', temp_t2_r2_3,
                                                   eris_ovvv[kl, ka, kw], optimize=True)
                            del eris_ovvv

                        del temp_s_a_1
                        del temp_t2_r2_2
                        del temp_t2_r2_3

            for kx in range(nkpts):
                for ky in range(nkpts):
                    ki = kconserv[ky, kshift, kx]
                    for kl in range(nkpts):
                        temp = np.zeros((nocc, nvir, nvir), dtype=np.complex128)
                        temp_1_1 = np.zeros((nocc, nvir, nvir), dtype=np.complex128)
                        temp_2_1 = np.zeros((nocc, nvir, nvir), dtype=np.complex128)
                        if eris.ovvv is None:
                            chnk_size = adc.chnk_size
                            chnk_size = min(chnk_size, nocc)
                            a = 0
                            for p in range(0, nocc, chnk_size):
                                kd = kconserv[kl, kshift, kx]
                                kb = kconserv[kl, kd, kx]

                                eris_ovvv = dfadc.get_ovvv_df(
                                    adc, eris.Lov[kl, kd], eris.Lvv[kx, kb], p,
                                    chnk_size).reshape(-1, nvir, nvir, nvir) / nkpts
                                k = eris_ovvv.shape[0]
                                temp_1_1[a:a + k] += lib.einsum('ldxb,b->lxd',
                                                                eris_ovvv, r1, optimize=True)
                                temp_2_1[a:a + k] += lib.einsum('ldxb,b->lxd',
                                                                eris_ovvv, r1, optimize=True)
                                del eris_ovvv

                                eris_ovvv = dfadc.get_ovvv_df(
                                    adc, eris.Lov[kl, kb], eris.Lvv[kx, kd], p,
                                    chnk_size).reshape(-1, nvir, nvir, nvir) / nkpts
                                temp_1_1[a:a + k] -= lib.einsum('lbxd,b->lxd',
                                                                eris_ovvv, r1, optimize=True)
                                del eris_ovvv

                                kd = kconserv[kl, kshift, ky]
                                kb = kconserv[ky, kd, kl]
                                eris_ovvv = dfadc.get_ovvv_df(
                                    adc, eris.Lov[kl, kb], eris.Lvv[ky, kd], p,
                                    chnk_size).reshape(-1, nvir, nvir, nvir) / nkpts
                                temp[a:a + k] -= lib.einsum('lbyd,b->lyd', eris_ovvv, r1, optimize=True)
                                del eris_ovvv
                                a += k
                        else:
                            kd = kconserv[ki, ky, kl]
                            kb = kconserv[kl, kd, kx]
                            eris_ovvv = eris.ovvv[:]
                            temp_1_1 += lib.einsum('ldxb,b->lxd',
                                                   eris_ovvv[kl, kd, kx], r1, optimize=True)
                            temp_1_1 -= lib.einsum('lbxd,b->lxd',
                                                   eris_ovvv[kl, kb, kx], r1, optimize=True)
                            temp_2_1 += lib.einsum('ldxb,b->lxd',
                                                   eris_ovvv[kl, kd, kx], r1, optimize=True)

                            kd = kconserv[kl, kshift, ky]
                            kb = kconserv[ky, kd, kl]
                            temp -= lib.einsum('lbyd,b->lyd', eris_ovvv[kl, kb, ky], r1, optimize=True)
                            del eris_ovvv

                        kd = kconserv[kl, ky, ki]
                        s2[ki, kx] += lib.einsum('lxd,lidy->ixy', temp_1_1.conj(),
                                                 t2_1[kl, ki, kd].conj(), optimize=True)
                        s2[ki, kx] += lib.einsum('lxd,ilyd->ixy', temp_2_1.conj(),
                                                 t2_1[ki, kl, ky].conj(), optimize=True)
                        s2[ki, kx] -= lib.einsum('lxd,ildy->ixy', temp_2_1.conj(),
                                                 t2_1[ki, kl, kd].conj(), optimize=True)

                        kd = kconserv[kl, kx, ki]
                        s2[ki, kx] += lib.einsum('lyd,lixd->ixy', temp.conj(),
                                                 t2_1[kl, ki, kx].conj(), optimize=True)
                        del temp
                        del temp_1_1
                        del temp_2_1

        s2 = s2.reshape(-1)
        s = np.hstack((s1, s2))
        del s1
        del s2

        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)

        return s
    return sigma_


def get_trans_moments(adc, kshift):

    nmo = adc.nmo
    T = []
    for orb in range(nmo):
        T_a = get_trans_moments_orbital(adc, orb, kshift)
        T.append(T_a)

    T = np.array(T)
    return T


def get_trans_moments_orbital(adc, orb, kshift):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    nkpts = adc.nkpts
    nocc = adc.nocc
    nvir = adc.nmo - adc.nocc
    t2_1 = adc.t2[0]

    idn_vir = np.identity(nvir)

    T1 = np.zeros((nvir), dtype=np.complex128)
    T2 = np.zeros((nkpts, nkpts, nocc, nvir, nvir), dtype=np.complex128)

######## ADC(2) 1h part  ############################################

    if orb < nocc:

        if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
            t1_2 = adc.t1[0]
            T1 -= t1_2[kshift][orb, :]

        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = adc.khelper.kconserv[kj, ka, kshift]
                ki = adc.khelper.kconserv[ka, kj, kb]

                t2_1_t = t2_1[ki, kj, ka].transpose(1, 0, 2, 3)
                T2[kj, ka] -= t2_1_t[:, orb, :, :].conj()

    else:

        T1 += idn_vir[(orb - nocc), :]
        for kk in range(nkpts):
            for kc in range(nkpts):
                kl = adc.khelper.kconserv[kc, kk, kshift]
                ka = adc.khelper.kconserv[kc, kl, kk]
                T1 -= 0.25 * \
                    lib.einsum('klc,klac->a', t2_1[kk, kl, kshift][:, :,
                               (orb - nocc), :], t2_1[kk, kl, ka].conj(), optimize=True)
                T1 -= 0.25 * \
                    lib.einsum('lkc,lkac->a', t2_1[kl, kk, kshift][:, :,
                               (orb - nocc), :], t2_1[kl, kk, ka].conj(), optimize=True)

                T1 -= 0.25 * \
                    lib.einsum('klc,klac->a', t2_1[kk, kl, kshift][:, :,
                               (orb - nocc), :], t2_1[kk, kl, ka].conj(), optimize=True)
                T1 += 0.25 * \
                    lib.einsum('lkc,klac->a', t2_1[kl, kk, kshift][:, :,
                               (orb - nocc), :], t2_1[kk, kl, ka].conj(), optimize=True)
                T1 += 0.25 * \
                    lib.einsum('klc,lkac->a', t2_1[kk, kl, kshift][:, :,
                               (orb - nocc), :], t2_1[kl, kk, ka].conj(), optimize=True)
                T1 -= 0.25 * \
                    lib.einsum('lkc,lkac->a', t2_1[kl, kk, kshift][:, :,
                               (orb - nocc), :], t2_1[kl, kk, ka].conj(), optimize=True)

######### ADC(3) 2p-1h  part  ############################################

    if (adc.method == "adc(2)-x" and adc.approx_trans_moments is False) or (adc.method == "adc(3)"):

        t2_2 = adc.t2[1]

        if orb < nocc:

            for kj in range(nkpts):
                for ka in range(nkpts):
                    kb = adc.khelper.kconserv[kj, ka, kshift]
                    ki = adc.khelper.kconserv[ka, kj, kb]

                    t2_2_t = t2_2[ki, kj, ka].conj().transpose(1, 0, 2, 3)

                    T2[kj, ka] -= t2_2_t[:, orb, :, :].conj()


########### ADC(3) 1p part  ############################################

    if (method == 'adc(3)'):
        if orb < nocc:
            for kk in range(nkpts):
                for kc in range(nkpts):
                    ka = adc.khelper.kconserv[kk, kc, kshift]
                    T1 += 0.5 * lib.einsum('kac,ck->a',
                                           t2_1[kk, kshift, kc][:, orb, :, :], t1_2[kc].T, optimize=True)
                    T1 -= 0.5 * lib.einsum('kac,ck->a',
                                           t2_1[kshift, kk, ka][orb, :, :, :], t1_2[kc].T, optimize=True)
                    T1 -= 0.5 * lib.einsum('kac,ck->a',
                                           t2_1[kshift, kk, ka][orb, :, :, :], t1_2[kc].T, optimize=True)

        else:
            for kk in range(nkpts):
                for kc in range(nkpts):
                    kl = adc.khelper.kconserv[kk, kc, kshift]
                    ka = adc.khelper.kconserv[kl, kc, kk]

                    T1 -= 0.25 * \
                        lib.einsum('klc,klac->a', t2_1[kk, kl, kshift][:, :,
                                   (orb - nocc), :], t2_2[kk, kl, ka].conj(), optimize=True)
                    T1 -= 0.25 * \
                        lib.einsum('lkc,lkac->a', t2_1[kl, kk, kshift][:, :,
                                   (orb - nocc), :], t2_2[kl, kk, ka].conj(), optimize=True)
                    T1 -= 0.25 * \
                        lib.einsum('klc,klac->a', t2_1[kk, kl, kshift][:, :,
                                   (orb - nocc), :], t2_2[kk, kl, ka].conj(), optimize=True)
                    T1 += 0.25 * \
                        lib.einsum('klc,lkac->a', t2_1[kk, kl, kshift][:, :,
                                   (orb - nocc), :], t2_2[kl, kk, ka].conj(), optimize=True)
                    T1 += 0.25 * \
                        lib.einsum('lkc,klac->a', t2_1[kl, kk, kshift][:, :,
                                   (orb - nocc), :], t2_2[kk, kl, ka].conj(), optimize=True)
                    T1 -= 0.25 * \
                        lib.einsum('lkc,lkac->a', t2_1[kl, kk, kshift][:, :,
                                   (orb - nocc), :], t2_2[kl, kk, ka].conj(), optimize=True)

                    T1 -= 0.25 * \
                        lib.einsum('klac,klc->a', t2_1[kk, kl, ka].conj(),
                                   t2_2[kk, kl, kshift][:, :, (orb - nocc), :], optimize=True)
                    T1 -= 0.25 * \
                        lib.einsum('lkac,lkc->a', t2_1[kl, kk, ka].conj(),
                                   t2_2[kl, kk, kshift][:, :, (orb - nocc), :], optimize=True)
                    T1 -= 0.25 * \
                        lib.einsum('klac,klc->a', t2_1[kk, kl, ka].conj(),
                                   t2_2[kk, kl, kshift][:, :, (orb - nocc), :], optimize=True)
                    T1 += 0.25 * \
                        lib.einsum('klac,lkc->a', t2_1[kk, kl, ka].conj(),
                                   t2_2[kl, kk, kshift][:, :, (orb - nocc), :], optimize=True)
                    T1 += 0.25 * \
                        lib.einsum('lkac,klc->a', t2_1[kl, kk, ka].conj(),
                                   t2_2[kk, kl, kshift][:, :, (orb - nocc), :], optimize=True)
                    T1 -= 0.25 * \
                        lib.einsum('lkac,lkc->a', t2_1[kl, kk, ka].conj(),
                                   t2_2[kl, kk, kshift][:, :, (orb - nocc), :], optimize=True)

        del t2_2
    del t2_1

    for ka in range(nkpts):
        for kb in range(nkpts):
            ki = adc.khelper.kconserv[kb, kshift, ka]
            T2[ki, ka] += T2[ki, ka] - T2[ki, kb].transpose(0, 2, 1)

    T2 = T2.reshape(-1)
    T = np.hstack((T1, T2))

    return T


def renormalize_eigenvectors(adc, kshift, U, nroots=1):

    nkpts = adc.nkpts
    nocc = adc.t2[0].shape[3]
    nvir = adc.nmo - adc.nocc
    n_singles = nvir

    for I in range(U.shape[1]):
        U1 = U[:n_singles, I]
        U2 = U[n_singles:, I].reshape(nkpts, nkpts, nocc, nvir, nvir)
        UdotU = np.dot(U1.conj().ravel(), U1.ravel())
        for ka in range(nkpts):
            for kb in range(nkpts):
                ki = adc.khelper.kconserv[kb, kshift, ka]
                UdotU += 2. * np.dot(U2[ki, ka].conj().ravel(), U2[ki, ka].ravel()) - \
                    np.dot(U2[ki, ka].conj().ravel(),
                           U2[ki, kb].transpose(0, 2, 1).ravel())
        U[:, I] /= np.sqrt(UdotU)

    U = U.reshape(-1, nroots)

    return U


def make_rdm1(adc, root=None, kptlist=None, K_idx=None, if_ss=False):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)

    rdm1 = []

    for kpt, kshift in enumerate(K_idx):
        k = kptlist.index(kshift)
        U = np.array(adc.U[k]).T.copy()
        U = adc.renormalize_eigenvectors(kshift, U, adc.U.shape[1])
        rdm1_k = []
        for i in root[kpt]:
            rdm1_k.append(make_rdm1_eigenvectors(adc, U[:, i], U[:, i], kshift, if_ss))
        rdm1_k = np.stack(rdm1_k, axis=0)
        rdm1.append(rdm1_k)
    cput0 = log.timer_debug1("completed OPDM calculation", *cput0)
    return rdm1


def make_rdm1_eigenvectors_slow(adc, L, R, kshift):

    L = np.array(L).ravel().conj()
    R = np.array(R).ravel()

    t2_ce = adc.t1[0]
    t1_ccee = adc.t2[0]

    einsum = lib.einsum
    kconserv = adc.khelper.kconserv

    nocc = adc.nocc
    nmo = adc.nmo
    nvir = nmo - nocc
    nkpts = adc.nkpts

    n_singles = nvir
    n_doubles = nkpts * nkpts * nocc * nvir * nvir

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    rdm1 = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
    kd_oc = np.identity(nocc)

    L1 = L[s1:f1]
    L2 = L[s2:f2]
    R1 = R[s1:f1]
    R2 = R[s2:f2]

    L2 = L2.reshape(nkpts, nkpts, nocc, nvir, nvir)
    R2 = R2.reshape(nkpts, nkpts, nocc, nvir, nvir)

# block- ij
    # 000
    for ki in range(nkpts):
        rdm1[ki][:nocc, :nocc] += 2 * einsum('ij,m,m->ij', kd_oc, L1, R1, optimize=True)
    # 101
    for ka in range(nkpts):
        for ki in range(nkpts):
            kb = kconserv[ki, ka, kshift]
            rdm1[ki][:nocc, :nocc] -= 2 * einsum('Jab,Iab->IJ', L2[ki][ka], R2[ki][ka], optimize=True)
            rdm1[ki][:nocc, :nocc] += einsum('Jab,Iba->IJ', L2[ki][ka], R2[ki][kb], optimize=True)
    for kj in range(nkpts):
        for ki in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kshift]
                rdm1[kj][:nocc, :nocc] += 4 * einsum('iab,iab,IJ->IJ', L2[ki][ka], R2[ki][ka], kd_oc, optimize=True)
                rdm1[kj][:nocc, :nocc] -= 2 * einsum('iab,iba,IJ->IJ', L2[ki][ka], R2[ki][kb], kd_oc, optimize=True)
    # 020
    for kj in range(nkpts):
        for ki in range(nkpts):
            for kb in range(nkpts):
                kc = kconserv[kj, kb, ki]
                rdm1[kj][:nocc, :nocc] -= 4 * einsum('a,a,Iibc,Jibc->IJ', L1, R1,
                                                     t1_ccee[kj][ki][kb], t1_ccee[kj][ki][kb].conj(), optimize=True)
                rdm1[kj][:nocc, :nocc] += 2 * einsum('a,a,Iibc,Jicb->IJ', L1, R1,
                                                     t1_ccee[kj][ki][kb], t1_ccee[kj][ki][kc].conj(), optimize=True)
            kb = kshift
            kc = kconserv[kj, kb, ki]
            rdm1[kj][:nocc, :nocc] += 2 * einsum('a,b,Iiac,Jibc->IJ', L1, R1,
                                                 t1_ccee[kj][ki][kb], t1_ccee[kj][ki][kb].conj(), optimize=True)
            rdm1[kj][:nocc, :nocc] -= einsum('a,b,Iiac,Jicb->IJ', L1, R1, t1_ccee[kj]
                                             [ki][kb], t1_ccee[kj][ki][kc].conj(), optimize=True)
            rdm1[kj][:nocc, :nocc] -= einsum('a,b,Iica,Jibc->IJ', L1, R1, t1_ccee[kj]
                                             [ki][kc], t1_ccee[kj][ki][kb].conj(), optimize=True)
            rdm1[kj][:nocc, :nocc] += 2 * einsum('a,b,Iica,Jicb->IJ', L1, R1,
                                                 t1_ccee[kj][ki][kc], t1_ccee[kj][ki][kc].conj(), optimize=True)

# block- ab
    # 000
    rdm1[kshift][nocc:, nocc:] += einsum('A,B->AB', L1, R1, optimize=True)
    # 101
    for ka in range(nkpts):
        for ki in range(nkpts):
            kb = kconserv[ki, ka, kshift]
            rdm1[kb][nocc:, nocc:] += 2 * einsum('iAa,iBa->AB', L2[ki][kb], R2[ki][kb], optimize=True)
            rdm1[kb][nocc:, nocc:] -= einsum('iAa,iaB->AB', L2[ki][kb], R2[ki][ka], optimize=True)
            rdm1[kb][nocc:, nocc:] -= einsum('iaA,iBa->AB', L2[ki][ka], R2[ki][kb], optimize=True)
            rdm1[kb][nocc:, nocc:] += 2 * einsum('iaA,iaB->AB', L2[ki][ka], R2[ki][ka], optimize=True)
    # 020
    for ki in range(nkpts):
        for kj in range(nkpts):
            ka = kshift
            kb = kconserv[ki, ka, kj]
            rdm1[ka][nocc:, nocc:] -= einsum('A,a,ijBb,ijab->AB', L1, R1, t1_ccee[ki]
                                             [kj][ka], t1_ccee[ki][kj][ka].conj(), optimize=True)
            rdm1[ka][nocc:, nocc:] += 1 / 2 * einsum('A,a,ijBb,ijba->AB', L1, R1,
                                                     t1_ccee[ki][kj][ka], t1_ccee[ki][kj][kb].conj(), optimize=True)
            rdm1[ka][nocc:, nocc:] -= einsum('a,B,ijab,ijAb->AB', L1, R1, t1_ccee[ki]
                                             [kj][ka], t1_ccee[ki][kj][ka].conj(), optimize=True)
            rdm1[ka][nocc:, nocc:] += 1 / 2 * einsum('a,B,ijab,ijbA->AB', L1, R1,
                                                     t1_ccee[ki][kj][ka], t1_ccee[ki][kj][kb].conj(), optimize=True)
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                rdm1[ka][nocc:, nocc:] += 4 * einsum('a,a,ijBb,ijAb->AB', L1, R1,
                                                     t1_ccee[ki][kj][ka], t1_ccee[ki][kj][ka].conj(), optimize=True)
                rdm1[ka][nocc:, nocc:] -= 2 * einsum('a,a,ijBb,ijbA->AB', L1, R1,
                                                     t1_ccee[ki][kj][ka], t1_ccee[ki][kj][kb].conj(), optimize=True)
            kb = kshift
            ka = kconserv[ki, kb, kj]
            rdm1[ka][nocc:, nocc:] -= 2 * einsum('a,b,ijBa,ijAb->AB', L1, R1,
                                                 t1_ccee[ki][kj][ka], t1_ccee[ki][kj][ka].conj(), optimize=True)
            rdm1[ka][nocc:, nocc:] += einsum('a,b,ijBa,ijbA->AB', L1, R1, t1_ccee[ki]
                                             [kj][ka], t1_ccee[ki][kj][kb].conj(), optimize=True)

# block- ia
    # 100
    for ki in range(nkpts):
        rdm1[ki][:nocc, nocc:] -= einsum('a,IAa->IA', L1, R2[ki][ki], optimize=True)
        ka = kshift
        rdm1[ki][:nocc, nocc:] += 2 * einsum('a,IaA->IA', L1, R2[ki][ka], optimize=True)
    # 110
    for ki in range(nkpts):
        for ka in range(nkpts):
            kj = kshift
            kb = kconserv[ki, ka, kj]
            rdm1[kj][:nocc, nocc:] -= 2 * einsum('iab,A,Iiab->IA', L2[ki][ka], R1, t1_ccee[kj][ki][ka], optimize=True)
            rdm1[kj][:nocc, nocc:] += einsum('iab,A,Iiba->IA', L2[ki][ka], R1, t1_ccee[kj][ki][kb], optimize=True)
        for kj in range(nkpts):
            ka = kshift
            kb = kconserv[kj, kj, ki]
            rdm1[kj][:nocc, nocc:] += 4 * einsum('iab,a,IiAb->IA', L2[ki][ka], R1, t1_ccee[kj][ki][kj], optimize=True)
            rdm1[kj][:nocc, nocc:] -= 2 * einsum('iab,a,IibA->IA', L2[ki][ka], R1, t1_ccee[kj][ki][kb], optimize=True)
            rdm1[kj][:nocc, nocc:] -= 2 * einsum('iab,b,IiAa->IA', L2[ki][ka], R1, t1_ccee[kj][ki][kj], optimize=True)
            rdm1[kj][:nocc, nocc:] += einsum('iab,b,IiaA->IA', L2[ki][ka], R1, t1_ccee[kj][ki][ka], optimize=True)

# block- ai
    for ki in range(nkpts):
        rdm1[ki][nocc:, :nocc] = rdm1[ki][:nocc, nocc:].conj().T

    if adc.approx_trans_moments is False or adc.method == "adc(3)":
        ki = kshift
        ### block- ia ###
        rdm1[ki][:nocc, nocc:] -= einsum('a,A,Ia->IA', L1, R1, t2_ce[ki], optimize=True)
        for kj in range(nkpts):
            rdm1[kj][:nocc, nocc:] += 2 * einsum('a,a,IA->IA', L1, R1, t2_ce[kj], optimize=True)
        ### block- ai ###
            rdm1[kj][nocc:, :nocc] = rdm1[kj][:nocc, nocc:].conj().T

    return rdm1


def make_rdm1_eigenvectors(adc, L, R, kshift, if_ss):

    L = np.array(L).ravel().conj()
    R = np.array(R).ravel()

    t2_ce = adc.t1[0]
    t1_ccee = adc.t2[0]

    einsum = lib.einsum
    kconserv = adc.khelper.kconserv
    kconserv_s = kconserv[kshift, :, :]

    nocc = adc.nocc
    nmo = adc.nmo
    nvir = nmo - nocc
    nkpts = adc.nkpts
    occ_list = range(nocc)

    t1_ccee_sjb = np.zeros_like(t1_ccee[0])
    t1_ccee_sja = t1_ccee[kshift]
    t1_ccee_ijs = t1_ccee[:, :, kshift, :, :, :]

    kj, ka = np.indices((nkpts, nkpts))
    kb = adc.khelper.kconserv[kshift, ka, kj]
    t1_ccee_sjb[kj, ka] = t1_ccee_sja[kj, kb]

    n_singles = nvir
    n_doubles = nkpts * nkpts * nocc * nvir * nvir

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    rdm1 = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)

    L1 = L[s1:f1]
    L2 = L[s2:f2]
    R1 = R[s1:f1]
    R2 = R[s2:f2]

    L2 = L2.reshape(nkpts, nkpts, nocc, nvir, nvir)
    R2 = R2.reshape(nkpts, nkpts, nocc, nvir, nvir)
    R2_iba = np.zeros_like(R2)
    L2_iba = np.zeros_like(L2)

    ka, ki = np.indices((nkpts, nkpts))
    kb = kconserv_s[ka, ki]
    R2_iba[ki, ka] = R2[ki, kb]
    L2_iba[ki, ka] = L2[ki, kb]
    R2_isb = R2[:, kshift, :, :, :]
    L2_isb = L2[:, kshift, :, :, :]
# block- ij
    # 000
    rdm1[:, occ_list, occ_list] += 2 * einsum('m,m->', L1, R1, optimize=True)
    # 101
    rdm1[:, :nocc, :nocc] -= 2 * einsum('KkJab,KkIab->KIJ', L2, R2, optimize=True)
    rdm1[:, :nocc, :nocc] += einsum('KkJab,KkIba->KIJ', L2, R2_iba, optimize=True)
    rdm1[:, occ_list, occ_list] += 4 * einsum('Kkiab,Kkiab->', L2, R2, optimize=True)
    rdm1[:, occ_list, occ_list] -= 2 * einsum('Kkiab,Kkiba->', L2, R2_iba, optimize=True)
    # 020
    rdm1[:, :nocc, :nocc] += 2 * einsum('a,b,KkIiac,KkJibc->KIJ', L1, R1,
                                        t1_ccee_ijs, t1_ccee_ijs.conj(), optimize=True)

# block- ab
    # 000
    rdm1[kshift, nocc:, nocc:] += einsum('A,B->AB', L1, R1, optimize=True)
    # 101
    rdm1[:, nocc:, nocc:] += 2 * einsum('kKiAa,kKiBa->KAB', L2, R2, optimize=True)
    rdm1[:, nocc:, nocc:] -= einsum('kKiAa,kKiaB->KAB', L2, R2_iba, optimize=True)
    rdm1[:, nocc:, nocc:] -= einsum('kKiaA,kKiBa->KAB', L2_iba, R2, optimize=True)
    rdm1[:, nocc:, nocc:] += 2 * einsum('kKiaA,kKiaB->KAB', L2_iba, R2_iba, optimize=True)
    # 020
    rdm1[kshift, nocc:, nocc:] -= einsum('A,a,kKijBb,kKijab->AB', L1, R1,
                                         t1_ccee_ijs, t1_ccee_ijs.conj(), optimize=True)
    rdm1[kshift, nocc:, nocc:] -= einsum('a,B,kKijab,kKijAb->AB', L1, R1,
                                         t1_ccee_ijs, t1_ccee_ijs.conj(), optimize=True)

# block- ia
    # 100
    rdm1[:, :nocc, nocc:] -= einsum('a,KKIAa->KIA', L1, R2, optimize=True)
    rdm1[:, :nocc, nocc:] += 2 * einsum('a,KIaA->KIA', L1, R2_isb, optimize=True)
    # 110
    rdm1[kshift, :nocc, nocc:] -= 2 * einsum('kKiab,A,kKIiab->IA', L2, R1, t1_ccee_sja, optimize=True)
    rdm1[kshift, :nocc, nocc:] += einsum('kKiab,A,kKIiba->IA', L2, R1, t1_ccee_sjb, optimize=True)
    rdm1[:, :nocc, nocc:] += einsum('kiab,b,KkIiaA->KIA', L2_isb, R1, t1_ccee_ijs, optimize=True)

    del (R2_iba)
    del (L2_iba)
    del (R2_isb)
    del (t1_ccee_sjb)
    del (t1_ccee_sja)

    kj, ka = np.indices((nkpts, nkpts))
    idx0 = np.arange(nkpts)[:, None]
    LR = np.dot(L1, R1)
    if if_ss:
        rdm1[:, :nocc, :nocc] += np.identity(nocc)
        LR += 1.
    t1_ccee_ijs_b = np.zeros_like(t1_ccee[0])
    oo = np.zeros((nkpts, nocc, nocc), dtype=np.complex128)
    vv = np.zeros((nkpts, nvir, nvir), dtype=np.complex128)
    ov = np.zeros((nkpts, nocc, nvir), dtype=np.complex128)
    for ki in range(nkpts):
        kb = adc.khelper.kconserv[ki, ka, kj]
        t1_ccee_np = np.array(t1_ccee[ki])
        t1_ccee_ijb = t1_ccee_np[idx0, kb]
        t1_ccee_ijs_b[ki] = t1_ccee_ijb[:, kshift].copy()
# block- ij
        oo[ki] -= 4 * einsum('KkIibc,KkJibc->IJ', t1_ccee_np, t1_ccee_np.conj(), optimize=True)
        oo[ki] += 2 * einsum('KkIibc,KkJicb->IJ', t1_ccee_np, t1_ccee_ijb.conj(), optimize=True)
# block- ab
        vv += 4 * einsum('kKijBb,kKijAb->KAB', t1_ccee_np, t1_ccee_np.conj(), optimize=True)
        vv -= 2 * einsum('kKijBb,kKijbA->KAB', t1_ccee_np, t1_ccee_ijb.conj(), optimize=True)
# block- ia
        ov[ki] += 4 * einsum('kiab,a,kIiAb->IA', L2_isb, R1, t1_ccee_np[:, ki], optimize=True)
        ov[ki] -= 2 * einsum('kiab,a,kkIibA->IA', L2_isb, R1, t1_ccee_np, optimize=True)
        ov[ki] -= 2 * einsum('kiab,b,kIiAa->IA', L2_isb, R1, t1_ccee_np[:, ki], optimize=True)
        del (t1_ccee_ijb)
        del (t1_ccee_np)

    oo *= LR
    vv *= LR
    rdm1[:, :nocc, :nocc] += oo
    rdm1[:, nocc:, nocc:] += vv
    rdm1[:, :nocc, nocc:] += ov
    del (oo)
    del (vv)
    del (ov)
    del (L2_isb)

    ki, ka = np.indices((nkpts, nkpts))
    kj = kconserv_s[ki, ka]
    t1_ccee_ijas = t1_ccee_ijs_b[idx0, kj]
    t1_ccee_ijsa = t1_ccee_ijs[idx0, kj]
# block- ij
    rdm1[:, :nocc, :nocc] -= einsum('a,b,KkIiac,KkJicb->KIJ', L1, R1,
                                    t1_ccee_ijs, t1_ccee_ijs_b.conj(), optimize=True)
    rdm1[:, :nocc, :nocc] -= einsum('a,b,KkIica,KkJibc->KIJ', L1, R1,
                                    t1_ccee_ijs_b, t1_ccee_ijs.conj(), optimize=True)
    rdm1[:, :nocc, :nocc] += 2 * einsum('a,b,KkIica,KkJicb->KIJ', L1, R1,
                                        t1_ccee_ijs_b, t1_ccee_ijs_b.conj(), optimize=True)
# block- ab
    rdm1[kshift, nocc:, nocc:] += 1 / 2 * einsum('A,a,kKijBb,kKijba->AB', L1, R1,
                                                 t1_ccee_ijs, t1_ccee_ijs_b.conj(), optimize=True)
    rdm1[kshift, nocc:, nocc:] += 1 / 2 * einsum('a,B,kKijab,kKijbA->AB', L1, R1,
                                                 t1_ccee_ijs, t1_ccee_ijs_b.conj(), optimize=True)
    rdm1[:, nocc:, nocc:] -= 2 * einsum('a,b,kKijBa,kKijAb->KAB', L1, R1,
                                        t1_ccee_ijas, t1_ccee_ijas.conj(), optimize=True)
    rdm1[:, nocc:, nocc:] += einsum('a,b,kKijBa,kKijbA->KAB', L1, R1,
                                    t1_ccee_ijas, t1_ccee_ijsa.conj(), optimize=True)
    del (t1_ccee_ijas)
    del (t1_ccee_ijsa)
    del (t1_ccee_ijs)
    del (t1_ccee_ijs_b)

# block- ai
    rdm1[:, nocc:, :nocc] = rdm1[:, :nocc, nocc:].conj().transpose(0, 2, 1)

    if adc.approx_trans_moments is False or adc.method == "adc(3)":
        rdm1[kshift, :nocc, nocc:] -= einsum('a,A,Ia->IA', L1, R1, t2_ce[kshift], optimize=True)
        rdm1[:, :nocc, nocc:] += 2 * LR * t2_ce
        ### block- ai ###
        rdm1[:, nocc:, :nocc] = rdm1[:, :nocc, nocc:].conj().transpose(0, 2, 1)

    return rdm1


def get_properties(adc, kshift, U, nroots=1):

    # Transition moments
    T = adc.get_trans_moments(kshift)

    # Spectroscopic amplitudes
    U = adc.renormalize_eigenvectors(kshift, U, nroots)
    X = np.dot(T, U).reshape(-1, nroots)

    # Spectroscopic factors
    P = 2.0 * lib.einsum("pi,pi->i", X.conj(), X)
    P = P.real

    return P, X


def mask_frozen_ea(adc, v1, v2, kshift, const=LARGE_DENOM):
    '''Replaces all frozen orbital indices of `vector` with the value `const`.'''
    nkpts = adc.nkpts
    kconserv = adc.khelper.kconserv

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(adc)

    new_v1 = const * np.ones_like(v1)
    new_v2 = const * np.ones_like(v2)

    new_v1[nonzero_vpadding[kshift]] = v1[nonzero_vpadding[kshift]]

    for ki in range(nkpts):
        for ka in range(nkpts):
            kb = kconserv[kshift, ka, ki]
            idx = np.ix_([ki], [ka], nonzero_opadding[ki], nonzero_vpadding[ka], nonzero_vpadding[kb])
            new_v2[idx] = v2[idx]

    return (new_v1, new_v2)


class RADCEA(kadc_rhf.RADC):
    '''restricted ADC for EA energies and spectroscopic amplitudes

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`
        incore_complete : bool
            Avoid all I/O. Default is False.
        method : string
            nth-order ADC method. Options are : ADC(2), ADC(2)-X, ADC(3). Default is ADC(2).
        conv_tol : float
            Convergence threshold for Davidson iterations.  Default is 1e-12.
        max_cycle : int
            Number of Davidson iterations.  Default is 50.
        max_space : int
            Space size to hold trial vectors for Davidson iterative diagonalization.  Default is 12.

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested. Default value is 1.

            >>> myadc = adc.RADC(mf).run()
            >>> myadcip = adc.RADC(myadc).run()

    Saved results

        e_ea : float or list of floats
            EA energy (eigenvalue). For nroots = 1, it is a single float number. If nroots > 1,
            it is a list of floats for the lowest nroots eigenvalues.
        v_ea : array
            Eigenvectors for each EA transition.
        p_ea : float
            Spectroscopic amplitudes for each EA transition.
    '''

    _keys = {
        'tol_residual', 'conv_tol', 'e_corr', 'method', 'mo_coeff', 'mo_energy_b',
        't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle',
        'kpts', 'exxdiv', 'khelper', 'cell', 'nkop_chk', 'kop_npick', 'chnk_size',
    }

    def __init__(self, adc):
        self.verbose = adc.verbose
        self.stdout = adc.stdout
        self.max_memory = adc.max_memory
        self.max_space = adc.max_space
        self.max_cycle = adc.max_cycle
        self.conv_tol = adc.conv_tol
        self.tol_residual = adc.tol_residual
        self.t1 = adc.t1
        self.t2 = adc.t2
        self.method = adc.method
        self.method_type = adc.method_type
        self._scf = adc._scf
        self.compute_properties = adc.compute_properties
        self.approx_trans_moments = adc.approx_trans_moments

        self.kpts = adc._scf.kpts
        self.exxdiv = adc.exxdiv
        self.verbose = adc.verbose
        self.max_memory = adc.max_memory
        self.method = adc.method

        self.khelper = adc.khelper
        self.cell = adc.cell
        self.mo_coeff = adc.mo_coeff
        self.mo_occ = adc.mo_occ
        self.frozen = adc.frozen

        self._nocc = adc._nocc
        self._nmo = adc._nmo
        self._nvir = adc._nvir
        self.nkop_chk = adc.nkop_chk
        self.kop_npick = adc.kop_npick

        self.t2 = adc.t2
        self.e_corr = adc.e_corr
        self.mo_energy = adc.mo_energy
        self.imds = adc.imds
        self.chnk_size = adc.chnk_size
        self.U = adc.U
        self.if_naf = adc.if_naf
        self.naux = adc.naux

    kernel = kadc_rhf.kernel
    get_imds = get_imds
    get_diag = get_diag
    matvec = matvec
    vector_size = vector_size
    get_trans_moments = get_trans_moments
    renormalize_eigenvectors = renormalize_eigenvectors
    get_properties = get_properties
    make_rdm1 = make_rdm1

    def get_init_guess(self, nroots=1, diag=None, ascending=True, type=None, ini=None, kshift=None):
        ncore = self.nocc
        nextern = self.nmo - ncore
        nkpts = self.nkpts
        n_singles = nextern
        n_doubles = nkpts * nkpts * ncore * nextern * nextern
        dim = n_singles + n_doubles
        if (type == "read"):
            logger.info(self, "obtain initial guess from input variable")
            g = ini.T
            if g.shape[0] != dim or g.shape[1] != nroots:
                raise ValueError(f"Shape of guess each k point should be ({dim},{nroots})")
        else:
            if diag is None:
                diag = self.get_diag()
            idx = None
            dtype = getattr(diag, 'dtype', np.complex128)
            g = np.zeros((diag.shape[0], nroots), dtype=dtype)
            if ascending:
                idx = np.argsort(diag)
            else:
                idx = np.argsort(diag)[::-1]
            guess = np.zeros((diag.shape[0], nroots), dtype=dtype)
            min_shape = min(diag.shape[0], nroots)
            guess[:min_shape, :min_shape] = np.identity(min_shape)
            g[idx] = guess.copy()
        if (self.frozen is not None or not np.all([x.shape[1] == self.nmo for x in self.mo_coeff])) \
                and (type == "read"):
            for p in range(g.shape[1]):
                singles = g[:n_singles, p]
                doubles = g[n_singles:, p].reshape(nkpts, nkpts, ncore, nextern, nextern)
                (singles, doubles) = mask_frozen_ea(self, singles, doubles, kshift, const=0.0)
                g[:, p] = np.hstack((singles, doubles.ravel()))

        guess = []
        for p in range(g.shape[1]):
            if (self.frozen is not None) or (not np.all([x.shape[1] == self.nmo for x in self.mo_coeff])) \
                    or (type == "read"):
                guess_norm = np.linalg.norm(g[:, p])
                guess_norm_tol = LOOSE_ZERO_TOL
                if guess_norm < guess_norm_tol:
                    raise ValueError('Guess vector for root %d with norm %.4g is below threshold %.4g.\n'
                                     'This could possibly be due to freezing orbitals.\n'
                                     'Check your guess vector to make sure it has sufficiently large norm.'
                                     % (p, guess_norm, guess_norm_tol))

            guess.append(g[:, p])
        return guess

    def gen_matvec(self, kshift, imds=None, eris=None):
        if imds is None:
            imds = self.get_imds(eris)
        diag = self.get_diag(kshift, imds, eris)
        matvec = self.matvec(kshift, imds, eris)
        return matvec, diag


def ea_contract_r_vvvv(adc, r2, vvvv, ka, kb, kc):

    nocc = r2.shape[0]
    nvir = r2.shape[1]
    nkpts = adc.nkpts
    kconserv = adc.khelper.kconserv

    kd = kconserv[ka, kc, kb]
    r2 = np.ascontiguousarray(r2.reshape(nocc, -1))
    r2_vvvv = np.zeros((nvir, nvir, nocc), dtype=r2.dtype)
    chnk_size = adc.chnk_size
    chnk_size = min(chnk_size, nvir)

    a = 0
    if isinstance(vvvv, np.ndarray):
        vv1 = vvvv[ka, kc]
        vv2 = vvvv[kb, kd]
        for p in range(0, nvir, chnk_size):
            vvvv_p = dfadc.get_vvvv_df(adc, vv1, vv2, p, chnk_size) / nkpts
            k = vvvv_p.shape[0]
            vvvv_p = vvvv_p.reshape(-1, nvir * nvir)
            r2_vvvv[a:a + k] += np.dot(vvvv_p.conj(), r2.T).reshape(-1, nvir, nocc)
            del vvvv_p
            a += k
    else:
        for p in range(0, nvir, chnk_size):
            vvvv_p = vvvv[ka, kb, kc][p:p + chnk_size].reshape(-1, nvir * nvir)
            k = vvvv_p.shape[0]
            r2_vvvv[a:a + k] += np.dot(vvvv_p.conj(), r2.T).reshape(-1, nvir, nocc)
            del vvvv_p
            a += k

    r2_vvvv = np.ascontiguousarray(r2_vvvv.transpose(2, 0, 1))

    return r2_vvvv
