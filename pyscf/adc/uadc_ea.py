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
# Author: Abdelrahman Ahmed <>
#         Samragni Banerjee <samragnibanerjee4@gmail.com>
#         James Serna <jamcar456@gmail.com>
#         Terrence Stahl <>
#         Ning-Yuan Chen <cny003@outlook.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

'''
Unrestricted algebraic diagrammatic construction
'''

import numpy as np
from pyscf import lib, symm, ao2mo
from pyscf.lib import logger
from pyscf.adc import uadc
from pyscf.adc import uadc_ao2mo
from pyscf.adc import radc_ao2mo
from pyscf.adc import dfadc
from pyscf.adc.uadc_amplitudes import contract_ladder
from pyscf.data.nist import HARTREE2EV


def get_imds(adc, eris=None):

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t1 = adc.t1
    t2 = adc.t2

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    if eris is None:
        eris = adc.transform_integrals()

    eris_ovvo = eris.ovvo
    eris_OVVO = eris.OVVO
    eris_ovVO = eris.ovVO
    eris_OVvo = eris.OVvo

    t1_1_a = t1_1_b = None
    if t1[2][0] is not None:
        t1_1_a = t1[2][0]
        t1_1_b = t1[2][1]
        f_ov_a, f_ov_b = adc.f_ov
    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]

    # a-b block
    # Zeroth-order terms

    M_ab_a = lib.einsum('ab,a->ab', idn_vir_a, e_vir_a)
    M_ab_b = lib.einsum('ab,a->ab', idn_vir_b, e_vir_b)

    # Second-order terms

    t2_1_a = t2[0][0][:]
    M_ab_a -= 0.5 *  lib.einsum('lmad,lbdm->ab',t2_1_a, eris_ovvo,optimize=True)
    M_ab_a -= 0.5 *  lib.einsum('lmbd,ladm->ab',t2_1_a,eris_ovvo,optimize=True)

    t2_1_b = t2[0][2][:]
    M_ab_b -= 0.5 *  lib.einsum('lmad,lbdm->ab',t2_1_b, eris_OVVO,optimize=True)
    M_ab_b -= 0.5 *  lib.einsum('lmbd,ladm->ab',t2_1_b, eris_OVVO,optimize=True)

    t2_1_ab = t2[0][1][:]
    M_ab_a -=    0.5 *    lib.einsum('lmad,lbdm->ab',t2_1_ab, eris_ovVO,optimize=True)
    M_ab_b -=    0.5 *    lib.einsum('mlda,mdbl->ab',t2_1_ab, eris_ovVO,optimize=True)
    M_ab_a -=    0.5 *    lib.einsum('lmbd,ladm->ab',t2_1_ab, eris_ovVO,optimize=True)
    M_ab_b -=    0.5 *    lib.einsum('mldb,mdal->ab',t2_1_ab, eris_ovVO,optimize=True)

    cput0 = log.timer_debug1("Completed M_ab second-order terms ADC(2) calculation", *cput0)

    if t1_1_a is not None:
        if eris.OVVV is None:
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            eris_OVVV = []
            for a, b in lib.prange(0, nocc_b, chnk_size):
                eris_OVVV.append(dfadc.get_ovvv_spin_df(adc, eris.LOV, eris.LVV, a, chnk_size))
            eris_OVVV = np.concatenate(eris_OVVV, axis=0)
        else:
            eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
        if eris.OVvv is None:
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            eris_OVvv = []
            for a, b in lib.prange(0, nocc_b, chnk_size):
                eris_OVvv.append(dfadc.get_ovvv_spin_df(adc, eris.LOV, eris.Lvv, a, chnk_size))
            eris_OVvv = np.concatenate(eris_OVvv, axis=0)
        else:
            eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
        if eris.ovVV is None:
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            eris_ovVV = []
            for a, b in lib.prange(0, nocc_a, chnk_size):
                eris_ovVV.append(dfadc.get_ovvv_spin_df(adc, eris.Lov, eris.LVV, a, chnk_size))
            eris_ovVV = np.concatenate(eris_ovVV, axis=0)
        else:
            eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
        if eris.ovvv is None:
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            eris_ovvv = []
            for a, b in lib.prange(0, nocc_a, chnk_size):
                eris_ovvv.append(dfadc.get_ovvv_spin_df(adc, eris.Lov, eris.Lvv, a, chnk_size))
            eris_ovvv = np.concatenate(eris_ovvv, axis=0)
        else:
            eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
        temp = lib.einsum('iA,iB->AB', f_ov_a, t1_1_a, optimize=True)
        M_ab_a -= temp + temp.T
        temp = lib.einsum('ia,iABa->AB', t1_1_a, eris_ovvv, optimize=True)
        M_ab_a -= temp + temp.T
        temp = lib.einsum('ia,iaAB->AB', t1_1_a, eris_ovvv, optimize=True)
        M_ab_a += temp + temp.T
        temp = lib.einsum('ia,iaAB->AB', t1_1_b, eris_OVvv, optimize=True)
        M_ab_a += temp + temp.T
        temp = lib.einsum('A,iA,iB->AB', e_vir_a, t1_1_a, t1_1_a, optimize=True)
        M_ab_a -= 1/2 * (temp + temp.T)
        M_ab_a += lib.einsum('i,iA,iB->AB', e_occ_a, t1_1_a, t1_1_a, optimize=True)
        temp = lib.einsum('iA,iB->AB', f_ov_b, t1_1_b, optimize=True)
        M_ab_b -= temp + temp.T
        temp = lib.einsum('ia,iaAB->AB', t1_1_a, eris_ovVV, optimize=True)
        M_ab_b += temp + temp.T
        temp = lib.einsum('ia,iABa->AB', t1_1_b, eris_OVVV, optimize=True)
        M_ab_b -= temp + temp.T
        temp = lib.einsum('ia,iaAB->AB', t1_1_b, eris_OVVV, optimize=True)
        M_ab_b += temp + temp.T
        temp = lib.einsum('A,iA,iB->AB', e_vir_b, t1_1_b, t1_1_b, optimize=True)
        M_ab_b -= 1/2 * (temp + temp.T)
        M_ab_b += lib.einsum('i,iA,iB->AB', e_occ_b, t1_1_b, t1_1_b, optimize=True)

    #Third-order terms
    if (method =='adc(3)'):

        t1_2_a, t1_2_b = t1[0]
        eris_oovv = eris.oovv
        eris_OOVV = eris.OOVV
        eris_OOvv = eris.OOvv
        eris_ooVV = eris.ooVV
        eris_ovvo = eris.ovvo
        eris_OVVO = eris.OVVO
        eris_OVvo = eris.OVvo
        eris_ovVO = eris.ovVO

        if t1_1_a is None:
            if eris.ovvv is None:
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                for a,b in lib.prange(0,nocc_a,chnk_size):
                    eris_ovvv = dfadc.get_ovvv_spin_df(
                        adc, eris.Lov, eris.Lvv, a, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
                    M_ab_a +=  2 * lib.einsum('ld,ldab->ab',t1_2_a[a:b], eris_ovvv,optimize=True)
                    M_ab_a -=  lib.einsum('ld,lbad->ab',t1_2_a[a:b], eris_ovvv,optimize=True)
                    M_ab_a -= lib.einsum('ld,ladb->ab',t1_2_a[a:b], eris_ovvv,optimize=True)
                    del eris_ovvv

            else :
                eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
                k = eris_ovvv.shape[0]
                M_ab_a +=  2 * lib.einsum('ld,ldab->ab',t1_2_a, eris_ovvv,optimize=True)
                M_ab_a -=  lib.einsum('ld,lbad->ab',t1_2_a, eris_ovvv,optimize=True)
                M_ab_a -= lib.einsum('ld,ladb->ab',t1_2_a, eris_ovvv,optimize=True)
                del eris_ovvv

            if eris.OVvv is None:
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                for a,b in lib.prange(0,nocc_b,chnk_size):
                    eris_OVvv = dfadc.get_ovvv_spin_df(
                        adc, eris.LOV, eris.Lvv, a, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
                    M_ab_a +=  2 * lib.einsum('ld,ldab->ab',t1_2_b[a:b], eris_OVvv,optimize=True)
                    del eris_OVvv
            else :
                eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
                M_ab_a += 2 * lib.einsum('ld,ldab->ab',t1_2_b, eris_OVvv,optimize=True)
                del eris_OVvv

            if eris.OVVV is None:
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                for a,b in lib.prange(0,nocc_b,chnk_size):
                    eris_OVVV = dfadc.get_ovvv_spin_df(
                        adc, eris.LOV, eris.LVV, a, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
                    M_ab_b +=  2 * lib.einsum('ld,ldab->ab',t1_2_b[a:b], eris_OVVV,optimize=True)
                    M_ab_b -=  lib.einsum('ld,lbad->ab',t1_2_b[a:b], eris_OVVV,optimize=True)
                    M_ab_b -= lib.einsum('ld,ladb->ab',t1_2_b[a:b], eris_OVVV,optimize=True)
                    del eris_OVVV
            else :
                eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
                M_ab_b += 2 * lib.einsum('ld,ldab->ab',t1_2_b, eris_OVVV,optimize=True)
                M_ab_b -=  lib.einsum('ld,lbad->ab',t1_2_b, eris_OVVV,optimize=True)
                M_ab_b -= lib.einsum('ld,ladb->ab',t1_2_b, eris_OVVV,optimize=True)
                del eris_OVVV

            if eris.ovVV is None:
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                for a,b in lib.prange(0,nocc_a,chnk_size):
                    eris_ovVV = dfadc.get_ovvv_spin_df(
                        adc, eris.Lov, eris.LVV, a, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
                    M_ab_b +=  2 * lib.einsum('ld,ldab->ab',t1_2_a[a:b], eris_ovVV,optimize=True)
                    del eris_ovVV
            else :
                eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
                k = eris_ovVV.shape[0]
                M_ab_b += 2 * lib.einsum('ld,ldab->ab',t1_2_a, eris_ovVV,optimize=True)
                del eris_ovVV

            cput0 = log.timer_debug1("Completed M_ab ovvv ADC(3) calculation", *cput0)

            t2_2_a = t2[1][0][:]
            M_ab_a -= 0.5 *  lib.einsum('lmad,lbdm->ab',t2_2_a, eris_ovvo,optimize=True)
            M_ab_a -= 0.5 *  lib.einsum('lmbd,ladm->ab',t2_2_a,eris_ovvo,optimize=True)

            t2_2_b = t2[1][2][:]
            M_ab_b -= 0.5 *  lib.einsum('lmad,lbdm->ab',t2_2_b, eris_OVVO,optimize=True)
            M_ab_b -= 0.5 *  lib.einsum('lmbd,ladm->ab',t2_2_b, eris_OVVO,optimize=True)

            t2_2_ab = t2[1][1][:]
            M_ab_a -=  0.5 *      lib.einsum('lmad,lbdm->ab',t2_2_ab, eris_ovVO,optimize=True)
            M_ab_b -=  0.5 *      lib.einsum('mlda,mdbl->ab',t2_2_ab, eris_ovVO,optimize=True)
            M_ab_a -=  0.5 *      lib.einsum('lmbd,ladm->ab',t2_2_ab, eris_ovVO,optimize=True)
            M_ab_b -=  0.5 *      lib.einsum('mldb,mdal->ab',t2_2_ab, eris_ovVO,optimize=True)

            t2_1_a = t2[0][0][:]
            t2_1_ab = t2[0][1][:]

            M_ab_a -= 0.5 * lib.einsum('lnde,mlbd,neam->ab',t2_1_ab, t2_1_a, eris_OVvo, optimize=True)
            M_ab_a += 0.5 * lib.einsum('lned,lmbd,nmae->ab',t2_1_ab, t2_1_ab, eris_OOvv, optimize=True)

            M_ab_t =  lib.einsum('lned,mlbd->nemb', t2_1_a,t2_1_a, optimize=True)
            M_ab_a -= 0.5 * lib.einsum('nemb,nmae->ab',M_ab_t, eris_oovv, optimize=True)
            M_ab_a += 0.5 * lib.einsum('nemb,maen->ab',M_ab_t, eris_ovvo, optimize=True)
            M_ab_a -= 0.5 * lib.einsum('name,nmeb->ab',M_ab_t, eris_oovv, optimize=True)
            M_ab_a += 0.5 * lib.einsum('name,nbem->ab',M_ab_t, eris_ovvo, optimize=True)
            del M_ab_t

            M_ab_t = lib.einsum('nled,mlbd->nemb', t2_1_ab,t2_1_ab, optimize=True)
            M_ab_a += 0.5 * lib.einsum('nemb,nmae->ab',M_ab_t, eris_oovv, optimize=True)
            M_ab_a -= 0.5 * lib.einsum('nemb,maen->ab',M_ab_t, eris_ovvo, optimize=True)
            del M_ab_t

            M_ab_t = lib.einsum('lnde,lmdb->nemb', t2_1_ab,t2_1_ab, optimize=True)
            M_ab_b += 0.5 * lib.einsum('nemb,nmae->ab',M_ab_t, eris_OOVV, optimize=True)
            M_ab_b -= 0.5 * lib.einsum('nemb,maen->ab',M_ab_t, eris_OVVO, optimize=True)
            del M_ab_t

            M_ab_b += 0.5 * lib.einsum('lned,lmdb,neam->ab',t2_1_a, t2_1_ab, eris_ovVO, optimize=True)
            M_ab_b += 0.5 * lib.einsum('nlde,mldb,nmae->ab',t2_1_ab, t2_1_ab, eris_ooVV, optimize=True)

            M_ab_a += 0.5 * lib.einsum('mled,nlad,nmeb->ab',t2_1_ab, t2_1_ab, eris_oovv, optimize=True)
            M_ab_a -= 0.5 * lib.einsum('mled,nlad,nbem->ab',t2_1_ab, t2_1_ab, eris_ovvo, optimize=True)
            M_ab_a += 0.5 * lib.einsum('lmed,lnad,nmeb->ab',t2_1_ab, t2_1_ab, eris_OOvv, optimize=True)
            M_ab_a += 0.5 * lib.einsum('lmde,lnad,nbem->ab',t2_1_ab, t2_1_a, eris_ovVO, optimize=True)

            M_ab_b += 0.5 * lib.einsum('lmde,lnda,nmeb->ab',t2_1_ab, t2_1_ab, eris_OOVV, optimize=True)
            M_ab_b -= 0.5 * lib.einsum('lmde,lnda,nbem->ab',t2_1_ab, t2_1_ab, eris_OVVO, optimize=True)
            M_ab_b += 0.5 * lib.einsum('mlde,nlda,nmeb->ab',t2_1_ab, t2_1_ab, eris_ooVV, optimize=True)
            M_ab_b -= 0.5 * lib.einsum('mled,lnda,nbem->ab',t2_1_a, t2_1_ab, eris_OVvo, optimize=True)

            M_ab_a +=  0.5*lib.einsum('lned,mled,nmab->ab',t2_1_a, t2_1_a, eris_oovv, optimize=True)
            M_ab_a -=  0.5*lib.einsum('lned,mled,nbam->ab',t2_1_a, t2_1_a, eris_ovvo, optimize=True)
            M_ab_a -=  lib.einsum('nled,mled,nmab->ab',t2_1_ab, t2_1_ab, eris_oovv, optimize=True)
            M_ab_a +=  lib.einsum('nled,mled,nbam->ab',t2_1_ab, t2_1_ab, eris_ovvo, optimize=True)

            M_ab_a -=  lib.einsum('lned,lmed,nmab->ab',t2_1_ab, t2_1_ab, eris_OOvv, optimize=True)
            M_ab_b -=  lib.einsum('lned,lmed,nmab->ab',t2_1_ab, t2_1_ab, eris_OOVV, optimize=True)
            M_ab_b +=  lib.einsum('lned,lmed,nbam->ab',t2_1_ab, t2_1_ab, eris_OVVO, optimize=True)
            M_ab_b +=  0.5*lib.einsum('lned,mled,nmab->ab',t2_1_a, t2_1_a, eris_ooVV, optimize=True)
            M_ab_b -=  lib.einsum('nled,mled,nmab->ab',t2_1_ab, t2_1_ab, eris_ooVV, optimize=True)

            t2_1_b = t2[0][2][:]
            M_ab_a += 0.5 * lib.einsum('lned,mlbd,neam->ab',t2_1_b, t2_1_ab, eris_OVvo, optimize=True)

            M_ab_t = lib.einsum('lned,mlbd->nemb', t2_1_b,t2_1_b, optimize=True)
            M_ab_b -= 0.5 * lib.einsum('nemb,nmae->ab',M_ab_t, eris_OOVV, optimize=True)
            M_ab_b += 0.5 * lib.einsum('nemb,maen->ab',M_ab_t, eris_OVVO, optimize=True)
            M_ab_b -= 0.5 * lib.einsum('name,nmeb->ab',M_ab_t, eris_OOVV, optimize=True)
            M_ab_b += 0.5 * lib.einsum('name,nbem->ab',M_ab_t, eris_OVVO, optimize=True)
            del M_ab_t

            M_ab_b -= 0.5 * lib.einsum('nled,mlbd,neam->ab',t2_1_ab, t2_1_b, eris_ovVO, optimize=True)
            M_ab_a -= 0.5 * lib.einsum('mled,nlad,nbem->ab',t2_1_b, t2_1_ab, eris_ovVO, optimize=True)
            M_ab_b += 0.5 * lib.einsum('mled,lnad,nbem->ab',t2_1_ab, t2_1_b, eris_OVvo, optimize=True)

            M_ab_a += 0.5 * lib.einsum('lned,mled,nmab->ab',t2_1_b, t2_1_b, eris_OOvv, optimize=True)
            M_ab_b += 0.5 * lib.einsum('lned,mled,nmab->ab',t2_1_b, t2_1_b, eris_OOVV, optimize=True)
            M_ab_b -= 0.5 * lib.einsum('lned,mled,nbam->ab',t2_1_b, t2_1_b, eris_OVVO, optimize=True)

            log.timer_debug1("Completed M_ab ADC(3) small integrals calculation")

            t2_1_a = t2[0][0][:]
            t2_1_ab = t2[0][1][:]

            if isinstance(eris.vvvv_p,np.ndarray):
                eris_vvvv = radc_ao2mo.unpack_eri_2(eris.vvvv_p, nvir_a)
                M_ab_a -= 0.5 * 0.25*lib.einsum('mlef,mlbd,adef->ab',
                                                t2_1_a, t2_1_a, eris_vvvv, optimize=True)
                M_ab_a -= 0.5*lib.einsum('mldf,mled,aebf->ab',t2_1_a, t2_1_a, eris_vvvv, optimize=True)
                M_ab_a += lib.einsum('mlfd,mled,aebf->ab',t2_1_ab, t2_1_ab, eris_vvvv, optimize=True)
                del eris_vvvv

                M_ab_a -= 2 * 0.5 * 0.25*lib.einsum('mlaf,mlbf->ab',
                                                    t2_1_a, adc.imds.t2_1_vvvv[0], optimize=True)
            else:
                M_ab_a -= 2*0.5*0.25*lib.einsum('mlad,mlbd->ab',
                                                adc.imds.t2_1_vvvv[0], t2_1_a, optimize=True)
                M_ab_a -= 2*0.5*0.25*lib.einsum('mlaf,mlbf->ab', t2_1_a,
                                                adc.imds.t2_1_vvvv[0], optimize=True)

            if isinstance(eris.vvvv_p, list):

                a = 0
                temp = np.zeros((nvir_a,nvir_a))
                for dataset in eris.vvvv_p:
                    k = dataset.shape[0]
                    vvvv = dataset[:]
                    eris_vvvv = np.zeros((k,nvir_a,nvir_a,nvir_a))
                    eris_vvvv[:,:,ab_ind_a[0],ab_ind_a[1]] = vvvv
                    eris_vvvv[:,:,ab_ind_a[1],ab_ind_a[0]] = -vvvv

                    temp[a:a+k]  -= 0.5*lib.einsum('mldf,mled,aebf->ab',
                                                   t2_1_a, t2_1_a,  eris_vvvv, optimize=True)
                    temp[a:a+k] += lib.einsum('mlfd,mled,aebf->ab',t2_1_ab,
                                              t2_1_ab, eris_vvvv, optimize=True)
                    del eris_vvvv
                    a += k
                M_ab_a  += temp

                a = 0
                temp = np.zeros((nvir_b,nvir_b))
                for dataset in eris.VvVv_p:
                    k = dataset.shape[0]
                    eris_VvVv = dataset[:].reshape(-1,nvir_a,nvir_b,nvir_a)
                    temp[a:a+k] -= 0.5*lib.einsum('mldf,mled,aebf->ab',
                                                  t2_1_a, t2_1_a, eris_VvVv, optimize=True)
                    temp[a:a+k] += lib.einsum('mlfd,mled,aebf->ab',t2_1_ab,
                                              t2_1_ab, eris_VvVv, optimize=True)
                    a += k
                M_ab_b  += temp

            elif eris.vvvv_p is None:

                temp = np.zeros((nvir_a,nvir_a))
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                for a,b in lib.prange(0,nvir_a,chnk_size):
                    vvvv = dfadc.get_vvvv_antisym_df(adc, eris.Lvv, a, chnk_size)

                    eris_vvvv = np.zeros((vvvv.shape[0],nvir_a,nvir_a,nvir_a))
                    eris_vvvv[:,:,ab_ind_a[0],ab_ind_a[1]] = vvvv
                    eris_vvvv[:,:,ab_ind_a[1],ab_ind_a[0]] = -vvvv

                    temp[a:b]  -= 0.5*lib.einsum('mldf,mled,aebf->ab',
                                              t2_1_a, t2_1_a,  eris_vvvv, optimize=True)
                    temp[a:b] += lib.einsum('mlfd,mled,aebf->ab',t2_1_ab,
                                              t2_1_ab, eris_vvvv, optimize=True)
                    del eris_vvvv
                M_ab_a  += temp
                del temp

                temp = np.zeros((nvir_b,nvir_b))
                for a,b in lib.prange(0,nvir_b,chnk_size):
                    eris_VvVv = dfadc.get_vVvV_df(adc, eris.LVV, eris.Lvv, a, chnk_size)

                    temp[a:b] -= 0.5*lib.einsum('mldf,mled,aebf->ab',
                                                  t2_1_a, t2_1_a, eris_VvVv, optimize=True)
                    temp[a:b] += lib.einsum('mlfd,mled,aebf->ab',t2_1_ab,
                                              t2_1_ab, eris_VvVv, optimize=True)
                M_ab_b  += temp
                del temp

            t2_1_b = t2[0][2][:]
            if isinstance(eris.vVvV_p,np.ndarray):

                eris_vVvV = eris.vVvV_p
                eris_vVvV = eris_vVvV.reshape(nvir_a,nvir_b,nvir_a,nvir_b)
                M_ab_a -= 0.5*lib.einsum('mlef,mlbd,adef->ab',t2_1_ab,
                                         t2_1_ab,   eris_vVvV, optimize=True)
                M_ab_a -= 0.5*lib.einsum('mldf,mled,aebf->ab',t2_1_b, t2_1_b, eris_vVvV, optimize=True)
                M_ab_a += lib.einsum('mldf,mlde,aebf->ab',t2_1_ab, t2_1_ab,   eris_vVvV, optimize=True)

                M_ab_b -= 0.5*lib.einsum('mlef,mldb,daef->ab',t2_1_ab,
                                         t2_1_ab,   eris_vVvV, optimize=True)
                M_ab_b -= 0.5*lib.einsum('mldf,mled,eafb->ab',t2_1_a, t2_1_a, eris_vVvV, optimize=True)
                M_ab_b += lib.einsum('mlfd,mled,eafb->ab',t2_1_ab, t2_1_ab,   eris_vVvV, optimize=True)

                eris_vVvV = eris_vVvV.reshape(nvir_a*nvir_b,nvir_a*nvir_b)

                M_ab_a -= 0.5*lib.einsum('mlaf,mlbf->ab',t2_1_ab, adc.imds.t2_1_vvvv[1], optimize=True)
                M_ab_b -= 0.5*lib.einsum('mlfa,mlfb->ab',t2_1_ab, adc.imds.t2_1_vvvv[1], optimize=True)

            else:
                M_ab_a -= 0.5 * lib.einsum('mlad,mlbd->ab',
                                           adc.imds.t2_1_vvvv[1], t2_1_ab, optimize=True)
                M_ab_b -= 0.5 * lib.einsum('mlda,mldb->ab',
                                           adc.imds.t2_1_vvvv[1], t2_1_ab, optimize=True)
                M_ab_a -= 0.5 * lib.einsum('mlaf,mlbf->ab', t2_1_ab,
                                           adc.imds.t2_1_vvvv[1], optimize=True)
                M_ab_b -= 0.5 * lib.einsum('mlfa,mlfb->ab', t2_1_ab,
                                           adc.imds.t2_1_vvvv[1], optimize=True)

            if isinstance(eris.VVVV_p,np.ndarray):
                eris_VVVV = radc_ao2mo.unpack_eri_2(eris.VVVV_p, nvir_b)
                M_ab_b -= 0.5*0.25*lib.einsum('mlef,mlbd,adef->ab',t2_1_b,
                                              t2_1_b, eris_VVVV, optimize=True)
                M_ab_b -= 0.5*lib.einsum('mldf,mled,aebf->ab',t2_1_b, t2_1_b, eris_VVVV, optimize=True)
                M_ab_b += lib.einsum('mldf,mlde,aebf->ab',t2_1_ab, t2_1_ab, eris_VVVV, optimize=True)
                del eris_VVVV

                M_ab_b -= 2 * 0.5 * 0.25*lib.einsum('mlaf,mlbf->ab',
                                                    t2_1_b, adc.imds.t2_1_vvvv[2], optimize=True)
            else:
                M_ab_b -= 2 * 0.5 * 0.25*lib.einsum('mlad,mlbd->ab',
                                                    adc.imds.t2_1_vvvv[2], t2_1_b, optimize=True)
                M_ab_b -= 2 * 0.5 * 0.25*lib.einsum('mlaf,mlbf->ab',
                                                    t2_1_b, adc.imds.t2_1_vvvv[2], optimize=True)

            if isinstance(eris.vvvv_p, list):

                a = 0
                temp = np.zeros((nvir_b,nvir_b))
                for dataset in eris.VVVV_p:
                    k = dataset.shape[0]
                    VVVV = dataset[:]
                    eris_VVVV = np.zeros((k,nvir_b,nvir_b,nvir_b))
                    eris_VVVV[:,:,ab_ind_b[0],ab_ind_b[1]] = VVVV
                    eris_VVVV[:,:,ab_ind_b[1],ab_ind_b[0]] = -VVVV

                    temp[a:a+k]  -= 0.5*lib.einsum('mldf,mled,aebf->ab',
                                                   t2_1_b, t2_1_b,  eris_VVVV, optimize=True)
                    temp[a:a+k]  += lib.einsum('mldf,mlde,aebf->ab',t2_1_ab,
                                               t2_1_ab, eris_VVVV, optimize=True)
                    del eris_VVVV
                    a += k
                M_ab_b  += temp

                a = 0
                temp = np.zeros((nvir_a,nvir_a))
                for dataset in eris.vVvV_p:
                    k = dataset.shape[0]
                    eris_vVvV = dataset[:].reshape(-1,nvir_b,nvir_a,nvir_b)
                    temp[a:a+k] -= 0.5*lib.einsum('mldf,mled,aebf->ab',
                                                  t2_1_b, t2_1_b, eris_vVvV, optimize=True)
                    temp[a:a+k] += lib.einsum('mldf,mlde,aebf->ab',t2_1_ab,
                                              t2_1_ab, eris_vVvV, optimize=True)
                    a += k
                M_ab_a  += temp

            elif eris.vvvv_p is None:

                temp = np.zeros((nvir_b,nvir_b))
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                for a,b in lib.prange(0,nvir_b,chnk_size):
                    VVVV = dfadc.get_vvvv_antisym_df(adc, eris.LVV, a, chnk_size)

                    eris_VVVV = np.zeros((VVVV.shape[0],nvir_b,nvir_b,nvir_b))
                    eris_VVVV[:,:,ab_ind_b[0],ab_ind_b[1]] = VVVV
                    eris_VVVV[:,:,ab_ind_b[1],ab_ind_b[0]] = -VVVV

                    temp[a:b]  -= 0.5*lib.einsum('mldf,mled,aebf->ab',
                                                   t2_1_b, t2_1_b,  eris_VVVV, optimize=True)
                    temp[a:b]  += lib.einsum('mldf,mlde,aebf->ab',t2_1_ab,
                                               t2_1_ab, eris_VVVV, optimize=True)
                    del eris_VVVV
                M_ab_b  += temp
                del temp

                temp = np.zeros((nvir_a,nvir_a))
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                for a,b in lib.prange(0,nvir_a,chnk_size):
                    eris_vVvV = dfadc.get_vVvV_df(adc, eris.Lvv, eris.LVV, a, chnk_size)

                    temp[a:b] -= 0.5*lib.einsum('mldf,mled,aebf->ab',
                                                  t2_1_b, t2_1_b, eris_vVvV, optimize=True)
                    temp[a:b] += lib.einsum('mldf,mlde,aebf->ab',t2_1_ab,
                                              t2_1_ab, eris_vVvV, optimize=True)
                M_ab_a  += temp
                del temp

        else:
            t2_1_a = t2[0][0][:]
            t2_1_ab = t2[0][1][:]
            t2_1_b = t2[0][2][:]
            t2_2_a = t2[1][0][:]
            t2_2_ab = t2[1][1][:]
            t2_2_b = t2[1][2][:]
            eris_OOOO = eris.OOOO
            eris_OVOO = eris.OVOO
            eris_OVoo = eris.OVoo
            eris_ooOO = eris.ooOO
            eris_oooo = eris.oooo
            eris_ovOO = eris.ovOO
            eris_ovoo = eris.ovoo
            if eris.OVVV is None:
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                eris_OVVV = []
                for a, b in lib.prange(0, nocc_b, chnk_size):
                    eris_OVVV.append(dfadc.get_ovvv_spin_df(adc, eris.LOV, eris.LVV, a, chnk_size))
                eris_OVVV = np.concatenate(eris_OVVV, axis=0)
            else:
                eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
            if eris.OVvv is None:
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                eris_OVvv = []
                for a, b in lib.prange(0, nocc_b, chnk_size):
                    eris_OVvv.append(dfadc.get_ovvv_spin_df(adc, eris.LOV, eris.Lvv, a, chnk_size))
                eris_OVvv = np.concatenate(eris_OVvv, axis=0)
            else:
                eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
            if eris.ovVV is None:
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                eris_ovVV = []
                for a, b in lib.prange(0, nocc_a, chnk_size):
                    eris_ovVV.append(dfadc.get_ovvv_spin_df(adc, eris.Lov, eris.LVV, a, chnk_size))
                eris_ovVV = np.concatenate(eris_ovVV, axis=0)
            else:
                eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
            if eris.ovvv is None:
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                eris_ovvv = []
                for a, b in lib.prange(0, nocc_a, chnk_size):
                    eris_ovvv.append(dfadc.get_ovvv_spin_df(adc, eris.Lov, eris.Lvv, a, chnk_size))
                eris_ovvv = np.concatenate(eris_ovvv, axis=0)
            else:
                eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
            temp = lib.einsum('iA,iB->AB', f_ov_a, t1_2_a, optimize=True)
            M_ab_a -= temp + temp.T
            temp = lib.einsum('ia,iABa->AB', t1_2_a, eris_ovvv, optimize=True)
            M_ab_a -= temp + temp.T
            temp = lib.einsum('ia,iaAB->AB', t1_2_a, eris_ovvv, optimize=True)
            M_ab_a += temp + temp.T
            temp = lib.einsum('ia,iaAB->AB', t1_2_b, eris_OVvv, optimize=True)
            M_ab_a += temp + temp.T
            temp = lib.einsum('ijAa,iBaj->AB', t2_2_a, eris_ovvo, optimize=True)
            M_ab_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ijAa,jBai->AB', t2_2_a, eris_ovvo, optimize=True)
            M_ab_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('ijAa,iBaj->AB', t2_2_ab, eris_ovVO, optimize=True)
            M_ab_a -= temp + temp.T
            temp = lib.einsum('A,iB,iA->AB', e_vir_a, t1_1_a, t1_2_a, optimize=True)
            M_ab_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('A,iA,iB->AB', e_vir_a, t1_1_a, t1_2_a, optimize=True)
            M_ab_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('i,iA,iB->AB', e_occ_a, t1_1_a, t1_2_a, optimize=True)
            M_ab_a += temp + temp.T
            temp = lib.einsum('A,ijBa,ijAa->AB', e_vir_a, t2_1_a, t2_2_a, optimize=True)
            M_ab_a -= 1/4 * (temp + temp.T)
            temp = lib.einsum('A,ijAa,ijBa->AB', e_vir_a, t2_1_a, t2_2_a, optimize=True)
            M_ab_a -= 1/4 * (temp + temp.T)
            temp = lib.einsum('a,ijAa,ijBa->AB', e_vir_a, t2_1_a, t2_2_a, optimize=True)
            M_ab_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('i,ijAa,ijBa->AB', e_occ_a, t2_1_a, t2_2_a, optimize=True)
            M_ab_a += temp + temp.T
            temp = lib.einsum('A,ijBa,ijAa->AB', e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
            M_ab_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('A,ijAa,ijBa->AB', e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
            M_ab_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('a,ijAa,ijBa->AB', e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
            M_ab_a -= temp + temp.T
            temp = lib.einsum('i,ijAa,ijBa->AB', e_occ_a, t2_1_ab, t2_2_ab, optimize=True)
            M_ab_a += temp + temp.T
            temp = lib.einsum('i,jiAa,jiBa->AB', e_occ_b, t2_1_ab, t2_2_ab, optimize=True)
            M_ab_a += temp + temp.T
            temp = lib.einsum('iA,ja,ijBa->AB', f_ov_a, t1_1_a, t2_1_a, optimize=True)
            M_ab_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,jA,ijBa->AB', f_ov_a, t1_1_a, t2_1_a, optimize=True)
            M_ab_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,jA,jiBa->AB', f_ov_b, t1_1_a, t2_1_ab, optimize=True)
            M_ab_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('iA,ja,ijBa->AB', f_ov_a, t1_1_b, t2_1_ab, optimize=True)
            M_ab_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('iA,ja,iBaj->AB', t1_1_a, t1_1_b, eris_ovVO, optimize=True)
            M_ab_a -= 2 * (temp + temp.T)
            temp = lib.einsum('iA,ijab,jaBb->AB', t1_1_a, t2_1_a, eris_ovvv, optimize=True)
            M_ab_a += 1/4 * (temp + temp.T)
            temp = lib.einsum('iA,ijab,jbBa->AB', t1_1_a, t2_1_a, eris_ovvv, optimize=True)
            M_ab_a -= 1/4 * (temp + temp.T)
            temp = lib.einsum('iA,jkBa,kaji->AB', t1_1_a, t2_1_a, eris_ovoo, optimize=True)
            M_ab_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('iA,jkBa,jaki->AB', t1_1_a, t2_1_a, eris_ovoo, optimize=True)
            M_ab_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,ijAb,jBab->AB', t1_1_a, t2_1_a, eris_ovvv, optimize=True)
            M_ab_a += temp + temp.T
            temp = lib.einsum('ia,ijAb,jbaB->AB', t1_1_a, t2_1_a, eris_ovvv, optimize=True)
            M_ab_a -= temp + temp.T
            temp = lib.einsum('ia,ijab,jABb->AB', t1_1_a, t2_1_a, eris_ovvv, optimize=True)
            M_ab_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,ijab,jbAB->AB', t1_1_a, t2_1_a, eris_ovvv, optimize=True)
            M_ab_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,jkAa,kBji->AB', t1_1_a, t2_1_a, eris_ovoo, optimize=True)
            M_ab_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,jkAa,jBki->AB', t1_1_a, t2_1_a, eris_ovoo, optimize=True)
            M_ab_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('iA,ijab,jbBa->AB', t1_1_a, t2_1_ab, eris_OVvv, optimize=True)
            M_ab_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('iA,jkBa,kaji->AB', t1_1_a, t2_1_ab, eris_OVoo, optimize=True)
            M_ab_a += temp + temp.T
            temp = lib.einsum('ia,ijAb,jbaB->AB', t1_1_a, t2_1_ab, eris_OVvv, optimize=True)
            M_ab_a -= temp + temp.T
            temp = lib.einsum('ia,ijab,jbAB->AB', t1_1_a, t2_1_ab, eris_OVvv, optimize=True)
            M_ab_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,jiAb,jBab->AB', t1_1_b, t2_1_ab, eris_ovVV, optimize=True)
            M_ab_a -= temp + temp.T
            temp = lib.einsum('ia,jiba,jABb->AB', t1_1_b, t2_1_ab, eris_ovvv, optimize=True)
            M_ab_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,jiba,jbAB->AB', t1_1_b, t2_1_ab, eris_ovvv, optimize=True)
            M_ab_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,jkAa,jBki->AB', t1_1_b, t2_1_ab, eris_ovOO, optimize=True)
            M_ab_a += temp + temp.T
            temp = lib.einsum('ia,ijab,jbAB->AB', t1_1_b, t2_1_b, eris_OVvv, optimize=True)
            M_ab_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('ijAa,ikBb,jabk->AB', t2_1_a, t2_1_ab, eris_ovVO, optimize=True)
            M_ab_a -= temp + temp.T
            temp = lib.einsum('ijAa,ikab,jBbk->AB', t2_1_a, t2_1_ab, eris_ovVO, optimize=True)
            M_ab_a += temp + temp.T
            temp = lib.einsum('ijAa,jkab,iBbk->AB', t2_1_ab, t2_1_b, eris_ovVO, optimize=True)
            M_ab_a -= temp + temp.T
            M_ab_a += lib.einsum('iABj,ja,ia->AB', eris_ovvo, t1_1_a, t1_1_a, optimize=True)
            temp = lib.einsum('iAaj,ja,iB->AB', eris_ovvo, t1_1_a, t1_1_a, optimize=True)
            M_ab_a -= 2 * (temp + temp.T)
            temp = lib.einsum('iAaj,ia,jB->AB', eris_ovvo, t1_1_a, t1_1_a, optimize=True)
            M_ab_a += temp + temp.T
            M_ab_a -= lib.einsum('ijAB,ia,ja->AB', eris_oovv, t1_1_a, t1_1_a, optimize=True)
            temp = lib.einsum('ijAa,ia,jB->AB', eris_oovv, t1_1_a, t1_1_a, optimize=True)
            M_ab_a += temp + temp.T
            M_ab_a -= lib.einsum('ijAB,ia,ja->AB', eris_OOvv, t1_1_b, t1_1_b, optimize=True)
            M_ab_a += 1/2 *  lib.einsum('iABj,jkab,ikab->AB', eris_ovvo, t2_1_a, t2_1_a, optimize=True)
            temp = lib.einsum('iAaj,jkab,ikBb->AB', eris_ovvo, t2_1_a, t2_1_a, optimize=True)
            M_ab_a -= temp + temp.T
            M_ab_a -= lib.einsum('iabj,ikAa,jkBb->AB', eris_ovvo, t2_1_a, t2_1_a, optimize=True)
            M_ab_a -= 1/2 *  lib.einsum('ijAB,ikab,jkab->AB', eris_oovv, t2_1_a, t2_1_a, optimize=True)
            temp = lib.einsum('ijAa,ikab,jkBb->AB', eris_oovv, t2_1_a, t2_1_a, optimize=True)
            M_ab_a += temp + temp.T
            M_ab_a += lib.einsum('ijab,ikAb,jkBa->AB', eris_oovv, t2_1_a, t2_1_a, optimize=True)
            temp = lib.einsum('ijkl,ikAa,jlBa->AB', eris_oooo, t2_1_a, t2_1_a, optimize=True)
            M_ab_a -= 1/4 * (temp + temp.T)
            M_ab_a += lib.einsum('iABj,jkab,ikab->AB', eris_ovvo, t2_1_ab, t2_1_ab, optimize=True)
            temp = lib.einsum('iAaj,jkab,ikBb->AB', eris_ovvo, t2_1_ab, t2_1_ab, optimize=True)
            M_ab_a -= temp + temp.T
            M_ab_a -= lib.einsum('iabj,kiAa,kjBb->AB', eris_OVVO, t2_1_ab, t2_1_ab, optimize=True)
            M_ab_a -= lib.einsum('ijAB,ikab,jkab->AB', eris_oovv, t2_1_ab, t2_1_ab, optimize=True)
            M_ab_a -= lib.einsum('ijAB,kiab,kjab->AB', eris_OOvv, t2_1_ab, t2_1_ab, optimize=True)
            temp = lib.einsum('ijAa,ikab,jkBb->AB', eris_oovv, t2_1_ab, t2_1_ab, optimize=True)
            M_ab_a += temp + temp.T
            temp = lib.einsum('ijAa,kiab,kjBb->AB', eris_OOvv, t2_1_ab, t2_1_ab, optimize=True)
            M_ab_a += temp + temp.T
            M_ab_a += lib.einsum('ijab,ikAb,jkBa->AB', eris_ooVV, t2_1_ab, t2_1_ab, optimize=True)
            M_ab_a += lib.einsum('ijab,kiAb,kjBa->AB', eris_OOVV, t2_1_ab, t2_1_ab, optimize=True)
            M_ab_a -= lib.einsum('ijkl,ikAa,jlBa->AB', eris_ooOO, t2_1_ab, t2_1_ab, optimize=True)
            M_ab_a -= 1/2 *  lib.einsum('ijAB,ikab,jkab->AB', eris_OOvv, t2_1_b, t2_1_b, optimize=True)
            temp = lib.einsum('A,iB,ja,ijAa->AB', e_vir_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            M_ab_a -= 1/3 * (temp + temp.T)
            temp = lib.einsum('A,iA,ja,ijBa->AB', e_vir_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            M_ab_a -= 1/6 * (temp + temp.T)
            temp = lib.einsum('a,iA,ja,ijBa->AB', e_vir_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            M_ab_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('i,iA,ja,ijBa->AB', e_occ_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            M_ab_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('i,jA,ia,jiBa->AB', e_occ_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            M_ab_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('A,ijBa,iA,ja->AB', e_vir_a, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            M_ab_a -= 1/6 * (temp + temp.T)
            temp = lib.einsum('A,ijAa,iB,ja->AB', e_vir_a, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            M_ab_a -= 1/3 * (temp + temp.T)
            temp = lib.einsum('a,ijAa,ia,jB->AB', e_vir_a, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            M_ab_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('i,ijAa,iB,ja->AB', e_occ_a, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            M_ab_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('i,ijAa,ia,jB->AB', e_occ_a, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            M_ab_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('iA,iB->AB', f_ov_b, t1_2_b, optimize=True)
            M_ab_b -= temp + temp.T
            temp = lib.einsum('ia,iaAB->AB', t1_2_a, eris_ovVV, optimize=True)
            M_ab_b += temp + temp.T
            temp = lib.einsum('ia,iABa->AB', t1_2_b, eris_OVVV, optimize=True)
            M_ab_b -= temp + temp.T
            temp = lib.einsum('ia,iaAB->AB', t1_2_b, eris_OVVV, optimize=True)
            M_ab_b += temp + temp.T
            temp = lib.einsum('ijaA,jBai->AB', t2_2_ab, eris_OVvo, optimize=True)
            M_ab_b -= temp + temp.T
            temp = lib.einsum('ijAa,iBaj->AB', t2_2_b, eris_OVVO, optimize=True)
            M_ab_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ijAa,jBai->AB', t2_2_b, eris_OVVO, optimize=True)
            M_ab_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('A,iB,iA->AB', e_vir_b, t1_1_b, t1_2_b, optimize=True)
            M_ab_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('A,iA,iB->AB', e_vir_b, t1_1_b, t1_2_b, optimize=True)
            M_ab_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('i,iA,iB->AB', e_occ_b, t1_1_b, t1_2_b, optimize=True)
            M_ab_b += temp + temp.T
            temp = lib.einsum('A,ijaB,ijaA->AB', e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
            M_ab_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('A,ijaA,ijaB->AB', e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
            M_ab_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('a,ijaA,ijaB->AB', e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
            M_ab_b -= temp + temp.T
            temp = lib.einsum('i,ijaA,ijaB->AB', e_occ_a, t2_1_ab, t2_2_ab, optimize=True)
            M_ab_b += temp + temp.T
            temp = lib.einsum('i,jiaA,jiaB->AB', e_occ_b, t2_1_ab, t2_2_ab, optimize=True)
            M_ab_b += temp + temp.T
            temp = lib.einsum('A,ijBa,ijAa->AB', e_vir_b, t2_1_b, t2_2_b, optimize=True)
            M_ab_b -= 1/4 * (temp + temp.T)
            temp = lib.einsum('A,ijAa,ijBa->AB', e_vir_b, t2_1_b, t2_2_b, optimize=True)
            M_ab_b -= 1/4 * (temp + temp.T)
            temp = lib.einsum('a,ijAa,ijBa->AB', e_vir_b, t2_1_b, t2_2_b, optimize=True)
            M_ab_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('i,ijAa,ijBa->AB', e_occ_b, t2_1_b, t2_2_b, optimize=True)
            M_ab_b += temp + temp.T
            temp = lib.einsum('iA,ja,jiaB->AB', f_ov_b, t1_1_a, t2_1_ab, optimize=True)
            M_ab_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,jA,ijaB->AB', f_ov_a, t1_1_b, t2_1_ab, optimize=True)
            M_ab_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('iA,ja,ijBa->AB', f_ov_b, t1_1_b, t2_1_b, optimize=True)
            M_ab_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,jA,ijBa->AB', f_ov_b, t1_1_b, t2_1_b, optimize=True)
            M_ab_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,jA,jBai->AB', t1_1_a, t1_1_b, eris_OVvo, optimize=True)
            M_ab_b -= 2 * (temp + temp.T)
            temp = lib.einsum('ia,ijab,jbAB->AB', t1_1_a, t2_1_a, eris_ovVV, optimize=True)
            M_ab_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,ijab,jABb->AB', t1_1_a, t2_1_ab, eris_OVVV, optimize=True)
            M_ab_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,ijab,jbAB->AB', t1_1_a, t2_1_ab, eris_OVVV, optimize=True)
            M_ab_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,ijbA,jBab->AB', t1_1_a, t2_1_ab, eris_OVvv, optimize=True)
            M_ab_b -= temp + temp.T
            temp = lib.einsum('ia,jkaA,kBji->AB', t1_1_a, t2_1_ab, eris_OVoo, optimize=True)
            M_ab_b += temp + temp.T
            temp = lib.einsum('iA,jiab,jaBb->AB', t1_1_b, t2_1_ab, eris_ovVV, optimize=True)
            M_ab_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('iA,jkaB,jaki->AB', t1_1_b, t2_1_ab, eris_ovOO, optimize=True)
            M_ab_b += temp + temp.T
            temp = lib.einsum('ia,jibA,jbaB->AB', t1_1_b, t2_1_ab, eris_ovVV, optimize=True)
            M_ab_b -= temp + temp.T
            temp = lib.einsum('ia,jiba,jbAB->AB', t1_1_b, t2_1_ab, eris_ovVV, optimize=True)
            M_ab_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('iA,ijab,jaBb->AB', t1_1_b, t2_1_b, eris_OVVV, optimize=True)
            M_ab_b += 1/4 * (temp + temp.T)
            temp = lib.einsum('iA,ijab,jbBa->AB', t1_1_b, t2_1_b, eris_OVVV, optimize=True)
            M_ab_b -= 1/4 * (temp + temp.T)
            temp = lib.einsum('iA,jkBa,kaji->AB', t1_1_b, t2_1_b, eris_OVOO, optimize=True)
            M_ab_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('iA,jkBa,jaki->AB', t1_1_b, t2_1_b, eris_OVOO, optimize=True)
            M_ab_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,ijAb,jBab->AB', t1_1_b, t2_1_b, eris_OVVV, optimize=True)
            M_ab_b += temp + temp.T
            temp = lib.einsum('ia,ijAb,jbaB->AB', t1_1_b, t2_1_b, eris_OVVV, optimize=True)
            M_ab_b -= temp + temp.T
            temp = lib.einsum('ia,ijab,jABb->AB', t1_1_b, t2_1_b, eris_OVVV, optimize=True)
            M_ab_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,ijab,jbAB->AB', t1_1_b, t2_1_b, eris_OVVV, optimize=True)
            M_ab_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,jkAa,kBji->AB', t1_1_b, t2_1_b, eris_OVOO, optimize=True)
            M_ab_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,jkAa,jBki->AB', t1_1_b, t2_1_b, eris_OVOO, optimize=True)
            M_ab_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('ijab,ikaA,kBbj->AB', t2_1_a, t2_1_ab, eris_OVvo, optimize=True)
            M_ab_b -= temp + temp.T
            temp = lib.einsum('ijaA,jkBb,iabk->AB', t2_1_ab, t2_1_b, eris_ovVO, optimize=True)
            M_ab_b -= temp + temp.T
            temp = lib.einsum('ijab,jkAb,kBai->AB', t2_1_ab, t2_1_b, eris_OVvo, optimize=True)
            M_ab_b += temp + temp.T
            M_ab_b -= lib.einsum('ijAB,ia,ja->AB', eris_ooVV, t1_1_a, t1_1_a, optimize=True)
            M_ab_b += lib.einsum('iABj,ja,ia->AB', eris_OVVO, t1_1_b, t1_1_b, optimize=True)
            temp = lib.einsum('iAaj,ja,iB->AB', eris_OVVO, t1_1_b, t1_1_b, optimize=True)
            M_ab_b -= 2 * (temp + temp.T)
            temp = lib.einsum('iAaj,ia,jB->AB', eris_OVVO, t1_1_b, t1_1_b, optimize=True)
            M_ab_b += temp + temp.T
            M_ab_b -= lib.einsum('ijAB,ia,ja->AB', eris_OOVV, t1_1_b, t1_1_b, optimize=True)
            temp = lib.einsum('ijAa,ia,jB->AB', eris_OOVV, t1_1_b, t1_1_b, optimize=True)
            M_ab_b += temp + temp.T
            M_ab_b -= 1/2 *  lib.einsum('ijAB,ikab,jkab->AB', eris_ooVV, t2_1_a, t2_1_a, optimize=True)
            M_ab_b += lib.einsum('iABj,kjab,kiab->AB', eris_OVVO, t2_1_ab, t2_1_ab, optimize=True)
            temp = lib.einsum('iAaj,kjba,kibB->AB', eris_OVVO, t2_1_ab, t2_1_ab, optimize=True)
            M_ab_b -= temp + temp.T
            M_ab_b -= lib.einsum('iabj,ikaA,jkbB->AB', eris_ovvo, t2_1_ab, t2_1_ab, optimize=True)
            M_ab_b -= lib.einsum('ijAB,ikab,jkab->AB', eris_ooVV, t2_1_ab, t2_1_ab, optimize=True)
            M_ab_b -= lib.einsum('ijAB,kiab,kjab->AB', eris_OOVV, t2_1_ab, t2_1_ab, optimize=True)
            temp = lib.einsum('ijAa,ikba,jkbB->AB', eris_ooVV, t2_1_ab, t2_1_ab, optimize=True)
            M_ab_b += temp + temp.T
            temp = lib.einsum('ijAa,kiba,kjbB->AB', eris_OOVV, t2_1_ab, t2_1_ab, optimize=True)
            M_ab_b += temp + temp.T
            M_ab_b += lib.einsum('ijab,ikbA,jkaB->AB', eris_oovv, t2_1_ab, t2_1_ab, optimize=True)
            M_ab_b += lib.einsum('ijab,kibA,kjaB->AB', eris_OOvv, t2_1_ab, t2_1_ab, optimize=True)
            M_ab_b -= lib.einsum('ijkl,ikaA,jlaB->AB', eris_ooOO, t2_1_ab, t2_1_ab, optimize=True)
            M_ab_b += 1/2 *  lib.einsum('iABj,jkab,ikab->AB', eris_OVVO, t2_1_b, t2_1_b, optimize=True)
            temp = lib.einsum('iAaj,jkab,ikBb->AB', eris_OVVO, t2_1_b, t2_1_b, optimize=True)
            M_ab_b -= temp + temp.T
            M_ab_b -= lib.einsum('iabj,ikAa,jkBb->AB', eris_OVVO, t2_1_b, t2_1_b, optimize=True)
            M_ab_b -= 1/2 *  lib.einsum('ijAB,ikab,jkab->AB', eris_OOVV, t2_1_b, t2_1_b, optimize=True)
            temp = lib.einsum('ijAa,ikab,jkBb->AB', eris_OOVV, t2_1_b, t2_1_b, optimize=True)
            M_ab_b += temp + temp.T
            M_ab_b += lib.einsum('ijab,ikAb,jkBa->AB', eris_OOVV, t2_1_b, t2_1_b, optimize=True)
            temp = lib.einsum('ijkl,ikAa,jlBa->AB', eris_OOOO, t2_1_b, t2_1_b, optimize=True)
            M_ab_b -= 1/4 * (temp + temp.T)
            temp = lib.einsum('A,ia,jB,ijaA->AB', e_vir_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            M_ab_b -= 1/3 * (temp + temp.T)
            temp = lib.einsum('A,ia,jA,ijaB->AB', e_vir_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            M_ab_b -= 1/6 * (temp + temp.T)
            temp = lib.einsum('a,ia,jA,ijaB->AB', e_vir_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            M_ab_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('i,ia,jA,ijaB->AB', e_occ_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            M_ab_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('i,ja,iA,jiaB->AB', e_occ_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            M_ab_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('A,ijBa,iA,ja->AB', e_vir_b, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            M_ab_b -= 1/6 * (temp + temp.T)
            temp = lib.einsum('A,ijAa,iB,ja->AB', e_vir_b, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            M_ab_b -= 1/3 * (temp + temp.T)
            temp = lib.einsum('a,ijAa,ia,jB->AB', e_vir_b, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            M_ab_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('i,ijAa,iB,ja->AB', e_occ_b, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            M_ab_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('i,ijAa,ia,jB->AB', e_occ_b, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            M_ab_b -= 1/2 * (temp + temp.T)

            if isinstance(eris.vvvv_p, np.ndarray):
                va = adc.mo_coeff[0][:, nocc_a:]
                vb = adc.mo_coeff[1][:, nocc_b:]
                v_eeee_aaaa = ao2mo.general(adc._scf._eri, (va, va, va, va), compact=False).reshape(nvir_a,
                    nvir_a, nvir_a, nvir_a)
                M_ab_a += lib.einsum('ABab,ia,ib->AB', v_eeee_aaaa, t1_1_a, t1_1_a, optimize=True)
                M_ab_a -= lib.einsum('AabB,ib,ia->AB', v_eeee_aaaa, t1_1_a, t1_1_a, optimize=True)
                M_ab_a += 1/2 *  lib.einsum('ABab,ijac,ijbc->AB', v_eeee_aaaa, t2_1_a, t2_1_a, optimize=True)
                M_ab_a -= 1/2 *  lib.einsum('AabB,ijbc,ijac->AB', v_eeee_aaaa, t2_1_a, t2_1_a, optimize=True)
                temp = lib.einsum('Aabc,ijBb,ijac->AB', v_eeee_aaaa, t2_1_a, t2_1_a, optimize=True)
                M_ab_a -= 1/4 * (temp + temp.T)
                temp = lib.einsum('Aabc,ijBb,ijca->AB', v_eeee_aaaa, t2_1_a, t2_1_a, optimize=True)
                M_ab_a += 1/4 * (temp + temp.T)
                M_ab_a += lib.einsum('ABab,ijac,ijbc->AB', v_eeee_aaaa, t2_1_ab, t2_1_ab, optimize=True)
                M_ab_a -= lib.einsum('AabB,ijbc,ijac->AB', v_eeee_aaaa, t2_1_ab, t2_1_ab, optimize=True)
                del v_eeee_aaaa
                v_eeee_abab = eris.vVvV_p.reshape(nvir_a, nvir_b, nvir_a, nvir_b)
                M_ab_a += lib.einsum('AaBb,ia,ib->AB', v_eeee_abab, t1_1_b, t1_1_b, optimize=True)
                M_ab_a += lib.einsum('AaBb,ijca,ijcb->AB', v_eeee_abab, t2_1_ab, t2_1_ab, optimize=True)
                temp = lib.einsum('Abac,ijBb,ijac->AB', v_eeee_abab, t2_1_ab, t2_1_ab, optimize=True)
                M_ab_a -= temp + temp.T
                M_ab_a += 1/2 *  lib.einsum('AaBb,ijac,ijbc->AB', v_eeee_abab, t2_1_b, t2_1_b, optimize=True)
                M_ab_b += lib.einsum('aAbB,ia,ib->AB', v_eeee_abab, t1_1_a, t1_1_a, optimize=True)
                M_ab_b += 1/2 *  lib.einsum('aAbB,ijac,ijbc->AB', v_eeee_abab, t2_1_a, t2_1_a, optimize=True)
                M_ab_b += lib.einsum('aAbB,ijac,ijbc->AB', v_eeee_abab, t2_1_ab, t2_1_ab, optimize=True)
                temp = lib.einsum('bAca,ijbB,ijca->AB', v_eeee_abab, t2_1_ab, t2_1_ab, optimize=True)
                M_ab_b -= temp + temp.T
                del v_eeee_abab
                v_eeee_bbbb = ao2mo.general(adc._scf._eri, (vb, vb, vb, vb), compact=False).reshape(nvir_b,
                    nvir_b, nvir_b, nvir_b)
                M_ab_b += lib.einsum('ABab,ia,ib->AB', v_eeee_bbbb, t1_1_b, t1_1_b, optimize=True)
                M_ab_b -= lib.einsum('AabB,ib,ia->AB', v_eeee_bbbb, t1_1_b, t1_1_b, optimize=True)
                M_ab_b += lib.einsum('ABab,ijca,ijcb->AB', v_eeee_bbbb, t2_1_ab, t2_1_ab, optimize=True)
                M_ab_b -= lib.einsum('AabB,ijcb,ijca->AB', v_eeee_bbbb, t2_1_ab, t2_1_ab, optimize=True)
                M_ab_b += 1/2 *  lib.einsum('ABab,ijac,ijbc->AB', v_eeee_bbbb, t2_1_b, t2_1_b, optimize=True)
                M_ab_b -= 1/2 *  lib.einsum('AabB,ijbc,ijac->AB', v_eeee_bbbb, t2_1_b, t2_1_b, optimize=True)
                temp = lib.einsum('Aabc,ijBb,ijac->AB', v_eeee_bbbb, t2_1_b, t2_1_b, optimize=True)
                M_ab_b -= 1/4 * (temp + temp.T)
                temp = lib.einsum('Aabc,ijBb,ijca->AB', v_eeee_bbbb, t2_1_b, t2_1_b, optimize=True)
                M_ab_b += 1/4 * (temp + temp.T)
                del v_eeee_bbbb
            elif isinstance(eris.vvvv_p, list):
                mol = adc.mol
                va = adc.mo_coeff[0][:, nocc_a:]
                vb = adc.mo_coeff[1][:, nocc_b:]
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                for p, q in lib.prange(0, nvir_a, chnk_size):
                    v_eeee_aaaa = ao2mo.general(mol, (va[:, p:q], va, va, va),
                        compact=False).reshape(q-p, nvir_a, nvir_a, nvir_a)
                    M_ab_a[p:q] += lib.einsum('ABab,ia,ib->AB', v_eeee_aaaa, t1_1_a, t1_1_a, optimize=True)
                    M_ab_a[p:q] -= lib.einsum('AabB,ib,ia->AB', v_eeee_aaaa, t1_1_a, t1_1_a, optimize=True)
                    M_ab_a[p:q] += 1/2 *  lib.einsum('ABab,ijac,ijbc->AB', v_eeee_aaaa, t2_1_a, t2_1_a, optimize=True)
                    M_ab_a[p:q] -= 1/2 *  lib.einsum('AabB,ijbc,ijac->AB', v_eeee_aaaa, t2_1_a, t2_1_a, optimize=True)
                    temp = lib.einsum('Aabc,ijBb,ijac->AB', v_eeee_aaaa, t2_1_a, t2_1_a, optimize=True)
                    M_ab_a[p:q] -= 1/4 * temp
                    M_ab_a[:, p:q] -= 1/4 * temp.T
                    temp = lib.einsum('Aabc,ijBb,ijca->AB', v_eeee_aaaa, t2_1_a, t2_1_a, optimize=True)
                    M_ab_a[p:q] += 1/4 * temp
                    M_ab_a[:, p:q] += 1/4 * temp.T
                    M_ab_a[p:q] += lib.einsum('ABab,ijac,ijbc->AB', v_eeee_aaaa, t2_1_ab, t2_1_ab, optimize=True)
                    M_ab_a[p:q] -= lib.einsum('AabB,ijbc,ijac->AB', v_eeee_aaaa, t2_1_ab, t2_1_ab, optimize=True)
                    del v_eeee_aaaa
                p = 0
                for dataset in eris.vVvV_p:
                    k = dataset.shape[0]
                    v_eeee_abab = dataset[:].reshape(k, nvir_b, nvir_a, nvir_b)
                    M_ab_a[p:p+k] += lib.einsum('AaBb,ia,ib->AB', v_eeee_abab, t1_1_b, t1_1_b, optimize=True)
                    M_ab_a[p:p+k] += lib.einsum('AaBb,ijca,ijcb->AB', v_eeee_abab, t2_1_ab, t2_1_ab, optimize=True)
                    temp = lib.einsum('Abac,ijBb,ijac->AB', v_eeee_abab, t2_1_ab, t2_1_ab, optimize=True)
                    M_ab_a[p:p+k] -= temp
                    M_ab_a[:, p:p+k] -= temp.T
                    M_ab_a[p:p+k] += 1/2 *  lib.einsum('AaBb,ijac,ijbc->AB', v_eeee_abab, t2_1_b, t2_1_b, optimize=True)
                    del v_eeee_abab
                    p += k
                p = 0
                for dataset in eris.VvVv_p:
                    k = dataset.shape[0]
                    v_eeee_abab = dataset[:].reshape(k, nvir_a, nvir_b, nvir_a)
                    M_ab_b[p:p+k] += lib.einsum('AaBb,ia,ib->AB', v_eeee_abab, t1_1_a, t1_1_a, optimize=True)
                    M_ab_b[p:p+k] += 1/2 *  lib.einsum('AaBb,ijac,ijbc->AB', v_eeee_abab, t2_1_a, t2_1_a, optimize=True)
                    M_ab_b[p:p+k] += lib.einsum('AaBb,ijac,ijbc->AB', v_eeee_abab, t2_1_ab, t2_1_ab, optimize=True)
                    temp = lib.einsum('Abac,ijbB,ijca->AB', v_eeee_abab, t2_1_ab, t2_1_ab, optimize=True)
                    M_ab_b[p:p+k] -= temp
                    M_ab_b[:, p:p+k] -= temp.T
                    del v_eeee_abab
                    p += k
                for p, q in lib.prange(0, nvir_b, chnk_size):
                    v_eeee_bbbb = ao2mo.general(mol, (vb[:, p:q], vb, vb, vb),
                        compact=False).reshape(q-p, nvir_b, nvir_b, nvir_b)
                    M_ab_b[p:q] += lib.einsum('ABab,ia,ib->AB', v_eeee_bbbb, t1_1_b, t1_1_b, optimize=True)
                    M_ab_b[p:q] -= lib.einsum('AabB,ib,ia->AB', v_eeee_bbbb, t1_1_b, t1_1_b, optimize=True)
                    M_ab_b[p:q] += lib.einsum('ABab,ijca,ijcb->AB', v_eeee_bbbb, t2_1_ab, t2_1_ab, optimize=True)
                    M_ab_b[p:q] -= lib.einsum('AabB,ijcb,ijca->AB', v_eeee_bbbb, t2_1_ab, t2_1_ab, optimize=True)
                    M_ab_b[p:q] += 1/2 *  lib.einsum('ABab,ijac,ijbc->AB', v_eeee_bbbb, t2_1_b, t2_1_b, optimize=True)
                    M_ab_b[p:q] -= 1/2 *  lib.einsum('AabB,ijbc,ijac->AB', v_eeee_bbbb, t2_1_b, t2_1_b, optimize=True)
                    temp = lib.einsum('Aabc,ijBb,ijac->AB', v_eeee_bbbb, t2_1_b, t2_1_b, optimize=True)
                    M_ab_b[p:q] -= 1/4 * temp
                    M_ab_b[:, p:q] -= 1/4 * temp.T
                    temp = lib.einsum('Aabc,ijBb,ijca->AB', v_eeee_bbbb, t2_1_b, t2_1_b, optimize=True)
                    M_ab_b[p:q] += 1/4 * temp
                    M_ab_b[:, p:q] += 1/4 * temp.T
                    del v_eeee_bbbb
            elif eris.vvvv_p is None:
                lad_aaaa = contract_ladder(adc, t2_1_a, (eris.Lvv, eris.Lvv))
                lad_aabb = contract_ladder(adc, t2_1_ab, (eris.Lvv, eris.LVV))
                lad_bbbb = contract_ladder(adc, t2_1_b, (eris.LVV, eris.LVV))
                temp = lib.einsum('Pab,ia,ib->P', eris.Lvv, t1_1_a, t1_1_a, optimize=True)
                M_ab_a += lib.einsum('PAB,P->AB', eris.Lvv, temp, optimize=True)
                temp = lib.einsum('PbB,ia,ib->PaB', eris.Lvv, t1_1_a, t1_1_a, optimize=True)
                M_ab_a -= lib.einsum('PAa,PaB->AB', eris.Lvv, temp, optimize=True)
                temp = lib.einsum('Pab,ia,ib->P', eris.LVV, t1_1_b, t1_1_b, optimize=True)
                M_ab_a += lib.einsum('PAB,P->AB', eris.Lvv, temp, optimize=True)
                temp = lib.einsum('ijac,ijbc->ab', t2_1_a, t2_1_a, optimize=True)
                temp = lib.einsum('Pab,ab->P', eris.Lvv, temp, optimize=True)
                M_ab_a += 1/2 *  lib.einsum('PAB,P->AB', eris.Lvv, temp, optimize=True)
                temp = lib.einsum('ijbc,ijac->ab', t2_1_a, t2_1_a, optimize=True)
                temp = lib.einsum('ab,PbB->PaB', temp, eris.Lvv, optimize=True)
                M_ab_a -= 1/2 *  lib.einsum('PAa,PaB->AB', eris.Lvv, temp, optimize=True)
                temp = lib.einsum('ijAb,ijBb->AB', lad_aaaa, t2_1_a, optimize=True)
                M_ab_a -= 1/4 * (temp + temp.T)
                temp = lib.einsum('ijAb,ijBb->AB', lad_aaaa.transpose(0,1,3,2), t2_1_a, optimize=True)
                M_ab_a += 1/4 * (temp + temp.T)
                temp = lib.einsum('ijac,ijbc->ab', t2_1_ab, t2_1_ab, optimize=True)
                temp = lib.einsum('Pab,ab->P', eris.Lvv, temp, optimize=True)
                M_ab_a += lib.einsum('PAB,P->AB', eris.Lvv, temp, optimize=True)
                temp = lib.einsum('ijca,ijcb->ab', t2_1_ab, t2_1_ab, optimize=True)
                temp1 = lib.einsum('Pab,ab->P', eris.LVV, temp, optimize=True)
                M_ab_a += lib.einsum('PAB,P->AB', eris.Lvv, temp1, optimize=True)
                temp = lib.einsum('ijbc,ijac->ab', t2_1_ab, t2_1_ab, optimize=True)
                temp = lib.einsum('ab,PbB->PaB', temp, eris.Lvv, optimize=True)
                M_ab_a -= lib.einsum('PAa,PaB->AB', eris.Lvv, temp, optimize=True)
                temp = lib.einsum('ijAb,ijBb->AB', lad_aabb, t2_1_ab, optimize=True)
                M_ab_a -= temp + temp.T
                temp = lib.einsum('ijac,ijbc->ab', t2_1_b, t2_1_b, optimize=True)
                temp = lib.einsum('Pab,ab->P', eris.LVV, temp, optimize=True)
                M_ab_a += 1/2 *  lib.einsum('PAB,P->AB', eris.Lvv, temp, optimize=True)
                temp = lib.einsum('Pab,ia,ib->P', eris.Lvv, t1_1_a, t1_1_a, optimize=True)
                M_ab_b += lib.einsum('PAB,P->AB', eris.LVV, temp, optimize=True)
                temp = lib.einsum('Pab,ia,ib->P', eris.LVV, t1_1_b, t1_1_b, optimize=True)
                M_ab_b += lib.einsum('PAB,P->AB', eris.LVV, temp, optimize=True)
                temp = lib.einsum('PbB,ia,ib->PaB', eris.LVV, t1_1_b, t1_1_b, optimize=True)
                M_ab_b -= lib.einsum('PAa,PaB->AB', eris.LVV, temp, optimize=True)
                temp = lib.einsum('ijac,ijbc->ab', t2_1_a, t2_1_a, optimize=True)
                temp = lib.einsum('Pab,ab->P', eris.Lvv, temp, optimize=True)
                M_ab_b += 1/2 *  lib.einsum('PAB,P->AB', eris.LVV, temp, optimize=True)
                temp = lib.einsum('ijac,ijbc->ab', t2_1_ab, t2_1_ab, optimize=True)
                temp = lib.einsum('Pab,ab->P', eris.Lvv, temp, optimize=True)
                M_ab_b += lib.einsum('PAB,P->AB', eris.LVV, temp, optimize=True)
                temp = lib.einsum('ijca,ijcb->ab', t2_1_ab, t2_1_ab, optimize=True)
                temp1 = lib.einsum('Pab,ab->P', eris.LVV, temp, optimize=True)
                M_ab_b += lib.einsum('PAB,P->AB', eris.LVV, temp1, optimize=True)
                temp = lib.einsum('ab,PbB->PaB', temp, eris.LVV, optimize=True)
                M_ab_b -= lib.einsum('PAa,PaB->AB', eris.LVV, temp, optimize=True)
                temp = lib.einsum('ijbA,ijbB->AB', lad_aabb, t2_1_ab, optimize=True)
                M_ab_b -= temp + temp.T
                temp = lib.einsum('ijac,ijbc->ab', t2_1_b, t2_1_b, optimize=True)
                temp = lib.einsum('Pab,ab->P', eris.LVV, temp, optimize=True)
                M_ab_b += 1/2 *  lib.einsum('PAB,P->AB', eris.LVV, temp, optimize=True)
                temp = lib.einsum('ijbc,ijac->ab', t2_1_b, t2_1_b, optimize=True)
                temp = lib.einsum('ab,PbB->PaB', temp, eris.LVV, optimize=True)
                M_ab_b -= 1/2 *  lib.einsum('PAa,PaB->AB', eris.LVV, temp, optimize=True)
                temp = lib.einsum('ijAb,ijBb->AB', lad_bbbb, t2_1_b, optimize=True)
                M_ab_b -= 1/4 * (temp + temp.T)
                temp = lib.einsum('ijAb,ijBb->AB', lad_bbbb.transpose(0,1,3,2), t2_1_b, optimize=True)
                M_ab_b += 1/4 * (temp + temp.T)

        del t2_1_a
        del t2_1_b
        del t2_1_ab

    M_ab = (M_ab_a, M_ab_b)

    cput0 = log.timer_debug1("Completed M_ab ADC(3) calculation", *cput0)
    return M_ab


def get_diag(adc,M_ab=None,eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    if M_ab is None:
        M_ab = adc.get_imds()

    M_ab_a, M_ab_b = M_ab[0], M_ab[1]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

    n_singles_a = nvir_a
    n_singles_b = nvir_b
    n_doubles_aaa = nvir_a * (nvir_a - 1) * nocc_a // 2
    n_doubles_bab = nocc_b * nvir_a * nvir_b
    n_doubles_aba = nocc_a * nvir_b * nvir_a
    n_doubles_bbb = nvir_b * (nvir_b - 1) * nocc_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb

    d_i_a = e_occ_a[:,None]
    d_ab_a = e_vir_a[:,None] + e_vir_a
    D_n_a = -d_i_a + d_ab_a.reshape(-1)
    D_n_a = D_n_a.reshape((nocc_a,nvir_a,nvir_a))
    D_iab_a = D_n_a.copy()[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

    d_i_b = e_occ_b[:,None]
    d_ab_b = e_vir_b[:,None] + e_vir_b
    D_n_b = -d_i_b + d_ab_b.reshape(-1)
    D_n_b = D_n_b.reshape((nocc_b,nvir_b,nvir_b))
    D_iab_b = D_n_b.copy()[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

    d_ab_ab = e_vir_a[:,None] + e_vir_b
    d_i_b = e_occ_b[:,None]
    D_n_bab = -d_i_b + d_ab_ab.reshape(-1)
    D_iab_bab = D_n_bab.reshape(-1)

    d_ab_ab = e_vir_b[:,None] + e_vir_a
    d_i_a = e_occ_a[:,None]
    D_n_aba = -d_i_a + d_ab_ab.reshape(-1)
    D_iab_aba = D_n_aba.reshape(-1)

    diag = np.zeros(dim)

    # Compute precond in p1-p1 block

    M_ab_a_diag = np.diagonal(M_ab_a)
    M_ab_b_diag = np.diagonal(M_ab_b)

    diag[s_a:f_a] = M_ab_a_diag.copy()
    diag[s_b:f_b] = M_ab_b_diag.copy()

    # Compute precond in 2p1h-2p1h block

    diag[s_aaa:f_aaa] = D_iab_a
    diag[s_bab:f_bab] = D_iab_bab
    diag[s_aba:f_aba] = D_iab_aba
    diag[s_bbb:f_bbb] = D_iab_b

#    ###### Additional terms for the preconditioner ####
#    if (method == "adc(2)-x" or method == "adc(3)"):
#
#        if eris is None:
#            eris = adc.transform_integrals()
#
#        if isinstance(eris.vvvv_p, np.ndarray):
#
#            eris_oovv = eris.oovv
#            eris_ovvo = eris.ovvo
#            eris_OOVV = eris.OOVV
#            eris_OVVO = eris.OVVO
#            eris_OOvv = eris.OOvv
#            eris_ooVV = eris.ooVV
#
#            eris_vvvv = eris.vvvv_p
#            temp = np.zeros((nocc_a, eris_vvvv.shape[0]))
#            temp[:] += np.diag(eris_vvvv)
#            diag[s_aaa:f_aaa] += temp.reshape(-1)
#
#            eris_VVVV = eris.VVVV_p
#            temp = np.zeros((nocc_b, eris_VVVV.shape[0]))
#            temp[:] += np.diag(eris_VVVV)
#            diag[s_bbb:f_bbb] += temp.reshape(-1)
#
#            eris_vVvV = eris.vVvV_p
#            temp = np.zeros((nocc_b, eris_vVvV.shape[0]))
#            temp[:] += np.diag(eris_vVvV)
#            diag[s_bab:f_bab] += temp.reshape(-1)
#
#            temp = np.zeros((nocc_a, nvir_a, nvir_b))
#            temp[:] += np.diag(eris_vVvV).reshape(nvir_a,nvir_b)
#            temp = np.ascontiguousarray(temp.transpose(0,2,1))
#            diag[s_aba:f_aba] += temp.reshape(-1)
#
#            eris_ovov_p = np.ascontiguousarray(eris_oovv.transpose(0,2,1,3))
#            eris_ovov_p -= np.ascontiguousarray(eris_ovvo.transpose(0,2,3,1))
#            eris_ovov_p = eris_ovov_p.reshape(nocc_a*nvir_a, nocc_a*nvir_a)
#
#            temp = np.zeros((eris_ovov_p.shape[0],nvir_a))
#            temp.T[:] += np.diagonal(eris_ovov_p)
#            temp = temp.reshape(nocc_a, nvir_a, nvir_a)
#            diag[s_aaa:f_aaa] += -temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)
#
#            temp = np.ascontiguousarray(temp.transpose(0,2,1))
#            diag[s_aaa:f_aaa] += -temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)
#
#            eris_OVOV_p = np.ascontiguousarray(eris_OOVV.transpose(0,2,1,3))
#            eris_OVOV_p -= np.ascontiguousarray(eris_OVVO.transpose(0,2,3,1))
#            eris_OVOV_p = eris_OVOV_p.reshape(nocc_b*nvir_b, nocc_b*nvir_b)
#
#            temp = np.zeros((eris_OVOV_p.shape[0],nvir_b))
#            temp.T[:] += np.diagonal(eris_OVOV_p)
#            temp = temp.reshape(nocc_b, nvir_b, nvir_b)
#            diag[s_bbb:f_bbb] += -temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)
#
#            temp = np.ascontiguousarray(temp.transpose(0,2,1))
#            diag[s_bbb:f_bbb] += -temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)
#
#            temp = np.zeros((nvir_a, nocc_b, nvir_b))
#            temp[:] += np.diagonal(eris_OVOV_p).reshape(nocc_b, nvir_b)
#            temp = np.ascontiguousarray(temp.transpose(1,0,2))
#            diag[s_bab:f_bab] += -temp.reshape(-1)
#
#            temp = np.zeros((nvir_b, nocc_a, nvir_a))
#            temp[:] += np.diagonal(eris_ovov_p).reshape(nocc_a, nvir_a)
#            temp = np.ascontiguousarray(temp.transpose(1,0,2))
#            diag[s_aba:f_aba] += -temp.reshape(-1)
#
#            eris_OvOv_p = np.ascontiguousarray(eris_OOvv.transpose(0,2,1,3))
#            eris_OvOv_p = eris_OvOv_p.reshape(nocc_b*nvir_a, nocc_b*nvir_a)
#
#            temp = np.zeros((nvir_b, nocc_b, nvir_a))
#            temp[:] += np.diagonal(eris_OvOv_p).reshape(nocc_b,nvir_a)
#            temp = np.ascontiguousarray(temp.transpose(1,2,0))
#            diag[s_bab:f_bab] += -temp.reshape(-1)
#
#            eris_oVoV_p = np.ascontiguousarray(eris_ooVV.transpose(0,2,1,3))
#            eris_oVoV_p = eris_oVoV_p.reshape(nocc_a*nvir_b, nocc_a*nvir_b)
#
#            temp = np.zeros((nvir_a, nocc_a, nvir_b))
#            temp[:] += np.diagonal(eris_oVoV_p).reshape(nocc_a,nvir_b)
#            temp = np.ascontiguousarray(temp.transpose(1,2,0))
#            diag[s_aba:f_aba] += -temp.reshape(-1)
#        else:
#           raise Exception("Precond not available for out-of-core and density-fitted algo")

    return diag


def matvec(adc, M_ab=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    n_singles_a = nvir_a
    n_singles_b = nvir_b
    n_doubles_aaa = nvir_a * (nvir_a - 1) * nocc_a // 2
    n_doubles_bab = nocc_b * nvir_a * nvir_b
    n_doubles_aba = nocc_a * nvir_b * nvir_a
    n_doubles_bbb = nvir_b * (nvir_b - 1) * nocc_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

    t1_1_a = t1_1_b = None
    if adc.t1[2][0] is not None:
        t1_1_a = adc.t1[2][0]
        t1_1_b = adc.t1[2][1]
        f_ov_a, f_ov_b = adc.f_ov

    if eris is None:
        eris = adc.transform_integrals()

    d_i_a = e_occ_a[:,None]
    d_ab_a = e_vir_a[:,None] + e_vir_a
    D_n_a = -d_i_a + d_ab_a.reshape(-1)
    D_n_a = D_n_a.reshape((nocc_a,nvir_a,nvir_a))
    D_iab_a = D_n_a.copy()[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

    d_i_b = e_occ_b[:,None]
    d_ab_b = e_vir_b[:,None] + e_vir_b
    D_n_b = -d_i_b + d_ab_b.reshape(-1)
    D_n_b = D_n_b.reshape((nocc_b,nvir_b,nvir_b))
    D_iab_b = D_n_b.copy()[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

    d_ab_ab = e_vir_a[:,None] + e_vir_b
    d_i_b = e_occ_b[:,None]
    D_n_bab = -d_i_b + d_ab_ab.reshape(-1)
    D_iab_bab = D_n_bab.reshape(-1)

    d_ab_ab = e_vir_b[:,None] + e_vir_a
    d_i_a = e_occ_a[:,None]
    D_n_aba = -d_i_a + d_ab_ab.reshape(-1)
    D_iab_aba = D_n_aba.reshape(-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb

    if M_ab is None:
        M_ab = adc.get_imds()
    M_ab_a, M_ab_b = M_ab

    #Calculate sigma vector
    def sigma_(r):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(adc.stdout, adc.verbose)

        s = np.zeros(dim)

        r_a = r[s_a:f_a]
        r_b = r[s_b:f_b]

        r_aaa = r[s_aaa:f_aaa]
        r_bab = r[s_bab:f_bab]
        r_aba = r[s_aba:f_aba]
        r_bbb = r[s_bbb:f_bbb]

        r_aaa_ = np.zeros((nocc_a, nvir_a, nvir_a))
        r_aaa_[:, ab_ind_a[0], ab_ind_a[1]] = r_aaa.reshape(nocc_a, -1)
        r_aaa_[:, ab_ind_a[1], ab_ind_a[0]] = -r_aaa.reshape(nocc_a, -1)
        r_bbb_ = np.zeros((nocc_b, nvir_b, nvir_b))
        r_bbb_[:, ab_ind_b[0], ab_ind_b[1]] = r_bbb.reshape(nocc_b, -1)
        r_bbb_[:, ab_ind_b[1], ab_ind_b[0]] = -r_bbb.reshape(nocc_b, -1)

        r_aba = r_aba.reshape(nocc_a,nvir_b,nvir_a)
        r_bab = r_bab.reshape(nocc_b,nvir_a,nvir_b)

############ ADC(2) ab block ############################

        s[s_a:f_a] = lib.einsum('ab,b->a',M_ab_a,r_a)
        s[s_b:f_b] = lib.einsum('ab,b->a',M_ab_b,r_b)

############ ADC(2) a - ibc and ibc - a coupling blocks #########################

        temp = np.zeros((nocc_a, nvir_a, nvir_a))
        if eris.ovvv is None:
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            for a,b in lib.prange(0,nocc_a,chnk_size):
                eris_ovvv = dfadc.get_ovvv_spin_df(
                    adc, eris.Lov, eris.Lvv, a, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
                s[s_a:f_a] += lib.einsum('icab,ibc->a',eris_ovvv, r_aaa_[a:b], optimize=True)
                temp[a:b] += lib.einsum('icab,a->ibc', eris_ovvv, r_a, optimize=True)
                temp[a:b] -= lib.einsum('ibac,a->ibc', eris_ovvv, r_a, optimize=True)
                del eris_ovvv
        else :
            eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
            s[s_a:f_a] += lib.einsum('icab,ibc->a',eris_ovvv, r_aaa_, optimize=True)
            temp += lib.einsum('icab,a->ibc', eris_ovvv, r_a, optimize=True)
            temp -= lib.einsum('ibac,a->ibc', eris_ovvv, r_a, optimize=True)
            del eris_ovvv

        s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)
        del temp

        temp = np.zeros((nocc_b, nvir_a, nvir_b))
        if eris.OVvv is None:
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            for a,b in lib.prange(0,nocc_b,chnk_size):
                eris_OVvv = dfadc.get_ovvv_spin_df(
                    adc, eris.LOV, eris.Lvv, a, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
                s[s_a:f_a] += lib.einsum('icab,ibc->a', eris_OVvv, r_bab[a:b], optimize=True)
                temp[a:b] += lib.einsum('icab,a->ibc', eris_OVvv, r_a, optimize=True)
                del eris_OVvv
        else :
            eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
            s[s_a:f_a] += lib.einsum('icab,ibc->a', eris_OVvv, r_bab, optimize=True)
            temp += lib.einsum('icab,a->ibc', eris_OVvv, r_a, optimize=True)
            del eris_OVvv

        s[s_bab:f_bab] += temp.reshape(-1)
        del temp

        temp = np.zeros((nocc_b, nvir_b, nvir_b))
        if eris.OVVV is None:
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            for a,b in lib.prange(0,nocc_b,chnk_size):
                eris_OVVV = dfadc.get_ovvv_spin_df(
                    adc, eris.LOV, eris.LVV, a, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
                s[s_b:f_b] += lib.einsum('icab,ibc->a',eris_OVVV, r_bbb_[a:b], optimize=True)
                temp[a:b] += lib.einsum('icab,a->ibc', eris_OVVV, r_b, optimize=True)
                temp[a:b] -= lib.einsum('ibac,a->ibc', eris_OVVV, r_b, optimize=True)
                del eris_OVVV
        else :
            eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
            s[s_b:f_b] += lib.einsum('icab,ibc->a',eris_OVVV, r_bbb_, optimize=True)
            temp += lib.einsum('icab,a->ibc', eris_OVVV, r_b, optimize=True)
            temp -= lib.einsum('ibac,a->ibc', eris_OVVV, r_b, optimize=True)
            del eris_OVVV

        s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)
        del temp

        temp = np.zeros((nocc_a, nvir_b, nvir_a))
        if eris.ovVV is None:
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            for a,b in lib.prange(0,nocc_a,chnk_size):
                eris_ovVV = dfadc.get_ovvv_spin_df(
                    adc, eris.Lov, eris.LVV, a, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
                s[s_b:f_b] += lib.einsum('icab,ibc->a', eris_ovVV, r_aba[a:b], optimize=True)
                temp[a:b] += lib.einsum('icab,a->ibc', eris_ovVV, r_b, optimize=True)
                del eris_ovVV
        else :
            eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
            s[s_b:f_b] += lib.einsum('icab,ibc->a', eris_ovVV, r_aba, optimize=True)
            temp += lib.einsum('icab,a->ibc', eris_ovVV, r_b, optimize=True)
            del eris_ovVV
        s[s_aba:f_aba] += temp.reshape(-1)
        del temp

        if t1_1_a is not None:
            s[s_a:f_a] += lib.einsum('iAa,a,ia->A', r_aaa_, e_vir_a, t1_1_a, optimize=True)
            s[s_a:f_a] -= lib.einsum('iAa,i,ia->A', r_aaa_, e_occ_a, t1_1_a, optimize=True)
            s[s_a:f_a] += lib.einsum('iAa,a,ia->A', r_bab, e_vir_b, t1_1_b, optimize=True)
            s[s_a:f_a] -= lib.einsum('iAa,i,ia->A', r_bab, e_occ_b, t1_1_b, optimize=True)
            s[s_b:f_b] += lib.einsum('iAa,a,ia->A', r_aba, e_vir_a, t1_1_a, optimize=True)
            s[s_b:f_b] -= lib.einsum('iAa,i,ia->A', r_aba, e_occ_a, t1_1_a, optimize=True)
            s[s_b:f_b] += lib.einsum('iAa,a,ia->A', r_bbb_, e_vir_b, t1_1_b, optimize=True)
            s[s_b:f_b] -= lib.einsum('iAa,i,ia->A', r_bbb_, e_occ_b, t1_1_b, optimize=True)
            s[s_a:f_a] += lib.einsum('iAa,ia->A', r_aaa_, f_ov_a, optimize=True)
            s[s_a:f_a] += lib.einsum('iAa,ia->A', r_bab, f_ov_b, optimize=True)
            s[s_b:f_b] += lib.einsum('iAa,ia->A', r_aba, f_ov_a, optimize=True)
            s[s_b:f_b] += lib.einsum('iAa,ia->A', r_bbb_, f_ov_b, optimize=True)

############### ADC(2) iab - jcd block ############################

        s[s_aaa:f_aaa] += D_iab_a * r_aaa
        s[s_bab:f_bab] += D_iab_bab * r_bab.reshape(-1)
        s[s_aba:f_aba] += D_iab_aba * r_aba.reshape(-1)
        s[s_bbb:f_bbb] += D_iab_b * r_bbb

############### ADC(3) iab - jcd block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

            eris_oovv = eris.oovv
            eris_OOVV = eris.OOVV
            eris_ooVV = eris.ooVV
            eris_OOvv = eris.OOvv
            eris_ovvo = eris.ovvo
            eris_OVVO = eris.OVVO
            eris_ovVO = eris.ovVO
            eris_OVvo = eris.OVvo

            r_aaa = r_aaa.reshape(nocc_a,-1)
            r_bbb = r_bbb.reshape(nocc_b,-1)

            r_aaa_u = np.zeros((nocc_a,nvir_a,nvir_a))
            r_aaa_u[:,ab_ind_a[0],ab_ind_a[1]]= r_aaa.copy()
            r_aaa_u[:,ab_ind_a[1],ab_ind_a[0]]= -r_aaa.copy()

            r_bbb_u = None
            r_bbb_u = np.zeros((nocc_b,nvir_b,nvir_b))
            r_bbb_u[:,ab_ind_b[0],ab_ind_b[1]]= r_bbb.copy()
            r_bbb_u[:,ab_ind_b[1],ab_ind_b[0]]= -r_bbb.copy()

            if isinstance(eris.vvvv_p, np.ndarray):
                eris_vvvv = eris.vvvv_p
                temp_1 = np.dot(r_aaa,eris_vvvv.T)
                del eris_vvvv
            elif isinstance(eris.vvvv_p, list):
                temp_1 = contract_r_vvvv_antisym(adc,r_aaa_u,eris.vvvv_p)
                temp_1 = temp_1[:,ab_ind_a[0],ab_ind_a[1]]
            else:
                temp_1 = contract_r_vvvv_antisym(adc,r_aaa_u,eris.Lvv)
                temp_1 = temp_1[:,ab_ind_a[0],ab_ind_a[1]]

            s[s_aaa:f_aaa] += temp_1.reshape(-1)

            if isinstance(eris.VVVV_p, np.ndarray):
                eris_VVVV = eris.VVVV_p
                temp_1 = np.dot(r_bbb,eris_VVVV.T)
                del eris_VVVV
            elif isinstance(eris.VVVV_p, list):
                temp_1 = contract_r_vvvv_antisym(adc,r_bbb_u,eris.VVVV_p)
                temp_1 = temp_1[:,ab_ind_b[0],ab_ind_b[1]]
            else:
                temp_1 = contract_r_vvvv_antisym(adc,r_bbb_u,eris.LVV)
                temp_1 = temp_1[:,ab_ind_b[0],ab_ind_b[1]]

            s[s_bbb:f_bbb] += temp_1.reshape(-1)

            if isinstance(eris.vVvV_p, np.ndarray):
                r_bab_t = r_bab.reshape(nocc_b,-1)
                r_aba_t = r_aba.transpose(0,2,1).reshape(nocc_a,-1)
                eris_vVvV = eris.vVvV_p
                s[s_bab:f_bab] += np.dot(r_bab_t,eris_vVvV.T).reshape(-1)
                temp_1 = np.dot(r_aba_t,eris_vVvV.T).reshape(nocc_a, nvir_a,nvir_b)
                s[s_aba:f_aba] += temp_1.transpose(0,2,1).copy().reshape(-1)
            elif isinstance(eris.vVvV_p, list):
                temp_1 = contract_r_vvvv(adc,r_bab,eris.vVvV_p)
                temp_2 = contract_r_vvvv(adc,r_aba,eris.VvVv_p)

                s[s_bab:f_bab] += temp_1.reshape(-1)
                s[s_aba:f_aba] += temp_2.reshape(-1)
            else:
                temp_1 = contract_r_vvvv(adc,r_bab,(eris.Lvv,eris.LVV))
                temp_2 = contract_r_vvvv(adc,r_aba,(eris.LVV,eris.Lvv))

                s[s_bab:f_bab] += temp_1.reshape(-1)
                s[s_aba:f_aba] += temp_2.reshape(-1)

            temp = 0.5*lib.einsum('jiyz,jzx->ixy',eris_oovv,r_aaa_u,optimize=True)
            temp -= 0.5*lib.einsum('jzyi,jzx->ixy',eris_ovvo,r_aaa_u,optimize=True)
            temp +=0.5*lib.einsum('jzyi,jxz->ixy',eris_OVvo,r_bab,optimize=True)
            s[s_aaa:f_aaa] += 2*temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

            s[s_bab:f_bab] -= lib.einsum('jzyi,jzx->ixy',eris_ovVO,
                                             r_aaa_u,optimize=True).reshape(-1)
            s[s_bab:f_bab] -= lib.einsum('jiyz,jxz->ixy',eris_OOVV,
                                             r_bab,optimize=True).reshape(-1)
            s[s_bab:f_bab] += lib.einsum('jzyi,jxz->ixy',eris_OVVO,
                                             r_bab,optimize=True).reshape(-1)

            temp = 0.5*lib.einsum('jiyz,jzx->ixy',eris_OOVV,r_bbb_u,optimize=True)
            temp -= 0.5*lib.einsum('jzyi,jzx->ixy',eris_OVVO,r_bbb_u,optimize=True)
            temp +=0.5* lib.einsum('jzyi,jxz->ixy',eris_ovVO,r_aba,optimize=True)
            s[s_bbb:f_bbb] += 2*temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

            s[s_aba:f_aba] -= lib.einsum('jiyz,jxz->ixy',eris_oovv,
                                             r_aba,optimize=True).reshape(-1)
            s[s_aba:f_aba] += lib.einsum('jzyi,jxz->ixy',eris_ovvo,
                                             r_aba,optimize=True).reshape(-1)
            s[s_aba:f_aba] -= lib.einsum('jzyi,jzx->ixy',eris_OVvo,
                                             r_bbb_u,optimize=True).reshape(-1)

            temp = -0.5*lib.einsum('jixz,jzy->ixy',eris_oovv,r_aaa_u,optimize=True)
            temp += 0.5*lib.einsum('jzxi,jzy->ixy',eris_ovvo,r_aaa_u,optimize=True)
            temp -= 0.5*lib.einsum('jzxi,jyz->ixy',eris_OVvo,r_bab,optimize=True)
            s[s_aaa:f_aaa] += 2*temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

            s[s_bab:f_bab] -=  lib.einsum('jixz,jzy->ixy',
                                              eris_OOvv,r_bab,optimize=True).reshape(-1)

            temp = -0.5*lib.einsum('jixz,jzy->ixy',eris_OOVV,r_bbb_u,optimize=True)
            temp += 0.5*lib.einsum('jzxi,jzy->ixy',eris_OVVO,r_bbb_u,optimize=True)
            temp -= 0.5*lib.einsum('jzxi,jyz->ixy',eris_ovVO,r_aba,optimize=True)
            s[s_bbb:f_bbb] += 2*temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

            s[s_aba:f_aba] -= lib.einsum('jixz,jzy->ixy',eris_ooVV,
                                             r_aba,optimize=True).reshape(-1)

        if (method == "adc(3)"):
            if t1_1_a is None:

                eris_ovoo = eris.ovoo
                eris_OVOO = eris.OVOO
                eris_ovOO = eris.ovOO
                eris_OVoo = eris.OVoo

    ############### ADC(3) a - ibc block and ibc-a coupling blocks ########################
                t2_1_a = adc.t2[0][0][:]
                t2_1_ab = adc.t2[0][1][:]

                t2_1_a_t = t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]]

                r_aaa = r_aaa.reshape(nocc_a,-1)
                temp = 0.5*lib.einsum('lmp,jp->lmj',t2_1_a_t,r_aaa)
                del t2_1_a_t
                s[s_a:f_a] += 2*lib.einsum('lmj,lamj->a',temp, eris_ovoo, optimize=True)
                del temp

                temp_1 = -lib.einsum('lmzw,jzw->jlm',t2_1_ab,r_bab)
                s[s_a:f_a] -= lib.einsum('jlm,lamj->a',temp_1, eris_ovOO, optimize=True)
                del temp_1

                temp_1 = -lib.einsum('mlwz,jzw->jlm',t2_1_ab,r_aba)
                s[s_b:f_b] -= lib.einsum('jlm,lamj->a',temp_1, eris_OVoo, optimize=True)
                del temp_1

                r_aaa_u = np.zeros((nocc_a,nvir_a,nvir_a))
                r_aaa_u[:,ab_ind_a[0],ab_ind_a[1]]= r_aaa.copy()
                r_aaa_u[:,ab_ind_a[1],ab_ind_a[0]]= -r_aaa.copy()

                r_bbb_u = np.zeros((nocc_b,nvir_b,nvir_b))
                r_bbb_u[:,ab_ind_b[0],ab_ind_b[1]]= r_bbb.copy()
                r_bbb_u[:,ab_ind_b[1],ab_ind_b[0]]= -r_bbb.copy()

                r_bab = r_bab.reshape(nocc_b,nvir_a,nvir_b)
                r_aba = r_aba.reshape(nocc_a,nvir_b,nvir_a)

                temp_s_a = np.zeros_like(r_bab)
                temp_s_a = lib.einsum('jlwd,jzw->lzd',t2_1_a,r_aaa_u,optimize=True)
                temp_s_a += lib.einsum('ljdw,jzw->lzd',t2_1_ab,r_bab,optimize=True)

                temp_1_1 = np.zeros((nocc_a,nvir_a,nvir_a))
                temp_1_2 = np.zeros((nocc_a,nvir_a,nvir_a))
                if eris.ovvv is None:
                    chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                    for a,b in lib.prange(0,nocc_a,chnk_size):
                        eris_ovvv = dfadc.get_ovvv_spin_df(
                            adc, eris.Lov, eris.Lvv, a, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
                        s[s_a:f_a] += lib.einsum('lzd,ldza->a',
                                                     temp_s_a[a:b],eris_ovvv,optimize=True)
                        s[s_a:f_a] -= lib.einsum('lzd,lazd->a',
                                                     temp_s_a[a:b],eris_ovvv,optimize=True)

                        temp_1_1[a:b] += lib.einsum('ldxb,b->lxd', eris_ovvv,r_a,optimize=True)
                        temp_1_1[a:b] -= lib.einsum('lbxd,b->lxd', eris_ovvv,r_a,optimize=True)

                        temp_1_2[a:b] += lib.einsum('ldyb,b->lyd', eris_ovvv,r_a,optimize=True)
                        temp_1_2[a:b] -= lib.einsum('lbyd,b->lyd', eris_ovvv,r_a,optimize=True)
                        del eris_ovvv
                else :
                    eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
                    s[s_a:f_a] += lib.einsum('lzd,ldza->a',temp_s_a,eris_ovvv,optimize=True)
                    s[s_a:f_a] -= lib.einsum('lzd,lazd->a',temp_s_a,eris_ovvv,optimize=True)

                    temp_1_1 += lib.einsum('ldxb,b->lxd', eris_ovvv,r_a,optimize=True)
                    temp_1_1 -= lib.einsum('lbxd,b->lxd', eris_ovvv,r_a,optimize=True)

                    temp_1_2 += lib.einsum('ldyb,b->lyd', eris_ovvv,r_a,optimize=True)
                    temp_1_2 -= lib.einsum('lbyd,b->lyd', eris_ovvv,r_a,optimize=True)
                    del eris_ovvv

                del temp_s_a

                r_bab_t = r_bab.reshape(nocc_b*nvir_a,-1)
                temp = np.ascontiguousarray(t2_1_ab.transpose(
                    0,3,1,2)).reshape(nocc_a*nvir_b,nocc_b*nvir_a)
                temp_2 = np.dot(temp,r_bab_t).reshape(nocc_a,nvir_b,nvir_b)
                del temp
                temp_2 = np.ascontiguousarray(temp_2.transpose(0,2,1))
                temp_new_1 = np.zeros_like(r_aba)
                temp_new_1 = lib.einsum('ljdw,jzw->ldz',t2_1_ab,r_bbb_u,optimize=True)
                temp_new_1 += lib.einsum('jlwd,jzw->ldz',t2_1_a,r_aba,optimize=True)

                temp_2_3 = np.zeros((nocc_a,nvir_b,nvir_a))
                temp_2_4 = np.zeros((nocc_a,nvir_b,nvir_a))

                temp = np.zeros((nocc_a,nvir_b,nvir_b))
                if eris.ovVV is None:
                    chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                    for a,b in lib.prange(0,nocc_a,chnk_size):
                        eris_ovVV = dfadc.get_ovvv_spin_df(
                            adc, eris.Lov, eris.LVV, a, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
                        s[s_a:f_a] -= lib.einsum('lzd,lazd->a',
                                                     temp_2[a:b],eris_ovVV,optimize=True)

                        s[s_b:f_b] += lib.einsum('ldz,ldza->a',temp_new_1[a:b],eris_ovVV)

                        eris_ovVV = eris_ovVV.reshape(-1, nvir_a, nvir_b, nvir_b)

                        temp_2_3[a:b] += lib.einsum('ldxb,b->lxd', eris_ovVV,r_b,optimize=True)
                        temp_2_4[a:b] += lib.einsum('ldyb,b->lyd', eris_ovVV,r_b,optimize=True)

                        temp[a:b]  -= lib.einsum('lbyd,b->lyd',eris_ovVV,r_a,optimize=True)
                        del eris_ovVV
                else :
                    eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
                    s[s_a:f_a] -= lib.einsum('lzd,lazd->a',temp_2,eris_ovVV,optimize=True)

                    s[s_b:f_b] += lib.einsum('ldz,ldza->a',temp_new_1,eris_ovVV)

                    eris_ovVV = eris_ovVV.reshape(-1, nvir_a, nvir_b, nvir_b)

                    temp_2_3 += lib.einsum('ldxb,b->lxd', eris_ovVV,r_b,optimize=True)
                    temp_2_4 += lib.einsum('ldyb,b->lyd', eris_ovVV,r_b,optimize=True)

                    temp  -= lib.einsum('lbyd,b->lyd',eris_ovVV,r_a,optimize=True)
                    del eris_ovVV

                temp = -lib.einsum('lyd,lixd->ixy',temp,t2_1_ab,optimize=True)
                s[s_bab:f_bab] -= temp.reshape(-1)
                del temp
                del temp_2
                del temp_new_1

                t2_1_a_t = t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]]
                temp = lib.einsum('b,lbmi->lmi',r_a,eris_ovoo)
                temp -= lib.einsum('b,mbli->lmi',r_a,eris_ovoo)
                s[s_aaa:f_aaa] += 0.5*lib.einsum('lmi,lmp->ip',temp,
                                                 t2_1_a_t, optimize=True).reshape(-1)

                temp  = lib.einsum('lxd,ilyd->ixy',temp_1_1,t2_1_a,optimize=True)
                s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1] ].reshape(-1)

                temp  = lib.einsum('lyd,ilxd->ixy',temp_1_2,t2_1_a,optimize=True)
                s[s_aaa:f_aaa] -= temp[:,ab_ind_a[0],ab_ind_a[1] ].reshape(-1)

                temp  = lib.einsum('lxd,ilyd->ixy',temp_2_3,t2_1_a,optimize=True)
                s[s_aba:f_aba] += temp.reshape(-1)

                t2_1_b = adc.t2[0][2][:]

                t2_1_b_t = t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]]
                r_bbb = r_bbb.reshape(nocc_b,-1)
                temp = 0.5*lib.einsum('lmp,jp->lmj',t2_1_b_t,r_bbb)
                del t2_1_b_t
                s[s_b:f_b] += 2*lib.einsum('lmj,lamj->a',temp, eris_OVOO, optimize=True)
                del temp

                temp_s_b = np.zeros_like(r_aba)
                temp_s_b = lib.einsum('jlwd,jzw->lzd',t2_1_b,r_bbb_u,optimize=True)
                temp_s_b += lib.einsum('jlwd,jzw->lzd',t2_1_ab,r_aba,optimize=True)

                temp_1_3 = np.zeros((nocc_b,nvir_b,nvir_b))
                temp_1_4 = np.zeros((nocc_b,nvir_b,nvir_b))

                if eris.OVVV is None:
                    chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                    for a,b in lib.prange(0,nocc_b,chnk_size):
                        eris_OVVV = dfadc.get_ovvv_spin_df(
                            adc, eris.LOV, eris.LVV, a, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
                        s[s_b:f_b] += lib.einsum('lzd,ldza->a',
                                                     temp_s_b[a:b],eris_OVVV,optimize=True)
                        s[s_b:f_b] -= lib.einsum('lzd,lazd->a',
                                                     temp_s_b[a:b],eris_OVVV,optimize=True)

                        temp_1_3[a:b] += lib.einsum('ldxb,b->lxd', eris_OVVV,r_b,optimize=True)
                        temp_1_3[a:b] -= lib.einsum('lbxd,b->lxd', eris_OVVV,r_b,optimize=True)

                        temp_1_4[a:b] += lib.einsum('ldyb,b->lyd', eris_OVVV,r_b,optimize=True)
                        temp_1_4[a:b] -= lib.einsum('lbyd,b->lyd', eris_OVVV,r_b,optimize=True)
                        del eris_OVVV
                else :
                    eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
                    s[s_b:f_b] += lib.einsum('lzd,ldza->a',temp_s_b,eris_OVVV,optimize=True)
                    s[s_b:f_b] -= lib.einsum('lzd,lazd->a',temp_s_b,eris_OVVV,optimize=True)

                    temp_1_3 += lib.einsum('ldxb,b->lxd', eris_OVVV,r_b,optimize=True)
                    temp_1_3 -= lib.einsum('lbxd,b->lxd', eris_OVVV,r_b,optimize=True)

                    temp_1_4 += lib.einsum('ldyb,b->lyd', eris_OVVV,r_b,optimize=True)
                    temp_1_4 -= lib.einsum('lbyd,b->lyd', eris_OVVV,r_b,optimize=True)
                    del eris_OVVV

                del temp_s_b

                temp_1 = np.zeros_like(r_bab)
                temp_1= lib.einsum('jlwd,jzw->lzd',t2_1_ab,r_aaa_u,optimize=True)
                temp_1 += lib.einsum('jlwd,jzw->lzd',t2_1_b,r_bab,optimize=True)
                temp_2 = lib.einsum('jldw,jwz->lzd',t2_1_ab,r_aba,optimize=True)
                temp_2_1 = np.zeros((nocc_b,nvir_a,nvir_b))
                temp_2_2 = np.zeros((nocc_b,nvir_a,nvir_b))
                temp = np.zeros((nocc_b,nvir_a,nvir_a))

                if eris.OVvv is None:
                    chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                    for a,b in lib.prange(0,nocc_b,chnk_size):
                        eris_OVvv = dfadc.get_ovvv_spin_df(
                            adc, eris.LOV, eris.Lvv, a, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
                        s[s_a:f_a] += lib.einsum('lzd,ldza->a',
                                                     temp_1[a:b],eris_OVvv,optimize=True)

                        s[s_b:f_b] -= lib.einsum('lzd,lazd->a',
                                                     temp_2[a:b],eris_OVvv,optimize=True)

                        temp_2_1[a:b] += lib.einsum('ldxb,b->lxd', eris_OVvv,r_a,optimize=True)
                        temp_2_2[a:b] += lib.einsum('ldyb,b->lyd', eris_OVvv,r_a,optimize=True)

                        temp[a:b]  -= lib.einsum('lbyd,b->lyd',eris_OVvv,r_b,optimize=True)
                        del eris_OVvv
                else :
                    eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
                    s[s_a:f_a] += lib.einsum('lzd,ldza->a',temp_1,eris_OVvv,optimize=True)

                    s[s_b:f_b] -= lib.einsum('lzd,lazd->a',temp_2,eris_OVvv,optimize=True)

                    temp_2_1 += lib.einsum('ldxb,b->lxd', eris_OVvv,r_a,optimize=True)
                    temp_2_2 += lib.einsum('ldyb,b->lyd', eris_OVvv,r_a,optimize=True)

                    temp  -= lib.einsum('lbyd,b->lyd',eris_OVvv,r_b,optimize=True)
                    del eris_OVvv

                temp_new = -lib.einsum('lyd,ildx->ixy',temp,t2_1_ab,optimize=True)
                s[s_aba:f_aba] -= temp_new.reshape(-1)
                del temp
                del temp_new
                del temp_1
                del temp_2

                t2_1_b_t = t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]]
                temp = lib.einsum('b,lbmi->lmi',r_b,eris_OVOO)
                temp -= lib.einsum('b,mbli->lmi',r_b,eris_OVOO)
                s[s_bbb:f_bbb] += 0.5*lib.einsum('lmi,lmp->ip',temp,
                                                 t2_1_b_t, optimize=True).reshape(-1)

                temp  = lib.einsum('lxd,ilyd->ixy',temp_2_1,t2_1_b,optimize=True)
                s[s_bab:f_bab] += temp.reshape(-1)

                temp  = lib.einsum('lxd,ilyd->ixy',temp_1_3,t2_1_b,optimize=True)
                s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1] ].reshape(-1)

                temp  = lib.einsum('lyd,ilxd->ixy',temp_1_4,t2_1_b,optimize=True)
                s[s_bbb:f_bbb] -= temp[:,ab_ind_b[0],ab_ind_b[1] ].reshape(-1)

                temp_1 = lib.einsum('b,lbmi->lmi',r_a,eris_ovOO)
                s[s_bab:f_bab] += lib.einsum('lmi,lmxy->ixy',temp_1, t2_1_ab, optimize=True).reshape(-1)

                temp_1 = lib.einsum('b,lbmi->mli',r_b,eris_OVoo)
                s[s_aba:f_aba] += lib.einsum('mli,mlyx->ixy',temp_1, t2_1_ab, optimize=True).reshape(-1)

                temp = lib.einsum('lxd,ilyd->ixy',temp_2_1,t2_1_ab,optimize=True)
                s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1] ].reshape(-1)

                temp = lib.einsum('lyd,ilxd->ixy',temp_2_2,t2_1_ab,optimize=True)
                s[s_aaa:f_aaa] -= temp[:,ab_ind_a[0],ab_ind_a[1] ].reshape(-1)

                temp  = lib.einsum('lxd,lidy->ixy',temp_1_1,t2_1_ab,optimize=True)
                s[s_bab:f_bab] += temp.reshape(-1)

                temp = lib.einsum('lxd,lidy->ixy',temp_2_3,t2_1_ab,optimize=True)
                s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1] ].reshape(-1)

                temp = lib.einsum('lyd,lidx->ixy',temp_2_4,t2_1_ab,optimize=True)
                s[s_bbb:f_bbb] -= temp[:,ab_ind_b[0],ab_ind_b[1] ].reshape(-1)

                temp  = lib.einsum('lxd,ilyd->ixy',temp_1_3,t2_1_ab,optimize=True)
                s[s_aba:f_aba] += temp.reshape(-1)

                del temp_2_1
                del temp_1_3
                del temp_1_4
                del temp_1_1
                del temp_1_2
                del temp_2_3

            else:
                t2_1_a = adc.t2[0][0][:]
                t2_1_ab = adc.t2[0][1][:]
                t2_1_b = adc.t2[0][2][:]
                t1_2_a = adc.t1[0][0][:]
                t1_2_b = adc.t1[0][1][:]
                eris_OOVV = eris.OOVV
                eris_OOvv = eris.OOvv
                eris_OVOO = eris.OVOO
                eris_OVVO = eris.OVVO
                eris_OVoo = eris.OVoo
                eris_OVvo = eris.OVvo
                eris_ooVV = eris.ooVV
                eris_oovv = eris.oovv
                eris_ovOO = eris.ovOO
                eris_ovVO = eris.ovVO
                eris_ovoo = eris.ovoo
                eris_ovvo = eris.ovvo
                if eris.OVVV is None:
                    chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                    eris_OVVV = []
                    for a, b in lib.prange(0, nocc_b, chnk_size):
                        eris_OVVV.append(dfadc.get_ovvv_spin_df(adc, eris.LOV, eris.LVV, a, chnk_size))
                    eris_OVVV = np.concatenate(eris_OVVV, axis=0)
                else:
                    eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
                if eris.OVvv is None:
                    chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                    eris_OVvv = []
                    for a, b in lib.prange(0, nocc_b, chnk_size):
                        eris_OVvv.append(dfadc.get_ovvv_spin_df(adc, eris.LOV, eris.Lvv, a, chnk_size))
                    eris_OVvv = np.concatenate(eris_OVvv, axis=0)
                else:
                    eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
                if eris.ovVV is None:
                    chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                    eris_ovVV = []
                    for a, b in lib.prange(0, nocc_a, chnk_size):
                        eris_ovVV.append(dfadc.get_ovvv_spin_df(adc, eris.Lov, eris.LVV, a, chnk_size))
                    eris_ovVV = np.concatenate(eris_ovVV, axis=0)
                else:
                    eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
                if eris.ovvv is None:
                    chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                    eris_ovvv = []
                    for a, b in lib.prange(0, nocc_a, chnk_size):
                        eris_ovvv.append(dfadc.get_ovvv_spin_df(adc, eris.Lov, eris.Lvv, a, chnk_size))
                    eris_ovvv = np.concatenate(eris_ovvv, axis=0)
                else:
                    eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
                s[s_a:f_a] += lib.einsum('iAa,a,ia->A', r_aaa_, e_vir_a, t1_2_a, optimize=True)
                s[s_a:f_a] -= lib.einsum('iAa,i,ia->A', r_aaa_, e_occ_a, t1_2_a, optimize=True)
                s[s_a:f_a] += lib.einsum('iAa,jb,ijab->A', r_aaa_, f_ov_a, t2_1_a, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('iab,jA,ijab->A', r_aaa_, f_ov_a, t2_1_a, optimize=True)
                s[s_a:f_a] += lib.einsum('iAa,jb,ijab->A', r_aaa_, f_ov_b, t2_1_ab, optimize=True)
                s[s_a:f_a] += 2 *  lib.einsum('iAa,jb,iabj->A', r_aaa_, t1_1_a, eris_ovvo, optimize=True)
                s[s_a:f_a] -= lib.einsum('iAa,jb,ibaj->A', r_aaa_, t1_1_a, eris_ovvo, optimize=True)
                s[s_a:f_a] -= lib.einsum('iAa,jb,ijba->A', r_aaa_, t1_1_a, eris_oovv, optimize=True)
                s[s_a:f_a] += lib.einsum('iab,jA,iabj->A', r_aaa_, t1_1_a, eris_ovvo, optimize=True)
                s[s_a:f_a] += lib.einsum('iab,ja,ijAb->A', r_aaa_, t1_1_a, eris_oovv, optimize=True)
                s[s_a:f_a] -= lib.einsum('iab,ja,jAbi->A', r_aaa_, t1_1_a, eris_ovvo, optimize=True)
                s[s_a:f_a] += 2 *  lib.einsum('iAa,jb,iabj->A', r_aaa_, t1_1_b, eris_ovVO, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('iAa,ijbc,jbac->A', r_aaa_, t2_1_a, eris_ovvv, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('iAa,ijbc,jcab->A', r_aaa_, t2_1_a, eris_ovvv, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('iAa,jkab,kbji->A', r_aaa_, t2_1_a, eris_ovoo, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('iAa,jkab,jbki->A', r_aaa_, t2_1_a, eris_ovoo, optimize=True)
                s[s_a:f_a] += lib.einsum('iab,ijac,jAbc->A', r_aaa_, t2_1_a, eris_ovvv, optimize=True)
                s[s_a:f_a] -= lib.einsum('iab,ijac,jcbA->A', r_aaa_, t2_1_a, eris_ovvv, optimize=True)
                s[s_a:f_a] -= 1/4 *  lib.einsum('iab,jkab,kAji->A', r_aaa_, t2_1_a, eris_ovoo, optimize=True)
                s[s_a:f_a] += 1/4 *  lib.einsum('iab,jkab,jAki->A', r_aaa_, t2_1_a, eris_ovoo, optimize=True)
                s[s_a:f_a] += lib.einsum('iAa,ijbc,jcab->A', r_aaa_, t2_1_ab, eris_OVvv, optimize=True)
                s[s_a:f_a] -= lib.einsum('iAa,jkab,kbji->A', r_aaa_, t2_1_ab, eris_OVoo, optimize=True)
                s[s_a:f_a] -= lib.einsum('iab,ijac,jcbA->A', r_aaa_, t2_1_ab, eris_OVvv, optimize=True)
                s[s_a:f_a] += lib.einsum('iAa,a,ia->A', r_bab, e_vir_b, t1_2_b, optimize=True)
                s[s_a:f_a] -= lib.einsum('iAa,i,ia->A', r_bab, e_occ_b, t1_2_b, optimize=True)
                s[s_a:f_a] += lib.einsum('iAa,jb,jiba->A', r_bab, f_ov_a, t2_1_ab, optimize=True)
                s[s_a:f_a] -= lib.einsum('iab,jA,jiab->A', r_bab, f_ov_a, t2_1_ab, optimize=True)
                s[s_a:f_a] += lib.einsum('iAa,jb,ijab->A', r_bab, f_ov_b, t2_1_b, optimize=True)
                s[s_a:f_a] += 2 *  lib.einsum('iAa,jb,iabj->A', r_bab, t1_1_a, eris_OVvo, optimize=True)
                s[s_a:f_a] -= lib.einsum('iab,jA,ibaj->A', r_bab, t1_1_a, eris_OVvo, optimize=True)
                s[s_a:f_a] -= lib.einsum('iab,ja,jAbi->A', r_bab, t1_1_a, eris_ovVO, optimize=True)
                s[s_a:f_a] += 2 *  lib.einsum('iAa,jb,iabj->A', r_bab, t1_1_b, eris_OVVO, optimize=True)
                s[s_a:f_a] -= lib.einsum('iAa,jb,ibaj->A', r_bab, t1_1_b, eris_OVVO, optimize=True)
                s[s_a:f_a] -= lib.einsum('iAa,jb,ijba->A', r_bab, t1_1_b, eris_OOVV, optimize=True)
                s[s_a:f_a] -= lib.einsum('iab,jb,ijAa->A', r_bab, t1_1_b, eris_OOvv, optimize=True)
                s[s_a:f_a] += lib.einsum('iAa,jibc,jbac->A', r_bab, t2_1_ab, eris_ovVV, optimize=True)
                s[s_a:f_a] -= lib.einsum('iAa,jkba,jbki->A', r_bab, t2_1_ab, eris_ovOO, optimize=True)
                s[s_a:f_a] -= lib.einsum('iab,jiac,jAbc->A', r_bab, t2_1_ab, eris_ovVV, optimize=True)
                s[s_a:f_a] -= lib.einsum('iab,jicb,jAac->A', r_bab, t2_1_ab, eris_ovvv, optimize=True)
                s[s_a:f_a] += lib.einsum('iab,jicb,jcaA->A', r_bab, t2_1_ab, eris_ovvv, optimize=True)
                s[s_a:f_a] += lib.einsum('iab,jkab,jAki->A', r_bab, t2_1_ab, eris_ovOO, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('iAa,ijbc,jbac->A', r_bab, t2_1_b, eris_OVVV, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('iAa,ijbc,jcab->A', r_bab, t2_1_b, eris_OVVV, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('iAa,jkab,kbji->A', r_bab, t2_1_b, eris_OVOO, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('iAa,jkab,jbki->A', r_bab, t2_1_b, eris_OVOO, optimize=True)
                s[s_a:f_a] += lib.einsum('iab,ijbc,jcaA->A', r_bab, t2_1_b, eris_OVvv, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('iAa,a,jb,ijab->A', r_aaa_, e_vir_a, t1_1_a, t2_1_a, optimize=True)
                s[s_a:f_a] += lib.einsum('iAa,b,jb,ijab->A', r_aaa_, e_vir_a, t1_1_a, t2_1_a, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('iAa,i,jb,ijab->A', r_aaa_, e_occ_a, t1_1_a, t2_1_a, optimize=True)
                s[s_a:f_a] -= lib.einsum('iAa,j,jb,ijab->A', r_aaa_, e_occ_a, t1_1_a, t2_1_a, optimize=True)
                s[s_a:f_a] += 1/4 *  lib.einsum('iab,A,jA,ijab->A', r_aaa_, e_vir_a, t1_1_a, t2_1_a, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('iab,b,jA,ijab->A', r_aaa_, e_vir_a, t1_1_a, t2_1_a, optimize=True)
                s[s_a:f_a] -= 1/4 *  lib.einsum('iab,i,jA,ijab->A', r_aaa_, e_occ_a, t1_1_a, t2_1_a, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('iab,j,jA,ijab->A', r_aaa_, e_occ_a, t1_1_a, t2_1_a, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('iAa,a,jb,ijab->A', r_aaa_, e_vir_a, t1_1_b, t2_1_ab, optimize=True)
                s[s_a:f_a] += lib.einsum('iAa,b,jb,ijab->A', r_aaa_, e_vir_b, t1_1_b, t2_1_ab, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('iAa,i,jb,ijab->A', r_aaa_, e_occ_a, t1_1_b, t2_1_ab, optimize=True)
                s[s_a:f_a] -= lib.einsum('iAa,j,jb,ijab->A', r_aaa_, e_occ_b, t1_1_b, t2_1_ab, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('iAa,a,jb,jiba->A', r_bab, e_vir_b, t1_1_a, t2_1_ab, optimize=True)
                s[s_a:f_a] += lib.einsum('iAa,b,jb,jiba->A', r_bab, e_vir_a, t1_1_a, t2_1_ab, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('iAa,i,jb,jiba->A', r_bab, e_occ_b, t1_1_a, t2_1_ab, optimize=True)
                s[s_a:f_a] -= lib.einsum('iAa,j,jb,jiba->A', r_bab, e_occ_a, t1_1_a, t2_1_ab, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('iab,A,jA,jiab->A', r_bab, e_vir_a, t1_1_a, t2_1_ab, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('iab,a,jA,jiab->A', r_bab, e_vir_a, t1_1_a, t2_1_ab, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('iab,b,jA,jiab->A', r_bab, e_vir_b, t1_1_a, t2_1_ab, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('iab,i,jA,jiab->A', r_bab, e_occ_b, t1_1_a, t2_1_ab, optimize=True)
                s[s_a:f_a] += lib.einsum('iab,j,jA,jiab->A', r_bab, e_occ_a, t1_1_a, t2_1_ab, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('iAa,a,jb,ijab->A', r_bab, e_vir_b, t1_1_b, t2_1_b, optimize=True)
                s[s_a:f_a] += lib.einsum('iAa,b,jb,ijab->A', r_bab, e_vir_b, t1_1_b, t2_1_b, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('iAa,i,jb,ijab->A', r_bab, e_occ_b, t1_1_b, t2_1_b, optimize=True)
                s[s_a:f_a] -= lib.einsum('iAa,j,jb,ijab->A', r_bab, e_occ_b, t1_1_b, t2_1_b, optimize=True)
                s[s_b:f_b] += lib.einsum('iAa,a,ia->A', r_aba, e_vir_a, t1_2_a, optimize=True)
                s[s_b:f_b] -= lib.einsum('iAa,i,ia->A', r_aba, e_occ_a, t1_2_a, optimize=True)
                s[s_b:f_b] += lib.einsum('iAa,jb,ijab->A', r_aba, f_ov_a, t2_1_a, optimize=True)
                s[s_b:f_b] += lib.einsum('iAa,jb,ijab->A', r_aba, f_ov_b, t2_1_ab, optimize=True)
                s[s_b:f_b] -= lib.einsum('iab,jA,ijba->A', r_aba, f_ov_b, t2_1_ab, optimize=True)
                s[s_b:f_b] += 2 *  lib.einsum('iAa,jb,iabj->A', r_aba, t1_1_a, eris_ovvo, optimize=True)
                s[s_b:f_b] -= lib.einsum('iAa,jb,ibaj->A', r_aba, t1_1_a, eris_ovvo, optimize=True)
                s[s_b:f_b] -= lib.einsum('iAa,jb,ijba->A', r_aba, t1_1_a, eris_oovv, optimize=True)
                s[s_b:f_b] -= lib.einsum('iab,jb,ijAa->A', r_aba, t1_1_a, eris_ooVV, optimize=True)
                s[s_b:f_b] += 2 *  lib.einsum('iAa,jb,iabj->A', r_aba, t1_1_b, eris_ovVO, optimize=True)
                s[s_b:f_b] -= lib.einsum('iab,jA,ibaj->A', r_aba, t1_1_b, eris_ovVO, optimize=True)
                s[s_b:f_b] -= lib.einsum('iab,ja,jAbi->A', r_aba, t1_1_b, eris_OVvo, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('iAa,ijbc,jbac->A', r_aba, t2_1_a, eris_ovvv, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('iAa,ijbc,jcab->A', r_aba, t2_1_a, eris_ovvv, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('iAa,jkab,kbji->A', r_aba, t2_1_a, eris_ovoo, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('iAa,jkab,jbki->A', r_aba, t2_1_a, eris_ovoo, optimize=True)
                s[s_b:f_b] += lib.einsum('iab,ijbc,jcaA->A', r_aba, t2_1_a, eris_ovVV, optimize=True)
                s[s_b:f_b] += lib.einsum('iAa,ijbc,jcab->A', r_aba, t2_1_ab, eris_OVvv, optimize=True)
                s[s_b:f_b] -= lib.einsum('iAa,jkab,kbji->A', r_aba, t2_1_ab, eris_OVoo, optimize=True)
                s[s_b:f_b] -= lib.einsum('iab,ijbc,jAac->A', r_aba, t2_1_ab, eris_OVVV, optimize=True)
                s[s_b:f_b] += lib.einsum('iab,ijbc,jcaA->A', r_aba, t2_1_ab, eris_OVVV, optimize=True)
                s[s_b:f_b] -= lib.einsum('iab,ijca,jAbc->A', r_aba, t2_1_ab, eris_OVvv, optimize=True)
                s[s_b:f_b] += lib.einsum('iab,jkba,kAji->A', r_aba, t2_1_ab, eris_OVoo, optimize=True)
                s[s_b:f_b] += lib.einsum('iAa,a,ia->A', r_bbb_, e_vir_b, t1_2_b, optimize=True)
                s[s_b:f_b] -= lib.einsum('iAa,i,ia->A', r_bbb_, e_occ_b, t1_2_b, optimize=True)
                s[s_b:f_b] += lib.einsum('iAa,jb,jiba->A', r_bbb_, f_ov_a, t2_1_ab, optimize=True)
                s[s_b:f_b] += lib.einsum('iAa,jb,ijab->A', r_bbb_, f_ov_b, t2_1_b, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('iab,jA,ijab->A', r_bbb_, f_ov_b, t2_1_b, optimize=True)
                s[s_b:f_b] += 2 *  lib.einsum('iAa,jb,iabj->A', r_bbb_, t1_1_a, eris_OVvo, optimize=True)
                s[s_b:f_b] += 2 *  lib.einsum('iAa,jb,iabj->A', r_bbb_, t1_1_b, eris_OVVO, optimize=True)
                s[s_b:f_b] -= lib.einsum('iAa,jb,ibaj->A', r_bbb_, t1_1_b, eris_OVVO, optimize=True)
                s[s_b:f_b] -= lib.einsum('iAa,jb,ijba->A', r_bbb_, t1_1_b, eris_OOVV, optimize=True)
                s[s_b:f_b] += lib.einsum('iab,jA,iabj->A', r_bbb_, t1_1_b, eris_OVVO, optimize=True)
                s[s_b:f_b] += lib.einsum('iab,ja,ijAb->A', r_bbb_, t1_1_b, eris_OOVV, optimize=True)
                s[s_b:f_b] -= lib.einsum('iab,ja,jAbi->A', r_bbb_, t1_1_b, eris_OVVO, optimize=True)
                s[s_b:f_b] += lib.einsum('iAa,jibc,jbac->A', r_bbb_, t2_1_ab, eris_ovVV, optimize=True)
                s[s_b:f_b] -= lib.einsum('iAa,jkba,jbki->A', r_bbb_, t2_1_ab, eris_ovOO, optimize=True)
                s[s_b:f_b] -= lib.einsum('iab,jica,jcbA->A', r_bbb_, t2_1_ab, eris_ovVV, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('iAa,ijbc,jbac->A', r_bbb_, t2_1_b, eris_OVVV, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('iAa,ijbc,jcab->A', r_bbb_, t2_1_b, eris_OVVV, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('iAa,jkab,kbji->A', r_bbb_, t2_1_b, eris_OVOO, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('iAa,jkab,jbki->A', r_bbb_, t2_1_b, eris_OVOO, optimize=True)
                s[s_b:f_b] += lib.einsum('iab,ijac,jAbc->A', r_bbb_, t2_1_b, eris_OVVV, optimize=True)
                s[s_b:f_b] -= lib.einsum('iab,ijac,jcbA->A', r_bbb_, t2_1_b, eris_OVVV, optimize=True)
                s[s_b:f_b] -= 1/4 *  lib.einsum('iab,jkab,kAji->A', r_bbb_, t2_1_b, eris_OVOO, optimize=True)
                s[s_b:f_b] += 1/4 *  lib.einsum('iab,jkab,jAki->A', r_bbb_, t2_1_b, eris_OVOO, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('iAa,a,jb,ijab->A', r_aba, e_vir_a, t1_1_a, t2_1_a, optimize=True)
                s[s_b:f_b] += lib.einsum('iAa,b,jb,ijab->A', r_aba, e_vir_a, t1_1_a, t2_1_a, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('iAa,i,jb,ijab->A', r_aba, e_occ_a, t1_1_a, t2_1_a, optimize=True)
                s[s_b:f_b] -= lib.einsum('iAa,j,jb,ijab->A', r_aba, e_occ_a, t1_1_a, t2_1_a, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('iAa,a,jb,ijab->A', r_aba, e_vir_a, t1_1_b, t2_1_ab, optimize=True)
                s[s_b:f_b] += lib.einsum('iAa,b,jb,ijab->A', r_aba, e_vir_b, t1_1_b, t2_1_ab, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('iAa,i,jb,ijab->A', r_aba, e_occ_a, t1_1_b, t2_1_ab, optimize=True)
                s[s_b:f_b] -= lib.einsum('iAa,j,jb,ijab->A', r_aba, e_occ_b, t1_1_b, t2_1_ab, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('iab,A,jA,ijba->A', r_aba, e_vir_b, t1_1_b, t2_1_ab, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('iab,a,jA,ijba->A', r_aba, e_vir_b, t1_1_b, t2_1_ab, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('iab,b,jA,ijba->A', r_aba, e_vir_a, t1_1_b, t2_1_ab, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('iab,i,jA,ijba->A', r_aba, e_occ_a, t1_1_b, t2_1_ab, optimize=True)
                s[s_b:f_b] += lib.einsum('iab,j,jA,ijba->A', r_aba, e_occ_b, t1_1_b, t2_1_ab, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('iAa,a,jb,jiba->A', r_bbb_, e_vir_b, t1_1_a, t2_1_ab, optimize=True)
                s[s_b:f_b] += lib.einsum('iAa,b,jb,jiba->A', r_bbb_, e_vir_a, t1_1_a, t2_1_ab, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('iAa,i,jb,jiba->A', r_bbb_, e_occ_b, t1_1_a, t2_1_ab, optimize=True)
                s[s_b:f_b] -= lib.einsum('iAa,j,jb,jiba->A', r_bbb_, e_occ_a, t1_1_a, t2_1_ab, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('iAa,a,jb,ijab->A', r_bbb_, e_vir_b, t1_1_b, t2_1_b, optimize=True)
                s[s_b:f_b] += lib.einsum('iAa,b,jb,ijab->A', r_bbb_, e_vir_b, t1_1_b, t2_1_b, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('iAa,i,jb,ijab->A', r_bbb_, e_occ_b, t1_1_b, t2_1_b, optimize=True)
                s[s_b:f_b] -= lib.einsum('iAa,j,jb,ijab->A', r_bbb_, e_occ_b, t1_1_b, t2_1_b, optimize=True)
                s[s_b:f_b] += 1/4 *  lib.einsum('iab,A,jA,ijab->A', r_bbb_, e_vir_b, t1_1_b, t2_1_b, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('iab,b,jA,ijab->A', r_bbb_, e_vir_b, t1_1_b, t2_1_b, optimize=True)
                s[s_b:f_b] -= 1/4 *  lib.einsum('iab,i,jA,ijab->A', r_bbb_, e_occ_b, t1_1_b, t2_1_b, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('iab,j,jA,ijab->A', r_bbb_, e_occ_b, t1_1_b, t2_1_b, optimize=True)
                s[s_aaa:f_aaa] += lib.einsum('a,ia,AiBC->ABC', r_a, f_ov_a, t2_1_a, optimize=True)[:, ab_ind_a[0],
                    ab_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] -= lib.einsum('a,iB,ACai->ABC', r_a, t1_1_a, eris_ovvo, optimize=True)[:, ab_ind_a[0],
                    ab_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] += lib.einsum('a,iB,AiaC->ABC', r_a, t1_1_a, eris_oovv, optimize=True)[:, ab_ind_a[0],
                    ab_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] += lib.einsum('a,iC,ABai->ABC', r_a, t1_1_a, eris_ovvo, optimize=True)[:, ab_ind_a[0],
                    ab_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] -= lib.einsum('a,iC,AiaB->ABC', r_a, t1_1_a, eris_oovv, optimize=True)[:, ab_ind_a[0],
                    ab_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] += lib.einsum('a,ia,ABCi->ABC', r_a, t1_1_a, eris_ovvo, optimize=True)[:, ab_ind_a[0],
                    ab_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] -= lib.einsum('a,ia,ACBi->ABC', r_a, t1_1_a, eris_ovvo, optimize=True)[:, ab_ind_a[0],
                    ab_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] += lib.einsum('a,AiBb,iaCb->ABC', r_a, t2_1_a, eris_ovvv, optimize=True)[:,
                    ab_ind_a[0], ab_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] -= lib.einsum('a,AiBb,ibCa->ABC', r_a, t2_1_a, eris_ovvv, optimize=True)[:,
                    ab_ind_a[0], ab_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] -= lib.einsum('a,AiCb,iaBb->ABC', r_a, t2_1_a, eris_ovvv, optimize=True)[:,
                    ab_ind_a[0], ab_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] += lib.einsum('a,AiCb,ibBa->ABC', r_a, t2_1_a, eris_ovvv, optimize=True)[:,
                    ab_ind_a[0], ab_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] -= 1/2 *  lib.einsum('a,ijBC,jaiA->ABC', r_a, t2_1_a, eris_ovoo, optimize=True)[:,
                    ab_ind_a[0], ab_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] += 1/2 *  lib.einsum('a,ijBC,iajA->ABC', r_a, t2_1_a, eris_ovoo, optimize=True)[:,
                    ab_ind_a[0], ab_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] -= lib.einsum('a,AiBb,ibCa->ABC', r_a, t2_1_ab, eris_OVvv, optimize=True)[:,
                    ab_ind_a[0], ab_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] += lib.einsum('a,AiCb,ibBa->ABC', r_a, t2_1_ab, eris_OVvv, optimize=True)[:,
                    ab_ind_a[0], ab_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] -= 1/2 *  lib.einsum('a,A,ia,AiBC->ABC', r_a, e_occ_a, t1_1_a, t2_1_a,
                    optimize=True)[:, ab_ind_a[0], ab_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] += 1/2 *  lib.einsum('a,B,ia,AiBC->ABC', r_a, e_vir_a, t1_1_a, t2_1_a,
                    optimize=True)[:, ab_ind_a[0], ab_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] += 1/2 *  lib.einsum('a,C,ia,AiBC->ABC', r_a, e_vir_a, t1_1_a, t2_1_a,
                    optimize=True)[:, ab_ind_a[0], ab_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] += 1/2 *  lib.einsum('a,a,ia,AiBC->ABC', r_a, e_vir_a, t1_1_a, t2_1_a,
                    optimize=True)[:, ab_ind_a[0], ab_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] -= lib.einsum('a,i,ia,AiBC->ABC', r_a, e_occ_a, t1_1_a, t2_1_a, optimize=True)[:,
                    ab_ind_a[0], ab_ind_a[1]].reshape(-1)
                s[s_bab:f_bab] -= lib.einsum('a,ia,iABC->ABC', r_a, f_ov_a, t2_1_ab, optimize=True).reshape(-1)
                s[s_bab:f_bab] -= lib.einsum('a,iB,ACai->ABC', r_a, t1_1_a, eris_OVvo, optimize=True).reshape(-1)
                s[s_bab:f_bab] -= lib.einsum('a,ia,ACBi->ABC', r_a, t1_1_a, eris_OVvo, optimize=True).reshape(-1)
                s[s_bab:f_bab] -= lib.einsum('a,iC,AiaB->ABC', r_a, t1_1_b, eris_OOvv, optimize=True).reshape(-1)
                s[s_bab:f_bab] -= lib.einsum('a,iABb,iaCb->ABC', r_a, t2_1_ab, eris_ovVV, optimize=True).reshape(-1)
                s[s_bab:f_bab] -= lib.einsum('a,iAbC,iaBb->ABC', r_a, t2_1_ab, eris_ovvv, optimize=True).reshape(-1)
                s[s_bab:f_bab] += lib.einsum('a,iAbC,ibBa->ABC', r_a, t2_1_ab, eris_ovvv, optimize=True).reshape(-1)
                s[s_bab:f_bab] += lib.einsum('a,ijBC,iajA->ABC', r_a, t2_1_ab, eris_ovOO, optimize=True).reshape(-1)
                s[s_bab:f_bab] += lib.einsum('a,AiCb,ibBa->ABC', r_a, t2_1_b, eris_OVvv, optimize=True).reshape(-1)
                s[s_bab:f_bab] += 1/2 *  lib.einsum('a,A,ia,iABC->ABC', r_a, e_occ_b, t1_1_a, t2_1_ab,
                    optimize=True).reshape(-1)
                s[s_bab:f_bab] -= 1/2 *  lib.einsum('a,B,ia,iABC->ABC', r_a, e_vir_a, t1_1_a, t2_1_ab,
                    optimize=True).reshape(-1)
                s[s_bab:f_bab] -= 1/2 *  lib.einsum('a,C,ia,iABC->ABC', r_a, e_vir_b, t1_1_a, t2_1_ab,
                    optimize=True).reshape(-1)
                s[s_bab:f_bab] -= 1/2 *  lib.einsum('a,a,ia,iABC->ABC', r_a, e_vir_a, t1_1_a, t2_1_ab,
                    optimize=True).reshape(-1)
                s[s_bab:f_bab] += lib.einsum('a,i,ia,iABC->ABC', r_a, e_occ_a, t1_1_a, t2_1_ab,
                    optimize=True).reshape(-1)
                s[s_aba:f_aba] -= lib.einsum('a,ia,AiCB->ABC', r_b, f_ov_b, t2_1_ab, optimize=True).reshape(-1)
                s[s_aba:f_aba] -= lib.einsum('a,iC,AiaB->ABC', r_b, t1_1_a, eris_ooVV, optimize=True).reshape(-1)
                s[s_aba:f_aba] -= lib.einsum('a,iB,ACai->ABC', r_b, t1_1_b, eris_ovVO, optimize=True).reshape(-1)
                s[s_aba:f_aba] -= lib.einsum('a,ia,ACBi->ABC', r_b, t1_1_b, eris_ovVO, optimize=True).reshape(-1)
                s[s_aba:f_aba] += lib.einsum('a,AiCb,ibBa->ABC', r_b, t2_1_a, eris_ovVV, optimize=True).reshape(-1)
                s[s_aba:f_aba] -= lib.einsum('a,AiCb,iaBb->ABC', r_b, t2_1_ab, eris_OVVV, optimize=True).reshape(-1)
                s[s_aba:f_aba] += lib.einsum('a,AiCb,ibBa->ABC', r_b, t2_1_ab, eris_OVVV, optimize=True).reshape(-1)
                s[s_aba:f_aba] -= lib.einsum('a,AibB,iaCb->ABC', r_b, t2_1_ab, eris_OVvv, optimize=True).reshape(-1)
                s[s_aba:f_aba] += lib.einsum('a,ijCB,jaiA->ABC', r_b, t2_1_ab, eris_OVoo, optimize=True).reshape(-1)
                s[s_aba:f_aba] += 1/2 *  lib.einsum('a,A,ia,AiCB->ABC', r_b, e_occ_a, t1_1_b, t2_1_ab,
                    optimize=True).reshape(-1)
                s[s_aba:f_aba] -= 1/2 *  lib.einsum('a,B,ia,AiCB->ABC', r_b, e_vir_b, t1_1_b, t2_1_ab,
                    optimize=True).reshape(-1)
                s[s_aba:f_aba] -= 1/2 *  lib.einsum('a,C,ia,AiCB->ABC', r_b, e_vir_a, t1_1_b, t2_1_ab,
                    optimize=True).reshape(-1)
                s[s_aba:f_aba] -= 1/2 *  lib.einsum('a,a,ia,AiCB->ABC', r_b, e_vir_b, t1_1_b, t2_1_ab,
                    optimize=True).reshape(-1)
                s[s_aba:f_aba] += lib.einsum('a,i,ia,AiCB->ABC', r_b, e_occ_b, t1_1_b, t2_1_ab,
                    optimize=True).reshape(-1)
                s[s_bbb:f_bbb] += lib.einsum('a,ia,AiBC->ABC', r_b, f_ov_b, t2_1_b, optimize=True)[:, ab_ind_b[0],
                    ab_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] -= lib.einsum('a,iB,ACai->ABC', r_b, t1_1_b, eris_OVVO, optimize=True)[:, ab_ind_b[0],
                    ab_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] += lib.einsum('a,iB,AiaC->ABC', r_b, t1_1_b, eris_OOVV, optimize=True)[:, ab_ind_b[0],
                    ab_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] += lib.einsum('a,iC,ABai->ABC', r_b, t1_1_b, eris_OVVO, optimize=True)[:, ab_ind_b[0],
                    ab_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] -= lib.einsum('a,iC,AiaB->ABC', r_b, t1_1_b, eris_OOVV, optimize=True)[:, ab_ind_b[0],
                    ab_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] += lib.einsum('a,ia,ABCi->ABC', r_b, t1_1_b, eris_OVVO, optimize=True)[:, ab_ind_b[0],
                    ab_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] -= lib.einsum('a,ia,ACBi->ABC', r_b, t1_1_b, eris_OVVO, optimize=True)[:, ab_ind_b[0],
                    ab_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] -= lib.einsum('a,iAbB,ibCa->ABC', r_b, t2_1_ab, eris_ovVV, optimize=True)[:,
                    ab_ind_b[0], ab_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] += lib.einsum('a,iAbC,ibBa->ABC', r_b, t2_1_ab, eris_ovVV, optimize=True)[:,
                    ab_ind_b[0], ab_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] += lib.einsum('a,AiBb,iaCb->ABC', r_b, t2_1_b, eris_OVVV, optimize=True)[:,
                    ab_ind_b[0], ab_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] -= lib.einsum('a,AiBb,ibCa->ABC', r_b, t2_1_b, eris_OVVV, optimize=True)[:,
                    ab_ind_b[0], ab_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] -= lib.einsum('a,AiCb,iaBb->ABC', r_b, t2_1_b, eris_OVVV, optimize=True)[:,
                    ab_ind_b[0], ab_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] += lib.einsum('a,AiCb,ibBa->ABC', r_b, t2_1_b, eris_OVVV, optimize=True)[:,
                    ab_ind_b[0], ab_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] -= 1/2 *  lib.einsum('a,ijBC,jaiA->ABC', r_b, t2_1_b, eris_OVOO, optimize=True)[:,
                    ab_ind_b[0], ab_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] += 1/2 *  lib.einsum('a,ijBC,iajA->ABC', r_b, t2_1_b, eris_OVOO, optimize=True)[:,
                    ab_ind_b[0], ab_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] -= 1/2 *  lib.einsum('a,A,ia,AiBC->ABC', r_b, e_occ_b, t1_1_b, t2_1_b,
                    optimize=True)[:, ab_ind_b[0], ab_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] += 1/2 *  lib.einsum('a,B,ia,AiBC->ABC', r_b, e_vir_b, t1_1_b, t2_1_b,
                    optimize=True)[:, ab_ind_b[0], ab_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] += 1/2 *  lib.einsum('a,C,ia,AiBC->ABC', r_b, e_vir_b, t1_1_b, t2_1_b,
                    optimize=True)[:, ab_ind_b[0], ab_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] += 1/2 *  lib.einsum('a,a,ia,AiBC->ABC', r_b, e_vir_b, t1_1_b, t2_1_b,
                    optimize=True)[:, ab_ind_b[0], ab_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] -= lib.einsum('a,i,ia,AiBC->ABC', r_b, e_occ_b, t1_1_b, t2_1_b, optimize=True)[:,
                    ab_ind_b[0], ab_ind_b[1]].reshape(-1)

                if isinstance(eris.vvvv_p, np.ndarray):
                    va = adc.mo_coeff[0][:, nocc_a:]
                    vb = adc.mo_coeff[1][:, nocc_b:]
                    v_eeee_aaaa = ao2mo.general(adc._scf._eri, (va, va, va, va),
                        compact=False).reshape(nvir_a, nvir_a, nvir_a, nvir_a)
                    s[s_a:f_a] += lib.einsum('iab,ic,Aacb->A', r_aaa_, t1_1_a, v_eeee_aaaa, optimize=True)
                    s[s_aaa:f_aaa] += lib.einsum('a,Ab,BaCb->ABC', r_a, t1_1_a, v_eeee_aaaa, optimize=True)[:,
                        ab_ind_a[0], ab_ind_a[1]].reshape(-1)
                    s[s_aaa:f_aaa] -= lib.einsum('a,Ab,BbCa->ABC', r_a, t1_1_a, v_eeee_aaaa, optimize=True)[:,
                        ab_ind_a[0], ab_ind_a[1]].reshape(-1)
                    del v_eeee_aaaa
                    v_eeee_abab = eris.vVvV_p.reshape(nvir_a, nvir_b, nvir_a, nvir_b)
                    s[s_a:f_a] += lib.einsum('iab,ic,Acab->A', r_bab, t1_1_b, v_eeee_abab, optimize=True)
                    s[s_b:f_b] += lib.einsum('iab,ic,cAba->A', r_aba, t1_1_a, v_eeee_abab, optimize=True)
                    s[s_bab:f_bab] += lib.einsum('a,Ab,BCab->ABC', r_a, t1_1_b, v_eeee_abab, optimize=True).reshape(-1)
                    s[s_aba:f_aba] += lib.einsum('a,Ab,CBba->ABC', r_b, t1_1_a, v_eeee_abab, optimize=True).reshape(-1)
                    del v_eeee_abab
                    v_eeee_bbbb = ao2mo.general(adc._scf._eri, (vb, vb, vb, vb),
                        compact=False).reshape(nvir_b, nvir_b, nvir_b, nvir_b)
                    s[s_b:f_b] += lib.einsum('iab,ic,Aacb->A', r_bbb_, t1_1_b, v_eeee_bbbb, optimize=True)
                    s[s_bbb:f_bbb] += lib.einsum('a,Ab,BaCb->ABC', r_b, t1_1_b, v_eeee_bbbb, optimize=True)[:,
                        ab_ind_b[0], ab_ind_b[1]].reshape(-1)
                    s[s_bbb:f_bbb] -= lib.einsum('a,Ab,BbCa->ABC', r_b, t1_1_b, v_eeee_bbbb, optimize=True)[:,
                        ab_ind_b[0], ab_ind_b[1]].reshape(-1)
                    del v_eeee_bbbb
                elif isinstance(eris.vvvv_p, list):
                    mol = adc.mol
                    va = adc.mo_coeff[0][:, nocc_a:]
                    vb = adc.mo_coeff[1][:, nocc_b:]
                    chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                    s_aaa_view = s[s_aaa:f_aaa].reshape(nocc_a, -1)
                    s_bab_view = s[s_bab:f_bab].reshape(nocc_b, nvir_a, nvir_b)
                    s_aba_view = s[s_aba:f_aba].reshape(nocc_a, nvir_b, nvir_a)
                    s_bbb_view = s[s_bbb:f_bbb].reshape(nocc_b, -1)
                    for p, q in lib.prange(0, nvir_a, chnk_size):
                        v_eeee_aaaa = ao2mo.general(mol, (va[:, p:q], va, va, va),
                            compact=False).reshape(q-p, nvir_a, nvir_a, nvir_a)
                        s[s_a+p:s_a+q] += lib.einsum('iab,ic,Aacb->A', r_aaa_, t1_1_a, v_eeee_aaaa, optimize=True)
                        p0 = p*(p-1)//2
                        q0 = q*(q-1)//2
                        rows = ab_ind_a[0][p0:q0] - p
                        cols = ab_ind_a[1][p0:q0]
                        temp = lib.einsum('a,Ab,BaCb->ABC', r_a, t1_1_a, v_eeee_aaaa, optimize=True)
                        s_aaa_view[:, p0:q0] += temp[:, rows, cols]
                        temp = lib.einsum('a,Ab,BbCa->ABC', r_a, t1_1_a, v_eeee_aaaa, optimize=True)
                        s_aaa_view[:, p0:q0] -= temp[:, rows, cols]
                        del v_eeee_aaaa
                    p = 0
                    for dataset in eris.vVvV_p:
                        k = dataset.shape[0]
                        v_eeee_abab = dataset[:].reshape(k, nvir_b, nvir_a, nvir_b)
                        s[s_a+p:s_a+k] += lib.einsum('iab,ic,Acab->A', r_bab, t1_1_b, v_eeee_abab, optimize=True)
                        s_bab_view[:, p:p+k] += lib.einsum('a,Ab,BCab->ABC', r_a, t1_1_b, v_eeee_abab, optimize=True)
                        s_aba_view[:, :, p:p+k] += lib.einsum('a,Ab,CBba->ABC', r_b, t1_1_a, v_eeee_abab,
                            optimize=True)
                        del v_eeee_abab
                        p += k
                    p = 0
                    for dataset in eris.VvVv_p:
                        k = dataset.shape[0]
                        v_eeee_abab = dataset[:].reshape(k, nvir_a, nvir_b, nvir_a)
                        s[s_b+p:s_b+k] += lib.einsum('iab,ic,Acab->A', r_aba, t1_1_a, v_eeee_abab, optimize=True)
                        del v_eeee_abab
                        p += k
                    for p, q in lib.prange(0, nvir_b, chnk_size):
                        v_eeee_bbbb = ao2mo.general(mol, (vb[:, p:q], vb, vb, vb),
                            compact=False).reshape(q-p, nvir_b, nvir_b, nvir_b)
                        s[s_b+p:s_b+q] += lib.einsum('iab,ic,Aacb->A', r_bbb_, t1_1_b, v_eeee_bbbb, optimize=True)
                        p0 = p*(p-1)//2
                        q0 = q*(q-1)//2
                        rows = ab_ind_b[0][p0:q0] - p
                        cols = ab_ind_b[1][p0:q0]
                        temp = lib.einsum('a,Ab,BaCb->ABC', r_b, t1_1_b, v_eeee_bbbb, optimize=True)
                        s_bbb_view[:, p0:q0] += temp[:, rows, cols]
                        temp = lib.einsum('a,Ab,BbCa->ABC', r_b, t1_1_b, v_eeee_bbbb, optimize=True)
                        s_bbb_view[:, p0:q0] -= temp[:, rows, cols]
                        del v_eeee_bbbb
                elif eris.vvvv_p is None:
                    temp = lib.einsum('ic,Pcb->iPb', t1_1_a, eris.Lvv, optimize=True)
                    temp = lib.einsum('iab,iPb->Pab', r_aaa_, temp, optimize=True)
                    s[s_a:f_a] += lib.einsum('PAa,Pab->A', eris.Lvv, temp, optimize=True)
                    temp = lib.einsum('ic,Pcb->iPb', t1_1_b, eris.LVV, optimize=True)
                    temp = lib.einsum('iab,iPb->Pab', r_bab, temp, optimize=True)
                    s[s_a:f_a] += lib.einsum('PAa,Pab->A', eris.Lvv, temp, optimize=True)
                    temp = lib.einsum('ic,Pcb->iPb', t1_1_a, eris.Lvv, optimize=True)
                    temp = lib.einsum('iab,iPb->Pab', r_aba, temp, optimize=True)
                    s[s_b:f_b] += lib.einsum('PAa,Pab->A', eris.LVV, temp, optimize=True)
                    temp = lib.einsum('ic,Pcb->iPb', t1_1_b, eris.LVV, optimize=True)
                    temp = lib.einsum('iab,iPb->Pab', r_bbb_, temp, optimize=True)
                    s[s_b:f_b] += lib.einsum('PAa,Pab->A', eris.LVV, temp, optimize=True)
                    temp = lib.einsum('a,PBa->PB', r_a, eris.Lvv, optimize=True)
                    temp1 = lib.einsum('Ab,PCb->PAC', t1_1_a, eris.Lvv, optimize=True)
                    s[s_aaa:f_aaa] += lib.einsum('PB,PAC->ABC', temp, temp1, optimize=True)[:,
                        ab_ind_a[0], ab_ind_a[1]].reshape(-1)
                    temp = lib.einsum('a,PCa->PC', r_a, eris.Lvv, optimize=True)
                    temp1 = lib.einsum('Ab,PBb->PAB', t1_1_a, eris.Lvv, optimize=True)
                    s[s_aaa:f_aaa] -= lib.einsum('PC,PAB->ABC', temp, temp1, optimize=True)[:,
                        ab_ind_a[0], ab_ind_a[1]].reshape(-1)
                    temp = lib.einsum('a,PBa->PB', r_a, eris.Lvv, optimize=True)
                    temp1 = lib.einsum('Ab,PCb->PAC', t1_1_b, eris.LVV, optimize=True)
                    s[s_bab:f_bab] += lib.einsum('PB,PAC->ABC', temp, temp1, optimize=True).reshape(-1)
                    temp = lib.einsum('a,PBa->PB', r_b, eris.LVV, optimize=True)
                    temp1 = lib.einsum('Ab,PCb->PAC', t1_1_a, eris.Lvv, optimize=True)
                    s[s_aba:f_aba] += lib.einsum('PB,PAC->ABC', temp, temp1, optimize=True).reshape(-1)
                    temp = lib.einsum('a,PBa->PB', r_b, eris.LVV, optimize=True)
                    temp1 = lib.einsum('Ab,PCb->PAC', t1_1_b, eris.LVV, optimize=True)
                    s[s_bbb:f_bbb] += lib.einsum('PB,PAC->ABC', temp, temp1, optimize=True)[:,
                        ab_ind_b[0], ab_ind_b[1]].reshape(-1)
                    temp = lib.einsum('a,PCa->PC', r_b, eris.LVV, optimize=True)
                    temp1 = lib.einsum('Ab,PBb->PAB', t1_1_b, eris.LVV, optimize=True)
                    s[s_bbb:f_bbb] -= lib.einsum('PC,PAB->ABC', temp, temp1, optimize=True)[:,
                        ab_ind_b[0], ab_ind_b[1]].reshape(-1)

            del t2_1_a
            del t2_1_b
            del t2_1_ab

        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)
        return s

    return sigma_


def get_trans_moments(adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)
    nmo_a  = adc.nmo_a
    nmo_b  = adc.nmo_b

    T_a = []
    T_b = []

    for orb in range(nmo_a):
        T_aa = get_trans_moments_orbital(adc,orb, spin="alpha")
        T_a.append(T_aa)

    for orb in range(nmo_b):
        T_bb = get_trans_moments_orbital(adc,orb, spin="beta")
        T_b.append(T_bb)

    cput0 = log.timer_debug1("completed spec vector calc in ADC(3) calculation", *cput0)
    return (T_a, T_b)


def get_trans_moments_orbital(adc, orb, spin="alpha"):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
        t1_2_a, t1_2_b = adc.t1[0]

    t1_1_a = t1_1_b = None
    if adc.t1[2][0] is not None:
        t1_1_a = adc.t1[2][0]
        t1_1_b = adc.t1[2][1]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    n_singles_a = nvir_a
    n_singles_b = nvir_b
    n_doubles_aaa = nvir_a* (nvir_a - 1) * nocc_a // 2
    n_doubles_bab = nocc_b * nvir_a* nvir_b
    n_doubles_aba = nocc_a * nvir_b* nvir_a
    n_doubles_bbb = nvir_b* (nvir_b - 1) * nocc_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb

    T = np.zeros(dim)

######## spin = alpha  ############################################

    if spin == "alpha":
        # placehold

        ######## ADC(2) part  ############################################

        t2_1_a = adc.t2[0][0][:]
        t2_1_ab = adc.t2[0][1][:]
        if orb < nocc_a:

            if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
                T[s_a:f_a] = -t1_2_a[orb,:]

            if t1_1_a is not None:
                T[s_a:f_a] -= t1_1_a[orb,:]
                T[s_a:f_a] += 0.5*lib.einsum('kac,ck->a',t2_1_a[:,orb,:,:], t1_1_a.T,optimize=True)
                T[s_a:f_a] -= 0.5*lib.einsum('kac,ck->a',t2_1_ab[orb,:,:,:], t1_1_b.T,optimize=True)

            t2_1_t = t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]].copy()
            t2_1_ab_t = -t2_1_ab.transpose(1,0,2,3)

            T[s_aaa:f_aaa] += t2_1_t[:,orb,:].reshape(-1)
            T[s_bab:f_bab] += t2_1_ab_t[:,orb,:,:].reshape(-1)

        else:
            T[s_a:f_a] += idn_vir_a[(orb-nocc_a), :]
            T[s_a:f_a] -= 0.25*lib.einsum('klc,klac->a',t2_1_a[:,:,
                                          (orb-nocc_a),:], t2_1_a, optimize=True)
            T[s_a:f_a] -= 0.25*lib.einsum('klc,klac->a',t2_1_ab[:,:,
                                          (orb-nocc_a),:], t2_1_ab, optimize=True)
            T[s_a:f_a] -= 0.25*lib.einsum('lkc,lkac->a',t2_1_ab[:,:,
                                          (orb-nocc_a),:], t2_1_ab, optimize=True)
            if t1_1_a is not None:
                T[s_a:f_a] -= 0.5*lib.einsum('ka,k->a',t1_1_a, t1_1_a[:,(orb-nocc_a)], optimize=True)
######## ADC(3) 2p-1h  part  ############################################

        if (adc.method == "adc(2)-x" and adc.approx_trans_moments is False) or (adc.method == "adc(3)"):

            t2_2_a = adc.t2[1][0][:]
            t2_2_ab = adc.t2[1][1][:]

            if orb < nocc_a:

                t2_2_t = t2_2_a[:,:,ab_ind_a[0],ab_ind_a[1]].copy()
                t2_2_ab_t = -t2_2_ab.transpose(1,0,2,3)

                T[s_aaa:f_aaa] += t2_2_t[:,orb,:].reshape(-1)
                T[s_bab:f_bab] += t2_2_ab_t[:,orb,:,:].reshape(-1)

            if orb >= nocc_a:
                if t1_1_a is not None:
                    T[s_aaa:f_aaa] += 0.5*lib.einsum('k,ikac->aci',t1_1_a[:,(orb-nocc_a)], t2_1_a,
                        optimize=True)[ab_ind_a[0],ab_ind_a[1],:].transpose(1,0).reshape(-1)
                    T[s_bab:f_bab] -= 0.5*lib.einsum('k,kica->aci',t1_1_a[:,(orb-nocc_a)], t2_1_ab,
                        optimize=True).transpose(2,1,0).reshape(-1)

######### ADC(3) 1p part  ############################################

        if (method=='adc(3)'):

            if (adc.approx_trans_moments is False):
                t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_a:

                T[s_a:f_a] += 0.5*lib.einsum('kac,ck->a',t2_1_a[:,orb,:,:], t1_2_a.T,optimize=True)
                T[s_a:f_a] -= 0.5*lib.einsum('kac,ck->a',t2_1_ab[orb,:,:,:], t1_2_b.T,optimize=True)

                if (adc.approx_trans_moments is False):
                    T[s_a:f_a] -= t1_3_a[orb,:]

                if t1_1_a is not None:
                    t2_1_b = adc.t2[0][2][:]
                    T[s_a:f_a] += 0.5*lib.einsum('kac,ck->a',t2_2_a[:,orb,:,:], t1_1_a.T,optimize=True)
                    T[s_a:f_a] -= 0.5*lib.einsum('kac,ck->a',t2_2_ab[orb,:,:,:], t1_1_b.T,optimize=True)
                    T[s_a:f_a] += 1/6*lib.einsum('c,ka,kc->a',t1_1_a[orb,:], t1_1_a, t1_1_a,optimize=True)
                    T[s_a:f_a] += 1/12*lib.einsum('c,klcd,klad->a',t1_1_a[orb,:], t2_1_a, t2_1_a,optimize=True)
                    T[s_a:f_a] += 1/12*lib.einsum('ka,klcd,lcd->a',t1_1_a, t2_1_a, t2_1_a[orb,:,:,:],optimize=True)
                    T[s_a:f_a] -= 1/6*lib.einsum('kc,klcd,lad->a',t1_1_a, t2_1_a, t2_1_a[orb,:,:,:],optimize=True)
                    T[s_a:f_a] += 1/6*lib.einsum('c,klcd,klad->a',t1_1_a[orb,:], t2_1_ab, t2_1_ab,optimize=True)
                    T[s_a:f_a] += 1/6*lib.einsum('ka,klcd,lcd->a',t1_1_a, t2_1_ab, t2_1_ab[orb,:,:,:],optimize=True)
                    T[s_a:f_a] -= 1/6*lib.einsum('kc,klcd,lad->a',t1_1_a, t2_1_ab, t2_1_ab[orb,:,:,:],optimize=True)
                    T[s_a:f_a] -= 1/6*lib.einsum('kc,lad,lkdc->a',t1_1_b, t2_1_a[orb,:,:,:], t2_1_ab,optimize=True)
                    T[s_a:f_a] -= 1/6*lib.einsum('kc,lad,klcd->a',t1_1_b, t2_1_ab[orb,:,:,:], t2_1_b,optimize=True)
                    del t2_1_b

            else:

                T[s_a:f_a] -= 0.25*lib.einsum('klc,klac->a',
                                              t2_1_a[:,:,(orb-nocc_a),:], t2_2_a, optimize=True)
                T[s_a:f_a] -= 0.25*lib.einsum('klc,klac->a',
                                              t2_1_ab[:,:,(orb-nocc_a),:], t2_2_ab, optimize=True)
                T[s_a:f_a] -= 0.25*lib.einsum('lkc,lkac->a',
                                              t2_1_ab[:,:,(orb-nocc_a),:], t2_2_ab, optimize=True)

                T[s_a:f_a] -= 0.25*lib.einsum('klac,klc->a',t2_1_a,
                                              t2_2_a[:,:,(orb-nocc_a),:],optimize=True)
                T[s_a:f_a] -= 0.25*lib.einsum('klac,klc->a',t2_1_ab,
                                              t2_2_ab[:,:,(orb-nocc_a),:],optimize=True)
                T[s_a:f_a] -= 0.25*lib.einsum('lkac,lkc->a',t2_1_ab,
                                              t2_2_ab[:,:,(orb-nocc_a),:],optimize=True)

                if t1_1_a is not None:
                    T[s_a:f_a] -= 0.5*lib.einsum('ka,k->a',t1_1_a, t1_2_a[:,(orb-nocc_a)],optimize=True)
                    T[s_a:f_a] -= 0.5*lib.einsum('k,ka->a',t1_1_a[:,(orb-nocc_a)], t1_2_a,optimize=True)
                    T[s_a:f_a] -= 1/3*lib.einsum('ka,lc,klc->a',t1_1_a, t1_1_b, t2_1_ab[:,:,(orb-nocc_a),:],
                                                 optimize=True)
                    T[s_a:f_a] -= 1/6*lib.einsum('k,lc,klac->a',t1_1_a[:,(orb-nocc_a)], t1_1_b, t2_1_ab,optimize=True)
                    T[s_a:f_a] -= 1/6*lib.einsum('klac,k,lc->a',t2_1_a, t1_1_a[:,(orb-nocc_a)], t1_1_a,optimize=True)
                    T[s_a:f_a] -= 1/3*lib.einsum('klc,ka,lc->a',t2_1_a[:,:,(orb-nocc_a),:], t1_1_a, t1_1_a,
                                                 optimize=True)

                del t2_2_a
                del t2_2_ab

        del t2_1_a
        del t2_1_ab

######### spin = beta  ############################################

    else:
        # placehold

        t2_1_b = adc.t2[0][2][:]
        t2_1_ab = adc.t2[0][1][:]
        if orb < nocc_b:

            if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
                T[s_b:f_b] = -t1_2_b[orb,:]

            if t1_1_b is not None:
                T[s_b:f_b] -= t1_1_b[orb,:]
                T[s_b:f_b] += 0.5*lib.einsum('kac,ck->a',t2_1_b[:,orb,:,:], t1_1_b.T,optimize=True)
                T[s_b:f_b] -= 0.5*lib.einsum('kca,ck->a',t2_1_ab[:,orb,:,:], t1_1_a.T,optimize=True)

            t2_1_t = t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]].copy()
            t2_1_ab_t = -t2_1_ab.transpose(0,1,3,2)

            T[s_bbb:f_bbb] += t2_1_t[:,orb,:].reshape(-1)
            T[s_aba:f_aba] += t2_1_ab_t[:,orb,:,:].reshape(-1)

        else:

            T[s_b:f_b] += idn_vir_b[(orb-nocc_b), :]
            T[s_b:f_b] -= 0.25*lib.einsum('klc,klac->a',t2_1_b[:,:,
                                          (orb-nocc_b),:], t2_1_b, optimize=True)
            T[s_b:f_b] -= 0.25*lib.einsum('lkc,lkca->a',t2_1_ab[:,:,:,
                                          (orb-nocc_b)], t2_1_ab, optimize=True)
            T[s_b:f_b] -= 0.25*lib.einsum('lkc,lkca->a',t2_1_ab[:,:,:,
                                          (orb-nocc_b)], t2_1_ab, optimize=True)
            if t1_1_b is not None:
                T[s_b:f_b] -= 0.5*lib.einsum('ka,k->a',t1_1_b, t1_1_b[:,(orb-nocc_b)], optimize=True)

######### ADC(3) 2p-1h part  ############################################

        if (adc.method == "adc(2)-x" and adc.approx_trans_moments is False) or (adc.method == "adc(3)"):

            t2_2_ab = adc.t2[1][1][:]
            t2_2_b = adc.t2[1][2][:]

            if orb < nocc_b:

                t2_2_t = t2_2_b[:,:,ab_ind_b[0],ab_ind_b[1]].copy()
                t2_2_ab_t = -t2_2_ab.transpose(0,1,3,2)

                T[s_bbb:f_bbb] += t2_2_t[:,orb,:].reshape(-1)
                T[s_aba:f_aba] += t2_2_ab_t[:,orb,:,:].reshape(-1)

            if orb >= nocc_b:
                if t1_1_b is not None:
                    T[s_bbb:f_bbb] += 0.5*lib.einsum('k,ikac->aci',t1_1_b[:,(orb-nocc_b)], t2_1_b,
                        optimize=True)[ab_ind_b[0],ab_ind_b[1],:].transpose(1,0).reshape(-1)
                    T[s_aba:f_aba] -= 0.5*lib.einsum('k,ikac->aci',t1_1_b[:,(orb-nocc_b)], t2_1_ab,
                        optimize=True).transpose(2,1,0).reshape(-1)

######### ADC(2) 1p part  ############################################

        if (method=='adc(3)'):

            if (adc.approx_trans_moments is False):
                t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_b:

                T[s_b:f_b] += 0.5*lib.einsum('kac,ck->a',t2_1_b[:,orb,:,:], t1_2_b.T,optimize=True)
                T[s_b:f_b] -= 0.5*lib.einsum('kca,ck->a',t2_1_ab[:,orb,:,:], t1_2_a.T,optimize=True)

                if (adc.approx_trans_moments is False):
                    T[s_b:f_b] -= t1_3_b[orb,:]

                if t1_1_b is not None:
                    t2_1_a = adc.t2[0][0][:]
                    T[s_b:f_b] -= 0.5*lib.einsum('kca,ck->a',t2_2_ab[:,orb,:,:], t1_1_a.T,optimize=True)
                    T[s_b:f_b] += 0.5*lib.einsum('kac,ck->a',t2_2_b[:,orb,:,:], t1_1_b.T,optimize=True)
                    T[s_b:f_b] -= 1/6*lib.einsum('kc,klcd,lda->a',t1_1_a, t2_1_a, t2_1_ab[:,orb,:,:],optimize=True)
                    T[s_b:f_b] -= 1/6*lib.einsum('kc,klcd,lad->a',t1_1_a, t2_1_ab, t2_1_b[orb,:,:,:],optimize=True)
                    T[s_b:f_b] += 1/6*lib.einsum('c,ka,kc->a',t1_1_b[orb,:], t1_1_b, t1_1_b,optimize=True)
                    T[s_b:f_b] += 1/6*lib.einsum('c,kldc,klda->a',t1_1_b[orb,:], t2_1_ab, t2_1_ab,optimize=True)
                    T[s_b:f_b] += 1/6*lib.einsum('ka,lkcd,lcd->a',t1_1_b, t2_1_ab, t2_1_ab[:,orb,:,:],optimize=True)
                    T[s_b:f_b] -= 1/6*lib.einsum('kc,lkdc,lda->a',t1_1_b, t2_1_ab, t2_1_ab[:,orb,:,:],optimize=True)
                    T[s_b:f_b] += 1/12*lib.einsum('c,klcd,klad->a',t1_1_b[orb,:], t2_1_b, t2_1_b,optimize=True)
                    T[s_b:f_b] += 1/12*lib.einsum('ka,klcd,lcd->a',t1_1_b, t2_1_b, t2_1_b[orb,:,:,:],optimize=True)
                    T[s_b:f_b] -= 1/6*lib.einsum('kc,klcd,lad->a',t1_1_b, t2_1_b, t2_1_b[orb,:,:,:],optimize=True)
                    del t2_1_a

            else:

                T[s_b:f_b] -= 0.25*lib.einsum('klc,klac->a',
                                              t2_1_b[:,:,(orb-nocc_b),:], t2_2_b, optimize=True)
                T[s_b:f_b] -= 0.25*lib.einsum('lkc,lkca->a',
                                              t2_1_ab[:,:,:,(orb-nocc_b)], t2_2_ab, optimize=True)
                T[s_b:f_b] -= 0.25*lib.einsum('lkc,lkca->a',
                                              t2_1_ab[:,:,:,(orb-nocc_b)], t2_2_ab, optimize=True)

                T[s_b:f_b] -= 0.25*lib.einsum('klac,klc->a',t2_1_b,
                                              t2_2_b[:,:,(orb-nocc_b),:],optimize=True)
                T[s_b:f_b] -= 0.25*lib.einsum('lkca,lkc->a',t2_1_ab,
                                              t2_2_ab[:,:,:,(orb-nocc_b)],optimize=True)
                T[s_b:f_b] -= 0.25*lib.einsum('klca,klc->a',t2_1_ab,
                                              t2_2_ab[:,:,:,(orb-nocc_b)],optimize=True)

                if t1_1_b is not None:
                    T[s_b:f_b] -= 0.5*lib.einsum('ka,k->a',t1_1_b, t1_2_b[:,(orb-nocc_b)],optimize=True)
                    T[s_b:f_b] -= 0.5*lib.einsum('k,ka->a',t1_1_b[:,(orb-nocc_b)], t1_2_b,optimize=True)
                    T[s_b:f_b] -= 1/3*lib.einsum('kc,la,klc->a',t1_1_a, t1_1_b, t2_1_ab[:,:,:,(orb-nocc_b)],
                                                 optimize=True)
                    T[s_b:f_b] -= 1/6*lib.einsum('kc,l,klca->a',t1_1_a, t1_1_b[:,(orb-nocc_b)], t2_1_ab,optimize=True)
                    T[s_b:f_b] -= 1/6*lib.einsum('klac,k,lc->a',t2_1_b, t1_1_b[:,(orb-nocc_b)], t1_1_b,optimize=True)
                    T[s_b:f_b] -= 1/3*lib.einsum('klc,ka,lc->a',t2_1_b[:,:,(orb-nocc_b),:], t1_1_b, t1_1_b,
                                                 optimize=True)

                del t2_2_b
                del t2_2_ab

        del t2_1_b
        del t2_1_ab

    return T


def analyze_eigenvector(adc):

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    evec_print_tol = adc.evec_print_tol

    logger.info(adc, "Number of alpha occupied orbitals = %d", nocc_a)
    logger.info(adc, "Number of beta occupied orbitals = %d", nocc_b)
    logger.info(adc, "Number of alpha virtual orbitals =  %d", nvir_a)
    logger.info(adc, "Number of beta virtual orbitals =  %d", nvir_b)
    logger.info(adc, "Print eigenvector elements > %f\n", evec_print_tol)
    ab_a = np.tril_indices(nvir_a, k=-1)
    ab_b = np.tril_indices(nvir_b, k=-1)

    n_singles_a = nvir_a
    n_singles_b = nvir_b
    n_doubles_aaa = nvir_a* (nvir_a - 1) * nocc_a // 2
    n_doubles_bab = nocc_b * nvir_a* nvir_b
    n_doubles_aba = nocc_a * nvir_b* nvir_a
    n_doubles_bbb = nvir_b* (nvir_b - 1) * nocc_b // 2

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb

    U = adc.U

    for I in range(U.shape[1]):
        U1 = U[:f_b, I]
        U2 = U[f_b:, I]
        U1dotU1 = np.dot(U1, U1)
        U2dotU2 = np.dot(U2, U2)

        temp_aaa = np.zeros((nocc_a, nvir_a, nvir_a))
        temp_aaa[:,ab_a[0],ab_a[1]] =  U[s_aaa:f_aaa,I].reshape(nocc_a,-1).copy()
        temp_aaa[:,ab_a[1],ab_a[0]] = -U[s_aaa:f_aaa,I].reshape(nocc_a,-1).copy()
        U_aaa = temp_aaa.reshape(-1).copy()

        temp_bbb = np.zeros((nocc_b, nvir_b, nvir_b))
        temp_bbb[:,ab_b[0],ab_b[1]] =  U[s_bbb:f_bbb,I].reshape(nocc_b,-1).copy()
        temp_bbb[:,ab_b[1],ab_b[0]] = -U[s_bbb:f_bbb,I].reshape(nocc_b,-1).copy()
        U_bbb = temp_bbb.reshape(-1).copy()

        U_sq = U[:,I].copy()**2
        ind_idx = np.argsort(-U_sq)
        U_sq = U_sq[ind_idx]
        U_sorted = U[ind_idx,I].copy()

        U_sq_aaa = U_aaa.copy()**2
        U_sq_bbb = U_bbb.copy()**2
        ind_idx_aaa = np.argsort(-U_sq_aaa)
        ind_idx_bbb = np.argsort(-U_sq_bbb)
        U_sq_aaa = U_sq_aaa[ind_idx_aaa]
        U_sq_bbb = U_sq_bbb[ind_idx_bbb]
        U_sorted_aaa = U_aaa[ind_idx_aaa].copy()
        U_sorted_bbb = U_bbb[ind_idx_bbb].copy()

        U_sorted = U_sorted[U_sq > evec_print_tol**2]
        ind_idx = ind_idx[U_sq > evec_print_tol**2]
        U_sorted_aaa = U_sorted_aaa[U_sq_aaa > evec_print_tol**2]
        U_sorted_bbb = U_sorted_bbb[U_sq_bbb > evec_print_tol**2]
        ind_idx_aaa = ind_idx_aaa[U_sq_aaa > evec_print_tol**2]
        ind_idx_bbb = ind_idx_bbb[U_sq_bbb > evec_print_tol**2]

        singles_a_idx = []
        singles_b_idx = []
        doubles_aaa_idx = []
        doubles_bab_idx = []
        doubles_aba_idx = []
        doubles_bbb_idx = []
        singles_a_val = []
        singles_b_val = []
        doubles_bab_val = []
        doubles_aba_val = []
        iter_idx = 0
        for orb_idx in ind_idx:

            if orb_idx in range(s_a,f_a):
                a_idx = orb_idx + 1 + nocc_a
                singles_a_idx.append(a_idx)
                singles_a_val.append(U_sorted[iter_idx])

            if orb_idx in range(s_b,f_b):
                a_idx = orb_idx - s_b + 1 + nocc_b
                singles_b_idx.append(a_idx)
                singles_b_val.append(U_sorted[iter_idx])

            if orb_idx in range(s_bab,f_bab):
                iab_idx = orb_idx - s_bab
                ab_rem = iab_idx % (nvir_a*nvir_b)
                i_idx = iab_idx//(nvir_a*nvir_b)
                a_idx = ab_rem//nvir_b
                b_idx = ab_rem % nvir_b
                doubles_bab_idx.append((i_idx + 1, a_idx + 1 + nocc_a, b_idx + 1 + nocc_b))
                doubles_bab_val.append(U_sorted[iter_idx])

            if orb_idx in range(s_aba,f_aba):
                iab_idx = orb_idx - s_aba
                ab_rem = iab_idx % (nvir_b*nvir_a)
                i_idx = iab_idx//(nvir_b*nvir_a)
                a_idx = ab_rem//nvir_a
                b_idx = ab_rem % nvir_a
                doubles_aba_idx.append((i_idx + 1, a_idx + 1 + nocc_b, b_idx + 1 + nocc_a))
                doubles_aba_val.append(U_sorted[iter_idx])

            iter_idx += 1

        for orb_aaa in ind_idx_aaa:
            ab_rem = orb_aaa % (nvir_a*nvir_a)
            i_idx = orb_aaa//(nvir_a*nvir_a)
            a_idx = ab_rem//nvir_a
            b_idx = ab_rem % nvir_a
            doubles_aaa_idx.append((i_idx + 1, a_idx + 1 + nocc_a, b_idx + 1 + nocc_a))

        for orb_bbb in ind_idx_bbb:
            ab_rem = orb_bbb % (nvir_b*nvir_b)
            i_idx = orb_bbb//(nvir_b*nvir_b)
            a_idx = ab_rem//nvir_b
            b_idx = ab_rem % nvir_b
            doubles_bbb_idx.append((i_idx + 1, a_idx + 1 + nocc_b, b_idx + 1 + nocc_b))

        doubles_aaa_val = list(U_sorted_aaa)
        doubles_bbb_val = list(U_sorted_bbb)

        logger.info(adc, '%s | root %d | Energy (eV) = %12.8f | norm(1p)  = %6.4f | norm(1h2p) = %6.4f ',
                    adc.method, I, adc.E[I]*HARTREE2EV, U1dotU1, U2dotU2)

        if singles_a_val:
            logger.info(adc, "\n1p(alpha) block: ")
            logger.info(adc, "     a     U(a)")
            logger.info(adc, "------------------")
            for idx, print_singles in enumerate(singles_a_idx):
                logger.info(adc, '  %4d   %7.4f', print_singles, singles_a_val[idx])

        if singles_b_val:
            logger.info(adc, "\n1p(beta) block: ")
            logger.info(adc, "     a     U(a)")
            logger.info(adc, "------------------")
            for idx, print_singles in enumerate(singles_b_idx):
                logger.info(adc, '  %4d   %7.4f', print_singles, singles_b_val[idx])

        if doubles_aaa_val:
            logger.info(adc, "\n1h2p(alpha|alpha|alpha) block: ")
            logger.info(adc, "     i     a     b     U(i,a,b)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_aaa_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f',
                            print_doubles[0], print_doubles[1], print_doubles[2], doubles_aaa_val[idx])

        if doubles_bab_val:
            logger.info(adc, "\n1h2p(beta|alpha|beta) block: ")
            logger.info(adc, "     i     a     b     U(i,a,b)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_bab_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f',
                            print_doubles[0], print_doubles[1], print_doubles[2], doubles_bab_val[idx])

        if doubles_aba_val:
            logger.info(adc, "\n1h2p(alpha|beta|alpha) block: ")
            logger.info(adc, "     i     a     b     U(i,a,b)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_aba_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f',
                            print_doubles[0], print_doubles[1], print_doubles[2], doubles_aba_val[idx])

        if doubles_bbb_val:
            logger.info(adc, "\n1h2p(beta|beta|beta) block: ")
            logger.info(adc, "     i     a     b     U(i,a,b)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_bbb_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f',
                            print_doubles[0], print_doubles[1], print_doubles[2], doubles_bbb_val[idx])

        logger.info(adc,
            "***************************************************************************************\n")


def analyze_spec_factor(adc):

    X_a = adc.X[0]
    X_b = adc.X[1]

    logger.info(adc, "Print spectroscopic factors > %E\n", adc.spec_factor_print_tol)

    X_tot = (X_a, X_b)

    for iter_idx, X in enumerate(X_tot):
        if iter_idx == 0:
            spin = "alpha"
        else:
            spin = "beta"

        X_2 = (X.copy()**2)

        thresh = adc.spec_factor_print_tol

        for i in range(X_2.shape[1]):

            sort = np.argsort(-X_2[:,i])
            X_2_row = X_2[:,i]

            X_2_row = X_2_row[sort]

            if not adc.mol.symmetry:
                sym = np.repeat(['A'], X_2_row.shape[0])
            else:
                if spin == "alpha":
                    sym = [symm.irrep_id2name(adc.mol.groupname, x)
                                              for x in adc._scf.mo_coeff[0].orbsym]
                    sym = np.array(sym)
                else:
                    sym = [symm.irrep_id2name(adc.mol.groupname, x)
                                              for x in adc._scf.mo_coeff[1].orbsym]
                    sym = np.array(sym)

                sym = sym[sort]

            spec_Contribution = X_2_row[X_2_row > thresh]
            index_mo = sort[X_2_row > thresh]+1

            if np.sum(spec_Contribution) == 0.0:
                continue

            logger.info(adc, '%s | root %d | Energy (eV) = %12.8f | %s\n',
                    adc.method, i, adc.E[i]*HARTREE2EV, spin)
            logger.info(adc, "     HF MO     Spec. Contribution     Orbital symmetry")
            logger.info(adc, "-----------------------------------------------------------")

            for c in range(index_mo.shape[0]):
                logger.info(adc, '     %3.d          %10.8f                %s',
                            index_mo[c], spec_Contribution[c], sym[c])

            logger.info(adc, '\nPartial spec. factor sum = %10.8f', np.sum(spec_Contribution))
            logger.info(adc,
            "***********************************************************\n")


def get_properties(adc, nroots=1):

    #Transition moments
    T = adc.get_trans_moments()

    T_a = T[0]
    T_b = T[1]

    T_a = np.array(T_a)
    T_b = np.array(T_b)

    U = adc.U

    #Spectroscopic amplitudes
    X_a = np.dot(T_a, U).reshape(-1,nroots)
    X_b = np.dot(T_b, U).reshape(-1,nroots)

    X = (X_a,X_b)

    #Spectroscopic factors
    P = lib.einsum("pi,pi->i", X_a, X_a)
    P += lib.einsum("pi,pi->i", X_b, X_b)

    return P, X


def analyze(myadc):

    header = ("\n*************************************************************"
              "\n           Eigenvector analysis summary"
              "\n*************************************************************")
    logger.info(myadc, header)

    myadc.analyze_eigenvector()

    if myadc.compute_properties:

        header = ("\n*************************************************************"
                  "\n            Spectroscopic factors analysis summary"
                  "\n*************************************************************")
        logger.info(myadc, header)

        myadc.analyze_spec_factor()


def compute_dyson_mo(myadc):

    X_a = myadc.X[0]
    X_b = myadc.X[1]

    if X_a is None:
        nroots = myadc.U.shape[1]
        P,X_a,X_b = myadc.get_properties(nroots)

    nroots = X_a.shape[1]
    dyson_mo_a = np.dot(myadc.mo_coeff[0],X_a)
    dyson_mo_b = np.dot(myadc.mo_coeff[1],X_b)

    dyson_mo = (dyson_mo_a,dyson_mo_b)

    return dyson_mo

def make_rdm1(adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)

    U = adc.U

    list_rdm1_a = []
    list_rdm1_b = []

    for i in range(U.shape[1]):
        rdm1_a, rdm1_b = make_rdm1_eigenvectors(adc, U[:,i], U[:,i])
        list_rdm1_a.append(rdm1_a)
        list_rdm1_b.append(rdm1_b)

    cput0 = log.timer_debug1("completed OPDM calculation", *cput0)
    return (list_rdm1_a, list_rdm1_b)

def make_rdm1_eigenvectors(adc, L, R):

    L = np.array(L).ravel()
    R = np.array(R).ravel()

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    nmo_a = nocc_a + nvir_a
    nmo_b = nocc_b + nvir_b

    occ_list_a = range(nocc_a)
    occ_list_b = range(nocc_b)

    t2_1_a = adc.t2[0][0][:]
    t2_1_ab = adc.t2[0][1][:]
    t2_1_b = adc.t2[0][2][:]
    if adc.t1[0][0] is not None:
        t1_2_a = adc.t1[0][0][:]
        t1_2_b = adc.t1[0][1][:]
    else:
        t1_2_a = np.zeros((nocc_a, nvir_a))
        t1_2_b = np.zeros((nocc_b, nvir_b))

    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)
    n_singles_a = nvir_a
    n_singles_b = nvir_b
    n_doubles_aaa = nvir_a * (nvir_a - 1) * nocc_a // 2
    n_doubles_bab = nocc_b * nvir_a * nvir_b
    n_doubles_aba = nocc_a * nvir_b * nvir_a
    n_doubles_bbb = nvir_b * (nvir_b - 1) * nocc_b // 2

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb

    rdm1_a  = np.zeros((nmo_a,nmo_a))
    rdm1_b  = np.zeros((nmo_b,nmo_b))

    L_a = L[s_a:f_a]
    L_b = L[s_b:f_b]
    L_aaa = L[s_aaa:f_aaa]
    L_bab = L[s_bab:f_bab]
    L_aba = L[s_aba:f_aba]
    L_bbb = L[s_bbb:f_bbb]

    R_a = R[s_a:f_a]
    R_b = R[s_b:f_b]
    R_aaa = R[s_aaa:f_aaa]
    R_bab = R[s_bab:f_bab]
    R_aba = R[s_aba:f_aba]
    R_bbb = R[s_bbb:f_bbb]

    L_aaa = L_aaa.reshape(nocc_a,-1)
    L_bbb = L_bbb.reshape(nocc_b,-1)
    L_aaa_u = None
    L_aaa_u = np.zeros((nocc_a,nvir_a,nvir_a))
    L_aaa_u[:,ab_ind_a[0],ab_ind_a[1]]= L_aaa.copy()
    L_aaa_u[:,ab_ind_a[1],ab_ind_a[0]]= -L_aaa.copy()

    L_bbb_u = None
    L_bbb_u = np.zeros((nocc_b,nvir_b,nvir_b))
    L_bbb_u[:,ab_ind_b[0],ab_ind_b[1]]= L_bbb.copy()
    L_bbb_u[:,ab_ind_b[1],ab_ind_b[0]]= -L_bbb.copy()

    L_aba = L_aba.reshape(nocc_a,nvir_b,nvir_a)
    L_bab = L_bab.reshape(nocc_b,nvir_a,nvir_b)


    R_aaa = R_aaa.reshape(nocc_a,-1)
    R_bbb = R_bbb.reshape(nocc_b,-1)
    R_aaa_u = None
    R_aaa_u = np.zeros((nocc_a,nvir_a,nvir_a))
    R_aaa_u[:,ab_ind_a[0],ab_ind_a[1]]= R_aaa.copy()
    R_aaa_u[:,ab_ind_a[1],ab_ind_a[0]]= -R_aaa.copy()

    R_bbb_u = None
    R_bbb_u = np.zeros((nocc_b,nvir_b,nvir_b))
    R_bbb_u[:,ab_ind_b[0],ab_ind_b[1]]= R_bbb.copy()
    R_bbb_u[:,ab_ind_b[1],ab_ind_b[0]]= -R_bbb.copy()

    R_aba = R_aba.reshape(nocc_a,nvir_b,nvir_a)
    R_bab = R_bab.reshape(nocc_b,nvir_a,nvir_b)

    t1_1_a = t1_1_b = None
    if adc.t1[2][0] is not None:
        t1_1_a = adc.t1[2][0]
        t1_1_b = adc.t1[2][1]

# block- ij
    rdm1_a[occ_list_a, occ_list_a] = lib.einsum('a,a->', L_a, R_a, optimize = True)
    rdm1_a[occ_list_a, occ_list_a] += lib.einsum('a,a->', L_b, R_b, optimize = True)
    rdm1_b[occ_list_b, occ_list_b] = lib.einsum('a,a->', L_a, R_a, optimize = True)
    rdm1_b[occ_list_b, occ_list_b] += lib.einsum('a,a->', L_b, R_b, optimize = True)
    rdm1_a[:nocc_a, :nocc_a] -= 1/2 * lib.einsum('a,a,Iibc,Jibc->IJ', L_a, R_a, t2_1_a, t2_1_a, optimize = True)
    rdm1_a[:nocc_a, :nocc_a] += lib.einsum('a,b,Iiac,Jibc->IJ', L_a, R_a, t2_1_a, t2_1_a, optimize = True)
    rdm1_a[:nocc_a, :nocc_a] -= lib.einsum('a,a,Iibc,Jibc->IJ', L_a, R_a, t2_1_ab, t2_1_ab, optimize = True)
    rdm1_a[:nocc_a, :nocc_a] += lib.einsum('a,b,Iiac,Jibc->IJ', L_a, R_a, t2_1_ab, t2_1_ab, optimize = True)
    rdm1_a[:nocc_a, :nocc_a] -= 1/2 * lib.einsum('a,a,Iibc,Jibc->IJ', L_b, R_b, t2_1_a, t2_1_a, optimize = True)
    rdm1_a[:nocc_a, :nocc_a] -= lib.einsum('a,a,Iibc,Jibc->IJ', L_b, R_b, t2_1_ab, t2_1_ab, optimize = True)
    rdm1_a[:nocc_a, :nocc_a] += lib.einsum('a,b,Iica,Jicb->IJ', L_b, R_b, t2_1_ab, t2_1_ab, optimize = True)
    rdm1_b[:nocc_b, :nocc_b] -= lib.einsum('a,a,iIbc,iJbc->IJ', L_a, R_a, t2_1_ab, t2_1_ab, optimize = True)
    rdm1_b[:nocc_b, :nocc_b] += lib.einsum('a,b,iIac,iJbc->IJ', L_a, R_a, t2_1_ab, t2_1_ab, optimize = True)
    rdm1_b[:nocc_b, :nocc_b] -= 1/2 * lib.einsum('a,a,Iibc,Jibc->IJ', L_a, R_a, t2_1_b, t2_1_b, optimize = True)
    rdm1_b[:nocc_b, :nocc_b] -= lib.einsum('a,a,iIbc,iJbc->IJ', L_b, R_b, t2_1_ab, t2_1_ab, optimize = True)
    rdm1_b[:nocc_b, :nocc_b] += lib.einsum('a,b,iIca,iJcb->IJ', L_b, R_b, t2_1_ab, t2_1_ab, optimize = True)
    rdm1_b[:nocc_b, :nocc_b] -= 1/2 * lib.einsum('a,a,Iibc,Jibc->IJ', L_b, R_b, t2_1_b, t2_1_b, optimize = True)
    rdm1_b[:nocc_b, :nocc_b] += lib.einsum('a,b,Iiac,Jibc->IJ', L_b, R_b, t2_1_b, t2_1_b, optimize = True)
    rdm1_a[:nocc_a, :nocc_a] -= 1/2 * lib.einsum('Jab,Iab->IJ', L_aaa_u, R_aaa_u, optimize = True)
    rdm1_a[:nocc_a, :nocc_a] -= lib.einsum('Jab,Iab->IJ', L_aba, R_aba, optimize = True)
    rdm1_a[occ_list_a, occ_list_a] += 1/2 * lib.einsum('iab,iab->', L_aaa_u, R_aaa_u, optimize = True)
    rdm1_a[occ_list_a, occ_list_a] += lib.einsum('iab,iab->', L_aba, R_aba, optimize = True)
    rdm1_a[occ_list_a, occ_list_a] += lib.einsum('iab,iab->', L_bab, R_bab, optimize = True)
    rdm1_a[occ_list_a, occ_list_a] += 1/2 * lib.einsum('iab,iab->', L_bbb_u, R_bbb_u, optimize = True)
    rdm1_b[:nocc_b, :nocc_b] -= lib.einsum('Jab,Iab->IJ', L_bab, R_bab, optimize = True)
    rdm1_b[:nocc_b, :nocc_b] -= 1/2 * lib.einsum('Jab,Iab->IJ', L_bbb_u, R_bbb_u, optimize = True)
    rdm1_b[occ_list_b, occ_list_b] += 1/2 * lib.einsum('iab,iab->', L_aaa_u, R_aaa_u, optimize = True)
    rdm1_b[occ_list_b, occ_list_b] += lib.einsum('iab,iab->', L_aba, R_aba, optimize = True)
    rdm1_b[occ_list_b, occ_list_b] += lib.einsum('iab,iab->', L_bab, R_bab, optimize = True)
    rdm1_b[occ_list_b, occ_list_b] += 1/2 * lib.einsum('iab,iab->', L_bbb_u, R_bbb_u, optimize = True)

    if t1_1_a is not None:
        rdm1_a[:nocc_a, :nocc_a] -= lib.einsum('a,a,Ib,Jb->IJ', L_a, R_a, t1_1_a, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, :nocc_a] += lib.einsum('a,b,Ia,Jb->IJ', L_a, R_a, t1_1_a, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, :nocc_a] -= lib.einsum('a,a,Ib,Jb->IJ', L_b, R_b, t1_1_a, t1_1_a, optimize=True)
        rdm1_b[:nocc_b, :nocc_b] -= lib.einsum('a,a,Ib,Jb->IJ', L_a, R_a, t1_1_b, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, :nocc_b] -= lib.einsum('a,a,Ib,Jb->IJ', L_b, R_b, t1_1_b, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, :nocc_b] += lib.einsum('a,b,Ia,Jb->IJ', L_b, R_b, t1_1_b, t1_1_b, optimize=True)
        temp = np.zeros((nocc_a, nocc_a))
        temp += lib.einsum('a,Iab,Jb->IJ', L_a, R_aaa_u, t1_1_a, optimize=True)
        temp += lib.einsum('Jab,a,Ib->IJ', L_aba, R_b, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, :nocc_a] -= temp + temp.T
        temp = np.zeros((nocc_b, nocc_b))
        temp += lib.einsum('a,Iab,Jb->IJ', L_a, R_bab, t1_1_b, optimize=True)
        temp += lib.einsum('a,Iab,Jb->IJ', L_b, R_bbb_u, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, :nocc_b] -= temp + temp.T

# block- ab
    rdm1_a[nocc_a:, nocc_a:] = lib.einsum('A,B->AB', L_a, R_a, optimize = True)
    rdm1_b[nocc_b:, nocc_b:] = lib.einsum('A,B->AB', L_b, R_b, optimize = True)
    temp = np.zeros((nvir_a, nvir_a))
    temp -= 1/4 * lib.einsum('A,a,ijab,ijBb->AB', L_a, R_a, t2_1_a, t2_1_a, optimize = True)
    temp -= 1/2 * lib.einsum('A,a,ijab,ijBb->AB', L_a, R_a, t2_1_ab, t2_1_ab, optimize = True)
    rdm1_a[nocc_a:, nocc_a:] += temp + temp.T
    rdm1_a[nocc_a:, nocc_a:] += 1/2 * lib.einsum('a,a,ijAb,ijBb->AB', L_a, R_a, t2_1_a, t2_1_a, optimize = True)
    rdm1_a[nocc_a:, nocc_a:] -= 1/2 * lib.einsum('a,b,ijBa,ijAb->AB', L_a, R_a, t2_1_a, t2_1_a, optimize = True)
    rdm1_a[nocc_a:, nocc_a:] += lib.einsum('a,a,ijAb,ijBb->AB', L_a, R_a, t2_1_ab, t2_1_ab, optimize = True)
    rdm1_a[nocc_a:, nocc_a:] += 1/2 * lib.einsum('a,a,ijAb,ijBb->AB', L_b, R_b, t2_1_a, t2_1_a, optimize = True)
    rdm1_a[nocc_a:, nocc_a:] += lib.einsum('a,a,ijAb,ijBb->AB', L_b, R_b, t2_1_ab, t2_1_ab, optimize = True)
    rdm1_a[nocc_a:, nocc_a:] -= lib.einsum('a,b,ijBa,ijAb->AB', L_b, R_b, t2_1_ab, t2_1_ab, optimize = True)
    rdm1_b[nocc_b:, nocc_b:] += lib.einsum('a,a,ijbA,ijbB->AB', L_a, R_a, t2_1_ab, t2_1_ab, optimize = True)
    rdm1_b[nocc_b:, nocc_b:] -= lib.einsum('a,b,ijaB,ijbA->AB', L_a, R_a, t2_1_ab, t2_1_ab, optimize = True)
    rdm1_b[nocc_b:, nocc_b:] += 1/2 * lib.einsum('a,a,ijAb,ijBb->AB', L_a, R_a, t2_1_b, t2_1_b, optimize = True)
    temp = np.zeros((nvir_b, nvir_b))
    temp -= 1/2 * lib.einsum('A,a,ijba,ijbB->AB', L_b, R_b, t2_1_ab, t2_1_ab, optimize = True)
    temp -= 1/4 * lib.einsum('A,a,ijab,ijBb->AB', L_b, R_b, t2_1_b, t2_1_b, optimize = True)
    rdm1_b[nocc_b:, nocc_b:] += temp + temp.T
    rdm1_b[nocc_b:, nocc_b:] += lib.einsum('a,a,ijbA,ijbB->AB', L_b, R_b, t2_1_ab, t2_1_ab, optimize = True)
    rdm1_b[nocc_b:, nocc_b:] += 1/2 * lib.einsum('a,a,ijAb,ijBb->AB', L_b, R_b, t2_1_b, t2_1_b, optimize = True)
    rdm1_b[nocc_b:, nocc_b:] -= 1/2 * lib.einsum('a,b,ijBa,ijAb->AB', L_b, R_b, t2_1_b, t2_1_b, optimize = True)
    rdm1_a[nocc_a:, nocc_a:] += lib.einsum('iAa,iBa->AB', L_aaa_u, R_aaa_u, optimize = True)
    rdm1_a[nocc_a:, nocc_a:] += lib.einsum('iaA,iaB->AB', L_aba, R_aba, optimize = True)
    rdm1_a[nocc_a:, nocc_a:] += lib.einsum('iAa,iBa->AB', L_bab, R_bab, optimize = True)
    rdm1_b[nocc_b:, nocc_b:] += lib.einsum('iAa,iBa->AB', L_aba, R_aba, optimize = True)
    rdm1_b[nocc_b:, nocc_b:] += lib.einsum('iaA,iaB->AB', L_bab, R_bab, optimize = True)
    rdm1_b[nocc_b:, nocc_b:] += lib.einsum('iAa,iBa->AB', L_bbb_u, R_bbb_u, optimize = True)

    if t1_1_a is not None:
        temp = 1/2 * lib.einsum('A,a,ia,iB->AB', L_a, R_a, t1_1_a, t1_1_a, optimize=True)
        rdm1_a[nocc_a:, nocc_a:] -= temp + temp.T
        rdm1_a[nocc_a:, nocc_a:] += lib.einsum('a,a,iA,iB->AB', L_a, R_a, t1_1_a, t1_1_a, optimize=True)
        rdm1_a[nocc_a:, nocc_a:] += lib.einsum('a,a,iA,iB->AB', L_b, R_b, t1_1_a, t1_1_a, optimize=True)
        rdm1_b[nocc_b:, nocc_b:] += lib.einsum('a,a,iA,iB->AB', L_a, R_a, t1_1_b, t1_1_b, optimize=True)
        temp = 1/2 * lib.einsum('A,a,ia,iB->AB', L_b, R_b, t1_1_b, t1_1_b, optimize=True)
        rdm1_b[nocc_b:, nocc_b:] -= temp + temp.T
        rdm1_b[nocc_b:, nocc_b:] += lib.einsum('a,a,iA,iB->AB', L_b, R_b, t1_1_b, t1_1_b, optimize=True)
        temp = np.zeros((nvir_a, nvir_a))
        temp -= lib.einsum('a,iBa,iA->AB', L_a, R_aaa_u, t1_1_a, optimize=True)
        temp += lib.einsum('iaA,a,iB->AB', L_aba, R_b, t1_1_a, optimize=True)
        rdm1_a[nocc_a:, nocc_a:] += temp + temp.T
        temp = np.zeros((nvir_b, nvir_b))
        temp += lib.einsum('a,iaB,iA->AB', L_a, R_bab, t1_1_b, optimize=True)
        temp -= lib.einsum('a,iBa,iA->AB', L_b, R_bbb_u, t1_1_b, optimize=True)
        rdm1_b[nocc_b:, nocc_b:] += temp + temp.T

# G^100#### block- ia
    rdm1_a[:nocc_a, nocc_a:] =- lib.einsum('a,IAa->IA', L_a, R_aaa_u, optimize = True)
    rdm1_a[:nocc_a, nocc_a:] += lib.einsum('a,IaA->IA', L_b, R_aba, optimize = True)

    rdm1_b[:nocc_b, nocc_b:]  = lib.einsum('a,IaA->IA', L_a, R_bab, optimize = True)
    rdm1_b[:nocc_b, nocc_b:] -= lib.einsum('a,IAa->IA', L_b, R_bbb_u, optimize = True)

    rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('iab,A,Iiab->IA', L_aaa_u, R_a, t2_1_a, optimize = True)
    rdm1_a[:nocc_a, nocc_a:] += lib.einsum('iab,a,IiAb->IA', L_aaa_u, R_a, t2_1_a, optimize = True)
    rdm1_a[:nocc_a, nocc_a:] += lib.einsum('iab,a,IiAb->IA', L_aba, R_b, t2_1_a, optimize = True)
    rdm1_a[:nocc_a, nocc_a:] -= lib.einsum('iab,A,Iiab->IA', L_bab, R_a, t2_1_ab, optimize = True)
    rdm1_a[:nocc_a, nocc_a:] += lib.einsum('iab,a,IiAb->IA', L_bab, R_a, t2_1_ab, optimize = True)
    rdm1_a[:nocc_a, nocc_a:] += lib.einsum('iab,a,IiAb->IA', L_bbb_u, R_b, t2_1_ab, optimize = True)

    rdm1_b[:nocc_b, nocc_b:] += lib.einsum('iab,a,iIbA->IA', L_aaa_u, R_a, t2_1_ab, optimize = True)
    rdm1_b[:nocc_b, nocc_b:] -= lib.einsum('iab,A,iIba->IA', L_aba, R_b, t2_1_ab, optimize = True)
    rdm1_b[:nocc_b, nocc_b:] += lib.einsum('iab,a,iIbA->IA', L_aba, R_b, t2_1_ab, optimize = True)
    rdm1_b[:nocc_b, nocc_b:] += lib.einsum('iab,a,IiAb->IA', L_bab, R_a, t2_1_b, optimize = True)
    rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('iab,A,Iiab->IA', L_bbb_u, R_b, t2_1_b, optimize = True)
    rdm1_b[:nocc_b, nocc_b:] += lib.einsum('iab,a,IiAb->IA', L_bbb_u, R_b, t2_1_b, optimize = True)

    if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
        rdm1_a[:nocc_a, nocc_a:] -= lib.einsum('a,A,Ia->IA', L_a, R_a, t1_2_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += lib.einsum('a,a,IA->IA', L_a, R_a, t1_2_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += lib.einsum('a,a,IA->IA', L_b, R_b, t1_2_a, optimize = True)

        rdm1_b[:nocc_b, nocc_b:] += lib.einsum('a,a,IA->IA', L_a, R_a, t1_2_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= lib.einsum('a,A,Ia->IA', L_b, R_b, t1_2_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += lib.einsum('a,a,IA->IA', L_b, R_b, t1_2_b, optimize = True)

    if t1_1_a is not None:
        rdm1_a[:nocc_a, nocc_a:] -= lib.einsum('a,A,Ia->IA', L_a, R_a, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += lib.einsum('a,a,IA->IA', L_a, R_a, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += lib.einsum('a,a,IA->IA', L_b, R_b, t1_1_a, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += lib.einsum('a,a,IA->IA', L_a, R_a, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= lib.einsum('a,A,Ia->IA', L_b, R_b, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += lib.einsum('a,a,IA->IA', L_b, R_b, t1_1_b, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('a,A,ib,Iiab->IA', L_a, R_a, t1_1_a, t2_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('a,a,ib,IiAb->IA', L_a, R_a, t1_1_a, t2_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('a,b,ib,IiAa->IA', L_a, R_a, t1_1_a, t2_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('a,A,ib,Iiab->IA', L_a, R_a, t1_1_b, t2_1_ab, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('a,a,ib,IiAb->IA', L_a, R_a, t1_1_b, t2_1_ab, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('a,a,ib,IiAb->IA', L_b, R_b, t1_1_a, t2_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('a,a,ib,IiAb->IA', L_b, R_b, t1_1_b, t2_1_ab, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('a,b,ib,IiAa->IA', L_b, R_b, t1_1_b, t2_1_ab, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('a,a,ib,iIbA->IA', L_a, R_a, t1_1_a, t2_1_ab, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('a,b,ib,iIaA->IA', L_a, R_a, t1_1_a, t2_1_ab, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('a,a,ib,IiAb->IA', L_a, R_a, t1_1_b, t2_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('a,A,ib,iIba->IA', L_b, R_b, t1_1_a, t2_1_ab, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('a,a,ib,iIbA->IA', L_b, R_b, t1_1_a, t2_1_ab, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('a,A,ib,Iiab->IA', L_b, R_b, t1_1_b, t2_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('a,a,ib,IiAb->IA', L_b, R_b, t1_1_b, t2_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('a,b,ib,IiAa->IA', L_b, R_b, t1_1_b, t2_1_b, optimize=True)

### 111 ###
    if adc.method in ("adc(2)-x", "adc(3)") and t1_1_a is not None:
        rdm1_a[:nocc_a, nocc_a:] -= 1/4 *  lib.einsum('iab,Iab,iA->IA', L_aaa_u, R_aaa_u, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += 1/4 *  lib.einsum('iab,Iba,iA->IA', L_aaa_u, R_aaa_u, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += 1/4 *  lib.einsum('iab,iAa,Ib->IA', L_aaa_u, R_aaa_u, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/4 *  lib.einsum('iab,iAb,Ia->IA', L_aaa_u, R_aaa_u, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/4 *  lib.einsum('iab,iaA,Ib->IA', L_aaa_u, R_aaa_u, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += 1/4 *  lib.einsum('iab,iab,IA->IA', L_aaa_u, R_aaa_u, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += 1/4 *  lib.einsum('iab,ibA,Ia->IA', L_aaa_u, R_aaa_u, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/4 *  lib.einsum('iab,iba,IA->IA', L_aaa_u, R_aaa_u, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] -= lib.einsum('iab,Iab,iA->IA', L_aba, R_aba, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] -= lib.einsum('iab,iaA,Ib->IA', L_aba, R_aba, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += lib.einsum('iab,iab,IA->IA', L_aba, R_aba, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] -= lib.einsum('iab,iAb,Ia->IA', L_bab, R_bab, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += lib.einsum('iab,iab,IA->IA', L_bab, R_bab, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += 1/4 *  lib.einsum('iab,iab,IA->IA', L_bbb_u, R_bbb_u, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/4 *  lib.einsum('iab,iba,IA->IA', L_bbb_u, R_bbb_u, t1_1_a, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += 1/4 *  lib.einsum('iab,iab,IA->IA', L_aaa_u, R_aaa_u, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/4 *  lib.einsum('iab,iba,IA->IA', L_aaa_u, R_aaa_u, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= lib.einsum('iab,iAb,Ia->IA', L_aba, R_aba, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += lib.einsum('iab,iab,IA->IA', L_aba, R_aba, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= lib.einsum('iab,Iab,iA->IA', L_bab, R_bab, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= lib.einsum('iab,iaA,Ib->IA', L_bab, R_bab, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += lib.einsum('iab,iab,IA->IA', L_bab, R_bab, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/4 *  lib.einsum('iab,Iab,iA->IA', L_bbb_u, R_bbb_u, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += 1/4 *  lib.einsum('iab,Iba,iA->IA', L_bbb_u, R_bbb_u, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += 1/4 *  lib.einsum('iab,iAa,Ib->IA', L_bbb_u, R_bbb_u, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/4 *  lib.einsum('iab,iAb,Ia->IA', L_bbb_u, R_bbb_u, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/4 *  lib.einsum('iab,iaA,Ib->IA', L_bbb_u, R_bbb_u, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += 1/4 *  lib.einsum('iab,iab,IA->IA', L_bbb_u, R_bbb_u, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += 1/4 *  lib.einsum('iab,ibA,Ia->IA', L_bbb_u, R_bbb_u, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/4 *  lib.einsum('iab,iba,IA->IA', L_bbb_u, R_bbb_u, t1_1_b, optimize=True)

    ####### ADC(3) SPIN ADAPTED EXCITED STATE OPDM WITH SQA ################
    if adc.method == "adc(3)":
        # Redundant Variables used for names from SQA
        t2_2_a = adc.t2[1][0][:]
        t2_2_ab = adc.t2[1][1][:]
        t2_2_b = adc.t2[1][2][:]
        if adc.t1[1][0] is not None:
            t1_3_a = adc.t1[1][0][:]
            t1_3_b = adc.t1[1][1][:]
        else:
            t1_3_a = np.zeros((nocc_a, nvir_a))
            t1_3_b = np.zeros((nocc_b, nvir_b))

        ###################################################

# block- ij
        ### 030 ###
        temp = np.zeros((nocc_a, nocc_a))
        temp -= 1/2 * lib.einsum('a,a,Iibc,Jibc->IJ', L_a, R_a, t2_1_a, t2_2_a, optimize = True)
        temp += lib.einsum('a,b,Iiac,Jibc->IJ', L_a, R_a, t2_1_a, t2_2_a, optimize = True)
        temp -= lib.einsum('a,a,Iibc,Jibc->IJ', L_a, R_a, t2_1_ab, t2_2_ab, optimize = True)
        temp += lib.einsum('a,b,Iiac,Jibc->IJ', L_a, R_a, t2_1_ab, t2_2_ab, optimize = True)
        temp -= 1/2 * lib.einsum('a,a,Iibc,Jibc->IJ', L_b, R_b, t2_1_a, t2_2_a, optimize = True)
        temp -= lib.einsum('a,a,Iibc,Jibc->IJ', L_b, R_b, t2_1_ab, t2_2_ab, optimize = True)
        temp += lib.einsum('a,b,Iica,Jicb->IJ', L_b, R_b, t2_1_ab, t2_2_ab, optimize = True)
        rdm1_a[:nocc_a, :nocc_a] += temp + temp.T
        temp = np.zeros((nocc_b, nocc_b))
        temp -= lib.einsum('a,a,iIbc,iJbc->IJ', L_a, R_a, t2_1_ab, t2_2_ab, optimize = True)
        temp += lib.einsum('a,b,iIac,iJbc->IJ', L_a, R_a, t2_1_ab, t2_2_ab, optimize = True)
        temp -= 1/2 * lib.einsum('a,a,Iibc,Jibc->IJ', L_a, R_a, t2_1_b, t2_2_b, optimize = True)
        temp -= lib.einsum('a,a,iIbc,iJbc->IJ', L_b, R_b, t2_1_ab, t2_2_ab, optimize = True)
        temp += lib.einsum('a,b,iIca,iJcb->IJ', L_b, R_b, t2_1_ab, t2_2_ab, optimize = True)
        temp -= 1/2 * lib.einsum('a,a,Iibc,Jibc->IJ', L_b, R_b, t2_1_b, t2_2_b, optimize = True)
        temp += lib.einsum('a,b,Iiac,Jibc->IJ', L_b, R_b, t2_1_b, t2_2_b, optimize = True)
        rdm1_b[:nocc_b, :nocc_b] += temp + temp.T

        if t1_1_a is not None:
            temp = np.zeros((nocc_a, nocc_a))
            temp -= lib.einsum('a,a,Ib,Jb->IJ', L_a, R_a, t1_1_a, t1_2_a, optimize=True)
            temp += lib.einsum('a,b,Ia,Jb->IJ', L_a, R_a, t1_1_a, t1_2_a, optimize=True)
            temp -= lib.einsum('a,a,Ib,Jb->IJ', L_b, R_b, t1_1_a, t1_2_a, optimize=True)
            temp -= 1/2 * lib.einsum('a,a,Ib,ic,Jibc->IJ', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('a,b,Ia,ic,Jibc->IJ', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            rdm1_a[:nocc_a, :nocc_a] += temp + temp.T
            rdm1_a[:nocc_a, :nocc_a] += 1/2 *  lib.einsum('a,a,Iibc,ib,Jc->IJ', L_a, R_a, t2_1_a, t1_1_a, t1_1_a,
                optimize=True)
            rdm1_a[:nocc_a, :nocc_a] += 1/2 *  lib.einsum('a,a,Jibc,ib,Ic->IJ', L_a, R_a, t2_1_a, t1_1_a, t1_1_a,
                optimize=True)
            temp = np.zeros((nocc_a, nocc_a))
            temp += 1/2 * lib.einsum('a,b,Iiac,Jb,ic->IJ', L_a, R_a, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            temp -= 1/2 * lib.einsum('a,b,Iiac,ib,Jc->IJ', L_a, R_a, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            temp -= 1/2 * lib.einsum('a,a,Ib,ic,Jibc->IJ', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('a,b,Ic,ia,Jicb->IJ', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            rdm1_a[:nocc_a, :nocc_a] += temp + temp.T
            rdm1_a[:nocc_a, :nocc_a] += 1/2 *  lib.einsum('a,a,Iibc,ib,Jc->IJ', L_b, R_b, t2_1_a, t1_1_a, t1_1_a,
                optimize=True)
            rdm1_a[:nocc_a, :nocc_a] += 1/2 *  lib.einsum('a,a,Jibc,ib,Ic->IJ', L_b, R_b, t2_1_a, t1_1_a, t1_1_a,
                optimize=True)
            temp = np.zeros((nocc_b, nocc_b))
            temp -= lib.einsum('a,a,Ib,Jb->IJ', L_a, R_a, t1_1_b, t1_2_b, optimize=True)
            temp -= lib.einsum('a,a,Ib,Jb->IJ', L_b, R_b, t1_1_b, t1_2_b, optimize=True)
            temp += lib.einsum('a,b,Ia,Jb->IJ', L_b, R_b, t1_1_b, t1_2_b, optimize=True)
            temp -= 1/2 * lib.einsum('a,a,ib,Ic,iJbc->IJ', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('a,b,ia,Ic,iJbc->IJ', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            rdm1_b[:nocc_b, :nocc_b] += temp + temp.T
            rdm1_b[:nocc_b, :nocc_b] += 1/2 *  lib.einsum('a,a,Iibc,ib,Jc->IJ', L_a, R_a, t2_1_b, t1_1_b, t1_1_b,
                optimize=True)
            rdm1_b[:nocc_b, :nocc_b] += 1/2 *  lib.einsum('a,a,Jibc,ib,Ic->IJ', L_a, R_a, t2_1_b, t1_1_b, t1_1_b,
                optimize=True)
            temp = np.zeros((nocc_b, nocc_b))
            temp -= 1/2 * lib.einsum('a,a,ib,Ic,iJbc->IJ', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('a,b,ic,Ia,iJcb->IJ', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            rdm1_b[:nocc_b, :nocc_b] += temp + temp.T
            rdm1_b[:nocc_b, :nocc_b] += 1/2 *  lib.einsum('a,a,Iibc,ib,Jc->IJ', L_b, R_b, t2_1_b, t1_1_b, t1_1_b,
                optimize=True)
            rdm1_b[:nocc_b, :nocc_b] += 1/2 *  lib.einsum('a,a,Jibc,ib,Ic->IJ', L_b, R_b, t2_1_b, t1_1_b, t1_1_b,
                optimize=True)
            temp = np.zeros((nocc_b, nocc_b))
            temp += 1/2 * lib.einsum('a,b,Iiac,Jb,ic->IJ', L_b, R_b, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            temp -= 1/2 * lib.einsum('a,b,Iiac,ib,Jc->IJ', L_b, R_b, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            rdm1_b[:nocc_b, :nocc_b] += temp + temp.T

        ### 021 & 120 ###
        temp = np.zeros((nocc_a, nocc_a))
        temp -= 1/2 * lib.einsum('a,Iab,Jb->IJ', L_a, R_aaa_u, t1_2_a, optimize = True)
        temp += 1/2 * lib.einsum('a,Iba,Jb->IJ', L_a, R_aaa_u, t1_2_a, optimize = True)
        temp -= lib.einsum('Jab,a,Ib->IJ', L_aba, R_b, t1_2_a, optimize = True)
        rdm1_a[:nocc_a, :nocc_a] += temp + temp.T
        temp = np.zeros((nocc_b, nocc_b))
        temp -= lib.einsum('a,Iab,Jb->IJ', L_a, R_bab, t1_2_b, optimize = True)
        temp -= 1/2 * lib.einsum('a,Iab,Jb->IJ', L_b, R_bbb_u, t1_2_b, optimize = True)
        temp += 1/2 * lib.einsum('a,Iba,Jb->IJ', L_b, R_bbb_u, t1_2_b, optimize = True)
        rdm1_b[:nocc_b, :nocc_b] += temp + temp.T

        if t1_1_a is not None:
            temp = np.zeros((nocc_a, nocc_a))
            temp -= 1/4 * lib.einsum('a,Iab,ic,Jibc->IJ', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp += 1/4 * lib.einsum('a,Iba,ic,Jibc->IJ', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp -= 1/4 * lib.einsum('a,Ibc,ia,Jibc->IJ', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp += 1/2 * lib.einsum('a,iab,Ic,Jibc->IJ', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp -= 1/2 * lib.einsum('a,iba,Ic,Jibc->IJ', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp += 1/2 * lib.einsum('a,ibc,Ia,Jibc->IJ', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp -= 1/4 * lib.einsum('a,Iab,ic,Jibc->IJ', L_a, R_aaa_u, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/4 * lib.einsum('a,Iba,ic,Jibc->IJ', L_a, R_aaa_u, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('Jab,a,ic,Iibc->IJ', L_aba, R_b, t1_1_a, t2_1_a, optimize=True)
            temp += lib.einsum('iab,a,Jc,Iibc->IJ', L_aba, R_b, t1_1_a, t2_1_a, optimize=True)
            temp -= 1/2 * lib.einsum('Jab,a,ic,Iibc->IJ', L_aba, R_b, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('Jab,c,ic,Iiba->IJ', L_aba, R_b, t1_1_b, t2_1_ab, optimize=True)
            temp -= lib.einsum('a,iab,Ic,Jicb->IJ', L_a, R_bab, t1_1_a, t2_1_ab, optimize=True)
            temp += lib.einsum('a,ibc,Ia,Jibc->IJ', L_a, R_bab, t1_1_a, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('a,iab,Ic,Jicb->IJ', L_b, R_bbb_u, t1_1_a, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('a,iba,Ic,Jicb->IJ', L_b, R_bbb_u, t1_1_a, t2_1_ab, optimize=True)
            rdm1_a[:nocc_a, :nocc_a] += temp + temp.T
            temp = np.zeros((nocc_b, nocc_b))
            temp -= 1/2 * lib.einsum('a,iab,Ic,iJbc->IJ', L_a, R_aaa_u, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('a,iba,Ic,iJbc->IJ', L_a, R_aaa_u, t1_1_b, t2_1_ab, optimize=True)
            temp -= lib.einsum('iab,a,Jc,iIbc->IJ', L_aba, R_b, t1_1_b, t2_1_ab, optimize=True)
            temp += lib.einsum('iab,c,Jc,iIba->IJ', L_aba, R_b, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('a,Iab,ic,iJcb->IJ', L_a, R_bab, t1_1_a, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('a,Ibc,ia,iJbc->IJ', L_a, R_bab, t1_1_a, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('a,Iab,ic,Jibc->IJ', L_a, R_bab, t1_1_b, t2_1_b, optimize=True)
            temp += lib.einsum('a,iab,Ic,Jibc->IJ', L_a, R_bab, t1_1_b, t2_1_b, optimize=True)
            temp -= 1/4 * lib.einsum('a,Iab,ic,iJcb->IJ', L_b, R_bbb_u, t1_1_a, t2_1_ab, optimize=True)
            temp += 1/4 * lib.einsum('a,Iba,ic,iJcb->IJ', L_b, R_bbb_u, t1_1_a, t2_1_ab, optimize=True)
            temp -= 1/4 * lib.einsum('a,Iab,ic,Jibc->IJ', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            temp += 1/4 * lib.einsum('a,Iba,ic,Jibc->IJ', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            temp -= 1/4 * lib.einsum('a,Ibc,ia,Jibc->IJ', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            temp += 1/2 * lib.einsum('a,iab,Ic,Jibc->IJ', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            temp -= 1/2 * lib.einsum('a,iba,Ic,Jibc->IJ', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            temp += 1/2 * lib.einsum('a,ibc,Ia,Jibc->IJ', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            rdm1_b[:nocc_b, :nocc_b] += temp + temp.T

# block- ab
        ### 030 ###
        temp = np.zeros((nvir_a, nvir_a))
        temp -= 1/4 * lib.einsum('A,a,ijBb,ijab->AB', L_a, R_a, t2_1_a, t2_2_a, optimize = True)
        temp -= 1/4 * lib.einsum('A,a,ijab,ijBb->AB', L_a, R_a, t2_1_a, t2_2_a, optimize = True)
        temp += 1/2 * lib.einsum('a,a,ijAb,ijBb->AB', L_a, R_a, t2_1_a, t2_2_a, optimize = True)
        temp -= 1/2 * lib.einsum('a,b,ijAb,ijBa->AB', L_a, R_a, t2_1_a, t2_2_a, optimize = True)
        temp -= 1/2 * lib.einsum('A,a,ijBb,ijab->AB', L_a, R_a, t2_1_ab, t2_2_ab, optimize = True)
        temp -= 1/2 * lib.einsum('A,a,ijab,ijBb->AB', L_a, R_a, t2_1_ab, t2_2_ab, optimize = True)
        temp += lib.einsum('a,a,ijAb,ijBb->AB', L_a, R_a, t2_1_ab, t2_2_ab, optimize = True)
        temp += 1/2 * lib.einsum('a,a,ijAb,ijBb->AB', L_b, R_b, t2_1_a, t2_2_a, optimize = True)
        temp += lib.einsum('a,a,ijAb,ijBb->AB', L_b, R_b, t2_1_ab, t2_2_ab, optimize = True)
        temp -= lib.einsum('a,b,ijAb,ijBa->AB', L_b, R_b, t2_1_ab, t2_2_ab, optimize = True)
        rdm1_a[nocc_a:, nocc_a:] += temp + temp.T
        temp = np.zeros((nvir_b, nvir_b))
        temp += lib.einsum('a,a,ijbA,ijbB->AB', L_a, R_a, t2_1_ab, t2_2_ab, optimize = True)
        temp -= lib.einsum('a,b,ijaB,ijbA->AB', L_a, R_a, t2_1_ab, t2_2_ab, optimize = True)
        temp += 1/2 * lib.einsum('a,a,ijAb,ijBb->AB', L_a, R_a, t2_1_b, t2_2_b, optimize = True)
        temp -= 1/2 * lib.einsum('A,a,ijbB,ijba->AB', L_b, R_b, t2_1_ab, t2_2_ab, optimize = True)
        temp -= 1/2 * lib.einsum('A,a,ijba,ijbB->AB', L_b, R_b, t2_1_ab, t2_2_ab, optimize = True)
        temp += lib.einsum('a,a,ijbA,ijbB->AB', L_b, R_b, t2_1_ab, t2_2_ab, optimize = True)
        temp -= 1/4 * lib.einsum('A,a,ijBb,ijab->AB', L_b, R_b, t2_1_b, t2_2_b, optimize = True)
        temp -= 1/4 * lib.einsum('A,a,ijab,ijBb->AB', L_b, R_b, t2_1_b, t2_2_b, optimize = True)
        temp += 1/2 * lib.einsum('a,a,ijAb,ijBb->AB', L_b, R_b, t2_1_b, t2_2_b, optimize = True)
        temp -= 1/2 * lib.einsum('a,b,ijAb,ijBa->AB', L_b, R_b, t2_1_b, t2_2_b, optimize = True)
        rdm1_b[nocc_b:, nocc_b:] += temp + temp.T

        if t1_1_a is not None:
            temp = np.zeros((nvir_a, nvir_a))
            temp -= 1/2 * lib.einsum('A,a,iB,ia->AB', L_a, R_a, t1_1_a, t1_2_a, optimize=True)
            temp -= 1/2 * lib.einsum('A,a,ia,iB->AB', L_a, R_a, t1_1_a, t1_2_a, optimize=True)
            temp += lib.einsum('a,a,iA,iB->AB', L_a, R_a, t1_1_a, t1_2_a, optimize=True)
            temp += lib.einsum('a,a,iA,iB->AB', L_b, R_b, t1_1_a, t1_2_a, optimize=True)
            temp -= 1/6 * lib.einsum('A,a,iB,jb,ijab->AB', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/3 * lib.einsum('A,a,ia,jb,ijBb->AB', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('a,a,iA,jb,ijBb->AB', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/3 * lib.einsum('A,a,ijBb,ia,jb->AB', L_a, R_a, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            temp -= 1/6 * lib.einsum('A,a,ijab,iB,jb->AB', L_a, R_a, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            temp += 1/2 * lib.einsum('a,a,ijAb,iB,jb->AB', L_a, R_a, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            temp += 1/2 * lib.einsum('a,b,ijAb,ia,jB->AB', L_a, R_a, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            temp += 1/2 * lib.einsum('a,a,iA,jb,ijBb->AB', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('a,b,iA,jb,ijBa->AB', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('a,a,ijAb,iB,jb->AB', L_b, R_b, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            rdm1_a[nocc_a:, nocc_a:] += temp + temp.T
            temp = np.zeros((nvir_b, nvir_b))
            temp += lib.einsum('a,a,iA,iB->AB', L_a, R_a, t1_1_b, t1_2_b, optimize=True)
            temp -= 1/2 * lib.einsum('A,a,iB,ia->AB', L_b, R_b, t1_1_b, t1_2_b, optimize=True)
            temp -= 1/2 * lib.einsum('A,a,ia,iB->AB', L_b, R_b, t1_1_b, t1_2_b, optimize=True)
            temp += lib.einsum('a,a,iA,iB->AB', L_b, R_b, t1_1_b, t1_2_b, optimize=True)
            temp += 1/2 * lib.einsum('a,a,ib,jA,ijbB->AB', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('a,b,ia,jB,ijbA->AB', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('a,a,ijAb,iB,jb->AB', L_a, R_a, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            temp -= 1/6 * lib.einsum('A,a,ib,jB,ijba->AB', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/3 * lib.einsum('A,a,ib,ja,ijbB->AB', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('a,a,ib,jA,ijbB->AB', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/3 * lib.einsum('A,a,ijBb,ia,jb->AB', L_b, R_b, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            temp -= 1/6 * lib.einsum('A,a,ijab,iB,jb->AB', L_b, R_b, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            temp += 1/2 * lib.einsum('a,a,ijAb,iB,jb->AB', L_b, R_b, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            temp += 1/2 * lib.einsum('a,b,ijAb,ia,jB->AB', L_b, R_b, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            rdm1_b[nocc_b:, nocc_b:] += temp + temp.T

        ### 021 & 120 ###
        temp = np.zeros((nvir_a, nvir_a))
        temp -= 1/2 * lib.einsum('a,iBa,iA->AB', L_a, R_aaa_u, t1_2_a, optimize = True)
        temp += 1/2 * lib.einsum('a,iaB,iA->AB', L_a, R_aaa_u, t1_2_a, optimize = True)
        temp += lib.einsum('iaA,a,iB->AB', L_aba, R_b, t1_2_a, optimize = True)
        rdm1_a[nocc_a:, nocc_a:] += temp + temp.T
        temp = np.zeros((nvir_b, nvir_b))
        temp += lib.einsum('a,iaB,iA->AB', L_a, R_bab, t1_2_b, optimize = True)
        temp -= 1/2 * lib.einsum('a,iBa,iA->AB', L_b, R_bbb_u, t1_2_b, optimize = True)
        temp += 1/2 * lib.einsum('a,iaB,iA->AB', L_b, R_bbb_u, t1_2_b, optimize = True)
        rdm1_b[nocc_b:, nocc_b:] += temp + temp.T

        if t1_1_a is not None:
            temp = np.zeros((nvir_a, nvir_a))
            temp -= 1/4 * lib.einsum('a,iBa,jb,ijAb->AB', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp += 1/4 * lib.einsum('a,iBb,ja,ijAb->AB', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp += 1/4 * lib.einsum('a,iaB,jb,ijAb->AB', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp -= 1/4 * lib.einsum('a,ibB,ja,ijAb->AB', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp += 1/4 * lib.einsum('A,iab,jB,ijab->AB', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp -= 1/2 * lib.einsum('a,iab,jB,ijAb->AB', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp += 1/2 * lib.einsum('a,iba,jB,ijAb->AB', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp -= 1/4 * lib.einsum('a,iBa,jb,ijAb->AB', L_a, R_aaa_u, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/4 * lib.einsum('a,iaB,jb,ijAb->AB', L_a, R_aaa_u, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('iaA,a,jb,ijBb->AB', L_aba, R_b, t1_1_a, t2_1_a, optimize=True)
            temp -= lib.einsum('iab,a,jA,ijBb->AB', L_aba, R_b, t1_1_a, t2_1_a, optimize=True)
            temp += 1/2 * lib.einsum('iaA,a,jb,ijBb->AB', L_aba, R_b, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('iaA,b,jb,ijBa->AB', L_aba, R_b, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('a,iBb,ja,jiAb->AB', L_a, R_bab, t1_1_a, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('A,iab,jB,jiab->AB', L_a, R_bab, t1_1_a, t2_1_ab, optimize=True)
            temp += lib.einsum('a,iab,jB,jiAb->AB', L_a, R_bab, t1_1_a, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('a,iab,jB,jiAb->AB', L_b, R_bbb_u, t1_1_a, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('a,iba,jB,jiAb->AB', L_b, R_bbb_u, t1_1_a, t2_1_ab, optimize=True)
            rdm1_a[nocc_a:, nocc_a:] += temp + temp.T
            temp = np.zeros((nvir_b, nvir_b))
            temp += 1/2 * lib.einsum('a,iab,jB,ijbA->AB', L_a, R_aaa_u, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('a,iba,jB,ijbA->AB', L_a, R_aaa_u, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('iab,B,jA,ijba->AB', L_aba, R_b, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('iAa,b,jb,ijaB->AB', L_aba, R_b, t1_1_b, t2_1_ab, optimize=True)
            temp += lib.einsum('iab,a,jA,ijbB->AB', L_aba, R_b, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('a,iaB,jb,jibA->AB', L_a, R_bab, t1_1_a, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('a,ibB,ja,jibA->AB', L_a, R_bab, t1_1_a, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('a,iaB,jb,ijAb->AB', L_a, R_bab, t1_1_b, t2_1_b, optimize=True)
            temp -= lib.einsum('a,iab,jB,ijAb->AB', L_a, R_bab, t1_1_b, t2_1_b, optimize=True)
            temp -= 1/4 * lib.einsum('a,iBa,jb,jibA->AB', L_b, R_bbb_u, t1_1_a, t2_1_ab, optimize=True)
            temp += 1/4 * lib.einsum('a,iaB,jb,jibA->AB', L_b, R_bbb_u, t1_1_a, t2_1_ab, optimize=True)
            temp -= 1/4 * lib.einsum('a,iBa,jb,ijAb->AB', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            temp += 1/4 * lib.einsum('a,iBb,ja,ijAb->AB', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            temp += 1/4 * lib.einsum('a,iaB,jb,ijAb->AB', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            temp -= 1/4 * lib.einsum('a,ibB,ja,ijAb->AB', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            temp += 1/4 * lib.einsum('A,iab,jB,ijab->AB', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            temp -= 1/2 * lib.einsum('a,iab,jB,ijAb->AB', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            temp += 1/2 * lib.einsum('a,iba,jB,ijAb->AB', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            rdm1_b[nocc_b:, nocc_b:] += temp + temp.T

# block- ia
        ### 030 ###
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('a,A,ib,Iiab->IA', L_a, R_a, t1_2_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('a,a,ib,IiAb->IA', L_a, R_a, t1_2_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('a,b,ib,IiAa->IA', L_a, R_a, t1_2_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('a,A,ib,Iiab->IA', L_a, R_a, t1_2_b, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('a,a,ib,IiAb->IA', L_a, R_a, t1_2_b, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('a,a,ib,IiAb->IA', L_b, R_b, t1_2_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('a,a,ib,IiAb->IA', L_b, R_b, t1_2_b, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('a,b,ib,IiAa->IA', L_b, R_b, t1_2_b, t2_1_ab, optimize = True)

        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('a,a,ib,iIbA->IA', L_a, R_a, t1_2_a, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('a,b,ib,iIaA->IA', L_a, R_a, t1_2_a, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('a,a,ib,IiAb->IA', L_a, R_a, t1_2_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('a,A,ib,iIba->IA', L_b, R_b, t1_2_a, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('a,a,ib,iIbA->IA', L_b, R_b, t1_2_a, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('a,A,ib,Iiab->IA', L_b, R_b, t1_2_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('a,a,ib,IiAb->IA', L_b, R_b, t1_2_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('a,b,ib,IiAa->IA', L_b, R_b, t1_2_b, t2_1_b, optimize = True)

        if adc.approx_trans_moments is False:
            rdm1_a[:nocc_a, nocc_a:] -= lib.einsum('a,A,Ia->IA', L_a, R_a, t1_3_a, optimize = True)
            rdm1_a[:nocc_a, nocc_a:] += lib.einsum('a,a,IA->IA', L_a, R_a, t1_3_a, optimize = True)
            rdm1_a[:nocc_a, nocc_a:] += lib.einsum('a,a,IA->IA', L_b, R_b, t1_3_a, optimize = True)
            rdm1_b[:nocc_b, nocc_b:] += lib.einsum('a,a,IA->IA', L_a, R_a, t1_3_b, optimize = True)
            rdm1_b[:nocc_b, nocc_b:] -= lib.einsum('a,A,Ia->IA', L_b, R_b, t1_3_b, optimize = True)
            rdm1_b[:nocc_b, nocc_b:] += lib.einsum('a,a,IA->IA', L_b, R_b, t1_3_b, optimize = True)

        if t1_1_a is not None:
            rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('a,A,ib,Iiab->IA', L_a, R_a, t1_1_a, t2_2_a, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('a,a,ib,IiAb->IA', L_a, R_a, t1_1_a, t2_2_a, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('a,b,ib,IiAa->IA', L_a, R_a, t1_1_a, t2_2_a, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('a,A,ib,Iiab->IA', L_a, R_a, t1_1_b, t2_2_ab, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('a,a,ib,IiAb->IA', L_a, R_a, t1_1_b, t2_2_ab, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('a,a,ib,IiAb->IA', L_b, R_b, t1_1_a, t2_2_a, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('a,a,ib,IiAb->IA', L_b, R_b, t1_1_b, t2_2_ab, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('a,b,ib,IiAa->IA', L_b, R_b, t1_1_b, t2_2_ab, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('a,A,ia,Ib,ib->IA', L_a, R_a, t1_1_a, t1_1_a, t1_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 2/3 *  lib.einsum('a,a,Ib,iA,ib->IA', L_a, R_a, t1_1_a, t1_1_a, t1_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('a,b,Ia,ib,iA->IA', L_a, R_a, t1_1_a, t1_1_a, t1_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/12 *  lib.einsum('a,A,Ib,ijac,ijbc->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/12 *  lib.einsum('a,A,ia,ijbc,Ijbc->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/6 *  lib.einsum('a,A,ib,Ijac,ijbc->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/3 *  lib.einsum('a,a,Ib,ijbc,ijAc->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/3 *  lib.einsum('a,a,iA,ijbc,Ijbc->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('a,a,ib,ijbc,IjAc->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/4 *  lib.einsum('a,b,Ia,ijbc,ijAc->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/3 *  lib.einsum('a,b,Ic,ijAa,ijbc->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 2/3 *  lib.einsum('a,b,iA,Ijac,ijbc->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/6 *  lib.einsum('a,b,ia,ijbc,IjAc->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('a,b,ic,IjAa,ijbc->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('a,A,Ib,ijac,ijbc->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('a,A,ia,ijbc,Ijbc->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/6 *  lib.einsum('a,A,ib,Ijac,ijbc->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 2/3 *  lib.einsum('a,a,Ib,ijbc,ijAc->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 2/3 *  lib.einsum('a,a,iA,ijbc,Ijbc->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('a,a,ib,ijbc,IjAc->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('a,b,Ia,ijbc,ijAc->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 2/3 *  lib.einsum('a,b,iA,Ijac,ijbc->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/6 *  lib.einsum('a,b,ia,ijbc,IjAc->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/6 *  lib.einsum('a,A,ib,Ijac,jicb->IA', L_a, R_a, t1_1_b, t2_1_a, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('a,a,ib,IjAc,jicb->IA', L_a, R_a, t1_1_b, t2_1_a, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/6 *  lib.einsum('a,b,ic,IjAa,jibc->IA', L_a, R_a, t1_1_b, t2_1_a, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/6 *  lib.einsum('a,A,ib,Ijac,ijbc->IA', L_a, R_a, t1_1_b, t2_1_ab, t2_1_b,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('a,a,ib,IjAc,ijbc->IA', L_a, R_a, t1_1_b, t2_1_ab, t2_1_b,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 2/3 *  lib.einsum('a,a,Ib,iA,ib->IA', L_b, R_b, t1_1_a, t1_1_a, t1_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/3 *  lib.einsum('a,a,Ib,ijbc,ijAc->IA', L_b, R_b, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/3 *  lib.einsum('a,a,iA,ijbc,Ijbc->IA', L_b, R_b, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('a,a,ib,ijbc,IjAc->IA', L_b, R_b, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 2/3 *  lib.einsum('a,a,Ib,ijbc,ijAc->IA', L_b, R_b, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 2/3 *  lib.einsum('a,a,iA,ijbc,Ijbc->IA', L_b, R_b, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('a,a,ib,ijbc,IjAc->IA', L_b, R_b, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 2/3 *  lib.einsum('a,b,Ic,ijAa,ijcb->IA', L_b, R_b, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 2/3 *  lib.einsum('a,b,iA,Ijca,ijcb->IA', L_b, R_b, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/6 *  lib.einsum('a,b,ic,IjAa,ijcb->IA', L_b, R_b, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('a,a,ib,IjAc,jicb->IA', L_b, R_b, t1_1_b, t2_1_a, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/6 *  lib.einsum('a,b,ia,IjAc,jicb->IA', L_b, R_b, t1_1_b, t2_1_a, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('a,a,ib,IjAc,ijbc->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_b,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/6 *  lib.einsum('a,b,ia,IjAc,ijbc->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_b,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('a,b,ic,IjAa,ijbc->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('a,a,ib,iIbA->IA', L_a, R_a, t1_1_a, t2_2_ab, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('a,b,ib,iIaA->IA', L_a, R_a, t1_1_a, t2_2_ab, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('a,a,ib,IiAb->IA', L_a, R_a, t1_1_b, t2_2_b, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('a,A,ib,iIba->IA', L_b, R_b, t1_1_a, t2_2_ab, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('a,a,ib,iIbA->IA', L_b, R_b, t1_1_a, t2_2_ab, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('a,A,ib,Iiab->IA', L_b, R_b, t1_1_b, t2_2_b, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('a,a,ib,IiAb->IA', L_b, R_b, t1_1_b, t2_2_b, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('a,b,ib,IiAa->IA', L_b, R_b, t1_1_b, t2_2_b, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('a,a,ib,ijbc,jIcA->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/6 *  lib.einsum('a,b,ia,ijbc,jIcA->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('a,b,ic,ijbc,jIaA->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('a,a,ib,ijbc,IjAc->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/6 *  lib.einsum('a,b,ia,ijbc,IjAc->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 2/3 *  lib.einsum('a,a,Ib,iA,ib->IA', L_a, R_a, t1_1_b, t1_1_b, t1_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 2/3 *  lib.einsum('a,a,Ib,ijcb,ijcA->IA', L_a, R_a, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 2/3 *  lib.einsum('a,a,iA,jibc,jIbc->IA', L_a, R_a, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('a,a,ib,jicb,jIcA->IA', L_a, R_a, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 2/3 *  lib.einsum('a,b,Ic,ijaA,ijbc->IA', L_a, R_a, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 2/3 *  lib.einsum('a,b,iA,jIac,jibc->IA', L_a, R_a, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/6 *  lib.einsum('a,b,ic,jIaA,jibc->IA', L_a, R_a, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/3 *  lib.einsum('a,a,Ib,ijbc,ijAc->IA', L_a, R_a, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/3 *  lib.einsum('a,a,iA,ijbc,Ijbc->IA', L_a, R_a, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('a,a,ib,ijbc,IjAc->IA', L_a, R_a, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/6 *  lib.einsum('a,A,ib,ijbc,jIca->IA', L_b, R_b, t1_1_a, t2_1_a, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('a,a,ib,ijbc,jIcA->IA', L_b, R_b, t1_1_a, t2_1_a, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/6 *  lib.einsum('a,A,ib,ijbc,Ijac->IA', L_b, R_b, t1_1_a, t2_1_ab, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('a,a,ib,ijbc,IjAc->IA', L_b, R_b, t1_1_a, t2_1_ab, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/6 *  lib.einsum('a,b,ic,ijcb,IjAa->IA', L_b, R_b, t1_1_a, t2_1_ab, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('a,A,ia,Ib,ib->IA', L_b, R_b, t1_1_b, t1_1_b, t1_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 2/3 *  lib.einsum('a,a,Ib,iA,ib->IA', L_b, R_b, t1_1_b, t1_1_b, t1_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('a,b,Ia,ib,iA->IA', L_b, R_b, t1_1_b, t1_1_b, t1_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('a,A,Ib,ijca,ijcb->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('a,A,ia,jibc,jIbc->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/6 *  lib.einsum('a,A,ib,jIca,jicb->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 2/3 *  lib.einsum('a,a,Ib,ijcb,ijcA->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 2/3 *  lib.einsum('a,a,iA,jibc,jIbc->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('a,a,ib,jicb,jIcA->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('a,b,Ia,ijcb,ijcA->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 2/3 *  lib.einsum('a,b,iA,jIca,jicb->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/6 *  lib.einsum('a,b,ia,jicb,jIcA->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/12 *  lib.einsum('a,A,Ib,ijac,ijbc->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/12 *  lib.einsum('a,A,ia,ijbc,Ijbc->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/6 *  lib.einsum('a,A,ib,Ijac,ijbc->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/3 *  lib.einsum('a,a,Ib,ijbc,ijAc->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/3 *  lib.einsum('a,a,iA,ijbc,Ijbc->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('a,a,ib,ijbc,IjAc->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/4 *  lib.einsum('a,b,Ia,ijbc,ijAc->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/3 *  lib.einsum('a,b,Ic,ijAa,ijbc->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 2/3 *  lib.einsum('a,b,iA,Ijac,ijbc->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/6 *  lib.einsum('a,b,ia,ijbc,IjAc->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('a,b,ic,IjAa,ijbc->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)

        ### 021 & 120 ###
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('iab,A,Iiab->IA', L_aaa_u, R_a, t2_2_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('iab,a,IiAb->IA', L_aaa_u, R_a, t2_2_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('iab,b,IiAa->IA', L_aaa_u, R_a, t2_2_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += lib.einsum('iab,a,IiAb->IA', L_aba, R_b, t2_2_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= lib.einsum('iab,A,Iiab->IA', L_bab, R_a, t2_2_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += lib.einsum('iab,a,IiAb->IA', L_bab, R_a, t2_2_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('iab,a,IiAb->IA', L_bbb_u, R_b, t2_2_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('iab,b,IiAa->IA', L_bbb_u, R_b, t2_2_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/8 * lib.einsum('a,Iab,ijbc,ijAc->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/8 * lib.einsum('a,Iba,ijbc,ijAc->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/8 * lib.einsum('a,Ibc,ijAa,ijbc->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/8 * lib.einsum('a,iAa,ijbc,Ijbc->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/4 * lib.einsum('a,iAb,Ijac,ijbc->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/8 * lib.einsum('a,iaA,ijbc,Ijbc->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/4 * lib.einsum('a,iab,ijbc,IjAc->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/4 * lib.einsum('a,ibA,Ijac,ijbc->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/4 * lib.einsum('a,iba,ijbc,IjAc->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/4 * lib.einsum('a,ibc,IjAa,ijbc->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/4 * lib.einsum('a,Iab,ijbc,ijAc->IA',
                                                    L_a, R_aaa_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/4 * lib.einsum('a,Iba,ijbc,ijAc->IA',
                                                    L_a, R_aaa_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/4 * lib.einsum('a,iAa,ijbc,Ijbc->IA',
                                                    L_a, R_aaa_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/4 * lib.einsum('a,iAb,Ijac,ijbc->IA',
                                                    L_a, R_aaa_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/4 * lib.einsum('a,iaA,ijbc,Ijbc->IA',
                                                    L_a, R_aaa_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/4 * lib.einsum('a,iab,ijbc,IjAc->IA',
                                                    L_a, R_aaa_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/4 * lib.einsum('a,ibA,Ijac,ijbc->IA',
                                                    L_a, R_aaa_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/4 * lib.einsum('a,iba,ijbc,IjAc->IA',
                                                    L_a, R_aaa_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('a,iAb,Ijac,jicb->IA',
                                                    L_a, R_bab, t2_1_a, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('a,iab,IjAc,jicb->IA',
                                                    L_a, R_bab, t2_1_a, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('a,ibc,IjAa,jibc->IA',
                                                    L_a, R_bab, t2_1_a, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('a,iAb,Ijac,ijbc->IA',
                                                    L_a, R_bab, t2_1_ab, t2_1_b, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('a,iab,IjAc,ijbc->IA',
                                                    L_a, R_bab, t2_1_ab, t2_1_b, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/4 * lib.einsum('a,Iab,ijbc,ijAc->IA', L_b, R_aba, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/4 * lib.einsum('a,iaA,ijbc,Ijbc->IA', L_b, R_aba, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('a,iab,ijbc,IjAc->IA', L_b, R_aba, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('a,Iab,ijbc,ijAc->IA',
                                                    L_b, R_aba, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('a,Ibc,ijAa,ijcb->IA',
                                                    L_b, R_aba, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('a,iaA,ijbc,Ijbc->IA',
                                                    L_b, R_aba, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('a,iab,ijbc,IjAc->IA',
                                                    L_b, R_aba, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('a,ibA,Ijca,ijcb->IA',
                                                    L_b, R_aba, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('a,ibc,IjAa,ijcb->IA',
                                                    L_b, R_aba, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/4 * lib.einsum('a,iab,IjAc,jicb->IA',
                                                    L_b, R_bbb_u, t2_1_a, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/4 * lib.einsum('a,iba,IjAc,jicb->IA',
                                                    L_b, R_bbb_u, t2_1_a, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/4 * lib.einsum('a,iab,IjAc,ijbc->IA',
                                                    L_b, R_bbb_u, t2_1_ab, t2_1_b, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/4 * lib.einsum('a,iba,IjAc,ijbc->IA',
                                                    L_b, R_bbb_u, t2_1_ab, t2_1_b, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/4 * lib.einsum('a,ibc,IjAa,ijbc->IA',
                                                    L_b, R_bbb_u, t2_1_ab, t2_1_b, optimize = True)

        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('iab,a,iIbA->IA', L_aaa_u, R_a, t2_2_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('iab,b,iIaA->IA', L_aaa_u, R_a, t2_2_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= lib.einsum('iab,A,iIba->IA', L_aba, R_b, t2_2_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += lib.einsum('iab,a,iIbA->IA', L_aba, R_b, t2_2_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += lib.einsum('iab,a,IiAb->IA', L_bab, R_a, t2_2_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('iab,A,Iiab->IA', L_bbb_u, R_b, t2_2_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('iab,a,IiAb->IA', L_bbb_u, R_b, t2_2_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('iab,b,IiAa->IA', L_bbb_u, R_b, t2_2_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/4 * lib.einsum('a,iab,ijbc,jIcA->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/4 * lib.einsum('a,iba,ijbc,jIcA->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/4 * lib.einsum('a,ibc,ijbc,jIaA->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/4 * lib.einsum('a,iab,ijbc,IjAc->IA',
                                                    L_a, R_aaa_u, t2_1_ab, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/4 * lib.einsum('a,iba,ijbc,IjAc->IA',
                                                    L_a, R_aaa_u, t2_1_ab, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('a,Iab,ijcb,ijcA->IA',
                                                    L_a, R_bab, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('a,Ibc,ijaA,ijbc->IA',
                                                    L_a, R_bab, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('a,iaA,jibc,jIbc->IA',
                                                    L_a, R_bab, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('a,iab,jicb,jIcA->IA',
                                                    L_a, R_bab, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('a,ibA,jIac,jibc->IA',
                                                    L_a, R_bab, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('a,ibc,jIaA,jibc->IA',
                                                    L_a, R_bab, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/4 * lib.einsum('a,Iab,ijbc,ijAc->IA', L_a, R_bab, t2_1_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/4 * lib.einsum('a,iaA,ijbc,Ijbc->IA', L_a, R_bab, t2_1_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('a,iab,ijbc,IjAc->IA', L_a, R_bab, t2_1_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('a,iAb,ijbc,jIca->IA',
                                                    L_b, R_aba, t2_1_a, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('a,iab,ijbc,jIcA->IA',
                                                    L_b, R_aba, t2_1_a, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('a,iAb,ijbc,Ijac->IA',
                                                    L_b, R_aba, t2_1_ab, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('a,iab,ijbc,IjAc->IA',
                                                    L_b, R_aba, t2_1_ab, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('a,ibc,ijcb,IjAa->IA',
                                                    L_b, R_aba, t2_1_ab, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/4 * lib.einsum('a,Iab,ijcb,ijcA->IA',
                                                    L_b, R_bbb_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/4 * lib.einsum('a,Iba,ijcb,ijcA->IA',
                                                    L_b, R_bbb_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/4 * lib.einsum('a,iAa,jibc,jIbc->IA',
                                                    L_b, R_bbb_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/4 * lib.einsum('a,iAb,jIca,jicb->IA',
                                                    L_b, R_bbb_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/4 * lib.einsum('a,iaA,jibc,jIbc->IA',
                                                    L_b, R_bbb_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/4 * lib.einsum('a,iab,jicb,jIcA->IA',
                                                    L_b, R_bbb_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/4 * lib.einsum('a,ibA,jIca,jicb->IA',
                                                    L_b, R_bbb_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/4 * lib.einsum('a,iba,jicb,jIcA->IA',
                                                    L_b, R_bbb_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/8 * lib.einsum('a,Iab,ijbc,ijAc->IA',
                                                    L_b, R_bbb_u, t2_1_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/8 * lib.einsum('a,Iba,ijbc,ijAc->IA',
                                                    L_b, R_bbb_u, t2_1_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/8 * lib.einsum('a,Ibc,ijAa,ijbc->IA',
                                                    L_b, R_bbb_u, t2_1_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/8 * lib.einsum('a,iAa,ijbc,Ijbc->IA',
                                                    L_b, R_bbb_u, t2_1_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/4 * lib.einsum('a,iAb,Ijac,ijbc->IA',
                                                    L_b, R_bbb_u, t2_1_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/8 * lib.einsum('a,iaA,ijbc,Ijbc->IA',
                                                    L_b, R_bbb_u, t2_1_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/4 * lib.einsum('a,iab,ijbc,IjAc->IA',
                                                    L_b, R_bbb_u, t2_1_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/4 * lib.einsum('a,ibA,Ijac,ijbc->IA',
                                                    L_b, R_bbb_u, t2_1_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/4 * lib.einsum('a,iba,ijbc,IjAc->IA',
                                                    L_b, R_bbb_u, t2_1_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/4 * lib.einsum('a,ibc,IjAa,ijbc->IA',
                                                    L_b, R_bbb_u, t2_1_b, t2_1_b, optimize = True)

        if t1_1_a is not None:
            rdm1_a[:nocc_a, nocc_a:] -= 1/4 *  lib.einsum('a,Iab,ib,iA->IA', L_a, R_aaa_u, t1_1_a, t1_1_a,
                                                          optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/4 *  lib.einsum('a,Iba,ib,iA->IA', L_a, R_aaa_u, t1_1_a, t1_1_a,
                                                          optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/4 *  lib.einsum('a,iAa,ib,Ib->IA', L_a, R_aaa_u, t1_1_a, t1_1_a,
                                                          optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/4 *  lib.einsum('a,iaA,ib,Ib->IA', L_a, R_aaa_u, t1_1_a, t1_1_a,
                                                          optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('iab,a,iA,Ib->IA', L_aaa_u, R_a, t1_1_a, t1_1_a,
                                                          optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('iab,b,iA,Ia->IA', L_aaa_u, R_a, t1_1_a, t1_1_a,
                                                          optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= lib.einsum('iab,a,iA,Ib->IA', L_aba, R_b, t1_1_a, t1_1_a, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('a,Iab,ib,iA->IA', L_b, R_aba, t1_1_a, t1_1_a, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('a,iaA,ib,Ib->IA', L_b, R_aba, t1_1_a, t1_1_a, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('a,Iab,ib,iA->IA', L_a, R_bab, t1_1_b, t1_1_b, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('a,iaA,ib,Ib->IA', L_a, R_bab, t1_1_b, t1_1_b, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/4 *  lib.einsum('a,Iab,ib,iA->IA', L_b, R_bbb_u, t1_1_b, t1_1_b,
                                                          optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/4 *  lib.einsum('a,Iba,ib,iA->IA', L_b, R_bbb_u, t1_1_b, t1_1_b,
                                                          optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/4 *  lib.einsum('a,iAa,ib,Ib->IA', L_b, R_bbb_u, t1_1_b, t1_1_b,
                                                          optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/4 *  lib.einsum('a,iaA,ib,Ib->IA', L_b, R_bbb_u, t1_1_b, t1_1_b,
                                                          optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= lib.einsum('iab,a,iA,Ib->IA', L_bab, R_a, t1_1_b, t1_1_b, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('iab,a,iA,Ib->IA', L_bbb_u, R_b, t1_1_b, t1_1_b,
                                                          optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('iab,b,iA,Ia->IA', L_bbb_u, R_b, t1_1_b, t1_1_b,
                                                          optimize=True)

    # block- ai
    rdm1_a[nocc_a:,:nocc_a] = rdm1_a[:nocc_a,nocc_a:].T
    rdm1_b[nocc_b:,:nocc_b] = rdm1_b[:nocc_b,nocc_b:].T

    return (rdm1_a, rdm1_b)

def get_spin_square(adc):
    '''
    <S^2> expectation values (2-RDM term of Eq. 17, J. Chem. Phys. 157, 044106)
    including the 3h1p/3p1h 2-RDM sectors (J. Chem. Phys. 164, 209901 (2026) erratum)
    and the cross-sector classes 100 (bare 1p/2p1h block, order 1) and
    110 (one-t2_1 1p/2p1h block, order 2), which are present in the exact
    effective-Liouvillian property matrix but missing from the reference
    implementation (its G^110/G^011 are t1_1-carried and vanish for canonical
    references)
    Auto-generated einsum groups from SQA;
    Hermitian block-pair symmetry exploited (valid for L = R, real vectors):
    the 2-RDM block (P,Q,R,T) and its conjugate (T,R,Q,P) contribute equal
    scalars in the self-conjugate classes (000/010/020/101/111/030), so each
    partner block is dropped and its canonical partner doubled.
    '''
    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    dm_a, dm_b = adc.make_rdm1()

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

    ovlp = adc._scf.get_ovlp(adc._scf.mol).copy()
    delta = np.dot(adc.mo_coeff[0].transpose(), np.dot(ovlp, adc.mo_coeff[1]))

    S_oo_ab = delta[:nocc_a, :nocc_b].copy()
    S_ov_ab = delta[:nocc_a, nocc_b:].copy()
    S_vo_ab = delta[nocc_a:, :nocc_b].copy()
    S_vv_ab = delta[nocc_a:, nocc_b:].copy()

    t2_1_a = adc.t2[0][0][:]
    t2_1_ab = adc.t2[0][1][:]
    t2_1_b = adc.t2[0][2][:]
    if adc.t1[0][0] is not None:
        t1_2_a = adc.t1[0][0][:]
        t1_2_b = adc.t1[0][1][:]
    else:
        t1_2_a = np.zeros((nocc_a, nvir_a))
        t1_2_b = np.zeros((nocc_b, nvir_b))
    if adc.t1[1][0] is not None:
        t1_3_a = adc.t1[1][0][:]
        t1_3_b = adc.t1[1][1][:]
    else:
        t1_3_a = np.zeros((nocc_a, nvir_a))
        t1_3_b = np.zeros((nocc_b, nvir_b))
    if adc.t2[1][0] is not None:
        t2_2_a = adc.t2[1][0][:]
        t2_2_ab = adc.t2[1][1][:]
        t2_2_b = adc.t2[1][2][:]
    else:
        t2_2_a = np.zeros((nocc_a, nocc_a, nvir_a, nvir_a))
        t2_2_ab = np.zeros((nocc_a, nocc_b, nvir_a, nvir_b))
        t2_2_b = np.zeros((nocc_b, nocc_b, nvir_b, nvir_b))

    if adc.frozen is not None:
        moidx_fr = adc.get_frozen_mask()
        act_a_fr = np.where(moidx_fr[0])[0]
        act_b_fr = np.where(moidx_fr[1])[0]
        cor_a_fr = np.where(~moidx_fr[0][:np.count_nonzero(adc.mo_occ[0] > 0)])[0]
        cor_b_fr = np.where(~moidx_fr[1][:np.count_nonzero(adc.mo_occ[1] > 0)])[0]
        delta_fr = np.dot(adc.mo_coeff_hf[0].transpose(), np.dot(ovlp, adc.mo_coeff_hf[1]))
        S_ac_fr = delta_fr[np.ix_(act_a_fr, cor_b_fr)]
        S_ca_fr = delta_fr[np.ix_(cor_a_fr, act_b_fr)]
        S_cc_fr = delta_fr[np.ix_(cor_a_fr, cor_b_fr)]

    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)
    n_singles_a = nvir_a
    n_singles_b = nvir_b
    n_doubles_aaa = nvir_a * (nvir_a - 1) * nocc_a // 2
    n_doubles_bab = nocc_b * nvir_a * nvir_b
    n_doubles_aba = nocc_a * nvir_b * nvir_a
    n_doubles_bbb = nvir_b * (nvir_b - 1) * nocc_b // 2

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb
    U = adc.U.T

    if method == "adc(3)":
        w_t2t2_icjd = lib.einsum('ikce,kjed->icjd', t2_1_a, t2_1_ab, optimize=True)
        w_t2t2_icjd_2 = lib.einsum('ikce,jkde->icjd', t2_1_ab, t2_1_b, optimize=True)
        w_t2t2_ijkl = lib.einsum('ijde,klde->ijkl', t2_1_ab, t2_1_ab, optimize=True)
        w_t2t2_cd = lib.einsum('jkce,jkde->cd', t2_1_a, t2_1_a, optimize=True)
        w_t2t2_cd_2 = lib.einsum('jkce,jkde->cd', t2_1_ab, t2_1_ab, optimize=True)
    spin = np.array([])
    trace_a = np.array([])
    trace_b = np.array([])

    t1_1_a = adc.t1[2][0]
    t1_1_b = adc.t1[2][1] if t1_1_a is not None else None

    for r in range(U.shape[0]):

        vec = U[r]

        L_a = vec[s_a:f_a].copy()
        L_b = vec[s_b:f_b].copy()
        L_aaa = vec[s_aaa:f_aaa].reshape(nocc_a, -1)
        L_bab = vec[s_bab:f_bab].reshape(nocc_b, nvir_a, nvir_b)
        L_aba = vec[s_aba:f_aba].reshape(nocc_a, nvir_b, nvir_a)
        L_bbb = vec[s_bbb:f_bbb].reshape(nocc_b, -1)

        L_aaa_u = np.zeros((nocc_a, nvir_a, nvir_a))
        L_aaa_u[:, ab_ind_a[0], ab_ind_a[1]] = L_aaa
        L_aaa_u[:, ab_ind_a[1], ab_ind_a[0]] = -L_aaa
        L_bbb_u = np.zeros((nocc_b, nvir_b, nvir_b))
        L_bbb_u[:, ab_ind_b[0], ab_ind_b[1]] = L_bbb
        L_bbb_u[:, ab_ind_b[1], ab_ind_b[0]] = -L_bbb

        R_a = L_a
        R_b = L_b
        R_aaa_u = L_aaa_u
        R_aba = L_aba
        R_bab = L_bab
        R_bbb_u = L_bbb_u
        S2 = 0.0

# 000
        # block AjlB
        S2 -= lib.einsum('a,b,ai,bi', L_a, R_a, S_vo_ab, S_vo_ab, optimize=True)
        # block IbdJ
        S2 -= lib.einsum('a,b,ia,ib', L_b, R_b, S_ov_ab, S_ov_ab, optimize=True)
        # block IjlJ
        S2 -= lib.einsum('a,a,ij,ij', L_a, R_a, S_oo_ab, S_oo_ab, optimize=True)
        S2 -= lib.einsum('a,a,ij,ij', L_b, R_b, S_oo_ab, S_oo_ab, optimize=True)
# 010
        if t1_1_a is not None:
            # block IjlB & AjlJ
            S2 -= 2 * lib.einsum('a,a,ij,bj,ib', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
            S2 += 2 * lib.einsum('a,b,ij,aj,ib', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ij,bj,ib', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
            # block IjdJ & IblJ
            S2 -= 2 * lib.einsum('a,a,ij,ib,jb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ij,ib,jb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
            S2 += 2 * lib.einsum('a,b,ij,ia,jb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
            # block AbdJ & IbdB
            S2 -= 2 * lib.einsum('a,b,ib,ca,ic', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, optimize=True)
            # block AblB & AjdB
            S2 -= 2 * lib.einsum('a,b,bi,ac,ic', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, optimize=True)

        # block IjdB & AblJ
        S2 -= 2 * lib.einsum('a,a,ib,cj,ijcb', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
        S2 += 2 * lib.einsum('a,b,ic,aj,ijbc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('a,a,ib,cj,ijcb', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
        S2 += 2 * lib.einsum('a,b,ia,cj,ijcb', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
# 020
        if t1_1_a is not None:
            # block IjlJ
            S2 += lib.einsum('a,a,ij,kj,ib,kb', L_a, R_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a, optimize=True)
            S2 -= lib.einsum('a,b,ij,kj,ia,kb', L_a, R_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a, optimize=True)
            S2 += lib.einsum('a,a,ij,ik,jb,kb', L_a, R_a, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b, optimize=True)
            S2 += lib.einsum('a,a,ij,kj,ib,kb', L_b, R_b, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a, optimize=True)
            S2 += lib.einsum('a,a,ij,ik,jb,kb', L_b, R_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b, optimize=True)
            S2 -= lib.einsum('a,b,ij,ik,ja,kb', L_b, R_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b, optimize=True)
            # block IjlB & AjlJ
            S2 -= lib.einsum('a,a,ij,bj,kc,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, optimize=True)
            S2 += lib.einsum('a,b,ij,aj,kc,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, optimize=True)
            S2 -= lib.einsum('a,b,ij,cj,ka,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, optimize=True)
            S2 -= lib.einsum('a,a,ij,bj,kc,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,a,ij,bk,jc,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,b,ij,aj,kc,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,b,ij,ak,jc,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,a,ij,bj,kc,ikbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, optimize=True)
            S2 -= lib.einsum('a,a,ij,bj,kc,ikbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,a,ij,bk,jc,ikbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,b,ij,cj,ka,ikcb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,b,ij,ck,ja,ikcb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
            # block IjdJ & IblJ
            S2 -= lib.einsum('a,a,ij,ib,kc,kjcb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,a,ij,kb,ic,kjcb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,b,ij,ic,ka,kjbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,b,ij,kc,ia,kjbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,a,ij,ib,kc,jkbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, optimize=True)
            S2 -= lib.einsum('a,a,ij,ib,kc,kjcb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,a,ij,kb,ic,kjcb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,b,ij,ia,kc,kjcb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,b,ij,ka,ic,kjcb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,a,ij,ib,kc,jkbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, optimize=True)
            S2 += lib.einsum('a,b,ij,ia,kc,jkbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, optimize=True)
            S2 -= lib.einsum('a,b,ij,ic,ka,jkbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, optimize=True)
            # block IjdB & AblJ
            S2 -= 2 * lib.einsum('a,a,ib,cj,ic,jb', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
            S2 += 2 * lib.einsum('a,b,ic,aj,ib,jc', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ib,cj,ic,jb', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
            S2 += 2 * lib.einsum('a,b,ia,cj,ic,jb', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
            # block IblB & AjdJ
            S2 -= 2 * lib.einsum('a,a,ij,bc,ib,jc', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
            S2 += 2 * lib.einsum('a,b,ij,ac,ib,jc', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ij,bc,ib,jc', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
            S2 += 2 * lib.einsum('a,b,ij,cb,ic,ja', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
            # block IbdJ
            S2 -= lib.einsum('a,a,ib,ic,jb,jc', L_a, R_a, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
            S2 += lib.einsum('a,b,ia,jb,ic,jc', L_b, R_b, S_ov_ab, S_ov_ab, t1_1_a, t1_1_a, optimize=True)
            S2 -= lib.einsum('a,a,ib,ic,jb,jc', L_b, R_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
            S2 += 1/2 * lib.einsum('a,b,ia,ic,jb,jc', L_b, R_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
            S2 += 1/2 * lib.einsum('a,b,ib,ic,ja,jc', L_b, R_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
            # block IbdB & AbdJ
            S2 -= 2 * lib.einsum('a,a,ib,cd,jd,ijcb', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,b,ic,ad,jd,ijbc', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,b,ia,cb,jd,ijcd', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_a, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ib,cd,jd,ijcb', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,b,ia,cb,jd,ijcd', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,b,ia,cd,jd,ijcb', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,b,ic,db,ja,ijdc', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
            # block AjlB
            S2 -= lib.einsum('a,a,bi,ci,jb,jc', L_a, R_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
            S2 += 1/2 * lib.einsum('a,b,ai,ci,jb,jc', L_a, R_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
            S2 += 1/2 * lib.einsum('a,b,bi,ci,ja,jc', L_a, R_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
            S2 += lib.einsum('a,b,ai,bj,ic,jc', L_a, R_a, S_vo_ab, S_vo_ab, t1_1_b, t1_1_b, optimize=True)
            S2 -= lib.einsum('a,a,bi,ci,jb,jc', L_b, R_b, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
            # block AjdB & AblB
            S2 -= 2 * lib.einsum('a,a,bi,cd,jc,jibd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,b,ai,bc,jd,jidc', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,b,ai,cd,jc,jibd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,b,ci,bd,ja,jicd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,b,ai,bc,jd,ijcd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('a,a,bi,cd,jc,jibd', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,b,ci,da,jd,jicb', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
            # block AbdB
            S2 -= lib.einsum('a,b,ac,bd,ic,id', L_a, R_a, S_vv_ab, S_vv_ab, t1_1_b, t1_1_b, optimize=True)
            S2 -= lib.einsum('a,b,ca,db,ic,id', L_b, R_b, S_vv_ab, S_vv_ab, t1_1_a, t1_1_a, optimize=True)

        # block AbdB
        S2 -= lib.einsum('a,a,bc,de,ijbe,ijdc', L_a, R_a, S_vv_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,b,ac,bd,ijec,ijed', L_a, R_a, S_vv_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,b,ac,de,ijbe,ijdc', L_a, R_a, S_vv_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,b,bc,de,ijae,ijdc', L_a, R_a, S_vv_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= 1 / 2 * lib.einsum('a,b,ac,bd,ijce,ijde', L_a, R_a, S_vv_ab, S_vv_ab, t2_1_b, t2_1_b, optimize=True)
        S2 -= 1 / 2 * lib.einsum('a,b,ca,db,ijce,ijde', L_b, R_b, S_vv_ab, S_vv_ab, t2_1_a, t2_1_a, optimize=True)
        S2 -= lib.einsum('a,a,bc,de,ijbe,ijdc', L_b, R_b, S_vv_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,b,ca,db,ijce,ijde', L_b, R_b, S_vv_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,b,ca,de,ijdb,ijce', L_b, R_b, S_vv_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,b,cb,de,ijda,ijce', L_b, R_b, S_vv_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
        # block AjlB
        S2 -= 1 / 2 * lib.einsum('a,a,bi,ci,jkbd,jkcd', L_a, R_a, S_vo_ab, S_vo_ab, t2_1_a, t2_1_a, optimize=True)
        S2 += 1 / 4 * lib.einsum('a,b,ai,ci,jkbd,jkcd', L_a, R_a, S_vo_ab, S_vo_ab, t2_1_a, t2_1_a, optimize=True)
        S2 += 1 / 4 * lib.einsum('a,b,bi,ci,jkad,jkcd', L_a, R_a, S_vo_ab, S_vo_ab, t2_1_a, t2_1_a, optimize=True)
        S2 += 1 / 2 * lib.einsum('a,b,ci,di,jkac,jkbd', L_a, R_a, S_vo_ab, S_vo_ab, t2_1_a, t2_1_a, optimize=True)
        S2 -= lib.einsum('a,a,bi,ci,jkbd,jkcd', L_a, R_a, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,a,bi,cj,kibd,kjcd', L_a, R_a, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,b,ai,bj,kicd,kjcd', L_a, R_a, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += 1 / 2 * lib.einsum('a,b,ai,ci,jkbd,jkcd', L_a, R_a, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,b,ai,cj,kibd,kjcd', L_a, R_a, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += 1 / 2 * lib.einsum('a,b,bi,ci,jkad,jkcd', L_a, R_a, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,b,bi,cj,kiad,kjcd', L_a, R_a, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += 1 / 2 * lib.einsum('a,b,ai,bj,ikcd,jkcd', L_a, R_a, S_vo_ab, S_vo_ab, t2_1_b, t2_1_b, optimize=True)
        S2 -= 1 / 2 * lib.einsum('a,a,bi,ci,jkbd,jkcd', L_b, R_b, S_vo_ab, S_vo_ab, t2_1_a, t2_1_a, optimize=True)
        S2 -= lib.einsum('a,a,bi,ci,jkbd,jkcd', L_b, R_b, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,a,bi,cj,kibd,kjcd', L_b, R_b, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,b,ci,di,jkca,jkdb', L_b, R_b, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,b,ci,dj,kica,kjdb', L_b, R_b, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        # block IbdJ
        S2 -= lib.einsum('a,a,ib,ic,jkdb,jkdc', L_a, R_a, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,a,ib,jc,ikdb,jkdc', L_a, R_a, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,b,ic,id,jkac,jkbd', L_a, R_a, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,b,ic,jd,ikac,jkbd', L_a, R_a, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= 1 / 2 * lib.einsum('a,a,ib,ic,jkbd,jkcd', L_a, R_a, S_ov_ab, S_ov_ab, t2_1_b, t2_1_b, optimize=True)
        S2 += 1 / 2 * lib.einsum('a,b,ia,jb,ikcd,jkcd', L_b, R_b, S_ov_ab, S_ov_ab, t2_1_a, t2_1_a, optimize=True)
        S2 -= lib.einsum('a,a,ib,ic,jkdb,jkdc', L_b, R_b, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,a,ib,jc,ikdb,jkdc', L_b, R_b, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += 1 / 2 * lib.einsum('a,b,ia,ic,jkdb,jkdc', L_b, R_b, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,b,ia,jb,ikcd,jkcd', L_b, R_b, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,b,ia,jc,ikdb,jkdc', L_b, R_b, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += 1 / 2 * lib.einsum('a,b,ib,ic,jkda,jkdc', L_b, R_b, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,b,ib,jc,ikda,jkdc', L_b, R_b, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= 1 / 2 * lib.einsum('a,a,ib,ic,jkbd,jkcd', L_b, R_b, S_ov_ab, S_ov_ab, t2_1_b, t2_1_b, optimize=True)
        S2 += 1 / 4 * lib.einsum('a,b,ia,ic,jkbd,jkcd', L_b, R_b, S_ov_ab, S_ov_ab, t2_1_b, t2_1_b, optimize=True)
        S2 += 1 / 4 * lib.einsum('a,b,ib,ic,jkad,jkcd', L_b, R_b, S_ov_ab, S_ov_ab, t2_1_b, t2_1_b, optimize=True)
        S2 += 1 / 2 * lib.einsum('a,b,ic,id,jkac,jkbd', L_b, R_b, S_ov_ab, S_ov_ab, t2_1_b, t2_1_b, optimize=True)
        # block IblB & AjdJ
        S2 -= 2 * lib.einsum('a,a,ij,bc,ikbd,kjdc', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
        S2 += 2 * lib.einsum('a,b,ij,ac,ikbd,kjdc', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('a,b,ij,cd,ikbc,kjad', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('a,a,ij,bc,ikbd,jkcd', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
        S2 += 2 * lib.einsum('a,b,ij,ac,ikbd,jkcd', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
        S2 -= 2 * lib.einsum('a,a,ij,bc,ikbd,kjdc', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
        S2 += 2 * lib.einsum('a,b,ij,cb,ikcd,kjda', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('a,a,ij,bc,ikbd,jkcd', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
        S2 += 2 * lib.einsum('a,b,ij,cb,ikcd,jkad', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
        S2 -= 2 * lib.einsum('a,b,ij,cd,ikcb,jkad', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
        # block IjlJ
        S2 += 1 / 2 * lib.einsum('a,a,ij,kj,ilbc,klbc', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_a, t2_1_a, optimize=True)
        S2 -= lib.einsum('a,b,ij,kj,ilac,klbc', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_a, t2_1_a, optimize=True)
        S2 += lib.einsum('a,a,ij,ik,ljbc,lkbc', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,a,ij,kj,ilbc,klbc', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,a,ij,kl,ilbc,kjbc', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,b,ij,ik,ljac,lkbc', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,b,ij,kj,ilac,klbc', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,b,ij,kl,ilac,kjbc', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += 1 / 2 * lib.einsum('a,a,ij,ik,jlbc,klbc', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_b, t2_1_b, optimize=True)
        S2 += 1 / 2 * lib.einsum('a,a,ij,kj,ilbc,klbc', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_a, t2_1_a, optimize=True)
        S2 += lib.einsum('a,a,ij,ik,ljbc,lkbc', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,a,ij,kj,ilbc,klbc', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,a,ij,kl,ilbc,kjbc', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,b,ij,ik,ljca,lkcb', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,b,ij,kj,ilca,klcb', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,b,ij,kl,ilca,kjcb', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += 1 / 2 * lib.einsum('a,a,ij,ik,jlbc,klbc', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_b, t2_1_b, optimize=True)
        S2 -= lib.einsum('a,b,ij,ik,jlac,klbc', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_b, t2_1_b, optimize=True)
        if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
            # block IjdB & AblJ
            S2 -= 2 * lib.einsum('a,a,ib,cj,ijcb', L_a, R_a, S_ov_ab, S_vo_ab, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('a,b,ic,aj,ijbc', L_a, R_a, S_ov_ab, S_vo_ab, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ib,cj,ijcb', L_b, R_b, S_ov_ab, S_vo_ab, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('a,b,ia,cj,ijcb', L_b, R_b, S_ov_ab, S_vo_ab, t2_2_ab, optimize=True)
            # block IjlB & AjlJ
            S2 -= 2 * lib.einsum('a,a,ij,bj,ib', L_a, R_a, S_oo_ab, S_vo_ab, t1_2_a, optimize=True)
            S2 += 2 * lib.einsum('a,b,ij,aj,ib', L_a, R_a, S_oo_ab, S_vo_ab, t1_2_a, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ij,bj,ib', L_b, R_b, S_oo_ab, S_vo_ab, t1_2_a, optimize=True)
            # block IjdJ & IblJ
            S2 -= 2 * lib.einsum('a,a,ij,ib,jb', L_a, R_a, S_oo_ab, S_ov_ab, t1_2_b, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ij,ib,jb', L_b, R_b, S_oo_ab, S_ov_ab, t1_2_b, optimize=True)
            S2 += 2 * lib.einsum('a,b,ij,ia,jb', L_b, R_b, S_oo_ab, S_ov_ab, t1_2_b, optimize=True)
            # block IbdB & AbdJ
            S2 -= 2 * lib.einsum('a,b,ia,cb,ic', L_b, R_b, S_ov_ab, S_vv_ab, t1_2_a, optimize=True)
            # block AjdB & AblB
            S2 -= 2 * lib.einsum('a,b,ai,bc,ic', L_a, R_a, S_vo_ab, S_vv_ab, t1_2_b, optimize=True)
# 101
        # block AbdB
        S2 -= lib.einsum('iab,icd,da,bc', L_aba, R_aba, S_vv_ab, S_vv_ab, optimize=True)
        S2 -= lib.einsum('iab,icd,ad,cb', L_bab, R_bab, S_vv_ab, S_vv_ab, optimize=True)
        # block AjlB
        S2 -= 1 / 4 * lib.einsum('iab,iac,bj,cj', L_aaa_u, R_aaa_u, S_vo_ab, S_vo_ab, optimize=True)
        S2 += 1 / 4 * lib.einsum('iab,ibc,aj,cj', L_aaa_u, R_aaa_u, S_vo_ab, S_vo_ab, optimize=True)
        S2 += 1 / 4 * lib.einsum('iab,ica,bj,cj', L_aaa_u, R_aaa_u, S_vo_ab, S_vo_ab, optimize=True)
        S2 -= 1 / 4 * lib.einsum('iab,icb,aj,cj', L_aaa_u, R_aaa_u, S_vo_ab, S_vo_ab, optimize=True)
        S2 -= lib.einsum('iab,iac,bj,cj', L_aba, R_aba, S_vo_ab, S_vo_ab, optimize=True)
        S2 -= lib.einsum('iab,icb,aj,cj', L_bab, R_bab, S_vo_ab, S_vo_ab, optimize=True)
        S2 += lib.einsum('iab,jcb,ai,cj', L_bab, R_bab, S_vo_ab, S_vo_ab, optimize=True)
        # block IbdJ
        S2 -= lib.einsum('iab,icb,ja,jc', L_aba, R_aba, S_ov_ab, S_ov_ab, optimize=True)
        S2 += lib.einsum('iab,jcb,ia,jc', L_aba, R_aba, S_ov_ab, S_ov_ab, optimize=True)
        S2 -= lib.einsum('iab,iac,jb,jc', L_bab, R_bab, S_ov_ab, S_ov_ab, optimize=True)
        S2 -= 1 / 4 * lib.einsum('iab,iac,jb,jc', L_bbb_u, R_bbb_u, S_ov_ab, S_ov_ab, optimize=True)
        S2 += 1 / 4 * lib.einsum('iab,ibc,ja,jc', L_bbb_u, R_bbb_u, S_ov_ab, S_ov_ab, optimize=True)
        S2 += 1 / 4 * lib.einsum('iab,ica,jb,jc', L_bbb_u, R_bbb_u, S_ov_ab, S_ov_ab, optimize=True)
        S2 -= 1 / 4 * lib.einsum('iab,icb,ja,jc', L_bbb_u, R_bbb_u, S_ov_ab, S_ov_ab, optimize=True)
        # block IblB & AjdJ
        S2 -= lib.einsum('iab,jac,ij,bc', L_aaa_u, R_bab, S_oo_ab, S_vv_ab, optimize=True)
        S2 += lib.einsum('iab,jbc,ij,ac', L_aaa_u, R_bab, S_oo_ab, S_vv_ab, optimize=True)
        S2 -= lib.einsum('iab,jac,ij,bc', L_aba, R_bbb_u, S_oo_ab, S_vv_ab, optimize=True)
        S2 += lib.einsum('iab,jca,ij,bc', L_aba, R_bbb_u, S_oo_ab, S_vv_ab, optimize=True)
        # block IjlJ
        S2 -= 1 / 4 * lib.einsum('iab,iab,jk,jk', L_aaa_u, R_aaa_u, S_oo_ab, S_oo_ab, optimize=True)
        S2 += 1 / 4 * lib.einsum('iab,iba,jk,jk', L_aaa_u, R_aaa_u, S_oo_ab, S_oo_ab, optimize=True)
        S2 += 1 / 4 * lib.einsum('iab,jab,ik,jk', L_aaa_u, R_aaa_u, S_oo_ab, S_oo_ab, optimize=True)
        S2 -= 1 / 4 * lib.einsum('iab,jba,ik,jk', L_aaa_u, R_aaa_u, S_oo_ab, S_oo_ab, optimize=True)
        S2 -= lib.einsum('iab,iab,jk,jk', L_aba, R_aba, S_oo_ab, S_oo_ab, optimize=True)
        S2 += lib.einsum('iab,jab,ik,jk', L_aba, R_aba, S_oo_ab, S_oo_ab, optimize=True)
        S2 -= lib.einsum('iab,iab,jk,jk', L_bab, R_bab, S_oo_ab, S_oo_ab, optimize=True)
        S2 += lib.einsum('iab,jab,ki,kj', L_bab, R_bab, S_oo_ab, S_oo_ab, optimize=True)
        S2 -= 1 / 4 * lib.einsum('iab,iab,jk,jk', L_bbb_u, R_bbb_u, S_oo_ab, S_oo_ab, optimize=True)
        S2 += 1 / 4 * lib.einsum('iab,iba,jk,jk', L_bbb_u, R_bbb_u, S_oo_ab, S_oo_ab, optimize=True)
        S2 += 1 / 4 * lib.einsum('iab,jab,ki,kj', L_bbb_u, R_bbb_u, S_oo_ab, S_oo_ab, optimize=True)
        S2 -= 1 / 4 * lib.einsum('iab,jba,ki,kj', L_bbb_u, R_bbb_u, S_oo_ab, S_oo_ab, optimize=True)
# 100
        # block AjdB & AblB
        S2 -= 2 * lib.einsum('iab,c,ai,cb', L_bab, R_a, S_vo_ab, S_vv_ab, optimize=True)
        # block IbdB & AbdJ
        S2 -= 2 * lib.einsum('iab,c,ia,bc', L_aba, R_b, S_ov_ab, S_vv_ab, optimize=True)
        # block IjdJ & IblJ
        S2 -= 2 * lib.einsum('iab,a,ji,jb', L_bab, R_a, S_oo_ab, S_ov_ab, optimize=True)
        S2 -= lib.einsum('iab,a,ji,jb', L_bbb_u, R_b, S_oo_ab, S_ov_ab, optimize=True)
        S2 += lib.einsum('iab,b,ji,ja', L_bbb_u, R_b, S_oo_ab, S_ov_ab, optimize=True)
        # block IjlB & AjlJ
        S2 -= lib.einsum('iab,a,ij,bj', L_aaa_u, R_a, S_oo_ab, S_vo_ab, optimize=True)
        S2 += lib.einsum('iab,b,ij,aj', L_aaa_u, R_a, S_oo_ab, S_vo_ab, optimize=True)
        S2 -= 2 * lib.einsum('iab,a,ij,bj', L_aba, R_b, S_oo_ab, S_vo_ab, optimize=True)
# 110
        if t1_1_a is not None:
            # block IjlJ
            S2 += 1/2 * lib.einsum('a,iab,jb,ik,jk', L_a, R_aaa_u, t1_1_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 1/2 * lib.einsum('a,iba,jb,ik,jk', L_a, R_aaa_u, t1_1_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('a,iab,jb,ki,kj', L_a, R_bab, t1_1_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 1/2 * lib.einsum('iab,a,jb,ik,jk', L_aaa_u, R_a, t1_1_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 1/2 * lib.einsum('iab,b,ja,ik,jk', L_aaa_u, R_a, t1_1_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('iab,a,jb,ik,jk', L_aba, R_b, t1_1_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('a,iab,jb,ik,jk', L_b, R_aba, t1_1_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 1/2 * lib.einsum('a,iab,jb,ki,kj', L_b, R_bbb_u, t1_1_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 1/2 * lib.einsum('a,iba,jb,ki,kj', L_b, R_bbb_u, t1_1_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('iab,a,jb,ki,kj', L_bab, R_a, t1_1_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 1/2 * lib.einsum('iab,a,jb,ki,kj', L_bbb_u, R_b, t1_1_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 1/2 * lib.einsum('iab,b,ja,ki,kj', L_bbb_u, R_b, t1_1_b, S_oo_ab, S_oo_ab, optimize=True)
            # block IjdB & AblJ
            S2 -= lib.einsum('iab,a,ic,bj,jc', L_aaa_u, R_a, S_ov_ab, S_vo_ab, t1_1_b, optimize=True)
            S2 += lib.einsum('iab,b,ic,aj,jc', L_aaa_u, R_a, S_ov_ab, S_vo_ab, t1_1_b, optimize=True)
            S2 -= 2 * lib.einsum('iab,a,ic,bj,jc', L_aba, R_b, S_ov_ab, S_vo_ab, t1_1_b, optimize=True)
            S2 += 2 * lib.einsum('iab,c,ia,bj,jc', L_aba, R_b, S_ov_ab, S_vo_ab, t1_1_b, optimize=True)
            S2 -= 2 * lib.einsum('iab,a,jb,ci,jc', L_bab, R_a, S_ov_ab, S_vo_ab, t1_1_a, optimize=True)
            S2 += 2 * lib.einsum('iab,c,jb,ai,jc', L_bab, R_a, S_ov_ab, S_vo_ab, t1_1_a, optimize=True)
            S2 -= lib.einsum('iab,a,jb,ci,jc', L_bbb_u, R_b, S_ov_ab, S_vo_ab, t1_1_a, optimize=True)
            S2 += lib.einsum('iab,b,ja,ci,jc', L_bbb_u, R_b, S_ov_ab, S_vo_ab, t1_1_a, optimize=True)
            # block IblB & AjdJ
            S2 -= 2 * lib.einsum('a,iab,ji,cb,jc', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_a, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,ji,ac,jb', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_a, optimize=True)
            S2 -= lib.einsum('iab,a,ij,bc,jc', L_aaa_u, R_a, S_oo_ab, S_vv_ab, t1_1_b, optimize=True)
            S2 += lib.einsum('iab,b,ij,ac,jc', L_aaa_u, R_a, S_oo_ab, S_vv_ab, t1_1_b, optimize=True)
            S2 -= 2 * lib.einsum('iab,a,ij,bc,jc', L_aba, R_b, S_oo_ab, S_vv_ab, t1_1_b, optimize=True)
            S2 += 2 * lib.einsum('iab,c,ij,bc,ja', L_aba, R_b, S_oo_ab, S_vv_ab, t1_1_b, optimize=True)
            S2 -= lib.einsum('a,iab,ji,cb,jc', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_a, optimize=True)
            S2 += lib.einsum('a,iba,ji,cb,jc', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_a, optimize=True)
            # block IbdJ
            S2 -= lib.einsum('a,iab,ic,jb,jc', L_a, R_bab, t1_1_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 += lib.einsum('iab,c,jb,ia,jc', L_aba, R_b, t1_1_a, S_ov_ab, S_ov_ab, optimize=True)
            S2 += lib.einsum('a,ibc,jc,ja,ib', L_b, R_aba, t1_1_a, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= 1/2 * lib.einsum('a,iab,ic,jb,jc', L_b, R_bbb_u, t1_1_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 += 1/2 * lib.einsum('a,iba,ic,jb,jc', L_b, R_bbb_u, t1_1_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= lib.einsum('iab,a,ic,jb,jc', L_bab, R_a, t1_1_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= 1/2 * lib.einsum('iab,a,ic,jb,jc', L_bbb_u, R_b, t1_1_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 += 1/2 * lib.einsum('iab,b,ic,ja,jc', L_bbb_u, R_b, t1_1_b, S_ov_ab, S_ov_ab, optimize=True)
            # block AjlB
            S2 -= 1/2 * lib.einsum('a,iab,ic,bj,cj', L_a, R_aaa_u, t1_1_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 += 1/2 * lib.einsum('a,iba,ic,bj,cj', L_a, R_aaa_u, t1_1_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 += lib.einsum('a,ibc,jc,aj,bi', L_a, R_bab, t1_1_b, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= 1/2 * lib.einsum('iab,a,ic,bj,cj', L_aaa_u, R_a, t1_1_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 += 1/2 * lib.einsum('iab,b,ic,aj,cj', L_aaa_u, R_a, t1_1_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= lib.einsum('iab,a,ic,bj,cj', L_aba, R_b, t1_1_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= lib.einsum('a,iab,ic,bj,cj', L_b, R_aba, t1_1_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 += lib.einsum('iab,c,jb,ai,cj', L_bab, R_a, t1_1_b, S_vo_ab, S_vo_ab, optimize=True)
            # block AbdB
            S2 -= lib.einsum('a,ibc,id,ac,bd', L_a, R_bab, t1_1_b, S_vv_ab, S_vv_ab, optimize=True)
            S2 -= lib.einsum('iab,c,id,da,bc', L_aba, R_b, t1_1_a, S_vv_ab, S_vv_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,id,ca,db', L_b, R_aba, t1_1_a, S_vv_ab, S_vv_ab, optimize=True)
            S2 -= lib.einsum('iab,c,id,ad,cb', L_bab, R_a, t1_1_b, S_vv_ab, S_vv_ab, optimize=True)

        # block AjdB & AblB
        S2 -= lib.einsum('a,iab,cj,bd,ijcd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,iba,cj,bd,ijcd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,ibc,aj,bd,ijcd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,ibc,aj,cd,ijbd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('a,ibc,aj,bd,ijcd', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_b, optimize=True)
        S2 -= 2 * lib.einsum('a,iab,cj,bd,ijcd', L_b, R_aba, S_vo_ab, S_vv_ab, t2_1_ab, optimize=True)
        S2 += 2 * lib.einsum('a,ibc,dj,ca,ijdb', L_b, R_aba, S_vo_ab, S_vv_ab, t2_1_ab, optimize=True)
        # block IbdB & AbdJ
        S2 -= 2 * lib.einsum('a,iab,jc,db,jidc', L_a, R_bab, S_ov_ab, S_vv_ab, t2_1_ab, optimize=True)
        S2 += 2 * lib.einsum('a,ibc,jd,ac,jibd', L_a, R_bab, S_ov_ab, S_vv_ab, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('a,ibc,ja,db,ijcd', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_a, optimize=True)
        S2 -= lib.einsum('a,iab,jc,db,jidc', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,iba,jc,db,jidc', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,ibc,ja,db,jidc', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,ibc,ja,dc,jidb', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_ab, optimize=True)
        # block IjdJ & IblJ
        S2 += lib.einsum('a,iab,ij,kc,kjbc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,iab,jk,jc,ikbc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,iba,ij,kc,kjbc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,iba,jk,jc,ikbc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('a,iab,jk,jc,ikbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_b, optimize=True)
        S2 += 2 * lib.einsum('a,iab,ij,kc,kjbc', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('a,iab,jk,jc,ikbc', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('a,ibc,ij,ka,kjcb', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_ab, optimize=True)
        S2 += 2 * lib.einsum('a,ibc,jk,ja,ikcb', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,iab,jk,jc,ikbc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b, optimize=True)
        S2 += lib.einsum('a,iba,jk,jc,ikbc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b, optimize=True)
        S2 -= lib.einsum('a,ibc,jk,ja,ikbc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b, optimize=True)
        # block IjlB & AjlJ
        S2 -= lib.einsum('a,iab,jk,ck,ijbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a, optimize=True)
        S2 += lib.einsum('a,iba,jk,ck,ijbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a, optimize=True)
        S2 -= lib.einsum('a,ibc,jk,ak,ijbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a, optimize=True)
        S2 += 2 * lib.einsum('a,iab,ji,ck,jkcb', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('a,iab,jk,ck,jicb', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('a,ibc,ji,ak,jkbc', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_ab, optimize=True)
        S2 += 2 * lib.einsum('a,ibc,jk,ak,jibc', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('a,iab,jk,ck,ijbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_a, optimize=True)
        S2 += lib.einsum('a,iab,ji,ck,jkcb', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,iab,jk,ck,jicb', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('a,iba,ji,ck,jkcb', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('a,iba,jk,ck,jicb', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab, optimize=True)
        if method in ("adc(2)-x", "adc(3)"):
            # 111
            if t1_1_a is not None:
                # block IjlB & AjlJ
                S2 -= 1/2 * lib.einsum('iab,iab,jk,ck,jc', L_aaa_u, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 += 1/2 * lib.einsum('iab,iac,jk,bk,jc', L_aaa_u, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 += 1/2 * lib.einsum('iab,iba,jk,ck,jc', L_aaa_u, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 -= 1/2 * lib.einsum('iab,ibc,jk,ak,jc', L_aaa_u, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 -= 1/2 * lib.einsum('iab,ica,jk,bk,jc', L_aaa_u, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 += 1/2 * lib.einsum('iab,icb,jk,ak,jc', L_aaa_u, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 += 1/2 * lib.einsum('iab,jab,ik,ck,jc', L_aaa_u, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 -= 1/2 * lib.einsum('iab,jba,ik,ck,jc', L_aaa_u, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 += lib.einsum('iab,jac,ij,bk,kc', L_aaa_u, R_bab, S_oo_ab, S_vo_ab, t1_1_b, optimize=True)
                S2 -= lib.einsum('iab,jbc,ij,ak,kc', L_aaa_u, R_bab, S_oo_ab, S_vo_ab, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('iab,iab,jk,ck,jc', L_aba, R_aba, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('iab,iac,jk,bk,jc', L_aba, R_aba, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('iab,jab,ik,ck,jc', L_aba, R_aba, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 += lib.einsum('iab,jac,ij,bk,kc', L_aba, R_bbb_u, S_oo_ab, S_vo_ab, t1_1_b, optimize=True)
                S2 -= lib.einsum('iab,jca,ij,bk,kc', L_aba, R_bbb_u, S_oo_ab, S_vo_ab, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('iab,iab,jk,ck,jc', L_bab, R_bab, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('iab,icb,jk,ak,jc', L_bab, R_bab, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('iab,jab,kj,ci,kc', L_bab, R_bab, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 -= 2 * lib.einsum('iab,jcb,kj,ai,kc', L_bab, R_bab, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 -= 1/2 * lib.einsum('iab,iab,jk,ck,jc', L_bbb_u, R_bbb_u, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 += 1/2 * lib.einsum('iab,iba,jk,ck,jc', L_bbb_u, R_bbb_u, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 += 1/2 * lib.einsum('iab,jab,kj,ci,kc', L_bbb_u, R_bbb_u, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 -= 1/2 * lib.einsum('iab,jba,kj,ci,kc', L_bbb_u, R_bbb_u, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                # block IjdJ & IblJ
                S2 -= 1/2 * lib.einsum('iab,iab,jk,jc,kc', L_aaa_u, R_aaa_u, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 += 1/2 * lib.einsum('iab,iba,jk,jc,kc', L_aaa_u, R_aaa_u, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 += 1/2 * lib.einsum('iab,jab,jk,ic,kc', L_aaa_u, R_aaa_u, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 -= 1/2 * lib.einsum('iab,jba,jk,ic,kc', L_aaa_u, R_aaa_u, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('iab,iab,jk,jc,kc', L_aba, R_aba, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('iab,icb,jk,ja,kc', L_aba, R_aba, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('iab,jab,jk,ic,kc', L_aba, R_aba, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('iab,jcb,jk,ia,kc', L_aba, R_aba, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 += lib.einsum('iab,jac,ji,kb,kc', L_bab, R_aaa_u, S_oo_ab, S_ov_ab, t1_1_a, optimize=True)
                S2 -= lib.einsum('iab,jca,ji,kb,kc', L_bab, R_aaa_u, S_oo_ab, S_ov_ab, t1_1_a, optimize=True)
                S2 -= 2 * lib.einsum('iab,iab,jk,jc,kc', L_bab, R_bab, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('iab,iac,jk,jb,kc', L_bab, R_bab, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('iab,jab,ki,kc,jc', L_bab, R_bab, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 += lib.einsum('iab,jac,ji,kb,kc', L_bbb_u, R_aba, S_oo_ab, S_ov_ab, t1_1_a, optimize=True)
                S2 -= lib.einsum('iab,jbc,ji,ka,kc', L_bbb_u, R_aba, S_oo_ab, S_ov_ab, t1_1_a, optimize=True)
                S2 -= 1/2 * lib.einsum('iab,iab,jk,jc,kc', L_bbb_u, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 += 1/2 * lib.einsum('iab,iac,jk,jb,kc', L_bbb_u, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 += 1/2 * lib.einsum('iab,iba,jk,jc,kc', L_bbb_u, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 -= 1/2 * lib.einsum('iab,ibc,jk,ja,kc', L_bbb_u, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 -= 1/2 * lib.einsum('iab,ica,jk,jb,kc', L_bbb_u, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 += 1/2 * lib.einsum('iab,icb,jk,ja,kc', L_bbb_u, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 += 1/2 * lib.einsum('iab,jab,ki,kc,jc', L_bbb_u, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 -= 1/2 * lib.einsum('iab,jba,ki,kc,jc', L_bbb_u, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                # block IbdB & AbdJ
                S2 -= lib.einsum('iab,jac,id,bc,jd', L_aaa_u, R_bab, S_ov_ab, S_vv_ab, t1_1_b, optimize=True)
                S2 += lib.einsum('iab,jbc,id,ac,jd', L_aaa_u, R_bab, S_ov_ab, S_vv_ab, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('iab,icb,ja,dc,jd', L_aba, R_aba, S_ov_ab, S_vv_ab, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('iab,icd,ja,bc,jd', L_aba, R_aba, S_ov_ab, S_vv_ab, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('iab,jcb,ia,dc,jd', L_aba, R_aba, S_ov_ab, S_vv_ab, t1_1_a, optimize=True)
                S2 -= lib.einsum('iab,jac,id,bc,jd', L_aba, R_bbb_u, S_ov_ab, S_vv_ab, t1_1_b, optimize=True)
                S2 += lib.einsum('iab,jca,id,bc,jd', L_aba, R_bbb_u, S_ov_ab, S_vv_ab, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('iab,iac,jb,dc,jd', L_bab, R_bab, S_ov_ab, S_vv_ab, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('iab,icd,jb,ad,jc', L_bab, R_bab, S_ov_ab, S_vv_ab, t1_1_a, optimize=True)
                S2 -= 1/2 * lib.einsum('iab,iac,jb,dc,jd', L_bbb_u, R_bbb_u, S_ov_ab, S_vv_ab, t1_1_a, optimize=True)
                S2 += 1/2 * lib.einsum('iab,ibc,ja,dc,jd', L_bbb_u, R_bbb_u, S_ov_ab, S_vv_ab, t1_1_a, optimize=True)
                S2 += 1/2 * lib.einsum('iab,ica,jb,dc,jd', L_bbb_u, R_bbb_u, S_ov_ab, S_vv_ab, t1_1_a, optimize=True)
                S2 -= 1/2 * lib.einsum('iab,icb,ja,dc,jd', L_bbb_u, R_bbb_u, S_ov_ab, S_vv_ab, t1_1_a, optimize=True)
                # block AjdB & AblB
                S2 -= 1/2 * lib.einsum('iab,iac,bj,cd,jd', L_aaa_u, R_aaa_u, S_vo_ab, S_vv_ab, t1_1_b, optimize=True)
                S2 += 1/2 * lib.einsum('iab,ibc,aj,cd,jd', L_aaa_u, R_aaa_u, S_vo_ab, S_vv_ab, t1_1_b, optimize=True)
                S2 += 1/2 * lib.einsum('iab,ica,bj,cd,jd', L_aaa_u, R_aaa_u, S_vo_ab, S_vv_ab, t1_1_b, optimize=True)
                S2 -= 1/2 * lib.einsum('iab,icb,aj,cd,jd', L_aaa_u, R_aaa_u, S_vo_ab, S_vv_ab, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('iab,iac,bj,cd,jd', L_aba, R_aba, S_vo_ab, S_vv_ab, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('iab,icd,bj,da,jc', L_aba, R_aba, S_vo_ab, S_vv_ab, t1_1_b, optimize=True)
                S2 -= lib.einsum('iab,jac,di,cb,jd', L_bab, R_aaa_u, S_vo_ab, S_vv_ab, t1_1_a, optimize=True)
                S2 += lib.einsum('iab,jca,di,cb,jd', L_bab, R_aaa_u, S_vo_ab, S_vv_ab, t1_1_a, optimize=True)
                S2 -= 2 * lib.einsum('iab,icb,aj,cd,jd', L_bab, R_bab, S_vo_ab, S_vv_ab, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('iab,icd,aj,cb,jd', L_bab, R_bab, S_vo_ab, S_vv_ab, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('iab,jcb,ai,cd,jd', L_bab, R_bab, S_vo_ab, S_vv_ab, t1_1_b, optimize=True)
                S2 -= lib.einsum('iab,jac,di,cb,jd', L_bbb_u, R_aba, S_vo_ab, S_vv_ab, t1_1_a, optimize=True)
                S2 += lib.einsum('iab,jbc,di,ca,jd', L_bbb_u, R_aba, S_vo_ab, S_vv_ab, t1_1_a, optimize=True)

            # block IjdB & AblJ
            S2 -= 1 / 2 * lib.einsum('iab,iab,jc,dk,jkdc', L_aaa_u, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('iab,iac,jd,bk,jkcd', L_aaa_u, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('iab,iba,jc,dk,jkdc', L_aaa_u, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('iab,ibc,jd,ak,jkcd', L_aaa_u, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('iab,ica,jd,bk,jkcd', L_aaa_u, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('iab,icb,jd,ak,jkcd', L_aaa_u, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('iab,jab,ic,dk,jkdc', L_aaa_u, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('iab,jac,id,bk,jkcd', L_aaa_u, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('iab,jba,ic,dk,jkdc', L_aaa_u, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('iab,jbc,id,ak,jkcd', L_aaa_u, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('iab,jca,id,bk,jkcd', L_aaa_u, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('iab,jcb,id,ak,jkcd', L_aaa_u, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('iab,jac,id,bk,jkcd', L_aaa_u, R_bab, S_ov_ab, S_vo_ab, t2_1_b, optimize=True)
            S2 += lib.einsum('iab,jbc,id,ak,jkcd', L_aaa_u, R_bab, S_ov_ab, S_vo_ab, t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('iab,iab,jc,dk,jkdc', L_aba, R_aba, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('iab,iac,jd,bk,jkcd', L_aba, R_aba, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('iab,icb,ja,dk,jkdc', L_aba, R_aba, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('iab,icd,ja,bk,jkdc', L_aba, R_aba, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('iab,jab,ic,dk,jkdc', L_aba, R_aba, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('iab,jac,id,bk,jkcd', L_aba, R_aba, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('iab,jcb,ia,dk,jkdc', L_aba, R_aba, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('iab,jcd,ia,bk,jkdc', L_aba, R_aba, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('iab,jac,id,bk,jkcd', L_aba, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_b, optimize=True)
            S2 += lib.einsum('iab,jca,id,bk,jkcd', L_aba, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_b, optimize=True)
            S2 -= lib.einsum('iab,jcd,ia,bk,jkcd', L_aba, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_b, optimize=True)
            S2 -= lib.einsum('iab,jac,kb,di,jkcd', L_bab, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_a, optimize=True)
            S2 += lib.einsum('iab,jca,kb,di,jkcd', L_bab, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_a, optimize=True)
            S2 -= lib.einsum('iab,jcd,kb,ai,jkcd', L_bab, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_a, optimize=True)
            S2 -= 2 * lib.einsum('iab,iab,jc,dk,jkdc', L_bab, R_bab, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('iab,iac,jb,dk,jkdc', L_bab, R_bab, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('iab,icb,jd,ak,jkcd', L_bab, R_bab, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('iab,icd,jb,ak,jkcd', L_bab, R_bab, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('iab,jab,kc,di,kjdc', L_bab, R_bab, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('iab,jac,kb,di,kjdc', L_bab, R_bab, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('iab,jcb,kd,ai,kjcd', L_bab, R_bab, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('iab,jcd,kb,ai,kjcd', L_bab, R_bab, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('iab,jac,kb,di,jkcd', L_bbb_u, R_aba, S_ov_ab, S_vo_ab, t2_1_a, optimize=True)
            S2 += lib.einsum('iab,jbc,ka,di,jkcd', L_bbb_u, R_aba, S_ov_ab, S_vo_ab, t2_1_a, optimize=True)
            S2 -= 1 / 2 * lib.einsum('iab,iab,jc,dk,jkdc', L_bbb_u, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('iab,iac,jb,dk,jkdc', L_bbb_u, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('iab,iba,jc,dk,jkdc', L_bbb_u, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('iab,ibc,ja,dk,jkdc', L_bbb_u, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('iab,ica,jb,dk,jkdc', L_bbb_u, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('iab,icb,ja,dk,jkdc', L_bbb_u, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('iab,jab,kc,di,kjdc', L_bbb_u, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('iab,jac,kb,di,kjdc', L_bbb_u, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('iab,jba,kc,di,kjdc', L_bbb_u, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('iab,jbc,ka,di,kjdc', L_bbb_u, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('iab,jca,kb,di,kjdc', L_bbb_u, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('iab,jcb,ka,di,kjdc', L_bbb_u, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)

        if method == "adc(3)":
            # 030
            if t1_1_a is not None:
                # block IjlJ
                S2 += 2 * lib.einsum('a,a,ib,jb,ik,jk', L_a, R_a, t1_1_a, t1_2_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('a,b,ia,jb,ik,jk', L_a, R_a, t1_1_a, t1_2_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('a,b,ib,ja,ik,jk', L_a, R_a, t1_1_a, t1_2_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 += 2 * lib.einsum('a,a,ib,jb,ki,kj', L_a, R_a, t1_1_b, t1_2_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 += 2 * lib.einsum('a,a,ib,jb,ik,jk', L_b, R_b, t1_1_a, t1_2_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 += 2 * lib.einsum('a,a,ib,jb,ki,kj', L_b, R_b, t1_1_b, t1_2_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('a,b,ia,jb,ki,kj', L_b, R_b, t1_1_b, t1_2_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('a,b,ib,ja,ki,kj', L_b, R_b, t1_1_b, t1_2_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 += lib.einsum('a,a,ib,jc,ikbc,lj,lk', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('a,a,ib,jc,kjbc,il,kl', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,ib,jc,klbc,il,kj', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ia,jc,ikbc,lj,lk', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ia,jc,kjbc,il,kl', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ia,jc,klbc,il,kj', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ib,jc,ikac,lj,lk', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ib,jc,kjac,il,kl', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ib,jc,klac,il,kj', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= lib.einsum('a,a,ijbc,ik,lk,jb,lc', L_a, R_a, t2_1_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ijac,ik,lk,jb,lc', L_a, R_a, t2_1_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ijac,ik,lk,lb,jc', L_a, R_a, t2_1_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ijbc,ik,lk,ja,lc', L_a, R_a, t2_1_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ijbc,ik,lk,la,jc', L_a, R_a, t2_1_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('a,a,ijbc,ki,kl,jb,lc', L_a, R_a, t2_1_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += lib.einsum('a,a,ib,jc,ikbc,lj,lk', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('a,a,ib,jc,kjbc,il,kl', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,ib,jc,klbc,il,kj', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ic,ja,ikcb,lj,lk', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ic,ja,kjcb,il,kl', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ic,ja,klcb,il,kj', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ic,jb,ikca,lj,lk', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ic,jb,kjca,il,kl', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ic,jb,klca,il,kj', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= lib.einsum('a,a,ijbc,ik,lk,jb,lc', L_b, R_b, t2_1_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('a,a,ijbc,ki,kl,jb,lc', L_b, R_b, t2_1_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ijac,ki,kl,jb,lc', L_b, R_b, t2_1_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ijac,ki,kl,lb,jc', L_b, R_b, t2_1_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ijbc,ki,kl,ja,lc', L_b, R_b, t2_1_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ijbc,ki,kl,la,jc', L_b, R_b, t2_1_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b,
                    optimize=True)
                # block IjlB & AjlJ
                S2 -= lib.einsum('a,a,ij,bj,kc,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_2_a, optimize=True)
                S2 += lib.einsum('a,b,ij,aj,kc,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_2_a, optimize=True)
                S2 -= lib.einsum('a,b,ij,cj,ka,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_2_a, optimize=True)
                S2 -= lib.einsum('a,a,ij,bj,kc,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,bk,jc,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 += lib.einsum('a,b,ij,aj,kc,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,ak,jc,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 -= lib.einsum('a,a,ij,bj,kc,ikbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_2_a, optimize=True)
                S2 -= lib.einsum('a,a,ij,bj,kc,ikbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,bk,jc,ikbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 += lib.einsum('a,b,ij,cj,ka,ikcb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,ck,ja,ikcb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 += 4/3 * lib.einsum('a,a,ij,bj,ic,kb,kc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ij,aj,kb,ic,kc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('a,b,ij,cj,ka,ib,kc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,bk,ib,jc,kc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,ak,ib,jc,kc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 2/3 * lib.einsum('a,a,ij,bj,ic,klbd,klcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 2/3 * lib.einsum('a,a,ij,bj,kb,ilcd,klcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,a,ij,bj,kc,ilbd,klcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 1/6 * lib.einsum('a,b,ij,aj,ic,klbd,klcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 1/6 * lib.einsum('a,b,ij,aj,kb,ilcd,klcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ij,aj,kc,ilbd,klcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ij,cj,ib,klad,klcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 2/3 * lib.einsum('a,b,ij,cj,id,klad,klbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ij,cj,kb,klad,ilcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('a,b,ij,cj,kc,klad,ilbd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ij,cj,kd,klad,ilbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,a,ij,bj,ic,klbd,klcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,a,ij,bj,kb,ilcd,klcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,a,ij,bj,kc,ilbd,klcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,bk,ib,ljcd,lkcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,ij,bk,ic,ljcd,lkbd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,ij,bk,lb,ikcd,ljcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,a,ij,bk,lc,ikbd,ljcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ij,aj,ic,klbd,klcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ij,aj,kb,ilcd,klcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ij,aj,kc,ilbd,klcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,ak,ib,ljcd,lkcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ij,ak,ic,lkbd,ljcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ij,ak,lb,ikcd,ljcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ij,ak,lc,ikbd,ljcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ij,cj,ib,klad,klcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ij,cj,kb,klad,ilcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('a,b,ij,cj,kc,klad,ilbd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ij,ck,ib,ljad,lkcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,ck,ic,ljad,lkbd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ij,ck,lb,ljad,ikcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ij,ck,lc,ljad,ikbd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,a,ij,bk,ib,jlcd,klcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,b,ij,ak,ib,jlcd,klcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,a,ij,bj,kc,ilbd,lkdc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,bk,kc,ilbd,ljdc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ij,aj,kc,ilbd,lkdc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,ak,kc,ilbd,ljdc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ij,cj,kd,ilbc,lkad', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ij,ck,kd,ilbc,ljad', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,a,ij,bj,kc,ilbd,klcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,bk,kc,ilbd,jlcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,a,ij,bk,lc,ikbd,jlcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ij,aj,kc,ilbd,klcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,ak,kc,ilbd,jlcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('a,b,ij,ak,lc,ikbd,jlcd', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,a,ij,bj,ic,kb,kc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,bk,ib,jc,kc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,ck,ic,ja,kb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 2/3 * lib.einsum('a,a,ij,bj,ic,klbd,klcd', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 2/3 * lib.einsum('a,a,ij,bj,kb,ilcd,klcd', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,a,ij,bj,kc,ilbd,klcd', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,a,ij,bj,ic,klbd,klcd', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,a,ij,bj,kb,ilcd,klcd', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,a,ij,bj,kc,ilbd,klcd', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,bk,ib,ljcd,lkcd', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,ij,bk,ic,ljcd,lkbd', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,ij,bk,lb,ikcd,ljcd', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,a,ij,bk,lc,ikbd,ljcd', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('a,b,ij,cj,id,klda,klcb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('a,b,ij,cj,kc,klda,ildb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ij,cj,kd,klda,ilcb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,ck,ic,ljda,lkdb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ij,ck,id,ljda,lkcb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ij,ck,lc,ljda,ikdb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ij,ck,ld,ljda,ikcb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,a,ij,bk,ib,jlcd,klcd', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,ck,ic,jlad,klbd', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,a,ij,bj,kc,ilbd,lkdc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,bk,kc,ilbd,ljdc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ij,cj,kb,ilcd,lkda', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,ck,kb,ilcd,ljda', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,a,ij,bj,kc,ilbd,klcd', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,bk,kc,ilbd,jlcd', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,a,ij,bk,lc,ikbd,jlcd', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ij,cj,kb,ilcd,klad', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ij,cj,kd,ilcb,klad', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,ck,kb,ilcd,jlad', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ij,ck,kd,ilcb,jlad', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('a,b,ij,ck,lb,ikcd,jlad', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,b,ij,ck,ld,ikcb,jlad', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                # block IjdJ & IblJ
                S2 -= lib.einsum('a,a,ij,ib,kc,kjcb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,kb,ic,kjcb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 += lib.einsum('a,b,ij,ic,ka,kjbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,kc,ia,kjbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 -= lib.einsum('a,a,ij,ib,kc,jkbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_2_b, optimize=True)
                S2 -= lib.einsum('a,a,ij,ib,kc,kjcb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,kb,ic,kjcb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 += lib.einsum('a,b,ij,ia,kc,kjcb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,ka,ic,kjcb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 -= lib.einsum('a,a,ij,ib,kc,jkbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_2_b, optimize=True)
                S2 += lib.einsum('a,b,ij,ia,kc,jkbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_2_b, optimize=True)
                S2 -= lib.einsum('a,b,ij,ic,ka,jkbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_2_b, optimize=True)
                S2 -= 1/3 * lib.einsum('a,a,ij,ib,kc,klcd,ljdb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,kb,kc,ilcd,ljdb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,a,ij,kb,lc,ilcd,kjdb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ij,ic,kb,klad,ljdc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ij,ic,kd,klad,ljbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,kc,kb,ilad,ljdc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ij,kc,kd,ilad,ljbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ij,kc,lb,ilad,kjdc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ij,kc,ld,ilad,kjbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,a,ij,ib,kc,klcd,jlbd', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,kb,kc,ilcd,jlbd', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ij,ic,kb,klad,jlcd', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,kc,kb,ilad,jlcd', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,kb,jb,ic,kc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,kc,jc,ia,kb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,a,ij,ib,jc,kb,kc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += lib.einsum('a,a,ij,kb,jb,ilcd,klcd', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,kc,jc,ilad,klbd', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,a,ij,ib,jc,kldb,kldc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,a,ij,ib,kb,ljcd,lkcd', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,a,ij,ib,kc,ljdb,lkdc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,kb,jb,ilcd,klcd', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,ij,kb,jc,ildc,kldb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,ij,kb,lb,ilcd,kjcd', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,a,ij,kb,lc,ildc,kjdb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('a,b,ij,ic,jd,klad,klbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('a,b,ij,ic,kc,lkad,ljbd', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ij,ic,kd,lkad,ljbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,kc,jc,ilad,klbd', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ij,kc,jd,ilad,klbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ij,kc,lc,ilad,kjbd', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ij,kc,ld,ilad,kjbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2/3 * lib.einsum('a,a,ij,ib,jc,klbd,klcd', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 += 2/3 * lib.einsum('a,a,ij,ib,kb,jlcd,klcd', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,a,ij,ib,kc,jlbd,klcd', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,a,ij,ib,kc,klcd,ljdb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,kb,kc,ilcd,ljdb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,a,ij,kb,lc,ilcd,kjdb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ij,ia,kc,klcd,ljdb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,ka,kc,ilcd,ljdb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ij,ka,lc,ilcd,kjdb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,a,ij,ib,kc,klcd,jlbd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,kb,kc,ilcd,jlbd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ij,ia,kc,klcd,jlbd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ij,ic,kd,klda,jlbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,ka,kc,ilcd,jlbd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ij,kc,kd,ilda,jlbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,kb,jb,ic,kc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,ka,jb,ic,kc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,a,ij,ib,jc,kb,kc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ij,ia,kb,jc,kc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,b,ij,ic,ka,jb,kc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += lib.einsum('a,a,ij,kb,jb,ilcd,klcd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= lib.einsum('a,b,ij,ka,jb,ilcd,klcd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,a,ij,ib,jc,kldb,kldc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,a,ij,ib,kb,ljcd,lkcd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,a,ij,ib,kc,ljdb,lkdc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,kb,jb,ilcd,klcd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,ij,kb,jc,ildc,kldb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,ij,kb,lb,ilcd,kjcd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,a,ij,kb,lc,ildc,kjdb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ij,ia,jc,kldb,kldc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ij,ia,kb,ljcd,lkcd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ij,ia,kc,ljdb,lkdc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ij,ic,jb,klda,kldc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ij,ic,kb,lkda,ljdc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('a,b,ij,ic,kc,lkda,ljdb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,ka,jb,ilcd,klcd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ij,ka,jc,kldb,ildc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ij,ka,lb,ilcd,kjcd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ij,ka,lc,kjdb,ildc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ij,kc,jb,ilda,kldc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,kc,jc,ilda,kldb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ij,kc,lb,ilda,kjdc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ij,kc,lc,ilda,kjdb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2/3 * lib.einsum('a,a,ij,ib,jc,klbd,klcd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 += 2/3 * lib.einsum('a,a,ij,ib,kb,jlcd,klcd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,a,ij,ib,kc,jlbd,klcd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 1/6 * lib.einsum('a,b,ij,ia,jc,klbd,klcd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 1/6 * lib.einsum('a,b,ij,ia,kb,jlcd,klcd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ij,ia,kc,jlbd,klcd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ij,ic,jb,klad,klcd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 2/3 * lib.einsum('a,b,ij,ic,jd,klad,klbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ij,ic,kb,klad,jlcd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('a,b,ij,ic,kc,klad,jlbd', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ij,ic,kd,klad,jlbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                # block IjdB & AblJ
                S2 -= 2 * lib.einsum('a,a,ib,cj,ic,jb', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_a, t1_2_b, optimize=True)
                S2 += 2 * lib.einsum('a,b,ic,aj,ib,jc', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_a, t1_2_b, optimize=True)
                S2 -= 2 * lib.einsum('a,a,ib,cj,jb,ic', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_b, t1_2_a, optimize=True)
                S2 += 2 * lib.einsum('a,b,ic,aj,jc,ib', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_b, t1_2_a, optimize=True)
                S2 -= 2 * lib.einsum('a,a,ib,cj,ic,jb', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_a, t1_2_b, optimize=True)
                S2 += 2 * lib.einsum('a,b,ia,cj,ic,jb', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_a, t1_2_b, optimize=True)
                S2 -= 2 * lib.einsum('a,a,ib,cj,jb,ic', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_b, t1_2_a, optimize=True)
                S2 += 2 * lib.einsum('a,b,ia,cj,jb,ic', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_b, t1_2_a, optimize=True)
                S2 -= lib.einsum('a,a,ib,cj,kd,jb,ikcd', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, t2_1_a,
                    optimize=True)
                S2 += lib.einsum('a,b,ic,aj,kd,jc,ikbd', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, t2_1_a,
                    optimize=True)
                S2 -= lib.einsum('a,b,ic,dj,ka,jc,ikbd', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, t2_1_a,
                    optimize=True)
                S2 -= lib.einsum('a,a,ib,cj,ic,kd,jkbd', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('a,b,ic,aj,ib,kd,jkcd', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, t2_1_b,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,a,ib,cj,ijdb,kc,kd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,a,ib,cj,kjcb,id,kd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('a,a,ib,cj,kjdb,ic,kd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ic,aj,ijdc,kb,kd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('a,b,ic,aj,kjbc,id,kd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += lib.einsum('a,b,ic,aj,kjdc,ib,kd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('a,b,ic,dj,ijbc,ka,kd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += lib.einsum('a,b,ic,dj,kjbc,ka,id', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('a,b,ic,dj,kjdc,ka,ib', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,a,ib,cj,ijcd,kb,kd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,a,ib,cj,ikcb,jd,kd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,a,ib,cj,ikcd,jb,kd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('a,b,ic,aj,ijbd,kc,kd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('a,b,ic,aj,ikbc,jd,kd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += lib.einsum('a,b,ic,aj,ikbd,jc,kd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,a,ib,cj,kd,jb,ikcd', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, t2_1_a,
                    optimize=True)
                S2 += lib.einsum('a,b,ia,cj,kd,jb,ikcd', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, t2_1_a,
                    optimize=True)
                S2 -= lib.einsum('a,a,ib,cj,ic,kd,jkbd', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('a,b,ia,cj,ic,kd,jkbd', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,b,ic,dj,id,ka,jkbc', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, t2_1_b,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,a,ib,cj,ijdb,kc,kd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,a,ib,cj,kjcb,id,kd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('a,a,ib,cj,kjdb,ic,kd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('a,b,ia,cj,ijdb,kc,kd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('a,b,ia,cj,kjcb,id,kd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += lib.einsum('a,b,ia,cj,kjdb,ic,kd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,a,ib,cj,ijcd,kb,kd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,a,ib,cj,ikcb,jd,kd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,a,ib,cj,ikcd,jb,kd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ia,cj,ijcd,kb,kd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('a,b,ia,cj,ikcb,jd,kd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += lib.einsum('a,b,ia,cj,ikcd,jb,kd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,b,ic,dj,ijdb,ka,kc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += lib.einsum('a,b,ic,dj,ikdb,ka,jc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,b,ic,dj,ikdc,ka,jb', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                # block IblB & AjdJ
                S2 -= 2 * lib.einsum('a,a,ij,bc,ib,jc', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_a, t1_2_b, optimize=True)
                S2 += 2 * lib.einsum('a,b,ij,ac,ib,jc', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_a, t1_2_b, optimize=True)
                S2 -= 2 * lib.einsum('a,a,ij,bc,jc,ib', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_b, t1_2_a, optimize=True)
                S2 += 2 * lib.einsum('a,b,ij,ac,jc,ib', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_b, t1_2_a, optimize=True)
                S2 -= 2 * lib.einsum('a,a,ij,bc,ib,jc', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_a, t1_2_b, optimize=True)
                S2 += 2 * lib.einsum('a,b,ij,cb,ic,ja', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_a, t1_2_b, optimize=True)
                S2 -= 2 * lib.einsum('a,a,ij,bc,jc,ib', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_b, t1_2_a, optimize=True)
                S2 += 2 * lib.einsum('a,b,ij,cb,ja,ic', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_b, t1_2_a, optimize=True)
                S2 -= lib.einsum('a,a,ij,bc,kd,jc,ikbd', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, t2_1_a,
                    optimize=True)
                S2 += lib.einsum('a,b,ij,ac,kd,jc,ikbd', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, t2_1_a,
                    optimize=True)
                S2 -= lib.einsum('a,b,ij,cd,ka,jd,ikbc', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, t2_1_a,
                    optimize=True)
                S2 -= lib.einsum('a,a,ij,bc,ib,kd,jkcd', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('a,b,ij,ac,ib,kd,jkcd', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,a,ij,bc,kjdc,ib,kd', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,bc,kjdc,id,kb', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += lib.einsum('a,b,ij,ac,kjdc,ib,kd', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('a,b,ij,ac,kjdc,kb,id', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,cd,kjad,ib,kc', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += lib.einsum('a,b,ij,cd,kjad,kb,ic', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('a,a,ij,bc,ikbd,jc,kd', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,bc,ikbd,jd,kc', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += lib.einsum('a,b,ij,ac,ikbd,jc,kd', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,ac,ikbd,jd,kc', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,a,ij,bc,kd,jc,ikbd', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, t2_1_a,
                    optimize=True)
                S2 += lib.einsum('a,b,ij,cb,kd,ja,ikcd', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, t2_1_a,
                    optimize=True)
                S2 -= lib.einsum('a,a,ij,bc,ib,kd,jkcd', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('a,b,ij,cb,ic,kd,jkad', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,b,ij,cd,ic,kb,jkad', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,a,ij,bc,kjdc,ib,kd', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,bc,kjdc,id,kb', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += lib.einsum('a,b,ij,cb,kjda,ic,kd', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,cb,kjda,id,kc', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('a,a,ij,bc,ikbd,jc,kd', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ij,bc,ikbd,jd,kc', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += lib.einsum('a,b,ij,cb,ikcd,ja,kd', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,b,ij,cb,ikcd,ka,jd', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ij,cd,ikcb,ja,kd', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += lib.einsum('a,b,ij,cd,ikcb,ka,jd', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                # block IbdJ
                S2 -= 2 * lib.einsum('a,a,ib,ic,jb,jc', L_a, R_a, t1_1_b, t1_2_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 += lib.einsum('a,b,ic,jc,ia,jb', L_b, R_b, t1_1_a, t1_2_a, S_ov_ab, S_ov_ab, optimize=True)
                S2 += lib.einsum('a,b,ic,jc,ja,ib', L_b, R_b, t1_1_a, t1_2_a, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,a,ib,ic,jb,jc', L_b, R_b, t1_1_b, t1_2_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ia,ic,jb,jc', L_b, R_b, t1_1_b, t1_2_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ib,ic,ja,jc', L_b, R_b, t1_1_b, t1_2_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ic,ia,jb,jc', L_b, R_b, t1_1_b, t1_2_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ic,ib,ja,jc', L_b, R_b, t1_1_b, t1_2_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= lib.einsum('a,a,ib,jc,ijbd,kc,kd', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ib,jc,kjbd,ic,kd', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ia,jc,ijbd,kc,kd', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ia,jc,kjbd,ic,kd', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ib,jc,ijad,kc,kd', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ib,jc,kjad,ic,kd', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += lib.einsum('a,a,ijbc,kb,kd,ic,jd', L_a, R_a, t2_1_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,a,ib,jc,ijbd,kc,kd', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ib,jc,kjbd,ic,kd', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ic,ja,ijcd,kb,kd', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ic,ja,kjcd,ib,kd', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ic,jb,ijcd,ka,kd', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ic,jb,kjcd,ia,kd', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 1/6 * lib.einsum('a,b,ic,jd,ijca,kb,kd', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 1/6 * lib.einsum('a,b,ic,jd,ijcb,ka,kd', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ic,jd,kjca,kb,id', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ic,jd,kjcb,ka,id', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ic,jd,kjcd,ia,kb', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ic,jd,kjcd,ka,ib', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ijcd,ia,kb,jc,kd', L_b, R_b, t2_1_a, S_ov_ab, S_ov_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ijcd,ka,ib,jc,kd', L_b, R_b, t2_1_a, S_ov_ab, S_ov_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += lib.einsum('a,a,ijbc,kb,kd,ic,jd', L_b, R_b, t2_1_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 1/6 * lib.einsum('a,b,ijac,kb,kd,ic,jd', L_b, R_b, t2_1_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ijac,kc,kd,ib,jd', L_b, R_b, t2_1_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 1/6 * lib.einsum('a,b,ijbc,ka,kd,ic,jd', L_b, R_b, t2_1_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ijbc,kc,kd,ia,jd', L_b, R_b, t2_1_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ijcd,ka,kc,ib,jd', L_b, R_b, t2_1_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ijcd,kb,kc,ia,jd', L_b, R_b, t2_1_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b,
                    optimize=True)
                # block IbdB & AbdJ
                S2 -= 2 * lib.einsum('a,a,ib,cd,jd,ijcb', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 += 2 * lib.einsum('a,b,ic,ad,jd,ijbc', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 -= lib.einsum('a,b,ia,cb,jd,ijcd', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_2_a, optimize=True)
                S2 -= 2 * lib.einsum('a,a,ib,cd,jd,ijcb', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 -= lib.einsum('a,b,ia,cb,jd,ijcd', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 += 2 * lib.einsum('a,b,ia,cd,jd,ijcb', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 += lib.einsum('a,b,ic,db,ja,ijdc', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,a,ib,cd,ic,jb,jd', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ic,ad,ib,jc,jd', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,ib,cd,ic,jkeb,jked', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ib,cd,ie,jkcb,jked', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ib,cd,jc,ikeb,jked', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,a,ib,cd,je,ikcb,jked', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ic,ad,ib,jkec,jked', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ic,ad,ie,jkbc,jked', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ic,ad,jb,ikec,jked', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ic,ad,je,ikbc,jked', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ic,de,ib,jkae,jkdc', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ic,de,id,jkae,jkbc', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ic,de,jb,jkae,ikdc', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ic,de,jd,jkae,ikbc', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,a,ib,cd,ic,jkbe,jkde', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_b, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('a,b,ic,ad,ib,jkce,jkde', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,ib,cd,jb,ikce,kjed', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ic,ad,jc,ikbe,kjed', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ic,de,jc,ikbd,kjae', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,ib,cd,jb,ikce,jkde', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('a,a,ib,cd,je,ikcb,jkde', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ic,ad,jc,ikbe,jkde', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,b,ic,ad,je,ikbc,jkde', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,b,ia,cb,id,jc,jd', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,ib,cd,ic,jb,jd', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += lib.einsum('a,b,ia,cd,ic,jb,jd', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += lib.einsum('a,b,ic,db,id,ja,jc', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 2/3 * lib.einsum('a,b,ia,cb,id,jkce,jkde', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 2/3 * lib.einsum('a,b,ia,cb,jc,ikde,jkde', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ia,cb,jd,ikce,jkde', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,ib,cd,ic,jkeb,jked', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ib,cd,ie,jkcb,jked', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ib,cd,jc,ikeb,jked', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,a,ib,cd,je,ikcb,jked', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,b,ia,cb,id,jkce,jkde', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,b,ia,cb,jc,ikde,jkde', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ia,cb,jd,ikce,jkde', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ia,cd,ic,jkeb,jked', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ia,cd,ie,jkcb,jked', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ia,cd,jc,ikeb,jked', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ia,cd,je,ikcb,jked', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ic,db,id,jkea,jkec', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('a,b,ic,db,ie,jkea,jkdc', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('a,b,ic,db,jd,jkea,ikec', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ic,db,je,jkea,ikdc', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,a,ib,cd,ic,jkbe,jkde', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_b, t2_1_b,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ia,cd,ic,jkbe,jkde', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_b, t2_1_b,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ic,db,id,jkae,jkce', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_b, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('a,b,ic,de,id,jkae,jkbc', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,ib,cd,jb,ikce,kjed', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ia,cb,jd,ikce,kjed', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ia,cd,jb,ikce,kjed', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,b,ic,db,jc,ikde,kjea', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,ib,cd,jb,ikce,jkde', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('a,a,ib,cd,je,ikcb,jkde', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ia,cb,jd,ikce,jkde', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('a,b,ia,cd,jb,ikce,jkde', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,b,ia,cd,je,ikcb,jkde', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,b,ic,db,jc,ikde,jkae', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ic,db,je,ikdc,jkae', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('a,b,ic,de,jb,ikdc,jkae', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ic,de,jc,ikdb,jkae', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                # block AjlB
                S2 -= 2 * lib.einsum('a,a,ib,ic,bj,cj', L_a, R_a, t1_1_a, t1_2_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ia,ic,bj,cj', L_a, R_a, t1_1_a, t1_2_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ib,ic,aj,cj', L_a, R_a, t1_1_a, t1_2_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ic,ia,bj,cj', L_a, R_a, t1_1_a, t1_2_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ic,ib,aj,cj', L_a, R_a, t1_1_a, t1_2_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 += lib.einsum('a,b,ic,jc,ai,bj', L_a, R_a, t1_1_b, t1_2_b, S_vo_ab, S_vo_ab, optimize=True)
                S2 += lib.einsum('a,b,ic,jc,aj,bi', L_a, R_a, t1_1_b, t1_2_b, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,a,ib,ic,bj,cj', L_b, R_b, t1_1_a, t1_2_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= lib.einsum('a,a,ib,jc,ijdc,bk,dk', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ib,jc,ikdc,bj,dk', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ia,jc,ijdc,bk,dk', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ia,jc,ikdc,bj,dk', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ib,jc,ijdc,ak,dk', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ib,jc,ikdc,aj,dk', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 1/6 * lib.einsum('a,b,ic,jd,ijad,bk,ck', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 1/6 * lib.einsum('a,b,ic,jd,ijbd,ak,ck', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ic,jd,ikad,bk,cj', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ic,jd,ikbd,ak,cj', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ic,jd,ikcd,aj,bk', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ic,jd,ikcd,ak,bj', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += lib.einsum('a,a,ijbc,bk,dk,ic,jd', L_a, R_a, t2_1_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 1/6 * lib.einsum('a,b,ijac,bk,dk,ic,jd', L_a, R_a, t2_1_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ijac,ck,dk,ib,jd', L_a, R_a, t2_1_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 1/6 * lib.einsum('a,b,ijbc,ak,dk,ic,jd', L_a, R_a, t2_1_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ijbc,ck,dk,ia,jd', L_a, R_a, t2_1_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ijcd,ak,ck,ib,jd', L_a, R_a, t2_1_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ijcd,bk,ck,ia,jd', L_a, R_a, t2_1_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ijcd,ai,bk,jc,kd', L_a, R_a, t2_1_b, S_vo_ab, S_vo_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ijcd,ak,bi,jc,kd', L_a, R_a, t2_1_b, S_vo_ab, S_vo_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,a,ib,jc,ijdc,bk,dk', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,ib,jc,ikdc,bj,dk', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ic,ja,ijdb,ck,dk', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ic,ja,ikdb,cj,dk', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ic,jb,ijda,ck,dk', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ic,jb,ikda,cj,dk', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += lib.einsum('a,a,ijbc,bk,dk,ic,jd', L_b, R_b, t2_1_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                # block AjdB & AblB
                S2 -= 2 * lib.einsum('a,a,bi,cd,jc,jibd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 -= lib.einsum('a,b,ai,bc,jd,jidc', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 += 2 * lib.einsum('a,b,ai,cd,jc,jibd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 += lib.einsum('a,b,ci,bd,ja,jicd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 -= lib.einsum('a,b,ai,bc,jd,ijcd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_2_b, optimize=True)
                S2 -= 2 * lib.einsum('a,a,bi,cd,jc,jibd', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 += 2 * lib.einsum('a,b,ci,da,jd,jicb', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,a,bi,cd,jb,jkce,kied', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,a,bi,cd,je,jkce,kibd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ai,bc,jd,jkde,kiec', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ai,cd,jb,jkce,kied', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ai,cd,je,jkce,kibd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,b,ci,bd,jc,jkae,kied', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ci,bd,je,jkae,kicd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ci,de,jb,jkad,kice', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ci,de,jc,jkad,kibe', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,bi,cd,jb,jkce,ikde', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ai,bc,jd,jkde,ikce', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('a,b,ai,cd,jb,jkce,ikde', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,b,ci,bd,jc,jkae,ikde', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,bi,cd,id,jb,jc', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += lib.einsum('a,b,ai,cd,id,jb,jc', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += lib.einsum('a,b,ci,bd,id,ja,jc', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,b,ai,bc,id,jc,jd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,a,bi,cd,id,jkbe,jkce', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ai,cd,id,jkbe,jkce', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ci,bd,id,jkae,jkce', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += lib.einsum('a,b,ci,de,ie,jkad,jkbc', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,bi,cd,id,jkbe,jkce', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,bi,cd,ie,jkbd,jkce', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,bi,cd,jd,kibe,kjce', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,a,bi,cd,je,kibd,kjce', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,b,ai,bc,id,jkec,jked', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 4/3 * lib.einsum('a,b,ai,bc,jc,kide,kjde', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ai,bc,jd,kiec,kjed', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ai,cd,id,jkbe,jkce', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ai,cd,ie,jkbd,jkce', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ai,cd,jd,kibe,kjce', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ai,cd,je,kibd,kjce', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ci,bd,id,jkae,jkce', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('a,b,ci,bd,ie,jkae,jkcd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('a,b,ci,bd,jd,kjae,kice', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('a,b,ci,bd,je,kjae,kicd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2/3 * lib.einsum('a,b,ai,bc,id,jkce,jkde', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 += 2/3 * lib.einsum('a,b,ai,bc,jc,ikde,jkde', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('a,b,ai,bc,jd,ikce,jkde', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,bi,cd,jb,jkce,kied', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,a,bi,cd,je,jkce,kibd', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ci,da,jc,jkde,kieb', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ci,da,je,jkde,kicb', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,bi,cd,jb,jkce,ikde', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ci,da,jc,jkde,ikbe', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ci,de,jc,jkda,ikbe', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,bi,cd,id,jb,jc', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ci,da,ib,jc,jd', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('a,a,bi,cd,id,jkbe,jkce', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += lib.einsum('a,b,ci,da,ib,jkce,jkde', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,bi,cd,id,jkbe,jkce', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,bi,cd,ie,jkbd,jkce', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,a,bi,cd,jd,kibe,kjce', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,a,bi,cd,je,kibd,kjce', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ci,da,ib,jkce,jkde', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ci,da,ie,jkcb,jkde', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,b,ci,da,jb,kice,kjde', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ci,da,je,kicb,kjde', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ci,de,ib,jkda,jkce', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,b,ci,de,ie,jkda,jkcb', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ci,de,jb,kjda,kice', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,b,ci,de,je,kjda,kicb', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                # block AbdB
                S2 -= lib.einsum('a,b,ic,id,ac,bd', L_a, R_a, t1_1_b, t1_2_b, S_vv_ab, S_vv_ab, optimize=True)
                S2 -= lib.einsum('a,b,ic,id,ad,bc', L_a, R_a, t1_1_b, t1_2_b, S_vv_ab, S_vv_ab, optimize=True)
                S2 -= lib.einsum('a,b,ic,id,ca,db', L_b, R_b, t1_1_a, t1_2_a, S_vv_ab, S_vv_ab, optimize=True)
                S2 -= lib.einsum('a,b,ic,id,da,cb', L_b, R_b, t1_1_a, t1_2_a, S_vv_ab, S_vv_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,a,ib,jc,ijde,be,dc', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ia,jc,ijde,be,dc', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ib,jc,ijde,ae,dc', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ic,jd,ijae,bd,ce', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ic,jd,ijbe,ad,ce', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ic,jd,ijce,ad,be', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ic,jd,ijce,ae,bd', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ijcd,ac,be,id,je', L_a, R_a, t2_1_b, S_vv_ab, S_vv_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ijcd,ae,bc,id,je', L_a, R_a, t2_1_b, S_vv_ab, S_vv_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,a,ib,jc,ijde,be,dc', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ic,ja,ijde,db,ce', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ic,jb,ijde,da,ce', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ic,jd,ijea,cb,ed', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 += lib.einsum('a,b,ic,jd,ijeb,ca,ed', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ic,jd,ijed,ca,eb', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,b,ic,jd,ijed,ea,cb', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ijcd,ca,eb,id,je', L_b, R_b, t2_1_a, S_vv_ab, S_vv_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,b,ijcd,ea,cb,id,je', L_b, R_b, t2_1_a, S_vv_ab, S_vv_ab, t1_1_a, t1_1_a,
                    optimize=True)

            # block AbdB
            S2 -= 2 * lib.einsum('a,a,ijbc,ijde,be,dc', L_a, R_a, t2_1_ab, t2_2_ab, S_vv_ab, S_vv_ab, optimize=True)
            S2 += lib.einsum('a,b,ijac,ijde,be,dc', L_a, R_a, t2_1_ab, t2_2_ab, S_vv_ab, S_vv_ab, optimize=True)
            S2 += lib.einsum('a,b,ijbc,ijde,ae,dc', L_a, R_a, t2_1_ab, t2_2_ab, S_vv_ab, S_vv_ab, optimize=True)
            S2 += lib.einsum('a,b,ijcd,ijae,bd,ce', L_a, R_a, t2_1_ab, t2_2_ab, S_vv_ab, S_vv_ab, optimize=True)
            S2 += lib.einsum('a,b,ijcd,ijbe,ad,ce', L_a, R_a, t2_1_ab, t2_2_ab, S_vv_ab, S_vv_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijcd,ijce,ad,be', L_a, R_a, t2_1_ab, t2_2_ab, S_vv_ab, S_vv_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijcd,ijce,ae,bd', L_a, R_a, t2_1_ab, t2_2_ab, S_vv_ab, S_vv_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,b,ijcd,ijce,ad,be', L_a, R_a, t2_1_b, t2_2_b, S_vv_ab, S_vv_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,b,ijcd,ijce,ae,bd', L_a, R_a, t2_1_b, t2_2_b, S_vv_ab, S_vv_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,b,ijcd,ijce,da,eb', L_b, R_b, t2_1_a, t2_2_a, S_vv_ab, S_vv_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,b,ijcd,ijce,ea,db', L_b, R_b, t2_1_a, t2_2_a, S_vv_ab, S_vv_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ijbc,ijde,be,dc', L_b, R_b, t2_1_ab, t2_2_ab, S_vv_ab, S_vv_ab, optimize=True)
            S2 += lib.einsum('a,b,ijca,ijde,db,ce', L_b, R_b, t2_1_ab, t2_2_ab, S_vv_ab, S_vv_ab, optimize=True)
            S2 += lib.einsum('a,b,ijcb,ijde,da,ce', L_b, R_b, t2_1_ab, t2_2_ab, S_vv_ab, S_vv_ab, optimize=True)
            S2 += lib.einsum('a,b,ijcd,ijea,cb,ed', L_b, R_b, t2_1_ab, t2_2_ab, S_vv_ab, S_vv_ab, optimize=True)
            S2 += lib.einsum('a,b,ijcd,ijeb,ca,ed', L_b, R_b, t2_1_ab, t2_2_ab, S_vv_ab, S_vv_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijcd,ijed,ca,eb', L_b, R_b, t2_1_ab, t2_2_ab, S_vv_ab, S_vv_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijcd,ijed,ea,cb', L_b, R_b, t2_1_ab, t2_2_ab, S_vv_ab, S_vv_ab, optimize=True)
            # block AjlB
            S2 -= lib.einsum('a,a,ijbc,ijbd,ck,dk', L_a, R_a, t2_1_a, t2_2_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,b,ijac,ijbd,ck,dk', L_a, R_a, t2_1_a, t2_2_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= 1 / 4 * lib.einsum('a,b,ijac,ijcd,bk,dk', L_a, R_a, t2_1_a, t2_2_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,b,ijbc,ijad,ck,dk', L_a, R_a, t2_1_a, t2_2_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= 1 / 4 * lib.einsum('a,b,ijbc,ijcd,ak,dk', L_a, R_a, t2_1_a, t2_2_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= 1 / 4 * lib.einsum('a,b,ijcd,ijac,bk,dk', L_a, R_a, t2_1_a, t2_2_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= 1 / 4 * lib.einsum('a,b,ijcd,ijbc,ak,dk', L_a, R_a, t2_1_a, t2_2_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ijbc,ijdc,bk,dk', L_a, R_a, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 += 2 * lib.einsum('a,a,ijbc,ikdc,bj,dk', L_a, R_a, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,b,ijac,ijdc,bk,dk', L_a, R_a, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijac,ikdc,bj,dk', L_a, R_a, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,b,ijbc,ijdc,ak,dk', L_a, R_a, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijbc,ikdc,aj,dk', L_a, R_a, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,b,ijcd,ijad,bk,ck', L_a, R_a, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,b,ijcd,ijbd,ak,ck', L_a, R_a, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijcd,ikad,bk,cj', L_a, R_a, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijcd,ikbd,ak,cj', L_a, R_a, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 += lib.einsum('a,b,ijcd,ikcd,aj,bk', L_a, R_a, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 += lib.einsum('a,b,ijcd,ikcd,ak,bj', L_a, R_a, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,b,ijcd,ikcd,aj,bk', L_a, R_a, t2_1_b, t2_2_b, S_vo_ab, S_vo_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,b,ijcd,ikcd,ak,bj', L_a, R_a, t2_1_b, t2_2_b, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= lib.einsum('a,a,ijbc,ijbd,ck,dk', L_b, R_b, t2_1_a, t2_2_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ijbc,ijdc,bk,dk', L_b, R_b, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 += 2 * lib.einsum('a,a,ijbc,ikdc,bj,dk', L_b, R_b, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 += lib.einsum('a,b,ijca,ijdb,ck,dk', L_b, R_b, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijca,ikdb,cj,dk', L_b, R_b, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 += lib.einsum('a,b,ijcb,ijda,ck,dk', L_b, R_b, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijcb,ikda,cj,dk', L_b, R_b, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            # block IbdJ
            S2 -= 2 * lib.einsum('a,a,ijbc,ijbd,kc,kd', L_a, R_a, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 += 2 * lib.einsum('a,a,ijbc,kjbd,ic,kd', L_a, R_a, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 += lib.einsum('a,b,ijac,ijbd,kc,kd', L_a, R_a, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijac,kjbd,ic,kd', L_a, R_a, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 += lib.einsum('a,b,ijbc,ijad,kc,kd', L_a, R_a, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijbc,kjad,ic,kd', L_a, R_a, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= lib.einsum('a,a,ijbc,ijbd,kc,kd', L_a, R_a, t2_1_b, t2_2_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,b,ijcd,ikcd,ja,kb', L_b, R_b, t2_1_a, t2_2_a, S_ov_ab, S_ov_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,b,ijcd,ikcd,ka,jb', L_b, R_b, t2_1_a, t2_2_a, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ijbc,ijbd,kc,kd', L_b, R_b, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 += 2 * lib.einsum('a,a,ijbc,kjbd,ic,kd', L_b, R_b, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,b,ijca,ijcd,kb,kd', L_b, R_b, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijca,kjcd,ib,kd', L_b, R_b, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,b,ijcb,ijcd,ka,kd', L_b, R_b, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijcb,kjcd,ia,kd', L_b, R_b, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,b,ijcd,ijca,kb,kd', L_b, R_b, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,b,ijcd,ijcb,ka,kd', L_b, R_b, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijcd,kjca,kb,id', L_b, R_b, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijcd,kjcb,ka,id', L_b, R_b, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 += lib.einsum('a,b,ijcd,kjcd,ia,kb', L_b, R_b, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 += lib.einsum('a,b,ijcd,kjcd,ka,ib', L_b, R_b, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= lib.einsum('a,a,ijbc,ijbd,kc,kd', L_b, R_b, t2_1_b, t2_2_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,b,ijac,ijbd,kc,kd', L_b, R_b, t2_1_b, t2_2_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= 1 / 4 * lib.einsum('a,b,ijac,ijcd,kb,kd', L_b, R_b, t2_1_b, t2_2_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,b,ijbc,ijad,kc,kd', L_b, R_b, t2_1_b, t2_2_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= 1 / 4 * lib.einsum('a,b,ijbc,ijcd,ka,kd', L_b, R_b, t2_1_b, t2_2_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= 1 / 4 * lib.einsum('a,b,ijcd,ijac,kb,kd', L_b, R_b, t2_1_b, t2_2_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= 1 / 4 * lib.einsum('a,b,ijcd,ijbc,ka,kd', L_b, R_b, t2_1_b, t2_2_b, S_ov_ab, S_ov_ab, optimize=True)
            # block IblB & AjdJ
            S2 -= 2 * lib.einsum('a,a,ij,bc,ikbd,kjdc', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_a, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('a,b,ij,ac,ikbd,kjdc', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_a, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,b,ij,cd,ikbc,kjad', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_a, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ij,bc,kjdc,ikbd', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t2_2_a, optimize=True)
            S2 += 2 * lib.einsum('a,b,ij,ac,kjdc,ikbd', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t2_2_a, optimize=True)
            S2 -= 2 * lib.einsum('a,b,ij,cd,kjad,ikbc', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t2_2_a, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ij,bc,ikbd,jkcd', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t2_2_b, optimize=True)
            S2 += 2 * lib.einsum('a,b,ij,ac,ikbd,jkcd', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t2_2_b, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ij,bc,jkcd,ikbd', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_b, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('a,b,ij,ac,jkcd,ikbd', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_b, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ij,bc,ikbd,kjdc', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_a, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('a,b,ij,cb,ikcd,kjda', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_a, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ij,bc,kjdc,ikbd', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t2_2_a, optimize=True)
            S2 += 2 * lib.einsum('a,b,ij,cb,kjda,ikcd', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t2_2_a, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ij,bc,ikbd,jkcd', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t2_2_b, optimize=True)
            S2 += 2 * lib.einsum('a,b,ij,cb,ikcd,jkad', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t2_2_b, optimize=True)
            S2 -= 2 * lib.einsum('a,b,ij,cd,ikcb,jkad', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t2_2_b, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ij,bc,jkcd,ikbd', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_b, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('a,b,ij,cb,jkad,ikcd', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_b, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,b,ij,cd,jkad,ikcb', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_b, t2_2_ab, optimize=True)
            # block IjdB & AblJ
            S2 -= 4 / 3 * lib.einsum('a,a,ib,cj,ikcd,klde,jlbe', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_a, t2_1_ab,
                t2_1_b, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ic,aj,ikbd,klde,jlce', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_a, t2_1_ab,
                t2_1_b, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,b,ic,dj,ikbd,klae,jlce', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_a, t2_1_ab,
                t2_1_b, optimize=True)
            S2 += 2 / 3 * lib.einsum('a,a,ib,cj,ijdb,klce,klde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 2 / 3 * lib.einsum('a,a,ib,cj,kjcb,ilde,klde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,a,ib,cj,kjdb,ilce,klde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 1 / 6 * lib.einsum('a,b,ic,aj,ijdc,klbe,klde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 2 / 3 * lib.einsum('a,b,ic,aj,kjbc,ilde,klde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ic,aj,kjdc,ilbe,klde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,b,ic,dj,ijbc,klae,klde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 2 / 3 * lib.einsum('a,b,ic,dj,ijec,klae,klbd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ic,dj,kjbc,klae,ilde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,b,ic,dj,kjdc,klae,ilbe', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ic,dj,kjec,klae,ilbd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,a,ib,cj,ijcd,kleb,kled', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,a,ib,cj,ijdb,klce,klde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,a,ib,cj,ijkl,klcb', L_a, R_a, S_ov_ab, S_vo_ab, w_t2t2_ijkl,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,a,ib,cj,ikcb,ljde,lkde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,a,ib,cj,ikcd,ljeb,lked', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,a,ib,cj,ikdb,lkde,ljce', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,a,ib,cj,ikde,ljcb,lkde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,b,ic,aj,ijbd,klec,kled', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,b,ic,aj,ikbc,ljde,lkde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ic,aj,ikbd,ljec,lked', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,b,ic,aj,kjbc,ilde,klde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ic,aj,kjbd,ilec,kled', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ic,aj,klbc,ijkl', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab,
                w_t2t2_ijkl, optimize=True)
            S2 -= 1 / 3 * lib.einsum('a,b,ic,aj,klbd,ijec,kled', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,b,ic,dj,klae,ijbc,klde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ic,dj,klae,ijbe,kldc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ic,dj,klae,ilbc,kjde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,b,ic,dj,klae,ilbe,kjdc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ic,dj,klae,kjbc,ilde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,b,ic,dj,klae,kjbe,ildc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,b,ic,dj,klae,klbc,ijde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 2 / 3 * lib.einsum('a,a,ib,cj,ijcd,klbe,klde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += 2 / 3 * lib.einsum('a,a,ib,cj,ikcb,jlde,klde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,a,ib,cj,ikcd,klde,jlbe', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 2 / 3 * lib.einsum('a,b,ic,aj,ijbd,klce,klde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 2 / 3 * lib.einsum('a,b,ic,aj,ikbc,jlde,klde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ic,aj,ikbd,jlce,klde', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,a,ib,cj,ikcd,klde,jlbe', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_a, t2_1_ab,
                t2_1_b, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ia,cj,ikcd,klde,jlbe', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_a, t2_1_ab,
                t2_1_b, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,b,ic,dj,ikde,klea,jlbc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_a, t2_1_ab,
                t2_1_b, optimize=True)
            S2 += 2 / 3 * lib.einsum('a,a,ib,cj,ijdb,klce,klde', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 2 / 3 * lib.einsum('a,a,ib,cj,kjcb,ilde,klde', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,a,ib,cj,kjdb,ilce,klde', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 2 / 3 * lib.einsum('a,b,ia,cj,ijdb,klce,klde', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 2 / 3 * lib.einsum('a,b,ia,cj,kjcb,ilde,klde', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ia,cj,kjdb,ilce,klde', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,a,ib,cj,ijcd,kleb,kled', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,a,ib,cj,ijdb,klce,klde', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,a,ib,cj,ijkl,klcb', L_b, R_b, S_ov_ab, S_vo_ab, w_t2t2_ijkl,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,a,ib,cj,ikcb,ljde,lkde', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,a,ib,cj,ikcd,ljeb,lked', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,a,ib,cj,ikdb,lkde,ljce', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,a,ib,cj,ikde,ljcb,lkde', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,b,ia,cj,ijdb,klce,klde', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,b,ia,cj,ikcb,ljde,lkde', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ia,cj,ikdb,lkde,ljce', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,b,ia,cj,kjcb,ilde,klde', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ia,cj,kjdb,ilce,klde', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ia,cj,klcb,ijkl', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab,
                w_t2t2_ijkl, optimize=True)
            S2 -= 1 / 3 * lib.einsum('a,b,ia,cj,kldb,ijce,klde', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,b,ic,dj,klea,ijdb,klec', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ic,dj,klea,ijeb,kldc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ic,dj,klea,ildb,kjec', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,b,ic,dj,klea,ileb,kjdc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ic,dj,klea,kjdb,ilec', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,b,ic,dj,klea,kjeb,ildc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,b,ic,dj,klea,kldb,ijec', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 2 / 3 * lib.einsum('a,a,ib,cj,ijcd,klbe,klde', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += 2 / 3 * lib.einsum('a,a,ib,cj,ikcb,jlde,klde', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,a,ib,cj,ikcd,klde,jlbe', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 1 / 6 * lib.einsum('a,b,ia,cj,ijcd,klbe,klde', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 2 / 3 * lib.einsum('a,b,ia,cj,ikcb,jlde,klde', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ia,cj,ikcd,klde,jlbe', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,b,ic,dj,ijdb,klae,klce', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 2 / 3 * lib.einsum('a,b,ic,dj,ijde,klae,klbc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ic,dj,ikdb,klae,jlce', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 4 / 3 * lib.einsum('a,b,ic,dj,ikdc,klae,jlbe', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += 4 / 3 * lib.einsum('a,b,ic,dj,ikde,klae,jlbc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            # block IjlB & AjlJ
            S2 -= lib.einsum('a,a,ij,bj,kc,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_2_a, t2_1_a, optimize=True)
            S2 += lib.einsum('a,b,ij,aj,kc,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_2_a, t2_1_a, optimize=True)
            S2 -= lib.einsum('a,b,ij,cj,ka,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_2_a, t2_1_a, optimize=True)
            S2 -= lib.einsum('a,a,ij,bj,kc,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,a,ij,bk,jc,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,b,ij,aj,kc,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,b,ij,ak,jc,ikbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,a,ij,bj,kc,ikbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_2_a, t2_1_a, optimize=True)
            S2 -= lib.einsum('a,a,ij,bj,kc,ikbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,a,ij,bk,jc,ikbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,b,ij,cj,ka,ikcb', L_b, R_b, S_oo_ab, S_vo_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,b,ij,ck,ja,ikcb', L_b, R_b, S_oo_ab, S_vo_ab, t1_2_b, t2_1_ab, optimize=True)
            # block IjdJ & IblJ
            S2 -= lib.einsum('a,a,ij,ib,kc,kjcb', L_a, R_a, S_oo_ab, S_ov_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,a,ij,kb,ic,kjcb', L_a, R_a, S_oo_ab, S_ov_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,b,ij,ic,ka,kjbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,b,ij,kc,ia,kjbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,a,ij,ib,kc,jkbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_2_b, t2_1_b, optimize=True)
            S2 -= lib.einsum('a,a,ij,ib,kc,kjcb', L_b, R_b, S_oo_ab, S_ov_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,a,ij,kb,ic,kjcb', L_b, R_b, S_oo_ab, S_ov_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,b,ij,ia,kc,kjcb', L_b, R_b, S_oo_ab, S_ov_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,b,ij,ka,ic,kjcb', L_b, R_b, S_oo_ab, S_ov_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,a,ij,ib,kc,jkbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_2_b, t2_1_b, optimize=True)
            S2 += lib.einsum('a,b,ij,ia,kc,jkbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_2_b, t2_1_b, optimize=True)
            S2 -= lib.einsum('a,b,ij,ic,ka,jkbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_2_b, t2_1_b, optimize=True)
            # block IbdB & AbdJ
            S2 -= 2 * lib.einsum('a,a,ib,cd,jd,ijcb', L_a, R_a, S_ov_ab, S_vv_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,b,ic,ad,jd,ijbc', L_a, R_a, S_ov_ab, S_vv_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,b,ia,cb,jd,ijcd', L_b, R_b, S_ov_ab, S_vv_ab, t1_2_a, t2_1_a, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ib,cd,jd,ijcb', L_b, R_b, S_ov_ab, S_vv_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,b,ia,cb,jd,ijcd', L_b, R_b, S_ov_ab, S_vv_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,b,ia,cd,jd,ijcb', L_b, R_b, S_ov_ab, S_vv_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,b,ic,db,ja,ijdc', L_b, R_b, S_ov_ab, S_vv_ab, t1_2_b, t2_1_ab, optimize=True)
            # block AjdB & AblB
            S2 -= 2 * lib.einsum('a,a,bi,cd,jc,jibd', L_a, R_a, S_vo_ab, S_vv_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,b,ai,bc,jd,jidc', L_a, R_a, S_vo_ab, S_vv_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,b,ai,cd,jc,jibd', L_a, R_a, S_vo_ab, S_vv_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,b,ci,bd,ja,jicd', L_a, R_a, S_vo_ab, S_vv_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,b,ai,bc,jd,ijcd', L_a, R_a, S_vo_ab, S_vv_ab, t1_2_b, t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('a,a,bi,cd,jc,jibd', L_b, R_b, S_vo_ab, S_vv_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,b,ci,da,jd,jicb', L_b, R_b, S_vo_ab, S_vv_ab, t1_2_a, t2_1_ab, optimize=True)
            # block IjlJ
            S2 += lib.einsum('a,a,ijbc,ikbc,jl,kl', L_a, R_a, t2_1_a, t2_2_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijac,ikbc,jl,kl', L_a, R_a, t2_1_a, t2_2_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijbc,ikac,jl,kl', L_a, R_a, t2_1_a, t2_2_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 2 * lib.einsum('a,a,ijbc,ikbc,lj,lk', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 2 * lib.einsum('a,a,ijbc,kjbc,il,kl', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ijbc,klbc,il,kj', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijac,ikbc,lj,lk', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijac,kjbc,il,kl', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('a,b,ijac,klbc,il,kj', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijbc,ikac,lj,lk', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijbc,kjac,il,kl', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('a,b,ijbc,klac,il,kj', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('a,a,ijbc,ikbc,lj,lk', L_a, R_a, t2_1_b, t2_2_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('a,a,ijbc,ikbc,jl,kl', L_b, R_b, t2_1_a, t2_2_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 2 * lib.einsum('a,a,ijbc,ikbc,lj,lk', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 2 * lib.einsum('a,a,ijbc,kjbc,il,kl', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,a,ijbc,klbc,il,kj', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijca,ikcb,lj,lk', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijca,kjcb,il,kl', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('a,b,ijca,klcb,il,kj', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijcb,ikca,lj,lk', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijcb,kjca,il,kl', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('a,b,ijcb,klca,il,kj', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('a,a,ijbc,ikbc,lj,lk', L_b, R_b, t2_1_b, t2_2_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijac,ikbc,lj,lk', L_b, R_b, t2_1_b, t2_2_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('a,b,ijbc,ikac,lj,lk', L_b, R_b, t2_1_b, t2_2_b, S_oo_ab, S_oo_ab, optimize=True)

            if adc.approx_trans_moments is False:
                # block IjlB & AjlJ
                S2 -= 2 * lib.einsum('a,a,ij,bj,ib', L_a, R_a, S_oo_ab, S_vo_ab, t1_3_a, optimize=True)
                S2 += 2 * lib.einsum('a,b,ij,aj,ib', L_a, R_a, S_oo_ab, S_vo_ab, t1_3_a, optimize=True)
                S2 -= 2 * lib.einsum('a,a,ij,bj,ib', L_b, R_b, S_oo_ab, S_vo_ab, t1_3_a, optimize=True)
                # block IjdJ & IblJ
                S2 -= 2 * lib.einsum('a,a,ij,ib,jb', L_a, R_a, S_oo_ab, S_ov_ab, t1_3_b, optimize=True)
                S2 -= 2 * lib.einsum('a,a,ij,ib,jb', L_b, R_b, S_oo_ab, S_ov_ab, t1_3_b, optimize=True)
                S2 += 2 * lib.einsum('a,b,ij,ia,jb', L_b, R_b, S_oo_ab, S_ov_ab, t1_3_b, optimize=True)
                # block IbdB & AbdJ
                S2 -= 2 * lib.einsum('a,b,ia,cb,ic', L_b, R_b, S_ov_ab, S_vv_ab, t1_3_a, optimize=True)
                # block AjdB & AblB
                S2 -= 2 * lib.einsum('a,b,ai,bc,ic', L_a, R_a, S_vo_ab, S_vv_ab, t1_3_b, optimize=True)

            # 120 & 021
            # block AbdB
            S2 -= 2 * lib.einsum('a,ibc,id,ac,bd', L_a, R_bab, t1_2_b, S_vv_ab, S_vv_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,id,ca,db', L_b, R_aba, t1_2_a, S_vv_ab, S_vv_ab, optimize=True)
            # block AblJ
            S2 -= lib.einsum('a,iab,ic,bj,jc', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_2_b, optimize=True)
            S2 += lib.einsum('a,iba,ic,bj,jc', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_2_b, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,jb,ci,jc', L_a, R_bab, S_ov_ab, S_vo_ab, t1_2_a, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,jc,bi,ja', L_a, R_bab, S_ov_ab, S_vo_ab, t1_2_a, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,ic,bj,jc', L_b, R_aba, S_ov_ab, S_vo_ab, t1_2_b, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,ib,cj,ja', L_b, R_aba, S_ov_ab, S_vo_ab, t1_2_b, optimize=True)
            S2 -= lib.einsum('a,iab,jb,ci,jc', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_2_a, optimize=True)
            S2 += lib.einsum('a,iba,jb,ci,jc', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_2_a, optimize=True)
            # block AjdJ
            S2 -= lib.einsum('a,iab,ij,bc,jc', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_2_b, optimize=True)
            S2 += lib.einsum('a,iba,ij,bc,jc', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_2_b, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,ij,bc,jc', L_b, R_aba, S_oo_ab, S_vv_ab, t1_2_b, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,ij,ca,jb', L_b, R_aba, S_oo_ab, S_vv_ab, t1_2_b, optimize=True)
            # block AjlB
            S2 -= lib.einsum('a,iab,ic,bj,cj', L_a, R_aaa_u, t1_2_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 += lib.einsum('a,iba,ic,bj,cj', L_a, R_aaa_u, t1_2_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,jc,aj,bi', L_a, R_bab, t1_2_b, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,ic,bj,cj', L_b, R_aba, t1_2_a, S_vo_ab, S_vo_ab, optimize=True)
            # block IbdJ
            S2 -= 2 * lib.einsum('a,iab,ic,jb,jc', L_a, R_bab, t1_2_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,jc,ja,ib', L_b, R_aba, t1_2_a, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= lib.einsum('a,iab,ic,jb,jc', L_b, R_bbb_u, t1_2_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 += lib.einsum('a,iba,ic,jb,jc', L_b, R_bbb_u, t1_2_b, S_ov_ab, S_ov_ab, optimize=True)
            # block IblB
            S2 -= 2 * lib.einsum('a,iab,ji,cb,jc', L_a, R_bab, S_oo_ab, S_vv_ab, t1_2_a, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,ji,ac,jb', L_a, R_bab, S_oo_ab, S_vv_ab, t1_2_a, optimize=True)
            S2 -= lib.einsum('a,iab,ji,cb,jc', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_2_a, optimize=True)
            S2 += lib.einsum('a,iba,ji,cb,jc', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_2_a, optimize=True)
            # block IjlJ
            S2 += lib.einsum('a,iab,jb,ik,jk', L_a, R_aaa_u, t1_2_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('a,iba,jb,ik,jk', L_a, R_aaa_u, t1_2_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 2 * lib.einsum('a,iab,jb,ki,kj', L_a, R_bab, t1_2_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 2 * lib.einsum('a,iab,jb,ik,jk', L_b, R_aba, t1_2_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('a,iab,jb,ki,kj', L_b, R_bbb_u, t1_2_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('a,iba,jb,ki,kj', L_b, R_bbb_u, t1_2_b, S_oo_ab, S_oo_ab, optimize=True)
            # block IjlB
            S2 -= lib.einsum('a,iab,jk,ck,ijbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_2_a, optimize=True)
            S2 += lib.einsum('a,iba,jk,ck,ijbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_2_a, optimize=True)
            S2 -= lib.einsum('a,ibc,jk,ak,ijbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_2_a, optimize=True)
            S2 += 2 * lib.einsum('a,iab,ji,ck,jkcb', L_a, R_bab, S_oo_ab, S_vo_ab, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,jk,ck,jicb', L_a, R_bab, S_oo_ab, S_vo_ab, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,ji,ak,jkbc', L_a, R_bab, S_oo_ab, S_vo_ab, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,jk,ak,jibc', L_a, R_bab, S_oo_ab, S_vo_ab, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,jk,ck,ijbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_2_a, optimize=True)
            S2 += lib.einsum('a,iab,ji,ck,jkcb', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_2_ab, optimize=True)
            S2 -= lib.einsum('a,iab,jk,ck,jicb', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_2_ab, optimize=True)
            S2 -= lib.einsum('a,iba,ji,ck,jkcb', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_2_ab, optimize=True)
            S2 += lib.einsum('a,iba,jk,ck,jicb', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_2_ab, optimize=True)
            # block IjdJ
            S2 += lib.einsum('a,iab,ij,kc,kjbc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_2_ab, optimize=True)
            S2 -= lib.einsum('a,iab,jk,jc,ikbc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_2_ab, optimize=True)
            S2 -= lib.einsum('a,iba,ij,kc,kjbc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_2_ab, optimize=True)
            S2 += lib.einsum('a,iba,jk,jc,ikbc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,jk,jc,ikbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_2_b, optimize=True)
            S2 += 2 * lib.einsum('a,iab,ij,kc,kjbc', L_b, R_aba, S_oo_ab, S_ov_ab, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,jk,jc,ikbc', L_b, R_aba, S_oo_ab, S_ov_ab, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,ij,ka,kjcb', L_b, R_aba, S_oo_ab, S_ov_ab, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,jk,ja,ikcb', L_b, R_aba, S_oo_ab, S_ov_ab, t2_2_ab, optimize=True)
            S2 -= lib.einsum('a,iab,jk,jc,ikbc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_2_b, optimize=True)
            S2 += lib.einsum('a,iba,jk,jc,ikbc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_2_b, optimize=True)
            S2 -= lib.einsum('a,ibc,jk,ja,ikbc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_2_b, optimize=True)
            # block IblJ
            S2 += lib.einsum('a,iab,jk,ic,jlbd,lkdc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,iab,jk,jc,ilbd,lkdc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_a,
                t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iab,jk,lc,ijbd,lkdc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iba,jk,ic,jlbd,lkdc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,iba,jk,jc,ilbd,lkdc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_a,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iba,jk,lc,ijbd,lkdc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,jk,id,jlbc,lkad', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,ibc,jk,jd,ilbc,lkad', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_a,
                t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,jk,ld,ijbc,lkad', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iab,jk,ic,jlbd,klcd', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,iab,jk,jc,ilbd,klcd', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_ab,
                t2_1_b, optimize=True)
            S2 -= lib.einsum('a,iba,jk,ic,jlbd,klcd', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,iba,jk,jc,ilbd,klcd', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_ab,
                t2_1_b, optimize=True)
            S2 += lib.einsum('a,iab,ji,kb,jlcd,klcd', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_a, t2_1_a, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,ji,kc,klad,jlbd', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_a, t2_1_a, optimize=True)
            S2 += lib.einsum('a,iab,ji,jc,kldb,kldc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,iab,ji,kb,jlcd,klcd', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,ji,kc,jldb,kldc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iab,jk,jb,licd,lkcd', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iab,jk,jc,lidb,lkdc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,jk,lb,jicd,lkcd', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,iab,jk,lc,jidb,lkdc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,ji,jd,klad,klbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,ji,kc,klad,jlbd', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,ji,kd,klad,jlbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,jk,jc,lkad,libd', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,jk,jd,lkad,libc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,jk,lc,lkad,jibd', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,jk,ld,lkad,jibc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,iab,ji,jc,klbd,klcd', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,iab,jk,jb,ilcd,klcd', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= lib.einsum('a,iab,jk,jc,ilbd,klcd', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_b, t2_1_b, optimize=True)
            S2 += 2 * lib.einsum('a,iab,jk,ic,jlbd,lkdc', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iab,jk,jc,ilbd,lkdc', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,iab,jk,lc,ijbd,lkdc', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,jk,ib,jlcd,lkda', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,jk,jb,ilcd,lkda', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,jk,lb,ijcd,lkda', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,iab,jk,ic,jlbd,klcd', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= lib.einsum('a,iab,jk,jc,ilbd,klcd', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,jk,ib,jlcd,klad', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,jk,id,jlcb,klad', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += lib.einsum('a,ibc,jk,jb,ilcd,klad', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= lib.einsum('a,ibc,jk,jd,ilcb,klad', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,iab,ji,kb,jlcd,klcd', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,iba,ji,kb,jlcd,klcd', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,iab,ji,jc,kldb,kldc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iab,ji,kb,jlcd,klcd', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iab,ji,kc,jldb,kldc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,iab,jk,jb,licd,lkcd', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,iab,jk,jc,lidb,lkdc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iab,jk,lb,jicd,lkcd', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iab,jk,lc,jidb,lkdc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,iba,ji,jc,kldb,kldc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iba,ji,kb,jlcd,klcd', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iba,ji,kc,jldb,kldc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,iba,jk,jb,licd,lkcd', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,iba,jk,jc,lidb,lkdc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iba,jk,lb,jicd,lkcd', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iba,jk,lc,jidb,lkdc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,ji,kb,klda,jldc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,ji,kc,klda,jldb', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,ibc,jk,jb,lkda,lidc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,ibc,jk,jc,lkda,lidb', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,jk,lb,lkda,jidc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,jk,lc,lkda,jidb', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 1 / 4 * lib.einsum('a,iab,ji,jc,klbd,klcd', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += 1 / 4 * lib.einsum('a,iab,jk,jb,ilcd,klcd', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,iab,jk,jc,ilbd,klcd', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 1 / 4 * lib.einsum('a,iba,ji,jc,klbd,klcd', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 1 / 4 * lib.einsum('a,iba,jk,jb,ilcd,klcd', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,iba,jk,jc,ilbd,klcd', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 1 / 4 * lib.einsum('a,ibc,ji,jd,klad,klbc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,ibc,jk,jb,klad,ilcd', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,ibc,jk,jc,klad,ilbd', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,ibc,jk,jd,klad,ilbc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b,
                t2_1_b, optimize=True)
            # block IbdB
            S2 -= 2 * lib.einsum('a,iab,jc,db,jidc', L_a, R_bab, S_ov_ab, S_vv_ab, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,jd,ac,jibd', L_a, R_bab, S_ov_ab, S_vv_ab, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,ja,db,ijcd', L_b, R_aba, S_ov_ab, S_vv_ab, t2_2_a, optimize=True)
            S2 -= lib.einsum('a,iab,jc,db,jidc', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_2_ab, optimize=True)
            S2 += lib.einsum('a,iba,jc,db,jidc', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_2_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,ja,db,jidc', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_2_ab, optimize=True)
            S2 += lib.einsum('a,ibc,ja,dc,jidb', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_2_ab, optimize=True)
            # block AjlJ
            S2 += 1 / 4 * lib.einsum('a,iab,ij,cj,klbd,klcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 1 / 4 * lib.einsum('a,iab,jk,bk,ilcd,jlcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,iab,jk,ck,ilbd,jlcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 1 / 4 * lib.einsum('a,iba,ij,cj,klbd,klcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 1 / 4 * lib.einsum('a,iba,jk,bk,ilcd,jlcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,iba,jk,ck,ilbd,jlcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 1 / 4 * lib.einsum('a,ibc,ij,dj,klad,klbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,ibc,jk,bk,jlad,ilcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,ibc,jk,ck,jlad,ilbd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,ibc,jk,dk,jlad,ilbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += lib.einsum('a,iab,ij,bk,ljcd,lkcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,iab,ij,cj,klbd,klcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iab,ij,ck,ljbd,lkcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,iab,jk,bk,ilcd,jlcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iab,jk,bl,ikcd,jlcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,iab,jk,ck,ilbd,jlcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iab,jk,cl,ikbd,jlcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iba,ij,bk,ljcd,lkcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,iba,ij,cj,klbd,klcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iba,ij,ck,ljbd,lkcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,iba,jk,bk,ilcd,jlcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iba,jk,bl,ikcd,jlcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,iba,jk,ck,ilbd,jlcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iba,jk,cl,ikbd,jlcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,ij,bk,lkad,ljcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,ij,ck,lkad,ljbd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,ibc,jk,bk,jlad,ilcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,jk,bl,jlad,ikcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,ibc,jk,ck,jlad,ilbd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,jk,cl,jlad,ikbd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,iab,ij,bk,jlcd,klcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,iba,ij,bk,jlcd,klcd', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += 2 * lib.einsum('a,iab,jk,ci,jlcd,lkdb', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iab,jk,ck,jlcd,lidb', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,jk,bi,jlad,lkdc', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,jk,bk,jlad,lidc', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,jk,di,jlad,lkbc', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,jk,dk,jlad,libc', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,iab,jk,ci,jlcd,klbd', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= lib.einsum('a,iab,jk,ck,jlcd,ilbd', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += 2 * lib.einsum('a,iab,jk,cl,jlcd,ikbd', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,jk,bi,jlad,klcd', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += lib.einsum('a,ibc,jk,bk,jlad,ilcd', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,jk,bl,jlad,ikcd', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,iab,ij,cj,klbd,klcd', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,iab,jk,bk,ilcd,jlcd', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= lib.einsum('a,iab,jk,ck,ilbd,jlcd', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_a, t2_1_a, optimize=True)
            S2 += 2 * lib.einsum('a,iab,ij,bk,ljcd,lkcd', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iab,ij,cj,klbd,klcd', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,ij,ck,ljbd,lkcd', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iab,jk,bk,ilcd,jlcd', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,jk,bl,ikcd,jlcd', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iab,jk,ck,ilbd,jlcd', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,iab,jk,cl,ikbd,jlcd', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,ij,ck,lkda,ljdb', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,ij,dj,klda,klcb', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,ij,dk,lkda,ljcb', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,jk,ck,jlda,ildb', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,jk,cl,jlda,ikdb', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,jk,dk,jlda,ilcb', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,jk,dl,jlda,ikcb', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iab,ij,bk,jlcd,klcd', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_b, t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,ij,ck,klad,jlbd', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_b, t2_1_b, optimize=True)
            S2 += lib.einsum('a,iab,jk,ci,jlcd,lkdb', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,iab,jk,ck,jlcd,lidb', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iba,jk,ci,jlcd,lkdb', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,iba,jk,ck,jlcd,lidb', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iab,jk,ci,jlcd,klbd', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,iab,jk,ck,jlcd,ilbd', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab,
                t2_1_b, optimize=True)
            S2 += lib.einsum('a,iab,jk,cl,jlcd,ikbd', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= lib.einsum('a,iba,jk,ci,jlcd,klbd', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,iba,jk,ck,jlcd,ilbd', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab,
                t2_1_b, optimize=True)
            S2 -= lib.einsum('a,iba,jk,cl,jlcd,ikbd', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += lib.einsum('a,ibc,jk,di,jlda,klbc', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,ibc,jk,dk,jlda,ilbc', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab,
                t2_1_b, optimize=True)
            S2 += lib.einsum('a,ibc,jk,dl,jlda,ikbc', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            # block AjdB
            S2 -= lib.einsum('a,iab,cj,bd,ijcd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_2_ab, optimize=True)
            S2 += lib.einsum('a,iba,cj,bd,ijcd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_2_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,aj,bd,ijcd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_2_ab, optimize=True)
            S2 += lib.einsum('a,ibc,aj,cd,ijbd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,aj,bd,ijcd', L_a, R_bab, S_vo_ab, S_vv_ab, t2_2_b, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,cj,bd,ijcd', L_b, R_aba, S_vo_ab, S_vv_ab, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,dj,ca,ijdb', L_b, R_aba, S_vo_ab, S_vv_ab, t2_2_ab, optimize=True)
            # block AblB
            S2 -= lib.einsum('a,iab,bj,cd,icjd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, w_t2t2_icjd, optimize=True)
            S2 -= lib.einsum('a,iab,cj,de,ikbd,kjce', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iba,bj,cd,icjd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, w_t2t2_icjd, optimize=True)
            S2 += lib.einsum('a,iba,cj,de,ikbd,kjce', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,ibc,bj,ad,icjd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, w_t2t2_icjd, optimize=True)
            S2 += lib.einsum('a,ibc,bj,de,ikcd,kjae', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,ibc,cj,ad,ikbe,kjed', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_1_a,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,cj,de,ikbd,kjae', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,ibc,dj,ae,ikbc,kjde', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_1_a,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iab,bj,cd,icjd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, w_t2t2_icjd_2, optimize=True)
            S2 += lib.einsum('a,iba,bj,cd,icjd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, w_t2t2_icjd_2, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,ibc,bj,ad,icjd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, w_t2t2_icjd_2, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,ibc,cj,ad,ikbe,jkde', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_1_ab,
                t2_1_b, optimize=True)
            S2 -= lib.einsum('a,iab,ci,db,cd', L_a, R_bab, S_vo_ab, S_vv_ab, w_t2t2_cd, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,ibc,bi,dc,jkae,jkde', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,ibc,di,ac,jkbe,jkde', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += lib.einsum('a,ibc,di,ec,jkad,jkbe', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_a, t2_1_a, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,ci,db,cd', L_a, R_bab, S_vo_ab, S_vv_ab, w_t2t2_cd_2, optimize=True)
            S2 += 2 * lib.einsum('a,iab,ci,de,jkdb,jkce', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,iab,cj,db,kide,kjce', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,cj,de,kidb,kjce', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,bi,ad,jkec,jked', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,bi,dc,jkae,jkde', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,bi,de,jkae,jkdc', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,bj,ac,kide,kjde', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,bj,ad,kiec,kjed', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,bj,dc,kjae,kide', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,bj,de,kjae,kidc', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,di,ac,jkbe,jkde', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,di,ae,jkbc,jkde', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,dj,ac,kibe,kjde', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,dj,ae,kibc,kjde', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,ibc,bi,ad,jkce,jkde', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,ibc,bj,ac,ikde,jkde', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= lib.einsum('a,ibc,bj,ad,ikce,jkde', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_b, t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,bj,cd,icjd', L_b, R_aba, S_vo_ab, S_vv_ab, w_t2t2_icjd, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,cj,de,ikbd,kjce', L_b, R_aba, S_vo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,cj,db,ikde,kjea', L_b, R_aba, S_vo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,dj,eb,ikce,kjda', L_b, R_aba, S_vo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,bj,cd,icjd', L_b, R_aba, S_vo_ab, S_vv_ab, w_t2t2_icjd_2, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,cj,db,ikde,jkae', L_b, R_aba, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,cj,de,ikdb,jkae', L_b, R_aba, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,iab,ci,db,cd', L_b, R_bbb_u, S_vo_ab, S_vv_ab, w_t2t2_cd, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,iba,ci,db,cd', L_b, R_bbb_u, S_vo_ab, S_vv_ab, w_t2t2_cd, optimize=True)
            S2 -= lib.einsum('a,iab,ci,db,cd', L_b, R_bbb_u, S_vo_ab, S_vv_ab, w_t2t2_cd_2, optimize=True)
            S2 += lib.einsum('a,iab,ci,de,jkdb,jkce', L_b, R_bbb_u, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iab,cj,db,kide,kjce', L_b, R_bbb_u, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iab,cj,de,kidb,kjce', L_b, R_bbb_u, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iba,ci,db,cd', L_b, R_bbb_u, S_vo_ab, S_vv_ab, w_t2t2_cd_2, optimize=True)
            S2 -= lib.einsum('a,iba,ci,de,jkdb,jkce', L_b, R_bbb_u, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iba,cj,db,kide,kjce', L_b, R_bbb_u, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iba,cj,de,kidb,kjce', L_b, R_bbb_u, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,di,eb,jkda,jkec', L_b, R_bbb_u, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,di,ec,jkda,jkeb', L_b, R_bbb_u, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,dj,eb,kjda,kiec', L_b, R_bbb_u, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,dj,ec,kjda,kieb', L_b, R_bbb_u, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            # block AbdJ
            S2 -= lib.einsum('a,iab,ic,bd,jkec,jked', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iab,ic,de,jkbe,jkdc', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iab,jc,bd,iked,jkec', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iab,jc,de,ikbe,jkdc', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iba,ic,bd,jkec,jked', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iba,ic,de,jkbe,jkdc', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iba,jc,bd,iked,jkec', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iba,jc,de,ikbe,jkdc', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,id,be,jkad,jkce', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,id,ce,jkad,jkbe', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,jd,be,jkad,ikce', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,jd,ce,jkad,ikbe', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,iab,ic,bd,jkce,jkde', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,iba,ic,bd,jkce,jkde', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,jb,cd,jkce,kied', L_a, R_bab, S_ov_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,jc,bd,jkae,kied', L_a, R_bab, S_ov_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,jc,de,jkad,kibe', L_a, R_bab, S_ov_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,jb,cd,jkce,ikde', L_a, R_bab, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,jc,de,jkdc,ikbe', L_a, R_bab, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,jc,bd,jkae,ikde', L_a, R_bab, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,jd,be,jkad,ikce', L_a, R_bab, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,ibc,ib,da,cd', L_b, R_aba, S_ov_ab, S_vv_ab, w_t2t2_cd, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,ibc,jb,ca,ikde,jkde', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= lib.einsum('a,ibc,jb,da,ikce,jkde', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_a, t2_1_a, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,ic,bd,jkec,jked', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,iab,ic,de,jkbe,jkdc', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,iab,jc,bd,iked,jkec', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,iab,jc,de,ikbe,jkdc', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,ib,cd,jkea,jked', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,ib,da,cd', L_b, R_aba, S_ov_ab, S_vv_ab, w_t2t2_cd_2, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,ib,de,jkda,jkce', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,id,ca,jkeb,jked', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,id,ea,jkcb,jked', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,jb,ca,ikde,jkde', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('a,ibc,jb,cd,jkea,iked', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,jb,da,ikce,jkde', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('a,ibc,jb,de,jkda,ikce', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,ibc,jd,ca,ikeb,jked', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,ibc,jd,ea,ikcb,jked', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iab,ic,bd,jkce,jkde', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_b, t2_1_b, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,ibc,ib,cd,jkae,jkde', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,ibc,id,ca,jkbe,jkde', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += lib.einsum('a,ibc,id,ce,jkad,jkbe', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_b, t2_1_b, optimize=True)
            S2 -= lib.einsum('a,iab,jb,cd,jkce,kied', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('a,iba,jb,cd,jkce,kied', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,ibc,jb,da,jkde,kiec', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_a,
                t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,ibc,jc,da,jkde,kieb', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_a,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('a,iab,jb,cd,jkce,ikde', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= lib.einsum('a,iab,jc,de,jkdc,ikbe', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += lib.einsum('a,iba,jb,cd,jkce,ikde', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += lib.einsum('a,iba,jc,de,jkdc,ikbe', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,ibc,jb,da,jkde,ikce', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_ab,
                t2_1_b, optimize=True)
            S2 += lib.einsum('a,ibc,jb,de,jkda,ikce', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += 1 / 2 * lib.einsum('a,ibc,jc,da,jkde,ikbe', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_ab,
                t2_1_b, optimize=True)
            S2 -= lib.einsum('a,ibc,jc,de,jkda,ikbe', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= 1 / 2 * lib.einsum('a,ibc,jd,ea,jked,ikbc', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_ab,
                t2_1_b, optimize=True)

            if t1_1_a is not None:
                # block IjlJ
                S2 += lib.einsum('a,iab,jc,ikbc,jl,kl', L_a, R_aaa_u, t1_1_a, t2_1_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('a,iab,jc,jkbc,il,kl', L_a, R_aaa_u, t1_1_a, t2_1_a, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= lib.einsum('a,iba,jc,ikbc,jl,kl', L_a, R_aaa_u, t1_1_a, t2_1_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 += 1/2 * lib.einsum('a,iba,jc,jkbc,il,kl', L_a, R_aaa_u, t1_1_a, t2_1_a, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('a,ibc,ja,ikbc,jl,kl', L_a, R_aaa_u, t1_1_a, t2_1_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('a,ibc,ja,jkbc,il,kl', L_a, R_aaa_u, t1_1_a, t2_1_a, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('a,iab,jc,ikbc,lj,lk', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 += 1/2 * lib.einsum('a,iab,jc,kjbc,il,kl', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= lib.einsum('a,iab,jc,klbc,il,kj', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('a,iba,jc,ikbc,lj,lk', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('a,iba,jc,kjbc,il,kl', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('a,iba,jc,klbc,il,kj', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 += lib.einsum('a,iab,jc,jkcb,li,lk', L_a, R_bab, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 += 2 * lib.einsum('a,iab,jc,kicb,jl,kl', L_a, R_bab, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,iab,jc,klcb,ki,jl', L_a, R_bab, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= lib.einsum('a,ibc,ja,jkbc,li,lk', L_a, R_bab, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,ja,kibc,jl,kl', L_a, R_bab, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,ibc,ja,klbc,ki,jl', L_a, R_bab, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,iab,jc,ikbc,lj,lk', L_a, R_bab, t1_1_b, t2_1_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('a,iab,jc,jkbc,li,lk', L_a, R_bab, t1_1_b, t2_1_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 += 2 * lib.einsum('a,iab,jc,ikbc,jl,kl', L_b, R_aba, t1_1_a, t2_1_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('a,iab,jc,jkbc,il,kl', L_b, R_aba, t1_1_a, t2_1_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 += 2 * lib.einsum('a,iab,jc,ikbc,lj,lk', L_b, R_aba, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('a,iab,jc,kjbc,il,kl', L_b, R_aba, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,iab,jc,klbc,il,kj', L_b, R_aba, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,ja,ikcb,lj,lk', L_b, R_aba, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= lib.einsum('a,ibc,ja,kjcb,il,kl', L_b, R_aba, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 += 2 * lib.einsum('a,ibc,ja,klcb,il,kj', L_b, R_aba, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,iab,jc,jkcb,li,lk', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('a,iab,jc,kicb,jl,kl', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('a,iab,jc,klcb,ki,jl', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('a,iba,jc,jkcb,li,lk', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= lib.einsum('a,iba,jc,kicb,jl,kl', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 += lib.einsum('a,iba,jc,klcb,ki,jl', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 += lib.einsum('a,iab,jc,ikbc,lj,lk', L_b, R_bbb_u, t1_1_b, t2_1_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('a,iab,jc,jkbc,li,lk', L_b, R_bbb_u, t1_1_b, t2_1_b, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= lib.einsum('a,iba,jc,ikbc,lj,lk', L_b, R_bbb_u, t1_1_b, t2_1_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 += 1/2 * lib.einsum('a,iba,jc,jkbc,li,lk', L_b, R_bbb_u, t1_1_b, t2_1_b, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('a,ibc,ja,ikbc,lj,lk', L_b, R_bbb_u, t1_1_b, t2_1_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('a,ibc,ja,jkbc,li,lk', L_b, R_bbb_u, t1_1_b, t2_1_b, S_oo_ab, S_oo_ab,
                    optimize=True)
                # block IjlB & AjlJ
                S2 += lib.einsum('a,iab,jk,ck,ic,jb', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
                S2 -= lib.einsum('a,iba,jk,ck,ic,jb', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('a,iab,ji,ck,jc,kb', L_a, R_bab, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,ji,ak,jb,kc', L_a, R_bab, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('a,iab,jk,ck,ic,jb', L_b, R_aba, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
                S2 += lib.einsum('a,iab,ji,ck,jc,kb', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
                S2 -= lib.einsum('a,iba,ji,ck,jc,kb', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
                # block IjdJ & IblJ
                S2 += lib.einsum('a,iab,ij,kc,kb,jc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t1_1_a, t1_1_b, optimize=True)
                S2 -= lib.einsum('a,iba,ij,kc,kb,jc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('a,iab,jk,jc,ic,kb', L_a, R_bab, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('a,iab,ij,kc,kb,jc', L_b, R_aba, S_oo_ab, S_ov_ab, t1_1_a, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,ij,ka,kc,jb', L_b, R_aba, S_oo_ab, S_ov_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += lib.einsum('a,iab,jk,jc,ic,kb', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
                S2 -= lib.einsum('a,iba,jk,jc,ic,kb', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
                # block IjdB & AblJ
                S2 += lib.einsum('a,iab,jc,dk,id,jkbc', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += lib.einsum('a,iab,jc,dk,jb,ikdc', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,iab,jc,dk,jd,ikbc', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,iba,jc,dk,id,jkbc', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,iba,jc,dk,jb,ikdc', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += lib.einsum('a,iba,jc,dk,jd,ikbc', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += lib.einsum('a,ibc,jd,ak,jb,ikcd', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,ibc,jd,ak,jc,ikbd', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,iab,jc,dk,kc,ijbd', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_a, optimize=True)
                S2 += lib.einsum('a,iba,jc,dk,kc,ijbd', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_a, optimize=True)
                S2 -= lib.einsum('a,ibc,jd,ak,kd,ijbc', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_a, optimize=True)
                S2 -= 2 * lib.einsum('a,iab,jc,dk,jd,ikbc', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_a, t2_1_b, optimize=True)
                S2 += 2 * lib.einsum('a,ibc,jd,ak,jb,ikcd', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_a, t2_1_b, optimize=True)
                S2 += 2 * lib.einsum('a,iab,jc,dk,ic,jkdb', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,iab,jc,dk,kb,jidc', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,iab,jc,dk,kc,jidb', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,jd,ak,id,jkbc', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,jd,ak,kc,jibd', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,ibc,jd,ak,kd,jibc', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,iab,jc,dk,id,jkbc', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,iab,jc,dk,jb,ikdc', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,iab,jc,dk,jd,ikbc', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,ja,dk,id,jkcb', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,ja,dk,jc,ikdb', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,ibc,ja,dk,jd,ikcb', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,iab,jc,dk,kc,ijbd', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_b, t2_1_a, optimize=True)
                S2 += 2 * lib.einsum('a,ibc,ja,dk,kb,ijcd', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_b, t2_1_a, optimize=True)
                S2 -= lib.einsum('a,iab,jc,dk,jd,ikbc', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_b, optimize=True)
                S2 += lib.einsum('a,iba,jc,dk,jd,ikbc', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_b, optimize=True)
                S2 -= lib.einsum('a,ibc,ja,dk,jd,ikbc', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_b, optimize=True)
                S2 += lib.einsum('a,iab,jc,dk,ic,jkdb', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += lib.einsum('a,iab,jc,dk,kb,jidc', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,iab,jc,dk,kc,jidb', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,iba,jc,dk,ic,jkdb', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,iba,jc,dk,kb,jidc', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += lib.einsum('a,iba,jc,dk,kc,jidb', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += lib.einsum('a,ibc,ja,dk,kb,jidc', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,ibc,ja,dk,kc,jidb', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                # block IblJ & IjdJ
                S2 += lib.einsum('a,iab,jk,ic,jb,kc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t1_1_a, t1_1_b, optimize=True)
                S2 -= lib.einsum('a,iba,jk,ic,jb,kc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('a,iab,ji,kb,jc,kc', L_a, R_bab, S_oo_ab, S_ov_ab, t1_1_a, t1_1_a, optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,ji,kc,ka,jb', L_a, R_bab, S_oo_ab, S_ov_ab, t1_1_a, t1_1_a, optimize=True)
                S2 += lib.einsum('a,iab,ji,jc,kb,kc', L_a, R_bab, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
                S2 += lib.einsum('a,iab,jk,jb,ic,kc', L_a, R_bab, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('a,iab,jk,ic,jb,kc', L_b, R_aba, S_oo_ab, S_ov_ab, t1_1_a, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,jk,ib,jc,ka', L_b, R_aba, S_oo_ab, S_ov_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += lib.einsum('a,iab,ji,kb,jc,kc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_a, t1_1_a, optimize=True)
                S2 -= lib.einsum('a,iba,ji,kb,jc,kc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_a, t1_1_a, optimize=True)
                S2 += 1/2 * lib.einsum('a,iab,ji,jc,kb,kc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,iab,jk,jb,ic,kc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,iba,ji,jc,kb,kc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,iba,jk,jb,ic,kc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b,
                    optimize=True)
                # block IblB & AjdJ
                S2 -= lib.einsum('a,iab,jk,cd,kd,ijbc', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_a, optimize=True)
                S2 += lib.einsum('a,iba,jk,cd,kd,ijbc', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_a, optimize=True)
                S2 -= lib.einsum('a,ibc,jk,ad,kd,ijbc', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_a, optimize=True)
                S2 -= lib.einsum('a,iab,ji,cb,kd,jkcd', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_a, t2_1_a, optimize=True)
                S2 += lib.einsum('a,ibc,ji,ac,kd,jkbd', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_a, t2_1_a, optimize=True)
                S2 -= lib.einsum('a,ibc,ji,dc,ka,jkbd', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_a, t2_1_a, optimize=True)
                S2 -= lib.einsum('a,iab,ji,cb,kd,jkcd', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += 2 * lib.einsum('a,iab,ji,cd,kd,jkcb', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,iab,jk,cb,kd,jicd', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,iab,jk,cd,kd,jicb', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,ibc,ji,ac,kd,jkbd', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,ji,ad,kd,jkbc', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,jk,ac,kd,jibd', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,ibc,jk,ad,kd,jibc', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,iab,jk,cd,kd,ijbc', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_b, t2_1_a, optimize=True)
                S2 += 2 * lib.einsum('a,ibc,jk,db,ka,ijcd', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_b, t2_1_a, optimize=True)
                S2 -= 1/2 * lib.einsum('a,iab,ji,cb,kd,jkcd', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_a,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,iba,ji,cb,kd,jkcd', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_a,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,iab,ji,cb,kd,jkcd', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,iab,ji,cd,kd,jkcb', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += lib.einsum('a,iab,jk,cb,kd,jicd', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,iab,jk,cd,kd,jicb', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += 1/2 * lib.einsum('a,iba,ji,cb,kd,jkcd', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,iba,ji,cd,kd,jkcb', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,iba,jk,cb,kd,jicd', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += lib.einsum('a,iba,jk,cd,kd,jicb', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('a,ibc,ji,db,ka,jkdc', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,ibc,ji,dc,ka,jkdb', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,ibc,jk,db,ka,jidc', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,ibc,jk,dc,ka,jidb', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
                # block IbdJ
                S2 -= lib.einsum('a,iab,jc,ijbd,kc,kd', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 += lib.einsum('a,iab,jc,kjbd,ic,kd', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 += lib.einsum('a,iba,jc,ijbd,kc,kd', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= lib.einsum('a,iba,jc,kjbd,ic,kd', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= lib.einsum('a,iab,jc,jicd,kb,kd', L_a, R_bab, t1_1_a, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 += 2 * lib.einsum('a,iab,jc,kicd,jb,kd', L_a, R_bab, t1_1_a, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += lib.einsum('a,ibc,ja,jibd,kc,kd', L_a, R_bab, t1_1_a, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,ja,kibd,jc,kd', L_a, R_bab, t1_1_a, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,iab,jc,ijbd,kc,kd', L_a, R_bab, t1_1_b, t2_1_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 += lib.einsum('a,iab,jc,ijcd,kb,kd', L_a, R_bab, t1_1_b, t2_1_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 += 2 * lib.einsum('a,ibc,jd,ikcd,ka,jb', L_b, R_aba, t1_1_a, t2_1_a, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= lib.einsum('a,ibc,jd,jkcd,ka,ib', L_b, R_aba, t1_1_a, t2_1_a, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,iab,jc,ijbd,kc,kd', L_b, R_aba, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,iab,jc,kjbd,ic,kd', L_b, R_aba, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += lib.einsum('a,ibc,ja,ijcd,kb,kd', L_b, R_aba, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= lib.einsum('a,ibc,ja,kjcd,ib,kd', L_b, R_aba, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 += lib.einsum('a,ibc,jd,ijcb,ka,kd', L_b, R_aba, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,jd,kjcb,ka,id', L_b, R_aba, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += lib.einsum('a,ibc,jd,kjcd,ka,ib', L_b, R_aba, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('a,iab,jc,jicd,kb,kd', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += lib.einsum('a,iab,jc,kicd,jb,kd', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 += 1/2 * lib.einsum('a,iba,jc,jicd,kb,kd', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 -= lib.einsum('a,iba,jc,kicd,jb,kd', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= lib.einsum('a,ibc,jd,kidb,ka,jc', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 += lib.einsum('a,ibc,jd,kidc,ka,jb', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= lib.einsum('a,iab,jc,ijbd,kc,kd', L_b, R_bbb_u, t1_1_b, t2_1_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 += 1/2 * lib.einsum('a,iab,jc,ijcd,kb,kd', L_b, R_bbb_u, t1_1_b, t2_1_b, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += lib.einsum('a,iba,jc,ijbd,kc,kd', L_b, R_bbb_u, t1_1_b, t2_1_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('a,iba,jc,ijcd,kb,kd', L_b, R_bbb_u, t1_1_b, t2_1_b, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,ibc,ja,ijbd,kc,kd', L_b, R_bbb_u, t1_1_b, t2_1_b, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,ibc,ja,ijcd,kb,kd', L_b, R_bbb_u, t1_1_b, t2_1_b, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,ibc,jd,ijbc,ka,kd', L_b, R_bbb_u, t1_1_b, t2_1_b, S_ov_ab, S_ov_ab,
                    optimize=True)
                # block IbdB & AbdJ
                S2 -= 2 * lib.einsum('a,iab,jc,db,jd,ic', L_a, R_bab, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('a,ibc,jd,ac,jb,id', L_a, R_bab, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('a,ibc,ja,db,id,jc', L_b, R_aba, S_ov_ab, S_vv_ab, t1_1_a, t1_1_a, optimize=True)
                S2 -= lib.einsum('a,iab,jc,db,jd,ic', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += lib.einsum('a,iba,jc,db,jd,ic', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                # block AjlJ & IjlB
                S2 += 1/2 * lib.einsum('a,iab,ij,cj,kb,kc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,iab,jk,bk,ic,jc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,iba,ij,cj,kb,kc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,iba,jk,bk,ic,jc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += lib.einsum('a,iab,ij,bk,jc,kc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_b, t1_1_b, optimize=True)
                S2 -= lib.einsum('a,iba,ij,bk,jc,kc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_b, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('a,iab,jk,ci,jc,kb', L_a, R_bab, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,jk,bi,ja,kc', L_a, R_bab, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += lib.einsum('a,iab,ij,cj,kb,kc', L_b, R_aba, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
                S2 += lib.einsum('a,iab,jk,bk,ic,jc', L_b, R_aba, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('a,iab,ij,bk,jc,kc', L_b, R_aba, S_oo_ab, S_vo_ab, t1_1_b, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,ij,ck,ka,jb', L_b, R_aba, S_oo_ab, S_vo_ab, t1_1_b, t1_1_b, optimize=True)
                S2 += lib.einsum('a,iab,jk,ci,jc,kb', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
                S2 -= lib.einsum('a,iba,jk,ci,jc,kb', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
                # block AjlB
                S2 -= lib.einsum('a,iab,jc,ijbd,ck,dk', L_a, R_aaa_u, t1_1_a, t2_1_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 += 1/2 * lib.einsum('a,iab,jc,ijcd,bk,dk', L_a, R_aaa_u, t1_1_a, t2_1_a, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += lib.einsum('a,iba,jc,ijbd,ck,dk', L_a, R_aaa_u, t1_1_a, t2_1_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('a,iba,jc,ijcd,bk,dk', L_a, R_aaa_u, t1_1_a, t2_1_a, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,ibc,ja,ijbd,ck,dk', L_a, R_aaa_u, t1_1_a, t2_1_a, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,ibc,ja,ijcd,bk,dk', L_a, R_aaa_u, t1_1_a, t2_1_a, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,ibc,jd,ijbc,ak,dk', L_a, R_aaa_u, t1_1_a, t2_1_a, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,iab,jc,ijdc,bk,dk', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += lib.einsum('a,iab,jc,ikdc,bj,dk', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 += 1/2 * lib.einsum('a,iba,jc,ijdc,bk,dk', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 -= lib.einsum('a,iba,jc,ikdc,bj,dk', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= lib.einsum('a,ibc,jd,ikbd,ak,cj', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 += lib.einsum('a,ibc,jd,ikcd,ak,bj', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,iab,jc,jidb,ck,dk', L_a, R_bab, t1_1_a, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,iab,jc,jkdb,ci,dk', L_a, R_bab, t1_1_a, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += lib.einsum('a,ibc,ja,jidc,bk,dk', L_a, R_bab, t1_1_a, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= lib.einsum('a,ibc,ja,jkdc,bi,dk', L_a, R_bab, t1_1_a, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 += lib.einsum('a,ibc,jd,jibc,ak,dk', L_a, R_bab, t1_1_a, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,jd,jkbc,ak,di', L_a, R_bab, t1_1_a, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += lib.einsum('a,ibc,jd,jkdc,ak,bi', L_a, R_bab, t1_1_a, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 += 2 * lib.einsum('a,ibc,jd,ikcd,ak,bj', L_a, R_bab, t1_1_b, t2_1_b, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= lib.einsum('a,ibc,jd,jkcd,ak,bi', L_a, R_bab, t1_1_b, t2_1_b, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,iab,jc,ijbd,ck,dk', L_b, R_aba, t1_1_a, t2_1_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 += lib.einsum('a,iab,jc,ijcd,bk,dk', L_b, R_aba, t1_1_a, t2_1_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= lib.einsum('a,iab,jc,ijdc,bk,dk', L_b, R_aba, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 += 2 * lib.einsum('a,iab,jc,ikdc,bj,dk', L_b, R_aba, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += lib.einsum('a,ibc,ja,ijdb,ck,dk', L_b, R_aba, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,ja,ikdb,cj,dk', L_b, R_aba, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 -= lib.einsum('a,iab,jc,jidb,ck,dk', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 += lib.einsum('a,iab,jc,jkdb,ci,dk', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 += lib.einsum('a,iba,jc,jidb,ck,dk', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= lib.einsum('a,iba,jc,jkdb,ci,dk', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                # block AjdJ & IblB
                S2 -= 1/2 * lib.einsum('a,iab,ij,bc,kd,kjdc', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,iab,ij,cd,kc,kjbd', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += lib.einsum('a,iab,jk,bc,jd,ikdc', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,iab,jk,cd,jc,ikbd', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += 1/2 * lib.einsum('a,iba,ij,bc,kd,kjdc', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,iba,ij,cd,kc,kjbd', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,iba,jk,bc,jd,ikdc', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += lib.einsum('a,iba,jk,cd,jc,ikbd', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('a,ibc,ij,bd,ka,kjcd', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,ibc,ij,cd,ka,kjbd', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,ibc,jk,bd,ja,ikcd', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,ibc,jk,cd,ja,ikbd', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('a,iab,ij,bc,kd,jkcd', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_b,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,iba,ij,bc,kd,jkcd', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,iab,jk,cd,jc,ikbd', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_a, t2_1_b, optimize=True)
                S2 += 2 * lib.einsum('a,ibc,jk,bd,ja,ikcd', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_a, t2_1_b, optimize=True)
                S2 -= lib.einsum('a,iab,ij,bc,kd,kjdc', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += 2 * lib.einsum('a,iab,ij,cd,kc,kjbd', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,iab,jk,bc,jd,ikdc', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,iab,jk,cd,jc,ikbd', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('a,ibc,ij,ca,kd,kjdb', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,ij,da,kd,kjcb', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,jk,ca,jd,ikdb', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('a,ibc,jk,da,jd,ikcb', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('a,iab,ij,bc,kd,jkcd', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_b, t2_1_b, optimize=True)
                S2 += lib.einsum('a,ibc,ij,ca,kd,jkbd', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_b, t2_1_b, optimize=True)
                S2 -= lib.einsum('a,ibc,ij,cd,ka,jkbd', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_b, t2_1_b, optimize=True)
                S2 -= lib.einsum('a,iab,jk,cd,jc,ikbd', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_b, optimize=True)
                S2 += lib.einsum('a,iba,jk,cd,jc,ikbd', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_b, optimize=True)
                S2 -= lib.einsum('a,ibc,jk,da,jd,ikbc', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_b, optimize=True)
                # block AjdB & AblB
                S2 -= lib.einsum('a,iab,cj,bd,ic,jd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += lib.einsum('a,iba,cj,bd,ic,jd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('a,ibc,aj,bd,id,jc', L_a, R_bab, S_vo_ab, S_vv_ab, t1_1_b, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('a,iab,cj,bd,ic,jd', L_b, R_aba, S_vo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('a,ibc,dj,ca,id,jb', L_b, R_aba, S_vo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                # block AblJ & IjdB
                S2 -= 1/2 * lib.einsum('a,iab,ic,bj,kd,kjdc', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,iab,ic,dj,kb,kjdc', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,iab,jc,bk,id,jkdc', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,iba,ic,bj,kd,kjdc', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,iba,ic,dj,kb,kjdc', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,iba,jc,bk,id,jkdc', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,ibc,id,bj,kc,kjad', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,ibc,id,cj,kb,kjad', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,iab,ic,bj,kd,jkcd', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_b,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,iba,ic,bj,kd,jkcd', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('a,iab,jb,ci,kd,jkcd', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_a, t2_1_a, optimize=True)
                S2 += lib.einsum('a,ibc,jc,bi,kd,jkad', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_a, t2_1_a, optimize=True)
                S2 -= lib.einsum('a,ibc,jc,di,kb,jkad', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_a, t2_1_a, optimize=True)
                S2 -= lib.einsum('a,iab,jb,ci,kd,jkcd', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += lib.einsum('a,iab,jb,ck,id,jkcd', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += lib.einsum('a,iab,jc,di,kb,jkdc', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += lib.einsum('a,ibc,jc,bi,kd,jkad', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,ibc,jc,bk,id,jkad', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,ibc,jd,bi,kc,jkad', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,iab,ic,bj,kd,kjdc', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += lib.einsum('a,iab,ic,dj,kb,kjdc', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += lib.einsum('a,iab,jc,bk,id,jkdc', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += lib.einsum('a,ibc,ib,cj,kd,kjda', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,ibc,ib,dj,kc,kjda', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,ibc,jb,ck,id,jkda', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= lib.einsum('a,iab,ic,bj,kd,jkcd', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_b, t2_1_b, optimize=True)
                S2 += lib.einsum('a,ibc,ib,cj,kd,jkad', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_b, t2_1_b, optimize=True)
                S2 -= lib.einsum('a,ibc,id,cj,kb,jkad', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_b, t2_1_b, optimize=True)
                S2 -= 1/2 * lib.einsum('a,iab,jb,ci,kd,jkcd', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_a,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,iba,jb,ci,kd,jkcd', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_a,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,iab,jb,ci,kd,jkcd', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,iab,jb,ck,id,jkcd', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,iab,jc,di,kb,jkdc', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,iba,jb,ci,kd,jkcd', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,iba,jb,ck,id,jkcd', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,iba,jc,di,kb,jkdc', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('a,ibc,jb,di,kc,jkda', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('a,ibc,jc,di,kb,jkda', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                # block AblB & AjdB
                S2 -= lib.einsum('a,iab,bj,cd,ic,jd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += lib.einsum('a,iba,bj,cd,ic,jd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('a,iab,ci,db,jc,jd', L_a, R_bab, S_vo_ab, S_vv_ab, t1_1_a, t1_1_a, optimize=True)
                S2 += lib.einsum('a,ibc,bi,dc,ja,jd', L_a, R_bab, S_vo_ab, S_vv_ab, t1_1_a, t1_1_a, optimize=True)
                S2 += lib.einsum('a,ibc,di,ac,jb,jd', L_a, R_bab, S_vo_ab, S_vv_ab, t1_1_a, t1_1_a, optimize=True)
                S2 += lib.einsum('a,ibc,bi,ad,jc,jd', L_a, R_bab, S_vo_ab, S_vv_ab, t1_1_b, t1_1_b, optimize=True)
                S2 += lib.einsum('a,ibc,bj,ac,id,jd', L_a, R_bab, S_vo_ab, S_vv_ab, t1_1_b, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('a,iab,bj,cd,ic,jd', L_b, R_aba, S_vo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('a,ibc,cj,db,id,ja', L_b, R_aba, S_vo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 -= lib.einsum('a,iab,ci,db,jc,jd', L_b, R_bbb_u, S_vo_ab, S_vv_ab, t1_1_a, t1_1_a, optimize=True)
                S2 += lib.einsum('a,iba,ci,db,jc,jd', L_b, R_bbb_u, S_vo_ab, S_vv_ab, t1_1_a, t1_1_a, optimize=True)
                # block AbdJ & IbdB
                S2 -= lib.einsum('a,iab,ic,bd,jc,jd', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t1_1_b, t1_1_b, optimize=True)
                S2 += lib.einsum('a,iba,ic,bd,jc,jd', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t1_1_b, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('a,iab,jb,cd,jc,id', L_a, R_bab, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('a,ibc,jc,bd,ja,id', L_a, R_bab, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += lib.einsum('a,ibc,ib,da,jc,jd', L_b, R_aba, S_ov_ab, S_vv_ab, t1_1_a, t1_1_a, optimize=True)
                S2 += lib.einsum('a,ibc,jb,ca,id,jd', L_b, R_aba, S_ov_ab, S_vv_ab, t1_1_a, t1_1_a, optimize=True)
                S2 -= 2 * lib.einsum('a,iab,ic,bd,jc,jd', L_b, R_aba, S_ov_ab, S_vv_ab, t1_1_b, t1_1_b, optimize=True)
                S2 += lib.einsum('a,ibc,ib,cd,ja,jd', L_b, R_aba, S_ov_ab, S_vv_ab, t1_1_b, t1_1_b, optimize=True)
                S2 += lib.einsum('a,ibc,id,ca,jb,jd', L_b, R_aba, S_ov_ab, S_vv_ab, t1_1_b, t1_1_b, optimize=True)
                S2 -= lib.einsum('a,iab,jb,cd,jc,id', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += lib.einsum('a,iba,jb,cd,jc,id', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                # block AbdB
                S2 -= lib.einsum('a,iab,jc,ijde,be,dc', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab, optimize=True)
                S2 += lib.einsum('a,iba,jc,ijde,be,dc', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab, optimize=True)
                S2 += lib.einsum('a,ibc,jd,ijbe,ad,ce', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab, optimize=True)
                S2 -= lib.einsum('a,ibc,jd,ijce,ad,be', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,iab,jc,jide,db,ce', L_a, R_bab, t1_1_a, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 += lib.einsum('a,ibc,ja,jide,be,dc', L_a, R_bab, t1_1_a, t2_1_ab, S_vv_ab, S_vv_ab, optimize=True)
                S2 += 2 * lib.einsum('a,ibc,jd,jibe,ac,de', L_a, R_bab, t1_1_a, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 -= lib.einsum('a,ibc,jd,jide,ac,be', L_a, R_bab, t1_1_a, t2_1_ab, S_vv_ab, S_vv_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,jd,ijce,ad,be', L_a, R_bab, t1_1_b, t2_1_b, S_vv_ab, S_vv_ab, optimize=True)
                S2 += lib.einsum('a,ibc,jd,ijde,ac,be', L_a, R_bab, t1_1_b, t2_1_b, S_vv_ab, S_vv_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,ibc,jd,ijce,da,eb', L_b, R_aba, t1_1_a, t2_1_a, S_vv_ab, S_vv_ab, optimize=True)
                S2 += lib.einsum('a,ibc,jd,ijde,ca,eb', L_b, R_aba, t1_1_a, t2_1_a, S_vv_ab, S_vv_ab, optimize=True)
                S2 -= 2 * lib.einsum('a,iab,jc,ijde,be,dc', L_b, R_aba, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 += lib.einsum('a,ibc,ja,ijde,db,ce', L_b, R_aba, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab, optimize=True)
                S2 += 2 * lib.einsum('a,ibc,jd,ijeb,ca,ed', L_b, R_aba, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 -= lib.einsum('a,ibc,jd,ijed,ca,eb', L_b, R_aba, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab, optimize=True)
                S2 -= lib.einsum('a,iab,jc,jide,db,ce', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_vv_ab, S_vv_ab, optimize=True)
                S2 += lib.einsum('a,iba,jc,jide,db,ce', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_vv_ab, S_vv_ab, optimize=True)
                S2 += lib.einsum('a,ibc,jd,jieb,da,ec', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_vv_ab, S_vv_ab, optimize=True)
                S2 -= lib.einsum('a,ibc,jd,jiec,da,eb', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_vv_ab, S_vv_ab, optimize=True)

        if adc.frozen is not None:
            ga_fr = dm_a[r][np.ix_(act_a_fr, act_a_fr)]
            gb_fr = dm_b[r][np.ix_(act_b_fr, act_b_fr)]
            S2 -= lib.einsum('pc,sc,sp->', S_ac_fr, S_ac_fr, ga_fr, optimize=True)
            S2 -= lib.einsum('cp,cq,pq->', S_ca_fr, S_ca_fr, gb_fr, optimize=True)
            S2 -= lib.einsum('cp,cp->', S_cc_fr, S_cc_fr, optimize=True)

        # 1-RDM and 2-RDM contributions to the <S^2> values
        na = np.trace(dm_a[r])
        nb = np.trace(dm_b[r])

        spin_square = 0.25 * ((na - nb)**2) + 0.5 * (na + nb) + S2

        spin = np.append(spin, spin_square)

        trace_a = np.append(trace_a, na)
        trace_b = np.append(trace_b, nb)

    if method == "adc(3)":
        del w_t2t2_icjd, w_t2t2_icjd_2, w_t2t2_ijkl, w_t2t2_cd, w_t2t2_cd_2

    return spin, (trace_a, trace_b)


class UADCEA(uadc.UADC):
    '''unrestricted ADC for EA energies and spectroscopic amplitudes

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
            Convergence threshold for Davidson iterations.  Default is 1e-8.
        max_cycle : int
            Number of Davidson iterations.  Default is 50.
        max_space : int
            Space size to hold trial vectors for Davidson iterative diagonalization.  Default is 12.

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested. Default value is 1.

            >>> myadc = adc.UADC(mf).run()
            >>> myadcea = adc.UADCEA(myadc).run()

    Saved results

        e_ea : float or list of floats
            EA energy (eigenvalue). For nroots = 1, it is a single float
            number. If nroots > 1, it is a list of floats for the lowest
            nroots eigenvalues.
        v_ea : array
            Eigenvectors for each EA transition.
        p_ea : float
            Spectroscopic factors for each EA transition.
        x_ea : float
            Spectroscopic amplitudes for each EA transition.
    '''

    _keys = {
        'tol_residual','conv_tol', 'e_corr', 'method',
        'method_type', 'mo_coeff', 'mo_coeff_hf', 'mo_energy_b', 'max_memory',
        't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle',
        'nocc_a', 'nocc_b', 'nvir_a', 'nvir_b', 'nmo_a', 'nmo_b', 'mol', 'transform_integrals',
        'with_df', 'spec_factor_print_tol', 'evec_print_tol',
        'compute_properties', 'approx_trans_moments', 'E', 'U', 'P', 'X',
        'compute_spin_square', '_make_rdm1', 'frozen', 'mo_occ', 'if_naf', 'naux', 'f_ov'
    }

    def __init__(self, adc):
        self.mol = adc.mol
        self.verbose = adc.verbose
        self.stdout = adc.stdout
        self.max_memory = adc.max_memory
        self.max_space = adc.max_space
        self.max_cycle = adc.max_cycle
        self.conv_tol  = adc.conv_tol
        self.tol_residual  = adc.tol_residual
        self.t1 = adc.t1
        self.t2 = adc.t2
        self.f_ov = adc.f_ov
        self.imds = adc.imds
        self.e_corr = adc.e_corr
        self.method = adc.method
        self.method_type = "ea"
        self._scf = adc._scf
        self._nocc = adc._nocc
        self._nvir = adc._nvir
        self._nmo = adc._nmo
        self.nocc_a = adc.nocc_a
        self.nocc_b = adc.nocc_b
        self.nvir_a = adc.nvir_a
        self.nvir_b = adc.nvir_b
        self.mo_coeff = adc.mo_coeff
        self.mo_coeff_hf = adc.mo_coeff_hf
        self.mo_energy_a = adc.mo_energy_a
        self.mo_energy_b = adc.mo_energy_b
        self.nmo_a = adc._nmo[0]
        self.nmo_b = adc._nmo[1]
        self.if_heri_eris = adc.if_heri_eris
        self.thresh_naf = adc.thresh_naf
        self.ncvs = adc.ncvs
        if getattr(adc.transform_integrals, '__name__', '') == 'df_transform':
            self.transform_integrals = lambda: uadc_ao2mo.transform_integrals_df(self)
        elif getattr(adc.transform_integrals, '__name__', '') == 'outcore_transform':
            self.transform_integrals = lambda: uadc_ao2mo.transform_integrals_outcore(self)
        self.with_df = adc.with_df
        self.compute_properties = adc.compute_properties
        self.approx_trans_moments = adc.approx_trans_moments
        self.frozen = adc.frozen
        self.mo_occ = adc.mo_occ
        self.if_naf = adc.if_naf
        self.naux = adc.naux

        self.spec_factor_print_tol = adc.spec_factor_print_tol
        self.evec_print_tol = adc.evec_print_tol

        self.compute_spin_square = adc.compute_spin_square

        self.E = adc.E
        self.U = adc.U
        self.P = adc.P
        self.X = adc.X

        self._adc_es = self

    kernel = uadc.kernel
    get_imds = get_imds
    matvec = matvec
    get_diag = get_diag
    get_trans_moments = get_trans_moments
    get_properties = get_properties

    analyze = analyze
    analyze_spec_factor = analyze_spec_factor
    analyze_eigenvector = analyze_eigenvector
    compute_dyson_mo = compute_dyson_mo
    _make_rdm1 = make_rdm1
    get_spin_square = get_spin_square

    def get_init_guess(self, nroots=1, diag=None, ascending=True, type=None, ini=None):
        if (type=="read"):
            logger.info(self,"obtain initial guess from input variable")
            nocc_a = self.nocc_a
            nocc_b = self.nocc_b
            nvir_a = self.nvir_a
            nvir_b = self.nvir_b
            n_singles_a = nvir_a
            n_singles_b = nvir_b
            n_doubles_aaa = nvir_a * (nvir_a - 1) * nocc_a // 2
            n_doubles_bab = nocc_b * nvir_a * nvir_b
            n_doubles_aba = nocc_a * nvir_b * nvir_a
            n_doubles_bbb = nvir_b * (nvir_b - 1) * nocc_b // 2
            dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb
            if isinstance(ini, list):
                g = np.array(ini)
            else:
                g = ini
            if g.shape[0] != dim or g.shape[1] != nroots:
                raise ValueError(f"Shape of guess should be ({dim},{nroots})")

        else:
            if diag is None :
                diag = self.get_diag()
            idx = None
            if ascending:
                idx = np.argsort(diag)
            else:
                idx = np.argsort(diag)[::-1]
            guess = np.zeros((diag.shape[0], nroots))
            min_shape = min(diag.shape[0], nroots)
            guess[:min_shape,:min_shape] = np.identity(min_shape)
            g = np.zeros((diag.shape[0], nroots))
            g[idx] = guess.copy()
        guess = []
        for p in range(g.shape[1]):
            guess.append(g[:,p])
        return guess

    def gen_matvec(self, imds=None, eris=None):
        if imds is None:
            imds = self.get_imds(eris)
        diag = self.get_diag(imds,eris)
        matvec = self.matvec(imds, eris)
        #matvec = lambda x: self.matvec()
        return matvec, diag


def contract_r_vvvv_antisym(myadc,r2,vvvv_d):

    nocc = r2.shape[0]
    nvir = r2.shape[1]

    nv_pair = nvir  *  (nvir - 1) // 2
    tril_idx = np.tril_indices(nvir, k=-1)

    r2 = r2[:,tril_idx[0],tril_idx[1]]
    r2 = np.ascontiguousarray(r2.reshape(nocc,-1))

    r2_vvvv = np.zeros((nocc,nvir,nvir))
    chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
    a = 0
    if isinstance(vvvv_d,list):
        for dataset in vvvv_d:
            k = dataset.shape[0]
            dataset = dataset[:].reshape(-1,nv_pair)
            r2_vvvv[:,a:a+k] = np.dot(r2,dataset.T).reshape(nocc,-1,nvir)
            a += k
    elif getattr(myadc, 'with_df', None):
        for a,b in lib.prange(0,nvir,chnk_size):
            vvvv = dfadc.get_vvvv_antisym_df(myadc, vvvv_d, a, chnk_size)
            vvvv = vvvv.reshape(-1,nv_pair)
            r2_vvvv[:,a:b] = np.dot(r2,vvvv.T).reshape(nocc,-1,nvir)
            del vvvv
    else:
        raise Exception("Unknown vvvv type")
    return r2_vvvv


def contract_r_vvvv(myadc,r2,vvvv_d):

    nocc_1 = r2.shape[0]
    nvir_1 = r2.shape[1]
    nvir_2 = r2.shape[2]

    r2 = r2.reshape(-1,nvir_1*nvir_2)
    r2_vvvv = np.zeros((nocc_1,nvir_1,nvir_2))
    chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)

    a = 0
    if isinstance(vvvv_d, list):
        for dataset in vvvv_d:
            k = dataset.shape[0]
            dataset = dataset[:].reshape(-1,nvir_1*nvir_2)
            r2_vvvv[:,a:a+k] = np.dot(r2,dataset.T).reshape(nocc_1,-1,nvir_2)
            a += k
    elif getattr(myadc, 'with_df', None):
        Lvv = vvvv_d[0]
        LVV = vvvv_d[1]
        for a,b in lib.prange(0,nvir_1,chnk_size):
            vvvv = dfadc.get_vVvV_df(myadc, Lvv, LVV, a, chnk_size)
            vvvv = vvvv.reshape(-1,nvir_1*nvir_2)
            r2_vvvv[:,a:b] = np.dot(r2,vvvv.T).reshape(nocc_1,-1,nvir_2)
            del vvvv
    else:
        raise Exception("Unknown vvvv type")

    return r2_vvvv
