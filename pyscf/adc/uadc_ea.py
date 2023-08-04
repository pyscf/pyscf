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

'''
Unrestricted algebraic diagrammatic construction
'''

import numpy as np
from pyscf import lib, symm
from pyscf.lib import logger
from pyscf.adc import uadc
from pyscf.adc import uadc_ao2mo
from pyscf.adc import radc_ao2mo
from pyscf.adc import dfadc
from pyscf import __config__
from pyscf import df


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

    # a-b block
    # Zeroth-order terms

    M_ab_a = lib.einsum('ab,a->ab', idn_vir_a, e_vir_a)
    M_ab_b = lib.einsum('ab,a->ab', idn_vir_b, e_vir_b)

    # Second-order terms

    t2_1_a = t2[0][0][:]
    M_ab_a -= 0.5 * 0.5 *  lib.einsum('lmad,lbdm->ab',t2_1_a, eris_ovvo,optimize=True)
    M_ab_a += 0.5 * 0.5 *  lib.einsum('lmad,ldbm->ab',t2_1_a, eris_ovvo,optimize=True)
    M_ab_a -= 0.5 * 0.5 *  lib.einsum('lmbd,ladm->ab',t2_1_a,eris_ovvo,optimize=True)
    M_ab_a += 0.5 * 0.5 *  lib.einsum('lmbd,ldam->ab',t2_1_a, eris_ovvo,optimize=True)
    del t2_1_a

    t2_1_b = t2[0][2][:]
    M_ab_b -= 0.5 * 0.5 *  lib.einsum('lmad,lbdm->ab',t2_1_b, eris_OVVO,optimize=True)
    M_ab_b += 0.5 * 0.5 *  lib.einsum('lmad,ldbm->ab',t2_1_b, eris_OVVO,optimize=True)
    M_ab_b -= 0.5 * 0.5 *  lib.einsum('lmbd,ladm->ab',t2_1_b, eris_OVVO,optimize=True)
    M_ab_b += 0.5 * 0.5 *  lib.einsum('lmbd,ldam->ab',t2_1_b, eris_OVVO,optimize=True)
    del t2_1_b

    t2_1_ab = t2[0][1][:]
    M_ab_a -=    0.5 *    lib.einsum('lmad,lbdm->ab',t2_1_ab, eris_ovVO,optimize=True)
    M_ab_b -=    0.5 *    lib.einsum('mlda,mdbl->ab',t2_1_ab, eris_ovVO,optimize=True)
    M_ab_a -=    0.5 *    lib.einsum('lmbd,ladm->ab',t2_1_ab, eris_ovVO,optimize=True)
    M_ab_b -=    0.5 *    lib.einsum('mldb,mdal->ab',t2_1_ab, eris_ovVO,optimize=True)
    del t2_1_ab

    cput0 = log.timer_debug1("Completed M_ab second-order terms ADC(2) calculation", *cput0)

    #Third-order terms
    if(method =='adc(3)'):

        t1_2_a, t1_2_b = t1[0]
        eris_oovv = eris.oovv
        eris_OOVV = eris.OOVV
        eris_OOvv = eris.OOvv
        eris_ooVV = eris.ooVV
        eris_ovvo = eris.ovvo
        eris_OVVO = eris.OVVO
        eris_OVvo = eris.OVvo
        eris_ovVO = eris.ovVO

        if isinstance(eris.ovvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc_a,chnk_size):
                eris_ovvv = dfadc.get_ovvv_spin_df(
                    adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
                k = eris_ovvv.shape[0]
                M_ab_a +=  lib.einsum('ld,ldab->ab',t1_2_a[a:a+k], eris_ovvv,optimize=True)
                M_ab_a -=  lib.einsum('ld,lbad->ab',t1_2_a[a:a+k], eris_ovvv,optimize=True)
                M_ab_a += lib.einsum('ld,ldab->ab',t1_2_a[a:a+k], eris_ovvv,optimize=True)
                M_ab_a -= lib.einsum('ld,ladb->ab',t1_2_a[a:a+k], eris_ovvv,optimize=True)
                del eris_ovvv
                a += k

        else :
            eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
            k = eris_ovvv.shape[0]
            M_ab_a +=  lib.einsum('ld,ldab->ab',t1_2_a, eris_ovvv,optimize=True)
            M_ab_a -=  lib.einsum('ld,lbad->ab',t1_2_a, eris_ovvv,optimize=True)
            M_ab_a += lib.einsum('ld,ldab->ab',t1_2_a, eris_ovvv,optimize=True)
            M_ab_a -= lib.einsum('ld,ladb->ab',t1_2_a, eris_ovvv,optimize=True)
            del eris_ovvv

        if isinstance(eris.OVvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc_b,chnk_size):
                eris_OVvv = dfadc.get_ovvv_spin_df(
                    adc, eris.LOV, eris.Lvv, p, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
                k = eris_OVvv.shape[0]
                M_ab_a +=  lib.einsum('ld,ldab->ab',t1_2_b[a:a+k], eris_OVvv,optimize=True)
                M_ab_a += lib.einsum('ld,ldab->ab',t1_2_b[a:a+k], eris_OVvv,optimize=True)
                del eris_OVvv
                a += k
        else :
            eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
            M_ab_a += lib.einsum('ld,ldab->ab',t1_2_b, eris_OVvv,optimize=True)
            M_ab_a += lib.einsum('ld,ldab->ab',t1_2_b, eris_OVvv,optimize=True)
            del eris_OVvv

        if isinstance(eris.OVVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc_b,chnk_size):
                eris_OVVV = dfadc.get_ovvv_spin_df(
                    adc, eris.LOV, eris.LVV, p, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
                k = eris_OVVV.shape[0]
                M_ab_b +=  lib.einsum('ld,ldab->ab',t1_2_b[a:a+k], eris_OVVV,optimize=True)
                M_ab_b -=  lib.einsum('ld,lbad->ab',t1_2_b[a:a+k], eris_OVVV,optimize=True)
                M_ab_b += lib.einsum('ld,ldab->ab',t1_2_b[a:a+k], eris_OVVV,optimize=True)
                M_ab_b -= lib.einsum('ld,ladb->ab',t1_2_b[a:a+k], eris_OVVV,optimize=True)
                del eris_OVVV
                a += k
        else :
            eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
            M_ab_b += lib.einsum('ld,ldab->ab',t1_2_b, eris_OVVV,optimize=True)
            M_ab_b -= lib.einsum('ld,lbad->ab',t1_2_b, eris_OVVV,optimize=True)
            M_ab_b += lib.einsum('ld,ldab->ab',t1_2_b, eris_OVVV,optimize=True)
            M_ab_b -= lib.einsum('ld,ladb->ab',t1_2_b, eris_OVVV,optimize=True)
            del eris_OVVV

        if isinstance(eris.ovVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc_a,chnk_size):
                eris_ovVV = dfadc.get_ovvv_spin_df(
                    adc, eris.Lov, eris.LVV, p, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
                k = eris_ovVV.shape[0]
                M_ab_b +=  lib.einsum('ld,ldab->ab',t1_2_a[a:a+k], eris_ovVV,optimize=True)
                M_ab_b += lib.einsum('ld,ldab->ab',t1_2_a[a:a+k], eris_ovVV,optimize=True)
                del eris_ovVV
                a += k
        else :
            eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
            k = eris_ovVV.shape[0]
            M_ab_b += lib.einsum('ld,ldab->ab',t1_2_a, eris_ovVV,optimize=True)
            M_ab_b += lib.einsum('ld,ldab->ab',t1_2_a, eris_ovVV,optimize=True)
            del eris_ovVV

        cput0 = log.timer_debug1("Completed M_ab ovvv ADC(3) calculation", *cput0)

        t2_2_a = t2[1][0][:]
        M_ab_a -= 0.5 * 0.5 *  lib.einsum('lmad,lbdm->ab',t2_2_a, eris_ovvo,optimize=True)
        M_ab_a += 0.5 * 0.5 *  lib.einsum('lmad,ldbm->ab',t2_2_a, eris_ovvo,optimize=True)
        M_ab_a -= 0.5 * 0.5 *  lib.einsum('lmbd,ladm->ab',t2_2_a,eris_ovvo,optimize=True)
        M_ab_a += 0.5 * 0.5 *  lib.einsum('lmbd,ldam->ab',t2_2_a, eris_ovvo,optimize=True)

        t2_2_b = t2[1][2][:]
        M_ab_b -= 0.5 * 0.5 *  lib.einsum('lmad,lbdm->ab',t2_2_b, eris_OVVO,optimize=True)
        M_ab_b += 0.5 * 0.5 *  lib.einsum('lmad,ldbm->ab',t2_2_b, eris_OVVO,optimize=True)
        M_ab_b -= 0.5 * 0.5 *  lib.einsum('lmbd,ladm->ab',t2_2_b, eris_OVVO,optimize=True)
        M_ab_b += 0.5 * 0.5 *  lib.einsum('lmbd,ldam->ab',t2_2_b, eris_OVVO,optimize=True)

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

            temp = np.zeros((nocc_a,nocc_a,nvir_a,nvir_a))
            temp[:,:,ab_ind_a[0],ab_ind_a[1]] =  adc.imds.t2_1_vvvv[0]
            temp[:,:,ab_ind_a[1],ab_ind_a[0]] = -adc.imds.t2_1_vvvv[0]

            M_ab_a -= 2 * 0.5 * 0.25*lib.einsum('mlaf,mlbf->ab',t2_1_a, temp, optimize=True)
            del temp

        else:

            temp_t2a_vvvv = np.zeros((nocc_a,nocc_a,nvir_a,nvir_a))
            temp_t2a_vvvv[:,:,ab_ind_a[0],ab_ind_a[1]] = adc.imds.t2_1_vvvv[0][:]
            temp_t2a_vvvv[:,:,ab_ind_a[1],ab_ind_a[0]] = -adc.imds.t2_1_vvvv[0][:]

            M_ab_a -= 2*0.5*0.25*lib.einsum('mlad,mlbd->ab',  temp_t2a_vvvv, t2_1_a, optimize=True)
            M_ab_a -= 2*0.5*0.25*lib.einsum('mlaf,mlbf->ab', t2_1_a, temp_t2a_vvvv, optimize=True)
            del temp_t2a_vvvv

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

        elif isinstance(eris.vvvv_p, type(None)):

            a = 0
            temp = np.zeros((nvir_a,nvir_a))
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            for p in range(0,nvir_a,chnk_size):
                vvvv = dfadc.get_vvvv_antisym_df(adc, eris.Lvv, p, chnk_size)
                k = vvvv.shape[0]

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
            del temp

            a = 0
            temp = np.zeros((nvir_b,nvir_b))
            for p in range(0,nvir_b,chnk_size):
                eris_VvVv = dfadc.get_vVvV_df(adc, eris.LVV, eris.Lvv, p, chnk_size)
                k = eris_VvVv.shape[0]

                temp[a:a+k] -= 0.5*lib.einsum('mldf,mled,aebf->ab',
                                              t2_1_a, t2_1_a, eris_VvVv, optimize=True)
                temp[a:a+k] += lib.einsum('mlfd,mled,aebf->ab',t2_1_ab,
                                          t2_1_ab, eris_VvVv, optimize=True)
                a += k
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
            temp = adc.imds.t2_1_vvvv[1]
            M_ab_a -= 0.5*lib.einsum('mlaf,mlbf->ab',t2_1_ab, temp, optimize=True)
            M_ab_b -= 0.5*lib.einsum('mlfa,mlfb->ab',t2_1_ab, temp, optimize=True)
            del temp
        else:
            t2_vVvV = adc.imds.t2_1_vvvv[1][:]

            M_ab_a -= 0.5 * lib.einsum('mlad,mlbd->ab', t2_vVvV, t2_1_ab, optimize=True)
            M_ab_b -= 0.5 * lib.einsum('mlda,mldb->ab', t2_vVvV, t2_1_ab, optimize=True)
            M_ab_a -= 0.5 * lib.einsum('mlaf,mlbf->ab',t2_1_ab, t2_vVvV, optimize=True)
            M_ab_b -= 0.5 * lib.einsum('mlfa,mlfb->ab',t2_1_ab, t2_vVvV, optimize=True)
            del t2_vVvV
        del t2_1_a

        if isinstance(eris.VVVV_p,np.ndarray):
            eris_VVVV = radc_ao2mo.unpack_eri_2(eris.VVVV_p, nvir_b)
            M_ab_b -= 0.5*0.25*lib.einsum('mlef,mlbd,adef->ab',t2_1_b,
                                          t2_1_b, eris_VVVV, optimize=True)
            M_ab_b -= 0.5*lib.einsum('mldf,mled,aebf->ab',t2_1_b, t2_1_b, eris_VVVV, optimize=True)
            M_ab_b += lib.einsum('mldf,mlde,aebf->ab',t2_1_ab, t2_1_ab, eris_VVVV, optimize=True)
            del eris_VVVV

            temp = np.zeros((nocc_b,nocc_b,nvir_b,nvir_b))
            temp[:,:,ab_ind_b[0],ab_ind_b[1]] =  adc.imds.t2_1_vvvv[2]
            temp[:,:,ab_ind_b[1],ab_ind_b[0]] = -adc.imds.t2_1_vvvv[2]
            M_ab_b -= 2 * 0.5 * 0.25*lib.einsum('mlaf,mlbf->ab',t2_1_b, temp, optimize=True)
            del temp
        else:

            temp_t2b_VVVV = np.zeros((nocc_b,nocc_b,nvir_b,nvir_b))
            temp_t2b_VVVV[:,:,ab_ind_b[0],ab_ind_b[1]] = adc.imds.t2_1_vvvv[2][:]
            temp_t2b_VVVV[:,:,ab_ind_b[1],ab_ind_b[0]] = -adc.imds.t2_1_vvvv[2][:]

            M_ab_b -= 2 * 0.5 * 0.25*lib.einsum('mlad,mlbd->ab',
                                                temp_t2b_VVVV, t2_1_b, optimize=True)
            M_ab_b -= 2 * 0.5 * 0.25*lib.einsum('mlaf,mlbf->ab',
                                                t2_1_b, temp_t2b_VVVV, optimize=True)
            del temp_t2b_VVVV

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

        elif isinstance(eris.vvvv_p, type(None)):

            a = 0
            temp = np.zeros((nvir_b,nvir_b))
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            for p in range(0,nvir_b,chnk_size):
                VVVV = dfadc.get_vvvv_antisym_df(adc, eris.LVV, p, chnk_size)
                k = VVVV.shape[0]

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
            del temp

            a = 0
            temp = np.zeros((nvir_a,nvir_a))
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            for p in range(0,nvir_a,chnk_size):
                eris_vVvV = dfadc.get_vVvV_df(adc, eris.Lvv, eris.LVV, p, chnk_size)
                k = eris_vVvV.shape[0]

                temp[a:a+k] -= 0.5*lib.einsum('mldf,mled,aebf->ab',
                                              t2_1_b, t2_1_b, eris_vVvV, optimize=True)
                temp[a:a+k] += lib.einsum('mldf,mlde,aebf->ab',t2_1_ab,
                                          t2_1_ab, eris_vVvV, optimize=True)
                a += k
            M_ab_a  += temp
            del temp

        del t2_1_ab, t2_1_b

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

        s = np.zeros((dim))

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
        if isinstance(eris.ovvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc_a,chnk_size):
                eris_ovvv = dfadc.get_ovvv_spin_df(
                    adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
                k = eris_ovvv.shape[0]
                s[s_a:f_a] += 0.5*lib.einsum('icab,ibc->a',eris_ovvv, r_aaa_[a:a+k], optimize=True)
                s[s_a:f_a] -= 0.5*lib.einsum('ibac,ibc->a',eris_ovvv, r_aaa_[a:a+k], optimize=True)
                temp[a:a+k] += lib.einsum('icab,a->ibc', eris_ovvv, r_a, optimize=True)
                temp[a:a+k] -= lib.einsum('ibac,a->ibc', eris_ovvv, r_a, optimize=True)
                del eris_ovvv
                a += k
        else :
            eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
            s[s_a:f_a] += 0.5*lib.einsum('icab,ibc->a',eris_ovvv, r_aaa_, optimize=True)
            s[s_a:f_a] -= 0.5*lib.einsum('ibac,ibc->a',eris_ovvv, r_aaa_, optimize=True)
            temp += lib.einsum('icab,a->ibc', eris_ovvv, r_a, optimize=True)
            temp -= lib.einsum('ibac,a->ibc', eris_ovvv, r_a, optimize=True)
            del eris_ovvv

        s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)
        del temp

        temp = np.zeros((nocc_b, nvir_a, nvir_b))
        if isinstance(eris.OVvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc_b,chnk_size):
                eris_OVvv = dfadc.get_ovvv_spin_df(
                    adc, eris.LOV, eris.Lvv, p, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
                k = eris_OVvv.shape[0]
                s[s_a:f_a] += lib.einsum('icab,ibc->a', eris_OVvv, r_bab[a:a+k], optimize=True)
                temp[a:a+k] += lib.einsum('icab,a->ibc', eris_OVvv, r_a, optimize=True)
                del eris_OVvv
                a += k
        else :
            eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
            s[s_a:f_a] += lib.einsum('icab,ibc->a', eris_OVvv, r_bab, optimize=True)
            temp += lib.einsum('icab,a->ibc', eris_OVvv, r_a, optimize=True)
            del eris_OVvv

        s[s_bab:f_bab] += temp.reshape(-1)
        del temp

        temp = np.zeros((nocc_b, nvir_b, nvir_b))
        if isinstance(eris.OVVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc_b,chnk_size):
                eris_OVVV = dfadc.get_ovvv_spin_df(
                    adc, eris.LOV, eris.LVV, p, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
                k = eris_OVVV.shape[0]
                s[s_b:f_b] += 0.5*lib.einsum('icab,ibc->a',eris_OVVV, r_bbb_[a:a+k], optimize=True)
                s[s_b:f_b] -= 0.5*lib.einsum('ibac,ibc->a',eris_OVVV, r_bbb_[a:a+k], optimize=True)
                temp[a:a+k] += lib.einsum('icab,a->ibc', eris_OVVV, r_b, optimize=True)
                temp[a:a+k] -= lib.einsum('ibac,a->ibc', eris_OVVV, r_b, optimize=True)
                del eris_OVVV
                a += k
        else :
            eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
            s[s_b:f_b] += 0.5*lib.einsum('icab,ibc->a',eris_OVVV, r_bbb_, optimize=True)
            s[s_b:f_b] -= 0.5*lib.einsum('ibac,ibc->a',eris_OVVV, r_bbb_, optimize=True)
            temp += lib.einsum('icab,a->ibc', eris_OVVV, r_b, optimize=True)
            temp -= lib.einsum('ibac,a->ibc', eris_OVVV, r_b, optimize=True)
            del eris_OVVV

        s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)
        del temp

        temp = np.zeros((nocc_a, nvir_b, nvir_a))
        if isinstance(eris.ovVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc_a,chnk_size):
                eris_ovVV = dfadc.get_ovvv_spin_df(
                    adc, eris.Lov, eris.LVV, p, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
                k = eris_ovVV.shape[0]
                s[s_b:f_b] += lib.einsum('icab,ibc->a', eris_ovVV, r_aba[a:a+k], optimize=True)
                temp[a:a+k] += lib.einsum('icab,a->ibc', eris_ovVV, r_b, optimize=True)
                del eris_ovVV
                a += k
        else :
            eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
            s[s_b:f_b] += lib.einsum('icab,ibc->a', eris_ovVV, r_aba, optimize=True)
            temp += lib.einsum('icab,a->ibc', eris_ovVV, r_b, optimize=True)
            del eris_ovVV
        s[s_aba:f_aba] += temp.reshape(-1)
        del temp

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
            s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

            s[s_bab:f_bab] -= 0.5*lib.einsum('jzyi,jzx->ixy',eris_ovVO,
                                             r_aaa_u,optimize=True).reshape(-1)
            s[s_bab:f_bab] -= 0.5*lib.einsum('jiyz,jxz->ixy',eris_OOVV,
                                             r_bab,optimize=True).reshape(-1)
            s[s_bab:f_bab] += 0.5*lib.einsum('jzyi,jxz->ixy',eris_OVVO,
                                             r_bab,optimize=True).reshape(-1)

            temp = 0.5*lib.einsum('jiyz,jzx->ixy',eris_OOVV,r_bbb_u,optimize=True)
            temp -= 0.5*lib.einsum('jzyi,jzx->ixy',eris_OVVO,r_bbb_u,optimize=True)
            temp +=0.5* lib.einsum('jzyi,jxz->ixy',eris_ovVO,r_aba,optimize=True)
            s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

            s[s_aba:f_aba] -= 0.5*lib.einsum('jiyz,jxz->ixy',eris_oovv,
                                             r_aba,optimize=True).reshape(-1)
            s[s_aba:f_aba] += 0.5*lib.einsum('jzyi,jxz->ixy',eris_ovvo,
                                             r_aba,optimize=True).reshape(-1)
            s[s_aba:f_aba] -= 0.5*lib.einsum('jzyi,jzx->ixy',eris_OVvo,
                                             r_bbb_u,optimize=True).reshape(-1)

            temp = -0.5*lib.einsum('jixz,jzy->ixy',eris_oovv,r_aaa_u,optimize=True)
            temp += 0.5*lib.einsum('jzxi,jzy->ixy',eris_ovvo,r_aaa_u,optimize=True)
            temp -= 0.5*lib.einsum('jzxi,jyz->ixy',eris_OVvo,r_bab,optimize=True)
            s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

            s[s_bab:f_bab] -=  0.5*lib.einsum('jixz,jzy->ixy',
                                              eris_OOvv,r_bab,optimize=True).reshape(-1)

            temp = -0.5*lib.einsum('jixz,jzy->ixy',eris_OOVV,r_bbb_u,optimize=True)
            temp += 0.5*lib.einsum('jzxi,jzy->ixy',eris_OVVO,r_bbb_u,optimize=True)
            temp -= 0.5*lib.einsum('jzxi,jyz->ixy',eris_ovVO,r_aba,optimize=True)
            s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

            s[s_aba:f_aba] -= 0.5*lib.einsum('jixz,jzy->ixy',eris_ooVV,
                                             r_aba,optimize=True).reshape(-1)

            temp = 0.5*lib.einsum('jixw,jyw->ixy',eris_oovv,r_aaa_u,optimize=True)
            temp -= 0.5*lib.einsum('jwxi,jyw->ixy',eris_ovvo,r_aaa_u,optimize=True)
            temp -= 0.5*lib.einsum('jwxi,jyw->ixy',eris_OVvo,r_bab,optimize=True)
            s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

            s[s_bab:f_bab] -= 0.5*lib.einsum('jixw,jwy->ixy',eris_OOvv,
                                             r_bab,optimize=True).reshape(-1)

            temp = 0.5*lib.einsum('jixw,jyw->ixy',eris_OOVV,r_bbb_u,optimize=True)
            temp -= 0.5*lib.einsum('jwxi,jyw->ixy',eris_OVVO,r_bbb_u,optimize=True)
            temp -= 0.5*lib.einsum('jwxi,jyw->ixy',eris_ovVO,r_aba,optimize=True)
            s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

            s[s_aba:f_aba] -= 0.5*lib.einsum('jixw,jwy->ixy',eris_ooVV,
                                             r_aba,optimize=True).reshape(-1)

            temp = -0.5*lib.einsum('jiyw,jxw->ixy',eris_oovv,r_aaa_u,optimize=True)
            temp += 0.5*lib.einsum('jwyi,jxw->ixy',eris_ovvo,r_aaa_u,optimize=True)
            temp += 0.5*lib.einsum('jwyi,jxw->ixy',eris_OVvo,r_bab,optimize=True)
            s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

            s[s_bab:f_bab] -= 0.5*lib.einsum('jiyw,jxw->ixy',eris_OOVV,
                                             r_bab,optimize=True).reshape(-1)
            s[s_bab:f_bab] += 0.5*lib.einsum('jwyi,jxw->ixy',eris_OVVO,
                                             r_bab,optimize=True).reshape(-1)
            s[s_bab:f_bab] += 0.5*lib.einsum('jwyi,jxw->ixy',eris_ovVO,
                                             r_aaa_u,optimize=True).reshape(-1)

            s[s_aba:f_aba] -= 0.5*lib.einsum('jiyw,jxw->ixy',eris_oovv,
                                             r_aba,optimize=True).reshape(-1)
            s[s_aba:f_aba] += 0.5*lib.einsum('jwyi,jxw->ixy',eris_ovvo,
                                             r_aba,optimize=True).reshape(-1)
            s[s_aba:f_aba] += 0.5*lib.einsum('jwyi,jxw->ixy',eris_OVvo,
                                             r_bbb_u,optimize=True).reshape(-1)

            temp = -0.5*lib.einsum('jiyw,jxw->ixy',eris_OOVV,r_bbb_u,optimize=True)
            temp += 0.5*lib.einsum('jwyi,jxw->ixy',eris_OVVO,r_bbb_u,optimize=True)
            temp += 0.5*lib.einsum('jwyi,jxw->ixy',eris_ovVO,r_aba,optimize=True)
            s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

        if (method == "adc(3)"):

            #print("Calculating additional terms for adc(3)")

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
            s[s_a:f_a] += lib.einsum('lmj,lamj->a',temp, eris_ovoo, optimize=True)
            s[s_a:f_a] -= lib.einsum('lmj,malj->a',temp, eris_ovoo, optimize=True)
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

            temp_s_a_1 = np.zeros_like(r_bab)
            temp_s_a_1 = -lib.einsum('jlzd,jwz->lwd',t2_1_a,r_aaa_u,optimize=True)
            temp_s_a_1 += -lib.einsum('ljdz,jwz->lwd',t2_1_ab,r_bab,optimize=True)

            temp_1_1 = np.zeros((nocc_a,nvir_a,nvir_a))
            temp_1_2 = np.zeros((nocc_a,nvir_a,nvir_a))
            if isinstance(eris.ovvv, type(None)):
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                a = 0
                for p in range(0,nocc_a,chnk_size):
                    eris_ovvv = dfadc.get_ovvv_spin_df(
                        adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
                    k = eris_ovvv.shape[0]
                    s[s_a:f_a] += 0.5*lib.einsum('lzd,ldza->a',
                                                 temp_s_a[a:a+k],eris_ovvv,optimize=True)
                    s[s_a:f_a] -= 0.5*lib.einsum('lzd,lazd->a',
                                                 temp_s_a[a:a+k],eris_ovvv,optimize=True)
                    s[s_a:f_a] -= 0.5*lib.einsum('lwd,ldwa->a',
                                                 temp_s_a_1[a:a+k],eris_ovvv,optimize=True)
                    s[s_a:f_a] += 0.5*lib.einsum('lwd,lawd->a',
                                                 temp_s_a_1[a:a+k],eris_ovvv,optimize=True)

                    temp_1_1[a:a+k] += lib.einsum('ldxb,b->lxd', eris_ovvv,r_a,optimize=True)
                    temp_1_1[a:a+k] -= lib.einsum('lbxd,b->lxd', eris_ovvv,r_a,optimize=True)

                    temp_1_2[a:a+k] += lib.einsum('ldyb,b->lyd', eris_ovvv,r_a,optimize=True)
                    temp_1_2[a:a+k] -= lib.einsum('lbyd,b->lyd', eris_ovvv,r_a,optimize=True)
                    del eris_ovvv
                    a += k
            else :
                eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
                s[s_a:f_a] += 0.5*lib.einsum('lzd,ldza->a',temp_s_a,eris_ovvv,optimize=True)
                s[s_a:f_a] -= 0.5*lib.einsum('lzd,lazd->a',temp_s_a,eris_ovvv,optimize=True)
                s[s_a:f_a] -= 0.5*lib.einsum('lwd,ldwa->a',temp_s_a_1,eris_ovvv,optimize=True)
                s[s_a:f_a] += 0.5*lib.einsum('lwd,lawd->a',temp_s_a_1,eris_ovvv,optimize=True)

                temp_1_1 += lib.einsum('ldxb,b->lxd', eris_ovvv,r_a,optimize=True)
                temp_1_1 -= lib.einsum('lbxd,b->lxd', eris_ovvv,r_a,optimize=True)

                temp_1_2 += lib.einsum('ldyb,b->lyd', eris_ovvv,r_a,optimize=True)
                temp_1_2 -= lib.einsum('lbyd,b->lyd', eris_ovvv,r_a,optimize=True)
                del eris_ovvv

            del temp_s_a
            del temp_s_a_1

            r_bab_t = r_bab.reshape(nocc_b*nvir_a,-1)
            temp = np.ascontiguousarray(t2_1_ab.transpose(
                0,3,1,2)).reshape(nocc_a*nvir_b,nocc_b*nvir_a)
            temp_2 = np.dot(temp,r_bab_t).reshape(nocc_a,nvir_b,nvir_b)
            del temp
            temp_2 = np.ascontiguousarray(temp_2.transpose(0,2,1))
            temp_2_new = -lib.einsum('ljzd,jzw->lwd',t2_1_ab,r_bab,optimize=True)


            temp_new_1 = np.zeros_like(r_aba)
            temp_new_1 = lib.einsum('ljdw,jzw->ldz',t2_1_ab,r_bbb_u,optimize=True)
            temp_new_1 += lib.einsum('jlwd,jzw->ldz',t2_1_a,r_aba,optimize=True)

            temp_new_2 = np.zeros_like(r_bab)
            temp_new_2 = -lib.einsum('ljdz,jwz->lwd',t2_1_ab,r_bbb_u,optimize=True)
            temp_new_2 += -lib.einsum('jlzd,jwz->lwd',t2_1_a,r_aba,optimize=True)

            temp_2_3 = np.zeros((nocc_a,nvir_b,nvir_a))
            temp_2_4 = np.zeros((nocc_a,nvir_b,nvir_a))

            temp = np.zeros((nocc_a,nvir_b,nvir_b))
            if isinstance(eris.ovVV, type(None)):
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                a = 0
                for p in range(0,nocc_a,chnk_size):
                    eris_ovVV = dfadc.get_ovvv_spin_df(
                        adc, eris.Lov, eris.LVV, p, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
                    k = eris_ovVV.shape[0]
                    s[s_a:f_a] -= 0.5*lib.einsum('lzd,lazd->a',
                                                 temp_2[a:a+k],eris_ovVV,optimize=True)

                    s[s_a:f_a] += 0.5*lib.einsum('lwd,lawd->a',
                                                 temp_2_new[a:a+k],eris_ovVV,optimize=True)

                    s[s_b:f_b] += 0.5*np.einsum('ldz,ldza->a',temp_new_1[a:a+k],eris_ovVV)
                    s[s_b:f_b] -= 0.5*np.einsum('lwd,ldwa->a',temp_new_2[a:a+k],eris_ovVV)

                    eris_ovVV = eris_ovVV.reshape(-1, nvir_a, nvir_b, nvir_b)

                    temp_2_3[a:a+k] += lib.einsum('ldxb,b->lxd', eris_ovVV,r_b,optimize=True)
                    temp_2_4[a:a+k] += lib.einsum('ldyb,b->lyd', eris_ovVV,r_b,optimize=True)

                    temp[a:a+k]  -= lib.einsum('lbyd,b->lyd',eris_ovVV,r_a,optimize=True)
                    del eris_ovVV
                    a += k
            else :
                eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
                s[s_a:f_a] -= 0.5*lib.einsum('lzd,lazd->a',temp_2,eris_ovVV,optimize=True)

                s[s_a:f_a] += 0.5*lib.einsum('lwd,lawd->a',temp_2_new,eris_ovVV,optimize=True)

                s[s_b:f_b] += 0.5*np.einsum('ldz,ldza->a',temp_new_1,eris_ovVV)
                s[s_b:f_b] -= 0.5*np.einsum('lwd,ldwa->a',temp_new_2,eris_ovVV)

                eris_ovVV = eris_ovVV.reshape(-1, nvir_a, nvir_b, nvir_b)

                temp_2_3 += lib.einsum('ldxb,b->lxd', eris_ovVV,r_b,optimize=True)
                temp_2_4 += lib.einsum('ldyb,b->lyd', eris_ovVV,r_b,optimize=True)

                temp  -= lib.einsum('lbyd,b->lyd',eris_ovVV,r_a,optimize=True)
                del eris_ovVV

            temp = -lib.einsum('lyd,lixd->ixy',temp,t2_1_ab,optimize=True)
            s[s_bab:f_bab] -= temp.reshape(-1)
            del temp
            del temp_2
            del temp_2_new
            del temp_new_1
            del temp_new_2

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

            del t2_1_a

            t2_1_b = adc.t2[0][2][:]

            t2_1_b_t = t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]]
            r_bbb = r_bbb.reshape(nocc_b,-1)
            temp = 0.5*lib.einsum('lmp,jp->lmj',t2_1_b_t,r_bbb)
            del t2_1_b_t
            s[s_b:f_b] += lib.einsum('lmj,lamj->a',temp, eris_OVOO, optimize=True)
            s[s_b:f_b] -= lib.einsum('lmj,malj->a',temp, eris_OVOO, optimize=True)
            del temp

            temp_s_b = np.zeros_like(r_aba)
            temp_s_b = lib.einsum('jlwd,jzw->lzd',t2_1_b,r_bbb_u,optimize=True)
            temp_s_b += lib.einsum('jlwd,jzw->lzd',t2_1_ab,r_aba,optimize=True)

            temp_s_b_1 = np.zeros_like(r_aba)
            temp_s_b_1 = -lib.einsum('jlzd,jwz->lwd',t2_1_b,r_bbb_u,optimize=True)
            temp_s_b_1 += -lib.einsum('jlzd,jwz->lwd',t2_1_ab,r_aba,optimize=True)

            temp_1_3 = np.zeros((nocc_b,nvir_b,nvir_b))
            temp_1_4 = np.zeros((nocc_b,nvir_b,nvir_b))

            if isinstance(eris.OVVV, type(None)):
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                a = 0
                for p in range(0,nocc_b,chnk_size):
                    eris_OVVV = dfadc.get_ovvv_spin_df(
                        adc, eris.LOV, eris.LVV, p, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
                    k = eris_OVVV.shape[0]
                    s[s_b:f_b] += 0.5*lib.einsum('lzd,ldza->a',
                                                 temp_s_b[a:a+k],eris_OVVV,optimize=True)
                    s[s_b:f_b] -= 0.5*lib.einsum('lzd,lazd->a',
                                                 temp_s_b[a:a+k],eris_OVVV,optimize=True)
                    s[s_b:f_b] -= 0.5*lib.einsum('lwd,ldwa->a',
                                                 temp_s_b_1[a:a+k],eris_OVVV,optimize=True)
                    s[s_b:f_b] += 0.5*lib.einsum('lwd,lawd->a',
                                                 temp_s_b_1[a:a+k],eris_OVVV,optimize=True)

                    temp_1_3[a:a+k] += lib.einsum('ldxb,b->lxd', eris_OVVV,r_b,optimize=True)
                    temp_1_3[a:a+k] -= lib.einsum('lbxd,b->lxd', eris_OVVV,r_b,optimize=True)

                    temp_1_4[a:a+k] += lib.einsum('ldyb,b->lyd', eris_OVVV,r_b,optimize=True)
                    temp_1_4[a:a+k] -= lib.einsum('lbyd,b->lyd', eris_OVVV,r_b,optimize=True)
                    del eris_OVVV
                    a += k
            else :
                eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
                s[s_b:f_b] += 0.5*lib.einsum('lzd,ldza->a',temp_s_b,eris_OVVV,optimize=True)
                s[s_b:f_b] -= 0.5*lib.einsum('lzd,lazd->a',temp_s_b,eris_OVVV,optimize=True)
                s[s_b:f_b] -= 0.5*lib.einsum('lwd,ldwa->a',temp_s_b_1,eris_OVVV,optimize=True)
                s[s_b:f_b] += 0.5*lib.einsum('lwd,lawd->a',temp_s_b_1,eris_OVVV,optimize=True)

                temp_1_3 += lib.einsum('ldxb,b->lxd', eris_OVVV,r_b,optimize=True)
                temp_1_3 -= lib.einsum('lbxd,b->lxd', eris_OVVV,r_b,optimize=True)

                temp_1_4 += lib.einsum('ldyb,b->lyd', eris_OVVV,r_b,optimize=True)
                temp_1_4 -= lib.einsum('lbyd,b->lyd', eris_OVVV,r_b,optimize=True)
                del eris_OVVV

            del temp_s_b
            del temp_s_b_1

            temp_1 = np.zeros_like(r_bab)
            temp_1= lib.einsum('jlwd,jzw->lzd',t2_1_ab,r_aaa_u,optimize=True)
            temp_1 += lib.einsum('jlwd,jzw->lzd',t2_1_b,r_bab,optimize=True)
            temp_2 = lib.einsum('jldw,jwz->lzd',t2_1_ab,r_aba,optimize=True)
            temp_1_new = np.zeros_like(r_bab)
            temp_1_new = -lib.einsum('jlzd,jwz->lwd',t2_1_ab,r_aaa_u,optimize=True)
            temp_1_new += -lib.einsum('jlzd,jwz->lwd',t2_1_b,r_bab,optimize=True)
            temp_2_new = -lib.einsum('jldz,jzw->lwd',t2_1_ab,r_aba,optimize=True)
            temp_2_1 = np.zeros((nocc_b,nvir_a,nvir_b))
            temp_2_2 = np.zeros((nocc_b,nvir_a,nvir_b))
            temp = np.zeros((nocc_b,nvir_a,nvir_a))

            if isinstance(eris.OVvv, type(None)):
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                a = 0
                for p in range(0,nocc_b,chnk_size):
                    eris_OVvv = dfadc.get_ovvv_spin_df(
                        adc, eris.LOV, eris.Lvv, p, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
                    k = eris_OVvv.shape[0]
                    s[s_a:f_a] += 0.5*lib.einsum('lzd,ldza->a',
                                                 temp_1[a:a+k],eris_OVvv,optimize=True)

                    s[s_b:f_b] -= 0.5*lib.einsum('lzd,lazd->a',
                                                 temp_2[a:a+k],eris_OVvv,optimize=True)

                    s[s_a:f_a] -= 0.5*lib.einsum('lwd,ldwa->a',
                                                 temp_1_new[a:a+k],eris_OVvv,optimize=True)

                    s[s_b:f_b] += 0.5*lib.einsum('lwd,lawd->a',
                                                 temp_2_new[a:a+k],eris_OVvv,optimize=True)

                    temp_2_1[a:a+k] += lib.einsum('ldxb,b->lxd', eris_OVvv,r_a,optimize=True)
                    temp_2_2[a:a+k] += lib.einsum('ldyb,b->lyd', eris_OVvv,r_a,optimize=True)

                    temp[a:a+k]  -= lib.einsum('lbyd,b->lyd',eris_OVvv,r_b,optimize=True)
                    del eris_OVvv
                    a += k
            else :
                eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
                s[s_a:f_a] += 0.5*lib.einsum('lzd,ldza->a',temp_1,eris_OVvv,optimize=True)

                s[s_b:f_b] -= 0.5*lib.einsum('lzd,lazd->a',temp_2,eris_OVvv,optimize=True)

                s[s_a:f_a] -= 0.5*lib.einsum('lwd,ldwa->a',temp_1_new,eris_OVvv,optimize=True)

                s[s_b:f_b] += 0.5*lib.einsum('lwd,lawd->a',temp_2_new,eris_OVvv,optimize=True)

                temp_2_1 += lib.einsum('ldxb,b->lxd', eris_OVvv,r_a,optimize=True)
                temp_2_2 += lib.einsum('ldyb,b->lyd', eris_OVvv,r_a,optimize=True)

                temp  -= lib.einsum('lbyd,b->lyd',eris_OVvv,r_b,optimize=True)
                del eris_OVvv

            temp_new = -lib.einsum('lyd,ildx->ixy',temp,t2_1_ab,optimize=True)
            s[s_aba:f_aba] -= temp_new.reshape(-1)
            del temp
            del temp_new
            del temp_1
            del temp_1_new
            del temp_2
            del temp_2_new

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
            del t2_1_b

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

            del t2_1_ab

        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)
        return s

        del temp_2_1
        del temp_1_3
        del temp_1_4
        del temp_1_1
        del temp_1_2
        del temp_2_3

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

    cput0 = log.timer_debug1("completed spec vactor calc in ADC(3) calculation", *cput0)
    return (T_a, T_b)


def get_trans_moments_orbital(adc, orb, spin="alpha"):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
        t1_2_a, t1_2_b = adc.t1[0]

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

    T = np.zeros((dim))

######## spin = alpha  ############################################

    if spin == "alpha":
        pass  # placehold

######## ADC(2) part  ############################################

        t2_1_a = adc.t2[0][0][:]
        t2_1_ab = adc.t2[0][1][:]
        if orb < nocc_a:

            if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
                T[s_a:f_a] = -t1_2_a[orb,:]

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
######## ADC(3) 2p-1h  part  ############################################

        if (adc.method == "adc(2)-x" and adc.approx_trans_moments is False) or (adc.method == "adc(3)"):

            t2_2_a = adc.t2[1][0][:]
            t2_2_ab = adc.t2[1][1][:]

            if orb < nocc_a:

                t2_2_t = t2_2_a[:,:,ab_ind_a[0],ab_ind_a[1]].copy()
                t2_2_ab_t = -t2_2_ab.transpose(1,0,2,3)

                T[s_aaa:f_aaa] += t2_2_t[:,orb,:].reshape(-1)
                T[s_bab:f_bab] += t2_2_ab_t[:,orb,:,:].reshape(-1)

######### ADC(3) 1p part  ############################################

        if (method=='adc(3)'):

            if (adc.approx_trans_moments is False):
                t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_a:

                T[s_a:f_a] += 0.5*lib.einsum('kac,ck->a',t2_1_a[:,orb,:,:], t1_2_a.T,optimize=True)
                T[s_a:f_a] -= 0.5*lib.einsum('kac,ck->a',t2_1_ab[orb,:,:,:], t1_2_b.T,optimize=True)

                if (adc.approx_trans_moments is False):
                    T[s_a:f_a] -= t1_3_a[orb,:]

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

                del t2_2_a
                del t2_2_ab

        del t2_1_a
        del t2_1_ab

######### spin = beta  ############################################

    else:
        pass  # placehold

        t2_1_b = adc.t2[0][2][:]
        t2_1_ab = adc.t2[0][1][:]
        if orb < nocc_b:

            if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
                T[s_b:f_b] = -t1_2_b[orb,:]

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

######### ADC(3) 2p-1h part  ############################################

        if (adc.method == "adc(2)-x" and adc.approx_trans_moments is False) or (adc.method == "adc(3)"):

            t2_2_ab = adc.t2[1][1][:]
            t2_2_b = adc.t2[1][2][:]

            if orb < nocc_b:

                t2_2_t = t2_2_b[:,:,ab_ind_b[0],ab_ind_b[1]].copy()
                t2_2_ab_t = -t2_2_ab.transpose(0,1,3,2)

                T[s_bbb:f_bbb] += t2_2_t[:,orb,:].reshape(-1)
                T[s_aba:f_aba] += t2_2_ab_t[:,orb,:,:].reshape(-1)

######### ADC(2) 1p part  ############################################

        if(method=='adc(3)'):

            if (adc.approx_trans_moments is False):
                t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_b:

                T[s_b:f_b] += 0.5*lib.einsum('kac,ck->a',t2_1_b[:,orb,:,:], t1_2_b.T,optimize=True)
                T[s_b:f_b] -= 0.5*lib.einsum('kca,ck->a',t2_1_ab[:,orb,:,:], t1_2_a.T,optimize=True)

                if (adc.approx_trans_moments is False):
                    T[s_b:f_b] -= t1_3_b[orb,:]

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

        logger.info(adc,'%s | root %d | norm(1p)  = %6.4f | norm(1h2p) = %6.4f ',
                    adc.method ,I, U1dotU1, U2dotU2)

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

        logger.info(adc, "\n*************************************************************\n")


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

            logger.info(adc, '%s | root %d %s\n', adc.method, i, spin)
            logger.info(adc, "     HF MO     Spec. Contribution     Orbital symmetry")
            logger.info(adc, "-----------------------------------------------------------")

            for c in range(index_mo.shape[0]):
                logger.info(adc, '     %3.d          %10.8f                %s',
                            index_mo[c], spec_Contribution[c], sym[c])

            logger.info(adc, '\nPartial spec. factor sum = %10.8f', np.sum(spec_Contribution))
            logger.info(adc, "\n*************************************************************\n")


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
            Convergence threshold for Davidson iterations.  Default is 1e-12.
        max_cycle : int
            Number of Davidson iterations.  Default is 50.
        max_space : int
            Space size to hold trial vectors for Davidson iterative diagonalization.  Default is 12.

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested. Default value is 1.

            >>> myadc = adc.UADC(mf).run()
            >>> myadcea = adc.UADC(myadc).run()

    Saved results

        e_ea : float or list of floats
            EA energy (eigenvalue). For nroots = 1, it is a single float
            number. If nroots > 1, it is a list of floats for the lowest
            nroots eigenvalues.
        v_ip : array
            Eigenvectors for each EA transition.
        p_ea : float
            Spectroscopic amplitudes for each EA transition.
    '''

    def __init__(self, adc):
        self.verbose = adc.verbose
        self.stdout = adc.stdout
        self.max_memory = adc.max_memory
        self.max_space = adc.max_space
        self.max_cycle = adc.max_cycle
        self.conv_tol  = adc.conv_tol
        self.tol_residual  = adc.tol_residual
        self.t1 = adc.t1
        self.t2 = adc.t2
        self.imds = adc.imds
        self.e_corr = adc.e_corr
        self.method = adc.method
        self.method_type = adc.method_type
        self._scf = adc._scf
        self._nocc = adc._nocc
        self._nvir = adc._nvir
        self.nocc_a = adc._nocc[0]
        self.nocc_b = adc._nocc[1]
        self.nvir_a = adc._nvir[0]
        self.nvir_b = adc._nvir[1]
        self.mo_coeff = adc.mo_coeff
        self.mo_energy_a = adc.mo_energy_a
        self.mo_energy_b = adc.mo_energy_b
        self.nmo_a = adc._nmo[0]
        self.nmo_b = adc._nmo[1]
        self.mol = adc.mol
        self.transform_integrals = adc.transform_integrals
        self.with_df = adc.with_df
        self.spec_factor_print_tol = adc.spec_factor_print_tol
        self.evec_print_tol = adc.evec_print_tol

        self.compute_properties = adc.compute_properties
        self.approx_trans_moments = adc.approx_trans_moments
        self.E = adc.E
        self.U = adc.U
        self.P = adc.P
        self.X = adc.X

        keys = set(('tol_residual','conv_tol', 'e_corr', 'method',
                    'method_type', 'mo_coeff', 'mo_energy_b', 'max_memory',
                    't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)

    kernel = uadc.kernel
    get_imds = get_imds
    matvec = matvec
    get_diag = get_diag
    get_trans_moments = get_trans_moments
    analyze_spec_factor = analyze_spec_factor
    get_properties = get_properties
    analyze = analyze
    compute_dyson_mo = compute_dyson_mo
    analyze_eigenvector = analyze_eigenvector

    def get_init_guess(self, nroots=1, diag=None, ascending=True):
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
        for p in range(0,nvir,chnk_size):
            vvvv = dfadc.get_vvvv_antisym_df(myadc, vvvv_d, p, chnk_size)
            k = vvvv.shape[0]
            vvvv = vvvv.reshape(-1,nv_pair)
            r2_vvvv[:,a:a+k] = np.dot(r2,vvvv.T).reshape(nocc,-1,nvir)
            del vvvv
            a += k
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
        for p in range(0,nvir_1,chnk_size):
            vvvv = dfadc.get_vVvV_df(myadc, Lvv, LVV, p, chnk_size)
            k = vvvv.shape[0]
            vvvv = vvvv.reshape(-1,nvir_1*nvir_2)
            r2_vvvv[:,a:a+k] = np.dot(r2,vvvv.T).reshape(nocc_1,-1,nvir_2)
            del vvvv
            a += k
    else:
        raise Exception("Unknown vvvv type")

    return r2_vvvv
