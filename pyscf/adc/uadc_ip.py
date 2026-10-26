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

    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)

    if eris is None:
        eris = adc.transform_integrals()

    eris_ovvo = eris.ovvo
    eris_OVVO = eris.OVVO
    eris_ovVO = eris.ovVO
    eris_OVvo = eris.OVvo
    eris_ovoo = eris.ovoo
    eris_OVOO = eris.OVOO
    eris_ovOO = eris.ovOO
    eris_OVoo = eris.OVoo

    t1_1_a = t1_1_b = None
    if t1[2][0] is not None:
        t1_1_a = t1[2][0]
        t1_1_b = t1[2][1]
        f_ov_a, f_ov_b = adc.f_ov
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

    # i-j block
    # Zeroth-order terms

    M_ij_a = lib.einsum('ij,j->ij', idn_occ_a ,e_occ_a)
    M_ij_b = lib.einsum('ij,j->ij', idn_occ_b ,e_occ_b)

    # Second-order terms

    t2_1_a = t2[0][0][:]
    M_ij_a += 0.5 *  lib.einsum('ilde,jdel->ij',t2_1_a, eris_ovvo, optimize=True)
    M_ij_a += 0.5 *  lib.einsum('jlde,idel->ij',t2_1_a, eris_ovvo, optimize=True)

    t2_1_b = t2[0][2][:]
    M_ij_b += 0.5 *  lib.einsum('ilde,jdel->ij',t2_1_b, eris_OVVO, optimize=True)
    M_ij_b += 0.5 *  lib.einsum('jlde,idel->ij',t2_1_b, eris_OVVO, optimize=True)

    t2_1_ab = t2[0][1][:]
    M_ij_a += 0.5 * lib.einsum('ilde,jdel->ij',t2_1_ab, eris_ovVO, optimize=True)
    M_ij_b += 0.5 * lib.einsum('lied,ledj->ij',t2_1_ab, eris_ovVO, optimize=True)
    M_ij_a += 0.5 * lib.einsum('jlde,idel->ij',t2_1_ab, eris_ovVO, optimize=True)
    M_ij_b += 0.5 * lib.einsum('ljed,ledi->ij',t2_1_ab, eris_ovVO, optimize=True)
    del t2_1_ab

    if t1_1_a is not None:
        temp = lib.einsum('Aa,Ba->AB', f_ov_a, t1_1_a, optimize=True)
        M_ij_a += temp + temp.T
        temp = lib.einsum('ia,iaAB->AB', t1_1_a, eris_ovoo, optimize=True)
        M_ij_a += temp + temp.T
        temp = lib.einsum('ia,BaiA->AB', t1_1_a, eris_ovoo, optimize=True)
        M_ij_a -= temp + temp.T
        temp = lib.einsum('ia,iaAB->AB', t1_1_b, eris_OVoo, optimize=True)
        M_ij_a += temp + temp.T
        temp = lib.einsum('A,Aa,Ba->AB', e_occ_a, t1_1_a, t1_1_a, optimize=True)
        M_ij_a -= 1/2 * (temp + temp.T)
        M_ij_a += lib.einsum('a,Aa,Ba->AB', e_vir_a, t1_1_a, t1_1_a, optimize=True)
        temp = lib.einsum('Aa,Ba->AB', f_ov_b, t1_1_b, optimize=True)
        M_ij_b += temp + temp.T
        temp = lib.einsum('ia,iaAB->AB', t1_1_a, eris_ovOO, optimize=True)
        M_ij_b += temp + temp.T
        temp = lib.einsum('ia,iaAB->AB', t1_1_b, eris_OVOO, optimize=True)
        M_ij_b += temp + temp.T
        temp = lib.einsum('ia,BaiA->AB', t1_1_b, eris_OVOO, optimize=True)
        M_ij_b -= temp + temp.T
        temp = lib.einsum('A,Aa,Ba->AB', e_occ_b, t1_1_b, t1_1_b, optimize=True)
        M_ij_b -= 1/2 * (temp + temp.T)
        M_ij_b += lib.einsum('a,Aa,Ba->AB', e_vir_b, t1_1_b, t1_1_b, optimize=True)

    # Third-order terms

    if (method == "adc(3)"):
        t1_2_a, t1_2_b = t1[0]

        eris_oovv = eris.oovv
        eris_OOVV = eris.OOVV
        eris_ooVV = eris.ooVV
        eris_OOvv = eris.OOvv
        eris_ovvo = eris.ovvo
        eris_OVVO = eris.OVVO
        eris_ovVO = eris.ovVO
        eris_OVvo = eris.OVvo
        eris_oooo = eris.oooo
        eris_OOOO = eris.OOOO
        eris_ooOO = eris.ooOO

        if t1_1_a is None:
            M_ij_a += 2 * lib.einsum('ld,ldji->ij',t1_2_a, eris_ovoo, optimize=True)
            temp = lib.einsum('ld,jdli->ij',t1_2_a,eris_ovoo, optimize=True)
            M_ij_a -= temp + temp.T
            M_ij_a += 2 * lib.einsum('ld,ldji->ij',t1_2_b, eris_OVoo, optimize=True)

            M_ij_b += 2 * lib.einsum('ld,ldji->ij',t1_2_b, eris_OVOO, optimize=True)
            temp = lib.einsum('ld,jdli->ij',t1_2_b,eris_OVOO, optimize=True)
            M_ij_b -= temp + temp.T
            del temp
            M_ij_b += 2 * lib.einsum('ld,ldji->ij',t1_2_a, eris_ovOO, optimize=True)

            t2_1_a = t2[0][0][:]
            t2_2_a = t2[1][0][:]

            M_ij_a += 0.5 * lib.einsum('ilde,jdel->ij',t2_2_a, eris_ovvo, optimize=True)

            M_ij_a += 0.5 * lib.einsum('jlde,ledi->ij',t2_2_a, eris_ovvo, optimize=True)

            t2_1_b = t2[0][2][:]
            t2_2_b = t2[1][2][:]

            M_ij_b += 0.5 * lib.einsum('ilde,jdel->ij',t2_2_b, eris_OVVO, optimize=True)
            M_ij_b += 0.5 * lib.einsum('jlde,ledi->ij',t2_2_b, eris_OVVO, optimize=True)

            t2_1_ab = t2[0][1][:]
            t2_2_ab = t2[1][1][:]

            M_ij_a += 0.5 * lib.einsum('ilde,jdel->ij',t2_2_ab, eris_ovVO, optimize=True)
            M_ij_b += 0.5 * lib.einsum('lied,ledj->ij',t2_2_ab, eris_ovVO, optimize=True)
            M_ij_a += 0.5 * lib.einsum('jlde,ledi->ij',t2_2_ab, eris_OVvo, optimize=True)
            M_ij_b += 0.5 * lib.einsum('ljed,ledi->ij',t2_2_ab, eris_ovVO, optimize=True)

            t2_1_a = t2[0][0][:]
            M_ij_t = lib.einsum('lmde,jldf->mejf', t2_1_a, t2_1_a, optimize=True)
            M_ij_a -= 0.5 * lib.einsum('mejf,mefi->ij',M_ij_t, eris_ovvo, optimize=True)
            M_ij_a -= 0.5 * lib.einsum('mejf,mefi->ji',M_ij_t, eris_ovvo, optimize=True)
            M_ij_a += 0.5 * lib.einsum('mejf,mife->ij',M_ij_t, eris_oovv, optimize=True)
            M_ij_a += 0.5 * lib.einsum('mejf,mife->ji',M_ij_t, eris_oovv, optimize=True)
            del M_ij_t

            M_ij_a += 0.5 * 0.5*lib.einsum('lmde,jnde,limn->ij',t2_1_a,
                                            t2_1_a,eris_oooo, optimize=True)

            M_ij_a += 0.5 *0.5*lib.einsum('inde,lmde,jlnm->ij',t2_1_a, t2_1_a,eris_oooo, optimize=True)

            M_ij_a += 0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_a, t2_1_a, eris_oovv, optimize=True)
            M_ij_a -= 0.5*lib.einsum('lmdf,lmde,jfei->ij',t2_1_a, t2_1_a, eris_ovvo, optimize=True)
            M_ij_b +=0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_a, t2_1_a, eris_OOvv , optimize=True)

            M_ij_a -= 0.5*lib.einsum('lnde,lmde,jinm->ij',t2_1_a, t2_1_a, eris_oooo, optimize=True)
            M_ij_a += 0.5*lib.einsum('lnde,lmde,jmni->ij',t2_1_a, t2_1_a, eris_oooo, optimize=True)
            M_ij_b -= 0.5 * lib.einsum('lnde,lmde,nmji->ij',t2_1_a, t2_1_a, eris_ooOO, optimize=True)

            t2_1_ab = t2[0][1][:]
            M_ij_a -= 0.5 * lib.einsum('lmde,jldf,mefi->ij',t2_1_ab, t2_1_a, eris_OVvo,optimize=True)
            M_ij_b += 0.5 * lib.einsum('lmde,ljdf,mefi->ij',t2_1_a, t2_1_ab, eris_ovVO ,optimize=True)
            M_ij_a -= 0.5 * lib.einsum('lmde,ildf,mefj->ij',t2_1_ab, t2_1_a, eris_OVvo,optimize=True)
            M_ij_b += 0.5 * lib.einsum('lmde,lidf,mefj->ij',t2_1_a, t2_1_ab, eris_ovVO ,optimize=True)

            t2_1_b = t2[0][2][:]
            M_ij_a += 0.5 * lib.einsum('lmde,jlfd,mefi->ij',t2_1_b, t2_1_ab, eris_OVvo ,optimize=True)
            M_ij_b -= 0.5 * lib.einsum('mled,jldf,mefi->ij',t2_1_ab, t2_1_b, eris_ovVO,optimize=True)
            M_ij_a += 0.5 * lib.einsum('lmde,ilfd,mefj->ij',t2_1_b, t2_1_ab, eris_OVvo ,optimize=True)
            M_ij_b -= 0.5 * lib.einsum('mled,ildf,mefj->ij',t2_1_ab, t2_1_b, eris_ovVO,optimize=True)

            M_ij_t = lib.einsum('lmde,jldf->mejf', t2_1_b, t2_1_b, optimize=True)
            M_ij_b -= 0.5 * lib.einsum('mejf,mefi->ij',M_ij_t, eris_OVVO, optimize=True)
            M_ij_b -= 0.5 * lib.einsum('mejf,mefi->ji',M_ij_t, eris_OVVO, optimize=True)
            M_ij_b += 0.5 * lib.einsum('mejf,mife->ij',M_ij_t, eris_OOVV, optimize=True)
            M_ij_b += 0.5 * lib.einsum('mejf,mife->ji',M_ij_t, eris_OOVV, optimize=True)
            del M_ij_t

            M_ij_b += 0.5 * 0.5*lib.einsum('lmde,jnde,limn->ij',t2_1_b,
                                            t2_1_b,eris_OOOO, optimize=True)

            M_ij_b += 0.5 * 0.5*lib.einsum('inde,lmde,jlnm->ij',t2_1_b,
                                            t2_1_b,eris_OOOO, optimize=True)

            M_ij_a += 0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_b, t2_1_b, eris_ooVV , optimize=True)
            M_ij_b += 0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_b, t2_1_b, eris_OOVV , optimize=True)
            M_ij_b -= 0.5*lib.einsum('lmdf,lmde,jfei->ij',t2_1_b, t2_1_b, eris_OVVO , optimize=True)

            M_ij_a -= 0.5 * lib.einsum('lnde,lmde,jinm->ij',t2_1_b, t2_1_b, eris_ooOO, optimize=True)
            M_ij_b -= 0.5*lib.einsum('lnde,lmde,jinm->ij',t2_1_b, t2_1_b, eris_OOOO, optimize=True)
            M_ij_b += 0.5*lib.einsum('lnde,lmde,jmni->ij',t2_1_b, t2_1_b, eris_OOOO, optimize=True)

            t2_1_ab = t2[0][1][:]

            M_ij_a += 0.5 * lib.einsum('mled,jlfd,mefi->ij',t2_1_ab, t2_1_ab, eris_ovvo ,optimize=True)
            M_ij_a -= 0.5 * lib.einsum('mled,jlfd,mife->ij',t2_1_ab, t2_1_ab, eris_oovv ,optimize=True)
            M_ij_a -= 0.5 * lib.einsum('mlde,jldf,mife->ij',t2_1_ab, t2_1_ab, eris_ooVV ,optimize=True)

            M_ij_b += 0.5 * lib.einsum('lmde,ljdf,mefi->ij',t2_1_ab, t2_1_ab, eris_OVVO,optimize=True)
            M_ij_b -= 0.5 * lib.einsum('lmde,ljdf,mife->ij',t2_1_ab, t2_1_ab, eris_OOVV,optimize=True)
            M_ij_b -= 0.5 * lib.einsum('lmed,ljfd,mife->ij',t2_1_ab, t2_1_ab, eris_OOvv ,optimize=True)

            M_ij_a += 0.5 * lib.einsum('mled,ilfd,mefj->ij',t2_1_ab, t2_1_ab, eris_ovvo ,optimize=True)
            M_ij_a -= 0.5 * lib.einsum('mled,ilfd,mjfe->ij',t2_1_ab, t2_1_ab, eris_oovv ,optimize=True)
            M_ij_a -= 0.5 * lib.einsum('mlde,ildf,mjfe->ij',t2_1_ab, t2_1_ab, eris_ooVV ,optimize=True)

            M_ij_b += 0.5 * lib.einsum('lmde,lidf,mefj->ij',t2_1_ab, t2_1_ab, eris_OVVO ,optimize=True)
            M_ij_b -= 0.5 * lib.einsum('lmde,lidf,mjfe->ij',t2_1_ab, t2_1_ab, eris_OOVV ,optimize=True)
            M_ij_b -= 0.5 * lib.einsum('lmed,lifd,mjfe->ij',t2_1_ab, t2_1_ab, eris_OOvv ,optimize=True)

            M_ij_a += 0.5 * lib.einsum('lmde,jnde,limn->ij',t2_1_ab ,t2_1_ab,eris_ooOO, optimize=True)
            M_ij_b += 0.5 * lib.einsum('mled,njed,mnli->ij',t2_1_ab ,t2_1_ab,eris_ooOO, optimize=True)

            M_ij_a +=0.5 * lib.einsum('inde,lmde,jlnm->ij',t2_1_ab, t2_1_ab,eris_ooOO, optimize=True)

            M_ij_b +=0.5 * lib.einsum('nied,mled,nmjl->ij',t2_1_ab, t2_1_ab,eris_ooOO, optimize=True)

            M_ij_a +=lib.einsum('mlfd,mled,jief->ij',t2_1_ab, t2_1_ab, eris_oovv , optimize=True)
            M_ij_a -=lib.einsum('mlfd,mled,jfei->ij',t2_1_ab, t2_1_ab, eris_ovvo , optimize=True)
            M_ij_a +=lib.einsum('lmdf,lmde,jief->ij',t2_1_ab, t2_1_ab, eris_ooVV , optimize=True)
            M_ij_b +=lib.einsum('lmdf,lmde,jief->ij',t2_1_ab, t2_1_ab, eris_OOVV , optimize=True)
            M_ij_b -=lib.einsum('lmdf,lmde,jfei->ij',t2_1_ab, t2_1_ab, eris_OVVO , optimize=True)
            M_ij_b +=lib.einsum('lmfd,lmed,jief->ij',t2_1_ab, t2_1_ab, eris_OOvv , optimize=True)

            M_ij_a -= lib.einsum('nled,mled,jinm->ij',t2_1_ab, t2_1_ab, eris_oooo, optimize=True)
            M_ij_a += lib.einsum('nled,mled,jmni->ij',t2_1_ab, t2_1_ab, eris_oooo, optimize=True)
            M_ij_a -= lib.einsum('lnde,lmde,jinm->ij',t2_1_ab, t2_1_ab, eris_ooOO, optimize=True)
            M_ij_b -= lib.einsum('lnde,lmde,jinm->ij',t2_1_ab, t2_1_ab, eris_OOOO, optimize=True)
            M_ij_b += lib.einsum('lnde,lmde,jmni->ij',t2_1_ab, t2_1_ab, eris_OOOO, optimize=True)
            M_ij_b -= lib.einsum('nled,mled,nmji->ij',t2_1_ab, t2_1_ab, eris_ooOO, optimize=True)

        else:
            t2_1_a = t2[0][0][:]
            t2_1_ab = t2[0][1][:]
            t2_1_b = t2[0][2][:]
            t2_2_a = t2[1][0][:]
            t2_2_ab = t2[1][1][:]
            t2_2_b = t2[1][2][:]
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
            temp = lib.einsum('Aa,Ba->AB', f_ov_a, t1_2_a, optimize=True)
            M_ij_a += temp + temp.T
            temp = lib.einsum('ia,iaAB->AB', t1_2_a, eris_ovoo, optimize=True)
            M_ij_a += temp + temp.T
            temp = lib.einsum('ia,BaiA->AB', t1_2_a, eris_ovoo, optimize=True)
            M_ij_a -= temp + temp.T
            temp = lib.einsum('ia,iaAB->AB', t1_2_b, eris_OVoo, optimize=True)
            M_ij_a += temp + temp.T
            temp = lib.einsum('Aiab,Babi->AB', t2_2_a, eris_ovvo, optimize=True)
            M_ij_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('Aiab,Bbai->AB', t2_2_a, eris_ovvo, optimize=True)
            M_ij_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('Aiab,Babi->AB', t2_2_ab, eris_ovVO, optimize=True)
            M_ij_a += temp + temp.T
            temp = lib.einsum('A,Ba,Aa->AB', e_occ_a, t1_1_a, t1_2_a, optimize=True)
            M_ij_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('A,Aa,Ba->AB', e_occ_a, t1_1_a, t1_2_a, optimize=True)
            M_ij_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('a,Aa,Ba->AB', e_vir_a, t1_1_a, t1_2_a, optimize=True)
            M_ij_a += temp + temp.T
            temp = lib.einsum('A,Biab,Aiab->AB', e_occ_a, t2_1_a, t2_2_a, optimize=True)
            M_ij_a -= 1/4 * (temp + temp.T)
            temp = lib.einsum('A,Aiab,Biab->AB', e_occ_a, t2_1_a, t2_2_a, optimize=True)
            M_ij_a -= 1/4 * (temp + temp.T)
            temp = lib.einsum('a,Aiab,Biab->AB', e_vir_a, t2_1_a, t2_2_a, optimize=True)
            M_ij_a += temp + temp.T
            temp = lib.einsum('i,Aiab,Biab->AB', e_occ_a, t2_1_a, t2_2_a, optimize=True)
            M_ij_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('A,Biab,Aiab->AB', e_occ_a, t2_1_ab, t2_2_ab, optimize=True)
            M_ij_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('A,Aiab,Biab->AB', e_occ_a, t2_1_ab, t2_2_ab, optimize=True)
            M_ij_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('a,Aiab,Biab->AB', e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
            M_ij_a += temp + temp.T
            temp = lib.einsum('a,Aiba,Biba->AB', e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
            M_ij_a += temp + temp.T
            temp = lib.einsum('i,Aiab,Biab->AB', e_occ_b, t2_1_ab, t2_2_ab, optimize=True)
            M_ij_a -= temp + temp.T
            temp = lib.einsum('Aa,ib,Biab->AB', f_ov_a, t1_1_a, t2_1_a, optimize=True)
            M_ij_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,Ab,Biab->AB', f_ov_a, t1_1_a, t2_1_a, optimize=True)
            M_ij_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,Ab,Biba->AB', f_ov_b, t1_1_a, t2_1_ab, optimize=True)
            M_ij_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('Aa,ib,Biab->AB', f_ov_a, t1_1_b, t2_1_ab, optimize=True)
            M_ij_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('Aa,ib,Babi->AB', t1_1_a, t1_1_b, eris_ovVO, optimize=True)
            M_ij_a += 2 * (temp + temp.T)
            temp = lib.einsum('Aa,Bibc,ibac->AB', t1_1_a, t2_1_a, eris_ovvv, optimize=True)
            M_ij_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('Aa,Bibc,icab->AB', t1_1_a, t2_1_a, eris_ovvv, optimize=True)
            M_ij_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('Aa,ijab,jbiB->AB', t1_1_a, t2_1_a, eris_ovoo, optimize=True)
            M_ij_a -= 1/4 * (temp + temp.T)
            temp = lib.einsum('Aa,ijab,ibjB->AB', t1_1_a, t2_1_a, eris_ovoo, optimize=True)
            M_ij_a += 1/4 * (temp + temp.T)
            temp = lib.einsum('ia,Aibc,Bbac->AB', t1_1_a, t2_1_a, eris_ovvv, optimize=True)
            M_ij_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,Aibc,Bcab->AB', t1_1_a, t2_1_a, eris_ovvv, optimize=True)
            M_ij_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,Ajab,jbBi->AB', t1_1_a, t2_1_a, eris_ovoo, optimize=True)
            M_ij_a -= temp + temp.T
            temp = lib.einsum('ia,Ajab,Bbji->AB', t1_1_a, t2_1_a, eris_ovoo, optimize=True)
            M_ij_a += temp + temp.T
            temp = lib.einsum('ia,ijab,jbAB->AB', t1_1_a, t2_1_a, eris_ovoo, optimize=True)
            M_ij_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,ijab,BbjA->AB', t1_1_a, t2_1_a, eris_ovoo, optimize=True)
            M_ij_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('Aa,Bibc,icab->AB', t1_1_a, t2_1_ab, eris_OVvv, optimize=True)
            M_ij_a += temp + temp.T
            temp = lib.einsum('Aa,ijab,jbiB->AB', t1_1_a, t2_1_ab, eris_OVoo, optimize=True)
            M_ij_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,Ajab,jbBi->AB', t1_1_a, t2_1_ab, eris_OVoo, optimize=True)
            M_ij_a -= temp + temp.T
            temp = lib.einsum('ia,ijab,jbAB->AB', t1_1_a, t2_1_ab, eris_OVoo, optimize=True)
            M_ij_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,Aibc,Bbac->AB', t1_1_b, t2_1_ab, eris_ovVV, optimize=True)
            M_ij_a += temp + temp.T
            temp = lib.einsum('ia,Ajba,Bbji->AB', t1_1_b, t2_1_ab, eris_ovOO, optimize=True)
            M_ij_a -= temp + temp.T
            temp = lib.einsum('ia,jiba,jbAB->AB', t1_1_b, t2_1_ab, eris_ovoo, optimize=True)
            M_ij_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,jiba,BbjA->AB', t1_1_b, t2_1_ab, eris_ovoo, optimize=True)
            M_ij_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,ijab,jbAB->AB', t1_1_b, t2_1_b, eris_OVoo, optimize=True)
            M_ij_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('Aiab,Bjac,ibcj->AB', t2_1_a, t2_1_ab, eris_ovVO, optimize=True)
            M_ij_a += temp + temp.T
            temp = lib.einsum('Aiab,ijac,Bbcj->AB', t2_1_a, t2_1_ab, eris_ovVO, optimize=True)
            M_ij_a -= temp + temp.T
            temp = lib.einsum('Aiab,ijbc,Bacj->AB', t2_1_ab, t2_1_b, eris_ovVO, optimize=True)
            M_ij_a += temp + temp.T
            M_ij_a += lib.einsum('ABab,ia,ib->AB', eris_oovv, t1_1_a, t1_1_a, optimize=True)
            M_ij_a -= lib.einsum('ABij,ia,ja->AB', eris_oooo, t1_1_a, t1_1_a, optimize=True)
            M_ij_a -= lib.einsum('AabB,ib,ia->AB', eris_ovvo, t1_1_a, t1_1_a, optimize=True)
            M_ij_a += lib.einsum('AijB,ja,ia->AB', eris_oooo, t1_1_a, t1_1_a, optimize=True)
            temp = lib.einsum('Aabi,ib,Ba->AB', eris_ovvo, t1_1_a, t1_1_a, optimize=True)
            M_ij_a += 2 * (temp + temp.T)
            temp = lib.einsum('Aabi,ia,Bb->AB', eris_ovvo, t1_1_a, t1_1_a, optimize=True)
            M_ij_a -= temp + temp.T
            temp = lib.einsum('Aiab,ia,Bb->AB', eris_oovv, t1_1_a, t1_1_a, optimize=True)
            M_ij_a -= temp + temp.T
            M_ij_a += lib.einsum('ABab,ia,ib->AB', eris_ooVV, t1_1_b, t1_1_b, optimize=True)
            M_ij_a -= lib.einsum('ABij,ia,ja->AB', eris_ooOO, t1_1_b, t1_1_b, optimize=True)
            M_ij_a += 1/2 *  lib.einsum('ABab,ijac,ijbc->AB', eris_oovv, t2_1_a, t2_1_a, optimize=True)
            M_ij_a -= 1/2 *  lib.einsum('ABij,ikab,jkab->AB', eris_oooo, t2_1_a, t2_1_a, optimize=True)
            M_ij_a -= 1/2 *  lib.einsum('AabB,ijbc,ijac->AB', eris_ovvo, t2_1_a, t2_1_a, optimize=True)
            M_ij_a += 1/2 *  lib.einsum('AijB,jkab,ikab->AB', eris_oooo, t2_1_a, t2_1_a, optimize=True)
            temp = lib.einsum('Aabi,ijbc,Bjac->AB', eris_ovvo, t2_1_a, t2_1_a, optimize=True)
            M_ij_a += temp + temp.T
            temp = lib.einsum('Aiab,ijac,Bjbc->AB', eris_oovv, t2_1_a, t2_1_a, optimize=True)
            M_ij_a -= temp + temp.T
            temp = lib.einsum('Aijk,Bjab,ikab->AB', eris_oooo, t2_1_a, t2_1_a, optimize=True)
            M_ij_a += 1/4 * (temp + temp.T)
            temp = lib.einsum('Aijk,Bjab,kiab->AB', eris_oooo, t2_1_a, t2_1_a, optimize=True)
            M_ij_a -= 1/4 * (temp + temp.T)
            M_ij_a += lib.einsum('iabj,Aiac,Bjbc->AB', eris_ovvo, t2_1_a, t2_1_a, optimize=True)
            M_ij_a -= lib.einsum('ijab,Aibc,Bjac->AB', eris_oovv, t2_1_a, t2_1_a, optimize=True)
            M_ij_a += lib.einsum('ABab,ijac,ijbc->AB', eris_oovv, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_a += lib.einsum('ABab,ijca,ijcb->AB', eris_ooVV, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_a -= lib.einsum('ABij,ikab,jkab->AB', eris_oooo, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_a -= lib.einsum('ABij,kiab,kjab->AB', eris_ooOO, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_a -= lib.einsum('AabB,ijbc,ijac->AB', eris_ovvo, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_a += lib.einsum('AijB,jkab,ikab->AB', eris_oooo, t2_1_ab, t2_1_ab, optimize=True)
            temp = lib.einsum('Aabi,ijbc,Bjac->AB', eris_ovvo, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_a += temp + temp.T
            temp = lib.einsum('Aiab,ijac,Bjbc->AB', eris_oovv, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_a -= temp + temp.T
            temp = lib.einsum('Aiab,ijca,Bjcb->AB', eris_ooVV, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_a -= temp + temp.T
            temp = lib.einsum('Aijk,Bjab,ikab->AB', eris_ooOO, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_a += temp + temp.T
            M_ij_a += lib.einsum('iabj,Aica,Bjcb->AB', eris_OVVO, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_a -= lib.einsum('ijab,Aibc,Bjac->AB', eris_OOvv, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_a -= lib.einsum('ijab,Aicb,Bjca->AB', eris_OOVV, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_a += 1/2 *  lib.einsum('ABab,ijac,ijbc->AB', eris_ooVV, t2_1_b, t2_1_b, optimize=True)
            M_ij_a -= 1/2 *  lib.einsum('ABij,ikab,jkab->AB', eris_ooOO, t2_1_b, t2_1_b, optimize=True)
            temp = lib.einsum('A,Ba,ib,Aiab->AB', e_occ_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            M_ij_a -= 1/3 * (temp + temp.T)
            temp = lib.einsum('A,Aa,ib,Biab->AB', e_occ_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            M_ij_a -= 1/6 * (temp + temp.T)
            temp = lib.einsum('a,Aa,ib,Biab->AB', e_vir_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            M_ij_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('a,Ab,ia,Biba->AB', e_vir_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            M_ij_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('i,Aa,ib,Biab->AB', e_occ_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            M_ij_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('A,Biab,ia,Ab->AB', e_occ_a, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            M_ij_a += 1/6 * (temp + temp.T)
            temp = lib.einsum('A,Aiab,ia,Bb->AB', e_occ_a, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            M_ij_a += 1/3 * (temp + temp.T)
            temp = lib.einsum('a,Aiab,Ba,ib->AB', e_vir_a, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            M_ij_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('a,Aiab,ia,Bb->AB', e_vir_a, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            M_ij_a -= 1/2 * (temp + temp.T)
            temp = lib.einsum('i,Aiab,ia,Bb->AB', e_occ_a, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            M_ij_a += 1/2 * (temp + temp.T)
            temp = lib.einsum('Aa,Ba->AB', f_ov_b, t1_2_b, optimize=True)
            M_ij_b += temp + temp.T
            temp = lib.einsum('ia,iaAB->AB', t1_2_a, eris_ovOO, optimize=True)
            M_ij_b += temp + temp.T
            temp = lib.einsum('ia,iaAB->AB', t1_2_b, eris_OVOO, optimize=True)
            M_ij_b += temp + temp.T
            temp = lib.einsum('ia,BaiA->AB', t1_2_b, eris_OVOO, optimize=True)
            M_ij_b -= temp + temp.T
            temp = lib.einsum('iAab,Bbai->AB', t2_2_ab, eris_OVvo, optimize=True)
            M_ij_b += temp + temp.T
            temp = lib.einsum('Aiab,Babi->AB', t2_2_b, eris_OVVO, optimize=True)
            M_ij_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('Aiab,Bbai->AB', t2_2_b, eris_OVVO, optimize=True)
            M_ij_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('A,Ba,Aa->AB', e_occ_b, t1_1_b, t1_2_b, optimize=True)
            M_ij_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('A,Aa,Ba->AB', e_occ_b, t1_1_b, t1_2_b, optimize=True)
            M_ij_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('a,Aa,Ba->AB', e_vir_b, t1_1_b, t1_2_b, optimize=True)
            M_ij_b += temp + temp.T
            temp = lib.einsum('A,iBab,iAab->AB', e_occ_b, t2_1_ab, t2_2_ab, optimize=True)
            M_ij_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('A,iAab,iBab->AB', e_occ_b, t2_1_ab, t2_2_ab, optimize=True)
            M_ij_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('a,iAab,iBab->AB', e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
            M_ij_b += temp + temp.T
            temp = lib.einsum('a,iAba,iBba->AB', e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
            M_ij_b += temp + temp.T
            temp = lib.einsum('i,iAab,iBab->AB', e_occ_a, t2_1_ab, t2_2_ab, optimize=True)
            M_ij_b -= temp + temp.T
            temp = lib.einsum('A,Biab,Aiab->AB', e_occ_b, t2_1_b, t2_2_b, optimize=True)
            M_ij_b -= 1/4 * (temp + temp.T)
            temp = lib.einsum('A,Aiab,Biab->AB', e_occ_b, t2_1_b, t2_2_b, optimize=True)
            M_ij_b -= 1/4 * (temp + temp.T)
            temp = lib.einsum('a,Aiab,Biab->AB', e_vir_b, t2_1_b, t2_2_b, optimize=True)
            M_ij_b += temp + temp.T
            temp = lib.einsum('i,Aiab,Biab->AB', e_occ_b, t2_1_b, t2_2_b, optimize=True)
            M_ij_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('Aa,ib,iBba->AB', f_ov_b, t1_1_a, t2_1_ab, optimize=True)
            M_ij_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,Ab,iBab->AB', f_ov_a, t1_1_b, t2_1_ab, optimize=True)
            M_ij_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('Aa,ib,Biab->AB', f_ov_b, t1_1_b, t2_1_b, optimize=True)
            M_ij_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,Ab,Biab->AB', f_ov_b, t1_1_b, t2_1_b, optimize=True)
            M_ij_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,Ab,Bbai->AB', t1_1_a, t1_1_b, eris_OVvo, optimize=True)
            M_ij_b += 2 * (temp + temp.T)
            temp = lib.einsum('ia,ijab,jbAB->AB', t1_1_a, t2_1_a, eris_ovOO, optimize=True)
            M_ij_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,iAbc,Bcab->AB', t1_1_a, t2_1_ab, eris_OVvv, optimize=True)
            M_ij_b += temp + temp.T
            temp = lib.einsum('ia,ijab,jbAB->AB', t1_1_a, t2_1_ab, eris_OVOO, optimize=True)
            M_ij_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,ijab,BbjA->AB', t1_1_a, t2_1_ab, eris_OVOO, optimize=True)
            M_ij_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,jAab,Bbji->AB', t1_1_a, t2_1_ab, eris_OVoo, optimize=True)
            M_ij_b -= temp + temp.T
            temp = lib.einsum('Aa,iBbc,ibac->AB', t1_1_b, t2_1_ab, eris_ovVV, optimize=True)
            M_ij_b += temp + temp.T
            temp = lib.einsum('Aa,ijba,ibjB->AB', t1_1_b, t2_1_ab, eris_ovOO, optimize=True)
            M_ij_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,jAba,jbBi->AB', t1_1_b, t2_1_ab, eris_ovOO, optimize=True)
            M_ij_b -= temp + temp.T
            temp = lib.einsum('ia,jiba,jbAB->AB', t1_1_b, t2_1_ab, eris_ovOO, optimize=True)
            M_ij_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('Aa,Bibc,ibac->AB', t1_1_b, t2_1_b, eris_OVVV, optimize=True)
            M_ij_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('Aa,Bibc,icab->AB', t1_1_b, t2_1_b, eris_OVVV, optimize=True)
            M_ij_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('Aa,ijab,jbiB->AB', t1_1_b, t2_1_b, eris_OVOO, optimize=True)
            M_ij_b -= 1/4 * (temp + temp.T)
            temp = lib.einsum('Aa,ijab,ibjB->AB', t1_1_b, t2_1_b, eris_OVOO, optimize=True)
            M_ij_b += 1/4 * (temp + temp.T)
            temp = lib.einsum('ia,Aibc,Bbac->AB', t1_1_b, t2_1_b, eris_OVVV, optimize=True)
            M_ij_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,Aibc,Bcab->AB', t1_1_b, t2_1_b, eris_OVVV, optimize=True)
            M_ij_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,Ajab,jbBi->AB', t1_1_b, t2_1_b, eris_OVOO, optimize=True)
            M_ij_b -= temp + temp.T
            temp = lib.einsum('ia,Ajab,Bbji->AB', t1_1_b, t2_1_b, eris_OVOO, optimize=True)
            M_ij_b += temp + temp.T
            temp = lib.einsum('ia,ijab,jbAB->AB', t1_1_b, t2_1_b, eris_OVOO, optimize=True)
            M_ij_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('ia,ijab,BbjA->AB', t1_1_b, t2_1_b, eris_OVOO, optimize=True)
            M_ij_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('ijab,iAac,Bcbj->AB', t2_1_a, t2_1_ab, eris_OVvo, optimize=True)
            M_ij_b += temp + temp.T
            temp = lib.einsum('iAab,Bjbc,iacj->AB', t2_1_ab, t2_1_b, eris_ovVO, optimize=True)
            M_ij_b += temp + temp.T
            temp = lib.einsum('ijab,Ajbc,Bcai->AB', t2_1_ab, t2_1_b, eris_OVvo, optimize=True)
            M_ij_b -= temp + temp.T
            M_ij_b += lib.einsum('ABab,ia,ib->AB', eris_OOvv, t1_1_a, t1_1_a, optimize=True)
            M_ij_b -= lib.einsum('ijAB,ia,ja->AB', eris_ooOO, t1_1_a, t1_1_a, optimize=True)
            M_ij_b += lib.einsum('ABab,ia,ib->AB', eris_OOVV, t1_1_b, t1_1_b, optimize=True)
            M_ij_b -= lib.einsum('ABij,ia,ja->AB', eris_OOOO, t1_1_b, t1_1_b, optimize=True)
            M_ij_b -= lib.einsum('AabB,ib,ia->AB', eris_OVVO, t1_1_b, t1_1_b, optimize=True)
            M_ij_b += lib.einsum('AijB,ja,ia->AB', eris_OOOO, t1_1_b, t1_1_b, optimize=True)
            temp = lib.einsum('Aabi,ib,Ba->AB', eris_OVVO, t1_1_b, t1_1_b, optimize=True)
            M_ij_b += 2 * (temp + temp.T)
            temp = lib.einsum('Aabi,ia,Bb->AB', eris_OVVO, t1_1_b, t1_1_b, optimize=True)
            M_ij_b -= temp + temp.T
            temp = lib.einsum('Aiab,ia,Bb->AB', eris_OOVV, t1_1_b, t1_1_b, optimize=True)
            M_ij_b -= temp + temp.T
            M_ij_b += 1/2 *  lib.einsum('ABab,ijac,ijbc->AB', eris_OOvv, t2_1_a, t2_1_a, optimize=True)
            M_ij_b -= 1/2 *  lib.einsum('ijAB,ikab,jkab->AB', eris_ooOO, t2_1_a, t2_1_a, optimize=True)
            M_ij_b += lib.einsum('ABab,ijac,ijbc->AB', eris_OOvv, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_b += lib.einsum('ABab,ijca,ijcb->AB', eris_OOVV, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_b -= lib.einsum('ijAB,ikab,jkab->AB', eris_ooOO, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_b -= lib.einsum('ABij,kiab,kjab->AB', eris_OOOO, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_b -= lib.einsum('AabB,ijcb,ijca->AB', eris_OVVO, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_b += lib.einsum('AijB,kjab,kiab->AB', eris_OOOO, t2_1_ab, t2_1_ab, optimize=True)
            temp = lib.einsum('Aabi,jicb,jBca->AB', eris_OVVO, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_b += temp + temp.T
            temp = lib.einsum('Aiab,jiac,jBbc->AB', eris_OOvv, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_b -= temp + temp.T
            temp = lib.einsum('Aiab,jica,jBcb->AB', eris_OOVV, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_b -= temp + temp.T
            temp = lib.einsum('jkAi,jBab,kiab->AB', eris_ooOO, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_b += temp + temp.T
            M_ij_b += lib.einsum('iabj,iAac,jBbc->AB', eris_ovvo, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_b -= lib.einsum('ijab,iAbc,jBac->AB', eris_oovv, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_b -= lib.einsum('ijab,iAcb,jBca->AB', eris_ooVV, t2_1_ab, t2_1_ab, optimize=True)
            M_ij_b += 1/2 *  lib.einsum('ABab,ijac,ijbc->AB', eris_OOVV, t2_1_b, t2_1_b, optimize=True)
            M_ij_b -= 1/2 *  lib.einsum('ABij,ikab,jkab->AB', eris_OOOO, t2_1_b, t2_1_b, optimize=True)
            M_ij_b -= 1/2 *  lib.einsum('AabB,ijbc,ijac->AB', eris_OVVO, t2_1_b, t2_1_b, optimize=True)
            M_ij_b += 1/2 *  lib.einsum('AijB,jkab,ikab->AB', eris_OOOO, t2_1_b, t2_1_b, optimize=True)
            temp = lib.einsum('Aabi,ijbc,Bjac->AB', eris_OVVO, t2_1_b, t2_1_b, optimize=True)
            M_ij_b += temp + temp.T
            temp = lib.einsum('Aiab,ijac,Bjbc->AB', eris_OOVV, t2_1_b, t2_1_b, optimize=True)
            M_ij_b -= temp + temp.T
            temp = lib.einsum('Aijk,Bjab,ikab->AB', eris_OOOO, t2_1_b, t2_1_b, optimize=True)
            M_ij_b += 1/4 * (temp + temp.T)
            temp = lib.einsum('Aijk,Bjab,kiab->AB', eris_OOOO, t2_1_b, t2_1_b, optimize=True)
            M_ij_b -= 1/4 * (temp + temp.T)
            M_ij_b += lib.einsum('iabj,Aiac,Bjbc->AB', eris_OVVO, t2_1_b, t2_1_b, optimize=True)
            M_ij_b -= lib.einsum('ijab,Aibc,Bjac->AB', eris_OOVV, t2_1_b, t2_1_b, optimize=True)
            temp = lib.einsum('A,ia,Bb,iAab->AB', e_occ_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            M_ij_b -= 1/3 * (temp + temp.T)
            temp = lib.einsum('A,ia,Ab,iBab->AB', e_occ_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            M_ij_b -= 1/6 * (temp + temp.T)
            temp = lib.einsum('a,ia,Ab,iBab->AB', e_vir_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            M_ij_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('a,ib,Aa,iBba->AB', e_vir_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            M_ij_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('i,ia,Ab,iBab->AB', e_occ_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            M_ij_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('A,Biab,ia,Ab->AB', e_occ_b, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            M_ij_b += 1/6 * (temp + temp.T)
            temp = lib.einsum('A,Aiab,ia,Bb->AB', e_occ_b, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            M_ij_b += 1/3 * (temp + temp.T)
            temp = lib.einsum('a,Aiab,Ba,ib->AB', e_vir_b, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            M_ij_b += 1/2 * (temp + temp.T)
            temp = lib.einsum('a,Aiab,ia,Bb->AB', e_vir_b, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            M_ij_b -= 1/2 * (temp + temp.T)
            temp = lib.einsum('i,Aiab,ia,Bb->AB', e_occ_b, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            M_ij_b += 1/2 * (temp + temp.T)

            if isinstance(eris.vvvv_p, np.ndarray):
                va = adc.mo_coeff[0][:, nocc_a:]
                vb = adc.mo_coeff[1][:, nocc_b:]
                v_eeee_aaaa = ao2mo.general(adc._scf._eri, (va, va, va, va), compact=False).reshape(nvir_a,
                    nvir_a, nvir_a, nvir_a)
                M_ij_a += 1/4 *  lib.einsum('abcd,Aiac,Bibd->AB', v_eeee_aaaa, t2_1_a, t2_1_a, optimize=True)
                M_ij_a -= 1/4 *  lib.einsum('abcd,Aiac,Bidb->AB', v_eeee_aaaa, t2_1_a, t2_1_a, optimize=True)
                del v_eeee_aaaa
                v_eeee_abab = eris.vVvV_p.reshape(nvir_a, nvir_b, nvir_a, nvir_b)
                M_ij_a += lib.einsum('acbd,Aiac,Bibd->AB', v_eeee_abab, t2_1_ab, t2_1_ab, optimize=True)
                M_ij_b += lib.einsum('acbd,iAac,iBbd->AB', v_eeee_abab, t2_1_ab, t2_1_ab, optimize=True)
                del v_eeee_abab
                v_eeee_bbbb = ao2mo.general(adc._scf._eri, (vb, vb, vb, vb), compact=False).reshape(nvir_b,
                    nvir_b, nvir_b, nvir_b)
                M_ij_b += 1/4 *  lib.einsum('abcd,Aiac,Bibd->AB', v_eeee_bbbb, t2_1_b, t2_1_b, optimize=True)
                M_ij_b -= 1/4 *  lib.einsum('abcd,Aiac,Bidb->AB', v_eeee_bbbb, t2_1_b, t2_1_b, optimize=True)
                del v_eeee_bbbb
            elif isinstance(eris.vvvv_p, list):
                mol = adc.mol
                va = adc.mo_coeff[0][:, nocc_a:]
                vb = adc.mo_coeff[1][:, nocc_b:]
                chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                for p, q in lib.prange(0, nvir_a, chnk_size):
                    v_eeee_aaaa = ao2mo.general(mol, (va[:, p:q], va, va, va),
                        compact=False).reshape(q-p, nvir_a, nvir_a, nvir_a)
                    t2_1_a_chunk = t2_1_a[:, :, p:q, :]
                    M_ij_a += 1/4 *  lib.einsum('abcd,Aiac,Bibd->AB', v_eeee_aaaa, t2_1_a_chunk, t2_1_a,
                        optimize=True)
                    M_ij_a -= 1/4 *  lib.einsum('abcd,Aiac,Bidb->AB', v_eeee_aaaa, t2_1_a_chunk, t2_1_a,
                        optimize=True)
                    del v_eeee_aaaa
                p = 0
                for dataset in eris.vVvV_p:
                    k = dataset.shape[0]
                    v_eeee_abab = dataset[:].reshape(k, nvir_b, nvir_a, nvir_b)
                    t2_1_ab_chunk = t2_1_ab[:, :, p:p+k, :]
                    M_ij_a += lib.einsum('acbd,Aiac,Bibd->AB', v_eeee_abab, t2_1_ab_chunk, t2_1_ab, optimize=True)
                    M_ij_b += lib.einsum('acbd,iAac,iBbd->AB', v_eeee_abab, t2_1_ab_chunk, t2_1_ab, optimize=True)
                    del v_eeee_abab
                    p += k
                for p, q in lib.prange(0, nvir_b, chnk_size):
                    v_eeee_bbbb = ao2mo.general(mol, (vb[:, p:q], vb, vb, vb),
                        compact=False).reshape(q-p, nvir_b, nvir_b, nvir_b)
                    t2_1_b_chunk = t2_1_b[:, :, p:q, :]
                    M_ij_b += 1/4 *  lib.einsum('abcd,Aiac,Bibd->AB', v_eeee_bbbb, t2_1_b_chunk, t2_1_b,
                        optimize=True)
                    M_ij_b -= 1/4 *  lib.einsum('abcd,Aiac,Bidb->AB', v_eeee_bbbb, t2_1_b_chunk, t2_1_b,
                        optimize=True)
                    del v_eeee_bbbb
            elif eris.vvvv_p is None:
                lad_aaaa = contract_ladder(adc, t2_1_a, (eris.Lvv, eris.Lvv))
                lad_aabb = contract_ladder(adc, t2_1_ab, (eris.Lvv, eris.LVV))
                lad_bbbb = contract_ladder(adc, t2_1_b, (eris.LVV, eris.LVV))
                M_ij_a += 1/4 *  lib.einsum('Aibd,Bibd->AB', lad_aaaa, t2_1_a, optimize=True)
                M_ij_a -= 1/4 *  lib.einsum('Aibd,Bidb->AB', lad_aaaa, t2_1_a, optimize=True)
                M_ij_a += lib.einsum('Aibd,Bibd->AB', lad_aabb, t2_1_ab, optimize=True)
                M_ij_b += lib.einsum('iAbd,iBbd->AB', lad_aabb, t2_1_ab, optimize=True)
                M_ij_b += 1/4 *  lib.einsum('Aibd,Bibd->AB', lad_bbbb, t2_1_b, optimize=True)
                M_ij_b -= 1/4 *  lib.einsum('Aibd,Bidb->AB', lad_bbbb, t2_1_b, optimize=True)

        del t2_1_a
        del t2_1_b
        del t2_1_ab

    M_ij = (M_ij_a, M_ij_b)
    cput0 = log.timer_debug1("Completed M_ab ADC(3) calculation", *cput0)

    return M_ij


def get_diag(adc,M_ij=None,eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    if M_ij is None:
        M_ij = adc.get_imds()

    M_ij_a, M_ij_b = M_ij[0], M_ij[1]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a * (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a * nocc_b
    n_doubles_aba = nvir_a * nocc_b * nocc_a
    n_doubles_bbb = nocc_b * (nocc_b - 1) * nvir_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)

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

    d_ij_a = e_occ_a[:,None] + e_occ_a
    d_a_a = e_vir_a[:,None]
    D_n_a = -d_a_a + d_ij_a.reshape(-1)
    D_n_a = D_n_a.reshape((nvir_a,nocc_a,nocc_a))
    D_aij_a = D_n_a.copy()[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

    d_ij_b = e_occ_b[:,None] + e_occ_b
    d_a_b = e_vir_b[:,None]
    D_n_b = -d_a_b + d_ij_b.reshape(-1)
    D_n_b = D_n_b.reshape((nvir_b,nocc_b,nocc_b))
    D_aij_b = D_n_b.copy()[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

    d_ij_ab = e_occ_b[:,None] + e_occ_a
    d_a_b = e_vir_b[:,None]
    D_n_bab = -d_a_b + d_ij_ab.reshape(-1)
    D_aij_bab = D_n_bab.reshape(-1)

    d_ij_ab = e_occ_a[:,None] + e_occ_b
    d_a_a = e_vir_a[:,None]
    D_n_aba = -d_a_a + d_ij_ab.reshape(-1)
    D_aij_aba = D_n_aba.reshape(-1)

    diag = np.zeros(dim)

    # Compute precond in h1-h1 block
    M_ij_a_diag = np.diagonal(M_ij_a)
    M_ij_b_diag = np.diagonal(M_ij_b)

    diag[s_a:f_a] = M_ij_a_diag.copy()
    diag[s_b:f_b] = M_ij_b_diag.copy()

    # Compute precond in 2p1h-2p1h block

    diag[s_aaa:f_aaa] = D_aij_a.copy()
    diag[s_bab:f_bab] = D_aij_bab.copy()
    diag[s_aba:f_aba] = D_aij_aba.copy()
    diag[s_bbb:f_bbb] = D_aij_b.copy()

#    ###### Additional terms for the preconditioner ####
#    if (method == "adc(2)-x" or method == "adc(3)"):
#
#        if eris is None:
#            eris = adc.transform_integrals()
#
#        if isinstance(eris.vvvv_p, np.ndarray):
#
#            eris_oooo = eris.oooo
#            eris_OOOO = eris.OOOO
#            eris_ooOO = eris.ooOO
#            eris_oovv = eris.oovv
#            eris_OOVV = eris.OOVV
#            eris_ooVV = eris.ooVV
#            eris_OOvv = eris.OOvv
#            eris_ovvo = eris.ovvo
#            eris_OVVO = eris.OVVO
#
#            eris_oooo_p = np.ascontiguousarray(eris_oooo.transpose(0,2,1,3))
#            eris_oooo_p -= np.ascontiguousarray(eris_oooo_p.transpose(0,1,3,2))
#            eris_oooo_p = eris_oooo_p.reshape(nocc_a*nocc_a, nocc_a*nocc_a)
#
#            temp = np.zeros((nvir_a,eris_oooo_p.shape[0]))
#            temp[:] += np.diagonal(eris_oooo_p)
#            temp = temp.reshape(nvir_a, nocc_a, nocc_a)
#            diag[s_aaa:f_aaa] += -temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
#
#            eris_OOOO_p = np.ascontiguousarray(eris_OOOO.transpose(0,2,1,3))
#            eris_OOOO_p -= np.ascontiguousarray(eris_OOOO_p.transpose(0,1,3,2))
#            eris_OOOO_p = eris_OOOO_p.reshape(nocc_b*nocc_b, nocc_b*nocc_b)
#
#            temp = np.zeros((nvir_b,eris_OOOO_p.shape[0]))
#            temp[:] += np.diagonal(eris_OOOO_p)
#            temp = temp.reshape(nvir_b, nocc_b, nocc_b)
#            diag[s_bbb:f_bbb] += -temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
#
#            eris_oOoO_p = np.ascontiguousarray(eris_ooOO.transpose(0,2,1,3))
#            eris_oOoO_p = eris_oOoO_p.reshape(nocc_a*nocc_b, nocc_a*nocc_b)
#
#            temp = np.zeros((nvir_b, eris_oOoO_p.shape[0]))
#            temp[:] += np.diag(eris_oOoO_p)
#            diag[s_bab:f_bab] += -temp.reshape(-1)
#
#            temp = np.zeros((nvir_a, eris_oOoO_p.shape[0]))
#            temp[:] += np.diag(eris_oOoO_p.T)
#            diag[s_aba:f_aba] += -temp.reshape(-1)
#
#            eris_ovov_p = np.ascontiguousarray(eris_oovv.transpose(0,2,1,3))
#            eris_ovov_p -= np.ascontiguousarray(eris_ovvo.transpose(0,2,3,1))
#            eris_ovov_p = eris_ovov_p.reshape(nocc_a*nvir_a, nocc_a*nvir_a)
#
#            temp = np.zeros((nocc_a,nocc_a,nvir_a))
#            temp[:] += np.diagonal(eris_ovov_p).reshape(nocc_a,nvir_a)
#            temp = np.ascontiguousarray(temp.T)
#            diag[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
#
#            temp = np.ascontiguousarray(temp.transpose(0,2,1))
#            diag[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
#
#            eris_OVOV_p = np.ascontiguousarray(eris_OOVV.transpose(0,2,1,3))
#            eris_OVOV_p -= np.ascontiguousarray(eris_OVVO.transpose(0,2,3,1))
#            eris_OVOV_p = eris_OVOV_p.reshape(nocc_b*nvir_b, nocc_b*nvir_b)
#
#            temp = np.zeros((nocc_b,nocc_b,nvir_b))
#            temp[:] += np.diagonal(eris_OVOV_p).reshape(nocc_b,nvir_b)
#            temp = np.ascontiguousarray(temp.T)
#            diag[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
#
#            temp = np.ascontiguousarray(temp.transpose(0,2,1))
#            diag[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
#
#            temp = np.zeros((nocc_a, nocc_b, nvir_b))
#            temp[:] += np.diagonal(eris_OVOV_p).reshape(nocc_b, nvir_b)
#            temp = np.ascontiguousarray(temp.transpose(2,0,1))
#            diag[s_bab:f_bab] += temp.reshape(-1)
#
#            temp = np.zeros((nocc_b, nocc_a, nvir_a))
#            temp[:] += np.diagonal(eris_ovov_p).reshape(nocc_a, nvir_a)
#            temp = np.ascontiguousarray(temp.transpose(2,0,1))
#            diag[s_aba:f_aba] += temp.reshape(-1)
#
#            eris_oVoV_p = np.ascontiguousarray(eris_ooVV.transpose(0,2,1,3))
#            eris_oVoV_p = eris_oVoV_p.reshape(nocc_a*nvir_b, nocc_a*nvir_b)
#
#            temp = np.zeros((nocc_b, nocc_a, nvir_b))
#            temp[:] += np.diagonal(eris_oVoV_p).reshape(nocc_a,nvir_b)
#            temp = np.ascontiguousarray(temp.transpose(2,1,0))
#            diag[s_bab:f_bab] += temp.reshape(-1)
#
#            eris_OvOv_p = np.ascontiguousarray(eris_OOvv.transpose(0,2,1,3))
#            eris_OvOv_p = eris_OvOv_p.reshape(nocc_b*nvir_a, nocc_b*nvir_a)
#
#            temp = np.zeros((nocc_a, nocc_b, nvir_a))
#            temp[:] += np.diagonal(eris_OvOv_p).reshape(nocc_b,nvir_a)
#            temp = np.ascontiguousarray(temp.transpose(2,1,0))
#            diag[s_aba:f_aba] += temp.reshape(-1)
#        else:
#           raise Exception("Precond not available for out-of-core and density-fitted algo")

    diag = -diag
    return diag


def matvec(adc, M_ij=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)

    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a * (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a * nocc_b
    n_doubles_aba = nvir_a * nocc_b * nocc_a
    n_doubles_bbb = nocc_b * (nocc_b - 1) * nvir_b // 2

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

    d_ij_a = e_occ_a[:,None] + e_occ_a
    d_a_a = e_vir_a[:,None]
    D_n_a = -d_a_a + d_ij_a.reshape(-1)
    D_n_a = D_n_a.reshape((nvir_a,nocc_a,nocc_a))
    D_aij_a = D_n_a.copy()[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

    d_ij_b = e_occ_b[:,None] + e_occ_b
    d_a_b = e_vir_b[:,None]
    D_n_b = -d_a_b + d_ij_b.reshape(-1)
    D_n_b = D_n_b.reshape((nvir_b,nocc_b,nocc_b))
    D_aij_b = D_n_b.copy()[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

    d_ij_ab = e_occ_b[:,None] + e_occ_a
    d_a_b = e_vir_b[:,None]
    D_n_bab = -d_a_b + d_ij_ab.reshape(-1)
    D_aij_bab = D_n_bab.reshape(-1)

    d_ij_ab = e_occ_a[:,None] + e_occ_b
    d_a_a = e_vir_a[:,None]
    D_n_aba = -d_a_a + d_ij_ab.reshape(-1)
    D_aij_aba = D_n_aba.reshape(-1)

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

    if M_ij is None:
        M_ij = adc.get_imds()
    M_ij_a, M_ij_b = M_ij

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

        r_aaa = r_aaa.reshape(nvir_a,-1)
        r_bbb = r_bbb.reshape(nvir_b,-1)

        r_aaa_u = None
        r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
        r_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= r_aaa.copy()
        r_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -r_aaa.copy()

        r_bbb_u = None
        r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))
        r_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= r_bbb.copy()
        r_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -r_bbb.copy()

        r_aba = r_aba.reshape(nvir_a,nocc_a,nocc_b)
        r_bab = r_bab.reshape(nvir_b,nocc_b,nocc_a)

        eris_ovoo = eris.ovoo
        eris_OVOO = eris.OVOO
        eris_OVoo = eris.OVoo
        eris_ovOO = eris.ovOO

############ ADC(2) ij block ############################

        s[s_a:f_a] = lib.einsum('ij,j->i',M_ij_a,r_a)
        s[s_b:f_b] = lib.einsum('ij,j->i',M_ij_b,r_b)

############# ADC(2) i - kja block #########################

        s[s_a:f_a] += lib.einsum('jaki,ajk->i', eris_ovoo, r_aaa_u, optimize=True)
        s[s_a:f_a] += lib.einsum('jaki,ajk->i', eris_OVoo, r_bab, optimize=True)

        s[s_b:f_b] += lib.einsum('jaki,ajk->i', eris_OVOO, r_bbb_u, optimize=True)
        s[s_b:f_b] += lib.einsum('jaki,ajk->i', eris_ovOO, r_aba, optimize=True)

        if t1_1_a is not None:
            s[s_a:f_a] += lib.einsum('aAi,a,ia->A', r_aaa_u, e_vir_a, t1_1_a, optimize=True)
            s[s_a:f_a] -= lib.einsum('aAi,i,ia->A', r_aaa_u, e_occ_a, t1_1_a, optimize=True)
            s[s_a:f_a] -= lib.einsum('aiA,a,ia->A', r_bab, e_vir_b, t1_1_b, optimize=True)
            s[s_a:f_a] += lib.einsum('aiA,i,ia->A', r_bab, e_occ_b, t1_1_b, optimize=True)
            s[s_b:f_b] -= lib.einsum('aiA,a,ia->A', r_aba, e_vir_a, t1_1_a, optimize=True)
            s[s_b:f_b] += lib.einsum('aiA,i,ia->A', r_aba, e_occ_a, t1_1_a, optimize=True)
            s[s_b:f_b] += lib.einsum('aAi,a,ia->A', r_bbb_u, e_vir_b, t1_1_b, optimize=True)
            s[s_b:f_b] -= lib.einsum('aAi,i,ia->A', r_bbb_u, e_occ_b, t1_1_b, optimize=True)
            s[s_a:f_a] += lib.einsum('aAi,ia->A', r_aaa_u, f_ov_a, optimize=True)
            s[s_a:f_a] -= lib.einsum('aiA,ia->A', r_bab, f_ov_b, optimize=True)
            s[s_b:f_b] -= lib.einsum('aiA,ia->A', r_aba, f_ov_a, optimize=True)
            s[s_b:f_b] += lib.einsum('aAi,ia->A', r_bbb_u, f_ov_b, optimize=True)

############## ADC(2) ajk - i block ############################

        temp = lib.einsum('jaki,i->ajk', eris_ovoo, r_a, optimize=True)
        temp -= lib.einsum('kaji,i->ajk', eris_ovoo, r_a, optimize=True)
        s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
        s[s_bab:f_bab] += lib.einsum('jaik,i->ajk', eris_OVoo, r_a, optimize=True).reshape(-1)
        s[s_aba:f_aba] += lib.einsum('jaki,i->ajk', eris_ovOO, r_b, optimize=True).reshape(-1)
        temp = lib.einsum('jaki,i->ajk', eris_OVOO, r_b, optimize=True)
        temp -= lib.einsum('kaji,i->ajk', eris_OVOO, r_b, optimize=True)
        s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

############ ADC(2) ajk - bil block ############################

        r_aaa = r_aaa.reshape(-1)
        r_bbb = r_bbb.reshape(-1)

        s[s_aaa:f_aaa] += D_aij_a * r_aaa
        s[s_bab:f_bab] += D_aij_bab * r_bab.reshape(-1)
        s[s_aba:f_aba] += D_aij_aba * r_aba.reshape(-1)
        s[s_bbb:f_bbb] += D_aij_b * r_bbb

############### ADC(3) ajk - bil block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

            eris_oooo = eris.oooo
            eris_OOOO = eris.OOOO
            eris_ooOO = eris.ooOO
            eris_oovv = eris.oovv
            eris_OOVV = eris.OOVV
            eris_ooVV = eris.ooVV
            eris_OOvv = eris.OOvv
            eris_ovvo = eris.ovvo
            eris_OVVO = eris.OVVO
            eris_ovVO = eris.ovVO
            eris_OVvo = eris.OVvo

            r_aaa = r_aaa.reshape(nvir_a,-1)
            r_bab = r_bab.reshape(nvir_b,nocc_b,nocc_a)
            r_aba = r_aba.reshape(nvir_a,nocc_a,nocc_b)
            r_bbb = r_bbb.reshape(nvir_b,-1)

            r_aaa_u = None
            r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
            r_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= r_aaa.copy()
            r_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -r_aaa.copy()

            r_bbb_u = None
            r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))
            r_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= r_bbb.copy()
            r_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -r_bbb.copy()

            temp = lib.einsum('jlki,ail->ajk',eris_oooo,r_aaa_u ,optimize=True)
            s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

            temp = lib.einsum('jlki,ail->ajk',eris_OOOO,r_bbb_u,optimize=True)
            s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

            s[s_bab:f_bab] -= lib.einsum('kijl,ali->ajk',eris_ooOO,
                                             r_bab,optimize=True).reshape(-1)

            s[s_aba:f_aba] -= lib.einsum('jlki,ali->ajk',eris_ooOO,
                                             r_aba,optimize=True).reshape(-1)

            temp = 0.5*lib.einsum('klba,bjl->ajk',eris_oovv,r_aaa_u,optimize=True)
            temp -= 0.5*lib.einsum('kabl,bjl->ajk',eris_ovvo,r_aaa_u,optimize=True)
            temp += 0.5* lib.einsum('kabl,blj->ajk',eris_ovVO,r_bab,optimize=True)
            s[s_aaa:f_aaa] += 2*temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

            s[s_bab:f_bab] += lib.einsum('klba,bjl->ajk',eris_ooVV,
                                             r_bab,optimize=True).reshape(-1)

            temp_1 = 0.5*lib.einsum('klba,bjl->ajk',eris_OOVV,r_bbb_u,optimize=True)
            temp_1 -= 0.5*lib.einsum('kabl,bjl->ajk',eris_OVVO,r_bbb_u,optimize=True)
            temp_1 += 0.5*lib.einsum('kabl,blj->ajk',eris_OVvo,r_aba,optimize=True)
            s[s_bbb:f_bbb] += 2*temp_1[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

            s[s_aba:f_aba] += lib.einsum('klba,bjl->ajk',eris_OOvv,
                                             r_aba,optimize=True).reshape(-1)

            temp = -0.5*lib.einsum('jlba,bkl->ajk',eris_oovv,r_aaa_u,optimize=True)
            temp += 0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo,r_aaa_u,optimize=True)
            temp -= 0.5*lib.einsum('jabl,blk->ajk',eris_ovVO,r_bab,optimize=True)
            s[s_aaa:f_aaa] += 2*temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

            s[s_bab:f_bab] +=  lib.einsum('jabl,bkl->ajk',
                                              eris_OVvo,r_aaa_u,optimize=True).reshape(-1)
            s[s_bab:f_bab] +=  lib.einsum('jlba,blk->ajk',
                                              eris_OOVV,r_bab,optimize=True).reshape(-1)
            s[s_bab:f_bab] -=  lib.einsum('jabl,blk->ajk',
                                              eris_OVVO,r_bab,optimize=True).reshape(-1)

            temp = -0.5*lib.einsum('jlba,bkl->ajk',eris_OOVV,r_bbb_u,optimize=True)
            temp += 0.5*lib.einsum('jabl,bkl->ajk',eris_OVVO,r_bbb_u,optimize=True)
            temp -= 0.5*lib.einsum('jabl,blk->ajk',eris_OVvo,r_aba,optimize=True)
            s[s_bbb:f_bbb] += 2*temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

            s[s_aba:f_aba] += lib.einsum('jlba,blk->ajk',eris_oovv,
                                             r_aba,optimize=True).reshape(-1)
            s[s_aba:f_aba] -= lib.einsum('jabl,blk->ajk',eris_ovvo,
                                             r_aba,optimize=True).reshape(-1)
            s[s_aba:f_aba] += lib.einsum('jabl,bkl->ajk',eris_ovVO,
                                             r_bbb_u,optimize=True).reshape(-1)

        if (method == "adc(3)"):
            if t1_1_a is None:

                eris_ovoo = eris.ovoo
                eris_OVOO = eris.OVOO
                eris_ovOO = eris.ovOO
                eris_OVoo = eris.OVoo

    ################ ADC(3) i - kja and ajk - i block ############################
                t2_1_a = adc.t2[0][0][:]
                t2_1_ab = adc.t2[0][1][:]

                temp_singles = np.zeros(nocc_a)
                temp_doubles = np.zeros((nvir_a, nvir_a, nvir_a))
                r_aaa = r_aaa.reshape(nvir_a,-1)
                t2_1_a_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:]
                temp_1 = lib.einsum('pbc,ap->abc',t2_1_a_t,r_aaa, optimize=True)
                if eris.ovvv is None:
                    chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                    for a,b in lib.prange(0,nocc_a,chnk_size):
                        eris_ovvv = dfadc.get_ovvv_spin_df(
                            adc, eris.Lov, eris.Lvv, a, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
                        temp_singles[a:b] += \
                            lib.einsum('abc,icab->i',temp_1, eris_ovvv, optimize=True)

                        temp_doubles += lib.einsum('i,icab->bca',r_a[a:b],eris_ovvv,optimize=True)
                        temp_doubles -= lib.einsum('i,ibac->bca',r_a[a:b],eris_ovvv,optimize=True)
                        del eris_ovvv
                else :
                    eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
                    temp_singles += lib.einsum('abc,icab->i',temp_1, eris_ovvv, optimize=True)

                    temp_doubles += lib.einsum('i,icab->bca',r_a,eris_ovvv,optimize=True)
                    temp_doubles -= lib.einsum('i,ibac->bca',r_a,eris_ovvv,optimize=True)
                    del eris_ovvv

                s[s_a:f_a] += temp_singles
                s[s_aaa:f_aaa] += 0.5*lib.einsum('bca,pbc->ap',temp_doubles,
                                                 t2_1_a_t,optimize=True).reshape(-1)
                del temp_singles
                del temp_doubles

                r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
                r_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= r_aaa.copy()
                r_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -r_aaa.copy()

                r_aba = r_aba.reshape(nvir_a,nocc_a,nocc_b)

                temp = lib.einsum('jlab,ajk->blk',t2_1_a,r_aaa_u,optimize=True)
                s[s_a:f_a] += lib.einsum('blk,lbik->i',temp,eris_ovoo,optimize=True)
                s[s_a:f_a] -= lib.einsum('blk,iblk->i',temp,eris_ovoo,optimize=True)

                temp_1 = lib.einsum('jlab,ajk->blk',t2_1_a,r_aba,optimize=True)
                s[s_b:f_b] += lib.einsum('blk,lbik->i',temp_1,eris_ovOO,optimize=True)

                temp_1 = lib.einsum('i,lbik->kbl',r_a, eris_ovoo)
                temp_1 -= lib.einsum('i,iblk->kbl',r_a, eris_ovoo)

                temp  = lib.einsum('kbl,jlab->ajk',temp_1,t2_1_a,optimize=True)
                s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1] ].reshape(-1)

                temp_2  = lib.einsum('i,lbik->kbl',r_b,eris_ovOO)
                temp = lib.einsum('kbl,jlab->ajk',temp_2,t2_1_a,optimize=True)
                s[s_aba:f_aba] += temp.reshape(-1)

                temp_1 = lib.einsum('i,lbij->jbl',r_a, eris_ovoo)
                temp_1 -= lib.einsum('i,iblj->jbl',r_a, eris_ovoo)

                temp  = lib.einsum('jbl,klab->ajk',temp_1,t2_1_a,optimize=True)
                s[s_aaa:f_aaa] -= temp[:,ij_ind_a[0],ij_ind_a[1] ].reshape(-1)

                t2_1_b = adc.t2[0][2][:]

                temp_singles = np.zeros(nocc_b)
                temp_doubles = np.zeros((nvir_b, nvir_b, nvir_b))
                r_bbb = r_bbb.reshape(nvir_b,-1)
                t2_1_b_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:]
                temp_1 = lib.einsum('pbc,ap->abc',t2_1_b_t,r_bbb, optimize=True)
                if eris.OVVV is None:
                    chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                    a = 0
                    for a,b in lib.prange(0,nocc_b,chnk_size):
                        eris_OVVV = dfadc.get_ovvv_spin_df(
                            adc, eris.LOV, eris.LVV, a, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
                        temp_singles[a:b] += \
                            lib.einsum('abc,icab->i',temp_1, eris_OVVV, optimize=True)

                        temp_doubles += lib.einsum('i,icab->bca',r_b[a:b],eris_OVVV,optimize=True)
                        temp_doubles -= lib.einsum('i,ibac->bca',r_b[a:b],eris_OVVV,optimize=True)
                        del eris_OVVV
                else :
                    eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
                    temp_singles += lib.einsum('abc,icab->i',temp_1, eris_OVVV, optimize=True)

                    temp_doubles += lib.einsum('i,icab->bca',r_b,eris_OVVV,optimize=True)
                    temp_doubles -= lib.einsum('i,ibac->bca',r_b,eris_OVVV,optimize=True)
                    del eris_OVVV

                s[s_b:f_b] += temp_singles
                s[s_bbb:f_bbb] += 0.5*lib.einsum('bca,pbc->ap',temp_doubles,
                                                 t2_1_b_t,optimize=True).reshape(-1)
                del temp_singles
                del temp_doubles

                r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))
                r_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= r_bbb.copy()
                r_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -r_bbb.copy()

                r_bab = r_bab.reshape(nvir_b,nocc_b,nocc_a)

                temp_1 = lib.einsum('jlab,ajk->blk',t2_1_b,r_bab,optimize=True)
                s[s_a:f_a] += lib.einsum('blk,lbik->i',temp_1,eris_OVoo,optimize=True)

                temp = lib.einsum('jlab,ajk->blk',t2_1_b,r_bbb_u,optimize=True)

                s[s_b:f_b] += lib.einsum('blk,lbik->i',temp,eris_OVOO,optimize=True)
                s[s_b:f_b] -= lib.einsum('blk,iblk->i',temp,eris_OVOO,optimize=True)

                temp_2  = lib.einsum('i,lbik->kbl',r_a,eris_OVoo)
                temp = lib.einsum('kbl,jlab->ajk',temp_2,t2_1_b,optimize=True)
                s[s_bab:f_bab] += temp.reshape(-1)

                temp_1 = lib.einsum('i,lbik->kbl',r_b, eris_OVOO)
                temp_1 -= lib.einsum('i,iblk->kbl',r_b, eris_OVOO)

                temp  = lib.einsum('kbl,jlab->ajk',temp_1,t2_1_b,optimize=True)
                s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1] ].reshape(-1)

                temp_1 = lib.einsum('i,lbij->jbl',r_b, eris_OVOO)
                temp_1 -= lib.einsum('i,iblj->jbl',r_b, eris_OVOO)

                temp  = lib.einsum('jbl,klab->ajk',temp_1,t2_1_b,optimize=True)
                s[s_bbb:f_bbb] -= temp[:,ij_ind_b[0],ij_ind_b[1] ].reshape(-1)

                temp_1 = lib.einsum('kjcb,ajk->abc',t2_1_ab,r_bab, optimize=True)
                temp_2 = np.zeros((nvir_a, nvir_b, nvir_b))
                if eris.ovVV is None:
                    chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                    for a,b in lib.prange(0,nocc_a,chnk_size):
                        eris_ovVV = dfadc.get_ovvv_spin_df(
                            adc, eris.Lov, eris.LVV, a, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)

                        s[s_a:f_a][a:b] += lib.einsum('abc,icab->i',temp_1, eris_ovVV, optimize=True)

                        temp_2 += lib.einsum('i,icab->cba',r_a[a:b],eris_ovVV,optimize=True)
                        del eris_ovVV
                else :
                    eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
                    s[s_a:f_a] += lib.einsum('abc,icab->i',temp_1, eris_ovVV, optimize=True)
                    temp_2 += lib.einsum('i,icab->cba',r_a,eris_ovVV,optimize=True)
                    del eris_ovVV

                s[s_bab:f_bab] += lib.einsum('cba,kjcb->ajk',temp_2, t2_1_ab, optimize=True).reshape(-1)
                del temp_1
                del temp_2

                temp_1 = lib.einsum('jkbc,ajk->abc',t2_1_ab,r_aba, optimize=True)
                temp_2 = np.zeros((nvir_a, nvir_b, nvir_a))
                if eris.OVvv is None:
                    chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
                    a = 0
                    for a,b in lib.prange(0,nocc_b,chnk_size):
                        eris_OVvv = dfadc.get_ovvv_spin_df(
                            adc, eris.LOV, eris.Lvv, a, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
                        s[s_b:f_b][a:b] += lib.einsum('abc,icab->i',temp_1, eris_OVvv, optimize=True)

                        temp_2 += lib.einsum('i,icab->bca',r_b[a:b],eris_OVvv,optimize=True)
                        del eris_OVvv
                else :
                    eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
                    s[s_b:f_b] += lib.einsum('abc,icab->i',temp_1, eris_OVvv, optimize=True)

                    temp_2 += lib.einsum('i,icab->bca',r_b,eris_OVvv,optimize=True)
                    del eris_OVvv

                s[s_aba:f_aba] += lib.einsum('bca,jkbc->ajk',temp_2, t2_1_ab, optimize=True).reshape(-1)
                del temp_1
                del temp_2

                temp = lib.einsum('ljba,ajk->blk',t2_1_ab,r_bab,optimize=True)
                temp_1 = lib.einsum('jlab,ajk->blk',t2_1_ab,r_aaa_u,optimize=True)
                temp_2 = lib.einsum('jlba,akj->blk',t2_1_ab,r_bab, optimize=True)

                s[s_a:f_a] += lib.einsum('blk,lbik->i',temp,eris_ovoo,optimize=True)
                s[s_a:f_a] -= lib.einsum('blk,iblk->i',temp,eris_ovoo,optimize=True)
                s[s_a:f_a] += lib.einsum('blk,lbik->i',temp_1,eris_OVoo,optimize=True)
                s[s_a:f_a] -= lib.einsum('blk,iblk->i',temp_2,eris_ovOO,optimize=True)

                temp = lib.einsum('jlab,ajk->blk',t2_1_ab,r_aba,optimize=True)
                temp_1 = lib.einsum('ljba,ajk->blk',t2_1_ab,r_bbb_u,optimize=True)
                temp_2 = lib.einsum('ljab,akj->blk',t2_1_ab,r_aba,optimize=True)

                s[s_b:f_b] += lib.einsum('blk,lbik->i',temp,eris_OVOO,optimize=True)
                s[s_b:f_b] -= lib.einsum('blk,iblk->i',temp,eris_OVOO,optimize=True)
                s[s_b:f_b] += lib.einsum('blk,lbik->i',temp_1,eris_ovOO,optimize=True)
                s[s_b:f_b] -= lib.einsum('blk,iblk->i',temp_2,eris_OVoo,optimize=True)

                temp_2 = lib.einsum('i,lbik->kbl',r_a, eris_OVoo)
                temp = lib.einsum('kbl,jlab->ajk',temp_2,t2_1_ab,optimize=True)
                s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1] ].reshape(-1)

                temp_1  = lib.einsum('i,lbik->kbl',r_a,eris_ovoo)
                temp_1  -= lib.einsum('i,iblk->kbl',r_a,eris_ovoo)

                temp  = lib.einsum('kbl,ljba->ajk',temp_1,t2_1_ab,optimize=True)
                s[s_bab:f_bab] += temp.reshape(-1)

                temp_2 = lib.einsum('i,lbik->kbl',r_b, eris_ovOO)

                temp = lib.einsum('kbl,ljba->ajk',temp_2,t2_1_ab,optimize=True)
                s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1] ].reshape(-1)

                temp_1  = lib.einsum('i,lbik->kbl',r_b,eris_OVOO)
                temp_1  -= lib.einsum('i,iblk->kbl',r_b,eris_OVOO)

                temp  = lib.einsum('kbl,jlab->ajk',temp_1,t2_1_ab,optimize=True)
                s[s_aba:f_aba] += temp.reshape(-1)

                temp_2 = lib.einsum('i,lbij->jbl',r_a, eris_OVoo)

                temp = lib.einsum('jbl,klab->ajk',temp_2,t2_1_ab,optimize=True)
                s[s_aaa:f_aaa] -= temp[:,ij_ind_a[0],ij_ind_a[1] ].reshape(-1)

                temp  = -lib.einsum('i,iblj->jbl',r_a,eris_ovOO,optimize=True)
                temp_1 = -lib.einsum('jbl,klba->ajk',temp,t2_1_ab,optimize=True)
                s[s_bab:f_bab] -= temp_1.reshape(-1)

                temp_2 = lib.einsum('i,lbij->jbl',r_b, eris_ovOO)
                temp = lib.einsum('jbl,lkba->ajk',temp_2,t2_1_ab,optimize=True)
                s[s_bbb:f_bbb] -= temp[:,ij_ind_b[0],ij_ind_b[1] ].reshape(-1)

                temp  = -lib.einsum('i,iblj->jbl',r_b,eris_OVoo,optimize=True)
                temp_1 = -lib.einsum('jbl,lkab->ajk',temp,t2_1_ab,optimize=True)
                s[s_aba:f_aba] -= temp_1.reshape(-1)

            else:
                t2_1_a = adc.t2[0][0][:]
                t2_1_ab = adc.t2[0][1][:]
                t2_1_b = adc.t2[0][2][:]
                t1_2_a = adc.t1[0][0][:]
                t1_2_b = adc.t1[0][1][:]
                eris_OOOO = eris.OOOO
                eris_OOVV = eris.OOVV
                eris_OOvv = eris.OOvv
                eris_OVOO = eris.OVOO
                eris_OVVO = eris.OVVO
                eris_OVoo = eris.OVoo
                eris_OVvo = eris.OVvo
                eris_ooOO = eris.ooOO
                eris_ooVV = eris.ooVV
                eris_oooo = eris.oooo
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
                s[s_a:f_a] += lib.einsum('aAi,a,ia->A', r_aaa_u, e_vir_a, t1_2_a, optimize=True)
                s[s_a:f_a] -= lib.einsum('aAi,i,ia->A', r_aaa_u, e_occ_a, t1_2_a, optimize=True)
                s[s_a:f_a] += lib.einsum('aAi,jb,ijab->A', r_aaa_u, f_ov_a, t2_1_a, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('aij,Ab,ijab->A', r_aaa_u, f_ov_a, t2_1_a, optimize=True)
                s[s_a:f_a] += lib.einsum('aAi,jb,ijab->A', r_aaa_u, f_ov_b, t2_1_ab, optimize=True)
                s[s_a:f_a] += 2 *  lib.einsum('aAi,jb,iabj->A', r_aaa_u, t1_1_a, eris_ovvo, optimize=True)
                s[s_a:f_a] -= lib.einsum('aAi,jb,jabi->A', r_aaa_u, t1_1_a, eris_ovvo, optimize=True)
                s[s_a:f_a] -= lib.einsum('aAi,jb,jiab->A', r_aaa_u, t1_1_a, eris_oovv, optimize=True)
                s[s_a:f_a] += lib.einsum('aij,Ab,iabj->A', r_aaa_u, t1_1_a, eris_ovvo, optimize=True)
                s[s_a:f_a] -= lib.einsum('aij,ib,Abaj->A', r_aaa_u, t1_1_a, eris_ovvo, optimize=True)
                s[s_a:f_a] += lib.einsum('aij,ib,Ajab->A', r_aaa_u, t1_1_a, eris_oovv, optimize=True)
                s[s_a:f_a] += lib.einsum('aij,ka,Aikj->A', r_aaa_u, t1_1_a, eris_oooo, optimize=True)
                s[s_a:f_a] += 2 *  lib.einsum('aAi,jb,iabj->A', r_aaa_u, t1_1_b, eris_ovVO, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('aAi,ijbc,jbac->A', r_aaa_u, t2_1_a, eris_ovvv, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('aAi,ijbc,jcab->A', r_aaa_u, t2_1_a, eris_ovvv, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('aAi,jkab,kbji->A', r_aaa_u, t2_1_a, eris_ovoo, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('aAi,jkab,jbki->A', r_aaa_u, t2_1_a, eris_ovoo, optimize=True)
                s[s_a:f_a] -= 1/4 *  lib.einsum('aij,ijbc,Abac->A', r_aaa_u, t2_1_a, eris_ovvv, optimize=True)
                s[s_a:f_a] += 1/4 *  lib.einsum('aij,ijbc,Acab->A', r_aaa_u, t2_1_a, eris_ovvv, optimize=True)
                s[s_a:f_a] += lib.einsum('aij,ikab,kbAj->A', r_aaa_u, t2_1_a, eris_ovoo, optimize=True)
                s[s_a:f_a] -= lib.einsum('aij,ikab,Abkj->A', r_aaa_u, t2_1_a, eris_ovoo, optimize=True)
                s[s_a:f_a] += lib.einsum('aAi,ijbc,jcab->A', r_aaa_u, t2_1_ab, eris_OVvv, optimize=True)
                s[s_a:f_a] -= lib.einsum('aAi,jkab,kbji->A', r_aaa_u, t2_1_ab, eris_OVoo, optimize=True)
                s[s_a:f_a] += lib.einsum('aij,ikab,kbAj->A', r_aaa_u, t2_1_ab, eris_OVoo, optimize=True)
                s[s_a:f_a] -= lib.einsum('aiA,a,ia->A', r_bab, e_vir_b, t1_2_b, optimize=True)
                s[s_a:f_a] += lib.einsum('aiA,i,ia->A', r_bab, e_occ_b, t1_2_b, optimize=True)
                s[s_a:f_a] -= lib.einsum('aiA,jb,jiba->A', r_bab, f_ov_a, t2_1_ab, optimize=True)
                s[s_a:f_a] += lib.einsum('aij,Ab,jiba->A', r_bab, f_ov_a, t2_1_ab, optimize=True)
                s[s_a:f_a] -= lib.einsum('aiA,jb,ijab->A', r_bab, f_ov_b, t2_1_b, optimize=True)
                s[s_a:f_a] -= 2 *  lib.einsum('aiA,jb,iabj->A', r_bab, t1_1_a, eris_OVvo, optimize=True)
                s[s_a:f_a] += lib.einsum('aij,Ab,iabj->A', r_bab, t1_1_a, eris_OVvo, optimize=True)
                s[s_a:f_a] += lib.einsum('aij,jb,Abai->A', r_bab, t1_1_a, eris_ovVO, optimize=True)
                s[s_a:f_a] -= 2 *  lib.einsum('aiA,jb,iabj->A', r_bab, t1_1_b, eris_OVVO, optimize=True)
                s[s_a:f_a] += lib.einsum('aiA,jb,jabi->A', r_bab, t1_1_b, eris_OVVO, optimize=True)
                s[s_a:f_a] += lib.einsum('aiA,jb,jiab->A', r_bab, t1_1_b, eris_OOVV, optimize=True)
                s[s_a:f_a] += lib.einsum('aij,ib,Ajab->A', r_bab, t1_1_b, eris_ooVV, optimize=True)
                s[s_a:f_a] -= lib.einsum('aij,ka,Ajki->A', r_bab, t1_1_b, eris_ooOO, optimize=True)
                s[s_a:f_a] -= lib.einsum('aiA,jibc,jbac->A', r_bab, t2_1_ab, eris_ovVV, optimize=True)
                s[s_a:f_a] += lib.einsum('aiA,jkba,jbki->A', r_bab, t2_1_ab, eris_ovOO, optimize=True)
                s[s_a:f_a] += lib.einsum('aij,jibc,Abac->A', r_bab, t2_1_ab, eris_ovVV, optimize=True)
                s[s_a:f_a] -= lib.einsum('aij,jkba,Abki->A', r_bab, t2_1_ab, eris_ovOO, optimize=True)
                s[s_a:f_a] += lib.einsum('aij,kiba,kbAj->A', r_bab, t2_1_ab, eris_ovoo, optimize=True)
                s[s_a:f_a] -= lib.einsum('aij,kiba,Abkj->A', r_bab, t2_1_ab, eris_ovoo, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('aiA,ijbc,jbac->A', r_bab, t2_1_b, eris_OVVV, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('aiA,ijbc,jcab->A', r_bab, t2_1_b, eris_OVVV, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('aiA,jkab,kbji->A', r_bab, t2_1_b, eris_OVOO, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('aiA,jkab,jbki->A', r_bab, t2_1_b, eris_OVOO, optimize=True)
                s[s_a:f_a] += lib.einsum('aij,ikab,kbAj->A', r_bab, t2_1_b, eris_OVoo, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('aAi,a,jb,ijab->A', r_aaa_u, e_vir_a, t1_1_a, t2_1_a, optimize=True)
                s[s_a:f_a] += lib.einsum('aAi,b,jb,ijab->A', r_aaa_u, e_vir_a, t1_1_a, t2_1_a, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('aAi,i,jb,ijab->A', r_aaa_u, e_occ_a, t1_1_a, t2_1_a, optimize=True)
                s[s_a:f_a] -= lib.einsum('aAi,j,jb,ijab->A', r_aaa_u, e_occ_a, t1_1_a, t2_1_a, optimize=True)
                s[s_a:f_a] -= 1/4 *  lib.einsum('aij,A,Ab,ijab->A', r_aaa_u, e_occ_a, t1_1_a, t2_1_a, optimize=True)
                s[s_a:f_a] += 1/4 *  lib.einsum('aij,a,Ab,ijab->A', r_aaa_u, e_vir_a, t1_1_a, t2_1_a, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('aij,b,Ab,ijab->A', r_aaa_u, e_vir_a, t1_1_a, t2_1_a, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('aij,j,Ab,ijab->A', r_aaa_u, e_occ_a, t1_1_a, t2_1_a, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('aAi,a,jb,ijab->A', r_aaa_u, e_vir_a, t1_1_b, t2_1_ab, optimize=True)
                s[s_a:f_a] += lib.einsum('aAi,b,jb,ijab->A', r_aaa_u, e_vir_b, t1_1_b, t2_1_ab, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('aAi,i,jb,ijab->A', r_aaa_u, e_occ_a, t1_1_b, t2_1_ab, optimize=True)
                s[s_a:f_a] -= lib.einsum('aAi,j,jb,ijab->A', r_aaa_u, e_occ_b, t1_1_b, t2_1_ab, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('aiA,a,jb,jiba->A', r_bab, e_vir_b, t1_1_a, t2_1_ab, optimize=True)
                s[s_a:f_a] -= lib.einsum('aiA,b,jb,jiba->A', r_bab, e_vir_a, t1_1_a, t2_1_ab, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('aiA,i,jb,jiba->A', r_bab, e_occ_b, t1_1_a, t2_1_ab, optimize=True)
                s[s_a:f_a] += lib.einsum('aiA,j,jb,jiba->A', r_bab, e_occ_a, t1_1_a, t2_1_ab, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('aij,A,Ab,jiba->A', r_bab, e_occ_a, t1_1_a, t2_1_ab, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('aij,a,Ab,jiba->A', r_bab, e_vir_b, t1_1_a, t2_1_ab, optimize=True)
                s[s_a:f_a] += lib.einsum('aij,b,Ab,jiba->A', r_bab, e_vir_a, t1_1_a, t2_1_ab, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('aij,i,Ab,jiba->A', r_bab, e_occ_b, t1_1_a, t2_1_ab, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('aij,j,Ab,jiba->A', r_bab, e_occ_a, t1_1_a, t2_1_ab, optimize=True)
                s[s_a:f_a] -= 1/2 *  lib.einsum('aiA,a,jb,ijab->A', r_bab, e_vir_b, t1_1_b, t2_1_b, optimize=True)
                s[s_a:f_a] -= lib.einsum('aiA,b,jb,ijab->A', r_bab, e_vir_b, t1_1_b, t2_1_b, optimize=True)
                s[s_a:f_a] += 1/2 *  lib.einsum('aiA,i,jb,ijab->A', r_bab, e_occ_b, t1_1_b, t2_1_b, optimize=True)
                s[s_a:f_a] += lib.einsum('aiA,j,jb,ijab->A', r_bab, e_occ_b, t1_1_b, t2_1_b, optimize=True)
                s[s_b:f_b] -= lib.einsum('aiA,a,ia->A', r_aba, e_vir_a, t1_2_a, optimize=True)
                s[s_b:f_b] += lib.einsum('aiA,i,ia->A', r_aba, e_occ_a, t1_2_a, optimize=True)
                s[s_b:f_b] -= lib.einsum('aiA,jb,ijab->A', r_aba, f_ov_a, t2_1_a, optimize=True)
                s[s_b:f_b] -= lib.einsum('aiA,jb,ijab->A', r_aba, f_ov_b, t2_1_ab, optimize=True)
                s[s_b:f_b] += lib.einsum('aij,Ab,ijab->A', r_aba, f_ov_b, t2_1_ab, optimize=True)
                s[s_b:f_b] -= 2 *  lib.einsum('aiA,jb,iabj->A', r_aba, t1_1_a, eris_ovvo, optimize=True)
                s[s_b:f_b] += lib.einsum('aiA,jb,jabi->A', r_aba, t1_1_a, eris_ovvo, optimize=True)
                s[s_b:f_b] += lib.einsum('aiA,jb,jiab->A', r_aba, t1_1_a, eris_oovv, optimize=True)
                s[s_b:f_b] += lib.einsum('aij,ib,Ajab->A', r_aba, t1_1_a, eris_OOvv, optimize=True)
                s[s_b:f_b] -= lib.einsum('aij,ka,kiAj->A', r_aba, t1_1_a, eris_ooOO, optimize=True)
                s[s_b:f_b] -= 2 *  lib.einsum('aiA,jb,iabj->A', r_aba, t1_1_b, eris_ovVO, optimize=True)
                s[s_b:f_b] += lib.einsum('aij,Ab,iabj->A', r_aba, t1_1_b, eris_ovVO, optimize=True)
                s[s_b:f_b] += lib.einsum('aij,jb,Abai->A', r_aba, t1_1_b, eris_OVvo, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('aiA,ijbc,jbac->A', r_aba, t2_1_a, eris_ovvv, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('aiA,ijbc,jcab->A', r_aba, t2_1_a, eris_ovvv, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('aiA,jkab,kbji->A', r_aba, t2_1_a, eris_ovoo, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('aiA,jkab,jbki->A', r_aba, t2_1_a, eris_ovoo, optimize=True)
                s[s_b:f_b] += lib.einsum('aij,ikab,kbAj->A', r_aba, t2_1_a, eris_ovOO, optimize=True)
                s[s_b:f_b] -= lib.einsum('aiA,ijbc,jcab->A', r_aba, t2_1_ab, eris_OVvv, optimize=True)
                s[s_b:f_b] += lib.einsum('aiA,jkab,kbji->A', r_aba, t2_1_ab, eris_OVoo, optimize=True)
                s[s_b:f_b] += lib.einsum('aij,ijbc,Acab->A', r_aba, t2_1_ab, eris_OVvv, optimize=True)
                s[s_b:f_b] += lib.einsum('aij,ikab,kbAj->A', r_aba, t2_1_ab, eris_OVOO, optimize=True)
                s[s_b:f_b] -= lib.einsum('aij,ikab,Abkj->A', r_aba, t2_1_ab, eris_OVOO, optimize=True)
                s[s_b:f_b] -= lib.einsum('aij,kjab,Abki->A', r_aba, t2_1_ab, eris_OVoo, optimize=True)
                s[s_b:f_b] += lib.einsum('aAi,a,ia->A', r_bbb_u, e_vir_b, t1_2_b, optimize=True)
                s[s_b:f_b] -= lib.einsum('aAi,i,ia->A', r_bbb_u, e_occ_b, t1_2_b, optimize=True)
                s[s_b:f_b] += lib.einsum('aAi,jb,jiba->A', r_bbb_u, f_ov_a, t2_1_ab, optimize=True)
                s[s_b:f_b] += lib.einsum('aAi,jb,ijab->A', r_bbb_u, f_ov_b, t2_1_b, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('aij,Ab,ijab->A', r_bbb_u, f_ov_b, t2_1_b, optimize=True)
                s[s_b:f_b] += 2 *  lib.einsum('aAi,jb,iabj->A', r_bbb_u, t1_1_a, eris_OVvo, optimize=True)
                s[s_b:f_b] += 2 *  lib.einsum('aAi,jb,iabj->A', r_bbb_u, t1_1_b, eris_OVVO, optimize=True)
                s[s_b:f_b] -= lib.einsum('aAi,jb,jabi->A', r_bbb_u, t1_1_b, eris_OVVO, optimize=True)
                s[s_b:f_b] -= lib.einsum('aAi,jb,jiab->A', r_bbb_u, t1_1_b, eris_OOVV, optimize=True)
                s[s_b:f_b] += lib.einsum('aij,Ab,iabj->A', r_bbb_u, t1_1_b, eris_OVVO, optimize=True)
                s[s_b:f_b] -= lib.einsum('aij,ib,Abaj->A', r_bbb_u, t1_1_b, eris_OVVO, optimize=True)
                s[s_b:f_b] += lib.einsum('aij,ib,Ajab->A', r_bbb_u, t1_1_b, eris_OOVV, optimize=True)
                s[s_b:f_b] += lib.einsum('aij,ka,Aikj->A', r_bbb_u, t1_1_b, eris_OOOO, optimize=True)
                s[s_b:f_b] += lib.einsum('aAi,jibc,jbac->A', r_bbb_u, t2_1_ab, eris_ovVV, optimize=True)
                s[s_b:f_b] -= lib.einsum('aAi,jkba,jbki->A', r_bbb_u, t2_1_ab, eris_ovOO, optimize=True)
                s[s_b:f_b] += lib.einsum('aij,kiba,kbAj->A', r_bbb_u, t2_1_ab, eris_ovOO, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('aAi,ijbc,jbac->A', r_bbb_u, t2_1_b, eris_OVVV, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('aAi,ijbc,jcab->A', r_bbb_u, t2_1_b, eris_OVVV, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('aAi,jkab,kbji->A', r_bbb_u, t2_1_b, eris_OVOO, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('aAi,jkab,jbki->A', r_bbb_u, t2_1_b, eris_OVOO, optimize=True)
                s[s_b:f_b] -= 1/4 *  lib.einsum('aij,ijbc,Abac->A', r_bbb_u, t2_1_b, eris_OVVV, optimize=True)
                s[s_b:f_b] += 1/4 *  lib.einsum('aij,ijbc,Acab->A', r_bbb_u, t2_1_b, eris_OVVV, optimize=True)
                s[s_b:f_b] += lib.einsum('aij,ikab,kbAj->A', r_bbb_u, t2_1_b, eris_OVOO, optimize=True)
                s[s_b:f_b] -= lib.einsum('aij,ikab,Abkj->A', r_bbb_u, t2_1_b, eris_OVOO, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('aiA,a,jb,ijab->A', r_aba, e_vir_a, t1_1_a, t2_1_a, optimize=True)
                s[s_b:f_b] -= lib.einsum('aiA,b,jb,ijab->A', r_aba, e_vir_a, t1_1_a, t2_1_a, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('aiA,i,jb,ijab->A', r_aba, e_occ_a, t1_1_a, t2_1_a, optimize=True)
                s[s_b:f_b] += lib.einsum('aiA,j,jb,ijab->A', r_aba, e_occ_a, t1_1_a, t2_1_a, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('aiA,a,jb,ijab->A', r_aba, e_vir_a, t1_1_b, t2_1_ab, optimize=True)
                s[s_b:f_b] -= lib.einsum('aiA,b,jb,ijab->A', r_aba, e_vir_b, t1_1_b, t2_1_ab, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('aiA,i,jb,ijab->A', r_aba, e_occ_a, t1_1_b, t2_1_ab, optimize=True)
                s[s_b:f_b] += lib.einsum('aiA,j,jb,ijab->A', r_aba, e_occ_b, t1_1_b, t2_1_ab, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('aij,A,Ab,ijab->A', r_aba, e_occ_b, t1_1_b, t2_1_ab, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('aij,a,Ab,ijab->A', r_aba, e_vir_a, t1_1_b, t2_1_ab, optimize=True)
                s[s_b:f_b] += lib.einsum('aij,b,Ab,ijab->A', r_aba, e_vir_b, t1_1_b, t2_1_ab, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('aij,i,Ab,ijab->A', r_aba, e_occ_a, t1_1_b, t2_1_ab, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('aij,j,Ab,ijab->A', r_aba, e_occ_b, t1_1_b, t2_1_ab, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('aAi,a,jb,jiba->A', r_bbb_u, e_vir_b, t1_1_a, t2_1_ab, optimize=True)
                s[s_b:f_b] += lib.einsum('aAi,b,jb,jiba->A', r_bbb_u, e_vir_a, t1_1_a, t2_1_ab, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('aAi,i,jb,jiba->A', r_bbb_u, e_occ_b, t1_1_a, t2_1_ab, optimize=True)
                s[s_b:f_b] -= lib.einsum('aAi,j,jb,jiba->A', r_bbb_u, e_occ_a, t1_1_a, t2_1_ab, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('aAi,a,jb,ijab->A', r_bbb_u, e_vir_b, t1_1_b, t2_1_b, optimize=True)
                s[s_b:f_b] += lib.einsum('aAi,b,jb,ijab->A', r_bbb_u, e_vir_b, t1_1_b, t2_1_b, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('aAi,i,jb,ijab->A', r_bbb_u, e_occ_b, t1_1_b, t2_1_b, optimize=True)
                s[s_b:f_b] -= lib.einsum('aAi,j,jb,ijab->A', r_bbb_u, e_occ_b, t1_1_b, t2_1_b, optimize=True)
                s[s_b:f_b] -= 1/4 *  lib.einsum('aij,A,Ab,ijab->A', r_bbb_u, e_occ_b, t1_1_b, t2_1_b, optimize=True)
                s[s_b:f_b] += 1/4 *  lib.einsum('aij,a,Ab,ijab->A', r_bbb_u, e_vir_b, t1_1_b, t2_1_b, optimize=True)
                s[s_b:f_b] += 1/2 *  lib.einsum('aij,b,Ab,ijab->A', r_bbb_u, e_vir_b, t1_1_b, t2_1_b, optimize=True)
                s[s_b:f_b] -= 1/2 *  lib.einsum('aij,j,Ab,ijab->A', r_bbb_u, e_occ_b, t1_1_b, t2_1_b, optimize=True)
                s[s_aaa:f_aaa] += lib.einsum('i,ia,BCAa->ABC', r_a, f_ov_a, t2_1_a, optimize=True)[:, ij_ind_a[0],
                    ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] -= lib.einsum('i,Ba,CAai->ABC', r_a, t1_1_a, eris_ovvo, optimize=True)[:, ij_ind_a[0],
                    ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] += lib.einsum('i,Ba,iCAa->ABC', r_a, t1_1_a, eris_oovv, optimize=True)[:, ij_ind_a[0],
                    ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] += lib.einsum('i,Ca,BAai->ABC', r_a, t1_1_a, eris_ovvo, optimize=True)[:, ij_ind_a[0],
                    ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] -= lib.einsum('i,Ca,iBAa->ABC', r_a, t1_1_a, eris_oovv, optimize=True)[:, ij_ind_a[0],
                    ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] += lib.einsum('i,ia,BAaC->ABC', r_a, t1_1_a, eris_ovvo, optimize=True)[:, ij_ind_a[0],
                    ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] -= lib.einsum('i,ia,CAaB->ABC', r_a, t1_1_a, eris_ovvo, optimize=True)[:, ij_ind_a[0],
                    ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] += lib.einsum('i,jA,BiCj->ABC', r_a, t1_1_a, eris_oooo, optimize=True)[:, ij_ind_a[0],
                    ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] -= lib.einsum('i,jA,BjCi->ABC', r_a, t1_1_a, eris_oooo, optimize=True)[:, ij_ind_a[0],
                    ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] -= 1/2 *  lib.einsum('i,BCab,iaAb->ABC', r_a, t2_1_a, eris_ovvv, optimize=True)[:,
                    ij_ind_a[0], ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] += 1/2 *  lib.einsum('i,BCab,ibAa->ABC', r_a, t2_1_a, eris_ovvv, optimize=True)[:,
                    ij_ind_a[0], ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] += lib.einsum('i,BjAa,jaiC->ABC', r_a, t2_1_a, eris_ovoo, optimize=True)[:,
                    ij_ind_a[0], ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] -= lib.einsum('i,BjAa,iajC->ABC', r_a, t2_1_a, eris_ovoo, optimize=True)[:,
                    ij_ind_a[0], ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] -= lib.einsum('i,CjAa,jaiB->ABC', r_a, t2_1_a, eris_ovoo, optimize=True)[:,
                    ij_ind_a[0], ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] += lib.einsum('i,CjAa,iajB->ABC', r_a, t2_1_a, eris_ovoo, optimize=True)[:,
                    ij_ind_a[0], ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] += lib.einsum('i,BjAa,jaiC->ABC', r_a, t2_1_ab, eris_OVoo, optimize=True)[:,
                    ij_ind_a[0], ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] -= lib.einsum('i,CjAa,jaiB->ABC', r_a, t2_1_ab, eris_OVoo, optimize=True)[:,
                    ij_ind_a[0], ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] += 1/2 *  lib.einsum('i,A,ia,BCAa->ABC', r_a, e_vir_a, t1_1_a, t2_1_a,
                    optimize=True)[:, ij_ind_a[0], ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] -= 1/2 *  lib.einsum('i,B,ia,BCAa->ABC', r_a, e_occ_a, t1_1_a, t2_1_a,
                    optimize=True)[:, ij_ind_a[0], ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] -= 1/2 *  lib.einsum('i,C,ia,BCAa->ABC', r_a, e_occ_a, t1_1_a, t2_1_a,
                    optimize=True)[:, ij_ind_a[0], ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] += lib.einsum('i,a,ia,BCAa->ABC', r_a, e_vir_a, t1_1_a, t2_1_a, optimize=True)[:,
                    ij_ind_a[0], ij_ind_a[1]].reshape(-1)
                s[s_aaa:f_aaa] -= 1/2 *  lib.einsum('i,i,ia,BCAa->ABC', r_a, e_occ_a, t1_1_a, t2_1_a,
                    optimize=True)[:, ij_ind_a[0], ij_ind_a[1]].reshape(-1)
                s[s_bab:f_bab] += lib.einsum('i,ia,CBaA->ABC', r_a, f_ov_a, t2_1_ab, optimize=True).reshape(-1)
                s[s_bab:f_bab] += lib.einsum('i,Ca,BAai->ABC', r_a, t1_1_a, eris_OVvo, optimize=True).reshape(-1)
                s[s_bab:f_bab] += lib.einsum('i,ia,BAaC->ABC', r_a, t1_1_a, eris_OVvo, optimize=True).reshape(-1)
                s[s_bab:f_bab] += lib.einsum('i,Ba,iCAa->ABC', r_a, t1_1_b, eris_ooVV, optimize=True).reshape(-1)
                s[s_bab:f_bab] -= lib.einsum('i,jA,CiBj->ABC', r_a, t1_1_b, eris_ooOO, optimize=True).reshape(-1)
                s[s_bab:f_bab] += lib.einsum('i,CBab,iaAb->ABC', r_a, t2_1_ab, eris_ovVV, optimize=True).reshape(-1)
                s[s_bab:f_bab] -= lib.einsum('i,CjaA,iajB->ABC', r_a, t2_1_ab, eris_ovOO, optimize=True).reshape(-1)
                s[s_bab:f_bab] += lib.einsum('i,jBaA,jaiC->ABC', r_a, t2_1_ab, eris_ovoo, optimize=True).reshape(-1)
                s[s_bab:f_bab] -= lib.einsum('i,jBaA,iajC->ABC', r_a, t2_1_ab, eris_ovoo, optimize=True).reshape(-1)
                s[s_bab:f_bab] += lib.einsum('i,BjAa,jaiC->ABC', r_a, t2_1_b, eris_OVoo, optimize=True).reshape(-1)
                s[s_bab:f_bab] += 1/2 *  lib.einsum('i,A,ia,CBaA->ABC', r_a, e_vir_b, t1_1_a, t2_1_ab,
                    optimize=True).reshape(-1)
                s[s_bab:f_bab] -= 1/2 *  lib.einsum('i,B,ia,CBaA->ABC', r_a, e_occ_b, t1_1_a, t2_1_ab,
                    optimize=True).reshape(-1)
                s[s_bab:f_bab] -= 1/2 *  lib.einsum('i,C,ia,CBaA->ABC', r_a, e_occ_a, t1_1_a, t2_1_ab,
                    optimize=True).reshape(-1)
                s[s_bab:f_bab] += lib.einsum('i,a,ia,CBaA->ABC', r_a, e_vir_a, t1_1_a, t2_1_ab,
                    optimize=True).reshape(-1)
                s[s_bab:f_bab] -= 1/2 *  lib.einsum('i,i,ia,CBaA->ABC', r_a, e_occ_a, t1_1_a, t2_1_ab,
                    optimize=True).reshape(-1)
                s[s_aba:f_aba] += lib.einsum('i,ia,BCAa->ABC', r_b, f_ov_b, t2_1_ab, optimize=True).reshape(-1)
                s[s_aba:f_aba] += lib.einsum('i,Ba,iCAa->ABC', r_b, t1_1_a, eris_OOvv, optimize=True).reshape(-1)
                s[s_aba:f_aba] -= lib.einsum('i,jA,BjCi->ABC', r_b, t1_1_a, eris_ooOO, optimize=True).reshape(-1)
                s[s_aba:f_aba] += lib.einsum('i,Ca,BAai->ABC', r_b, t1_1_b, eris_ovVO, optimize=True).reshape(-1)
                s[s_aba:f_aba] += lib.einsum('i,ia,BAaC->ABC', r_b, t1_1_b, eris_ovVO, optimize=True).reshape(-1)
                s[s_aba:f_aba] += lib.einsum('i,BjAa,jaiC->ABC', r_b, t2_1_a, eris_ovOO, optimize=True).reshape(-1)
                s[s_aba:f_aba] += lib.einsum('i,BCab,ibAa->ABC', r_b, t2_1_ab, eris_OVvv, optimize=True).reshape(-1)
                s[s_aba:f_aba] += lib.einsum('i,BjAa,jaiC->ABC', r_b, t2_1_ab, eris_OVOO, optimize=True).reshape(-1)
                s[s_aba:f_aba] -= lib.einsum('i,BjAa,iajC->ABC', r_b, t2_1_ab, eris_OVOO, optimize=True).reshape(-1)
                s[s_aba:f_aba] -= lib.einsum('i,jCAa,iajB->ABC', r_b, t2_1_ab, eris_OVoo, optimize=True).reshape(-1)
                s[s_aba:f_aba] += 1/2 *  lib.einsum('i,A,ia,BCAa->ABC', r_b, e_vir_a, t1_1_b, t2_1_ab,
                    optimize=True).reshape(-1)
                s[s_aba:f_aba] -= 1/2 *  lib.einsum('i,B,ia,BCAa->ABC', r_b, e_occ_a, t1_1_b, t2_1_ab,
                    optimize=True).reshape(-1)
                s[s_aba:f_aba] -= 1/2 *  lib.einsum('i,C,ia,BCAa->ABC', r_b, e_occ_b, t1_1_b, t2_1_ab,
                    optimize=True).reshape(-1)
                s[s_aba:f_aba] += lib.einsum('i,a,ia,BCAa->ABC', r_b, e_vir_b, t1_1_b, t2_1_ab,
                    optimize=True).reshape(-1)
                s[s_aba:f_aba] -= 1/2 *  lib.einsum('i,i,ia,BCAa->ABC', r_b, e_occ_b, t1_1_b, t2_1_ab,
                    optimize=True).reshape(-1)
                s[s_bbb:f_bbb] += lib.einsum('i,ia,BCAa->ABC', r_b, f_ov_b, t2_1_b, optimize=True)[:, ij_ind_b[0],
                    ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] -= lib.einsum('i,Ba,CAai->ABC', r_b, t1_1_b, eris_OVVO, optimize=True)[:, ij_ind_b[0],
                    ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] += lib.einsum('i,Ba,iCAa->ABC', r_b, t1_1_b, eris_OOVV, optimize=True)[:, ij_ind_b[0],
                    ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] += lib.einsum('i,Ca,BAai->ABC', r_b, t1_1_b, eris_OVVO, optimize=True)[:, ij_ind_b[0],
                    ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] -= lib.einsum('i,Ca,iBAa->ABC', r_b, t1_1_b, eris_OOVV, optimize=True)[:, ij_ind_b[0],
                    ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] += lib.einsum('i,ia,BAaC->ABC', r_b, t1_1_b, eris_OVVO, optimize=True)[:, ij_ind_b[0],
                    ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] -= lib.einsum('i,ia,CAaB->ABC', r_b, t1_1_b, eris_OVVO, optimize=True)[:, ij_ind_b[0],
                    ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] += lib.einsum('i,jA,BiCj->ABC', r_b, t1_1_b, eris_OOOO, optimize=True)[:, ij_ind_b[0],
                    ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] -= lib.einsum('i,jA,BjCi->ABC', r_b, t1_1_b, eris_OOOO, optimize=True)[:, ij_ind_b[0],
                    ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] += lib.einsum('i,jBaA,jaiC->ABC', r_b, t2_1_ab, eris_ovOO, optimize=True)[:,
                    ij_ind_b[0], ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] -= lib.einsum('i,jCaA,jaiB->ABC', r_b, t2_1_ab, eris_ovOO, optimize=True)[:,
                    ij_ind_b[0], ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] -= 1/2 *  lib.einsum('i,BCab,iaAb->ABC', r_b, t2_1_b, eris_OVVV, optimize=True)[:,
                    ij_ind_b[0], ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] += 1/2 *  lib.einsum('i,BCab,ibAa->ABC', r_b, t2_1_b, eris_OVVV, optimize=True)[:,
                    ij_ind_b[0], ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] += lib.einsum('i,BjAa,jaiC->ABC', r_b, t2_1_b, eris_OVOO, optimize=True)[:,
                    ij_ind_b[0], ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] -= lib.einsum('i,BjAa,iajC->ABC', r_b, t2_1_b, eris_OVOO, optimize=True)[:,
                    ij_ind_b[0], ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] -= lib.einsum('i,CjAa,jaiB->ABC', r_b, t2_1_b, eris_OVOO, optimize=True)[:,
                    ij_ind_b[0], ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] += lib.einsum('i,CjAa,iajB->ABC', r_b, t2_1_b, eris_OVOO, optimize=True)[:,
                    ij_ind_b[0], ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] += 1/2 *  lib.einsum('i,A,ia,BCAa->ABC', r_b, e_vir_b, t1_1_b, t2_1_b,
                    optimize=True)[:, ij_ind_b[0], ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] -= 1/2 *  lib.einsum('i,B,ia,BCAa->ABC', r_b, e_occ_b, t1_1_b, t2_1_b,
                    optimize=True)[:, ij_ind_b[0], ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] -= 1/2 *  lib.einsum('i,C,ia,BCAa->ABC', r_b, e_occ_b, t1_1_b, t2_1_b,
                    optimize=True)[:, ij_ind_b[0], ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] += lib.einsum('i,a,ia,BCAa->ABC', r_b, e_vir_b, t1_1_b, t2_1_b, optimize=True)[:,
                    ij_ind_b[0], ij_ind_b[1]].reshape(-1)
                s[s_bbb:f_bbb] -= 1/2 *  lib.einsum('i,i,ia,BCAa->ABC', r_b, e_occ_b, t1_1_b, t2_1_b,
                    optimize=True)[:, ij_ind_b[0], ij_ind_b[1]].reshape(-1)
            del t2_1_a
            del t2_1_b
            del t2_1_ab


        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)
        s *= -1.0

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

    t2_1_a = adc.t2[0][0][:]
    t2_1_ab = adc.t2[0][1][:]
    t2_1_b = adc.t2[0][2][:]

    t1_1_a = t1_1_b = None
    if adc.t1[2][0] is not None:
        t1_1_a = adc.t1[2][0]
        t1_1_b = adc.t1[2][1]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)

    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a* (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a* nocc_b
    n_doubles_aba = nvir_a * nocc_b* nocc_a
    n_doubles_bbb = nocc_b* (nocc_b - 1) * nvir_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)

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
        # placeholder to mute flake8 warning

        ######## ADC(2) 1h part  ############################################

        t2_1_a = adc.t2[0][0][:]
        t2_1_ab = adc.t2[0][1][:]
        if orb < nocc_a:
            T[s_a:f_a]  = idn_occ_a[orb, :]
            T[s_a:f_a] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_a[:,orb,:,:], t2_1_a, optimize=True)
            T[s_a:f_a] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1_ab[orb,:,:,:], t2_1_ab, optimize=True)
            T[s_a:f_a] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1_ab[orb,:,:,:], t2_1_ab, optimize=True)
            if t1_1_a is not None:
                T[s_a:f_a] -= 0.5*lib.einsum('c,ic->i',t1_1_a[orb,:], t1_1_a, optimize=True)
        else:
            if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
                T[s_a:f_a] += t1_2_a[:,(orb-nocc_a)]
            if t1_1_a is not None:
                T[s_a:f_a] += t1_1_a[:,(orb-nocc_a)]
                T[s_a:f_a] += 0.5*lib.einsum('kc,ikc->i',t1_1_a, t2_1_a[:,:,(orb-nocc_a),:], optimize=True)
                T[s_a:f_a] += 0.5*lib.einsum('kc,ikc->i',t1_1_b, t2_1_ab[:,:,(orb-nocc_a),:], optimize=True)

######## ADC(2) 2h-1p  part  ############################################

            t2_1_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:]
            t2_1_t_a = t2_1_t.transpose(2,1,0)
            t2_1_t_ab = t2_1_ab.transpose(2,3,1,0)

            T[s_aaa:f_aaa] = t2_1_t_a[(orb-nocc_a),:,:].reshape(-1)
            T[s_bab:f_bab] = t2_1_t_ab[(orb-nocc_a),:,:,:].reshape(-1)

######## ADC(3) 2h-1p  part  ############################################

        if (adc.method == "adc(2)-x" and adc.approx_trans_moments is False) or (adc.method == "adc(3)"):

            t2_2_a = adc.t2[1][0][:]
            t2_2_ab = adc.t2[1][1][:]

            if orb >= nocc_a:
                t2_2_t = t2_2_a[ij_ind_a[0],ij_ind_a[1],:,:]
                t2_2_t_a = t2_2_t.transpose(2,1,0)
                t2_2_t_ab = t2_2_ab.transpose(2,3,1,0)

                T[s_aaa:f_aaa] += t2_2_t_a[(orb-nocc_a),:,:].reshape(-1)
                T[s_bab:f_bab] += t2_2_t_ab[(orb-nocc_a),:,:,:].reshape(-1)

            if orb < nocc_a:
                if t1_1_a is not None:
                    T[s_aaa:f_aaa] -= 0.5*lib.einsum('c,ikac->ika',t1_1_a[orb,:], t2_1_a,
                        optimize=True)[ij_ind_a[0],ij_ind_a[1],:].transpose(1,0).reshape(-1)
                    T[s_bab:f_bab] -= 0.5*lib.einsum('c,kica->ika',t1_1_a[orb,:], t2_1_ab,
                        optimize=True).transpose(2,0,1).reshape(-1)

######## ADC(3) 1h part  ############################################

        if (method == 'adc(3)'):

            if (adc.approx_trans_moments is False):
                t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_a:

                t2_1_a_tmp = np.ascontiguousarray(t2_1_a[:,orb,:,:])
                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[orb,:,:,:])

                T[s_a:f_a] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_a_tmp, t2_2_a, optimize=True)
                T[s_a:f_a] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1_ab_tmp, t2_2_ab, optimize=True)
                T[s_a:f_a] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1_ab_tmp, t2_2_ab, optimize=True)

                del t2_1_a_tmp, t2_1_ab_tmp

                t2_2_a_tmp = np.ascontiguousarray(t2_2_a[:,orb,:,:])
                t2_2_ab_tmp = np.ascontiguousarray(t2_2_ab[orb,:,:,:])

                T[s_a:f_a] += 0.25*lib.einsum('ikdc,kdc->i',t2_1_a,  t2_2_a_tmp,optimize=True)
                T[s_a:f_a] -= 0.25*lib.einsum('ikcd,kcd->i',t2_1_ab, t2_2_ab_tmp,optimize=True)
                T[s_a:f_a] -= 0.25*lib.einsum('ikdc,kdc->i',t2_1_ab, t2_2_ab_tmp,optimize=True)

                del t2_2_a_tmp, t2_2_ab_tmp

                if t1_1_a is not None:
                    T[s_a:f_a] -= 0.5*lib.einsum('ic,c->i',t1_1_a, t1_2_a[orb,:],optimize=True)
                    T[s_a:f_a] -= 0.5*lib.einsum('c,ic->i',t1_1_a[orb,:], t1_2_a,optimize=True)
                    T[s_a:f_a] -= 1/3*lib.einsum('ic,kd,kcd->i',t1_1_a, t1_1_b, t2_1_ab[orb,:,:,:],optimize=True)
                    T[s_a:f_a] -= 1/6*lib.einsum('c,kd,ikcd->i',t1_1_a[orb,:], t1_1_b, t2_1_ab,optimize=True)
                    T[s_a:f_a] += 1/6*lib.einsum('ikcd,kc,d->i',t2_1_a, t1_1_a, t1_1_a[orb,:],optimize=True)
                    T[s_a:f_a] += 1/3*lib.einsum('kcd,kc,id->i',t2_1_a[orb,:,:,:], t1_1_a, t1_1_a,optimize=True)
            else:
                t2_1_a_tmp =  np.ascontiguousarray(t2_1_a[:,:,(orb-nocc_a),:])
                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,:,(orb-nocc_a),:])

                T[s_a:f_a] += 0.5*lib.einsum('ikc,kc->i',t2_1_a_tmp, t1_2_a,optimize=True)
                T[s_a:f_a] += 0.5*lib.einsum('ikc,kc->i',t2_1_ab_tmp, t1_2_b,optimize=True)
                if (adc.approx_trans_moments is False):
                    T[s_a:f_a] += t1_3_a[:,(orb-nocc_a)]
                if t1_1_a is not None:
                    T[s_a:f_a] += 0.5*lib.einsum('kc,ikc->i',t1_1_a, t2_2_a[:,:,(orb-nocc_a),:],optimize=True)
                    T[s_a:f_a] += 0.5*lib.einsum('kc,ikc->i',t1_1_b, t2_2_ab[:,:,(orb-nocc_a),:],optimize=True)
                    T[s_a:f_a] -= 1/6*lib.einsum('ic,k,kc->i',t1_1_a, t1_1_a[:,(orb-nocc_a)], t1_1_a,optimize=True)
                    T[s_a:f_a] -= 1/12*lib.einsum('ic,klcd,kld->i',t1_1_a, t2_1_a, t2_1_a_tmp,optimize=True)
                    T[s_a:f_a] -= 1/12*lib.einsum('k,klcd,ilcd->i',t1_1_a[:,(orb-nocc_a)], t2_1_a, t2_1_a,optimize=True)
                    T[s_a:f_a] += 1/6*lib.einsum('kc,klcd,ild->i',t1_1_a, t2_1_a, t2_1_a_tmp,optimize=True)
                    T[s_a:f_a] -= 1/6*lib.einsum('ic,klcd,kld->i',t1_1_a, t2_1_ab, t2_1_ab_tmp,optimize=True)
                    T[s_a:f_a] -= 1/6*lib.einsum('k,klcd,ilcd->i',t1_1_a[:,(orb-nocc_a)], t2_1_ab, t2_1_ab,
                                                 optimize=True)
                    T[s_a:f_a] += 1/6*lib.einsum('kc,klcd,ild->i',t1_1_a, t2_1_ab, t2_1_ab_tmp,optimize=True)
                    T[s_a:f_a] += 1/6*lib.einsum('kc,ild,lkdc->i',t1_1_b, t2_1_a_tmp, t2_1_ab,optimize=True)
                    T[s_a:f_a] += 1/6*lib.einsum('kc,ild,klcd->i',t1_1_b, t2_1_ab_tmp, t2_1_b,optimize=True)
                del t2_1_a_tmp, t2_1_ab_tmp

                del t2_2_a
                del t2_2_ab

        del t2_1_a
        del t2_1_ab
######## spin = beta  ############################################

    else:
        # placeholder

        ######## ADC(2) 1h part  ############################################

        t2_1_b = adc.t2[0][2][:]
        t2_1_ab = adc.t2[0][1][:]
        if orb < nocc_b:

            t2_1_b_tmp = np.ascontiguousarray(t2_1_b[:,orb,:,:])
            t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,orb,:,:])

            T[s_b:f_b] = idn_occ_b[orb, :]
            T[s_b:f_b]+= 0.25*lib.einsum('kdc,ikdc->i',t2_1_b_tmp, t2_1_b, optimize=True)
            T[s_b:f_b]-= 0.25*lib.einsum('kdc,kidc->i',t2_1_ab_tmp, t2_1_ab, optimize=True)
            T[s_b:f_b]-= 0.25*lib.einsum('kcd,kicd->i',t2_1_ab_tmp, t2_1_ab, optimize=True)
            del t2_1_b_tmp, t2_1_ab_tmp
            if t1_1_b is not None:
                T[s_b:f_b] -= 0.5*lib.einsum('c,ic->i',t1_1_b[orb,:], t1_1_b, optimize=True)
        else:
            if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
                T[s_b:f_b] += t1_2_b[:,(orb-nocc_b)]
            if t1_1_b is not None:
                T[s_b:f_b] += t1_1_b[:,(orb-nocc_b)]
                T[s_b:f_b] += 0.5*lib.einsum('kc,kic->i',t1_1_a, t2_1_ab[:,:,:,(orb-nocc_b)], optimize=True)
                T[s_b:f_b] += 0.5*lib.einsum('kc,ikc->i',t1_1_b, t2_1_b[:,:,(orb-nocc_b),:], optimize=True)

######## ADC(2) 2h-1p part  ############################################

            t2_1_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:]
            t2_1_t_b = t2_1_t.transpose(2,1,0)
            t2_1_t_ab = t2_1_ab.transpose(2,3,0,1)

            T[s_bbb:f_bbb] = t2_1_t_b[(orb-nocc_b),:,:].reshape(-1)
            T[s_aba:f_aba] = t2_1_t_ab[:,(orb-nocc_b),:,:].reshape(-1)

######## ADC(3) 2h-1p part  ############################################

        if (adc.method == "adc(2)-x" and adc.approx_trans_moments is False) or (adc.method == "adc(3)"):

            t2_2_a = adc.t2[1][0][:]
            t2_2_ab = adc.t2[1][1][:]
            t2_2_b = adc.t2[1][2][:]

            if orb >= nocc_b:
                t2_2_t = t2_2_b[ij_ind_b[0],ij_ind_b[1],:,:]
                t2_2_t_b = t2_2_t.transpose(2,1,0)

                t2_2_t_ab = t2_2_ab.transpose(2,3,0,1)

                T[s_bbb:f_bbb] += t2_2_t_b[(orb-nocc_b),:,:].reshape(-1)
                T[s_aba:f_aba] += t2_2_t_ab[:,(orb-nocc_b),:,:].reshape(-1)

            if orb < nocc_b:
                if t1_1_b is not None:
                    T[s_bbb:f_bbb] -= 0.5*lib.einsum('c,ikac->ika',t1_1_b[orb,:], t2_1_b,
                        optimize=True)[ij_ind_b[0],ij_ind_b[1],:].transpose(1,0).reshape(-1)
                    T[s_aba:f_aba] -= 0.5*lib.einsum('c,ikac->ika',t1_1_b[orb,:], t2_1_ab,
                        optimize=True).transpose(2,0,1).reshape(-1)

######## ADC(3) 1h part  ############################################

        if (method=='adc(3)'):

            if (adc.approx_trans_moments is False):
                t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_b:

                t2_1_b_tmp = np.ascontiguousarray(t2_1_b[:,orb,:,:])
                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,orb,:,:])

                T[s_b:f_b] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_b_tmp, t2_2_b, optimize=True)
                T[s_b:f_b] -= 0.25*lib.einsum('kdc,kidc->i',t2_1_ab_tmp, t2_2_ab, optimize=True)
                T[s_b:f_b] -= 0.25*lib.einsum('kcd,kicd->i',t2_1_ab_tmp, t2_2_ab, optimize=True)

                del t2_1_b_tmp, t2_1_ab_tmp

                t2_2_b_tmp = np.ascontiguousarray(t2_2_b[:,orb,:,:])
                t2_2_ab_tmp = np.ascontiguousarray(t2_2_ab[:,orb,:,:])

                T[s_b:f_b] += 0.25*lib.einsum('ikdc,kdc->i',t2_1_b,  t2_2_b_tmp ,optimize=True)
                T[s_b:f_b] -= 0.25*lib.einsum('kicd,kcd->i',t2_1_ab, t2_2_ab_tmp,optimize=True)
                T[s_b:f_b] -= 0.25*lib.einsum('kidc,kdc->i',t2_1_ab, t2_2_ab_tmp,optimize=True)

                del t2_2_b_tmp, t2_2_ab_tmp

                if t1_1_b is not None:
                    T[s_b:f_b] -= 0.5*lib.einsum('ic,c->i',t1_1_b, t1_2_b[orb,:],optimize=True)
                    T[s_b:f_b] -= 0.5*lib.einsum('c,ic->i',t1_1_b[orb,:], t1_2_b,optimize=True)
                    T[s_b:f_b] -= 1/3*lib.einsum('kc,id,kcd->i',t1_1_a, t1_1_b, t2_1_ab[:,orb,:,:],optimize=True)
                    T[s_b:f_b] -= 1/6*lib.einsum('kc,d,kicd->i',t1_1_a, t1_1_b[orb,:], t2_1_ab,optimize=True)
                    T[s_b:f_b] += 1/6*lib.einsum('ikcd,kc,d->i',t2_1_b, t1_1_b, t1_1_b[orb,:],optimize=True)
                    T[s_b:f_b] += 1/3*lib.einsum('kcd,kc,id->i',t2_1_b[orb,:,:,:], t1_1_b, t1_1_b,optimize=True)

            else:
                t2_1_b_tmp  = np.ascontiguousarray(t2_1_b[:,:,(orb-nocc_b),:])
                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,:,:,(orb-nocc_b)])

                T[s_b:f_b] += 0.5*lib.einsum('ikc,kc->i',t2_1_b_tmp, t1_2_b,optimize=True)
                T[s_b:f_b] += 0.5*lib.einsum('kic,kc->i',t2_1_ab_tmp, t1_2_a,optimize=True)
                if (adc.approx_trans_moments is False):
                    T[s_b:f_b] += t1_3_b[:,(orb-nocc_b)]
                if t1_1_b is not None:
                    T[s_b:f_b] += 0.5*lib.einsum('kc,kic->i',t1_1_a, t2_2_ab[:,:,:,(orb-nocc_b)],optimize=True)
                    T[s_b:f_b] += 0.5*lib.einsum('kc,ikc->i',t1_1_b, t2_2_b[:,:,(orb-nocc_b),:],optimize=True)
                    T[s_b:f_b] += 1/6*lib.einsum('kc,klcd,lid->i',t1_1_a, t2_1_a, t2_1_ab_tmp,optimize=True)
                    T[s_b:f_b] += 1/6*lib.einsum('kc,klcd,ild->i',t1_1_a, t2_1_ab, t2_1_b_tmp,optimize=True)
                    T[s_b:f_b] -= 1/6*lib.einsum('ic,k,kc->i',t1_1_b, t1_1_b[:,(orb-nocc_b)], t1_1_b,optimize=True)
                    T[s_b:f_b] -= 1/6*lib.einsum('ic,kldc,kld->i',t1_1_b, t2_1_ab, t2_1_ab_tmp,optimize=True)
                    T[s_b:f_b] -= 1/6*lib.einsum('k,lkcd,licd->i',t1_1_b[:,(orb-nocc_b)], t2_1_ab, t2_1_ab,
                                                 optimize=True)
                    T[s_b:f_b] += 1/6*lib.einsum('kc,lkdc,lid->i',t1_1_b, t2_1_ab, t2_1_ab_tmp,optimize=True)
                    T[s_b:f_b] -= 1/12*lib.einsum('ic,klcd,kld->i',t1_1_b, t2_1_b, t2_1_b_tmp,optimize=True)
                    T[s_b:f_b] -= 1/12*lib.einsum('k,klcd,ilcd->i',t1_1_b[:,(orb-nocc_b)], t2_1_b, t2_1_b,optimize=True)
                    T[s_b:f_b] += 1/6*lib.einsum('kc,klcd,ild->i',t1_1_b, t2_1_b, t2_1_b_tmp,optimize=True)
                del t2_1_b_tmp, t2_1_ab_tmp
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

    ij_a = np.tril_indices(nocc_a, k=-1)
    ij_b = np.tril_indices(nocc_b, k=-1)

    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a* (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a* nocc_b
    n_doubles_aba = nvir_a * nocc_b* nocc_a
    n_doubles_bbb = nocc_b* (nocc_b - 1) * nvir_b // 2

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
        U1 = U[:f_b,I]
        U2 = U[f_b:,I]
        U1dotU1 = np.dot(U1, U1)
        U2dotU2 = np.dot(U2, U2)

        temp_aaa = np.zeros((nvir_a, nocc_a, nocc_a))
        temp_aaa[:,ij_a[0],ij_a[1]] =  U[s_aaa:f_aaa,I].reshape(nvir_a,-1).copy()
        temp_aaa[:,ij_a[1],ij_a[0]] = -U[s_aaa:f_aaa,I].reshape(nvir_a,-1).copy()
        U_aaa = temp_aaa.reshape(-1).copy()

        temp_bbb = np.zeros((nvir_b, nocc_b, nocc_b))
        temp_bbb[:,ij_b[0],ij_b[1]] =  U[s_bbb:f_bbb,I].reshape(nvir_b,-1).copy()
        temp_bbb[:,ij_b[1],ij_b[0]] = -U[s_bbb:f_bbb,I].reshape(nvir_b,-1).copy()
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
                i_idx = orb_idx + 1
                singles_a_idx.append(i_idx)
                singles_a_val.append(U_sorted[iter_idx])

            if orb_idx in range(s_b,f_b):
                i_idx = orb_idx - s_b + 1
                singles_b_idx.append(i_idx)
                singles_b_val.append(U_sorted[iter_idx])

            if orb_idx in range(s_bab,f_bab):
                aij_idx = orb_idx - s_bab
                ij_rem = aij_idx % (nocc_a*nocc_b)
                a_idx = aij_idx//(nocc_a*nocc_b)
                i_idx = ij_rem//nocc_a
                j_idx = ij_rem % nocc_a
                doubles_bab_idx.append((a_idx + 1 + nocc_b, i_idx + 1, j_idx + 1))
                doubles_bab_val.append(U_sorted[iter_idx])

            if orb_idx in range(s_aba,f_aba):
                aij_idx = orb_idx - s_aba
                ij_rem = aij_idx % (nocc_b*nocc_a)
                a_idx = aij_idx//(nocc_b*nocc_a)
                i_idx = ij_rem//nocc_b
                j_idx = ij_rem % nocc_b
                doubles_aba_idx.append((a_idx + 1 + nocc_a, i_idx + 1, j_idx + 1))
                doubles_aba_val.append(U_sorted[iter_idx])

            iter_idx += 1

        for orb_aaa in ind_idx_aaa:
            ij_rem = orb_aaa % (nocc_a*nocc_a)
            a_idx = orb_aaa//(nocc_a*nocc_a)
            i_idx = ij_rem//nocc_a
            j_idx = ij_rem % nocc_a
            doubles_aaa_idx.append((a_idx + 1 + nocc_a, i_idx + 1, j_idx + 1))

        for orb_bbb in ind_idx_bbb:
            ij_rem = orb_bbb % (nocc_b*nocc_b)
            a_idx = orb_bbb//(nocc_b*nocc_b)
            i_idx = ij_rem//nocc_b
            j_idx = ij_rem % nocc_b
            doubles_bbb_idx.append((a_idx + 1 + nocc_b, i_idx + 1, j_idx + 1))

        doubles_aaa_val = list(U_sorted_aaa)
        doubles_bbb_val = list(U_sorted_bbb)

        logger.info(adc,'%s | root %d | Energy (eV) = %12.8f | norm(1h)  = %6.4f | norm(2h1p) = %6.4f ',
                    adc.method, I, adc.E[I]*HARTREE2EV, U1dotU1, U2dotU2)

        if singles_a_val:
            logger.info(adc, "\n1h(alpha) block: ")
            logger.info(adc, "     i     U(i)")
            logger.info(adc, "------------------")
            for idx, print_singles in enumerate(singles_a_idx):
                logger.info(adc, '  %4d   %7.4f', print_singles, singles_a_val[idx])

        if singles_b_val:
            logger.info(adc, "\n1h(beta) block: ")
            logger.info(adc, "     i     U(i)")
            logger.info(adc, "------------------")
            for idx, print_singles in enumerate(singles_b_idx):
                logger.info(adc, '  %4d   %7.4f', print_singles, singles_b_val[idx])

        if doubles_aaa_val:
            logger.info(adc, "\n2h1p(alpha|alpha|alpha) block: ")
            logger.info(adc, "     i     j     a     U(i,j,a)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_aaa_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f',
                            print_doubles[1], print_doubles[2], print_doubles[0], doubles_aaa_val[idx])

        if doubles_bab_val:
            logger.info(adc, "\n2h1p(beta|alpha|beta) block: ")
            logger.info(adc, "     i     j     a     U(i,j,a)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_bab_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f', print_doubles[1],
                            print_doubles[2], print_doubles[0], doubles_bab_val[idx])

        if doubles_aba_val:
            logger.info(adc, "\n2h1p(alpha|beta|alpha) block: ")
            logger.info(adc, "     i     j     a     U(i,j,a)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_aba_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f',
                            print_doubles[1], print_doubles[2], print_doubles[0], doubles_aba_val[idx])

        if doubles_bbb_val:
            logger.info(adc, "\n2h1p(beta|beta|beta) block: ")
            logger.info(adc, "     i     j     a     U(i,j,a)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_bbb_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f',
                            print_doubles[1], print_doubles[2], print_doubles[0], doubles_bbb_val[idx])

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

    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)
    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a * (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a * nocc_b
    n_doubles_aba = nvir_a * nocc_b * nocc_a
    n_doubles_bbb = nocc_b * (nocc_b - 1) * nvir_b // 2

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

    L_aaa = L_aaa.reshape(nvir_a,-1)
    L_bbb = L_bbb.reshape(nvir_b,-1)
    L_aaa_u = None
    L_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
    L_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= L_aaa.copy()
    L_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -L_aaa.copy()

    L_bbb_u = None
    L_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))
    L_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= L_bbb.copy()
    L_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -L_bbb.copy()

    L_aba = L_aba.reshape(nvir_a,nocc_a,nocc_b)
    L_bab = L_bab.reshape(nvir_b,nocc_b,nocc_a)


    R_aaa = R_aaa.reshape(nvir_a,-1)
    R_bbb = R_bbb.reshape(nvir_b,-1)
    R_aaa_u = None
    R_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
    R_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= R_aaa.copy()
    R_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -R_aaa.copy()

    R_bbb_u = None
    R_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))
    R_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= R_bbb.copy()
    R_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -R_bbb.copy()

    R_aba = R_aba.reshape(nvir_a,nocc_a,nocc_b)
    R_bab = R_bab.reshape(nvir_b,nocc_b,nocc_a)

    t1_1_a = t1_1_b = None
    if adc.t1[2][0] is not None:
        t1_1_a = adc.t1[2][0]
        t1_1_b = adc.t1[2][1]

# block- ij
    rdm1_a[occ_list_a,occ_list_a] = lib.einsum('m,m->',L_a,R_a,optimize=True)
    rdm1_a[:nocc_a,:nocc_a] -= lib.einsum('i,j->ij',L_a,R_a,optimize=True)
    rdm1_a[occ_list_a,occ_list_a] += lib.einsum('m,m->',L_b,R_b,optimize=True)
    rdm1_b[occ_list_b,occ_list_b] = lib.einsum('m,m->',L_b,R_b,optimize=True)
    rdm1_b[:nocc_b,:nocc_b] -= lib.einsum('i,j->ij',L_b,R_b,optimize=True)
    rdm1_b[occ_list_b,occ_list_b] += lib.einsum('m,m->',L_a,R_a,optimize=True)
    rdm1_a[occ_list_a,occ_list_a] += 0.5*lib.einsum('etu,etu->',L_aaa_u,R_aaa_u,optimize=True)
    rdm1_a[occ_list_a,occ_list_a] += lib.einsum('etu,etu->',L_bab,R_bab,optimize=True)
    rdm1_a[occ_list_a,occ_list_a] += lib.einsum('etu,etu->',L_aba,R_aba,optimize=True)
    rdm1_a[occ_list_a,occ_list_a] += 0.5*lib.einsum('etu,etu->',L_bbb_u,R_bbb_u,optimize=True)
    rdm1_a[:nocc_a,:nocc_a] -= lib.einsum('eti,etj->ij',L_aaa_u,R_aaa_u,optimize=True)
    rdm1_a[:nocc_a,:nocc_a] -= lib.einsum('eti,etj->ij',L_bab,R_bab,optimize=True)
    rdm1_a[:nocc_a,:nocc_a] -= lib.einsum('eit,ejt->ij',L_aba,R_aba,optimize=True)
    rdm1_b[occ_list_b,occ_list_b] += 0.5*lib.einsum('etu,etu->',L_aaa_u,R_aaa_u,optimize=True)
    rdm1_b[occ_list_b,occ_list_b] += lib.einsum('etu,etu->',L_bab,R_bab,optimize=True)
    rdm1_b[occ_list_b,occ_list_b] += lib.einsum('etu,etu->',L_aba,R_aba,optimize=True)
    rdm1_b[occ_list_b,occ_list_b] += 0.5*lib.einsum('etu,etu->',L_bbb_u,R_bbb_u,optimize=True)
    rdm1_b[:nocc_b,:nocc_b] -= lib.einsum('eti,etj->ij',L_bbb_u,R_bbb_u,optimize=True)
    rdm1_b[:nocc_b,:nocc_b] -= lib.einsum('eti,etj->ij',L_aba,R_aba,optimize=True)
    rdm1_b[:nocc_b,:nocc_b] -= lib.einsum('eit,ejt->ij',L_bab,R_bab,optimize=True)
    rdm1_a[:nocc_a,:nocc_a] -= 0.5* lib.einsum('g,g,hjcd,hicd->ij', L_a,R_a,t2_1_a,t2_1_a,optimize=True)
    rdm1_a[:nocc_a,:nocc_a] -= 0.5* lib.einsum('g,g,hjcd,hicd->ij', L_b,R_b,t2_1_a,t2_1_a,optimize=True)
    rdm1_a[:nocc_a,:nocc_a] -= lib.einsum('g,g,jhcd,ihcd->ij', L_a,R_a,t2_1_ab,t2_1_ab,optimize=True)
    rdm1_a[:nocc_a,:nocc_a] -= lib.einsum('g,g,jhcd,ihcd->ij', L_b,R_b,t2_1_ab,t2_1_ab,optimize=True)
    rdm1_a[:nocc_a,:nocc_a] += 0.5* lib.einsum('g,h,jgcd,ihcd->ij', L_a,R_a,t2_1_a,t2_1_a,optimize=True)
    rdm1_a[:nocc_a,:nocc_a] += lib.einsum('g,h,jgcd,ihcd->ij', L_b,R_b,t2_1_ab,t2_1_ab,optimize=True)
    temp = np.zeros((nocc_a, nocc_a))
    temp += 0.25* lib.einsum('g,j,ghcd,ihcd->ij',L_a,R_a,t2_1_a,t2_1_a,optimize=True)
    temp += 0.5* lib.einsum('g,j,ghcd,ihcd->ij',L_a,R_a,t2_1_ab,t2_1_ab,optimize=True)
    rdm1_a[:nocc_a,:nocc_a] += temp + temp.T
    rdm1_b[:nocc_b,:nocc_b] -= 0.5* lib.einsum('g,g,hjcd,hicd->ij', L_b,R_b,t2_1_b,t2_1_b,optimize=True)
    rdm1_b[:nocc_b,:nocc_b] -= 0.5* lib.einsum('g,g,hjcd,hicd->ij', L_a,R_a,t2_1_b,t2_1_b,optimize=True)
    rdm1_b[:nocc_b,:nocc_b] -= lib.einsum('g,g,hjcd,hicd->ij', L_b,R_b,t2_1_ab,t2_1_ab,optimize=True)
    rdm1_b[:nocc_b,:nocc_b] -= lib.einsum('g,g,hjcd,hicd->ij', L_a,R_a,t2_1_ab,t2_1_ab,optimize=True)
    rdm1_b[:nocc_b,:nocc_b] += 0.5* lib.einsum('g,h,jgcd,ihcd->ij', L_b,R_b,t2_1_b,t2_1_b,optimize=True)
    rdm1_b[:nocc_b,:nocc_b] += lib.einsum('g,h,gjcd,hicd->ij', L_a,R_a,t2_1_ab,t2_1_ab,optimize=True)
    temp = np.zeros((nocc_b, nocc_b))
    temp += 0.25* lib.einsum('g,j,ghcd,ihcd->ij',L_b,R_b,t2_1_b,t2_1_b,optimize=True)
    temp += 0.5* lib.einsum('g,j,hgcd,hicd->ij',L_b,R_b,t2_1_ab,t2_1_ab,optimize=True)
    rdm1_b[:nocc_b,:nocc_b] += temp + temp.T

    if t1_1_a is not None:
        temp = 1/2 * lib.einsum('J,i,ia,Ia->IJ', L_a, R_a, t1_1_a, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, :nocc_a] += temp + temp.T
        rdm1_a[:nocc_a, :nocc_a] -= lib.einsum('i,i,Ia,Ja->IJ', L_a, R_a, t1_1_a, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, :nocc_a] -= lib.einsum('i,i,Ia,Ja->IJ', L_b, R_b, t1_1_a, t1_1_a, optimize=True)
        rdm1_b[:nocc_b, :nocc_b] -= lib.einsum('i,i,Ia,Ja->IJ', L_a, R_a, t1_1_b, t1_1_b, optimize=True)
        temp = 1/2 * lib.einsum('J,i,ia,Ia->IJ', L_b, R_b, t1_1_b, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, :nocc_b] += temp + temp.T
        rdm1_b[:nocc_b, :nocc_b] -= lib.einsum('i,i,Ia,Ja->IJ', L_b, R_b, t1_1_b, t1_1_b, optimize=True)
        temp = np.zeros((nocc_a, nocc_a))
        temp += lib.einsum('i,aIi,Ja->IJ', L_a, R_aaa_u, t1_1_a, optimize=True)
        temp += lib.einsum('aJi,i,Ia->IJ', L_aba, R_b, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, :nocc_a] -= temp + temp.T
        temp = np.zeros((nocc_b, nocc_b))
        temp += lib.einsum('i,aIi,Ja->IJ', L_a, R_bab, t1_1_b, optimize=True)
        temp += lib.einsum('i,aIi,Ja->IJ', L_b, R_bbb_u, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, :nocc_b] -= temp + temp.T

# block- ab

    rdm1_a[nocc_a:,nocc_a:] = 0.5*lib.einsum('atu,btu->ab', L_aaa_u,R_aaa_u,optimize=True)
    rdm1_a[nocc_a:,nocc_a:] += lib.einsum('atu,btu->ab', L_aba,R_aba,optimize=True)
    rdm1_b[nocc_b:,nocc_b:] = 0.5*lib.einsum('atu,btu->ab', L_bbb_u,R_bbb_u,optimize=True)
    rdm1_b[nocc_b:,nocc_b:] += lib.einsum('atu,btu->ab', L_bab,R_bab,optimize=True)
    rdm1_a[nocc_a:,nocc_a:] += 0.5* lib.einsum('g,g,hmbc,hmac->ab', L_a,R_a,t2_1_a,t2_1_a,optimize=True)
    rdm1_a[nocc_a:,nocc_a:] += 0.5* lib.einsum('g,g,hmbc,hmac->ab', L_b,R_b,t2_1_a,t2_1_a,optimize=True)
    rdm1_a[nocc_a:,nocc_a:] += lib.einsum('g,g,hmbc,hmac->ab', L_a,R_a,t2_1_ab,t2_1_ab,optimize=True)
    rdm1_a[nocc_a:,nocc_a:] += lib.einsum('g,g,hmbc,hmac->ab', L_b,R_b,t2_1_ab,t2_1_ab,optimize=True)
    rdm1_a[nocc_a:,nocc_a:] -= lib.einsum('g,h,hmbc,gmac->ab', L_a,R_a,t2_1_a,t2_1_a,optimize=True)
    rdm1_a[nocc_a:,nocc_a:] -= lib.einsum('g,h,hmbc,gmac->ab', L_a,R_a,t2_1_ab,t2_1_ab,optimize=True)
    rdm1_a[nocc_a:,nocc_a:] -= lib.einsum('g,h,mhbc,mgac->ab', L_b,R_b,t2_1_ab,t2_1_ab,optimize=True)
    rdm1_b[nocc_b:,nocc_b:] += 0.5* lib.einsum('g,g,hmbc,hmac->ab', L_b,R_b,t2_1_b,t2_1_b,optimize=True)
    rdm1_b[nocc_b:,nocc_b:] += 0.5* lib.einsum('g,g,hmbc,hmac->ab', L_a,R_a,t2_1_b,t2_1_b,optimize=True)
    rdm1_b[nocc_b:,nocc_b:] += lib.einsum('g,g,hmcb,hmca->ab', L_b,R_b,t2_1_ab,t2_1_ab,optimize=True)
    rdm1_b[nocc_b:,nocc_b:] += lib.einsum('g,g,hmcb,hmca->ab', L_a,R_a,t2_1_ab,t2_1_ab,optimize=True)
    rdm1_b[nocc_b:,nocc_b:] -= lib.einsum('g,h,hmbc,gmac->ab', L_b,R_b,t2_1_b,t2_1_b,optimize=True)
    rdm1_b[nocc_b:,nocc_b:] -= lib.einsum('g,h,mhcb,mgca->ab', L_b,R_b,t2_1_ab,t2_1_ab,optimize=True)
    rdm1_b[nocc_b:,nocc_b:] -= lib.einsum('g,h,hmcb,gmca->ab', L_a,R_a,t2_1_ab,t2_1_ab,optimize=True)

    if t1_1_a is not None:
        rdm1_a[nocc_a:, nocc_a:] += lib.einsum('i,i,jA,jB->AB', L_a, R_a, t1_1_a, t1_1_a, optimize=True)
        rdm1_a[nocc_a:, nocc_a:] -= lib.einsum('i,j,iB,jA->AB', L_a, R_a, t1_1_a, t1_1_a, optimize=True)
        rdm1_a[nocc_a:, nocc_a:] += lib.einsum('i,i,jA,jB->AB', L_b, R_b, t1_1_a, t1_1_a, optimize=True)
        rdm1_b[nocc_b:, nocc_b:] += lib.einsum('i,i,jA,jB->AB', L_a, R_a, t1_1_b, t1_1_b, optimize=True)
        rdm1_b[nocc_b:, nocc_b:] += lib.einsum('i,i,jA,jB->AB', L_b, R_b, t1_1_b, t1_1_b, optimize=True)
        rdm1_b[nocc_b:, nocc_b:] -= lib.einsum('i,j,iB,jA->AB', L_b, R_b, t1_1_b, t1_1_b, optimize=True)
        temp = np.zeros((nvir_a, nvir_a))
        temp -= lib.einsum('i,Bij,jA->AB', L_a, R_aaa_u, t1_1_a, optimize=True)
        temp += lib.einsum('Aij,j,iB->AB', L_aba, R_b, t1_1_a, optimize=True)
        rdm1_a[nocc_a:, nocc_a:] += temp + temp.T
        temp = np.zeros((nvir_b, nvir_b))
        temp += lib.einsum('i,Bji,jA->AB', L_a, R_bab, t1_1_b, optimize=True)
        temp -= lib.einsum('i,Bij,jA->AB', L_b, R_bbb_u, t1_1_b, optimize=True)
        rdm1_b[nocc_b:, nocc_b:] += temp + temp.T

# G^100#### block- ia
    rdm1_a[:nocc_a,nocc_a:] = -lib.einsum('n,ani->ia', R_a,L_aaa_u,optimize=True)
    rdm1_a[:nocc_a,nocc_a:] += lib.einsum('n,ain->ia', R_b,L_aba,optimize=True)

    rdm1_b[:nocc_b,nocc_b:] = -lib.einsum('n,ani->ia', R_b,L_bbb_u,optimize=True)
    rdm1_b[:nocc_b,nocc_b:] += lib.einsum('n,ain->ia', R_a,L_bab,optimize=True)

    rdm1_a[:nocc_a,nocc_a:] -= lib.einsum('g,cgh,ihac->ia', L_a,R_aaa_u,t2_1_a,optimize=True)
    rdm1_a[:nocc_a,nocc_a:] += lib.einsum('g,chg,ihac->ia', L_a,R_bab,t2_1_ab,optimize=True)
    rdm1_a[:nocc_a,nocc_a:] += lib.einsum('g,chg,ihac->ia', L_b,R_aba,t2_1_a,optimize=True)
    rdm1_a[:nocc_a,nocc_a:] -= lib.einsum('g,cgh,ihac->ia', L_b,R_bbb_u,t2_1_ab,optimize=True)
    rdm1_a[:nocc_a,nocc_a:] += 0.5*lib.einsum('i,cgh,ghac->ia', L_a,R_aaa_u,t2_1_a,optimize=True)
    rdm1_a[:nocc_a,nocc_a:] -= lib.einsum('i,chg,ghac->ia', L_a,R_bab,t2_1_ab,optimize=True)

    rdm1_b[:nocc_b,nocc_b:] -= lib.einsum('g,cgh,ihac->ia', L_b,R_bbb_u,t2_1_b,optimize=True)
    rdm1_b[:nocc_b,nocc_b:] += lib.einsum('g,chg,hica->ia', L_b,R_aba,t2_1_ab,optimize=True)
    rdm1_b[:nocc_b,nocc_b:] += lib.einsum('g,chg,hica->ia', L_a,R_bab,t2_1_b,optimize=True)
    rdm1_b[:nocc_b,nocc_b:] -= lib.einsum('g,cgh,hica->ia', L_a,R_aaa_u,t2_1_ab,optimize=True)
    rdm1_b[:nocc_b,nocc_b:] += 0.5*lib.einsum('i,cgh,ghac->ia', L_b,R_bbb_u,t2_1_b,optimize=True)
    rdm1_b[:nocc_b,nocc_b:] -= lib.einsum('i,chg,hgca->ia', L_b,R_aba,t2_1_ab,optimize=True)

    if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
        rdm1_a[:nocc_a,nocc_a:] += lib.einsum('g,g,ia->ia', L_a,R_a,t1_2_a,optimize=True)
        rdm1_a[:nocc_a,nocc_a:] += lib.einsum('g,g,ia->ia', L_b,R_b,t1_2_a,optimize=True)
        rdm1_a[:nocc_a,nocc_a:] -= lib.einsum('g,i,ga->ia', R_a,L_a,t1_2_a,optimize=True)

        rdm1_b[:nocc_b,nocc_b:] += lib.einsum('g,g,ia->ia', L_b,R_b,t1_2_b,optimize=True)
        rdm1_b[:nocc_b,nocc_b:] += lib.einsum('g,g,ia->ia', L_a,R_a,t1_2_b,optimize=True)
        rdm1_b[:nocc_b,nocc_b:] -= lib.einsum('g,i,ga->ia', R_b,L_b,t1_2_b,optimize=True)

    if t1_1_a is not None:
        rdm1_a[:nocc_a, nocc_a:] -= lib.einsum('i,I,iA->IA', L_a, R_a, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += lib.einsum('i,i,IA->IA', L_a, R_a, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += lib.einsum('i,i,IA->IA', L_b, R_b, t1_1_a, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += lib.einsum('i,i,IA->IA', L_a, R_a, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= lib.einsum('i,I,iA->IA', L_b, R_b, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += lib.einsum('i,i,IA->IA', L_b, R_b, t1_1_b, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('i,I,ja,ijAa->IA', L_a, R_a, t1_1_a, t2_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('i,i,ja,IjAa->IA', L_a, R_a, t1_1_a, t2_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('i,j,ja,IiAa->IA', L_a, R_a, t1_1_a, t2_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('i,I,ja,ijAa->IA', L_a, R_a, t1_1_b, t2_1_ab, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('i,i,ja,IjAa->IA', L_a, R_a, t1_1_b, t2_1_ab, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('i,i,ja,IjAa->IA', L_b, R_b, t1_1_a, t2_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('i,i,ja,IjAa->IA', L_b, R_b, t1_1_b, t2_1_ab, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('i,j,ja,IiAa->IA', L_b, R_b, t1_1_b, t2_1_ab, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('i,i,ja,jIaA->IA', L_a, R_a, t1_1_a, t2_1_ab, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('i,j,ja,iIaA->IA', L_a, R_a, t1_1_a, t2_1_ab, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('i,i,ja,IjAa->IA', L_a, R_a, t1_1_b, t2_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('i,I,ja,jiaA->IA', L_b, R_b, t1_1_a, t2_1_ab, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('i,i,ja,jIaA->IA', L_b, R_b, t1_1_a, t2_1_ab, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('i,I,ja,ijAa->IA', L_b, R_b, t1_1_b, t2_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('i,i,ja,IjAa->IA', L_b, R_b, t1_1_b, t2_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('i,j,ja,IiAa->IA', L_b, R_b, t1_1_b, t2_1_b, optimize=True)

### 111 ###
    if adc.method in ("adc(2)-x", "adc(3)") and t1_1_a is not None:
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('aij,Aij,Ia->IA', L_aaa_u, R_aaa_u, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += lib.einsum('aij,aIi,jA->IA', L_aaa_u, R_aaa_u, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('aij,aij,IA->IA', L_aaa_u, R_aaa_u, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] -= lib.einsum('aij,Aij,Ia->IA', L_aba, R_aba, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] -= lib.einsum('aij,aIj,iA->IA', L_aba, R_aba, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += lib.einsum('aij,aij,IA->IA', L_aba, R_aba, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] -= lib.einsum('aij,aiI,jA->IA', L_bab, R_bab, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += lib.einsum('aij,aij,IA->IA', L_bab, R_bab, t1_1_a, optimize=True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('aij,aij,IA->IA', L_bbb_u, R_bbb_u, t1_1_a, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('aij,aij,IA->IA', L_aaa_u, R_aaa_u, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= lib.einsum('aij,aiI,jA->IA', L_aba, R_aba, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += lib.einsum('aij,aij,IA->IA', L_aba, R_aba, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= lib.einsum('aij,Aij,Ia->IA', L_bab, R_bab, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= lib.einsum('aij,aIj,iA->IA', L_bab, R_bab, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += lib.einsum('aij,aij,IA->IA', L_bab, R_bab, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('aij,Aij,Ia->IA', L_bbb_u, R_bbb_u, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += lib.einsum('aij,aIi,jA->IA', L_bbb_u, R_bbb_u, t1_1_b, optimize=True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('aij,aij,IA->IA', L_bbb_u, R_bbb_u, t1_1_b, optimize=True)

    ####### ADC(3) SPIN ADAPTED EXCITED STATE OPDM WITH SQA ################
    if adc.method == "adc(3)":
        # Redudant Variables used for names from SQA
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
        temp += 1/4 * lib.einsum('J,i,Ijab,ijab->IJ', L_a, R_a, t2_1_a, t2_2_a, optimize = True)
        temp += 1/4 * lib.einsum('J,i,ijab,Ijab->IJ', L_a, R_a, t2_1_a, t2_2_a, optimize = True)
        temp -= 1/2 * lib.einsum('i,i,Ijab,Jjab->IJ', L_a, R_a, t2_1_a, t2_2_a, optimize = True)
        temp += 1/2 * lib.einsum('i,j,Iiab,Jjab->IJ', L_a, R_a, t2_1_a, t2_2_a, optimize = True)
        temp += 1/2 * lib.einsum('J,i,Ijab,ijab->IJ', L_a, R_a, t2_1_ab, t2_2_ab, optimize = True)
        temp += 1/2 * lib.einsum('J,i,ijab,Ijab->IJ', L_a, R_a, t2_1_ab, t2_2_ab, optimize = True)
        temp -= lib.einsum('i,i,Ijab,Jjab->IJ', L_a, R_a, t2_1_ab, t2_2_ab, optimize = True)
        temp -= 1/2 * lib.einsum('i,i,Ijab,Jjab->IJ', L_b, R_b, t2_1_a, t2_2_a, optimize = True)
        temp -= lib.einsum('i,i,Ijab,Jjab->IJ', L_b, R_b, t2_1_ab, t2_2_ab, optimize = True)
        temp += lib.einsum('i,j,Iiab,Jjab->IJ', L_b, R_b, t2_1_ab, t2_2_ab, optimize = True)
        rdm1_a[:nocc_a, :nocc_a] += temp + temp.T
        temp = np.zeros((nocc_b, nocc_b))
        temp -= lib.einsum('i,i,jIab,jJab->IJ', L_a, R_a, t2_1_ab, t2_2_ab, optimize = True)
        temp += lib.einsum('i,j,iIab,jJab->IJ', L_a, R_a, t2_1_ab, t2_2_ab, optimize = True)
        temp -= 1/2 * lib.einsum('i,i,Ijab,Jjab->IJ', L_a, R_a, t2_1_b, t2_2_b, optimize = True)
        temp += 1/2 * lib.einsum('J,i,jIab,jiab->IJ', L_b, R_b, t2_1_ab, t2_2_ab, optimize = True)
        temp += 1/2 * lib.einsum('J,i,jiab,jIab->IJ', L_b, R_b, t2_1_ab, t2_2_ab, optimize = True)
        temp -= lib.einsum('i,i,jIab,jJab->IJ', L_b, R_b, t2_1_ab, t2_2_ab, optimize = True)
        temp += 1/4 * lib.einsum('J,i,Ijab,ijab->IJ', L_b, R_b, t2_1_b, t2_2_b, optimize = True)
        temp += 1/4 * lib.einsum('J,i,ijab,Ijab->IJ', L_b, R_b, t2_1_b, t2_2_b, optimize = True)
        temp -= 1/2 * lib.einsum('i,i,Ijab,Jjab->IJ', L_b, R_b, t2_1_b, t2_2_b, optimize = True)
        temp += 1/2 * lib.einsum('i,j,Iiab,Jjab->IJ', L_b, R_b, t2_1_b, t2_2_b, optimize = True)
        rdm1_b[:nocc_b, :nocc_b] += temp + temp.T

        if t1_1_a is not None:
            temp = np.zeros((nocc_a, nocc_a))
            temp += 1/2 * lib.einsum('J,i,Ia,ia->IJ', L_a, R_a, t1_1_a, t1_2_a, optimize=True)
            temp += 1/2 * lib.einsum('J,i,ia,Ia->IJ', L_a, R_a, t1_1_a, t1_2_a, optimize=True)
            temp -= lib.einsum('i,i,Ia,Ja->IJ', L_a, R_a, t1_1_a, t1_2_a, optimize=True)
            temp -= lib.einsum('i,i,Ia,Ja->IJ', L_b, R_b, t1_1_a, t1_2_a, optimize=True)
            temp += 1/6 * lib.einsum('J,i,Ia,jb,ijab->IJ', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/3 * lib.einsum('J,i,ia,jb,Ijab->IJ', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('i,i,Ia,jb,Jjab->IJ', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            rdm1_a[:nocc_a, :nocc_a] += temp + temp.T
            rdm1_a[:nocc_a, :nocc_a] += 1/3 *  lib.einsum('J,i,Ijab,ia,jb->IJ', L_a, R_a, t2_1_a, t1_1_a, t1_1_a,
                optimize=True)
            rdm1_a[:nocc_a, :nocc_a] -= 1/6 *  lib.einsum('J,i,ijab,ja,Ib->IJ', L_a, R_a, t2_1_a, t1_1_a, t1_1_a,
                optimize=True)
            rdm1_a[:nocc_a, :nocc_a] += 1/3 *  lib.einsum('i,I,Jjab,ia,jb->IJ', L_a, R_a, t2_1_a, t1_1_a, t1_1_a,
                optimize=True)
            rdm1_a[:nocc_a, :nocc_a] -= 1/6 *  lib.einsum('i,I,ijab,ja,Jb->IJ', L_a, R_a, t2_1_a, t1_1_a, t1_1_a,
                optimize=True)
            rdm1_a[:nocc_a, :nocc_a] += 1/2 *  lib.einsum('i,i,Ijab,ja,Jb->IJ', L_a, R_a, t2_1_a, t1_1_a, t1_1_a,
                optimize=True)
            rdm1_a[:nocc_a, :nocc_a] += 1/2 *  lib.einsum('i,i,Jjab,ja,Ib->IJ', L_a, R_a, t2_1_a, t1_1_a, t1_1_a,
                optimize=True)
            rdm1_a[:nocc_a, :nocc_a] -= 1/2 *  lib.einsum('i,j,Iiab,ja,Jb->IJ', L_a, R_a, t2_1_a, t1_1_a, t1_1_a,
                optimize=True)
            rdm1_a[:nocc_a, :nocc_a] -= 1/2 *  lib.einsum('i,j,Jjab,ia,Ib->IJ', L_a, R_a, t2_1_a, t1_1_a, t1_1_a,
                optimize=True)
            temp = np.zeros((nocc_a, nocc_a))
            temp -= 1/2 * lib.einsum('i,i,Ia,jb,Jjab->IJ', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('i,j,Ia,ib,Jjab->IJ', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            rdm1_a[:nocc_a, :nocc_a] += temp + temp.T
            rdm1_a[:nocc_a, :nocc_a] += 1/2 *  lib.einsum('i,i,Ijab,ja,Jb->IJ', L_b, R_b, t2_1_a, t1_1_a, t1_1_a,
                optimize=True)
            rdm1_a[:nocc_a, :nocc_a] += 1/2 *  lib.einsum('i,i,Jjab,ja,Ib->IJ', L_b, R_b, t2_1_a, t1_1_a, t1_1_a,
                optimize=True)
            temp = np.zeros((nocc_b, nocc_b))
            temp -= lib.einsum('i,i,Ia,Ja->IJ', L_a, R_a, t1_1_b, t1_2_b, optimize=True)
            temp += 1/2 * lib.einsum('J,i,Ia,ia->IJ', L_b, R_b, t1_1_b, t1_2_b, optimize=True)
            temp += 1/2 * lib.einsum('J,i,ia,Ia->IJ', L_b, R_b, t1_1_b, t1_2_b, optimize=True)
            temp -= lib.einsum('i,i,Ia,Ja->IJ', L_b, R_b, t1_1_b, t1_2_b, optimize=True)
            temp -= 1/2 * lib.einsum('i,i,ja,Ib,jJab->IJ', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('i,j,ia,Ib,jJab->IJ', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            rdm1_b[:nocc_b, :nocc_b] += temp + temp.T
            rdm1_b[:nocc_b, :nocc_b] += 1/2 *  lib.einsum('i,i,Ijab,ja,Jb->IJ', L_a, R_a, t2_1_b, t1_1_b, t1_1_b,
                optimize=True)
            rdm1_b[:nocc_b, :nocc_b] += 1/2 *  lib.einsum('i,i,Jjab,ja,Ib->IJ', L_a, R_a, t2_1_b, t1_1_b, t1_1_b,
                optimize=True)
            temp = np.zeros((nocc_b, nocc_b))
            temp += 1/6 * lib.einsum('J,i,ja,Ib,jiab->IJ', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/3 * lib.einsum('J,i,ja,ib,jIab->IJ', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('i,i,ja,Ib,jJab->IJ', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            rdm1_b[:nocc_b, :nocc_b] += temp + temp.T
            rdm1_b[:nocc_b, :nocc_b] += 1/3 *  lib.einsum('J,i,Ijab,ia,jb->IJ', L_b, R_b, t2_1_b, t1_1_b, t1_1_b,
                optimize=True)
            rdm1_b[:nocc_b, :nocc_b] -= 1/6 *  lib.einsum('J,i,ijab,ja,Ib->IJ', L_b, R_b, t2_1_b, t1_1_b, t1_1_b,
                optimize=True)
            rdm1_b[:nocc_b, :nocc_b] += 1/3 *  lib.einsum('i,I,Jjab,ia,jb->IJ', L_b, R_b, t2_1_b, t1_1_b, t1_1_b,
                optimize=True)
            rdm1_b[:nocc_b, :nocc_b] -= 1/6 *  lib.einsum('i,I,ijab,ja,Jb->IJ', L_b, R_b, t2_1_b, t1_1_b, t1_1_b,
                optimize=True)
            rdm1_b[:nocc_b, :nocc_b] += 1/2 *  lib.einsum('i,i,Ijab,ja,Jb->IJ', L_b, R_b, t2_1_b, t1_1_b, t1_1_b,
                optimize=True)
            rdm1_b[:nocc_b, :nocc_b] += 1/2 *  lib.einsum('i,i,Jjab,ja,Ib->IJ', L_b, R_b, t2_1_b, t1_1_b, t1_1_b,
                optimize=True)
            rdm1_b[:nocc_b, :nocc_b] -= 1/2 *  lib.einsum('i,j,Iiab,ja,Jb->IJ', L_b, R_b, t2_1_b, t1_1_b, t1_1_b,
                optimize=True)
            rdm1_b[:nocc_b, :nocc_b] -= 1/2 *  lib.einsum('i,j,Jjab,ia,Ib->IJ', L_b, R_b, t2_1_b, t1_1_b, t1_1_b,
                optimize=True)

        ### 021 & 120 ###
        temp = np.zeros((nocc_a, nocc_a))
        temp -= lib.einsum('i,aIi,Ja->IJ', L_a, R_aaa_u, t1_2_a, optimize = True)
        temp -= lib.einsum('aJi,i,Ia->IJ', L_aba, R_b, t1_2_a, optimize = True)
        rdm1_a[:nocc_a, :nocc_a] += temp + temp.T
        temp = np.zeros((nocc_b, nocc_b))
        temp -= lib.einsum('i,aIi,Ja->IJ', L_a, R_bab, t1_2_b, optimize = True)
        temp -= lib.einsum('i,aIi,Ja->IJ', L_b, R_bbb_u, t1_2_b, optimize = True)
        rdm1_b[:nocc_b, :nocc_b] += temp + temp.T

        if t1_1_a is not None:
            temp = np.zeros((nocc_a, nocc_a))
            temp -= 1/2 * lib.einsum('i,aIi,jb,Jjab->IJ', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp += 1/2 * lib.einsum('i,aIj,ib,Jjab->IJ', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp += 1/4 * lib.einsum('J,aij,Ib,ijab->IJ', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp -= lib.einsum('i,aij,Ib,Jjab->IJ', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp -= 1/2 * lib.einsum('i,aIi,jb,Jjab->IJ', L_a, R_aaa_u, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('aJi,i,jb,Ijab->IJ', L_aba, R_b, t1_1_a, t2_1_a, optimize=True)
            temp += lib.einsum('aij,j,Jb,Iiab->IJ', L_aba, R_b, t1_1_a, t2_1_a, optimize=True)
            temp -= 1/2 * lib.einsum('aJi,i,jb,Ijab->IJ', L_aba, R_b, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('aJi,j,jb,Iiab->IJ', L_aba, R_b, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('i,ajI,ib,Jjba->IJ', L_a, R_bab, t1_1_a, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('J,aij,Ib,jiba->IJ', L_a, R_bab, t1_1_a, t2_1_ab, optimize=True)
            temp -= lib.einsum('i,aji,Ib,Jjba->IJ', L_a, R_bab, t1_1_a, t2_1_ab, optimize=True)
            temp += lib.einsum('i,aij,Ib,Jjba->IJ', L_b, R_bbb_u, t1_1_a, t2_1_ab, optimize=True)
            rdm1_a[:nocc_a, :nocc_a] += temp + temp.T
            temp = np.zeros((nocc_b, nocc_b))
            temp += lib.einsum('i,aij,Ib,jJab->IJ', L_a, R_aaa_u, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('aij,I,Jb,ijab->IJ', L_aba, R_b, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('aiJ,j,jb,iIab->IJ', L_aba, R_b, t1_1_b, t2_1_ab, optimize=True)
            temp -= lib.einsum('aij,j,Jb,iIab->IJ', L_aba, R_b, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('i,aIi,jb,jJba->IJ', L_a, R_bab, t1_1_a, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('i,aIj,ib,jJba->IJ', L_a, R_bab, t1_1_a, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('i,aIi,jb,Jjab->IJ', L_a, R_bab, t1_1_b, t2_1_b, optimize=True)
            temp += lib.einsum('i,aji,Ib,Jjab->IJ', L_a, R_bab, t1_1_b, t2_1_b, optimize=True)
            temp -= 1/2 * lib.einsum('i,aIi,jb,jJba->IJ', L_b, R_bbb_u, t1_1_a, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('i,aIi,jb,Jjab->IJ', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            temp += 1/2 * lib.einsum('i,aIj,ib,Jjab->IJ', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            temp += 1/4 * lib.einsum('J,aij,Ib,ijab->IJ', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            temp -= lib.einsum('i,aij,Ib,Jjab->IJ', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            rdm1_b[:nocc_b, :nocc_b] += temp + temp.T

# block- ab
        ### 030 ###
        temp = np.zeros((nvir_a, nvir_a))
        temp += 1/2 * lib.einsum('i,i,jkAa,jkBa->AB', L_a, R_a, t2_1_a, t2_2_a, optimize = True)
        temp -= lib.einsum('i,j,ikBa,jkAa->AB', L_a, R_a, t2_1_a, t2_2_a, optimize = True)
        temp += lib.einsum('i,i,jkAa,jkBa->AB', L_a, R_a, t2_1_ab, t2_2_ab, optimize = True)
        temp -= lib.einsum('i,j,ikBa,jkAa->AB', L_a, R_a, t2_1_ab, t2_2_ab, optimize = True)
        temp += 1/2 * lib.einsum('i,i,jkAa,jkBa->AB', L_b, R_b, t2_1_a, t2_2_a, optimize = True)
        temp += lib.einsum('i,i,jkAa,jkBa->AB', L_b, R_b, t2_1_ab, t2_2_ab, optimize = True)
        temp -= lib.einsum('i,j,kiBa,kjAa->AB', L_b, R_b, t2_1_ab, t2_2_ab, optimize = True)
        rdm1_a[nocc_a:, nocc_a:] += temp + temp.T
        temp = np.zeros((nvir_b, nvir_b))
        temp += lib.einsum('i,i,jkaA,jkaB->AB', L_a, R_a, t2_1_ab, t2_2_ab, optimize = True)
        temp -= lib.einsum('i,j,ikaB,jkaA->AB', L_a, R_a, t2_1_ab, t2_2_ab, optimize = True)
        temp += 1/2 * lib.einsum('i,i,jkAa,jkBa->AB', L_a, R_a, t2_1_b, t2_2_b, optimize = True)
        temp += lib.einsum('i,i,jkaA,jkaB->AB', L_b, R_b, t2_1_ab, t2_2_ab, optimize = True)
        temp -= lib.einsum('i,j,kiaB,kjaA->AB', L_b, R_b, t2_1_ab, t2_2_ab, optimize = True)
        temp += 1/2 * lib.einsum('i,i,jkAa,jkBa->AB', L_b, R_b, t2_1_b, t2_2_b, optimize = True)
        temp -= lib.einsum('i,j,ikBa,jkAa->AB', L_b, R_b, t2_1_b, t2_2_b, optimize = True)
        rdm1_b[nocc_b:, nocc_b:] += temp + temp.T

        if t1_1_a is not None:
            temp = np.zeros((nvir_a, nvir_a))
            temp += lib.einsum('i,i,jA,jB->AB', L_a, R_a, t1_1_a, t1_2_a, optimize=True)
            temp -= lib.einsum('i,j,iB,jA->AB', L_a, R_a, t1_1_a, t1_2_a, optimize=True)
            temp += lib.einsum('i,i,jA,jB->AB', L_b, R_b, t1_1_a, t1_2_a, optimize=True)
            temp += 1/2 * lib.einsum('i,i,jA,ka,jkBa->AB', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('i,j,iB,ka,jkAa->AB', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('i,i,jkAa,jB,ka->AB', L_a, R_a, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            temp -= 1/2 * lib.einsum('i,j,ikBa,jA,ka->AB', L_a, R_a, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            temp += 1/2 * lib.einsum('i,j,ikBa,ja,kA->AB', L_a, R_a, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            temp += 1/2 * lib.einsum('i,i,jA,ka,jkBa->AB', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('i,j,kA,ja,kiBa->AB', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('i,i,jkAa,jB,ka->AB', L_b, R_b, t2_1_a, t1_1_a, t1_1_a, optimize=True)
            rdm1_a[nocc_a:, nocc_a:] += temp + temp.T
            temp = np.zeros((nvir_b, nvir_b))
            temp += lib.einsum('i,i,jA,jB->AB', L_a, R_a, t1_1_b, t1_2_b, optimize=True)
            temp += lib.einsum('i,i,jA,jB->AB', L_b, R_b, t1_1_b, t1_2_b, optimize=True)
            temp -= lib.einsum('i,j,iB,jA->AB', L_b, R_b, t1_1_b, t1_2_b, optimize=True)
            temp += 1/2 * lib.einsum('i,i,ja,kA,jkaB->AB', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('i,j,ia,kB,jkaA->AB', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('i,i,jkAa,jB,ka->AB', L_a, R_a, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            temp += 1/2 * lib.einsum('i,i,ja,kA,jkaB->AB', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('i,j,ka,iB,kjaA->AB', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('i,i,jkAa,jB,ka->AB', L_b, R_b, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            temp -= 1/2 * lib.einsum('i,j,ikBa,jA,ka->AB', L_b, R_b, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            temp += 1/2 * lib.einsum('i,j,ikBa,ja,kA->AB', L_b, R_b, t2_1_b, t1_1_b, t1_1_b, optimize=True)
            rdm1_b[nocc_b:, nocc_b:] += temp + temp.T

        ### 021 & 120 ###
        temp = np.zeros((nvir_a, nvir_a))
        temp -= lib.einsum('i,Bij,jA->AB', L_a, R_aaa_u, t1_2_a, optimize = True)
        temp += lib.einsum('Aij,j,iB->AB', L_aba, R_b, t1_2_a, optimize = True)
        rdm1_a[nocc_a:, nocc_a:] += temp + temp.T
        temp = np.zeros((nvir_b, nvir_b))
        temp += lib.einsum('i,Bji,jA->AB', L_a, R_bab, t1_2_b, optimize = True)
        temp -= lib.einsum('i,Bij,jA->AB', L_b, R_bbb_u, t1_2_b, optimize = True)
        rdm1_b[nocc_b:, nocc_b:] += temp + temp.T

        if t1_1_a is not None:
            temp = np.zeros((nvir_a, nvir_a))
            temp -= 1/2 * lib.einsum('i,Bij,ka,jkAa->AB', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp -= 1/4 * lib.einsum('i,Bjk,ia,jkAa->AB', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp += lib.einsum('i,aij,kB,jkAa->AB', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp += 1/2 * lib.einsum('i,ajk,iB,jkAa->AB', L_a, R_aaa_u, t1_1_a, t2_1_a, optimize=True)
            temp -= 1/2 * lib.einsum('i,Bij,ka,jkAa->AB', L_a, R_aaa_u, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('Aij,j,ka,ikBa->AB', L_aba, R_b, t1_1_a, t2_1_a, optimize=True)
            temp -= lib.einsum('aij,j,kA,ikBa->AB', L_aba, R_b, t1_1_a, t2_1_a, optimize=True)
            temp += 1/2 * lib.einsum('Aij,j,ka,ikBa->AB', L_aba, R_b, t1_1_b, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('Aij,k,ka,ijBa->AB', L_aba, R_b, t1_1_b, t2_1_ab, optimize=True)
            temp += lib.einsum('i,aji,kB,kjAa->AB', L_a, R_bab, t1_1_a, t2_1_ab, optimize=True)
            temp -= lib.einsum('i,ajk,iB,kjAa->AB', L_a, R_bab, t1_1_a, t2_1_ab, optimize=True)
            temp -= lib.einsum('i,aij,kB,kjAa->AB', L_b, R_bbb_u, t1_1_a, t2_1_ab, optimize=True)
            rdm1_a[nocc_a:, nocc_a:] += temp + temp.T
            temp = np.zeros((nvir_b, nvir_b))
            temp -= lib.einsum('i,aij,kB,jkaA->AB', L_a, R_aaa_u, t1_1_b, t2_1_ab, optimize=True)
            temp += lib.einsum('aij,j,kA,ikaB->AB', L_aba, R_b, t1_1_b, t2_1_ab, optimize=True)
            temp -= lib.einsum('aij,k,kA,ijaB->AB', L_aba, R_b, t1_1_b, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('i,Bji,ka,kjaA->AB', L_a, R_bab, t1_1_a, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('i,Bjk,ia,kjaA->AB', L_a, R_bab, t1_1_a, t2_1_ab, optimize=True)
            temp += 1/2 * lib.einsum('i,Bji,ka,jkAa->AB', L_a, R_bab, t1_1_b, t2_1_b, optimize=True)
            temp -= lib.einsum('i,aji,kB,jkAa->AB', L_a, R_bab, t1_1_b, t2_1_b, optimize=True)
            temp -= 1/2 * lib.einsum('i,Bij,ka,kjaA->AB', L_b, R_bbb_u, t1_1_a, t2_1_ab, optimize=True)
            temp -= 1/2 * lib.einsum('i,Bij,ka,jkAa->AB', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            temp -= 1/4 * lib.einsum('i,Bjk,ia,jkAa->AB', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            temp += lib.einsum('i,aij,kB,jkAa->AB', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            temp += 1/2 * lib.einsum('i,ajk,iB,jkAa->AB', L_b, R_bbb_u, t1_1_b, t2_1_b, optimize=True)
            rdm1_b[nocc_b:, nocc_b:] += temp + temp.T

# block- ia
        ### 030 ###
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('i,I,ja,ijAa->IA', L_a, R_a, t1_2_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('i,i,ja,IjAa->IA', L_a, R_a, t1_2_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('i,j,ja,IiAa->IA', L_a, R_a, t1_2_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('i,I,ja,ijAa->IA', L_a, R_a, t1_2_b, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('i,i,ja,IjAa->IA', L_a, R_a, t1_2_b, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('i,i,ja,IjAa->IA', L_b, R_b, t1_2_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('i,i,ja,IjAa->IA', L_b, R_b, t1_2_b, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('i,j,ja,IiAa->IA', L_b, R_b, t1_2_b, t2_1_ab, optimize = True)

        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('i,i,ja,jIaA->IA', L_a, R_a, t1_2_a, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('i,j,ja,iIaA->IA', L_a, R_a, t1_2_a, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('i,i,ja,IjAa->IA', L_a, R_a, t1_2_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('i,I,ja,jiaA->IA', L_b, R_b, t1_2_a, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('i,i,ja,jIaA->IA', L_b, R_b, t1_2_a, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('i,I,ja,ijAa->IA', L_b, R_b, t1_2_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('i,i,ja,IjAa->IA', L_b, R_b, t1_2_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('i,j,ja,IiAa->IA', L_b, R_b, t1_2_b, t2_1_b, optimize = True)

        if adc.approx_trans_moments is False:
            rdm1_a[:nocc_a, nocc_a:] -= lib.einsum('i,I,iA->IA', L_a, R_a, t1_3_a, optimize = True)
            rdm1_a[:nocc_a, nocc_a:] += lib.einsum('i,i,IA->IA', L_a, R_a, t1_3_a, optimize = True)
            rdm1_a[:nocc_a, nocc_a:] += lib.einsum('i,i,IA->IA', L_b, R_b, t1_3_a, optimize = True)
            rdm1_b[:nocc_b, nocc_b:] += lib.einsum('i,i,IA->IA', L_a, R_a, t1_3_b, optimize = True)
            rdm1_b[:nocc_b, nocc_b:] -= lib.einsum('i,I,iA->IA', L_b, R_b, t1_3_b, optimize = True)
            rdm1_b[:nocc_b, nocc_b:] += lib.einsum('i,i,IA->IA', L_b, R_b, t1_3_b, optimize = True)

        if t1_1_a is not None:
            rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('i,I,ja,ijAa->IA', L_a, R_a, t1_1_a, t2_2_a, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('i,i,ja,IjAa->IA', L_a, R_a, t1_1_a, t2_2_a, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('i,j,ja,IiAa->IA', L_a, R_a, t1_1_a, t2_2_a, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('i,I,ja,ijAa->IA', L_a, R_a, t1_1_b, t2_2_ab, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('i,i,ja,IjAa->IA', L_a, R_a, t1_1_b, t2_2_ab, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('i,i,ja,IjAa->IA', L_b, R_b, t1_1_a, t2_2_a, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('i,i,ja,IjAa->IA', L_b, R_b, t1_1_b, t2_2_ab, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('i,j,ja,IiAa->IA', L_b, R_b, t1_1_b, t2_2_ab, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('i,I,ia,jA,ja->IA', L_a, R_a, t1_1_a, t1_1_a, t1_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 2/3 *  lib.einsum('i,i,Ia,jA,ja->IA', L_a, R_a, t1_1_a, t1_1_a, t1_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('i,j,iA,ja,Ia->IA', L_a, R_a, t1_1_a, t1_1_a, t1_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/12 *  lib.einsum('i,I,ia,jkab,jkAb->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/12 *  lib.einsum('i,I,jA,ikab,jkab->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/6 *  lib.einsum('i,I,ja,ikAb,jkab->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/3 *  lib.einsum('i,i,Ia,jkab,jkAb->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/3 *  lib.einsum('i,i,jA,jkab,Ikab->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('i,i,ja,jkab,IkAb->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 2/3 *  lib.einsum('i,j,Ia,ikAb,jkab->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/4 *  lib.einsum('i,j,iA,jkab,Ikab->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/6 *  lib.einsum('i,j,ia,jkab,IkAb->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/3 *  lib.einsum('i,j,kA,Iiab,jkab->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('i,j,ka,IiAb,jkab->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('i,I,ia,jkab,jkAb->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('i,I,jA,ikab,jkab->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/6 *  lib.einsum('i,I,ja,ikAb,jkab->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 2/3 *  lib.einsum('i,i,Ia,jkab,jkAb->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 2/3 *  lib.einsum('i,i,jA,jkab,Ikab->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('i,i,ja,jkab,IkAb->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 2/3 *  lib.einsum('i,j,Ia,ikAb,jkab->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('i,j,iA,jkab,Ikab->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/6 *  lib.einsum('i,j,ia,jkab,IkAb->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/6 *  lib.einsum('i,I,ja,ikAb,kjba->IA', L_a, R_a, t1_1_b, t2_1_a, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('i,i,ja,IkAb,kjba->IA', L_a, R_a, t1_1_b, t2_1_a, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/6 *  lib.einsum('i,j,ka,IiAb,jkba->IA', L_a, R_a, t1_1_b, t2_1_a, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/6 *  lib.einsum('i,I,ja,ikAb,jkab->IA', L_a, R_a, t1_1_b, t2_1_ab, t2_1_b,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('i,i,ja,IkAb,jkab->IA', L_a, R_a, t1_1_b, t2_1_ab, t2_1_b,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 2/3 *  lib.einsum('i,i,Ia,jA,ja->IA', L_b, R_b, t1_1_a, t1_1_a, t1_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/3 *  lib.einsum('i,i,Ia,jkab,jkAb->IA', L_b, R_b, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/3 *  lib.einsum('i,i,jA,jkab,Ikab->IA', L_b, R_b, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('i,i,ja,jkab,IkAb->IA', L_b, R_b, t1_1_a, t2_1_a, t2_1_a,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 2/3 *  lib.einsum('i,i,Ia,jkab,jkAb->IA', L_b, R_b, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 2/3 *  lib.einsum('i,i,jA,jkab,Ikab->IA', L_b, R_b, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('i,i,ja,jkab,IkAb->IA', L_b, R_b, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 2/3 *  lib.einsum('i,j,Ia,kiAb,kjab->IA', L_b, R_b, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 2/3 *  lib.einsum('i,j,kA,Iiab,kjab->IA', L_b, R_b, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/6 *  lib.einsum('i,j,ka,IiAb,kjab->IA', L_b, R_b, t1_1_a, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('i,i,ja,IkAb,kjba->IA', L_b, R_b, t1_1_b, t2_1_a, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/6 *  lib.einsum('i,j,ia,IkAb,kjba->IA', L_b, R_b, t1_1_b, t2_1_a, t2_1_ab,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('i,i,ja,IkAb,jkab->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_b,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/6 *  lib.einsum('i,j,ia,IkAb,jkab->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_b,
                optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += 1/6 *  lib.einsum('i,j,ka,IiAb,jkab->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('i,i,ja,jIaA->IA', L_a, R_a, t1_1_a, t2_2_ab, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('i,j,ja,iIaA->IA', L_a, R_a, t1_1_a, t2_2_ab, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('i,i,ja,IjAa->IA', L_a, R_a, t1_1_b, t2_2_b, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('i,I,ja,jiaA->IA', L_b, R_b, t1_1_a, t2_2_ab, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('i,i,ja,jIaA->IA', L_b, R_b, t1_1_a, t2_2_ab, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('i,I,ja,ijAa->IA', L_b, R_b, t1_1_b, t2_2_b, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('i,i,ja,IjAa->IA', L_b, R_b, t1_1_b, t2_2_b, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('i,j,ja,IiAa->IA', L_b, R_b, t1_1_b, t2_2_b, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('i,i,ja,jkab,kIbA->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/6 *  lib.einsum('i,j,ia,jkab,kIbA->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('i,j,ka,jkab,iIbA->IA', L_a, R_a, t1_1_a, t2_1_a, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('i,i,ja,jkab,IkAb->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/6 *  lib.einsum('i,j,ia,jkab,IkAb->IA', L_a, R_a, t1_1_a, t2_1_ab, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 2/3 *  lib.einsum('i,i,Ia,jA,ja->IA', L_a, R_a, t1_1_b, t1_1_b, t1_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 2/3 *  lib.einsum('i,i,Ia,jkba,jkbA->IA', L_a, R_a, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 2/3 *  lib.einsum('i,i,jA,kjab,kIab->IA', L_a, R_a, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('i,i,ja,kjba,kIbA->IA', L_a, R_a, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 2/3 *  lib.einsum('i,j,Ia,ikbA,jkba->IA', L_a, R_a, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 2/3 *  lib.einsum('i,j,kA,iIab,jkab->IA', L_a, R_a, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/6 *  lib.einsum('i,j,ka,iIbA,jkba->IA', L_a, R_a, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/3 *  lib.einsum('i,i,Ia,jkab,jkAb->IA', L_a, R_a, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/3 *  lib.einsum('i,i,jA,jkab,Ikab->IA', L_a, R_a, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('i,i,ja,jkab,IkAb->IA', L_a, R_a, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/6 *  lib.einsum('i,I,ja,jkab,kibA->IA', L_b, R_b, t1_1_a, t2_1_a, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('i,i,ja,jkab,kIbA->IA', L_b, R_b, t1_1_a, t2_1_a, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/6 *  lib.einsum('i,I,ja,jkab,ikAb->IA', L_b, R_b, t1_1_a, t2_1_ab, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('i,i,ja,jkab,IkAb->IA', L_b, R_b, t1_1_a, t2_1_ab, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/6 *  lib.einsum('i,j,ka,kjab,IiAb->IA', L_b, R_b, t1_1_a, t2_1_ab, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('i,I,ia,jA,ja->IA', L_b, R_b, t1_1_b, t1_1_b, t1_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 2/3 *  lib.einsum('i,i,Ia,jA,ja->IA', L_b, R_b, t1_1_b, t1_1_b, t1_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('i,j,iA,ja,Ia->IA', L_b, R_b, t1_1_b, t1_1_b, t1_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('i,I,ia,jkba,jkbA->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('i,I,jA,kiab,kjab->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/6 *  lib.einsum('i,I,ja,kibA,kjba->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 2/3 *  lib.einsum('i,i,Ia,jkba,jkbA->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 2/3 *  lib.einsum('i,i,jA,kjab,kIab->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('i,i,ja,kjba,kIbA->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 2/3 *  lib.einsum('i,j,Ia,kibA,kjba->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('i,j,iA,kjab,kIab->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/6 *  lib.einsum('i,j,ia,kjba,kIbA->IA', L_b, R_b, t1_1_b, t2_1_ab, t2_1_ab,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/12 *  lib.einsum('i,I,ia,jkab,jkAb->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/12 *  lib.einsum('i,I,jA,ikab,jkab->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/6 *  lib.einsum('i,I,ja,ikAb,jkab->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/3 *  lib.einsum('i,i,Ia,jkab,jkAb->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/3 *  lib.einsum('i,i,jA,jkab,Ikab->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('i,i,ja,jkab,IkAb->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 2/3 *  lib.einsum('i,j,Ia,ikAb,jkab->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/4 *  lib.einsum('i,j,iA,jkab,Ikab->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/6 *  lib.einsum('i,j,ia,jkab,IkAb->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/3 *  lib.einsum('i,j,kA,Iiab,jkab->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/6 *  lib.einsum('i,j,ka,IiAb,jkab->IA', L_b, R_b, t1_1_b, t2_1_b, t2_1_b,
                optimize=True)

        ### 021 & 120 ###
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('aij,I,ijAa->IA', L_aaa_u, R_a, t2_2_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= lib.einsum('aij,i,IjAa->IA', L_aaa_u, R_a, t2_2_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += lib.einsum('aij,j,IiAa->IA', L_aba, R_b, t2_2_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= lib.einsum('aij,I,jiAa->IA', L_bab, R_a, t2_2_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += lib.einsum('aij,j,IiAa->IA', L_bab, R_a, t2_2_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= lib.einsum('aij,i,IjAa->IA', L_bbb_u, R_b, t2_2_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/4 * lib.einsum('i,Aij,jkab,Ikab->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/8 * lib.einsum('i,Ajk,Iiab,jkab->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/4 * lib.einsum('i,aIi,jkab,jkAb->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('i,aIj,ikAb,jkab->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('i,aij,jkab,IkAb->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/4 * lib.einsum('i,ajk,IiAb,jkab->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('i,Aij,jkab,Ikab->IA',
                                                    L_a, R_aaa_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('i,aIi,jkab,jkAb->IA',
                                                    L_a, R_aaa_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('i,aIj,ikAb,jkab->IA',
                                                    L_a, R_aaa_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('i,aij,jkab,IkAb->IA',
                                                    L_a, R_aaa_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('i,ajI,ikAb,kjba->IA', L_a, R_bab, t2_1_a, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('i,aji,IkAb,kjba->IA', L_a, R_bab, t2_1_a, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('i,ajk,IiAb,kjba->IA', L_a, R_bab, t2_1_a, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('i,ajI,ikAb,jkab->IA', L_a, R_bab, t2_1_ab, t2_1_b, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('i,aji,IkAb,jkab->IA', L_a, R_bab, t2_1_ab, t2_1_b, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/4 * lib.einsum('i,Aji,jkab,Ikab->IA', L_b, R_aba, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/4 * lib.einsum('i,aIi,jkab,jkAb->IA', L_b, R_aba, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('i,aji,jkab,IkAb->IA', L_b, R_aba, t2_1_a, t2_1_a, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('i,Aji,jkab,Ikab->IA',
                                                    L_b, R_aba, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('i,Ajk,Iiab,jkab->IA',
                                                    L_b, R_aba, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('i,aIi,jkab,jkAb->IA',
                                                    L_b, R_aba, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('i,aIj,kiAb,kjab->IA',
                                                    L_b, R_aba, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] += 1/2 * lib.einsum('i,aji,jkab,IkAb->IA',
                                                    L_b, R_aba, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('i,ajk,IiAb,jkab->IA',
                                                    L_b, R_aba, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('i,aij,IkAb,kjba->IA',
                                                    L_b, R_bbb_u, t2_1_a, t2_1_ab, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/2 * lib.einsum('i,aij,IkAb,jkab->IA',
                                                    L_b, R_bbb_u, t2_1_ab, t2_1_b, optimize = True)
        rdm1_a[:nocc_a, nocc_a:] -= 1/4 * lib.einsum('i,ajk,IiAb,jkab->IA',
                                                    L_b, R_bbb_u, t2_1_ab, t2_1_b, optimize = True)

        rdm1_b[:nocc_b, nocc_b:] -= lib.einsum('aij,i,jIaA->IA', L_aaa_u, R_a, t2_2_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= lib.einsum('aij,I,ijaA->IA', L_aba, R_b, t2_2_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += lib.einsum('aij,j,iIaA->IA', L_aba, R_b, t2_2_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += lib.einsum('aij,j,IiAa->IA', L_bab, R_a, t2_2_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('aij,I,ijAa->IA', L_bbb_u, R_b, t2_2_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= lib.einsum('aij,i,IjAa->IA', L_bbb_u, R_b, t2_2_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('i,aij,jkab,kIbA->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/4 * lib.einsum('i,ajk,jkab,iIbA->IA',
                                                    L_a, R_aaa_u, t2_1_a, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('i,aij,jkab,IkAb->IA',
                                                    L_a, R_aaa_u, t2_1_ab, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('i,Aji,kjab,kIab->IA',
                                                    L_a, R_bab, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('i,Ajk,iIab,kjab->IA',
                                                    L_a, R_bab, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('i,aIi,jkba,jkbA->IA',
                                                    L_a, R_bab, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('i,aIj,ikbA,jkba->IA',
                                                    L_a, R_bab, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('i,aji,kjba,kIbA->IA',
                                                    L_a, R_bab, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('i,ajk,iIbA,kjba->IA',
                                                    L_a, R_bab, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/4 * lib.einsum('i,Aji,jkab,Ikab->IA', L_a, R_bab, t2_1_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/4 * lib.einsum('i,aIi,jkab,jkAb->IA', L_a, R_bab, t2_1_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('i,aji,jkab,IkAb->IA', L_a, R_bab, t2_1_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('i,ajI,jkab,kibA->IA', L_b, R_aba, t2_1_a, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('i,aji,jkab,kIbA->IA', L_b, R_aba, t2_1_a, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('i,ajI,jkab,ikAb->IA', L_b, R_aba, t2_1_ab, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('i,aji,jkab,IkAb->IA', L_b, R_aba, t2_1_ab, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('i,ajk,jkab,IiAb->IA', L_b, R_aba, t2_1_ab, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('i,Aij,kjab,kIab->IA',
                                                    L_b, R_bbb_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('i,aIi,jkba,jkbA->IA',
                                                    L_b, R_bbb_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('i,aIj,kibA,kjba->IA',
                                                    L_b, R_bbb_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('i,aij,kjba,kIbA->IA',
                                                    L_b, R_bbb_u, t2_1_ab, t2_1_ab, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/4 * lib.einsum('i,Aij,jkab,Ikab->IA',
                                                    L_b, R_bbb_u, t2_1_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/8 * lib.einsum('i,Ajk,Iiab,jkab->IA',
                                                    L_b, R_bbb_u, t2_1_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/4 * lib.einsum('i,aIi,jkab,jkAb->IA',
                                                    L_b, R_bbb_u, t2_1_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] += 1/2 * lib.einsum('i,aIj,ikAb,jkab->IA',
                                                    L_b, R_bbb_u, t2_1_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/2 * lib.einsum('i,aij,jkab,IkAb->IA',
                                                    L_b, R_bbb_u, t2_1_b, t2_1_b, optimize = True)
        rdm1_b[:nocc_b, nocc_b:] -= 1/4 * lib.einsum('i,ajk,IiAb,jkab->IA',
                                                    L_b, R_bbb_u, t2_1_b, t2_1_b, optimize = True)

        if t1_1_a is not None:
            rdm1_a[:nocc_a, nocc_a:] += 1/2 *  lib.einsum('i,Aij,ja,Ia->IA', L_a, R_aaa_u, t1_1_a, t1_1_a, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('i,aIi,ja,jA->IA', L_a, R_aaa_u, t1_1_a, t1_1_a, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] += lib.einsum('aij,i,Ia,jA->IA', L_aaa_u, R_a, t1_1_a, t1_1_a, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= lib.einsum('aij,j,Ia,iA->IA', L_aba, R_b, t1_1_a, t1_1_a, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('i,Aji,ja,Ia->IA', L_b, R_aba, t1_1_a, t1_1_a, optimize=True)
            rdm1_a[:nocc_a, nocc_a:] -= 1/2 *  lib.einsum('i,aIi,ja,jA->IA', L_b, R_aba, t1_1_a, t1_1_a, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('i,Aji,ja,Ia->IA', L_a, R_bab, t1_1_b, t1_1_b, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('i,aIi,ja,jA->IA', L_a, R_bab, t1_1_b, t1_1_b, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += 1/2 *  lib.einsum('i,Aij,ja,Ia->IA', L_b, R_bbb_u, t1_1_b, t1_1_b, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= 1/2 *  lib.einsum('i,aIi,ja,jA->IA', L_b, R_bbb_u, t1_1_b, t1_1_b, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] -= lib.einsum('aij,j,Ia,iA->IA', L_bab, R_a, t1_1_b, t1_1_b, optimize=True)
            rdm1_b[:nocc_b, nocc_b:] += lib.einsum('aij,i,Ia,jA->IA', L_bbb_u, R_b, t1_1_b, t1_1_b, optimize=True)

    # block- ai
    rdm1_a[nocc_a:,:nocc_a] = rdm1_a[:nocc_a,nocc_a:].T
    rdm1_b[nocc_b:,:nocc_b] = rdm1_b[:nocc_b,nocc_b:].T

    return (rdm1_a, rdm1_b)

def get_spin_square(adc):
    '''
    <S^2> expectation values (2-RDM term of Eq. 17, J. Chem. Phys. 157, 044106)
    including the 3h1p/3p1h 2-RDM sectors (J. Chem. Phys. 164, 209901 (2026) erratum)
    and the cross-sector classes 100 (bare 1h/2h1p block, order 1) and
    110 (one-t2_1 1h/2h1p block, order 2), which are present in the exact
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

    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)
    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a * (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a * nocc_b
    n_doubles_aba = nvir_a * nocc_b * nocc_a
    n_doubles_bbb = nocc_b * (nocc_b - 1) * nvir_b // 2

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
        w_t2t2_bc = lib.einsum('lmbd,lmcd->bc', t2_1_a, t2_1_a, optimize=True)
        w_t2t2_ac = lib.einsum('lmda,lmdc->ac', t2_1_ab, t2_1_ab, optimize=True)
        w_t2t2_bc_2 = lib.einsum('lmbd,lmcd->bc', t2_1_ab, t2_1_ab, optimize=True)
        w_t2t2_ac_2 = lib.einsum('lmad,lmcd->ac', t2_1_b, t2_1_b, optimize=True)
    spin = np.array([])
    trace_a = np.array([])
    trace_b = np.array([])

    t1_1_a = adc.t1[2][0]
    t1_1_b = adc.t1[2][1] if t1_1_a is not None else None

    for r in range(U.shape[0]):

        vec = U[r]

        L_a = vec[s_a:f_a].copy()
        L_b = vec[s_b:f_b].copy()
        L_aaa = vec[s_aaa:f_aaa].reshape(nvir_a, -1)
        L_bab = vec[s_bab:f_bab].reshape(nvir_b, nocc_b, nocc_a)
        L_aba = vec[s_aba:f_aba].reshape(nvir_a, nocc_a, nocc_b)
        L_bbb = vec[s_bbb:f_bbb].reshape(nvir_b, -1)

        L_aaa_u = np.zeros((nvir_a, nocc_a, nocc_a))
        L_aaa_u[:, ij_ind_a[0], ij_ind_a[1]] = L_aaa
        L_aaa_u[:, ij_ind_a[1], ij_ind_a[0]] = -L_aaa
        L_bbb_u = np.zeros((nvir_b, nocc_b, nocc_b))
        L_bbb_u[:, ij_ind_b[0], ij_ind_b[1]] = L_bbb
        L_bbb_u[:, ij_ind_b[1], ij_ind_b[0]] = -L_bbb

        R_a = L_a
        R_b = L_b
        R_aaa_u = L_aaa_u
        R_aba = L_aba
        R_bab = L_bab
        R_bbb_u = L_bbb_u
        S2 = 0.0
# 000
        # block IjlJ
        S2 -= lib.einsum('i,i,jk,jk', L_a, R_a, S_oo_ab, S_oo_ab, optimize=True)
        S2 += lib.einsum('i,j,ik,jk', L_a, R_a, S_oo_ab, S_oo_ab, optimize=True)
        S2 -= lib.einsum('i,i,jk,jk', L_b, R_b, S_oo_ab, S_oo_ab, optimize=True)
        S2 += lib.einsum('i,j,ki,kj', L_b, R_b, S_oo_ab, S_oo_ab, optimize=True)
# 010
        if t1_1_a is not None:
            # block IjlB & AjlJ
            S2 -= 2 * lib.einsum('i,i,jk,ak,ja', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
            S2 += 2 * lib.einsum('i,j,ik,ak,ja', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
            S2 -= 2 * lib.einsum('i,i,jk,ak,ja', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
            S2 += 2 * lib.einsum('i,j,kj,ai,ka', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
            # block IjdJ & IblJ
            S2 -= 2 * lib.einsum('i,i,jk,ja,ka', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
            S2 += 2 * lib.einsum('i,j,jk,ia,ka', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
            S2 -= 2 * lib.einsum('i,i,jk,ja,ka', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
            S2 += 2 * lib.einsum('i,j,ki,ka,ja', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)

        # block IjdB & AblJ
        S2 -= 2 * lib.einsum('i,i,ja,bk,jkba', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
        S2 += 2 * lib.einsum('i,j,ia,bk,jkba', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('i,i,ja,bk,jkba', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
        S2 += 2 * lib.einsum('i,j,ka,bi,kjba', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
# 020
        if t1_1_a is not None:
            # block IjlJ
            S2 += lib.einsum('i,i,jk,lk,ja,la', L_a, R_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a, optimize=True)
            S2 -= 1/2 * lib.einsum('i,j,ik,lk,ja,la', L_a, R_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a, optimize=True)
            S2 -= 1/2 * lib.einsum('i,j,jk,lk,ia,la', L_a, R_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a, optimize=True)
            S2 += lib.einsum('i,i,jk,jl,ka,la', L_a, R_a, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b, optimize=True)
            S2 -= lib.einsum('i,j,ik,jl,ka,la', L_a, R_a, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b, optimize=True)
            S2 += lib.einsum('i,i,jk,lk,ja,la', L_b, R_b, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a, optimize=True)
            S2 -= lib.einsum('i,j,ki,lj,ka,la', L_b, R_b, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a, optimize=True)
            S2 += lib.einsum('i,i,jk,jl,ka,la', L_b, R_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b, optimize=True)
            S2 -= 1/2 * lib.einsum('i,j,ki,kl,ja,la', L_b, R_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b, optimize=True)
            S2 -= 1/2 * lib.einsum('i,j,kj,kl,ia,la', L_b, R_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b, optimize=True)
            # block IjlB & AjlJ
            S2 -= lib.einsum('i,i,jk,ak,lb,jlab', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, optimize=True)
            S2 += lib.einsum('i,j,ik,ak,lb,jlab', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, optimize=True)
            S2 -= lib.einsum('i,j,kl,al,ib,jkab', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, optimize=True)
            S2 -= lib.einsum('i,i,jk,ak,lb,jlab', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,i,jk,al,kb,jlab', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,j,ik,ak,lb,jlab', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,j,ik,al,kb,jlab', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,i,jk,ak,lb,jlab', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, optimize=True)
            S2 += lib.einsum('i,j,kj,ai,lb,klab', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, optimize=True)
            S2 -= lib.einsum('i,i,jk,ak,lb,jlab', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,i,jk,al,kb,jlab', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,j,kj,ai,lb,klab', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,j,kj,al,ib,klab', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,j,kl,ai,lb,kjab', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,j,kl,al,ib,kjab', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
            # block IjdJ & IblJ
            S2 -= lib.einsum('i,i,jk,ja,lb,lkba', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,i,jk,la,jb,lkba', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,j,jk,ia,lb,lkba', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,j,jk,la,ib,lkba', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,j,kl,ia,kb,jlba', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,j,kl,ka,ib,jlba', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,i,jk,ja,lb,klab', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, optimize=True)
            S2 += lib.einsum('i,j,jk,ia,lb,klab', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, optimize=True)
            S2 -= lib.einsum('i,i,jk,ja,lb,lkba', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,i,jk,la,jb,lkba', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,j,ki,ka,lb,ljba', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,j,ki,la,kb,ljba', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,i,jk,ja,lb,klab', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, optimize=True)
            S2 += lib.einsum('i,j,ki,ka,lb,jlab', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, optimize=True)
            S2 -= lib.einsum('i,j,kl,ka,ib,jlab', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, optimize=True)
            # block IjdB & AblJ
            S2 -= 2 * lib.einsum('i,i,ja,bk,jb,ka', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
            S2 += 2 * lib.einsum('i,j,ia,bk,jb,ka', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
            S2 -= 2 * lib.einsum('i,i,ja,bk,jb,ka', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
            S2 += 2 * lib.einsum('i,j,ka,bi,kb,ja', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
            # block IblB & AjdJ
            S2 -= 2 * lib.einsum('i,i,jk,ab,ja,kb', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
            S2 += 2 * lib.einsum('i,j,ik,ab,ja,kb', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
            S2 -= 2 * lib.einsum('i,i,jk,ab,ja,kb', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
            S2 += 2 * lib.einsum('i,j,kj,ab,ka,ib', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
            # block IbdJ
            S2 -= lib.einsum('i,i,ja,jb,ka,kb', L_a, R_a, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
            S2 += lib.einsum('i,j,ia,jb,ka,kb', L_a, R_a, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
            S2 -= lib.einsum('i,i,ja,jb,ka,kb', L_b, R_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
            S2 += lib.einsum('i,j,ka,kb,ia,jb', L_b, R_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
            # block IbdB & AbdJ
            S2 -= 2 * lib.einsum('i,i,ja,bc,kc,jkba', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,j,ia,bc,kc,jkba', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,i,ja,bc,kc,jkba', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,j,ka,bc,ic,kjba', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
            # block AjlB
            S2 -= lib.einsum('i,i,aj,bj,ka,kb', L_a, R_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
            S2 += lib.einsum('i,j,ak,bk,ia,jb', L_a, R_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
            S2 -= lib.einsum('i,i,aj,bj,ka,kb', L_b, R_b, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
            S2 += lib.einsum('i,j,ai,bj,ka,kb', L_b, R_b, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
            # block AjdB & AblB
            S2 -= 2 * lib.einsum('i,i,aj,bc,kb,kjac', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,j,ak,bc,ib,jkac', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,i,aj,bc,kb,kjac', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,j,ai,bc,kb,kjac', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)

        # block AbdB
        S2 -= lib.einsum('i,i,ab,cd,jkad,jkcb', L_a, R_a, S_vv_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('i,j,ab,cd,ikad,jkcb', L_a, R_a, S_vv_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('i,i,ab,cd,jkad,jkcb', L_b, R_b, S_vv_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('i,j,ab,cd,kiad,kjcb', L_b, R_b, S_vv_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
        # block AjlB
        S2 -= 1 / 2 * lib.einsum('i,i,aj,bj,klac,klbc', L_a, R_a, S_vo_ab, S_vo_ab, t2_1_a, t2_1_a, optimize=True)
        S2 += lib.einsum('i,j,ak,bk,ilac,jlbc', L_a, R_a, S_vo_ab, S_vo_ab, t2_1_a, t2_1_a, optimize=True)
        S2 -= lib.einsum('i,i,aj,bj,klac,klbc', L_a, R_a, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('i,i,aj,bk,ljac,lkbc', L_a, R_a, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('i,j,ak,bk,ilac,jlbc', L_a, R_a, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('i,j,ak,bl,ikac,jlbc', L_a, R_a, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= 1 / 2 * lib.einsum('i,i,aj,bj,klac,klbc', L_b, R_b, S_vo_ab, S_vo_ab, t2_1_a, t2_1_a, optimize=True)
        S2 += 1 / 2 * lib.einsum('i,j,ai,bj,klac,klbc', L_b, R_b, S_vo_ab, S_vo_ab, t2_1_a, t2_1_a, optimize=True)
        S2 -= lib.einsum('i,i,aj,bj,klac,klbc', L_b, R_b, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('i,i,aj,bk,ljac,lkbc', L_b, R_b, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('i,j,ai,bj,klac,klbc', L_b, R_b, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('i,j,ai,bk,ljac,lkbc', L_b, R_b, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('i,j,aj,bk,liac,lkbc', L_b, R_b, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('i,j,ak,bk,liac,ljbc', L_b, R_b, S_vo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
        # block IbdJ
        S2 -= lib.einsum('i,i,ja,jb,klca,klcb', L_a, R_a, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('i,i,ja,kb,jlca,klcb', L_a, R_a, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('i,j,ia,jb,klca,klcb', L_a, R_a, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('i,j,ia,kb,jlca,klcb', L_a, R_a, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('i,j,ja,kb,ilca,klcb', L_a, R_a, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('i,j,ka,kb,ilca,jlcb', L_a, R_a, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= 1 / 2 * lib.einsum('i,i,ja,jb,klac,klbc', L_a, R_a, S_ov_ab, S_ov_ab, t2_1_b, t2_1_b, optimize=True)
        S2 += 1 / 2 * lib.einsum('i,j,ia,jb,klac,klbc', L_a, R_a, S_ov_ab, S_ov_ab, t2_1_b, t2_1_b, optimize=True)
        S2 -= lib.einsum('i,i,ja,jb,klca,klcb', L_b, R_b, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('i,i,ja,kb,jlca,klcb', L_b, R_b, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('i,j,ka,kb,lica,ljcb', L_b, R_b, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('i,j,ka,lb,kica,ljcb', L_b, R_b, S_ov_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= 1 / 2 * lib.einsum('i,i,ja,jb,klac,klbc', L_b, R_b, S_ov_ab, S_ov_ab, t2_1_b, t2_1_b, optimize=True)
        S2 += lib.einsum('i,j,ka,kb,ilac,jlbc', L_b, R_b, S_ov_ab, S_ov_ab, t2_1_b, t2_1_b, optimize=True)
        # block IblB & AjdJ
        S2 -= 2 * lib.einsum('i,i,jk,ab,jlac,lkcb', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
        S2 += 2 * lib.einsum('i,j,ik,ab,jlac,lkcb', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('i,j,kl,ab,jkac,ilcb', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('i,i,jk,ab,jlac,klbc', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
        S2 += 2 * lib.einsum('i,j,ik,ab,jlac,klbc', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
        S2 -= 2 * lib.einsum('i,i,jk,ab,jlac,lkcb', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
        S2 += 2 * lib.einsum('i,j,kj,ab,klac,licb', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('i,i,jk,ab,jlac,klbc', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
        S2 += 2 * lib.einsum('i,j,kj,ab,klac,ilbc', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
        S2 -= 2 * lib.einsum('i,j,kl,ab,kjac,ilbc', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
        # block IjlJ
        S2 += 1 / 2 * lib.einsum('i,i,jk,lk,jmab,lmab', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_a, t2_1_a, optimize=True)
        S2 -= 1 / 4 * lib.einsum('i,j,ik,lk,jmab,lmab', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_a, t2_1_a, optimize=True)
        S2 -= 1 / 4 * lib.einsum('i,j,jk,lk,imab,lmab', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_a, t2_1_a, optimize=True)
        S2 -= 1 / 2 * lib.einsum('i,j,kl,ml,ikab,jmab', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_a, t2_1_a, optimize=True)
        S2 += lib.einsum('i,i,jk,jl,mkab,mlab', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('i,i,jk,lk,jmab,lmab', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('i,i,jk,lm,jmab,lkab', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('i,j,ik,jl,mkab,mlab', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= 1 / 2 * lib.einsum('i,j,ik,lk,jmab,lmab', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('i,j,ik,lm,jmab,lkab', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= 1 / 2 * lib.einsum('i,j,jk,lk,imab,lmab', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('i,j,jk,lm,imab,lkab', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('i,j,kl,km,ilab,jmab', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += 1 / 2 * lib.einsum('i,i,jk,jl,kmab,lmab', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_b, t2_1_b, optimize=True)
        S2 -= 1 / 2 * lib.einsum('i,j,ik,jl,kmab,lmab', L_a, R_a, S_oo_ab, S_oo_ab, t2_1_b, t2_1_b, optimize=True)
        S2 += 1 / 2 * lib.einsum('i,i,jk,lk,jmab,lmab', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_a, t2_1_a, optimize=True)
        S2 -= 1 / 2 * lib.einsum('i,j,ki,lj,kmab,lmab', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_a, t2_1_a, optimize=True)
        S2 += lib.einsum('i,i,jk,jl,mkab,mlab', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('i,i,jk,lk,jmab,lmab', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('i,i,jk,lm,jmab,lkab', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= 1 / 2 * lib.einsum('i,j,ki,kl,mjab,mlab', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('i,j,ki,lj,kmab,lmab', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('i,j,ki,lm,ljab,kmab', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= 1 / 2 * lib.einsum('i,j,kj,kl,miab,mlab', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += lib.einsum('i,j,kj,lm,liab,kmab', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 -= lib.einsum('i,j,kl,ml,kiab,mjab', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_ab, t2_1_ab, optimize=True)
        S2 += 1 / 2 * lib.einsum('i,i,jk,jl,kmab,lmab', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_b, t2_1_b, optimize=True)
        S2 -= 1 / 4 * lib.einsum('i,j,ki,kl,jmab,lmab', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_b, t2_1_b, optimize=True)
        S2 -= 1 / 4 * lib.einsum('i,j,kj,kl,imab,lmab', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_b, t2_1_b, optimize=True)
        S2 -= 1 / 2 * lib.einsum('i,j,kl,km,ilab,jmab', L_b, R_b, S_oo_ab, S_oo_ab, t2_1_b, t2_1_b, optimize=True)

        if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
            # block IjdB & AblJ
            S2 -= 2 * lib.einsum('i,i,ja,bk,jkba', L_a, R_a, S_ov_ab, S_vo_ab, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('i,j,ia,bk,jkba', L_a, R_a, S_ov_ab, S_vo_ab, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,i,ja,bk,jkba', L_b, R_b, S_ov_ab, S_vo_ab, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('i,j,ka,bi,kjba', L_b, R_b, S_ov_ab, S_vo_ab, t2_2_ab, optimize=True)
            # block IjlB & AjlJ
            S2 -= 2 * lib.einsum('i,i,jk,ak,ja', L_a, R_a, S_oo_ab, S_vo_ab, t1_2_a, optimize=True)
            S2 += 2 * lib.einsum('i,j,ik,ak,ja', L_a, R_a, S_oo_ab, S_vo_ab, t1_2_a, optimize=True)
            S2 -= 2 * lib.einsum('i,i,jk,ak,ja', L_b, R_b, S_oo_ab, S_vo_ab, t1_2_a, optimize=True)
            S2 += 2 * lib.einsum('i,j,kj,ai,ka', L_b, R_b, S_oo_ab, S_vo_ab, t1_2_a, optimize=True)
            # block IjdJ & IblJ
            S2 -= 2 * lib.einsum('i,i,jk,ja,ka', L_a, R_a, S_oo_ab, S_ov_ab, t1_2_b, optimize=True)
            S2 += 2 * lib.einsum('i,j,jk,ia,ka', L_a, R_a, S_oo_ab, S_ov_ab, t1_2_b, optimize=True)
            S2 -= 2 * lib.einsum('i,i,jk,ja,ka', L_b, R_b, S_oo_ab, S_ov_ab, t1_2_b, optimize=True)
            S2 += 2 * lib.einsum('i,j,ki,ka,ja', L_b, R_b, S_oo_ab, S_ov_ab, t1_2_b, optimize=True)
# 101
        # block AjlB
        S2 -= 1 / 2 * lib.einsum('aij,bij,ak,bk', L_aaa_u, R_aaa_u, S_vo_ab, S_vo_ab, optimize=True)
        S2 -= lib.einsum('aij,bij,ak,bk', L_aba, R_aba, S_vo_ab, S_vo_ab, optimize=True)
        S2 += lib.einsum('aij,bik,aj,bk', L_aba, R_aba, S_vo_ab, S_vo_ab, optimize=True)
        # block IbdJ
        S2 -= lib.einsum('aij,bij,ka,kb', L_bab, R_bab, S_ov_ab, S_ov_ab, optimize=True)
        S2 += lib.einsum('aij,bik,ja,kb', L_bab, R_bab, S_ov_ab, S_ov_ab, optimize=True)
        S2 -= 1 / 2 * lib.einsum('aij,bij,ka,kb', L_bbb_u, R_bbb_u, S_ov_ab, S_ov_ab, optimize=True)
        # block IblB & AjdJ
        S2 += 2 * lib.einsum('aij,bki,jk,ab', L_aaa_u, R_bab, S_oo_ab, S_vv_ab, optimize=True)
        S2 += 2 * lib.einsum('aij,bjk,ik,ab', L_aba, R_bbb_u, S_oo_ab, S_vv_ab, optimize=True)
        # block IjlJ
        S2 -= 1 / 2 * lib.einsum('aij,aij,kl,kl', L_aaa_u, R_aaa_u, S_oo_ab, S_oo_ab, optimize=True)
        S2 += lib.einsum('aij,aik,jl,kl', L_aaa_u, R_aaa_u, S_oo_ab, S_oo_ab, optimize=True)
        S2 -= lib.einsum('aij,aij,kl,kl', L_aba, R_aba, S_oo_ab, S_oo_ab, optimize=True)
        S2 += lib.einsum('aij,aik,lj,lk', L_aba, R_aba, S_oo_ab, S_oo_ab, optimize=True)
        S2 += lib.einsum('aij,akj,il,kl', L_aba, R_aba, S_oo_ab, S_oo_ab, optimize=True)
        S2 -= lib.einsum('aij,akl,il,kj', L_aba, R_aba, S_oo_ab, S_oo_ab, optimize=True)
        S2 -= lib.einsum('aij,aij,kl,kl', L_bab, R_bab, S_oo_ab, S_oo_ab, optimize=True)
        S2 += lib.einsum('aij,aik,jl,kl', L_bab, R_bab, S_oo_ab, S_oo_ab, optimize=True)
        S2 += lib.einsum('aij,akj,li,lk', L_bab, R_bab, S_oo_ab, S_oo_ab, optimize=True)
        S2 -= lib.einsum('aij,akl,li,jk', L_bab, R_bab, S_oo_ab, S_oo_ab, optimize=True)
        S2 -= 1 / 2 * lib.einsum('aij,aij,kl,kl', L_bbb_u, R_bbb_u, S_oo_ab, S_oo_ab, optimize=True)
        S2 += lib.einsum('aij,aik,lj,lk', L_bbb_u, R_bbb_u, S_oo_ab, S_oo_ab, optimize=True)
# 100
        # block IjdJ & IblJ
        S2 -= 2 * lib.einsum('aij,j,ki,ka', L_bab, R_a, S_oo_ab, S_ov_ab, optimize=True)
        S2 += 2 * lib.einsum('aij,k,ki,ja', L_bab, R_a, S_oo_ab, S_ov_ab, optimize=True)
        S2 += 2 * lib.einsum('aij,i,kj,ka', L_bbb_u, R_b, S_oo_ab, S_ov_ab, optimize=True)
        # block IjlB & AjlJ
        S2 += 2 * lib.einsum('aij,i,jk,ak', L_aaa_u, R_a, S_oo_ab, S_vo_ab, optimize=True)
        S2 -= 2 * lib.einsum('aij,j,ik,ak', L_aba, R_b, S_oo_ab, S_vo_ab, optimize=True)
        S2 += 2 * lib.einsum('aij,k,ik,aj', L_aba, R_b, S_oo_ab, S_vo_ab, optimize=True)
# 110
        if t1_1_a is not None:
            # block IjlJ
            S2 -= lib.einsum('i,aij,ka,jl,kl', L_a, R_aaa_u, t1_1_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('i,aji,ka,lj,lk', L_a, R_bab, t1_1_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,la,ij,kl', L_a, R_bab, t1_1_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('aij,i,ka,jl,kl', L_aaa_u, R_a, t1_1_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('aij,j,ka,il,kl', L_aba, R_b, t1_1_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('aij,k,la,ik,lj', L_aba, R_b, t1_1_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('i,aji,ka,jl,kl', L_b, R_aba, t1_1_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,la,ji,lk', L_b, R_aba, t1_1_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('i,aij,ka,lj,lk', L_b, R_bbb_u, t1_1_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('aij,j,ka,li,lk', L_bab, R_a, t1_1_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('aij,k,la,ki,jl', L_bab, R_a, t1_1_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('aij,i,ka,lj,lk', L_bbb_u, R_b, t1_1_b, S_oo_ab, S_oo_ab, optimize=True)
            # block IjdB & AblJ
            S2 += 2 * lib.einsum('aij,i,jb,ak,kb', L_aaa_u, R_a, S_ov_ab, S_vo_ab, t1_1_b, optimize=True)
            S2 -= 2 * lib.einsum('aij,j,ib,ak,kb', L_aba, R_b, S_ov_ab, S_vo_ab, t1_1_b, optimize=True)
            S2 += 2 * lib.einsum('aij,k,ib,aj,kb', L_aba, R_b, S_ov_ab, S_vo_ab, t1_1_b, optimize=True)
            S2 -= 2 * lib.einsum('aij,j,ka,bi,kb', L_bab, R_a, S_ov_ab, S_vo_ab, t1_1_a, optimize=True)
            S2 += 2 * lib.einsum('aij,k,ja,bi,kb', L_bab, R_a, S_ov_ab, S_vo_ab, t1_1_a, optimize=True)
            S2 += 2 * lib.einsum('aij,i,ka,bj,kb', L_bbb_u, R_b, S_ov_ab, S_vo_ab, t1_1_a, optimize=True)
            # block IblB & AjdJ
            S2 -= 2 * lib.einsum('i,aji,kj,ba,kb', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_a, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,ij,ba,kb', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_a, optimize=True)
            S2 += 2 * lib.einsum('aij,i,jk,ab,kb', L_aaa_u, R_a, S_oo_ab, S_vv_ab, t1_1_b, optimize=True)
            S2 -= 2 * lib.einsum('aij,j,ik,ab,kb', L_aba, R_b, S_oo_ab, S_vv_ab, t1_1_b, optimize=True)
            S2 += 2 * lib.einsum('aij,k,ik,ab,jb', L_aba, R_b, S_oo_ab, S_vv_ab, t1_1_b, optimize=True)
            S2 += 2 * lib.einsum('i,aij,kj,ba,kb', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_a, optimize=True)
            # block IbdJ
            S2 -= lib.einsum('i,aji,jb,ka,kb', L_a, R_bab, t1_1_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 += lib.einsum('i,ajk,jb,ib,ka', L_a, R_bab, t1_1_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 += lib.einsum('i,aij,jb,ka,kb', L_b, R_bbb_u, t1_1_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= lib.einsum('aij,j,ib,ka,kb', L_bab, R_a, t1_1_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 += lib.einsum('aij,k,ib,ja,kb', L_bab, R_a, t1_1_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 += lib.einsum('aij,i,jb,ka,kb', L_bbb_u, R_b, t1_1_b, S_ov_ab, S_ov_ab, optimize=True)
            # block AjlB
            S2 += lib.einsum('i,aij,jb,ak,bk', L_a, R_aaa_u, t1_1_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 += lib.einsum('aij,i,jb,ak,bk', L_aaa_u, R_a, t1_1_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= lib.einsum('aij,j,ib,ak,bk', L_aba, R_b, t1_1_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 += lib.einsum('aij,k,ib,aj,bk', L_aba, R_b, t1_1_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= lib.einsum('i,aji,jb,ak,bk', L_b, R_aba, t1_1_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 += lib.einsum('i,ajk,jb,bi,ak', L_b, R_aba, t1_1_a, S_vo_ab, S_vo_ab, optimize=True)

        # block AjdB & AblB
        S2 += 2 * lib.einsum('i,aij,bk,ac,jkbc', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('i,aji,bk,ac,jkbc', L_b, R_aba, S_vo_ab, S_vv_ab, t2_1_ab, optimize=True)
        S2 += 2 * lib.einsum('i,ajk,bi,ac,jkbc', L_b, R_aba, S_vo_ab, S_vv_ab, t2_1_ab, optimize=True)
        # block IbdB & AbdJ
        S2 -= 2 * lib.einsum('i,aji,kb,ca,kjcb', L_a, R_bab, S_ov_ab, S_vv_ab, t2_1_ab, optimize=True)
        S2 += 2 * lib.einsum('i,ajk,ib,ca,kjcb', L_a, R_bab, S_ov_ab, S_vv_ab, t2_1_ab, optimize=True)
        S2 += 2 * lib.einsum('i,aij,kb,ca,kjcb', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_ab, optimize=True)
        # block IjdJ & IblJ
        S2 -= 2 * lib.einsum('i,aij,jk,lb,lkab', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_ab, optimize=True)
        S2 += 2 * lib.einsum('i,aij,kl,kb,jlab', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('i,ajk,jl,ib,klab', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('i,aji,kl,kb,jlab', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_b, optimize=True)
        S2 += 2 * lib.einsum('i,ajk,kl,ib,jlab', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_b, optimize=True)
        S2 += 2 * lib.einsum('i,aji,jk,lb,lkab', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('i,aji,kl,kb,jlab', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('i,ajk,ji,lb,lkab', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_ab, optimize=True)
        S2 += 2 * lib.einsum('i,ajk,li,lb,jkab', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_ab, optimize=True)
        S2 += 2 * lib.einsum('i,aij,kl,kb,jlab', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b, optimize=True)
        S2 += lib.einsum('i,ajk,li,lb,jkab', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b, optimize=True)
        # block IjlB & AjlJ
        S2 += 2 * lib.einsum('i,aij,kl,bl,jkab', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a, optimize=True)
        S2 += lib.einsum('i,ajk,il,bl,jkab', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a, optimize=True)
        S2 += 2 * lib.einsum('i,aji,kj,bl,klba', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('i,aji,kl,bl,kjba', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('i,ajk,ij,bl,klba', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_ab, optimize=True)
        S2 += 2 * lib.einsum('i,ajk,il,bl,kjba', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('i,aji,kl,bl,jkab', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_a, optimize=True)
        S2 += 2 * lib.einsum('i,ajk,lk,bi,jlab', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_a, optimize=True)
        S2 -= 2 * lib.einsum('i,aij,kj,bl,klba', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab, optimize=True)
        S2 += 2 * lib.einsum('i,aij,kl,bl,kjba', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab, optimize=True)
        S2 -= 2 * lib.einsum('i,ajk,lj,bi,lkba', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab, optimize=True)
        if method in ("adc(2)-x", "adc(3)"):
            # 111
            if t1_1_a is not None:
                # block IjlB & AjlJ
                S2 -= lib.einsum('aij,aij,kl,bl,kb', L_aaa_u, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('aij,aik,jl,bl,kb', L_aaa_u, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 += lib.einsum('aij,bij,kl,al,kb', L_aaa_u, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 -= 2 * lib.einsum('aij,bki,jk,al,lb', L_aaa_u, R_bab, S_oo_ab, S_vo_ab, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('aij,aij,kl,bl,kb', L_aba, R_aba, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('aij,aik,lk,bj,lb', L_aba, R_aba, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('aij,akj,il,bl,kb', L_aba, R_aba, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 -= 2 * lib.einsum('aij,akl,il,bj,kb', L_aba, R_aba, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('aij,bij,kl,al,kb', L_aba, R_aba, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 -= 2 * lib.einsum('aij,bik,lk,aj,lb', L_aba, R_aba, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 -= 2 * lib.einsum('aij,bjk,ik,al,lb', L_aba, R_bbb_u, S_oo_ab, S_vo_ab, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('aij,aij,kl,bl,kb', L_bab, R_bab, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('aij,aik,jl,bl,kb', L_bab, R_bab, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('aij,akj,lk,bi,lb', L_bab, R_bab, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 -= 2 * lib.einsum('aij,akl,jk,bi,lb', L_bab, R_bab, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 -= lib.einsum('aij,aij,kl,bl,kb', L_bbb_u, R_bbb_u, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('aij,aik,lk,bj,lb', L_bbb_u, R_bbb_u, S_oo_ab, S_vo_ab, t1_1_a, optimize=True)
                # block IjdJ & IblJ
                S2 -= lib.einsum('aij,aij,kl,kb,lb', L_aaa_u, R_aaa_u, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('aij,aik,kl,jb,lb', L_aaa_u, R_aaa_u, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('aij,aij,kl,kb,lb', L_aba, R_aba, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('aij,aik,lj,lb,kb', L_aba, R_aba, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('aij,akj,kl,ib,lb', L_aba, R_aba, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('aij,akl,kj,ib,lb', L_aba, R_aba, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('aij,bjk,ki,la,lb', L_bab, R_aaa_u, S_oo_ab, S_ov_ab, t1_1_a, optimize=True)
                S2 -= 2 * lib.einsum('aij,aij,kl,kb,lb', L_bab, R_bab, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('aij,aik,kl,jb,lb', L_bab, R_bab, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('aij,akj,li,lb,kb', L_bab, R_bab, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('aij,akl,li,jb,kb', L_bab, R_bab, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('aij,bij,kl,ka,lb', L_bab, R_bab, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('aij,bik,kl,ja,lb', L_bab, R_bab, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('aij,bki,kj,la,lb', L_bbb_u, R_aba, S_oo_ab, S_ov_ab, t1_1_a, optimize=True)
                S2 -= lib.einsum('aij,aij,kl,kb,lb', L_bbb_u, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('aij,aik,lj,lb,kb', L_bbb_u, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                S2 += lib.einsum('aij,bij,kl,ka,lb', L_bbb_u, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_b, optimize=True)
                # block IbdB & AbdJ
                S2 += 2 * lib.einsum('aij,bki,jc,ab,kc', L_aaa_u, R_bab, S_ov_ab, S_vv_ab, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('aij,bjk,ic,ab,kc', L_aba, R_bbb_u, S_ov_ab, S_vv_ab, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('aij,bij,ka,cb,kc', L_bab, R_bab, S_ov_ab, S_vv_ab, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('aij,bik,ja,cb,kc', L_bab, R_bab, S_ov_ab, S_vv_ab, t1_1_a, optimize=True)
                S2 -= lib.einsum('aij,bij,ka,cb,kc', L_bbb_u, R_bbb_u, S_ov_ab, S_vv_ab, t1_1_a, optimize=True)
                # block AjdB & AblB
                S2 -= lib.einsum('aij,bij,ak,bc,kc', L_aaa_u, R_aaa_u, S_vo_ab, S_vv_ab, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('aij,bij,ak,bc,kc', L_aba, R_aba, S_vo_ab, S_vv_ab, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('aij,bik,aj,bc,kc', L_aba, R_aba, S_vo_ab, S_vv_ab, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('aij,bjk,ci,ba,kc', L_bab, R_aaa_u, S_vo_ab, S_vv_ab, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('aij,bki,cj,ba,kc', L_bbb_u, R_aba, S_vo_ab, S_vv_ab, t1_1_a, optimize=True)

            # block IjdB & AblJ
            S2 -= lib.einsum('aij,aij,kb,cl,klcb', L_aaa_u, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('aij,aik,jb,cl,klcb', L_aaa_u, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('aij,bij,kc,al,klbc', L_aaa_u, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('aij,bik,jc,al,klbc', L_aaa_u, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('aij,bki,jc,al,klbc', L_aaa_u, R_bab, S_ov_ab, S_vo_ab, t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('aij,aij,kb,cl,klcb', L_aba, R_aba, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('aij,aik,lb,cj,lkcb', L_aba, R_aba, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('aij,akj,ib,cl,klcb', L_aba, R_aba, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('aij,akl,ib,cj,klcb', L_aba, R_aba, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('aij,bij,kc,al,klbc', L_aba, R_aba, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('aij,bik,lc,aj,lkbc', L_aba, R_aba, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('aij,bkj,ic,al,klbc', L_aba, R_aba, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('aij,bkl,ic,aj,klbc', L_aba, R_aba, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('aij,bjk,ic,al,klbc', L_aba, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_b, optimize=True)
            S2 += lib.einsum('aij,bkl,ic,aj,klbc', L_aba, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_b, optimize=True)
            S2 += 2 * lib.einsum('aij,bjk,la,ci,klbc', L_bab, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_a, optimize=True)
            S2 += lib.einsum('aij,bkl,ja,ci,klbc', L_bab, R_aaa_u, S_ov_ab, S_vo_ab, t2_1_a, optimize=True)
            S2 -= 2 * lib.einsum('aij,aij,kb,cl,klcb', L_bab, R_bab, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('aij,aik,jb,cl,klcb', L_bab, R_bab, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('aij,akj,lb,ci,lkcb', L_bab, R_bab, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('aij,akl,jb,ci,lkcb', L_bab, R_bab, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('aij,bij,ka,cl,klcb', L_bab, R_bab, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('aij,bik,ja,cl,klcb', L_bab, R_bab, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('aij,bkj,la,ci,lkcb', L_bab, R_bab, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('aij,bkl,ja,ci,lkcb', L_bab, R_bab, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('aij,bki,la,cj,klbc', L_bbb_u, R_aba, S_ov_ab, S_vo_ab, t2_1_a, optimize=True)
            S2 -= lib.einsum('aij,aij,kb,cl,klcb', L_bbb_u, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('aij,aik,lb,cj,lkcb', L_bbb_u, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('aij,bij,ka,cl,klcb', L_bbb_u, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('aij,bik,la,cj,lkcb', L_bbb_u, R_bbb_u, S_ov_ab, S_vo_ab, t2_1_ab, optimize=True)

        if method == "adc(3)":
            # 030
            # block AbdB
            S2 -= 2 * lib.einsum('i,i,jkab,jkcd,ad,cb', L_a, R_a, t2_1_ab, t2_2_ab, S_vv_ab, S_vv_ab, optimize=True)
            S2 += lib.einsum('i,j,ikab,jkcd,ad,cb', L_a, R_a, t2_1_ab, t2_2_ab, S_vv_ab, S_vv_ab, optimize=True)
            S2 += lib.einsum('i,j,jkab,ikcd,ad,cb', L_a, R_a, t2_1_ab, t2_2_ab, S_vv_ab, S_vv_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,i,jkab,jkcd,ad,cb', L_b, R_b, t2_1_ab, t2_2_ab, S_vv_ab, S_vv_ab, optimize=True)
            S2 += lib.einsum('i,j,kiab,kjcd,ad,cb', L_b, R_b, t2_1_ab, t2_2_ab, S_vv_ab, S_vv_ab, optimize=True)
            S2 += lib.einsum('i,j,kjab,kicd,ad,cb', L_b, R_b, t2_1_ab, t2_2_ab, S_vv_ab, S_vv_ab, optimize=True)
            # block AjlB
            S2 -= lib.einsum('i,i,jkab,jkac,bl,cl', L_a, R_a, t2_1_a, t2_2_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 += lib.einsum('i,j,ikab,jkac,bl,cl', L_a, R_a, t2_1_a, t2_2_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 += lib.einsum('i,j,jkab,ikac,bl,cl', L_a, R_a, t2_1_a, t2_2_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,i,jkab,jkcb,al,cl', L_a, R_a, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 += 2 * lib.einsum('i,i,jkab,jlcb,ak,cl', L_a, R_a, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 += lib.einsum('i,j,ikab,jkcb,al,cl', L_a, R_a, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= lib.einsum('i,j,ikab,jlcb,ak,cl', L_a, R_a, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 += lib.einsum('i,j,jkab,ikcb,al,cl', L_a, R_a, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= lib.einsum('i,j,jkab,ilcb,ak,cl', L_a, R_a, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= lib.einsum('i,i,jkab,jkac,bl,cl', L_b, R_b, t2_1_a, t2_2_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('i,j,klab,klac,bi,cj', L_b, R_b, t2_1_a, t2_2_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('i,j,klab,klac,ci,bj', L_b, R_b, t2_1_a, t2_2_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,i,jkab,jkcb,al,cl', L_b, R_b, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 += 2 * lib.einsum('i,i,jkab,jlcb,ak,cl', L_b, R_b, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 += lib.einsum('i,j,kiab,kjcb,al,cl', L_b, R_b, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= lib.einsum('i,j,kiab,klcb,aj,cl', L_b, R_b, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 += lib.einsum('i,j,kjab,kicb,al,cl', L_b, R_b, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= lib.einsum('i,j,kjab,klcb,ai,cl', L_b, R_b, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= lib.einsum('i,j,klab,kicb,cj,al', L_b, R_b, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= lib.einsum('i,j,klab,kjcb,ci,al', L_b, R_b, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 += lib.einsum('i,j,klab,klcb,ai,cj', L_b, R_b, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            S2 += lib.einsum('i,j,klab,klcb,ci,aj', L_b, R_b, t2_1_ab, t2_2_ab, S_vo_ab, S_vo_ab, optimize=True)
            # block IbdJ
            S2 -= 2 * lib.einsum('i,i,jkab,jkac,lb,lc', L_a, R_a, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 += 2 * lib.einsum('i,i,jkab,lkac,jb,lc', L_a, R_a, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 += lib.einsum('i,j,ikab,jkac,lb,lc', L_a, R_a, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= lib.einsum('i,j,ikab,lkac,jb,lc', L_a, R_a, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 += lib.einsum('i,j,jkab,ikac,lb,lc', L_a, R_a, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= lib.einsum('i,j,jkab,lkac,ib,lc', L_a, R_a, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= lib.einsum('i,j,klab,ilac,jc,kb', L_a, R_a, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= lib.einsum('i,j,klab,jlac,ic,kb', L_a, R_a, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 += lib.einsum('i,j,klab,klac,ib,jc', L_a, R_a, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 += lib.einsum('i,j,klab,klac,ic,jb', L_a, R_a, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= lib.einsum('i,i,jkab,jkac,lb,lc', L_a, R_a, t2_1_b, t2_2_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('i,j,klab,klac,ib,jc', L_a, R_a, t2_1_b, t2_2_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('i,j,klab,klac,ic,jb', L_a, R_a, t2_1_b, t2_2_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,i,jkab,jkac,lb,lc', L_b, R_b, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 += 2 * lib.einsum('i,i,jkab,lkac,jb,lc', L_b, R_b, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 += lib.einsum('i,j,kiab,kjac,lb,lc', L_b, R_b, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= lib.einsum('i,j,kiab,ljac,kb,lc', L_b, R_b, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 += lib.einsum('i,j,kjab,kiac,lb,lc', L_b, R_b, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= lib.einsum('i,j,kjab,liac,kb,lc', L_b, R_b, t2_1_ab, t2_2_ab, S_ov_ab, S_ov_ab, optimize=True)
            S2 -= lib.einsum('i,i,jkab,jkac,lb,lc', L_b, R_b, t2_1_b, t2_2_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 += lib.einsum('i,j,ikab,jkac,lb,lc', L_b, R_b, t2_1_b, t2_2_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 += lib.einsum('i,j,jkab,ikac,lb,lc', L_b, R_b, t2_1_b, t2_2_b, S_ov_ab, S_ov_ab, optimize=True)
            # block IblB & AjdJ
            S2 -= 2 * lib.einsum('i,i,jk,ab,jlac,lkcb', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_a, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('i,j,ik,ab,jlac,lkcb', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_a, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,j,kl,ab,jkac,ilcb', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_a, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,i,jk,ab,lkcb,jlac', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t2_2_a, optimize=True)
            S2 += 2 * lib.einsum('i,j,ik,ab,lkcb,jlac', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t2_2_a, optimize=True)
            S2 -= 2 * lib.einsum('i,j,kl,ab,ilcb,jkac', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t2_2_a, optimize=True)
            S2 -= 2 * lib.einsum('i,i,jk,ab,jlac,klbc', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t2_2_b, optimize=True)
            S2 += 2 * lib.einsum('i,j,ik,ab,jlac,klbc', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t2_2_b, optimize=True)
            S2 -= 2 * lib.einsum('i,i,jk,ab,klbc,jlac', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_b, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('i,j,ik,ab,klbc,jlac', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_b, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,i,jk,ab,jlac,lkcb', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_a, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('i,j,kj,ab,klac,licb', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_a, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,i,jk,ab,lkcb,jlac', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t2_2_a, optimize=True)
            S2 += 2 * lib.einsum('i,j,kj,ab,licb,klac', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t2_2_a, optimize=True)
            S2 -= 2 * lib.einsum('i,i,jk,ab,jlac,klbc', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t2_2_b, optimize=True)
            S2 += 2 * lib.einsum('i,j,kj,ab,klac,ilbc', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t2_2_b, optimize=True)
            S2 -= 2 * lib.einsum('i,j,kl,ab,kjac,ilbc', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t2_2_b, optimize=True)
            S2 -= 2 * lib.einsum('i,i,jk,ab,klbc,jlac', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_b, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('i,j,kj,ab,ilbc,klac', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_b, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,j,kl,ab,ilbc,kjac', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_b, t2_2_ab, optimize=True)
            # block IjdB & AblJ
            S2 -= 4 / 3 * lib.einsum('i,i,ja,bk,jlbc,lmcd,kmad', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_a, t2_1_ab,
                t2_1_b, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ia,bk,jlbc,lmcd,kmad', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_a, t2_1_ab,
                t2_1_b, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,j,ka,bl,jkbc,imcd,lmad', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_a, t2_1_ab,
                t2_1_b, optimize=True)
            S2 += 2 / 3 * lib.einsum('i,i,ja,bk,jkca,bc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, w_t2t2_bc, optimize=True)
            S2 += 2 / 3 * lib.einsum('i,i,ja,bk,lkba,jmcd,lmcd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,i,ja,bk,lkca,jmbd,lmcd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 2 / 3 * lib.einsum('i,j,ia,bk,jkca,bc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, w_t2t2_bc, optimize=True)
            S2 -= 1 / 6 * lib.einsum('i,j,ia,bk,lkba,jmcd,lmcd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ia,bk,lkca,jmbd,lmcd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,j,ka,bl,jlba,imcd,kmcd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ka,bl,jlca,imcd,kmbd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,j,ka,bl,klca,imcd,jmbd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 2 / 3 * lib.einsum('i,j,ka,bl,mlba,imcd,jkcd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ka,bl,mlca,imcd,jkbd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,i,ja,bk,jkbc,ac', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, w_t2t2_ac, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,i,ja,bk,jkca,bc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab,
                w_t2t2_bc_2, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,i,ja,bk,jkcd,lmcd,lmba', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,i,ja,bk,jlba,mkcd,mlcd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,i,ja,bk,jlbc,mldc,mkda', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,i,ja,bk,jlca,mlcd,mkbd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,i,ja,bk,jlcd,mkba,mlcd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,j,ia,bk,jkbc,ac', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, w_t2t2_ac, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,j,ia,bk,jkca,bc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab,
                w_t2t2_bc_2, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ia,bk,jkcd,lmcd,lmba', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,j,ia,bk,jlba,mkcd,mlcd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ia,bk,jlbc,mldc,mkda', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ia,bk,jlca,mlcd,mkbd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 1 / 3 * lib.einsum('i,j,ia,bk,jlcd,mkba,mlcd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,j,ka,bl,imcd,jlba,kmcd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ka,bl,imcd,jlbd,kmca', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ka,bl,imcd,jlca,kmbd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,j,ka,bl,imcd,jlcd,kmba', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ka,bl,imcd,jmba,klcd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,j,ka,bl,imcd,jmbd,klca', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,j,ka,bl,imcd,jmca,klbd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 2 / 3 * lib.einsum('i,i,ja,bk,jkbc,ac', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab,
                w_t2t2_ac_2, optimize=True)
            S2 += 2 / 3 * lib.einsum('i,i,ja,bk,jlba,kmcd,lmcd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,i,ja,bk,jlbc,lmcd,kmad', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 2 / 3 * lib.einsum('i,j,ia,bk,jkbc,ac', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab,
                w_t2t2_ac_2, optimize=True)
            S2 -= 2 / 3 * lib.einsum('i,j,ia,bk,jlba,kmcd,lmcd', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ia,bk,jlbc,lmcd,kmad', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,i,ja,bk,jlbc,lmcd,kmad', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_a, t2_1_ab,
                t2_1_b, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ka,bi,klbc,lmcd,jmad', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_a, t2_1_ab,
                t2_1_b, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,j,ka,bl,kmbc,micd,jlad', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_a, t2_1_ab,
                t2_1_b, optimize=True)
            S2 += 2 / 3 * lib.einsum('i,i,ja,bk,jkca,bc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, w_t2t2_bc, optimize=True)
            S2 += 2 / 3 * lib.einsum('i,i,ja,bk,lkba,jmcd,lmcd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,i,ja,bk,lkca,jmbd,lmcd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 2 / 3 * lib.einsum('i,j,ka,bi,kjca,bc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, w_t2t2_bc, optimize=True)
            S2 -= 2 / 3 * lib.einsum('i,j,ka,bi,ljba,kmcd,lmcd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ka,bi,ljca,kmbd,lmcd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,i,ja,bk,jkbc,ac', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, w_t2t2_ac, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,i,ja,bk,jkca,bc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab,
                w_t2t2_bc_2, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,i,ja,bk,jkcd,lmcd,lmba', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,i,ja,bk,jlba,mkcd,mlcd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,i,ja,bk,jlbc,mldc,mkda', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,i,ja,bk,jlca,mlcd,mkbd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,i,ja,bk,jlcd,mkba,mlcd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,j,ka,bi,kjbc,ac', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, w_t2t2_ac, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,j,ka,bi,kjca,bc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab,
                w_t2t2_bc_2, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ka,bi,kjcd,lmba,lmcd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,j,ka,bi,ljba,kmcd,lmcd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ka,bi,ljbc,kmda,lmdc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ka,bi,ljca,kmbd,lmcd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 1 / 3 * lib.einsum('i,j,ka,bi,ljcd,kmba,lmcd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,j,ka,bl,micd,kjba,mlcd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ka,bl,micd,kjbd,mlca', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ka,bl,micd,kjca,mlbd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,j,ka,bl,micd,kjcd,mlba', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ka,bl,micd,mjba,klcd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,j,ka,bl,micd,mjbd,klca', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,j,ka,bl,micd,mjca,klbd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 2 / 3 * lib.einsum('i,i,ja,bk,jkbc,ac', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab,
                w_t2t2_ac_2, optimize=True)
            S2 += 2 / 3 * lib.einsum('i,i,ja,bk,jlba,kmcd,lmcd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,i,ja,bk,jlbc,lmcd,kmad', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 2 / 3 * lib.einsum('i,j,ka,bi,kjbc,ac', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab,
                w_t2t2_ac_2, optimize=True)
            S2 -= 1 / 6 * lib.einsum('i,j,ka,bi,klba,jmcd,lmcd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ka,bi,klbc,jmad,lmcd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,j,ka,bl,kjba,imcd,lmcd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ka,bl,kjbc,imcd,lmad', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 4 / 3 * lib.einsum('i,j,ka,bl,klbc,imcd,jmad', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 2 / 3 * lib.einsum('i,j,ka,bl,kmba,imcd,jlcd', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += 4 / 3 * lib.einsum('i,j,ka,bl,kmbc,imcd,jlad', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t2_1_b,
                t2_1_b, optimize=True)
            # block IjlB & AjlJ
            S2 -= lib.einsum('i,i,jk,ak,lb,jlab', L_a, R_a, S_oo_ab, S_vo_ab, t1_2_a, t2_1_a, optimize=True)
            S2 += lib.einsum('i,j,ik,ak,lb,jlab', L_a, R_a, S_oo_ab, S_vo_ab, t1_2_a, t2_1_a, optimize=True)
            S2 -= lib.einsum('i,j,kl,al,ib,jkab', L_a, R_a, S_oo_ab, S_vo_ab, t1_2_a, t2_1_a, optimize=True)
            S2 -= lib.einsum('i,i,jk,ak,lb,jlab', L_a, R_a, S_oo_ab, S_vo_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,i,jk,al,kb,jlab', L_a, R_a, S_oo_ab, S_vo_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,j,ik,ak,lb,jlab', L_a, R_a, S_oo_ab, S_vo_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,j,ik,al,kb,jlab', L_a, R_a, S_oo_ab, S_vo_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,i,jk,ak,lb,jlab', L_b, R_b, S_oo_ab, S_vo_ab, t1_2_a, t2_1_a, optimize=True)
            S2 += lib.einsum('i,j,kj,ai,lb,klab', L_b, R_b, S_oo_ab, S_vo_ab, t1_2_a, t2_1_a, optimize=True)
            S2 -= lib.einsum('i,i,jk,ak,lb,jlab', L_b, R_b, S_oo_ab, S_vo_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,i,jk,al,kb,jlab', L_b, R_b, S_oo_ab, S_vo_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,j,kj,ai,lb,klab', L_b, R_b, S_oo_ab, S_vo_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,j,kj,al,ib,klab', L_b, R_b, S_oo_ab, S_vo_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,j,kl,ai,lb,kjab', L_b, R_b, S_oo_ab, S_vo_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,j,kl,al,ib,kjab', L_b, R_b, S_oo_ab, S_vo_ab, t1_2_b, t2_1_ab, optimize=True)
            # block IjdJ & IblJ
            S2 -= lib.einsum('i,i,jk,ja,lb,lkba', L_a, R_a, S_oo_ab, S_ov_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,i,jk,la,jb,lkba', L_a, R_a, S_oo_ab, S_ov_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,j,jk,ia,lb,lkba', L_a, R_a, S_oo_ab, S_ov_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,j,jk,la,ib,lkba', L_a, R_a, S_oo_ab, S_ov_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,j,kl,ia,kb,jlba', L_a, R_a, S_oo_ab, S_ov_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,j,kl,ka,ib,jlba', L_a, R_a, S_oo_ab, S_ov_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,i,jk,ja,lb,klab', L_a, R_a, S_oo_ab, S_ov_ab, t1_2_b, t2_1_b, optimize=True)
            S2 += lib.einsum('i,j,jk,ia,lb,klab', L_a, R_a, S_oo_ab, S_ov_ab, t1_2_b, t2_1_b, optimize=True)
            S2 -= lib.einsum('i,i,jk,ja,lb,lkba', L_b, R_b, S_oo_ab, S_ov_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,i,jk,la,jb,lkba', L_b, R_b, S_oo_ab, S_ov_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,j,ki,ka,lb,ljba', L_b, R_b, S_oo_ab, S_ov_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,j,ki,la,kb,ljba', L_b, R_b, S_oo_ab, S_ov_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,i,jk,ja,lb,klab', L_b, R_b, S_oo_ab, S_ov_ab, t1_2_b, t2_1_b, optimize=True)
            S2 += lib.einsum('i,j,ki,ka,lb,jlab', L_b, R_b, S_oo_ab, S_ov_ab, t1_2_b, t2_1_b, optimize=True)
            S2 -= lib.einsum('i,j,kl,ka,ib,jlab', L_b, R_b, S_oo_ab, S_ov_ab, t1_2_b, t2_1_b, optimize=True)
            # block IbdB & AbdJ
            S2 -= 2 * lib.einsum('i,i,ja,bc,kc,jkba', L_a, R_a, S_ov_ab, S_vv_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,j,ia,bc,kc,jkba', L_a, R_a, S_ov_ab, S_vv_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,i,ja,bc,kc,jkba', L_b, R_b, S_ov_ab, S_vv_ab, t1_2_b, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,j,ka,bc,ic,kjba', L_b, R_b, S_ov_ab, S_vv_ab, t1_2_b, t2_1_ab, optimize=True)
            # block AjdB & AblB
            S2 -= 2 * lib.einsum('i,i,aj,bc,kb,kjac', L_a, R_a, S_vo_ab, S_vv_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,j,ak,bc,ib,jkac', L_a, R_a, S_vo_ab, S_vv_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,i,aj,bc,kb,kjac', L_b, R_b, S_vo_ab, S_vv_ab, t1_2_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,j,ai,bc,kb,kjac', L_b, R_b, S_vo_ab, S_vv_ab, t1_2_a, t2_1_ab, optimize=True)
            # block IjlJ
            S2 += lib.einsum('i,i,jkab,jlab,km,lm', L_a, R_a, t2_1_a, t2_2_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,j,ikab,jlab,km,lm', L_a, R_a, t2_1_a, t2_2_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 1 / 4 * lib.einsum('i,j,ikab,klab,jm,lm', L_a, R_a, t2_1_a, t2_2_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,j,jkab,ilab,km,lm', L_a, R_a, t2_1_a, t2_2_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 1 / 4 * lib.einsum('i,j,jkab,klab,im,lm', L_a, R_a, t2_1_a, t2_2_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 1 / 4 * lib.einsum('i,j,klab,ikab,jm,lm', L_a, R_a, t2_1_a, t2_2_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 1 / 4 * lib.einsum('i,j,klab,jkab,im,lm', L_a, R_a, t2_1_a, t2_2_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 2 * lib.einsum('i,i,jkab,jlab,mk,ml', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 2 * lib.einsum('i,i,jkab,lkab,jm,lm', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,i,jkab,lmab,jm,lk', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('i,j,ikab,jlab,mk,ml', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,j,ikab,lkab,jm,lm', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('i,j,ikab,lmab,jm,lk', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('i,j,jkab,ilab,mk,ml', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,j,jkab,lkab,im,lm', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('i,j,jkab,lmab,im,lk', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,j,klab,ilab,jm,km', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('i,j,klab,imab,jl,km', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,j,klab,jlab,im,km', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('i,j,klab,jmab,il,km', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('i,j,klab,kmab,il,jm', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('i,j,klab,kmab,im,jl', L_a, R_a, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('i,i,jkab,jlab,mk,ml', L_a, R_a, t2_1_b, t2_2_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,j,klab,kmab,il,jm', L_a, R_a, t2_1_b, t2_2_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,j,klab,kmab,im,jl', L_a, R_a, t2_1_b, t2_2_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('i,i,jkab,jlab,km,lm', L_b, R_b, t2_1_a, t2_2_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,j,klab,kmab,li,mj', L_b, R_b, t2_1_a, t2_2_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,j,klab,kmab,mi,lj', L_b, R_b, t2_1_a, t2_2_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 2 * lib.einsum('i,i,jkab,jlab,mk,ml', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 2 * lib.einsum('i,i,jkab,lkab,jm,lm', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,i,jkab,lmab,jm,lk', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,j,kiab,klab,mj,ml', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('i,j,kiab,ljab,km,lm', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('i,j,kiab,lmab,lj,km', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,j,kjab,klab,mi,ml', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('i,j,kjab,liab,km,lm', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('i,j,kjab,lmab,li,km', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,j,klab,kiab,mj,ml', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,j,klab,kjab,mi,ml', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('i,j,klab,miab,kj,ml', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('i,j,klab,mjab,ki,ml', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('i,j,klab,mlab,ki,mj', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= lib.einsum('i,j,klab,mlab,mi,kj', L_b, R_b, t2_1_ab, t2_2_ab, S_oo_ab, S_oo_ab, optimize=True)
            S2 += lib.einsum('i,i,jkab,jlab,mk,ml', L_b, R_b, t2_1_b, t2_2_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,j,ikab,jlab,mk,ml', L_b, R_b, t2_1_b, t2_2_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 1 / 4 * lib.einsum('i,j,ikab,klab,mj,ml', L_b, R_b, t2_1_b, t2_2_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,j,jkab,ilab,mk,ml', L_b, R_b, t2_1_b, t2_2_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 1 / 4 * lib.einsum('i,j,jkab,klab,mi,ml', L_b, R_b, t2_1_b, t2_2_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 1 / 4 * lib.einsum('i,j,klab,ikab,mj,ml', L_b, R_b, t2_1_b, t2_2_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 1 / 4 * lib.einsum('i,j,klab,jkab,mi,ml', L_b, R_b, t2_1_b, t2_2_b, S_oo_ab, S_oo_ab, optimize=True)

            if adc.approx_trans_moments is False:
                # block IjlB & AjlJ
                S2 -= 2 * lib.einsum('i,i,jk,ak,ja', L_a, R_a, S_oo_ab, S_vo_ab, t1_3_a, optimize=True)
                S2 += 2 * lib.einsum('i,j,ik,ak,ja', L_a, R_a, S_oo_ab, S_vo_ab, t1_3_a, optimize=True)
                S2 -= 2 * lib.einsum('i,i,jk,ak,ja', L_b, R_b, S_oo_ab, S_vo_ab, t1_3_a, optimize=True)
                S2 += 2 * lib.einsum('i,j,kj,ai,ka', L_b, R_b, S_oo_ab, S_vo_ab, t1_3_a, optimize=True)
                # block IjdJ & IblJ
                S2 -= 2 * lib.einsum('i,i,jk,ja,ka', L_a, R_a, S_oo_ab, S_ov_ab, t1_3_b, optimize=True)
                S2 += 2 * lib.einsum('i,j,jk,ia,ka', L_a, R_a, S_oo_ab, S_ov_ab, t1_3_b, optimize=True)
                S2 -= 2 * lib.einsum('i,i,jk,ja,ka', L_b, R_b, S_oo_ab, S_ov_ab, t1_3_b, optimize=True)
                S2 += 2 * lib.einsum('i,j,ki,ka,ja', L_b, R_b, S_oo_ab, S_ov_ab, t1_3_b, optimize=True)

            if t1_1_a is not None:
                # block IjlJ
                S2 += 2 * lib.einsum('i,i,ja,ka,jl,kl', L_a, R_a, t1_1_a, t1_2_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ia,ka,jl,kl', L_a, R_a, t1_1_a, t1_2_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ja,ka,il,kl', L_a, R_a, t1_1_a, t1_2_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ka,ia,jl,kl', L_a, R_a, t1_1_a, t1_2_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ka,ja,il,kl', L_a, R_a, t1_1_a, t1_2_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 += 2 * lib.einsum('i,i,ja,ka,lj,lk', L_a, R_a, t1_1_b, t1_2_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('i,j,ka,la,ik,jl', L_a, R_a, t1_1_b, t1_2_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('i,j,ka,la,il,jk', L_a, R_a, t1_1_b, t1_2_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 += 2 * lib.einsum('i,i,ja,ka,jl,kl', L_b, R_b, t1_1_a, t1_2_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('i,j,ka,la,ki,lj', L_b, R_b, t1_1_a, t1_2_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('i,j,ka,la,li,kj', L_b, R_b, t1_1_a, t1_2_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 += 2 * lib.einsum('i,i,ja,ka,lj,lk', L_b, R_b, t1_1_b, t1_2_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ia,ka,lj,lk', L_b, R_b, t1_1_b, t1_2_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ja,ka,li,lk', L_b, R_b, t1_1_b, t1_2_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ka,ia,lj,lk', L_b, R_b, t1_1_b, t1_2_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ka,ja,li,lk', L_b, R_b, t1_1_b, t1_2_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 += lib.einsum('i,i,ja,kb,jlab,mk,ml', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('i,i,ja,kb,lkab,jm,lm', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,ja,kb,lmab,jm,lk', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ia,kb,jlab,mk,ml', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,ia,kb,lkab,jm,lm', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,ia,kb,lmab,jm,lk', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ja,kb,ilab,mk,ml', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,ja,kb,lkab,im,lm', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,ja,kb,lmab,im,lk', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/6 * lib.einsum('i,j,ka,lb,ilab,jm,km', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('i,j,ka,lb,imab,jl,km', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/6 * lib.einsum('i,j,ka,lb,jlab,im,km', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('i,j,ka,lb,jmab,il,km', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ka,lb,kmab,il,jm', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ka,lb,kmab,im,jl', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= lib.einsum('i,i,jkab,jl,ml,ka,mb', L_a, R_a, t2_1_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 1/6 * lib.einsum('i,j,ikab,jl,ml,ka,mb', L_a, R_a, t2_1_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ikab,kl,ml,ja,mb', L_a, R_a, t2_1_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 1/6 * lib.einsum('i,j,jkab,il,ml,ka,mb', L_a, R_a, t2_1_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,jkab,kl,ml,ia,mb', L_a, R_a, t2_1_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,klab,im,km,ja,lb', L_a, R_a, t2_1_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,klab,jm,km,ia,lb', L_a, R_a, t2_1_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,i,jkab,lj,lm,ka,mb', L_a, R_a, t2_1_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,klab,ik,jm,la,mb', L_a, R_a, t2_1_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,klab,im,jk,la,mb', L_a, R_a, t2_1_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += lib.einsum('i,i,ja,kb,jlab,mk,ml', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('i,i,ja,kb,lkab,jm,lm', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,ja,kb,lmab,jm,lk', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,ka,ib,klab,mj,ml', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ka,ib,ljab,km,lm', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,ka,ib,lmab,lj,km', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,ka,jb,klab,mi,ml', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ka,jb,liab,km,lm', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,ka,jb,lmab,li,km', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/6 * lib.einsum('i,j,ka,lb,kiab,mj,ml', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/6 * lib.einsum('i,j,ka,lb,kjab,mi,ml', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('i,j,ka,lb,miab,kj,ml', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('i,j,ka,lb,mjab,ki,ml', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ka,lb,mlab,ki,mj', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ka,lb,mlab,mi,kj', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= lib.einsum('i,i,jkab,jl,ml,ka,mb', L_b, R_b, t2_1_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,klab,ki,mj,la,mb', L_b, R_b, t2_1_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,klab,mi,kj,la,mb', L_b, R_b, t2_1_a, S_oo_ab, S_oo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,i,jkab,lj,lm,ka,mb', L_b, R_b, t2_1_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 1/6 * lib.einsum('i,j,ikab,lj,lm,ka,mb', L_b, R_b, t2_1_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ikab,lk,lm,ja,mb', L_b, R_b, t2_1_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 1/6 * lib.einsum('i,j,jkab,li,lm,ka,mb', L_b, R_b, t2_1_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,jkab,lk,lm,ia,mb', L_b, R_b, t2_1_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,klab,mi,mk,ja,lb', L_b, R_b, t2_1_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,klab,mj,mk,ia,lb', L_b, R_b, t2_1_b, S_oo_ab, S_oo_ab, t1_1_b, t1_1_b,
                    optimize=True)
                # block IjlB & AjlJ
                S2 -= lib.einsum('i,i,jk,ak,lb,jlab', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_2_a, optimize=True)
                S2 += lib.einsum('i,j,ik,ak,lb,jlab', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_2_a, optimize=True)
                S2 -= lib.einsum('i,j,kl,al,ib,jkab', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_2_a, optimize=True)
                S2 -= lib.einsum('i,i,jk,ak,lb,jlab', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,al,kb,jlab', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 += lib.einsum('i,j,ik,ak,lb,jlab', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,j,ik,al,kb,jlab', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 -= lib.einsum('i,i,jk,ak,lb,jlab', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_2_a, optimize=True)
                S2 += lib.einsum('i,j,kj,ai,lb,klab', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_2_a, optimize=True)
                S2 -= lib.einsum('i,i,jk,ak,lb,jlab', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,al,kb,jlab', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 += lib.einsum('i,j,kj,ai,lb,klab', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 -= lib.einsum('i,j,kj,al,ib,klab', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,j,kl,ai,lb,kjab', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 += lib.einsum('i,j,kl,al,ib,kjab', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 += 4/3 * lib.einsum('i,i,jk,ak,jb,la,lb', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,ik,ak,jb,la,lb', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,j,kl,al,ib,ja,kb', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,al,ja,kb,lb', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ik,al,ja,kb,lb', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 2/3 * lib.einsum('i,i,jk,ak,jb,lmac,lmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 2/3 * lib.einsum('i,i,jk,ak,la,jmbc,lmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,i,jk,ak,lb,jmac,lmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 1/6 * lib.einsum('i,j,ik,ak,jb,lmac,lmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 1/6 * lib.einsum('i,j,ik,ak,la,jmbc,lmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,ik,ak,lb,jmac,lmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,kl,al,ja,imbc,kmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,kl,al,jb,imbc,kmac', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,kl,al,kb,imbc,jmac', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 2/3 * lib.einsum('i,j,kl,al,ma,imbc,jkbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,kl,al,mb,imbc,jkac', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,i,jk,ak,jb,lmac,lmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,i,jk,ak,la,jmbc,lmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,i,jk,ak,lb,jmac,lmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,al,ja,mkbc,mlbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,jk,al,jb,mkbc,mlac', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,jk,al,ma,jlbc,mkbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,i,jk,al,mb,jlac,mkbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,ik,ak,jb,lmac,lmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,ik,ak,la,jmbc,lmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,ik,ak,lb,jmac,lmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ik,al,ja,mkbc,mlbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,j,ik,al,jb,mkbc,mlac', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ik,al,ma,jlbc,mkbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,ik,al,mb,jlac,mkbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,kl,al,ja,imbc,kmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,kl,al,jb,imbc,kmac', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,kl,al,kb,imbc,jmac', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,kl,am,ja,ilbc,kmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,kl,am,jb,ilbc,kmac', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,kl,am,ka,ilbc,jmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,kl,am,kb,ilbc,jmac', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,i,jk,al,ja,kmbc,lmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,j,ik,al,ja,kmbc,lmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_a, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,i,jk,ak,lb,jmac,mlcb', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,al,lb,jmac,mkcb', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,ik,ak,lb,jmac,mlcb', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ik,al,lb,jmac,mkcb', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,kl,al,mb,jkac,imcb', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,kl,am,mb,jkac,ilcb', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,i,jk,ak,lb,jmac,lmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,al,lb,jmac,kmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,i,jk,al,mb,jlac,kmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,ik,ak,lb,jmac,lmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ik,al,lb,jmac,kmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('i,j,ik,al,mb,jlac,kmbc', L_a, R_a, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,i,jk,ak,jb,la,lb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,kj,ai,kb,la,lb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,al,ja,kb,lb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,j,kj,al,ka,ib,lb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, t1_1_b, optimize=True)
                S2 -= lib.einsum('i,j,kl,ai,ka,jb,lb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, t1_1_b, optimize=True)
                S2 += 2/3 * lib.einsum('i,i,jk,ak,jb,lmac,lmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 2/3 * lib.einsum('i,i,jk,ak,la,jmbc,lmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,i,jk,ak,lb,jmac,lmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 2/3 * lib.einsum('i,j,kj,ai,kb,lmac,lmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 2/3 * lib.einsum('i,j,kj,ai,la,kmbc,lmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,kj,ai,lb,kmac,lmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,i,jk,ak,jb,lmac,lmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,i,jk,ak,la,jmbc,lmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,i,jk,ak,lb,jmac,lmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,al,ja,mkbc,mlbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,jk,al,jb,mkbc,mlac', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,jk,al,ma,jlbc,mkbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,i,jk,al,mb,jlac,mkbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,kj,ai,kb,lmac,lmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,kj,ai,la,kmbc,lmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,kj,ai,lb,kmac,lmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,kj,al,ka,mibc,mlbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,j,kj,al,kb,mibc,mlac', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,j,kj,al,ma,mibc,klbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,kj,al,mb,mibc,klac', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,kl,ai,ka,mjbc,mlbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,kl,ai,kb,mjac,mlbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,kl,ai,ma,kjbc,mlbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,kl,ai,mb,kjac,mlbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,kl,al,kb,mibc,mjac', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,kl,al,ma,mibc,kjbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,kl,al,mb,mibc,kjac', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,i,jk,al,ja,kmbc,lmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,kj,al,ka,imbc,lmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,kl,ai,ka,jmbc,lmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,j,kl,am,ka,ilbc,jmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_a, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,i,jk,ak,lb,jmac,mlcb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,al,lb,jmac,mkcb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,kj,ai,lb,kmac,mlcb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,kj,al,lb,kmac,micb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,kl,ai,jb,kmac,mlcb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,kl,al,jb,kmac,micb', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,i,jk,ak,lb,jmac,lmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,al,lb,jmac,kmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,i,jk,al,mb,jlac,kmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,kj,ai,lb,kmac,lmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,kj,al,lb,kmac,imbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,kj,al,mb,klac,imbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,j,kl,ai,jb,kmac,lmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('i,j,kl,ai,mb,kjac,lmbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,kl,al,jb,kmac,imbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,kl,al,mb,kjac,imbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,j,kl,am,jb,kmac,ilbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,kl,am,mb,kjac,ilbc', L_b, R_b, S_oo_ab, S_vo_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                # block IjdJ & IblJ
                S2 -= lib.einsum('i,i,jk,ja,lb,lkba', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,la,jb,lkba', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 += lib.einsum('i,j,jk,ia,lb,lkba', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 -= lib.einsum('i,j,jk,la,ib,lkba', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,j,kl,ia,kb,jlba', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 += lib.einsum('i,j,kl,ka,ib,jlba', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 -= lib.einsum('i,i,jk,ja,lb,klab', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_2_b, optimize=True)
                S2 += lib.einsum('i,j,jk,ia,lb,klab', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_2_b, optimize=True)
                S2 -= lib.einsum('i,i,jk,ja,lb,lkba', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,la,jb,lkba', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 += lib.einsum('i,j,ki,ka,lb,ljba', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,j,ki,la,kb,ljba', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 -= lib.einsum('i,i,jk,ja,lb,klab', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_2_b, optimize=True)
                S2 += lib.einsum('i,j,ki,ka,lb,jlab', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_2_b, optimize=True)
                S2 -= lib.einsum('i,j,kl,ka,ib,jlab', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_2_b, optimize=True)
                S2 -= 1/3 * lib.einsum('i,i,jk,ja,lb,lmbc,mkca', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,la,lb,jmbc,mkca', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,i,jk,la,mb,jmbc,lkca', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,jk,ia,lb,lmbc,mkca', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,jk,la,lb,imbc,mkca', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,jk,la,mb,imbc,lkca', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,kl,ia,jb,kmbc,mlca', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,j,kl,ia,mb,kmbc,jlca', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,kl,ka,jb,imbc,mlca', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,kl,ka,mb,imbc,jlca', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,kl,ma,jb,ikbc,mlca', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,kl,ma,mb,ikbc,jlca', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,i,jk,ja,lb,lmbc,kmac', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,la,lb,jmbc,kmac', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,jk,ia,lb,lmbc,kmac', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,jk,la,lb,imbc,kmac', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,j,kl,ia,jb,kmbc,lmac', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,kl,ka,jb,imbc,lmac', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,la,ka,jb,lb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,j,jk,la,ka,ib,lb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t1_1_a, t1_1_a, optimize=True)
                S2 -= lib.einsum('i,j,kl,ia,la,jb,kb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t1_1_a, t1_1_a, optimize=True)
                S2 += 4/3 * lib.einsum('i,i,jk,ja,kb,la,lb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,jk,ia,kb,la,lb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += lib.einsum('i,i,jk,la,ka,jmbc,lmbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,jk,la,ka,imbc,lmbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,kl,ia,la,jmbc,kmbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,j,kl,ma,la,ikbc,jmbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,i,jk,ja,kb,lmca,lmcb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,i,jk,ja,la,mkbc,mlbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,i,jk,ja,lb,mkca,mlcb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,la,ka,jmbc,lmbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,jk,la,kb,jmcb,lmca', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,jk,la,ma,jmbc,lkbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,i,jk,la,mb,jmcb,lkca', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,jk,ia,kb,lmca,lmcb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,jk,ia,la,mkbc,mlbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,jk,ia,lb,mkca,mlcb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,jk,la,ka,imbc,lmbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,j,jk,la,kb,imcb,lmca', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,j,jk,la,ma,imbc,lkbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,jk,la,mb,imcb,lkca', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,kl,ia,la,jmbc,kmbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,kl,ia,lb,jmca,kmcb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,kl,ia,ma,jlbc,kmbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,kl,ia,mb,jlca,kmcb', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,kl,ka,lb,imcb,jmca', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,kl,ka,ma,imbc,jlbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,kl,ka,mb,imcb,jlca', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2/3 * lib.einsum('i,i,jk,ja,kb,lmac,lmbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 += 2/3 * lib.einsum('i,i,jk,ja,la,kmbc,lmbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,i,jk,ja,lb,kmac,lmbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 2/3 * lib.einsum('i,j,jk,ia,kb,lmac,lmbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 2/3 * lib.einsum('i,j,jk,ia,la,kmbc,lmbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,jk,ia,lb,kmac,lmbc', L_a, R_a, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,i,jk,ja,lb,lmbc,mkca', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,la,lb,jmbc,mkca', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,i,jk,la,mb,jmbc,lkca', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,ki,ka,lb,lmbc,mjca', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ki,la,lb,kmbc,mjca', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,j,ki,la,mb,kmbc,ljca', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,i,jk,ja,lb,lmbc,kmac', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,la,lb,jmbc,kmac', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,ki,ka,lb,lmbc,jmac', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ki,la,lb,kmbc,jmac', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,kl,ka,mb,mibc,jlac', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,kl,ma,mb,kibc,jlac', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,la,ka,jb,lb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ki,la,ja,kb,lb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,i,jk,ja,kb,la,lb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,ki,ka,jb,la,lb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,j,kl,ka,ib,ja,lb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, t1_1_b, optimize=True)
                S2 += lib.einsum('i,i,jk,la,ka,jmbc,lmbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,j,ki,la,ja,kmbc,lmbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,i,jk,ja,kb,lmca,lmcb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,i,jk,ja,la,mkbc,mlbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,i,jk,ja,lb,mkca,mlcb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,la,ka,jmbc,lmbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,jk,la,kb,jmcb,lmca', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,jk,la,ma,jmbc,lkbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,i,jk,la,mb,jmcb,lkca', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,ki,ka,jb,lmca,lmcb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,ki,ka,la,mjbc,mlbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,ki,ka,lb,mjca,mlcb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ki,la,ja,kmbc,lmbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,j,ki,la,jb,kmcb,lmca', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ki,la,ma,ljbc,kmbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,ki,la,mb,ljca,kmcb', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,kl,ka,ja,mibc,mlbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,kl,ka,jb,micb,mlca', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,kl,ka,lb,micb,mjca', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,kl,ma,ja,kibc,mlbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,kl,ma,jb,kicb,mlca', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,kl,ma,la,kibc,mjbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,kl,ma,lb,kicb,mjca', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2/3 * lib.einsum('i,i,jk,ja,kb,lmac,lmbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 += 2/3 * lib.einsum('i,i,jk,ja,la,kmbc,lmbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,i,jk,ja,lb,kmac,lmbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 1/6 * lib.einsum('i,j,ki,ka,jb,lmac,lmbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 1/6 * lib.einsum('i,j,ki,ka,la,jmbc,lmbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,ki,ka,lb,jmac,lmbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,kl,ka,ja,imbc,lmbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,kl,ka,jb,imbc,lmac', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,kl,ka,lb,imbc,jmac', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 2/3 * lib.einsum('i,j,kl,ka,ma,imbc,jlbc', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                S2 += 1/3 * lib.einsum('i,j,kl,ka,mb,imbc,jlac', L_b, R_b, S_oo_ab, S_ov_ab, t1_1_b, t2_1_b, t2_1_b,
                    optimize=True)
                # block IjdB & AblJ
                S2 -= 2 * lib.einsum('i,i,ja,bk,jb,ka', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_a, t1_2_b, optimize=True)
                S2 += 2 * lib.einsum('i,j,ia,bk,jb,ka', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_a, t1_2_b, optimize=True)
                S2 -= 2 * lib.einsum('i,i,ja,bk,ka,jb', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_b, t1_2_a, optimize=True)
                S2 += 2 * lib.einsum('i,j,ia,bk,ka,jb', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_b, t1_2_a, optimize=True)
                S2 -= 2 * lib.einsum('i,i,ja,bk,jb,ka', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_a, t1_2_b, optimize=True)
                S2 += 2 * lib.einsum('i,j,ka,bi,kb,ja', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_a, t1_2_b, optimize=True)
                S2 -= 2 * lib.einsum('i,i,ja,bk,ka,jb', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_b, t1_2_a, optimize=True)
                S2 += 2 * lib.einsum('i,j,ka,bi,ja,kb', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_b, t1_2_a, optimize=True)
                S2 -= lib.einsum('i,i,ja,bk,lc,ka,jlbc', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, t2_1_a,
                    optimize=True)
                S2 += lib.einsum('i,j,ia,bk,lc,ka,jlbc', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, t2_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,j,ka,bl,ic,la,jkbc', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, t2_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,i,ja,bk,jb,lc,klac', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('i,j,ia,bk,jb,lc,klac', L_a, R_a, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, t2_1_b,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,i,ja,bk,jkca,lb,lc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,i,ja,bk,lkba,jc,lc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,i,ja,bk,lkca,jb,lc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,ia,bk,jkca,lb,lc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,ia,bk,lkba,jc,lc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += lib.einsum('i,j,ia,bk,lkca,jb,lc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,j,ka,bl,jlba,ic,kc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += lib.einsum('i,j,ka,bl,jlca,ic,kb', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,j,ka,bl,klca,ic,jb', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,i,ja,bk,jkbc,la,lc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,i,ja,bk,jlba,kc,lc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,i,ja,bk,jlbc,ka,lc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,ia,bk,jkbc,la,lc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,ia,bk,jlba,kc,lc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += lib.einsum('i,j,ia,bk,jlbc,ka,lc', L_a, R_a, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,i,ja,bk,lc,ka,jlbc', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, t2_1_a,
                    optimize=True)
                S2 += lib.einsum('i,j,ka,bi,lc,ja,klbc', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, t2_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,i,ja,bk,jb,lc,klac', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('i,j,ka,bi,kb,lc,jlac', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,j,ka,bl,kb,ic,jlac', L_b, R_b, S_ov_ab, S_vo_ab, t1_1_a, t1_1_b, t2_1_b,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,i,ja,bk,jkca,lb,lc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,i,ja,bk,lkba,jc,lc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,i,ja,bk,lkca,jb,lc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,ka,bi,kjca,lb,lc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,ka,bi,ljba,kc,lc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += lib.einsum('i,j,ka,bi,ljca,kb,lc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,i,ja,bk,jkbc,la,lc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 4/3 * lib.einsum('i,i,ja,bk,jlba,kc,lc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,i,ja,bk,jlbc,ka,lc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 4/3 * lib.einsum('i,j,ka,bi,kjbc,la,lc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 1/3 * lib.einsum('i,j,ka,bi,klba,jc,lc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += lib.einsum('i,j,ka,bi,klbc,ja,lc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,j,ka,bl,kjba,ic,lc', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += lib.einsum('i,j,ka,bl,kjbc,ic,la', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,j,ka,bl,klbc,ic,ja', L_b, R_b, S_ov_ab, S_vo_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                # block IblB & AjdJ
                S2 -= 2 * lib.einsum('i,i,jk,ab,ja,kb', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_a, t1_2_b, optimize=True)
                S2 += 2 * lib.einsum('i,j,ik,ab,ja,kb', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_a, t1_2_b, optimize=True)
                S2 -= 2 * lib.einsum('i,i,jk,ab,kb,ja', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_b, t1_2_a, optimize=True)
                S2 += 2 * lib.einsum('i,j,ik,ab,kb,ja', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_b, t1_2_a, optimize=True)
                S2 -= 2 * lib.einsum('i,i,jk,ab,ja,kb', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_a, t1_2_b, optimize=True)
                S2 += 2 * lib.einsum('i,j,kj,ab,ka,ib', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_a, t1_2_b, optimize=True)
                S2 -= 2 * lib.einsum('i,i,jk,ab,kb,ja', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_b, t1_2_a, optimize=True)
                S2 += 2 * lib.einsum('i,j,kj,ab,ib,ka', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_b, t1_2_a, optimize=True)
                S2 -= lib.einsum('i,i,jk,ab,lc,kb,jlac', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, t2_1_a,
                    optimize=True)
                S2 += lib.einsum('i,j,ik,ab,lc,kb,jlac', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, t2_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,j,kl,ab,ic,lb,jkac', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, t2_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,i,jk,ab,ja,lc,klbc', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('i,j,ik,ab,ja,lc,klbc', L_a, R_a, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,i,jk,ab,lkcb,ja,lc', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,ab,lkcb,jc,la', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += lib.einsum('i,j,ik,ab,lkcb,ja,lc', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,j,ik,ab,lkcb,jc,la', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,kl,ab,ilcb,ja,kc', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += lib.einsum('i,j,kl,ab,ilcb,jc,ka', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,i,jk,ab,jlac,kb,lc', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,ab,jlac,kc,lb', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += lib.einsum('i,j,ik,ab,jlac,kb,lc', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ik,ab,jlac,kc,lb', L_a, R_a, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,i,jk,ab,lc,kb,jlac', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, t2_1_a,
                    optimize=True)
                S2 += lib.einsum('i,j,kj,ab,lc,ib,klac', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, t2_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,i,jk,ab,ja,lc,klbc', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('i,j,kj,ab,ka,lc,ilbc', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,j,kl,ab,ka,jc,ilbc', L_b, R_b, S_oo_ab, S_vv_ab, t1_1_a, t1_1_b, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,i,jk,ab,lkcb,ja,lc', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,ab,lkcb,jc,la', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += lib.einsum('i,j,kj,ab,licb,ka,lc', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,kj,ab,licb,kc,la', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,i,jk,ab,jlac,kb,lc', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,jk,ab,jlac,kc,lb', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += lib.einsum('i,j,kj,ab,klac,ib,lc', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,j,kj,ab,klac,ic,lb', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,kl,ab,kjac,ib,lc', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += lib.einsum('i,j,kl,ab,kjac,ic,lb', L_b, R_b, S_oo_ab, S_vv_ab, t2_1_ab, t1_1_b, t1_1_b,
                    optimize=True)
                # block IbdJ
                S2 -= 2 * lib.einsum('i,i,ja,jb,ka,kb', L_a, R_a, t1_1_b, t1_2_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 += lib.einsum('i,j,ka,kb,ia,jb', L_a, R_a, t1_1_b, t1_2_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 += lib.einsum('i,j,ka,kb,ib,ja', L_a, R_a, t1_1_b, t1_2_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,i,ja,jb,ka,kb', L_b, R_b, t1_1_b, t1_2_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 += lib.einsum('i,j,ia,jb,ka,kb', L_b, R_b, t1_1_b, t1_2_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 += lib.einsum('i,j,ja,ib,ka,kb', L_b, R_b, t1_1_b, t1_2_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= lib.einsum('i,i,ja,kb,jkac,lb,lc', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,ja,kb,lkac,jb,lc', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,ia,kb,jkac,lb,lc', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ia,kb,lkac,jb,lc', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,ja,kb,ikac,lb,lc', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ja,kb,lkac,ib,lc', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,ka,lb,ilac,jc,kb', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,ka,lb,jlac,ic,kb', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,ka,lb,klac,ib,jc', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,ka,lb,klac,ic,jb', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += lib.einsum('i,i,jkab,la,lc,jb,kc', L_a, R_a, t2_1_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,klab,ia,jc,kb,lc', L_a, R_a, t2_1_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,klab,ic,ja,kb,lc', L_a, R_a, t2_1_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,i,ja,kb,jkac,lb,lc', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,ja,kb,lkac,jb,lc', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,ka,ib,kjac,lb,lc', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,ka,ib,ljac,kb,lc', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,ka,jb,kiac,lb,lc', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,ka,jb,liac,kb,lc', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += lib.einsum('i,i,jkab,la,lc,jb,kc', L_b, R_b, t2_1_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ikab,la,lc,jb,kc', L_b, R_b, t2_1_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,ikab,la,lc,jc,kb', L_b, R_b, t2_1_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,jkab,la,lc,ib,kc', L_b, R_b, t2_1_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,jkab,la,lc,ic,kb', L_b, R_b, t2_1_b, S_ov_ab, S_ov_ab, t1_1_b, t1_1_b,
                    optimize=True)
                # block IbdB & AbdJ
                S2 -= 2 * lib.einsum('i,i,ja,bc,kc,jkba', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 += 2 * lib.einsum('i,j,ia,bc,kc,jkba', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,i,ja,bc,kc,jkba', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 += 2 * lib.einsum('i,j,ka,bc,ic,kjba', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_2_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,i,ja,bc,jb,ka,kc', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ia,bc,jb,ka,kc', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,ja,bc,jb,klda,kldc', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,ja,bc,jd,klba,kldc', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,ja,bc,kb,jlda,kldc', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,i,ja,bc,kd,jlba,kldc', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ia,bc,jb,klda,kldc', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,ia,bc,jd,klba,kldc', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ia,bc,kb,jlda,kldc', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,j,ia,bc,kd,jlba,kldc', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ka,bc,jb,ildc,klda', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,j,ka,bc,jd,ildc,klba', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ka,bc,kb,ildc,jlda', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ka,bc,kd,ildc,jlba', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,i,ja,bc,jb,klad,klcd', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_b, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('i,j,ia,bc,jb,klad,klcd', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_a, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,ja,bc,ka,jlbd,lkdc', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ia,bc,ka,jlbd,lkdc', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ka,bc,la,jkbd,ildc', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,ja,bc,ka,jlbd,klcd', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('i,i,ja,bc,kd,jlba,klcd', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ia,bc,ka,jlbd,klcd', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,j,ia,bc,kd,jlba,klcd', L_a, R_a, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,ja,bc,jb,ka,kc', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, t1_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ka,bc,kb,ic,ja', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, t1_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,ja,bc,jb,klda,kldc', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,ja,bc,jd,klba,kldc', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,ja,bc,kb,jlda,kldc', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,i,ja,bc,kd,jlba,kldc', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ka,bc,kb,lidc,ljda', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ka,bc,kd,lidc,ljba', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ka,bc,lb,lidc,kjda', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,j,ka,bc,ld,lidc,kjba', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,i,ja,bc,jb,klad,klcd', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_b, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ka,bc,kb,ilcd,jlad', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_a, t2_1_b, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,ja,bc,ka,jlbd,lkdc', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ka,bc,ja,klbd,lidc', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,ja,bc,ka,jlbd,klcd', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('i,i,ja,bc,kd,jlba,klcd', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ka,bc,ja,klbd,ilcd', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= lib.einsum('i,j,ka,bc,jd,klba,ilcd', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ka,bc,la,kjbd,ilcd', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('i,j,ka,bc,ld,kjba,ilcd', L_b, R_b, S_ov_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_b,
                    optimize=True)
                # block AjlB
                S2 -= 2 * lib.einsum('i,i,ja,jb,ak,bk', L_a, R_a, t1_1_a, t1_2_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 += lib.einsum('i,j,ia,jb,ak,bk', L_a, R_a, t1_1_a, t1_2_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 += lib.einsum('i,j,ja,ib,ak,bk', L_a, R_a, t1_1_a, t1_2_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,i,ja,jb,ak,bk', L_b, R_b, t1_1_a, t1_2_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 += lib.einsum('i,j,ka,kb,ai,bj', L_b, R_b, t1_1_a, t1_2_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 += lib.einsum('i,j,ka,kb,bi,aj', L_b, R_b, t1_1_a, t1_2_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= lib.einsum('i,i,ja,kb,jkcb,al,cl', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,ja,kb,jlcb,ak,cl', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,ia,kb,jkcb,al,cl', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,ia,kb,jlcb,ak,cl', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,ja,kb,ikcb,al,cl', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,ja,kb,ilcb,ak,cl', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += lib.einsum('i,i,jkab,al,cl,jb,kc', L_a, R_a, t2_1_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ikab,al,cl,jb,kc', L_a, R_a, t2_1_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,ikab,al,cl,jc,kb', L_a, R_a, t2_1_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,jkab,al,cl,ib,kc', L_a, R_a, t2_1_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,jkab,al,cl,ic,kb', L_a, R_a, t2_1_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,i,ja,kb,jkcb,al,cl', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,ja,kb,jlcb,ak,cl', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,ka,ib,kjcb,al,cl', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ka,ib,klcb,aj,cl', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,ka,jb,kicb,al,cl', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,ka,jb,klcb,ai,cl', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,ka,lb,kicb,cj,al', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,ka,lb,kjcb,ci,al', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,ka,lb,klcb,ai,cj', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 1/2 * lib.einsum('i,j,ka,lb,klcb,ci,aj', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += lib.einsum('i,i,jkab,al,cl,jb,kc', L_b, R_b, t2_1_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,klab,ai,cj,kb,lc', L_b, R_b, t2_1_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= 1/2 * lib.einsum('i,j,klab,ci,aj,kb,lc', L_b, R_b, t2_1_a, S_vo_ab, S_vo_ab, t1_1_a, t1_1_a,
                    optimize=True)
                # block AjdB & AblB
                S2 -= 2 * lib.einsum('i,i,aj,bc,kb,kjac', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 += 2 * lib.einsum('i,j,ak,bc,ib,jkac', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,i,aj,bc,kb,kjac', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 += 2 * lib.einsum('i,j,ai,bc,kb,kjac', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_2_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,i,aj,bc,ka,klbd,ljdc', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,i,aj,bc,kd,klbd,ljac', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ak,bc,ja,ilbd,lkdc', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,ak,bc,jd,ilbd,lkac', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ak,bc,la,ilbd,jkdc', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,j,ak,bc,ld,ilbd,jkac', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,aj,bc,ka,klbd,jlcd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ak,bc,ja,ilbd,klcd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,aj,bc,jc,ka,kb', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ak,bc,kc,ib,ja', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,i,aj,bc,jc,klad,klbd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ak,bc,kc,ilbd,jlad', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,aj,bc,jc,klad,klbd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,aj,bc,jd,klac,klbd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,aj,bc,kc,ljad,lkbd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,i,aj,bc,kd,ljac,lkbd', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ak,bc,kc,ilbd,jlad', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ak,bc,kd,ilbd,jlac', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ak,bc,lc,ilbd,jkad', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,j,ak,bc,ld,ilbd,jkac', L_a, R_a, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,aj,bc,ka,klbd,ljdc', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,i,aj,bc,kd,klbd,ljac', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ai,bc,ka,klbd,ljdc', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,ai,bc,kd,klbd,ljac', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,aj,bc,ka,klbd,jlcd', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ai,bc,ka,klbd,jlcd', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ak,bc,la,libd,jkcd', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_a, t2_1_ab, t2_1_b,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,aj,bc,jc,ka,kb', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t1_1_a, t1_1_a,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ai,bc,jc,ka,kb', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t1_1_a, t1_1_a,
                    optimize=True)
                S2 -= lib.einsum('i,i,aj,bc,jc,klad,klbd', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_a,
                    optimize=True)
                S2 += lib.einsum('i,j,ai,bc,jc,klad,klbd', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_a, t2_1_a,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,aj,bc,jc,klad,klbd', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,aj,bc,jd,klac,klbd', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,i,aj,bc,kc,ljad,lkbd', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,i,aj,bc,kd,ljac,lkbd', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ai,bc,jc,klad,klbd', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= lib.einsum('i,j,ai,bc,jd,klac,klbd', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ai,bc,kc,ljad,lkbd', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,j,ai,bc,kd,ljac,lkbd', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ak,bc,jc,libd,lkad', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,j,ak,bc,jd,libd,lkac', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,j,ak,bc,kc,libd,ljad', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,j,ak,bc,kd,libd,ljac', L_b, R_b, S_vo_ab, S_vv_ab, t1_1_b, t2_1_ab, t2_1_ab,
                    optimize=True)
                # block AbdB
                S2 -= 2 * lib.einsum('i,i,ja,kb,jkcd,ad,cb', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 += lib.einsum('i,j,ia,kb,jkcd,ad,cb', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 += lib.einsum('i,j,ja,kb,ikcd,ad,cb', L_a, R_a, t1_1_a, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,i,ja,kb,jkcd,ad,cb', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 += lib.einsum('i,j,ka,ib,kjcd,ad,cb', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 += lib.einsum('i,j,ka,jb,kicd,ad,cb', L_b, R_b, t1_1_a, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)

            # 120 & 021
            # block AblJ
            S2 += 2 * lib.einsum('i,aij,jb,ak,kb', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_2_b, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,ka,bj,kb', L_a, R_bab, S_ov_ab, S_vo_ab, t1_2_a, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,ka,bj,ib', L_a, R_bab, S_ov_ab, S_vo_ab, t1_2_a, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,jb,ak,kb', L_b, R_aba, S_ov_ab, S_vo_ab, t1_2_b, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,jb,ak,ib', L_b, R_aba, S_ov_ab, S_vo_ab, t1_2_b, optimize=True)
            S2 += 2 * lib.einsum('i,aij,ka,bj,kb', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_2_a, optimize=True)
            # block AjdJ
            S2 += 2 * lib.einsum('i,aij,jk,ab,kb', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_2_b, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,jk,ab,kb', L_b, R_aba, S_oo_ab, S_vv_ab, t1_2_b, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,ji,ab,kb', L_b, R_aba, S_oo_ab, S_vv_ab, t1_2_b, optimize=True)
            # block AjlB
            S2 += 2 * lib.einsum('i,aij,jb,ak,bk', L_a, R_aaa_u, t1_2_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,jb,ak,bk', L_b, R_aba, t1_2_a, S_vo_ab, S_vo_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,jb,bi,ak', L_b, R_aba, t1_2_a, S_vo_ab, S_vo_ab, optimize=True)
            # block IbdJ
            S2 -= 2 * lib.einsum('i,aji,jb,ka,kb', L_a, R_bab, t1_2_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,jb,ib,ka', L_a, R_bab, t1_2_b, S_ov_ab, S_ov_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aij,jb,ka,kb', L_b, R_bbb_u, t1_2_b, S_ov_ab, S_ov_ab, optimize=True)
            # block IblB
            S2 -= 2 * lib.einsum('i,aji,kj,ba,kb', L_a, R_bab, S_oo_ab, S_vv_ab, t1_2_a, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,ij,ba,kb', L_a, R_bab, S_oo_ab, S_vv_ab, t1_2_a, optimize=True)
            S2 += 2 * lib.einsum('i,aij,kj,ba,kb', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_2_a, optimize=True)
            # block IjlJ
            S2 -= 2 * lib.einsum('i,aij,ka,jl,kl', L_a, R_aaa_u, t1_2_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aji,ka,lj,lk', L_a, R_bab, t1_2_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,la,ij,kl', L_a, R_bab, t1_2_b, S_oo_ab, S_oo_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aji,ka,jl,kl', L_b, R_aba, t1_2_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,la,ji,lk', L_b, R_aba, t1_2_a, S_oo_ab, S_oo_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aij,ka,lj,lk', L_b, R_bbb_u, t1_2_b, S_oo_ab, S_oo_ab, optimize=True)
            # block IjlB
            S2 += 2 * lib.einsum('i,aij,kl,bl,jkab', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_2_a, optimize=True)
            S2 += lib.einsum('i,ajk,il,bl,jkab', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_2_a, optimize=True)
            S2 += 2 * lib.einsum('i,aji,kj,bl,klba', L_a, R_bab, S_oo_ab, S_vo_ab, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,kl,bl,kjba', L_a, R_bab, S_oo_ab, S_vo_ab, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,ij,bl,klba', L_a, R_bab, S_oo_ab, S_vo_ab, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,il,bl,kjba', L_a, R_bab, S_oo_ab, S_vo_ab, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,kl,bl,jkab', L_b, R_aba, S_oo_ab, S_vo_ab, t2_2_a, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,lk,bi,jlab', L_b, R_aba, S_oo_ab, S_vo_ab, t2_2_a, optimize=True)
            S2 -= 2 * lib.einsum('i,aij,kj,bl,klba', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aij,kl,bl,kjba', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,lj,bi,lkba', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_2_ab, optimize=True)
            # block IjdJ
            S2 -= 2 * lib.einsum('i,aij,jk,lb,lkab', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aij,kl,kb,jlab', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,jl,ib,klab', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,kl,kb,jlab', L_a, R_bab, S_oo_ab, S_ov_ab, t2_2_b, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,kl,ib,jlab', L_a, R_bab, S_oo_ab, S_ov_ab, t2_2_b, optimize=True)
            S2 += 2 * lib.einsum('i,aji,jk,lb,lkab', L_b, R_aba, S_oo_ab, S_ov_ab, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,kl,kb,jlab', L_b, R_aba, S_oo_ab, S_ov_ab, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,ji,lb,lkab', L_b, R_aba, S_oo_ab, S_ov_ab, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,li,lb,jkab', L_b, R_aba, S_oo_ab, S_ov_ab, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aij,kl,kb,jlab', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_2_b, optimize=True)
            S2 += lib.einsum('i,ajk,li,lb,jkab', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_2_b, optimize=True)
            # block IblJ
            S2 -= 2 * lib.einsum('i,aij,kl,jb,kmac,mlcb', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,aij,kl,kb,jmac,mlcb', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aij,kl,mb,jkac,mlcb', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,il,jb,kmac,mlcb', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,ajk,il,mb,jkac,mlcb', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_a,
                t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,lm,jb,klac,imcb', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('i,ajk,lm,lb,jkac,imcb', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_a,
                t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aij,kl,jb,kmac,lmbc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += lib.einsum('i,aij,kl,kb,jmac,lmbc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= lib.einsum('i,ajk,il,jb,kmac,lmbc', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += lib.einsum('i,aji,kj,la,kmbc,lmbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_a, t2_1_a, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,ajk,ij,la,kmbc,lmbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,ajk,lj,ka,imbc,lmbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= lib.einsum('i,ajk,lj,ma,imbc,klbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_a, t2_1_a, optimize=True)
            S2 += lib.einsum('i,aji,kj,kb,lmca,lmcb', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aji,kj,la,kmbc,lmbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,kj,lb,kmca,lmcb', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,aji,kl,ka,mjbc,mlbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,aji,kl,kb,mjca,mlcb', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,kl,ma,kjbc,mlbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aji,kl,mb,kjca,mlcb', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,ij,kb,lmca,lmcb', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,ij,la,kmbc,lmbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,ajk,ij,lb,kmca,lmcb', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,il,ka,mjbc,mlbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,ajk,il,kb,mjca,mlcb', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,ajk,il,ma,kjbc,mlbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,il,mb,kjca,mlcb', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,lj,ka,imbc,lmbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,lj,kb,imcb,lmca', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,lj,lb,imcb,kmca', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,lm,ka,imbc,ljbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,lm,kb,imcb,ljca', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,lm,la,imbc,kjbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,ajk,lm,lb,imcb,kjca', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 1 / 2 * lib.einsum('i,aji,kj,kb,lmac,lmbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += 1 / 2 * lib.einsum('i,aji,kl,ka,jmbc,lmbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= lib.einsum('i,aji,kl,kb,jmac,lmbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_b, t2_1_b, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,ajk,ij,kb,lmac,lmbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,ajk,il,ka,jmbc,lmbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += lib.einsum('i,ajk,il,kb,jmac,lmbc', L_a, R_bab, S_oo_ab, S_ov_ab, t2_1_b, t2_1_b, optimize=True)
            S2 += 2 * lib.einsum('i,aji,kl,jb,kmac,mlcb', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,aji,kl,kb,jmac,mlcb', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aji,kl,mb,jkac,mlcb', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,lk,jb,lmac,micb', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,ajk,lk,lb,jmac,micb', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,lk,mb,jlac,micb', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aji,kl,jb,kmac,lmbc', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= lib.einsum('i,aji,kl,kb,jmac,lmbc', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,lk,jb,lmac,imbc', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += lib.einsum('i,ajk,lk,lb,jmac,imbc', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,lm,jb,lkac,imbc', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= lib.einsum('i,ajk,lm,lb,jkac,imbc', L_b, R_aba, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= lib.einsum('i,aij,kj,la,kmbc,lmbc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_a, t2_1_a, optimize=True)
            S2 -= lib.einsum('i,aij,kj,kb,lmca,lmcb', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aij,kj,la,kmbc,lmbc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aij,kj,lb,kmca,lmcb', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,aij,kl,ka,mjbc,mlbc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,aij,kl,kb,mjca,mlcb', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aij,kl,ma,kjbc,mlbc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aij,kl,mb,kjca,mlcb', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,lj,lb,micb,mkca', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,lj,ma,mibc,lkbc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,lj,mb,micb,lkca', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,aij,kj,kb,lmac,lmbc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,aij,kl,ka,jmbc,lmbc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 += lib.einsum('i,aij,kl,kb,jmac,lmbc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b, t2_1_b, optimize=True)
            S2 -= lib.einsum('i,ajk,lj,lb,imbc,kmac', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b, t2_1_b, optimize=True)
            S2 += 1 / 4 * lib.einsum('i,ajk,lm,la,imbc,jkbc', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,ajk,lm,lb,imbc,jkac', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t2_1_b,
                t2_1_b, optimize=True)
            # block IbdB
            S2 -= 2 * lib.einsum('i,aji,kb,ca,kjcb', L_a, R_bab, S_ov_ab, S_vv_ab, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,ib,ca,kjcb', L_a, R_bab, S_ov_ab, S_vv_ab, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aij,kb,ca,kjcb', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_2_ab, optimize=True)
            # block AjlJ
            S2 -= 1 / 2 * lib.einsum('i,aij,jk,bk,lmac,lmbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,aij,kl,al,jmbc,kmbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += lib.einsum('i,aij,kl,bl,jmac,kmbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a, t2_1_a, optimize=True)
            S2 -= lib.einsum('i,ajk,jl,bl,imbc,kmac', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a, t2_1_a, optimize=True)
            S2 += 1 / 4 * lib.einsum('i,ajk,lm,am,ilbc,jkbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,ajk,lm,bm,ilbc,jkac', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 2 * lib.einsum('i,aij,jk,al,mkbc,mlbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,aij,jk,bk,lmac,lmbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aij,jk,bl,mkac,mlbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,aij,kl,al,jmbc,kmbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aij,kl,am,jlbc,kmbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += lib.einsum('i,aij,kl,bl,jmac,kmbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aij,kl,bm,jlac,kmbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,jl,am,imbc,klbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,jl,bl,imbc,kmac', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,jl,bm,imbc,klac', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,aij,jk,al,kmbc,lmbc', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t2_1_b, t2_1_b, optimize=True)
            S2 += 2 * lib.einsum('i,aji,kl,bj,kmbc,mlca', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,aji,kl,bl,kmbc,mjca', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,kl,bj,imbc,mlca', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,ajk,kl,bl,imbc,mjca', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,lm,bj,ilbc,kmca', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,lm,bm,ilbc,kjca', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aji,kl,bj,kmbc,lmac', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= lib.einsum('i,aji,kl,bl,kmbc,jmac', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += 2 * lib.einsum('i,aji,kl,bm,kmbc,jlac', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,kl,bj,imbc,lmac', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += lib.einsum('i,ajk,kl,bl,imbc,jmac', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,kl,bm,imbc,jlac', L_a, R_bab, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += 1 / 2 * lib.einsum('i,aji,jk,bk,lmac,lmbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += 1 / 2 * lib.einsum('i,aji,kl,al,jmbc,kmbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= lib.einsum('i,aji,kl,bl,jmac,kmbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_a, t2_1_a, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,ajk,ji,bk,lmac,lmbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,ajk,li,ak,jmbc,lmbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_a,
                t2_1_a, optimize=True)
            S2 += lib.einsum('i,ajk,li,bk,jmac,lmbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_a, t2_1_a, optimize=True)
            S2 += 2 * lib.einsum('i,aji,jk,al,mkbc,mlbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,aji,jk,bk,lmac,lmbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,jk,bl,mkac,mlbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,aji,kl,al,jmbc,kmbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,kl,am,jlbc,kmbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,aji,kl,bl,jmac,kmbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aji,kl,bm,jlac,kmbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,ji,al,mkbc,mlbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,ji,bk,lmac,lmbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,ajk,ji,bl,mkac,mlbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,jl,ak,mibc,mlbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,jl,bk,mibc,mlac', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,jl,bl,mibc,mkac', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,li,ak,jmbc,lmbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,ajk,li,am,jkbc,lmbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,ajk,li,bk,jmac,lmbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,li,bm,jkac,lmbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,lm,ak,libc,jmbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,lm,am,libc,jkbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,lm,bk,libc,jmac', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,ajk,lm,bm,libc,jkac', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,aji,jk,al,kmbc,lmbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_b, t2_1_b, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,ajk,ji,al,kmbc,lmbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,ajk,jl,ak,imbc,lmbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_b,
                t2_1_b, optimize=True)
            S2 -= lib.einsum('i,ajk,jl,am,imbc,klbc', L_b, R_aba, S_oo_ab, S_vo_ab, t2_1_b, t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('i,aij,kl,bj,kmbc,mlca', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,aij,kl,bl,kmbc,mjca', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,ajk,li,bj,lmbc,mkca', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aij,kl,bj,kmbc,lmac', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += lib.einsum('i,aij,kl,bl,kmbc,jmac', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('i,aij,kl,bm,kmbc,jlac', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= lib.einsum('i,ajk,li,bj,lmbc,kmac', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= 1 / 2 * lib.einsum('i,ajk,li,bm,lmbc,jkac', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab,
                t2_1_b, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,lm,bj,libc,kmac', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += 1 / 2 * lib.einsum('i,ajk,lm,bm,libc,jkac', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t2_1_ab,
                t2_1_b, optimize=True)
            # block AjdB
            S2 += 2 * lib.einsum('i,aij,bk,ac,jkbc', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_2_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,bk,ac,jkbc', L_b, R_aba, S_vo_ab, S_vv_ab, t2_2_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,bi,ac,jkbc', L_b, R_aba, S_vo_ab, S_vv_ab, t2_2_ab, optimize=True)
            # block AblB
            S2 += 2 * lib.einsum('i,aij,ak,bc,jlbd,lkdc', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aij,bk,cd,jlac,lkbd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,ajk,al,bc,jkbd,ildc', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += lib.einsum('i,ajk,bl,cd,jkac,ilbd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aij,ak,bc,jlbd,klcd', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= lib.einsum('i,aji,bj,ca,klbd,klcd', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_a, t2_1_a, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,bj,ca,ilbd,klcd', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_a, t2_1_a, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,bj,ca,klbd,klcd', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aji,bj,cd,klca,klbd', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aji,bk,ca,ljcd,lkbd', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,bk,cd,ljca,lkbd', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,bj,ca,ilbd,klcd', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,bj,cd,ilbd,klca', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,bl,ca,ilbd,kjcd', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,bl,cd,ilbd,kjca', L_a, R_bab, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,ak,bc,jlbd,lkdc', L_b, R_aba, S_vo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,bk,cd,jlac,lkbd', L_b, R_aba, S_vo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,ak,bc,jlbd,lidc', L_b, R_aba, S_vo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,bk,cd,jlac,libd', L_b, R_aba, S_vo_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,ak,bc,jlbd,klcd', L_b, R_aba, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,ak,bc,jlbd,ilcd', L_b, R_aba, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,al,bc,jkbd,ilcd', L_b, R_aba, S_vo_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += lib.einsum('i,aij,bj,ca,klbd,klcd', L_b, R_bbb_u, S_vo_ab, S_vv_ab, t2_1_a, t2_1_a, optimize=True)
            S2 += 2 * lib.einsum('i,aij,bj,ca,klbd,klcd', L_b, R_bbb_u, S_vo_ab, S_vv_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aij,bj,cd,klca,klbd', L_b, R_bbb_u, S_vo_ab, S_vv_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aij,bk,ca,ljcd,lkbd', L_b, R_bbb_u, S_vo_ab, S_vv_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aij,bk,cd,ljca,lkbd', L_b, R_bbb_u, S_vo_ab, S_vv_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,bj,ca,libd,lkcd', L_b, R_bbb_u, S_vo_ab, S_vv_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,bj,cd,libd,lkca', L_b, R_bbb_u, S_vo_ab, S_vv_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            # block AbdJ
            S2 += 2 * lib.einsum('i,aij,jb,ac,kldb,kldc', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aij,jb,cd,klad,klcb', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aij,kb,ac,jldc,kldb', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aij,kb,cd,jlad,klcb', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,jb,ac,ildb,kldc', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,jb,cd,ilcb,klad', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_ab,
                t2_1_ab, optimize=True)
            S2 += lib.einsum('i,aij,jb,ac,klbd,klcd', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t2_1_b, t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,ka,bc,klbd,ljdc', L_a, R_bab, S_ov_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,ka,bc,ilbd,ljdc', L_a, R_bab, S_ov_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,la,bc,ilbd,kjdc', L_a, R_bab, S_ov_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,ka,bc,klbd,jlcd', L_a, R_bab, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,kb,cd,klcb,jlad', L_a, R_bab, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,ka,bc,ilbd,jlcd', L_a, R_bab, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,kb,cd,ilcb,jlad', L_a, R_bab, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,jb,ac,kldb,kldc', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aji,jb,cd,klad,klcb', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aji,kb,ac,jldc,kldb', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,aji,kb,cd,jlad,klcb', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,jb,ac,lidb,lkdc', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,jb,cd,licb,lkad', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= 2 * lib.einsum('i,ajk,lb,ac,lidb,jkdc', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,lb,cd,licb,jkad', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_ab, optimize=True)
            S2 -= lib.einsum('i,aji,jb,ac,klbd,klcd', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_b, t2_1_b, optimize=True)
            S2 += 2 * lib.einsum('i,ajk,jb,ac,ilbd,klcd', L_b, R_aba, S_ov_ab, S_vv_ab, t2_1_b, t2_1_b, optimize=True)
            S2 += 2 * lib.einsum('i,aij,ka,bc,klbd,ljdc', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_a, t2_1_ab, optimize=True)
            S2 += 2 * lib.einsum('i,aij,ka,bc,klbd,jlcd', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += 2 * lib.einsum('i,aij,kb,cd,klcb,jlad', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += lib.einsum('i,ajk,la,bc,libd,jkcd', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)
            S2 += lib.einsum('i,ajk,lb,cd,licb,jkad', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t2_1_ab, t2_1_b, optimize=True)

            if t1_1_a is not None:
                # block IjlJ
                S2 -= 2 * lib.einsum('i,aij,kb,jlab,km,lm', L_a, R_aaa_u, t1_1_a, t2_1_a, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('i,aij,kb,klab,jm,lm', L_a, R_aaa_u, t1_1_a, t2_1_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('i,ajk,ib,jlab,km,lm', L_a, R_aaa_u, t1_1_a, t2_1_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('i,ajk,lb,jkab,im,lm', L_a, R_aaa_u, t1_1_a, t2_1_a, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,aij,kb,jlab,mk,ml', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= lib.einsum('i,aij,kb,lkab,jm,lm', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 += 2 * lib.einsum('i,aij,kb,lmab,jm,lk', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,lb,jmab,il,km', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('i,aji,kb,klba,mj,ml', L_a, R_bab, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kb,ljba,km,lm', L_a, R_bab, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,aji,kb,lmba,lj,km', L_a, R_bab, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('i,ajk,ib,klba,mj,ml', L_a, R_bab, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('i,ajk,ib,ljba,km,lm', L_a, R_bab, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 += lib.einsum('i,ajk,ib,lmba,lj,km', L_a, R_bab, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('i,ajk,lb,kjba,im,lm', L_a, R_bab, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,lb,kmba,ij,lm', L_a, R_bab, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('i,ajk,lb,lmba,ij,km', L_a, R_bab, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kb,jlab,mk,ml', L_a, R_bab, t1_1_b, t2_1_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('i,aji,kb,klab,mj,ml', L_a, R_bab, t1_1_b, t2_1_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,lb,jmab,il,km', L_a, R_bab, t1_1_b, t2_1_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 += lib.einsum('i,ajk,lb,lmab,ij,km', L_a, R_bab, t1_1_b, t2_1_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kb,jlab,km,lm', L_b, R_aba, t1_1_a, t2_1_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('i,aji,kb,klab,jm,lm', L_b, R_aba, t1_1_a, t2_1_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,lb,jmab,li,mk', L_b, R_aba, t1_1_a, t2_1_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 += lib.einsum('i,ajk,lb,lmab,ji,mk', L_b, R_aba, t1_1_a, t2_1_a, S_oo_ab, S_oo_ab, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kb,jlab,mk,ml', L_b, R_aba, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 += lib.einsum('i,aji,kb,lkab,jm,lm', L_b, R_aba, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,aji,kb,lmab,jm,lk', L_b, R_aba, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('i,ajk,ib,jlab,mk,ml', L_b, R_aba, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('i,ajk,ib,lkab,jm,lm', L_b, R_aba, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 += lib.einsum('i,ajk,ib,lmab,jm,lk', L_b, R_aba, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('i,ajk,lb,jkab,mi,ml', L_b, R_aba, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,lb,mkab,ji,ml', L_b, R_aba, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('i,ajk,lb,mlab,ji,mk', L_b, R_aba, t1_1_b, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('i,aij,kb,klba,mj,ml', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,aij,kb,ljba,km,lm', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,aij,kb,lmba,lj,km', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,lb,mjba,li,mk', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,aij,kb,jlab,mk,ml', L_b, R_bbb_u, t1_1_b, t2_1_b, S_oo_ab, S_oo_ab,
                    optimize=True)
                S2 += lib.einsum('i,aij,kb,klab,mj,ml', L_b, R_bbb_u, t1_1_b, t2_1_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= lib.einsum('i,ajk,ib,jlab,mk,ml', L_b, R_bbb_u, t1_1_b, t2_1_b, S_oo_ab, S_oo_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('i,ajk,lb,jkab,mi,ml', L_b, R_bbb_u, t1_1_b, t2_1_b, S_oo_ab, S_oo_ab,
                    optimize=True)
                # block IjlB & AjlJ
                S2 -= 2 * lib.einsum('i,aij,kl,bl,ka,jb', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kj,bl,kb,la', L_a, R_bab, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,ij,bl,kb,la', L_a, R_bab, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kl,bl,ka,jb', L_b, R_aba, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,lk,bi,la,jb', L_b, R_aba, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
                S2 -= 2 * lib.einsum('i,aij,kj,bl,kb,la', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
                # block IjdJ & IblJ
                S2 -= 2 * lib.einsum('i,aij,jk,lb,la,kb', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kl,kb,la,jb', L_a, R_bab, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,kl,ib,la,jb', L_a, R_bab, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('i,aji,jk,lb,la,kb', L_b, R_aba, S_oo_ab, S_ov_ab, t1_1_a, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,ji,lb,la,kb', L_b, R_aba, S_oo_ab, S_ov_ab, t1_1_a, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('i,aij,kl,kb,la,jb', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
                # block IjdB & AblJ
                S2 -= 2 * lib.einsum('i,aij,kb,cl,jc,klab', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,aij,kb,cl,ka,jlcb', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,aij,kb,cl,kc,jlab', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,ib,cl,jc,klab', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,aij,kb,cl,lb,jkac', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_a,
                    optimize=True)
                S2 += lib.einsum('i,ajk,ib,cl,lb,jkac', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_a, optimize=True)
                S2 -= 2 * lib.einsum('i,aji,kb,cl,kc,jlab', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_a, t2_1_b, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,ib,cl,kc,jlab', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_a, t2_1_b, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kb,cl,jb,klca', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kb,cl,la,kjcb', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,aji,kb,cl,lb,kjca', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,ib,cl,jb,klca', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,ib,cl,la,kjcb', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,ib,cl,lb,kjca', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kb,cl,jc,klab', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kb,cl,ka,jlcb', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,aji,kb,cl,kc,jlab', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,lb,ci,jc,lkab', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,lb,ci,la,jkcb', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,lb,ci,lc,jkab', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,aji,kb,cl,lb,jkac', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_b, t2_1_a, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,lb,ci,kb,jlac', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_b, t2_1_a, optimize=True)
                S2 += 2 * lib.einsum('i,aij,kb,cl,kc,jlab', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('i,ajk,lb,ci,lc,jkab', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_b, optimize=True)
                S2 -= 2 * lib.einsum('i,aij,kb,cl,jb,klca', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,aij,kb,cl,la,kjcb', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,aij,kb,cl,lb,kjca', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,lb,ci,jb,lkca', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                # block IblJ & IjdJ
                S2 -= 2 * lib.einsum('i,aij,kl,jb,ka,lb', L_a, R_aaa_u, S_oo_ab, S_ov_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kj,la,kb,lb', L_a, R_bab, S_oo_ab, S_ov_ab, t1_1_a, t1_1_a, optimize=True)
                S2 -= lib.einsum('i,ajk,ij,la,kb,lb', L_a, R_bab, S_oo_ab, S_ov_ab, t1_1_a, t1_1_a, optimize=True)
                S2 -= lib.einsum('i,ajk,lj,ka,ib,lb', L_a, R_bab, S_oo_ab, S_ov_ab, t1_1_a, t1_1_a, optimize=True)
                S2 += lib.einsum('i,aji,kj,kb,la,lb', L_a, R_bab, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
                S2 += lib.einsum('i,aji,kl,ka,jb,lb', L_a, R_bab, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
                S2 -= lib.einsum('i,ajk,ij,kb,la,lb', L_a, R_bab, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
                S2 -= lib.einsum('i,ajk,il,ka,jb,lb', L_a, R_bab, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kl,jb,ka,lb', L_b, R_aba, S_oo_ab, S_ov_ab, t1_1_a, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,lk,jb,la,ib', L_b, R_aba, S_oo_ab, S_ov_ab, t1_1_a, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('i,aij,kj,la,kb,lb', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_a, t1_1_a, optimize=True)
                S2 -= lib.einsum('i,aij,kj,kb,la,lb', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
                S2 -= lib.einsum('i,aij,kl,ka,jb,lb', L_b, R_bbb_u, S_oo_ab, S_ov_ab, t1_1_b, t1_1_b, optimize=True)
                # block IblB & AjdJ
                S2 += 2 * lib.einsum('i,aij,kl,bc,lc,jkab', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_a,
                    optimize=True)
                S2 += lib.einsum('i,ajk,il,bc,lc,jkab', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_a, optimize=True)
                S2 -= lib.einsum('i,aji,kj,ba,lc,klbc', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_a, t2_1_a, optimize=True)
                S2 += lib.einsum('i,ajk,ij,ba,lc,klbc', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_a, t2_1_a, optimize=True)
                S2 -= lib.einsum('i,ajk,lj,ba,ic,klbc', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_a, t2_1_a, optimize=True)
                S2 -= lib.einsum('i,aji,kj,ba,lc,klbc', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kj,bc,lc,klba', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kl,ba,lc,kjbc', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,aji,kl,bc,lc,kjba', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += lib.einsum('i,ajk,ij,ba,lc,klbc', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,ij,bc,lc,klba', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,il,ba,lc,kjbc', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,il,bc,lc,kjba', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,aji,kl,bc,lc,jkab', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_b, t2_1_a, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,lk,bc,ic,jlab', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_b, t2_1_a, optimize=True)
                S2 += lib.einsum('i,aij,kj,ba,lc,klbc', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_a, optimize=True)
                S2 += lib.einsum('i,aij,kj,ba,lc,klbc', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,aij,kj,bc,lc,klba', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,aij,kl,ba,lc,kjbc', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,aij,kl,bc,lc,kjba', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,ajk,lj,ba,ic,lkbc', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,lj,bc,ic,lkba', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_ab,
                    optimize=True)
                # block IbdJ
                S2 += 2 * lib.einsum('i,aij,kb,jkac,lb,lc', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,aij,kb,lkac,jb,lc', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,ajk,lb,jlac,ic,kb', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 -= lib.einsum('i,aji,kb,kjbc,la,lc', L_a, R_bab, t1_1_a, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kb,ljbc,ka,lc', L_a, R_bab, t1_1_a, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 += lib.einsum('i,ajk,ib,kjbc,la,lc', L_a, R_bab, t1_1_a, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= lib.einsum('i,ajk,ib,ljbc,ka,lc', L_a, R_bab, t1_1_a, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,lb,kjbc,ic,la', L_a, R_bab, t1_1_a, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 += lib.einsum('i,ajk,lb,ljbc,ic,ka', L_a, R_bab, t1_1_a, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,aji,kb,jkac,lb,lc', L_a, R_bab, t1_1_b, t2_1_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 += lib.einsum('i,aji,kb,jkbc,la,lc', L_a, R_bab, t1_1_b, t2_1_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,lb,jlac,ic,kb', L_a, R_bab, t1_1_b, t2_1_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= lib.einsum('i,ajk,lb,jlbc,ic,ka', L_a, R_bab, t1_1_b, t2_1_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,aji,kb,jkac,lb,lc', L_b, R_aba, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kb,lkac,jb,lc', L_b, R_aba, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,ib,jkac,lb,lc', L_b, R_aba, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,ib,lkac,jb,lc', L_b, R_aba, t1_1_b, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 += lib.einsum('i,aij,kb,kjbc,la,lc', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,aij,kb,ljbc,ka,lc', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,aij,kb,jkac,lb,lc', L_b, R_bbb_u, t1_1_b, t2_1_b, S_ov_ab, S_ov_ab,
                    optimize=True)
                S2 -= lib.einsum('i,aij,kb,jkbc,la,lc', L_b, R_bbb_u, t1_1_b, t2_1_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 += lib.einsum('i,ajk,ib,jkac,lb,lc', L_b, R_bbb_u, t1_1_b, t2_1_b, S_ov_ab, S_ov_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('i,ajk,ib,jkbc,la,lc', L_b, R_bbb_u, t1_1_b, t2_1_b, S_ov_ab, S_ov_ab,
                    optimize=True)
                # block IbdB & AbdJ
                S2 -= 2 * lib.einsum('i,aji,kb,ca,kc,jb', L_a, R_bab, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,ib,ca,kc,jb', L_a, R_bab, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('i,aij,kb,ca,kc,jb', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                # block AjlJ & IjlB
                S2 -= lib.einsum('i,aij,jk,bk,la,lb', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
                S2 -= lib.einsum('i,aij,kl,al,jb,kb', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
                S2 -= 2 * lib.einsum('i,aij,jk,al,kb,lb', L_a, R_aaa_u, S_oo_ab, S_vo_ab, t1_1_b, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kl,bj,kb,la', L_a, R_bab, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,kl,bj,ib,la', L_a, R_bab, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += lib.einsum('i,aji,jk,bk,la,lb', L_b, R_aba, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
                S2 += lib.einsum('i,aji,kl,al,jb,kb', L_b, R_aba, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
                S2 -= lib.einsum('i,ajk,ji,bk,la,lb', L_b, R_aba, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
                S2 -= lib.einsum('i,ajk,li,ak,jb,lb', L_b, R_aba, S_oo_ab, S_vo_ab, t1_1_a, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('i,aji,jk,al,kb,lb', L_b, R_aba, S_oo_ab, S_vo_ab, t1_1_b, t1_1_b, optimize=True)
                S2 -= lib.einsum('i,ajk,ji,al,kb,lb', L_b, R_aba, S_oo_ab, S_vo_ab, t1_1_b, t1_1_b, optimize=True)
                S2 -= lib.einsum('i,ajk,jl,ak,ib,lb', L_b, R_aba, S_oo_ab, S_vo_ab, t1_1_b, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('i,aij,kl,bj,kb,la', L_b, R_bbb_u, S_oo_ab, S_vo_ab, t1_1_a, t1_1_b, optimize=True)
                # block AjlB
                S2 += 2 * lib.einsum('i,aij,kb,jkac,bl,cl', L_a, R_aaa_u, t1_1_a, t2_1_a, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 -= lib.einsum('i,aij,kb,jkbc,al,cl', L_a, R_aaa_u, t1_1_a, t2_1_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 += lib.einsum('i,ajk,ib,jkac,bl,cl', L_a, R_aaa_u, t1_1_a, t2_1_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= 1/2 * lib.einsum('i,ajk,ib,jkbc,al,cl', L_a, R_aaa_u, t1_1_a, t2_1_a, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += lib.einsum('i,aij,kb,jkcb,al,cl', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,aij,kb,jlcb,ak,cl', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,aji,kb,kjca,bl,cl', L_a, R_bab, t1_1_a, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kb,klca,bj,cl', L_a, R_bab, t1_1_a, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,ib,kjca,bl,cl', L_a, R_bab, t1_1_a, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,ib,klca,bj,cl', L_a, R_bab, t1_1_a, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,aji,kb,jkac,bl,cl', L_b, R_aba, t1_1_a, t2_1_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 += lib.einsum('i,aji,kb,jkbc,al,cl', L_b, R_aba, t1_1_a, t2_1_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,lb,jlac,ci,bk', L_b, R_aba, t1_1_a, t2_1_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= lib.einsum('i,ajk,lb,jlbc,ci,ak', L_b, R_aba, t1_1_a, t2_1_a, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= lib.einsum('i,aji,kb,jkcb,al,cl', L_b, R_aba, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kb,jlcb,ak,cl', L_b, R_aba, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 += lib.einsum('i,ajk,ib,jkcb,al,cl', L_b, R_aba, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= lib.einsum('i,ajk,ib,jlcb,ak,cl', L_b, R_aba, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,lb,jkcb,ci,al', L_b, R_aba, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 += lib.einsum('i,ajk,lb,jlcb,ci,ak', L_b, R_aba, t1_1_b, t2_1_ab, S_vo_ab, S_vo_ab, optimize=True)
                S2 += 2 * lib.einsum('i,aij,kb,kjca,bl,cl', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,aij,kb,klca,bj,cl', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,ajk,lb,ljca,ci,bk', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_vo_ab, S_vo_ab,
                    optimize=True)
                # block AjdJ & IblB
                S2 += lib.einsum('i,aij,jk,ab,lc,lkcb', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,aij,jk,bc,lb,lkac', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,aij,kl,ab,kc,jlcb', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 += 2 * lib.einsum('i,aij,kl,bc,kb,jlac', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,ajk,jl,ab,ic,klcb', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,jl,bc,ib,klac', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab,
                    optimize=True)
                S2 += lib.einsum('i,aij,jk,ab,lc,klbc', L_a, R_aaa_u, S_oo_ab, S_vv_ab, t1_1_b, t2_1_b, optimize=True)
                S2 -= 2 * lib.einsum('i,aji,kl,bc,kb,jlac', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_a, t2_1_b, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,kl,bc,ib,jlac', L_a, R_bab, S_oo_ab, S_vv_ab, t1_1_a, t2_1_b, optimize=True)
                S2 -= lib.einsum('i,aji,jk,ab,lc,lkcb', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += 2 * lib.einsum('i,aji,jk,bc,lb,lkac', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += 2 * lib.einsum('i,aji,kl,ab,kc,jlcb', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,aji,kl,bc,kb,jlac', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += lib.einsum('i,ajk,ji,ab,lc,lkcb', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,ji,bc,lb,lkac', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,ajk,li,ab,lc,jkcb', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,li,bc,lb,jkac', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= lib.einsum('i,aji,jk,ab,lc,klbc', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_b, t2_1_b, optimize=True)
                S2 += lib.einsum('i,ajk,ji,ab,lc,klbc', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_b, t2_1_b, optimize=True)
                S2 -= lib.einsum('i,ajk,jl,ab,ic,klbc', L_b, R_aba, S_oo_ab, S_vv_ab, t1_1_b, t2_1_b, optimize=True)
                S2 += 2 * lib.einsum('i,aij,kl,bc,kb,jlac', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_b,
                    optimize=True)
                S2 += lib.einsum('i,ajk,li,bc,lb,jkac', L_b, R_bbb_u, S_oo_ab, S_vv_ab, t1_1_a, t2_1_b, optimize=True)
                # block AjdB & AblB
                S2 += 2 * lib.einsum('i,aij,bk,ac,jb,kc', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('i,aji,bk,ac,jb,kc', L_b, R_aba, S_vo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,bi,ac,jb,kc', L_b, R_aba, S_vo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                # block AblJ & IjdB
                S2 += lib.einsum('i,aij,jb,ak,lc,lkcb', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= lib.einsum('i,aij,jb,ck,la,lkcb', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= lib.einsum('i,aij,kb,al,jc,klcb', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += lib.einsum('i,ajk,jb,al,kc,ilcb', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += lib.einsum('i,aij,jb,ak,lc,klbc', L_a, R_aaa_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_b, optimize=True)
                S2 -= lib.einsum('i,aji,ka,bj,lc,klbc', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_a, t2_1_a, optimize=True)
                S2 += lib.einsum('i,ajk,ka,bj,lc,ilbc', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_a, t2_1_a, optimize=True)
                S2 -= lib.einsum('i,ajk,la,bj,kc,ilbc', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_a, t2_1_a, optimize=True)
                S2 -= lib.einsum('i,aji,ka,bj,lc,klbc', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += lib.einsum('i,aji,ka,bl,jc,klbc', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += lib.einsum('i,aji,kb,cj,la,klcb', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += lib.einsum('i,ajk,ka,bj,lc,ilbc', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= lib.einsum('i,ajk,ka,bl,jc,ilbc', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= lib.einsum('i,ajk,kb,cj,la,ilcb', L_a, R_bab, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= lib.einsum('i,aji,jb,ak,lc,lkcb', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += lib.einsum('i,aji,jb,ck,la,lkcb', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += lib.einsum('i,aji,kb,al,jc,klcb', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 += lib.einsum('i,ajk,jb,ak,lc,licb', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= lib.einsum('i,ajk,jb,ck,la,licb', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= lib.einsum('i,ajk,lb,ak,jc,licb', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_a, t2_1_ab, optimize=True)
                S2 -= lib.einsum('i,aji,jb,ak,lc,klbc', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_b, t2_1_b, optimize=True)
                S2 += lib.einsum('i,ajk,jb,ak,lc,ilbc', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_b, t2_1_b, optimize=True)
                S2 -= lib.einsum('i,ajk,jb,al,kc,ilbc', L_b, R_aba, S_ov_ab, S_vo_ab, t1_1_b, t2_1_b, optimize=True)
                S2 += lib.einsum('i,aij,ka,bj,lc,klbc', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_a, t2_1_a, optimize=True)
                S2 += lib.einsum('i,aij,ka,bj,lc,klbc', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= lib.einsum('i,aij,ka,bl,jc,klbc', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 -= lib.einsum('i,aij,kb,cj,la,klcb', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                S2 += lib.einsum('i,ajk,la,bj,kc,libc', L_b, R_bbb_u, S_ov_ab, S_vo_ab, t1_1_b, t2_1_ab, optimize=True)
                # block AblB & AjdB
                S2 += 2 * lib.einsum('i,aij,ak,bc,jb,kc', L_a, R_aaa_u, S_vo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('i,aji,bj,ca,kb,kc', L_a, R_bab, S_vo_ab, S_vv_ab, t1_1_a, t1_1_a, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,bj,ca,ib,kc', L_a, R_bab, S_vo_ab, S_vv_ab, t1_1_a, t1_1_a, optimize=True)
                S2 -= 2 * lib.einsum('i,aji,ak,bc,jb,kc', L_b, R_aba, S_vo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,ak,bc,jb,ic', L_b, R_aba, S_vo_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('i,aij,bj,ca,kb,kc', L_b, R_bbb_u, S_vo_ab, S_vv_ab, t1_1_a, t1_1_a, optimize=True)
                # block AbdJ & IbdB
                S2 += 2 * lib.einsum('i,aij,jb,ac,kb,kc', L_a, R_aaa_u, S_ov_ab, S_vv_ab, t1_1_b, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('i,aji,ka,bc,kb,jc', L_a, R_bab, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,ka,bc,ib,jc', L_a, R_bab, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                S2 -= 2 * lib.einsum('i,aji,jb,ac,kb,kc', L_b, R_aba, S_ov_ab, S_vv_ab, t1_1_b, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,jb,ac,ib,kc', L_b, R_aba, S_ov_ab, S_vv_ab, t1_1_b, t1_1_b, optimize=True)
                S2 += 2 * lib.einsum('i,aij,ka,bc,kb,jc', L_b, R_bbb_u, S_ov_ab, S_vv_ab, t1_1_a, t1_1_b, optimize=True)
                # block AbdB
                S2 += 2 * lib.einsum('i,aij,kb,jkcd,ad,cb', L_a, R_aaa_u, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)
                S2 -= 2 * lib.einsum('i,aji,kb,kjcd,ca,bd', L_a, R_bab, t1_1_a, t2_1_ab, S_vv_ab, S_vv_ab, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,ib,kjcd,ca,bd', L_a, R_bab, t1_1_a, t2_1_ab, S_vv_ab, S_vv_ab, optimize=True)
                S2 -= 2 * lib.einsum('i,aji,kb,jkcd,ad,cb', L_b, R_aba, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab, optimize=True)
                S2 += 2 * lib.einsum('i,ajk,ib,jkcd,ad,cb', L_b, R_aba, t1_1_b, t2_1_ab, S_vv_ab, S_vv_ab, optimize=True)
                S2 += 2 * lib.einsum('i,aij,kb,kjcd,ca,bd', L_b, R_bbb_u, t1_1_a, t2_1_ab, S_vv_ab, S_vv_ab,
                    optimize=True)

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
        del w_t2t2_bc, w_t2t2_ac, w_t2t2_bc_2, w_t2t2_ac_2

    return spin, (trace_a, trace_b)


class UADCIP(uadc.UADC):
    '''unrestricted ADC for IP energies and spectroscopic amplitudes

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
            >>> myadcip = adc.UADC(myadc).run()

    Saved results

        e_ip : float or list of floats
            IP energy (eigenvalue). For nroots = 1, it is a single float
            number. If nroots > 1, it is a list of floats for the lowest
            nroots eigenvalues.
        v_ip : array
            Eigenvectors for each IP transition.
        p_ip : float
            Spectroscopic amplitudes for each IP transition.
    '''

    _keys = {
        'tol_residual','conv_tol', 'e_corr', 'method',
        'method_type', 'mo_coeff', 'mo_energy_b', 'max_memory',
        't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle',
        'nocc_a', 'nocc_b', 'nvir_a', 'nvir_b', 'mo_coeff_hf',
        'nmo_a', 'nmo_b', 'mol', 'transform_integrals',
        'with_df', 'spec_factor_print_tol', 'evec_print_tol',
        'compute_properties', 'approx_trans_moments', 'E', 'U', 'P', 'X',
        'compute_spin_square', '_make_rdm1', 'mo_occ', 'if_naf', 'naux', 'f_ov'
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
        self.method_type = "ip"
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
        self.frozen = adc.frozen

        self._adc_es = self

    kernel = uadc.kernel
    get_imds = get_imds
    get_diag = get_diag
    matvec = matvec
    get_trans_moments = get_trans_moments
    get_properties = get_properties

    analyze_spec_factor = analyze_spec_factor
    analyze_eigenvector = analyze_eigenvector
    analyze = analyze
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
            n_singles_a = nocc_a
            n_singles_b = nocc_b
            n_doubles_aaa = nocc_a* (nocc_a - 1) * nvir_a // 2
            n_doubles_bab = nvir_b * nocc_a* nocc_b
            n_doubles_aba = nvir_a * nocc_b* nocc_a
            n_doubles_bbb = nocc_b* (nocc_b - 1) * nvir_b // 2

            dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb
            if isinstance(ini, list):
                g = np.array(ini)
            else:
                g = ini
            if g.shape[0] != dim or g.shape[1] != nroots:
                if self.frozen is None:
                    raise ValueError(f"Shape of guess should be ({dim},{nroots})")
                else:
                    g = self.fro_guess(g,self.frozen)
                    if (g.shape[0] != dim or g.shape[1] != nroots):
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

    def fro_guess(self,ini,frozen):
        nocc = self._scf.nelec
        nocc_a = nocc[0]
        nocc_b = nocc[1]
        if frozen[0] is None:
            pass
        elif isinstance(frozen[0], (int, np.integer)):
            nvir_a = self.nvir_a
            olist_a = list(range(frozen[0],nocc_a))
            vlist_a = list(range(nvir_a))
        elif hasattr(frozen[0], '__len__'):
            nvir_a = self._nmo[0] + len(frozen[0]) - nocc_a
            vlist_a = list(range(nvir_a))
            olist_a = list(range(nocc_a))
        for n in frozen[0]:
            if n < nocc_a:
                olist_a.remove(n)
            else:
                vlist_a.remove(n-nocc_a)

        if frozen[1] is None:
            pass
        elif isinstance(frozen[1], (int, np.integer)):
            nvir_b = self.nvir_b
            olist_b = list(range(frozen[1],nocc_b))
            vlist_b = list(range(self.nvir_b))
        elif hasattr(frozen[1], '__len__'):
            nvir_b = self._nmo[1] + len(frozen[1]) - nocc_b
            vlist_b = list(range(nvir_b))
            olist_b = list(range(nocc_b))
        for n in frozen[1]:
            if n < nocc_b:
                olist_b.remove(n)
            else:
                vlist_b.remove(n-nocc_b)

        sidx_a = np.zeros(nocc_a, dtype=bool)
        sidx_b = np.zeros(nocc_b, dtype=bool)
        didx_aaa = np.zeros((nvir_a, nocc_a, nocc_a), dtype=bool)
        didx_bab = np.zeros((nvir_b, nocc_a, nocc_b), dtype=bool)
        didx_aba = np.zeros((nvir_a, nocc_b, nocc_a), dtype=bool)
        didx_bbb = np.zeros((nvir_b, nocc_b, nocc_b), dtype=bool)
        for n, i in enumerate(olist_a):
            sidx_a[i] = True
            for j in vlist_a:
                for k in olist_a[n:]:
                    didx_aaa[j][i][k] = True
            for j in vlist_b:
                for k in olist_b:
                    didx_bab[j][i][k] = True
        for n, i in enumerate(olist_b):
            sidx_b[i] = True
            for j in vlist_b:
                for k in olist_b[n:]:
                    didx_bbb[j][i][k] = True
            for j in vlist_a:
                for k in olist_a:
                    didx_aba[j][i][k] = True
        mask_a = np.triu(np.ones(nocc_a, dtype=bool), k=1)
        mask_b = np.triu(np.ones(nocc_a, dtype=bool), k=1)
        flat_mask_a = mask_a.ravel()
        flat_mask_b = mask_b.ravel()
        didx_aaa = didx_aaa.reshape(nvir_a, -1)[:, flat_mask_a].ravel()
        didx_bbb = didx_bbb.reshape(nvir_b, -1)[:, flat_mask_b].ravel()
        ini = ini[np.concatenate((sidx_a, sidx_b, didx_aaa, didx_bab.ravel(), didx_aba.ravel(), didx_bbb))]
        return ini
