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
Restricted algebraic diagrammatic construction
'''
import numpy as np
import pyscf.ao2mo as ao2mo
from pyscf import lib
from pyscf.lib import logger
from pyscf.adc import radc_ao2mo
from pyscf.adc import dfadc
from pyscf import __config__
from pyscf import df
from pyscf import symm
import h5py
import tempfile


def compute_amplitudes_energy(myadc, eris, verbose=None):

    t1, t2, myadc.imds.t2_1_vvvv = myadc.compute_amplitudes(eris)
    e_corr = myadc.compute_energy(t2, eris)

    return e_corr, t1, t2


def compute_amplitudes(myadc, eris):

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(myadc.stdout, myadc.verbose)

    if myadc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(myadc.method)

    nocc = myadc._nocc
    nvir = myadc._nvir

    eris_oooo = eris.oooo
    eris_ovoo = eris.ovoo
    eris_oovv = eris.oovv
    eris_ovvo = eris.ovvo

    # Compute first-order doubles t2 (tijab)

    v2e_oovv = eris_ovvo[:].transpose(0,3,1,2).copy()

    e = myadc.mo_energy
    d_ij = e[:nocc][:,None] + e[:nocc]
    d_ab = e[nocc:][:,None] + e[nocc:]

    D2 = d_ij.reshape(-1,1) - d_ab.reshape(-1)
    D2 = D2.reshape((nocc,nocc,nvir,nvir))

    D1 = e[:nocc][:None].reshape(-1,1) - e[nocc:].reshape(-1)
    D1 = D1.reshape((nocc,nvir))

    t2_1 = v2e_oovv/D2
    h5cache_t2 = _create_t2_h5cache()
    if not isinstance(eris.oooo, np.ndarray):
        t2_1 = h5cache_t2.create_dataset('t2_1', data=t2_1)

    del v2e_oovv
    del D2

    cput0 = log.timer_debug1("Completed t2_1 amplitude calculation", *cput0)

    t1_2 = None

    if myadc.approx_trans_moments is False or myadc.method == "adc(3)":
        # Compute second-order singles t1 (tij)
        t1_2 = np.zeros((nocc,nvir))

        if isinstance(eris.ovvv, type(None)):
            chnk_size = radc_ao2mo.calculate_chunk_size(myadc)
            a = 0
            for p in range(0,nocc,chnk_size):
                eris_ovvv = dfadc.get_ovvv_df(
                    myadc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
                k = eris_ovvv.shape[0]
                t1_2 += 1.5*lib.einsum('kdac,ikcd->ia',eris_ovvv,t2_1[:,a:a+k],optimize=True)
                t1_2 -= 0.5*lib.einsum('kdac,kicd->ia',eris_ovvv,t2_1[a:a+k,:],optimize=True)
                t1_2 -= 0.5*lib.einsum('kcad,ikcd->ia',eris_ovvv,t2_1[:,a:a+k],optimize=True)
                t1_2 += 0.5*lib.einsum('kcad,kicd->ia',eris_ovvv,t2_1[a:a+k,:],optimize=True)

                del eris_ovvv
                a += k
        else :
            eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
            t1_2 += 1.5*lib.einsum('kdac,ikcd->ia',eris_ovvv,t2_1,optimize=True)
            t1_2 -= 0.5*lib.einsum('kdac,kicd->ia',eris_ovvv,t2_1,optimize=True)
            t1_2 -= 0.5*lib.einsum('kcad,ikcd->ia',eris_ovvv,t2_1,optimize=True)
            t1_2 += 0.5*lib.einsum('kcad,kicd->ia',eris_ovvv,t2_1,optimize=True)
            del eris_ovvv

        t1_2 -= 1.5*lib.einsum('lcki,klac->ia',eris_ovoo,t2_1[:],optimize=True)
        t1_2 += 0.5*lib.einsum('lcki,lkac->ia',eris_ovoo,t2_1[:],optimize=True)
        t1_2 -= 0.5*lib.einsum('kcli,lkac->ia',eris_ovoo,t2_1[:],optimize=True)
        t1_2 += 0.5*lib.einsum('kcli,klac->ia',eris_ovoo,t2_1[:],optimize=True)

        t1_2 = t1_2/D1

        cput0 = log.timer_debug1("Completed t1_2 amplitude calculation", *cput0)

    t2_2 = None
    t1_3 = None
    t2_1_vvvv = None

    if (myadc.method == "adc(2)-x" and myadc.approx_trans_moments is False) or (myadc.method == "adc(3)"):

        # Compute second-order doubles t2 (tijab)

        eris_oooo = eris.oooo
        eris_ovvo = eris.ovvo

        if isinstance(eris.vvvv, np.ndarray):
            eris_vvvv = eris.vvvv
            temp = t2_1.reshape(nocc*nocc,nvir*nvir)
            t2_1_vvvv = np.dot(temp,eris_vvvv.T).reshape(nocc,nocc,nvir,nvir)
        elif isinstance(eris.vvvv, list):
            t2_1_vvvv = contract_ladder(myadc,t2_1[:],eris.vvvv)
        else:
            t2_1_vvvv = contract_ladder(myadc,t2_1[:],eris.Lvv)

        if not isinstance(eris.oooo, np.ndarray):
            t2_1_vvvv = h5cache_t2.create_dataset('t2_1_vvvv', data=t2_1_vvvv)

        t2_2 = t2_1_vvvv[:].copy()

        t2_2 += lib.einsum('kilj,klab->ijab',eris_oooo,t2_1[:],optimize=True)
        t2_2 += 2 * lib.einsum('kcbj,kica->ijab',eris_ovvo,t2_1[:],optimize=True)
        t2_2 -= lib.einsum('kcbj,ikca->ijab',eris_ovvo,t2_1[:],optimize=True)
        t2_2 -= lib.einsum('kjbc,ikac->ijab',eris_oovv,t2_1[:],optimize=True)
        t2_2 -= lib.einsum('kibc,kjac->ijab',eris_oovv,t2_1[:],optimize=True)
        t2_2 -= lib.einsum('kjac,ikcb->ijab',eris_oovv,t2_1[:],optimize=True)
        t2_2 += 2 * lib.einsum('kcai,kjcb->ijab',eris_ovvo,t2_1[:],optimize=True)
        t2_2 -= lib.einsum('kcai,jkcb->ijab',eris_ovvo,t2_1[:],optimize=True)
        t2_2 -= lib.einsum('kiac,kjcb->ijab',eris_oovv,t2_1[:],optimize=True)

        D2 = d_ij.reshape(-1,1) - d_ab.reshape(-1)
        D2 = D2.reshape((nocc,nocc,nvir,nvir))

        t2_2 = t2_2/D2
        if not isinstance(eris.oooo, np.ndarray):
            t2_2 = h5cache_t2.create_dataset('t2_2', data=t2_2)
        del D2

        cput0 = log.timer_debug1("Completed t2_2 amplitude calculation", *cput0)

    if (myadc.method == "adc(3)" and myadc.approx_trans_moments is False):

        eris_ovoo = eris.ovoo

        t1_3 =  lib.einsum('d,ilad,ld->ia',e[nocc:],t2_1[:],t1_2,optimize=True)
        t1_3 -= lib.einsum('d,liad,ld->ia',e[nocc:],t2_1[:],t1_2,optimize=True)
        t1_3 += lib.einsum('d,ilad,ld->ia',e[nocc:],t2_1[:],t1_2,optimize=True)

        t1_3 -= lib.einsum('l,ilad,ld->ia',e[:nocc],t2_1[:], t1_2,optimize=True)
        t1_3 += lib.einsum('l,liad,ld->ia',e[:nocc],t2_1[:], t1_2,optimize=True)
        t1_3 -= lib.einsum('l,ilad,ld->ia',e[:nocc],t2_1[:],t1_2,optimize=True)

        t1_3 += 0.5*lib.einsum('a,ilad,ld->ia',e[nocc:],t2_1[:], t1_2,optimize=True)
        t1_3 -= 0.5*lib.einsum('a,liad,ld->ia',e[nocc:],t2_1[:], t1_2,optimize=True)
        t1_3 += 0.5*lib.einsum('a,ilad,ld->ia',e[nocc:],t2_1[:],t1_2,optimize=True)

        t1_3 -= 0.5*lib.einsum('i,ilad,ld->ia',e[:nocc],t2_1[:], t1_2,optimize=True)
        t1_3 += 0.5*lib.einsum('i,liad,ld->ia',e[:nocc],t2_1[:], t1_2,optimize=True)
        t1_3 -= 0.5*lib.einsum('i,ilad,ld->ia',e[:nocc],t2_1[:],t1_2,optimize=True)

        t1_3 += lib.einsum('ld,iadl->ia',t1_2,eris_ovvo,optimize=True)
        t1_3 -= lib.einsum('ld,ladi->ia',t1_2,eris_ovvo,optimize=True)
        t1_3 += lib.einsum('ld,iadl->ia',t1_2,eris_ovvo,optimize=True)

        t1_3 += lib.einsum('ld,ldai->ia',t1_2,eris_ovvo ,optimize=True)
        t1_3 -= lib.einsum('ld,liad->ia',t1_2,eris_oovv ,optimize=True)
        t1_3 += lib.einsum('ld,ldai->ia',t1_2,eris_ovvo,optimize=True)

        t1_3 -= 0.5*lib.einsum('lmad,mdli->ia',t2_2[:],eris_ovoo,optimize=True)
        t1_3 += 0.5*lib.einsum('mlad,mdli->ia',t2_2[:],eris_ovoo,optimize=True)
        t1_3 += 0.5*lib.einsum('lmad,ldmi->ia',t2_2[:],eris_ovoo,optimize=True)
        t1_3 -= 0.5*lib.einsum('mlad,ldmi->ia',t2_2[:],eris_ovoo,optimize=True)
        t1_3 -=     lib.einsum('lmad,mdli->ia',t2_2[:],eris_ovoo,optimize=True)

        if isinstance(eris.ovvv, type(None)):
            chnk_size = radc_ao2mo.calculate_chunk_size(myadc)
            a = 0
            for p in range(0,nocc,chnk_size):
                eris_ovvv = dfadc.get_ovvv_df(
                    myadc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
                k = eris_ovvv.shape[0]

                t1_3 += 0.5*lib.einsum('ilde,lead->ia', t2_2[:,a:a+k],eris_ovvv,optimize=True)
                t1_3 -= 0.5*lib.einsum('lide,lead->ia', t2_2[a:a+k],eris_ovvv,optimize=True)

                t1_3 -= 0.5*lib.einsum('ilde,ldae->ia', t2_2[:,a:a+k],eris_ovvv,optimize=True)
                t1_3 += 0.5*lib.einsum('lide,ldae->ia', t2_2[a:a+k],eris_ovvv,optimize=True)

                t1_3 -= lib.einsum('ildf,mefa,lmde->ia',
                                   t2_1[:], eris_ovvv,  t2_1[:,a:a+k] ,optimize=True)
                t1_3 += lib.einsum('ildf,mefa,mlde->ia',
                                   t2_1[:], eris_ovvv,  t2_1[a:a+k] ,optimize=True)
                t1_3 += lib.einsum('lidf,mefa,lmde->ia',
                                   t2_1[:], eris_ovvv,  t2_1[:,a:a+k] ,optimize=True)
                t1_3 -= lib.einsum('lidf,mefa,mlde->ia',
                                   t2_1[:], eris_ovvv,  t2_1[a:a+k] ,optimize=True)

                t1_3 += lib.einsum('ildf,mafe,lmde->ia',
                                   t2_1[:], eris_ovvv,  t2_1[:,a:a+k] ,optimize=True)
                t1_3 -= lib.einsum('ildf,mafe,mlde->ia',
                                   t2_1[:], eris_ovvv,  t2_1[a:a+k] ,optimize=True)
                t1_3 -= lib.einsum('lidf,mafe,lmde->ia',
                                   t2_1[:], eris_ovvv,  t2_1[:,a:a+k] ,optimize=True)
                t1_3 += lib.einsum('lidf,mafe,mlde->ia',
                                   t2_1[:], eris_ovvv,  t2_1[a:a+k] ,optimize=True)

                t1_3 += lib.einsum('ilfd,mefa,mled->ia',
                                   t2_1[:],eris_ovvv, t2_1[a:a+k],optimize=True)
                t1_3 -= lib.einsum('ilfd,mafe,mled->ia',
                                   t2_1[:],eris_ovvv, t2_1[a:a+k],optimize=True)

                t1_3 += 0.5*lib.einsum('ilaf,mefd,lmde->ia',
                                       t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
                t1_3 -= 0.5*lib.einsum('ilaf,mefd,mlde->ia',
                                       t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)
                t1_3 -= 0.5*lib.einsum('liaf,mefd,lmde->ia',
                                       t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
                t1_3 += 0.5*lib.einsum('liaf,mefd,mlde->ia',
                                       t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)

                t1_3 -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',
                                       t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
                t1_3 += 0.5*lib.einsum('ilaf,mdfe,mlde->ia',
                                       t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)
                t1_3 += 0.5*lib.einsum('liaf,mdfe,lmde->ia',
                                       t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
                t1_3 -= 0.5*lib.einsum('liaf,mdfe,mlde->ia',
                                       t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)

                t1_3[a:a+k] += 0.5*lib.einsum('lmdf,iaef,lmde->ia',
                                              t2_1[:],eris_ovvv,t2_1[:],optimize=True)
                t1_3[a:a+k] -= 0.5*lib.einsum('lmdf,iaef,mlde->ia',
                                              t2_1[:],eris_ovvv,t2_1[:],optimize=True)
                t1_3[a:a+k] -= 0.5*lib.einsum('mldf,iaef,lmde->ia',
                                              t2_1[:],eris_ovvv,t2_1[:],optimize=True)
                t1_3[a:a+k] += 0.5*lib.einsum('mldf,iaef,mlde->ia',
                                              t2_1[:],eris_ovvv,t2_1[:],optimize=True)

                t1_3[a:a+k] -= 0.5*lib.einsum('lmdf,ifea,lmde->ia',
                                              t2_1[:],eris_ovvv,t2_1[:],optimize=True)
                t1_3[a:a+k] += 0.5*lib.einsum('lmdf,ifea,mlde->ia',
                                              t2_1[:],eris_ovvv,t2_1[:],optimize=True)
                t1_3[a:a+k] += 0.5*lib.einsum('mldf,ifea,lmde->ia',
                                              t2_1[:],eris_ovvv,t2_1[:],optimize=True)
                t1_3[a:a+k] -= 0.5*lib.einsum('mldf,ifea,mlde->ia',
                                              t2_1[:],eris_ovvv,t2_1[:],optimize=True)

                t1_3[a:a+k] += lib.einsum('mlfd,iaef,mled->ia',t2_1[:],
                                          eris_ovvv,t2_1[:],optimize=True)
                t1_3[a:a+k] -= lib.einsum('mlfd,ifea,mled->ia',t2_1[:],
                                          eris_ovvv,t2_1[:],optimize=True)

                t1_3[a:a+k] -= 0.25*lib.einsum('lmef,iedf,lmad->ia',
                                               t2_1[:],eris_ovvv,t2_1[:],optimize=True)
                t1_3[a:a+k] += 0.25*lib.einsum('lmef,iedf,mlad->ia',
                                               t2_1[:],eris_ovvv,t2_1[:],optimize=True)
                t1_3[a:a+k] += 0.25*lib.einsum('mlef,iedf,lmad->ia',
                                               t2_1[:],eris_ovvv,t2_1[:],optimize=True)
                t1_3[a:a+k] -= 0.25*lib.einsum('mlef,iedf,mlad->ia',
                                               t2_1[:],eris_ovvv,t2_1[:],optimize=True)

                t1_3[a:a+k] += 0.25*lib.einsum('lmef,ifde,lmad->ia',
                                               t2_1[:],eris_ovvv,t2_1[:],optimize=True)
                t1_3[a:a+k] -= 0.25*lib.einsum('lmef,ifde,mlad->ia',
                                               t2_1[:],eris_ovvv,t2_1[:],optimize=True)
                t1_3[a:a+k] -= 0.25*lib.einsum('mlef,ifde,lmad->ia',
                                               t2_1[:],eris_ovvv,t2_1[:],optimize=True)
                t1_3[a:a+k] += 0.25*lib.einsum('mlef,ifde,mlad->ia',
                                               t2_1[:],eris_ovvv,t2_1[:],optimize=True)

                t1_3 += 0.5*lib.einsum('ilaf,mefd,lmde->ia',
                                       t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
                t1_3 -= 0.5*lib.einsum('ilaf,mefd,mlde->ia',
                                       t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)

                t1_3 -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',
                                       t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
                t1_3 += 0.5*lib.einsum('ilaf,mdfe,mlde->ia',
                                       t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)

                t1_3 -= lib.einsum('ildf,mafe,mlde->ia',t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)
                t1_3 += lib.einsum('ilaf,mefd,mled->ia',t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)

                t1_3[a:a+k] += 0.5*lib.einsum('lmdf,iaef,lmde->ia',
                                              t2_1[:],eris_ovvv,t2_1[:],optimize=True)
                t1_3[a:a+k] -= 0.5*lib.einsum('lmdf,iaef,mlde->ia',
                                              t2_1[:],eris_ovvv,t2_1[:],optimize=True)
                t1_3[a:a+k] -= 0.5*lib.einsum('mldf,iaef,lmde->ia',
                                              t2_1[:],eris_ovvv,t2_1[:],optimize=True)
                t1_3[a:a+k] += 0.5*lib.einsum('mldf,iaef,mlde->ia',
                                              t2_1[:],eris_ovvv,t2_1[:],optimize=True)

                t1_3[a:a+k] += lib.einsum('lmdf,iaef,lmde->ia',t2_1[:],
                                          eris_ovvv,t2_1[:],optimize=True)
                t1_3[a:a+k] -= lib.einsum('lmef,iedf,lmad->ia',t2_1[:],
                                          eris_ovvv,t2_1[:],optimize=True)

                t1_3 += lib.einsum('ilde,lead->ia',t2_2[:,a:a+k],eris_ovvv,optimize=True)

                t1_3 -= lib.einsum('ildf,mefa,lmde->ia',
                                   t2_1[:],eris_ovvv, t2_1[:,a:a+k],optimize=True)
                t1_3 += lib.einsum('lidf,mefa,lmde->ia',
                                   t2_1[:],eris_ovvv, t2_1[:,a:a+k],optimize=True)

                t1_3 += lib.einsum('ilfd,mefa,lmde->ia',
                                   t2_1[:],eris_ovvv,t2_1[:,a:a+k] ,optimize=True)
                t1_3 -= lib.einsum('ilfd,mefa,mlde->ia',
                                   t2_1[:],eris_ovvv,t2_1[a:a+k] ,optimize=True)

                t1_3 += lib.einsum('ilaf,mefd,lmde->ia',
                                   t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
                t1_3 -= lib.einsum('liaf,mefd,lmde->ia',
                                   t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)

                del eris_ovvv
                a += k

        else :
            eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)

            t1_3 += 0.5*lib.einsum('ilde,lead->ia', t2_2[:],eris_ovvv,optimize=True)
            t1_3 -= 0.5*lib.einsum('lide,lead->ia', t2_2[:],eris_ovvv,optimize=True)

            t1_3 -= 0.5*lib.einsum('ilde,ldae->ia', t2_2[:],eris_ovvv,optimize=True)
            t1_3 += 0.5*lib.einsum('lide,ldae->ia', t2_2[:],eris_ovvv,optimize=True)

            t1_3 -= lib.einsum('ildf,mefa,lmde->ia',t2_1[:], eris_ovvv,  t2_1[:] ,optimize=True)
            t1_3 += lib.einsum('ildf,mefa,mlde->ia',t2_1[:], eris_ovvv,  t2_1[:] ,optimize=True)
            t1_3 += lib.einsum('lidf,mefa,lmde->ia',t2_1[:], eris_ovvv,  t2_1[:] ,optimize=True)
            t1_3 -= lib.einsum('lidf,mefa,mlde->ia',t2_1[:], eris_ovvv,  t2_1[:] ,optimize=True)

            t1_3 += lib.einsum('ildf,mafe,lmde->ia',t2_1[:], eris_ovvv,  t2_1[:] ,optimize=True)
            t1_3 -= lib.einsum('ildf,mafe,mlde->ia',t2_1[:], eris_ovvv,  t2_1[:] ,optimize=True)
            t1_3 -= lib.einsum('lidf,mafe,lmde->ia',t2_1[:], eris_ovvv,  t2_1[:] ,optimize=True)
            t1_3 += lib.einsum('lidf,mafe,mlde->ia',t2_1[:], eris_ovvv,  t2_1[:] ,optimize=True)

            t1_3 += lib.einsum('ilfd,mefa,mled->ia',  t2_1[:],eris_ovvv, t2_1[:],optimize=True)
            t1_3 -= lib.einsum('ilfd,mafe,mled->ia',  t2_1[:],eris_ovvv, t2_1[:],optimize=True)

            t1_3 += 0.5*lib.einsum('ilaf,mefd,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 -= 0.5*lib.einsum('ilaf,mefd,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 -= 0.5*lib.einsum('liaf,mefd,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 += 0.5*lib.einsum('liaf,mefd,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)

            t1_3 -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 += 0.5*lib.einsum('ilaf,mdfe,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 += 0.5*lib.einsum('liaf,mdfe,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 -= 0.5*lib.einsum('liaf,mdfe,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)

            t1_3 += 0.5*lib.einsum('lmdf,iaef,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 -= 0.5*lib.einsum('lmdf,iaef,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 -= 0.5*lib.einsum('mldf,iaef,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 += 0.5*lib.einsum('mldf,iaef,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)

            t1_3 -= 0.5*lib.einsum('lmdf,ifea,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 += 0.5*lib.einsum('lmdf,ifea,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 += 0.5*lib.einsum('mldf,ifea,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 -= 0.5*lib.einsum('mldf,ifea,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)

            t1_3 += lib.einsum('mlfd,iaef,mled->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 -= lib.einsum('mlfd,ifea,mled->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)

            t1_3 -= 0.25*lib.einsum('lmef,iedf,lmad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 += 0.25*lib.einsum('lmef,iedf,mlad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 += 0.25*lib.einsum('mlef,iedf,lmad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 -= 0.25*lib.einsum('mlef,iedf,mlad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)

            t1_3 += 0.25*lib.einsum('lmef,ifde,lmad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 -= 0.25*lib.einsum('lmef,ifde,mlad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 -= 0.25*lib.einsum('mlef,ifde,lmad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 += 0.25*lib.einsum('mlef,ifde,mlad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)

            t1_3 += 0.5*lib.einsum('ilaf,mefd,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 -= 0.5*lib.einsum('ilaf,mefd,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)

            t1_3 -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 += 0.5*lib.einsum('ilaf,mdfe,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)

            t1_3 -= lib.einsum('ildf,mafe,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 += lib.einsum('ilaf,mefd,mled->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)

            t1_3 += 0.5*lib.einsum('lmdf,iaef,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 -= 0.5*lib.einsum('lmdf,iaef,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 -= 0.5*lib.einsum('mldf,iaef,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 += 0.5*lib.einsum('mldf,iaef,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)

            t1_3 += lib.einsum('lmdf,iaef,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 -= lib.einsum('lmef,iedf,lmad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)

            t1_3 += lib.einsum('ilde,lead->ia',t2_2[:],eris_ovvv,optimize=True)

            t1_3 -= lib.einsum('ildf,mefa,lmde->ia',t2_1[:],eris_ovvv, t2_1[:],optimize=True)
            t1_3 += lib.einsum('lidf,mefa,lmde->ia',t2_1[:],eris_ovvv, t2_1[:],optimize=True)

            t1_3 += lib.einsum('ilfd,mefa,lmde->ia',t2_1[:],eris_ovvv,t2_1[:] ,optimize=True)
            t1_3 -= lib.einsum('ilfd,mefa,mlde->ia',t2_1[:],eris_ovvv,t2_1[:] ,optimize=True)

            t1_3 += lib.einsum('ilaf,mefd,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3 -= lib.einsum('liaf,mefd,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)

            del eris_ovvv

        t1_3 += 0.25*lib.einsum('inde,lamn,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 -= 0.25*lib.einsum('inde,lamn,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 -= 0.25*lib.einsum('nide,lamn,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += 0.25*lib.einsum('nide,lamn,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 -= 0.25*lib.einsum('inde,maln,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += 0.25*lib.einsum('inde,maln,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += 0.25*lib.einsum('nide,maln,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 -= 0.25*lib.einsum('nide,maln,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += lib.einsum('inde,lamn,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)

        t1_3 += 0.5 * lib.einsum('inad,lemn,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 -= 0.5 * lib.einsum('inad,lemn,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 -= 0.5 * lib.einsum('niad,lemn,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += 0.5 * lib.einsum('niad,lemn,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)

        t1_3 -= 0.5 * lib.einsum('inad,meln,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += 0.5 * lib.einsum('inad,meln,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += 0.5 * lib.einsum('niad,meln,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 -= 0.5 * lib.einsum('niad,meln,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)

        t1_3 -= 0.5 * lib.einsum('inad,lemn,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += 0.5 * lib.einsum('niad,lemn,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)

        t1_3 -= 0.5 * lib.einsum('inad,meln,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += 0.5 * lib.einsum('niad,meln,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)

        t1_3 -= 0.5 * lib.einsum('inad,lemn,lmed->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 -= 0.5 * lib.einsum('inad,meln,mled->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)

        t1_3 += 0.5 * lib.einsum('inad,lemn,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 -= 0.5 * lib.einsum('inad,lemn,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)

        t1_3 -= 0.5 * lib.einsum('inad,meln,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += 0.5 * lib.einsum('inad,meln,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)

        t1_3 -= 0.5 * lib.einsum('lnde,ianm,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += 0.5 * lib.einsum('lnde,ianm,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += 0.5 * lib.einsum('nlde,ianm,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 -= 0.5 * lib.einsum('nlde,ianm,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)

        t1_3 += 0.5 * lib.einsum('lnde,naim,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 -= 0.5 * lib.einsum('lnde,naim,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 -= 0.5 * lib.einsum('nlde,naim,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += 0.5 * lib.einsum('nlde,naim,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)

        t1_3 -= lib.einsum('nled,ianm,mled->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += lib.einsum('nled,naim,mled->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)

        t1_3 -= 0.5*lib.einsum('lnde,ianm,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += 0.5*lib.einsum('lnde,ianm,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += 0.5*lib.einsum('nlde,ianm,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 -= 0.5*lib.einsum('nlde,ianm,mlde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)

        t1_3 -= lib.einsum('lnde,ianm,lmde->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)

        t1_3 -= lib.einsum('lnde,ienm,lmad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += lib.einsum('lnde,ienm,mlad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += lib.einsum('nlde,ienm,lmad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 -= lib.einsum('nlde,ienm,mlad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)

        t1_3 += lib.einsum('lnde,neim,lmad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 -= lib.einsum('lnde,neim,mlad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 -= lib.einsum('nlde,neim,lmad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += lib.einsum('nlde,neim,mlad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)

        t1_3 += lib.einsum('lnde,neim,lmad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 -= lib.einsum('lnde,neim,mlad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)

        t1_3 += lib.einsum('nled,ienm,mlad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 -= lib.einsum('nled,neim,mlad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += lib.einsum('lned,ienm,lmad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)

        t1_3 -= lib.einsum('lnde,neim,mlad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)
        t1_3 += lib.einsum('nlde,neim,mlad->ia',t2_1[:],eris_ovoo,t2_1[:],optimize=True)

        t1_3 = t1_3/D1

    cput0 = log.timer_debug1("Completed amplitude calculation", *cput0)

    t1 = (t1_2, t1_3)
    t2 = (t2_1, t2_2)

    return t1, t2, t2_1_vvvv


def compute_energy(myadc, t2, eris):

    if myadc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(myadc.method)

    eris_ovvo = eris.ovvo

    t2_new  = t2[0][:].copy()

    if (myadc.method == "adc(3)"):
        t2_new += t2[1][:]

    #Compute MPn correlation energy

    e_mp = 2 * lib.einsum('ijab,iabj', t2_new, eris_ovvo,optimize=True)
    e_mp -= lib.einsum('ijab,ibaj', t2_new, eris_ovvo,optimize=True)

    del t2_new
    return e_mp


def contract_ladder(myadc,t_amp,vvvv):

    nocc = myadc._nocc
    nvir = myadc._nvir

    t_amp = np.ascontiguousarray(t_amp.reshape(nocc*nocc,nvir*nvir).T)
    t = np.zeros((nvir,nvir, nocc*nocc))
    chnk_size = radc_ao2mo.calculate_chunk_size(myadc)

    a = 0
    if isinstance(vvvv, list):
        for dataset in vvvv:
            k = dataset.shape[0]
            dataset = dataset[:].reshape(-1,nvir*nvir)
            t[a:a+k] = np.dot(dataset,t_amp).reshape(-1,nvir,nocc*nocc)
            a += k
    elif getattr(myadc, 'with_df', None):
        for p in range(0,nvir,chnk_size):
            vvvv_p = dfadc.get_vvvv_df(myadc, vvvv, p, chnk_size)
            k = vvvv_p.shape[0]
            vvvv_p = vvvv_p.reshape(-1,nvir*nvir)
            t[a:a+k] = np.dot(vvvv_p,t_amp).reshape(-1,nvir,nocc*nocc)
            del vvvv_p
            a += k
    else:
        raise Exception("Unknown vvvv type")

    del t_amp
    t = np.ascontiguousarray(t.transpose(2,0,1)).reshape(nocc, nocc, nvir, nvir)

    return t

def _create_t2_h5cache():
    '''Create an unclosed and unlinked h5 temporary file to cache t2 data so as
    to pass t2 between iterations. This is not a good practice though. Use this
    as a temporary workaround before figuring out a better solution to handle
    big t2 amplitudes.
    '''
    tmpfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    return h5py.File(tmpfile.name, 'w')

