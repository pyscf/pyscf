# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
import time
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.adc import uadc_ao2mo
from pyscf import __config__

def kernel(adc, nroots=1, guess=None, eris=None, verbose=None):

    adc.method = adc.method.lower()
    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
       raise NotImplementedError(adc.method)

    cput0 = (time.clock(), time.time())
    log = logger.Logger(adc.stdout, adc.verbose)
    if adc.verbose >= logger.WARN:
        adc.check_sanity()
    adc.dump_flags()

    if eris is None:
        eris = uadc_ao2mo.transform_integrals_incore(adc)

    imds = adc.get_imds(eris)
    matvec, diag = adc.gen_matvec(imds, eris)

    guess = adc.get_init_guess(nroots, diag, ascending = True)

    E, U = lib.linalg_helper.davidson(matvec, guess, diag, nroots=nroots, verbose=log, max_cycle=adc.max_cycle, max_space=adc.max_space)

    U = np.array(U)

    T = adc.get_trans_moments()

    spec_factors = adc.get_spec_factors(T, U, nroots)

    if adc.verbose >= logger.INFO:
        if nroots == 1:
            logger.info(adc, '%s root %d    Energy (Eh) = %.8f    Energy (eV) = %.8f    Spec factors = %.8f',
                         adc.method, 0, E, E*27.2114, spec_factors)
        else :
            for n, en, pn in zip(range(nroots), E, spec_factors):
                logger.info(adc, '%s root %d    Energy (Eh) = %.8f    Energy (eV) = %.8f    Spec factors = %.8f',
                          adc.method, n, en, en*27.2114, pn)
        log.timer('ADC', *cput0)

    return E, U, spec_factors


def compute_amplitudes_energy(myadc, eris, verbose=None):

    t1, t2 = myadc.compute_amplitudes(eris)
    e_corr = myadc.compute_energy(t1, t2, eris)

    return e_corr, t1, t2


def compute_amplitudes(myadc, eris):

    if myadc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(myadc.method)

    nocc_a = myadc._nocc[0]
    nocc_b = myadc._nocc[1]
    nvir_a = myadc._nvir[0]
    nvir_b = myadc._nvir[1]

    eris_oovv = eris.oovv
    eris_OOVV = eris.OOVV
    eris_ooVV = eris.ooVV
    eris_OOvv = eris.OOvv
    eris_ovov = eris.ovov
    eris_OVOV = eris.OVOV
    eris_ovOV = eris.ovOV
    eris_OVov = eris.OVov
    eris_ovoo = eris.ovoo
    eris_OVoo = eris.OVoo
    eris_ovOO = eris.ovOO
    eris_OVOO = eris.OVOO

    e_a = myadc.mo_energy_a
    e_b = myadc.mo_energy_b

    d_ij_a = e_a[:nocc_a][:,None] + e_a[:nocc_a]
    d_ij_b = e_b[:nocc_b][:,None] + e_b[:nocc_b]
    d_ij_ab = e_a[:nocc_a][:,None] + e_b[:nocc_b]

    d_ab_a = e_a[nocc_a:][:,None] + e_a[nocc_a:]
    d_ab_b = e_b[nocc_b:][:,None] + e_b[nocc_b:]
    d_ab_ab = e_a[nocc_a:][:,None] + e_b[nocc_b:]

    D2_a = d_ij_a.reshape(-1,1) - d_ab_a.reshape(-1)
    D2_b = d_ij_b.reshape(-1,1) - d_ab_b.reshape(-1)
    D2_ab = d_ij_ab.reshape(-1,1) - d_ab_ab.reshape(-1)

    D2_a = D2_a.reshape((nocc_a,nocc_a,nvir_a,nvir_a))
    D2_b = D2_b.reshape((nocc_b,nocc_b,nvir_b,nvir_b))
    D2_ab = D2_ab.reshape((nocc_a,nocc_b,nvir_a,nvir_b))

    D1_a = e_a[:nocc_a][:None].reshape(-1,1) - e_a[nocc_a:].reshape(-1)
    D1_b = e_b[:nocc_b][:None].reshape(-1,1) - e_b[nocc_b:].reshape(-1)
    D1_a = D1_a.reshape((nocc_a,nvir_a))
    D1_b = D1_b.reshape((nocc_b,nvir_b))

    # Compute first-order doubles t2 (tijab)

    v2e_oovv = eris_ovov.transpose(0,2,1,3).copy()
    v2e_oovv -= eris_ovov.transpose(0,2,3,1).copy()
    v2e_OOVV = eris_OVOV.transpose(0,2,1,3).copy()
    v2e_OOVV -= eris_OVOV.transpose(0,2,3,1).copy()
    v2e_oOvV = eris_ovOV.transpose(0,2,1,3).copy()

    t2_1_a = v2e_oovv/D2_a
    t2_1_b = v2e_OOVV/D2_b
    t2_1_ab = v2e_oOvV/D2_ab

    t2_1 = (t2_1_a , t2_1_ab, t2_1_b)

    # Compute second-order singles t1 (tij)

    eris_ovvv = uadc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
    t1_2_a = 0.5*np.einsum('kdac,ikcd->ia',eris_ovvv,t2_1_a)
    t1_2_a -= 0.5*np.einsum('kcad,ikcd->ia',eris_ovvv,t2_1_a)
    del eris_ovvv
    t1_2_a -= 0.5*np.einsum('lcki,klac->ia',eris_ovoo,t2_1_a)
    t1_2_a += 0.5*np.einsum('kcli,klac->ia',eris_ovoo,t2_1_a)
    eris_OVvv = uadc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
    t1_2_a += np.einsum('kdac,ikcd->ia',eris_OVvv,t2_1_ab)
    del eris_OVvv
    t1_2_a -= np.einsum('lcki,klac->ia',eris_OVoo,t2_1_ab)
    eris_OVVV = uadc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
    t1_2_b = 0.5*np.einsum('kdac,ikcd->ia',eris_OVVV,t2_1_b)
    t1_2_b -= 0.5*np.einsum('kcad,ikcd->ia',eris_OVVV,t2_1_b)
    del eris_OVVV
    t1_2_b -= 0.5*np.einsum('lcki,klac->ia',eris_OVOO,t2_1_b)
    t1_2_b += 0.5*np.einsum('kcli,klac->ia',eris_OVOO,t2_1_b)
    eris_ovVV = uadc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
    t1_2_b += np.einsum('kdac,kidc->ia',eris_ovVV,t2_1_ab)
    del eris_ovVV
    t1_2_b -= np.einsum('lcki,lkca->ia',eris_ovOO,t2_1_ab)

    t1_2_a = t1_2_a/D1_a
    t1_2_b = t1_2_b/D1_b

    t1_2 = (t1_2_a , t1_2_b)
    t2_2 = (None,)
    t1_3 = (None,)

    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):

    # Compute second-order doubles t2 (tijab)

        eris_oooo = eris.oooo
        eris_OOOO = eris.OOOO
        eris_ooOO = eris.ooOO
        eris_ovvo = eris.ovvo
        eris_OVVO = eris.OVVO
        eris_OVvo = eris.OVvo
        eris_ovVO = eris.ovVO

        temp = t2_1_a.reshape(nocc_a*nocc_a,nvir_a*nvir_a)
        eris_vvvv = uadc_ao2mo.unpack_eri_2s(eris.vvvv, nvir_a)
        eris_vvvv = eris_vvvv.transpose(0,2,1,3)
        eris_vvvv = eris_vvvv.copy()[:].reshape(nvir_a*nvir_a,nvir_a*nvir_a)
        t2_2_a = 0.5*np.dot(temp,eris_vvvv.T).reshape(nocc_a,nocc_a,nvir_a,nvir_a)
        eris_vvvv = eris_vvvv[:].reshape(nvir_a,nvir_a,nvir_a,nvir_a)
        eris_vvvv = eris_vvvv.transpose(0,1,3,2)
        eris_vvvv = eris_vvvv.copy()[:].reshape(nvir_a*nvir_a,nvir_a*nvir_a)
        t2_2_a -= 0.5*np.dot(temp,eris_vvvv.T).reshape(nocc_a,nocc_a,nvir_a,nvir_a)
        del eris_vvvv
        t2_2_a += 0.5*np.einsum('kilj,klab->ijab', eris_oooo, t2_1_a,optimize=True)
        t2_2_a -= 0.5*np.einsum('kjli,klab->ijab', eris_oooo, t2_1_a,optimize=True)

        temp = np.einsum('kcbj,kica->ijab',eris_ovvo,t2_1_a,optimize=True)
        temp -= np.einsum('kjbc,kica->ijab',eris_oovv,t2_1_a,optimize=True)
        temp_1 = np.einsum('kcbj,ikac->ijab',eris_OVvo,t2_1_ab,optimize=True)

        t2_2_a += temp - temp.transpose(1,0,2,3) - temp.transpose(0,1,3,2) + temp.transpose(1,0,3,2)
        t2_2_a += temp_1 - temp_1.transpose(1,0,2,3) - temp_1.transpose(0,1,3,2) + temp_1.transpose(1,0,3,2)

        temp = t2_1_b.reshape(nocc_b*nocc_b,nvir_b*nvir_b)
        eris_VVVV = uadc_ao2mo.unpack_eri_2s(eris.VVVV, nvir_b)
        eris_VVVV = eris_VVVV.transpose(0,2,1,3)
        eris_VVVV = eris_VVVV.copy()[:].reshape(nvir_b*nvir_b,nvir_b*nvir_b)
        t2_2_b = 0.5*np.dot(temp,eris_VVVV.T).reshape(nocc_b,nocc_b,nvir_b,nvir_b)
        eris_VVVV = eris_VVVV[:].reshape(nvir_b,nvir_b,nvir_b,nvir_b)
        eris_VVVV = eris_VVVV.transpose(0,1,3,2)
        eris_VVVV = eris_VVVV.copy()[:].reshape(nvir_b*nvir_b,nvir_b*nvir_b)
        t2_2_b -= 0.5*np.dot(temp,eris_VVVV.T).reshape(nocc_b,nocc_b,nvir_b,nvir_b)
        del eris_VVVV
        t2_2_b += 0.5*np.einsum('kilj,klab->ijab', eris_OOOO, t2_1_b,optimize=True)
        t2_2_b -= 0.5*np.einsum('kjli,klab->ijab', eris_OOOO, t2_1_b,optimize=True)

        temp = np.einsum('kcbj,kica->ijab',eris_OVVO,t2_1_b,optimize=True)
        temp -= np.einsum('kjbc,kica->ijab',eris_OOVV,t2_1_b,optimize=True)
        temp_1 = np.einsum('kcbj,kica->ijab',eris_ovVO,t2_1_ab,optimize=True)

        t2_2_b += temp - temp.transpose(1,0,2,3) - temp.transpose(0,1,3,2) + temp.transpose(1,0,3,2)
        t2_2_b += temp_1 - temp_1.transpose(1,0,2,3) - temp_1.transpose(0,1,3,2) + temp_1.transpose(1,0,3,2)

        temp = t2_1_ab.reshape(nocc_a*nocc_b,nvir_a*nvir_b)
        eris_vvVV = uadc_ao2mo.unpack_eri_2(eris.vvVV, nvir_a, nvir_b)
        eris_vvVV = eris_vvVV.transpose(0,2,1,3)
        eris_vvVV = eris_vvVV.copy()[:].reshape(nvir_a*nvir_b,nvir_a*nvir_b)
        t2_2_ab = np.dot(temp,eris_vvVV.T).reshape(nocc_a,nocc_b,nvir_a,nvir_b)
        del eris_vvVV

        t2_2_ab += np.einsum('kilj,klab->ijab',eris_ooOO,t2_1_ab,optimize=True)
        t2_2_ab += np.einsum('kcbj,kica->ijab',eris_ovVO,t2_1_a,optimize=True)
        t2_2_ab += np.einsum('kcbj,ikac->ijab',eris_OVVO,t2_1_ab,optimize=True)
        t2_2_ab -= np.einsum('kjbc,ikac->ijab',eris_OOVV,t2_1_ab,optimize=True)
        t2_2_ab -= np.einsum('kibc,kjac->ijab',eris_ooVV,t2_1_ab,optimize=True)
        t2_2_ab -= np.einsum('kjac,ikcb->ijab',eris_OOvv,t2_1_ab,optimize=True)
        t2_2_ab += np.einsum('kcai,kjcb->ijab',eris_OVvo,t2_1_b,optimize=True)
        t2_2_ab += np.einsum('kcai,kjcb->ijab',eris_ovvo,t2_1_ab,optimize=True)
        t2_2_ab -= np.einsum('kiac,kjcb->ijab',eris_oovv,t2_1_ab,optimize=True)

        t2_2_a = t2_2_a/D2_a
        t2_2_b = t2_2_b/D2_b
        t2_2_ab = t2_2_ab/D2_ab

        t2_2 = (t2_2_a , t2_2_ab, t2_2_b)

    if (myadc.method == "adc(3)"):
    # Compute third-order singles (tij)

        eris_ovoo = eris.ovoo
        eris_OVOO = eris.OVOO
        eris_OVoo = eris.OVoo
        eris_ovOO = eris.ovOO

        t1_3 = (None,)

        t1_3_a = np.einsum('d,ilad,ld->ia',e_a[nocc_a:],t2_1_a,t1_2_a,optimize=True)
        t1_3_a += np.einsum('d,ilad,ld->ia',e_b[nocc_b:],t2_1_ab,t1_2_b,optimize=True)

        t1_3_b  = np.einsum('d,ilad,ld->ia',e_b[nocc_b:],t2_1_b, t1_2_b,optimize=True)
        t1_3_b += np.einsum('d,lida,ld->ia',e_a[nocc_a:],t2_1_ab,t1_2_a,optimize=True)

        t1_3_a -= np.einsum('l,ilad,ld->ia',e_a[:nocc_a],t2_1_a, t1_2_a,optimize=True)
        t1_3_a -= np.einsum('l,ilad,ld->ia',e_b[:nocc_b],t2_1_ab,t1_2_b,optimize=True)

        t1_3_b -= np.einsum('l,ilad,ld->ia',e_b[:nocc_b],t2_1_b, t1_2_b,optimize=True)
        t1_3_b -= np.einsum('l,lida,ld->ia',e_a[:nocc_a],t2_1_ab,t1_2_a,optimize=True)

        t1_3_a += 0.5*np.einsum('a,ilad,ld->ia',e_a[nocc_a:],t2_1_a, t1_2_a,optimize=True)
        t1_3_a += 0.5*np.einsum('a,ilad,ld->ia',e_a[nocc_a:],t2_1_ab,t1_2_b,optimize=True)

        t1_3_b += 0.5*np.einsum('a,ilad,ld->ia',e_b[nocc_b:],t2_1_b, t1_2_b,optimize=True)
        t1_3_b += 0.5*np.einsum('a,lida,ld->ia',e_b[nocc_b:],t2_1_ab,t1_2_a,optimize=True)

        t1_3_a -= 0.5*np.einsum('i,ilad,ld->ia',e_a[:nocc_a],t2_1_a, t1_2_a,optimize=True)
        t1_3_a -= 0.5*np.einsum('i,ilad,ld->ia',e_a[:nocc_a],t2_1_ab,t1_2_b,optimize=True)

        t1_3_b -= 0.5*np.einsum('i,ilad,ld->ia',e_b[:nocc_b],t2_1_b, t1_2_b,optimize=True)
        t1_3_b -= 0.5*np.einsum('i,lida,ld->ia',e_b[:nocc_b],t2_1_ab,t1_2_a,optimize=True)

        t1_3_a += np.einsum('ld,iald->ia',t1_2_a,eris_ovov,optimize=True)
        t1_3_a -= np.einsum('ld,laid->ia',t1_2_a,eris_ovov,optimize=True)
        t1_3_a += np.einsum('ld,iald->ia',t1_2_b,eris_ovOV,optimize=True)

        t1_3_b += np.einsum('ld,iald->ia',t1_2_b,eris_OVOV ,optimize=True)
        t1_3_b -= np.einsum('ld,laid->ia',t1_2_b,eris_OVOV ,optimize=True)
        t1_3_b += np.einsum('ld,ldia->ia',t1_2_a,eris_ovOV,optimize=True)

        t1_3_a += np.einsum('ld,ldai->ia',t1_2_a,eris_ovvo ,optimize=True)
        t1_3_a -= np.einsum('ld,liad->ia',t1_2_a,eris_oovv ,optimize=True)
        t1_3_a += np.einsum('ld,ldai->ia',t1_2_b,eris_OVvo,optimize=True)

        t1_3_b += np.einsum('ld,ldai->ia',t1_2_b,eris_OVVO ,optimize=True)
        t1_3_b -= np.einsum('ld,liad->ia',t1_2_b,eris_OOVV ,optimize=True)
        t1_3_b += np.einsum('ld,ldai->ia',t1_2_a,eris_ovVO,optimize=True)

        t1_3_a -= 0.5*np.einsum('lmad,mdli->ia',t2_2_a,eris_ovoo,optimize=True)
        t1_3_a += 0.5*np.einsum('lmad,ldmi->ia',t2_2_a,eris_ovoo,optimize=True)
        t1_3_a -=     np.einsum('lmad,mdli->ia',t2_2_ab,eris_OVoo,optimize=True)

        t1_3_b -= 0.5*np.einsum('lmad,mdli->ia',t2_2_b,eris_OVOO,optimize=True)
        t1_3_b += 0.5*np.einsum('lmad,ldmi->ia',t2_2_b,eris_OVOO,optimize=True)
        t1_3_b -=     np.einsum('mlda,mdli->ia',t2_2_ab,eris_ovOO,optimize=True)

        eris_ovvv = uadc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
        t1_3_a += 0.5*np.einsum('ilde,lead->ia',t2_2_a,eris_ovvv,optimize=True)
        t1_3_a -= 0.5*np.einsum('ilde,ldae->ia',t2_2_a,eris_ovvv,optimize=True)
        t1_3_a -= np.einsum('ildf,mefa,lmde->ia',t2_1_a, eris_ovvv,  t2_1_a ,optimize=True)
        t1_3_a += np.einsum('ildf,mafe,lmde->ia',t2_1_a, eris_ovvv,  t2_1_a ,optimize=True)
        t1_3_a += np.einsum('ilfd,mefa,mled->ia',t2_1_ab,eris_ovvv, t2_1_ab,optimize=True)
        t1_3_a -= np.einsum('ilfd,mafe,mled->ia',t2_1_ab,eris_ovvv, t2_1_ab,optimize=True)
        t1_3_a += 0.5*np.einsum('ilaf,mefd,lmde->ia',t2_1_a,eris_ovvv,t2_1_a,optimize=True)
        t1_3_a -= 0.5*np.einsum('ilaf,mdfe,lmde->ia',t2_1_a,eris_ovvv,t2_1_a,optimize=True)
        t1_3_b += 0.5*np.einsum('lifa,mefd,lmde->ia',t2_1_ab,eris_ovvv,t2_1_a,optimize=True)
        t1_3_b -= 0.5*np.einsum('lifa,mdfe,lmde->ia',t2_1_ab,eris_ovvv,t2_1_a,optimize=True)
        t1_3_a += 0.5*np.einsum('lmdf,iaef,lmde->ia',t2_1_a,eris_ovvv,t2_1_a,optimize=True)
        t1_3_a -= 0.5*np.einsum('lmdf,ifea,lmde->ia',t2_1_a,eris_ovvv,t2_1_a,optimize=True)
        t1_3_a += np.einsum('mlfd,iaef,mled->ia',t2_1_ab,eris_ovvv,t2_1_ab,optimize=True)
        t1_3_a -= np.einsum('mlfd,ifea,mled->ia',t2_1_ab,eris_ovvv,t2_1_ab,optimize=True)
        t1_3_a -= 0.25*np.einsum('lmef,iedf,lmad->ia',t2_1_a,eris_ovvv,t2_1_a,optimize=True)
        t1_3_a += 0.25*np.einsum('lmef,ifde,lmad->ia',t2_1_a,eris_ovvv,t2_1_a,optimize=True)
        del eris_ovvv

        eris_OVVV = uadc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
        t1_3_b += 0.5*np.einsum('ilde,lead->ia',t2_2_b,eris_OVVV,optimize=True)
        t1_3_b -= 0.5*np.einsum('ilde,ldae->ia',t2_2_b,eris_OVVV,optimize=True)
        t1_3_b -= np.einsum('ildf,mefa,lmde->ia',t2_1_b,eris_OVVV,t2_1_b,optimize=True)
        t1_3_b += np.einsum('ildf,mafe,lmde->ia',t2_1_b,eris_OVVV,t2_1_b,optimize=True)
        t1_3_b += np.einsum('lidf,mefa,lmde->ia',t2_1_ab,eris_OVVV,t2_1_ab,optimize=True)
        t1_3_b -= np.einsum('lidf,mafe,lmde->ia',t2_1_ab,eris_OVVV,t2_1_ab,optimize=True)
        t1_3_a += 0.5*np.einsum('ilaf,mefd,lmde->ia',t2_1_ab,eris_OVVV,t2_1_b,optimize=True)
        t1_3_a -= 0.5*np.einsum('ilaf,mdfe,lmde->ia',t2_1_ab,eris_OVVV,t2_1_b,optimize=True)
        t1_3_b += 0.5*np.einsum('ilaf,mefd,lmde->ia',t2_1_b,eris_OVVV,t2_1_b,optimize=True)
        t1_3_b -= 0.5*np.einsum('ilaf,mdfe,lmde->ia',t2_1_b,eris_OVVV,t2_1_b,optimize=True)
        t1_3_b += 0.5*np.einsum('lmdf,iaef,lmde->ia',t2_1_b,eris_OVVV,t2_1_b,optimize=True)
        t1_3_b -= 0.5*np.einsum('lmdf,ifea,lmde->ia',t2_1_b,eris_OVVV,t2_1_b,optimize=True)
        t1_3_b += np.einsum('lmdf,iaef,lmde->ia',t2_1_ab,eris_OVVV,t2_1_ab,optimize=True)
        t1_3_b -= np.einsum('lmdf,ifea,lmde->ia',t2_1_ab,eris_OVVV,t2_1_ab,optimize=True)
        t1_3_b -= 0.25*np.einsum('lmef,iedf,lmad->ia',t2_1_b,eris_OVVV,t2_1_b,optimize=True)
        t1_3_b += 0.25*np.einsum('lmef,ifde,lmad->ia',t2_1_b,eris_OVVV,t2_1_b,optimize=True)
        del eris_OVVV

        eris_ovVV = uadc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
        t1_3_b += np.einsum('lied,lead->ia',t2_2_ab,eris_ovVV,optimize=True)
        t1_3_a -= np.einsum('ildf,mafe,mlde->ia',t2_1_ab,eris_ovVV,t2_1_ab,optimize=True)
        t1_3_b -= np.einsum('ildf,mefa,mled->ia',t2_1_b,eris_ovVV,t2_1_ab,optimize=True)
        t1_3_b += np.einsum('lidf,mefa,lmde->ia',t2_1_ab,eris_ovVV,t2_1_a,optimize=True)
        t1_3_a += np.einsum('ilaf,mefd,mled->ia',t2_1_ab,eris_ovVV,t2_1_ab,optimize=True)
        t1_3_b += np.einsum('ilaf,mefd,mled->ia',t2_1_b,eris_ovVV,t2_1_ab,optimize=True)
        t1_3_a += 0.5*np.einsum('lmdf,iaef,lmde->ia',t2_1_b,eris_ovVV,t2_1_b,optimize=True)
        t1_3_a += np.einsum('lmdf,iaef,lmde->ia',t2_1_ab,eris_ovVV,t2_1_ab,optimize=True)
        t1_3_a -= np.einsum('lmef,iedf,lmad->ia',t2_1_ab,eris_ovVV,t2_1_ab,optimize=True)
        del eris_ovVV

        eris_OVvv = uadc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
        t1_3_a += np.einsum('ilde,lead->ia',t2_2_ab,eris_OVvv,optimize=True)
        t1_3_a -= np.einsum('ildf,mefa,lmde->ia',t2_1_a,eris_OVvv, t2_1_ab,optimize=True)
        t1_3_a += np.einsum('ilfd,mefa,lmde->ia',t2_1_ab,eris_OVvv,t2_1_b ,optimize=True)
        t1_3_b -= np.einsum('lifd,mafe,lmed->ia',t2_1_ab,eris_OVvv,t2_1_ab,optimize=True)
        t1_3_a += np.einsum('ilaf,mefd,lmde->ia',t2_1_a,eris_OVvv,t2_1_ab,optimize=True)
        t1_3_b += np.einsum('lifa,mefd,lmde->ia',t2_1_ab,eris_OVvv,t2_1_ab,optimize=True)
        t1_3_b += 0.5*np.einsum('lmdf,iaef,lmde->ia',t2_1_a,eris_OVvv,t2_1_a,optimize=True)
        t1_3_b += np.einsum('mlfd,iaef,mled->ia',t2_1_ab,eris_OVvv,t2_1_ab,optimize=True)
        temp = t2_1_ab.reshape(nocc_a*nocc_b,-1)
        eris_OVvv = eris_OVvv.transpose(3,1,2,0)
        eris_OVvv = eris_OVvv.copy()[:].reshape(nvir_a*nvir_b,-1)
        temp_2 = t2_1_ab.reshape(nocc_a*nocc_b*nvir_a,-1)
        int_1 = np.dot(temp,eris_OVvv).reshape(nocc_a*nocc_b*nvir_a,-1)
        t1_3_b -= np.dot(int_1.T,temp_2).reshape(nocc_b,nvir_b)
        del eris_OVvv

        t1_3_a += 0.25*np.einsum('inde,lamn,lmde->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3_a -= 0.25*np.einsum('inde,maln,lmde->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3_a += np.einsum('inde,lamn,lmde->ia',t2_1_ab,eris_ovOO,t2_1_ab,optimize=True)

        t1_3_b += 0.25*np.einsum('inde,lamn,lmde->ia',t2_1_b,eris_OVOO,t2_1_b,optimize=True)
        t1_3_b -= 0.25*np.einsum('inde,maln,lmde->ia',t2_1_b,eris_OVOO,t2_1_b,optimize=True)
        t1_3_b += np.einsum('nied,lamn,mled->ia',t2_1_ab,eris_OVoo,t2_1_ab,optimize=True)

        t1_3_a += 0.5*np.einsum('inad,lemn,lmde->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3_a -= 0.5*np.einsum('inad,meln,lmde->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3_a -= 0.5 * np.einsum('inad,lemn,mlde->ia',t2_1_a,eris_OVoo,t2_1_ab,optimize=True)
        t1_3_a -= 0.5 * np.einsum('inad,meln,lmde->ia',t2_1_a,eris_OVoo,t2_1_ab,optimize=True)
        t1_3_a -= 0.5 *np.einsum('inad,lemn,lmed->ia',t2_1_ab,eris_ovOO,t2_1_ab,optimize=True)
        t1_3_a -= 0.5*np.einsum('inad,meln,mled->ia',t2_1_ab,eris_ovOO,t2_1_ab,optimize=True)
        t1_3_a += 0.5*np.einsum('inad,lemn,lmde->ia',t2_1_ab,eris_OVOO,t2_1_b,optimize=True)
        t1_3_a -= 0.5*np.einsum('inad,meln,lmde->ia',t2_1_ab,eris_OVOO,t2_1_b,optimize=True)

        t1_3_b += 0.5*np.einsum('inad,lemn,lmde->ia',t2_1_b,eris_OVOO,t2_1_b,optimize=True)
        t1_3_b -= 0.5*np.einsum('inad,meln,lmde->ia',t2_1_b,eris_OVOO,t2_1_b,optimize=True)
        t1_3_b -= 0.5 * np.einsum('inad,meln,mled->ia',t2_1_b,eris_ovOO,t2_1_ab,optimize=True)
        t1_3_b -= 0.5 * np.einsum('inad,lemn,lmed->ia',t2_1_b,eris_ovOO,t2_1_ab,optimize=True)
        t1_3_b -= 0.5 *np.einsum('nida,meln,lmde->ia',t2_1_ab,eris_OVoo,t2_1_ab,optimize=True)
        t1_3_b -= 0.5*np.einsum('nida,lemn,mlde->ia',t2_1_ab,eris_OVoo,t2_1_ab,optimize=True)
        t1_3_b += 0.5*np.einsum('nida,lemn,lmde->ia',t2_1_ab,eris_ovoo,t2_1_a,optimize=True)
        t1_3_b -= 0.5*np.einsum('nida,meln,lmde->ia',t2_1_ab,eris_ovoo,t2_1_a,optimize=True)

        t1_3_a -= 0.5*np.einsum('lnde,ianm,lmde->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3_a += 0.5*np.einsum('lnde,naim,lmde->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3_a -= np.einsum('nled,ianm,mled->ia',t2_1_ab,eris_ovoo,t2_1_ab,optimize=True)
        t1_3_a += np.einsum('nled,naim,mled->ia',t2_1_ab,eris_ovoo,t2_1_ab,optimize=True)
        t1_3_a -= 0.5*np.einsum('lnde,ianm,lmde->ia',t2_1_b,eris_ovOO,t2_1_b,optimize=True)
        t1_3_a -= np.einsum('lnde,ianm,lmde->ia',t2_1_ab,eris_ovOO,t2_1_ab,optimize=True)

        t1_3_b -= 0.5*np.einsum('lnde,ianm,lmde->ia',t2_1_b,eris_OVOO,t2_1_b,optimize=True)
        t1_3_b += 0.5*np.einsum('lnde,naim,lmde->ia',t2_1_b,eris_OVOO,t2_1_b,optimize=True)
        t1_3_b -= np.einsum('lnde,ianm,lmde->ia',t2_1_ab,eris_OVOO,t2_1_ab,optimize=True)
        t1_3_b += np.einsum('lnde,naim,lmde->ia',t2_1_ab,eris_OVOO,t2_1_ab,optimize=True)
        t1_3_b -= 0.5*np.einsum('lnde,ianm,lmde->ia',t2_1_a,eris_OVoo,t2_1_a,optimize=True)
        t1_3_b -= np.einsum('nled,ianm,mled->ia',t2_1_ab,eris_OVoo,t2_1_ab,optimize=True)

        t1_3_a -= np.einsum('lnde,ienm,lmad->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3_a += np.einsum('lnde,neim,lmad->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3_a += np.einsum('lnde,neim,lmad->ia',t2_1_ab,eris_OVoo,t2_1_a,optimize=True)
        t1_3_a += np.einsum('nled,ienm,mlad->ia',t2_1_ab,eris_ovoo,t2_1_ab,optimize=True)
        t1_3_a -= np.einsum('nled,neim,mlad->ia',t2_1_ab,eris_ovoo,t2_1_ab,optimize=True)
        t1_3_a += np.einsum('lned,ienm,lmad->ia',t2_1_ab,eris_ovOO,t2_1_ab,optimize=True)
        t1_3_a -= np.einsum('lnde,neim,mlad->ia',t2_1_b,eris_OVoo,t2_1_ab,optimize=True)

        t1_3_b -= np.einsum('lnde,ienm,lmad->ia',t2_1_b,eris_OVOO,t2_1_b,optimize=True)
        t1_3_b += np.einsum('lnde,neim,lmad->ia',t2_1_b,eris_OVOO,t2_1_b,optimize=True)
        t1_3_b += np.einsum('nled,neim,lmad->ia',t2_1_ab,eris_ovOO,t2_1_b,optimize=True)
        t1_3_b += np.einsum('lnde,ienm,lmda->ia',t2_1_ab,eris_OVOO,t2_1_ab,optimize=True)
        t1_3_b -= np.einsum('lnde,neim,lmda->ia',t2_1_ab,eris_OVOO,t2_1_ab,optimize=True)
        t1_3_b += np.einsum('nlde,ienm,mlda->ia',t2_1_ab,eris_OVoo,t2_1_ab,optimize=True)
        t1_3_b -= np.einsum('lnde,neim,lmda->ia',t2_1_a,eris_ovOO,t2_1_ab,optimize=True)

        t1_3_a = t1_3_a/D1_a
        t1_3_b = t1_3_b/D1_b

        t1_3 = (t1_3_a, t1_3_b)

    t1 = (t1_2, t1_3)
    t2 = (t2_1, t2_2)

    return t1, t2


def compute_energy(myadc, t1, t2, eris):

    if myadc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(myadc.method)

    nocc_a = myadc._nocc[0]
    nocc_b = myadc._nocc[1]
    nvir_a = myadc._nvir[0]
    nvir_b = myadc._nvir[1]

    eris_ovov = eris.ovov
    eris_OVOV = eris.OVOV
    eris_ovOV = eris.ovOV

    t2_1_a, t2_1_ab, t2_1_b  = t2[0]

    #Compute MP2 correlation energy

    e_mp2 = 0.25 * np.einsum('ijab,iajb', t2_1_a, eris_ovov)
    e_mp2 -= 0.25 * np.einsum('ijab,ibja', t2_1_a, eris_ovov)
    e_mp2 += np.einsum('ijab,iajb', t2_1_ab, eris_ovOV)
    e_mp2 += 0.25 * np.einsum('ijab,iajb', t2_1_b, eris_OVOV)
    e_mp2 -= 0.25 * np.einsum('ijab,ibja', t2_1_b, eris_OVOV)

    e_corr = e_mp2

    if (myadc.method == "adc(3)"):

        #Compute MP3 correlation energy
        eris_oovv = eris.oovv
        eris_OOVV = eris.OOVV
        eris_OOvv = eris.OOvv
        eris_ooVV = eris.ooVV
        eris_ovvo = eris.ovvo
        eris_OVVO = eris.OVVO
        eris_OVvo = eris.OVvo
        eris_ovVO = eris.ovVO
        eris_oooo = eris.oooo
        eris_OOOO = eris.OOOO
        eris_ooOO = eris.ooOO

        #eris_vvvv = uadc_ao2mo.unpack_eri_2s(eris.vvvv, nvir_a)
        #temp_1_a = np.einsum('ijab,acbd->ijcd',t2_1_a, eris_vvvv)
        #temp_1_a -= np.einsum('ijab,adbc->ijcd',t2_1_a, eris_vvvv)
        #del eris_vvvv
        #e_mp3 = 0.125 * np.einsum('ijcd,ijcd',temp_1_a, t2_1_a)
        #del temp_1_a

        temp = t2_1_a.reshape(nocc_a*nocc_a,nvir_a*nvir_a)
        eris_vvvv = uadc_ao2mo.unpack_eri_2s(eris.vvvv, nvir_a)
        eris_vvvv = eris_vvvv.transpose(0,2,1,3)
        eris_vvvv = eris_vvvv.copy()[:].reshape(nvir_a*nvir_a,nvir_a*nvir_a)
        temp_1_a = np.dot(temp,eris_vvvv.T).reshape(nocc_a,nocc_a,nvir_a,nvir_a)
        eris_vvvv = eris_vvvv[:].reshape(nvir_a,nvir_a,nvir_a,nvir_a)
        eris_vvvv = eris_vvvv.transpose(0,1,3,2)
        eris_vvvv = eris_vvvv.copy()[:].reshape(nvir_a*nvir_a,nvir_a*nvir_a)
        temp_1_a -= np.dot(temp,eris_vvvv.T).reshape(nocc_a,nocc_a,nvir_a,nvir_a)
        del eris_vvvv
        e_mp3 = 0.125 * np.einsum('ijcd,ijcd',temp_1_a, t2_1_a)
        del temp_1_a

        #eris_VVVV = uadc_ao2mo.unpack_eri_2s(eris.VVVV, nvir_b)
        #temp_1_b =  np.einsum('ijab,acbd->ijcd', t2_1_b, eris_VVVV)
        #temp_1_b -=  np.einsum('ijab,adbc->ijcd', t2_1_b, eris_VVVV)
        #del eris_VVVV
        #e_mp3 += 0.125 * np.einsum('ijcd,ijcd',temp_1_b, t2_1_b)
        #del temp_1_b

        temp = t2_1_b.reshape(nocc_b*nocc_b,nvir_b*nvir_b)
        eris_VVVV = uadc_ao2mo.unpack_eri_2s(eris.VVVV, nvir_b)
        eris_VVVV = eris_VVVV.transpose(0,2,1,3)
        eris_VVVV = eris_VVVV.copy()[:].reshape(nvir_b*nvir_b,nvir_b*nvir_b)
        temp_1_b = np.dot(temp,eris_VVVV.T).reshape(nocc_b,nocc_b,nvir_b,nvir_b)
        eris_VVVV = eris_VVVV[:].reshape(nvir_b,nvir_b,nvir_b,nvir_b)
        eris_VVVV = eris_VVVV.transpose(0,1,3,2)
        eris_VVVV = eris_VVVV.copy()[:].reshape(nvir_b*nvir_b,nvir_b*nvir_b)
        temp_1_b -= np.dot(temp,eris_VVVV.T).reshape(nocc_b,nocc_b,nvir_b,nvir_b)
        del eris_VVVV
        e_mp3 += 0.125 * np.einsum('ijcd,ijcd',temp_1_b, t2_1_b)
        del temp_1_b

        #eris_vvVV = uadc_ao2mo.unpack_eri_2(eris.vvVV, nvir_a, nvir_b)
        #temp_1_ab_1 =  np.einsum('ijab,acbd->ijcd', t2_1_ab, eris_vvVV)
        #del eris_vvVV
        #e_mp3 +=  np.einsum('ijcd,ijcd',temp_1_ab_1, t2_1_ab)
        #del temp_1_ab_1

        temp = t2_1_ab.reshape(nocc_a*nocc_b,nvir_a*nvir_b)
        eris_vvVV = uadc_ao2mo.unpack_eri_2(eris.vvVV, nvir_a, nvir_b)
        eris_vvVV = eris_vvVV.transpose(0,2,1,3)
        eris_vvVV = eris_vvVV.copy()[:].reshape(nvir_a*nvir_b,nvir_a*nvir_b)
        temp_1_ab = np.dot(temp,eris_vvVV.T).reshape(nocc_a,nocc_b,nvir_a,nvir_b)
        del eris_vvVV
        e_mp3 +=  np.einsum('ijcd,ijcd',temp_1_ab, t2_1_ab)
        del temp_1_ab

        temp_2_a =  np.einsum('ijab,klab', t2_1_a, t2_1_a)
        e_mp3 += 0.125 * np.einsum('ijkl,ikjl',temp_2_a, eris_oooo)
        e_mp3 -= 0.125 * np.einsum('ijkl,iljk',temp_2_a, eris_oooo)
        del temp_2_a

        temp_2_b =  np.einsum('ijab,klab', t2_1_b, t2_1_b)
        e_mp3 += 0.125 * np.einsum('ijkl,ikjl',temp_2_b, eris_OOOO)
        e_mp3 -= 0.125 * np.einsum('ijkl,iljk',temp_2_b, eris_OOOO)
        del temp_2_b

        temp_2_ab_1 =  np.einsum('ijab,klab', t2_1_ab, t2_1_ab)
        e_mp3 +=  np.einsum('ijkl,ikjl',temp_2_ab_1, eris_ooOO)
        del temp_2_ab_1

        temp_3_a = np.einsum('ijab,ikcb->akcj', t2_1_a, t2_1_a)
        temp_3_a += np.einsum('jiab,kicb->akcj', t2_1_ab, t2_1_ab)
        e_mp3 -= np.einsum('akcj,kjac',temp_3_a, eris_oovv)
        e_mp3 += np.einsum('akcj,kcaj',temp_3_a, eris_ovvo)
        del temp_3_a

        temp_3_b = np.einsum('ijab,ikcb->akcj', t2_1_b, t2_1_b)
        temp_3_b += np.einsum('ijba,ikbc->akcj', t2_1_ab, t2_1_ab)
        e_mp3 -= np.einsum('akcj,kjac',temp_3_b, eris_OOVV)
        e_mp3 += np.einsum('akcj,kcaj',temp_3_b, eris_OVVO)
        del temp_3_b

        temp_3_ab_1 = np.einsum('ijab,ikcb->akcj', t2_1_ab, t2_1_ab)
        e_mp3 -= np.einsum('akcj,kjac',temp_3_ab_1, eris_OOvv)
        del temp_3_ab_1

        temp_3_ab_2 = np.einsum('jiba,kibc->akcj', t2_1_ab, t2_1_ab)
        e_mp3 -= np.einsum('akcj,kjac',temp_3_ab_2, eris_ooVV)
        del temp_3_ab_2

        temp_3_ab_3 = -np.einsum('ijab,ikbc->akcj', t2_1_a, t2_1_ab)
        temp_3_ab_3 -= np.einsum('jiab,ikcb->akcj', t2_1_ab, t2_1_b)
        e_mp3 += np.einsum('akcj,kcaj',temp_3_ab_3, eris_OVvo)
        del temp_3_ab_3

        temp_3_ab_4 = -np.einsum('ijba,ikcb->akcj', t2_1_ab, t2_1_a)
        temp_3_ab_4 -= np.einsum('ijab,kicb->akcj', t2_1_b, t2_1_ab)
        e_mp3 += np.einsum('akcj,kcaj',temp_3_ab_4, eris_ovVO)
        del temp_3_ab_4

        e_corr += e_mp3

    return e_corr


class UADC(lib.StreamObject):
    '''Ground state calculations

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`
        incore_complete : bool
            Avoid all I/O. Default is False.
        method : string
            nth-order ADC method. Options are : ADC(2), ADC(2)-X, ADC(3). Default is ADC(2).

            >>> mol = gto.M(atom = 'H 0 0 0; F 0 0 1.1', basis = 'ccpvdz')
            >>> mf = scf.RHF(mol).run()
            >>> myadc = adc.UADC(mf).run()

    Saved results

        e_corr : float
            MPn correlation correction
        e_tot : float
            Total energy (HF + correlation)
        t1, t2 :
            T amplitudes t1[i,a], t2[i,j,a,b]  (i,j in occ, a,b in virt)
    '''
    incore_complete = getattr(__config__, 'adc_uadc_UADC_incore_complete', False)

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        if 'dft' in str(mf.__module__):
            raise NotImplementedError('DFT reference for UADC')

        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.max_space = getattr(__config__, 'adc_uadc_UADC_max_space', 12)
        self.max_cycle = getattr(__config__, 'adc_uadc_UADC_max_cycle', 50)
        self.conv_tol = getattr(__config__, 'adc_uadc_UADC_conv_tol', 1e-12)
        self.scf_energy = mf.scf()

        self.frozen = frozen
        self.incore_complete = self.incore_complete or self.mol.incore_anyway

        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.e_corr = None
        self.e_tot = None
        self.t1 = None
        self.t2 = None
        self._nocc = mf.nelec
        self._nmo = (mo_coeff[0].shape[1], mo_coeff[1].shape[1])
        self._nvir = (self._nmo[0] - self._nocc[0], self._nmo[1] - self._nocc[1])
        self.mo_energy_a = mf.mo_energy[0]
        self.mo_energy_b = mf.mo_energy[1]
        self.chkfile = mf.chkfile
        self.method = "adc(2)"

        keys = set(('e_corr', 'method', 'mo_coeff', 'mol', 'mo_energy_b', 'max_memory', 'scf_energy', 'e_tot', 't1', 'frozen', 'mo_energy_a', 'chkfile', 'max_space', 't2', 'mo_occ', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)

    compute_amplitudes = compute_amplitudes
    compute_energy = compute_energy

    def dump_flags(self, verbose=None):
        logger.info(self, '')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'max_space = %d', self.max_space)
        logger.info(self, 'max_cycle = %d', self.max_cycle)
        logger.info(self, 'conv_tol = %s', self.conv_tol)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        return self

    def dump_flags_gs(self, verbose=None):
        logger.info(self, '')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        return self

    def kernel(self):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)

        self.method = self.method.lower()
        if self.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
            raise NotImplementedError(self.method)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags_gs()

        eris = uadc_ao2mo.transform_integrals_incore(self)
        self.e_corr, self.t1, self.t2 = compute_amplitudes_energy(self, eris, verbose=self.verbose)
        self.e_tot = self.scf_energy + self.e_corr

        self._finalize()

        return self.e_corr, self.t1, self.t2

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        logger.note(self, 'E_corr = %.8f  E_tot = %.8f',
                    self.e_corr, self.e_tot)
        return self

    def ea_adc(self, nroots=1, guess=None):
        return UADCEA(self).kernel(nroots, guess)

    def ip_adc(self, nroots=1, guess=None):
        return UADCIP(self).kernel(nroots, guess)


def get_imds_ea(adc, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t1 = adc.t1
    t2 = adc.t2

    t1_2_a, t1_2_b = t1[0]
    t2_1_a, t2_1_ab, t2_1_b = t2[0]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    if eris is None:
        eris = uadc_ao2mo.transform_integrals_incore(adc)

    eris_ovov = eris.ovov
    eris_OVOV = eris.OVOV
    eris_ovOV = eris.ovOV
    eris_OVov = eris.OVov

    # a-b block
    # Zeroth-order terms

    M_ab_a = np.einsum('ab,a->ab', idn_vir_a, e_vir_a)
    M_ab_b = np.einsum('ab,a->ab', idn_vir_b, e_vir_b)

   # Second-order terms

    M_ab_a +=  np.einsum('l,lmad,lmbd->ab',e_occ_a,t2_1_a, t2_1_a)
    M_ab_a +=  np.einsum('l,lmad,lmbd->ab',e_occ_a,t2_1_ab, t2_1_ab)
    M_ab_a +=  np.einsum('l,mlad,mlbd->ab',e_occ_b,t2_1_ab, t2_1_ab)

    M_ab_b +=  np.einsum('l,lmad,lmbd->ab',e_occ_b,t2_1_b, t2_1_b)
    M_ab_b +=  np.einsum('l,mlda,mldb->ab',e_occ_b,t2_1_ab, t2_1_ab)
    M_ab_b +=  np.einsum('l,lmda,lmdb->ab',e_occ_a,t2_1_ab, t2_1_ab)

    M_ab_a -= 0.5 *  np.einsum('d,lmad,lmbd->ab',e_vir_a,t2_1_a, t2_1_a)
    M_ab_a -= 0.5 *  np.einsum('d,lmad,lmbd->ab',e_vir_b,t2_1_ab, t2_1_ab)
    M_ab_a -= 0.5 *  np.einsum('d,mlad,mlbd->ab',e_vir_b,t2_1_ab, t2_1_ab)

    M_ab_b -= 0.5 *  np.einsum('d,lmad,lmbd->ab',e_vir_b,t2_1_b, t2_1_b)
    M_ab_b -= 0.5 *  np.einsum('d,mlda,mldb->ab',e_vir_a,t2_1_ab, t2_1_ab)
    M_ab_b -= 0.5 *  np.einsum('d,lmda,lmdb->ab',e_vir_a,t2_1_ab, t2_1_ab)

    M_ab_a -= 0.25 *  np.einsum('a,lmad,lmbd->ab',e_vir_a,t2_1_a, t2_1_a)
    M_ab_a -= 0.25 *  np.einsum('a,lmad,lmbd->ab',e_vir_a,t2_1_ab, t2_1_ab)
    M_ab_a -= 0.25 *  np.einsum('a,mlad,mlbd->ab',e_vir_a,t2_1_ab, t2_1_ab)

    M_ab_b -= 0.25 *  np.einsum('a,lmad,lmbd->ab',e_vir_b,t2_1_b, t2_1_b)
    M_ab_b -= 0.25 *  np.einsum('a,mlda,mldb->ab',e_vir_b,t2_1_ab, t2_1_ab)
    M_ab_b -= 0.25 *  np.einsum('a,lmda,lmdb->ab',e_vir_b,t2_1_ab, t2_1_ab)

    M_ab_a -= 0.25 *  np.einsum('b,lmad,lmbd->ab',e_vir_a,t2_1_a, t2_1_a)
    M_ab_a -= 0.25 *  np.einsum('b,lmad,lmbd->ab',e_vir_a,t2_1_ab, t2_1_ab)
    M_ab_a -= 0.25 *  np.einsum('b,mlad,mlbd->ab',e_vir_a,t2_1_ab, t2_1_ab)

    M_ab_b -= 0.25 *  np.einsum('b,lmad,lmbd->ab',e_vir_b,t2_1_b, t2_1_b)
    M_ab_b -= 0.25 *  np.einsum('b,mlda,mldb->ab',e_vir_b,t2_1_ab, t2_1_ab)
    M_ab_b -= 0.25 *  np.einsum('b,lmda,lmdb->ab',e_vir_b,t2_1_ab, t2_1_ab)

    M_ab_a -= 0.5 *  np.einsum('lmad,lbmd->ab',t2_1_a, eris_ovov)
    M_ab_a += 0.5 *  np.einsum('lmad,ldmb->ab',t2_1_a, eris_ovov)
    M_ab_a -=        np.einsum('lmad,lbmd->ab',t2_1_ab, eris_ovOV)

    M_ab_b -= 0.5 *  np.einsum('lmad,lbmd->ab',t2_1_b, eris_OVOV)
    M_ab_b += 0.5 *  np.einsum('lmad,ldmb->ab',t2_1_b, eris_OVOV)
    M_ab_b -=        np.einsum('mlda,mdlb->ab',t2_1_ab, eris_ovOV)

    M_ab_a -= 0.5 *  np.einsum('lmbd,lamd->ab',t2_1_a,eris_ovov)
    M_ab_a += 0.5 *  np.einsum('lmbd,ldma->ab',t2_1_a, eris_ovov)
    M_ab_a -=        np.einsum('lmbd,lamd->ab',t2_1_ab, eris_ovOV)

    M_ab_b -= 0.5 *  np.einsum('lmbd,lamd->ab',t2_1_b, eris_OVOV)
    M_ab_b += 0.5 *  np.einsum('lmbd,ldma->ab',t2_1_b, eris_OVOV)
    M_ab_b -=        np.einsum('mldb,mdla->ab',t2_1_ab, eris_ovOV)

    #Third-order terms

    if(method =='adc(3)'):

        t2_2_a, t2_2_ab, t2_2_b = t2[1]

        eris_oovv = eris.oovv
        eris_OOVV = eris.OOVV
        eris_OOvv = eris.OOvv
        eris_ooVV = eris.ooVV
        eris_ovvo = eris.ovvo
        eris_OVVO = eris.OVVO
        eris_OVvo = eris.OVvo
        eris_ovVO = eris.ovVO
        eris_oooo = eris.oooo
        eris_OOOO = eris.OOOO
        eris_ooOO = eris.ooOO

        eris_ovvv = uadc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
        M_ab_a +=  np.einsum('ld,ldab->ab',t1_2_a, eris_ovvv)
        M_ab_a -=  np.einsum('ld,lbad->ab',t1_2_a, eris_ovvv)
        M_ab_a += np.einsum('ld,ldab->ab',t1_2_a, eris_ovvv)
        M_ab_a -= np.einsum('ld,ladb->ab',t1_2_a, eris_ovvv)
        del eris_ovvv

        eris_OVvv = uadc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
        M_ab_a +=  np.einsum('ld,ldab->ab',t1_2_b, eris_OVvv)
        M_ab_a += np.einsum('ld,ldab->ab',t1_2_b, eris_OVvv)
        del eris_OVvv

        eris_OVVV = uadc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
        M_ab_b +=  np.einsum('ld,ldab->ab',t1_2_b, eris_OVVV)
        M_ab_b -=  np.einsum('ld,lbad->ab',t1_2_b, eris_OVVV)
        M_ab_b += np.einsum('ld,ldab->ab',t1_2_b, eris_OVVV)
        M_ab_b -= np.einsum('ld,ladb->ab',t1_2_b, eris_OVVV)
        del eris_OVVV

        eris_ovVV = uadc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
        M_ab_b +=  np.einsum('ld,ldab->ab',t1_2_a, eris_ovVV)
        M_ab_b += np.einsum('ld,ldab->ab',t1_2_a, eris_ovVV)
        eris_ovVV

        M_ab_a -= 0.5 *  np.einsum('lmad,lbmd->ab',t2_2_a, eris_ovov)
        M_ab_a += 0.5 *  np.einsum('lmad,ldmb->ab',t2_2_a, eris_ovov)
        M_ab_a -=        np.einsum('lmad,lbmd->ab',t2_2_ab, eris_ovOV)

        M_ab_b -= 0.5 *  np.einsum('lmad,lbmd->ab',t2_2_b, eris_OVOV)
        M_ab_b += 0.5 *  np.einsum('lmad,ldmb->ab',t2_2_b, eris_OVOV)
        M_ab_b -=        np.einsum('mlda,mdlb->ab',t2_2_ab, eris_ovOV)

        M_ab_a -= 0.5 *  np.einsum('lmbd,lamd->ab',t2_2_a,eris_ovov)
        M_ab_a += 0.5 *  np.einsum('lmbd,ldma->ab',t2_2_a, eris_ovov)
        M_ab_a -=        np.einsum('lmbd,lamd->ab',t2_2_ab, eris_ovOV)

        M_ab_b -= 0.5 *  np.einsum('lmbd,lamd->ab',t2_2_b, eris_OVOV)
        M_ab_b += 0.5 *  np.einsum('lmbd,ldma->ab',t2_2_b, eris_OVOV)
        M_ab_b -=        np.einsum('mldb,mdla->ab',t2_2_ab, eris_ovOV)

        M_ab_a += np.einsum('l,lmbd,lmad->ab',e_occ_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a += np.einsum('l,lmbd,lmad->ab',e_occ_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a += np.einsum('l,mlbd,mlad->ab',e_occ_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b += np.einsum('l,lmbd,lmad->ab',e_occ_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b += np.einsum('l,mldb,mlda->ab',e_occ_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b += np.einsum('l,lmdb,lmda->ab',e_occ_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a += np.einsum('l,lmad,lmbd->ab',e_occ_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a += np.einsum('l,lmad,lmbd->ab',e_occ_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a += np.einsum('l,mlad,mlbd->ab',e_occ_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b += np.einsum('l,lmad,lmbd->ab',e_occ_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b += np.einsum('l,mlda,mldb->ab',e_occ_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b += np.einsum('l,lmda,lmdb->ab',e_occ_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a -= 0.5*np.einsum('d,lmbd,lmad->ab', e_vir_a, t2_1_a ,t2_2_a, optimize=True)
        M_ab_a -= 0.5*np.einsum('d,lmbd,lmad->ab', e_vir_b, t2_1_ab ,t2_2_ab, optimize=True)
        M_ab_a -= 0.5*np.einsum('d,mlbd,mlad->ab', e_vir_b, t2_1_ab ,t2_2_ab, optimize=True)

        M_ab_b -= 0.5*np.einsum('d,lmbd,lmad->ab', e_vir_b, t2_1_b ,t2_2_b, optimize=True)
        M_ab_b -= 0.5*np.einsum('d,mldb,mlda->ab', e_vir_a, t2_1_ab ,t2_2_ab, optimize=True)
        M_ab_b -= 0.5*np.einsum('d,lmdb,lmda->ab', e_vir_a, t2_1_ab ,t2_2_ab, optimize=True)

        M_ab_a -= 0.5*np.einsum('d,lmad,lmbd->ab', e_vir_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a -= 0.5*np.einsum('d,lmad,lmbd->ab', e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a -= 0.5*np.einsum('d,mlad,mlbd->ab', e_vir_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b -= 0.5*np.einsum('d,lmad,lmbd->ab', e_vir_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b -= 0.5*np.einsum('d,mlda,mldb->ab', e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b -= 0.5*np.einsum('d,lmda,lmdb->ab', e_vir_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a -= 0.25*np.einsum('a,lmbd,lmad->ab',e_vir_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a -= 0.25*np.einsum('a,lmbd,lmad->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a -= 0.25*np.einsum('a,mlbd,mlad->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b -= 0.25*np.einsum('a,lmbd,lmad->ab',e_vir_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b -= 0.25*np.einsum('a,mldb,mlda->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b -= 0.25*np.einsum('a,lmdb,lmda->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a -= 0.25*np.einsum('a,lmad,lmbd->ab',e_vir_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a -= 0.25*np.einsum('a,lmad,lmbd->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a -= 0.25*np.einsum('a,mlad,mlbd->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b -= 0.25*np.einsum('a,lmad,lmbd->ab',e_vir_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b -= 0.25*np.einsum('a,mlda,mldb->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b -= 0.25*np.einsum('a,lmda,lmdb->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a -= 0.25*np.einsum('b,lmbd,lmad->ab',e_vir_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a -= 0.25*np.einsum('b,lmbd,lmad->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a -= 0.25*np.einsum('b,mlbd,mlad->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b -= 0.25*np.einsum('b,lmbd,lmad->ab',e_vir_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b -= 0.25*np.einsum('b,mldb,mlda->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b -= 0.25*np.einsum('b,lmdb,lmda->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a -= 0.25*np.einsum('b,lmad,lmbd->ab',e_vir_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a -= 0.25*np.einsum('b,lmad,lmbd->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a -= 0.25*np.einsum('b,mlad,mlbd->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b -= 0.25*np.einsum('b,lmad,lmbd->ab',e_vir_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b -= 0.25*np.einsum('b,mlda,mldb->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b -= 0.25*np.einsum('b,lmda,lmdb->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a -= np.einsum('lned,mlbd,nmae->ab',t2_1_a, t2_1_a, eris_oovv, optimize=True)
        M_ab_a += np.einsum('lned,mlbd,mane->ab',t2_1_a, t2_1_a, eris_ovov, optimize=True)
        M_ab_a += np.einsum('nled,mlbd,nmae->ab',t2_1_ab, t2_1_ab, eris_oovv, optimize=True)
        M_ab_a -= np.einsum('nled,mlbd,mane->ab',t2_1_ab, t2_1_ab, eris_ovov, optimize=True)
        M_ab_a -= np.einsum('lnde,mlbd,neam->ab',t2_1_ab, t2_1_a, eris_OVvo, optimize=True)
        M_ab_a += np.einsum('lned,mlbd,neam->ab',t2_1_b, t2_1_ab, eris_OVvo, optimize=True)
        M_ab_a += np.einsum('lned,lmbd,nmae->ab',t2_1_ab, t2_1_ab, eris_OOvv, optimize=True)

        M_ab_b -= np.einsum('lned,mlbd,nmae->ab',t2_1_b, t2_1_b, eris_OOVV, optimize=True)
        M_ab_b += np.einsum('lned,mlbd,mane->ab',t2_1_b, t2_1_b, eris_OVOV, optimize=True)
        M_ab_b += np.einsum('lnde,lmdb,nmae->ab',t2_1_ab, t2_1_ab, eris_OOVV, optimize=True)
        M_ab_b -= np.einsum('lnde,lmdb,mane->ab',t2_1_ab, t2_1_ab, eris_OVOV, optimize=True)
        M_ab_b -= np.einsum('nled,mlbd,neam->ab',t2_1_ab, t2_1_b, eris_ovVO, optimize=True)
        M_ab_b += np.einsum('lned,lmdb,neam->ab',t2_1_a, t2_1_ab, eris_ovVO, optimize=True)
        M_ab_b += np.einsum('nlde,mldb,nmae->ab',t2_1_ab, t2_1_ab, eris_ooVV, optimize=True)

        M_ab_a -= np.einsum('mled,lnad,nmeb->ab',t2_1_a, t2_1_a, eris_oovv, optimize=True)
        M_ab_a += np.einsum('mled,lnad,nbem->ab',t2_1_a, t2_1_a, eris_ovvo, optimize=True)
        M_ab_a += np.einsum('mled,nlad,nmeb->ab',t2_1_ab, t2_1_ab, eris_oovv, optimize=True)
        M_ab_a -= np.einsum('mled,nlad,nbem->ab',t2_1_ab, t2_1_ab, eris_ovvo, optimize=True)
        M_ab_a += np.einsum('lmed,lnad,nmeb->ab',t2_1_ab, t2_1_ab, eris_OOvv, optimize=True)
        M_ab_a -= np.einsum('mled,nlad,nbem->ab',t2_1_b, t2_1_ab, eris_ovVO, optimize=True)
        M_ab_a += np.einsum('lmde,lnad,nbem->ab',t2_1_ab, t2_1_a, eris_ovVO, optimize=True)

        M_ab_b -= np.einsum('mled,lnad,nmeb->ab',t2_1_b, t2_1_b, eris_OOVV, optimize=True)
        M_ab_b += np.einsum('mled,lnad,nbem->ab',t2_1_b, t2_1_b, eris_OVVO, optimize=True)
        M_ab_b += np.einsum('lmde,lnda,nmeb->ab',t2_1_ab, t2_1_ab, eris_OOVV, optimize=True)
        M_ab_b -= np.einsum('lmde,lnda,nbem->ab',t2_1_ab, t2_1_ab, eris_OVVO, optimize=True)
        M_ab_b += np.einsum('mlde,nlda,nmeb->ab',t2_1_ab, t2_1_ab, eris_ooVV, optimize=True)
        M_ab_b -= np.einsum('mled,lnda,nbem->ab',t2_1_a, t2_1_ab, eris_OVvo, optimize=True)
        M_ab_b += np.einsum('mled,lnad,nbem->ab',t2_1_ab, t2_1_b, eris_OVvo, optimize=True)

        M_ab_a -= np.einsum('mlbd,lnae,nmde->ab',t2_1_a, t2_1_a,   eris_oovv, optimize=True)
        M_ab_a += np.einsum('mlbd,lnae,nedm->ab',t2_1_a, t2_1_a,   eris_ovvo, optimize=True)
        M_ab_a += np.einsum('lmbd,lnae,nmde->ab',t2_1_ab, t2_1_ab, eris_OOVV, optimize=True)
        M_ab_a -= np.einsum('lmbd,lnae,nedm->ab',t2_1_ab, t2_1_ab, eris_OVVO, optimize=True)
        M_ab_a += np.einsum('mlbd,lnae,nedm->ab',t2_1_a, t2_1_ab,  eris_OVvo, optimize=True)
        M_ab_a -= np.einsum('lmbd,lnae,nedm->ab',t2_1_ab, t2_1_a,  eris_ovVO, optimize=True)
        M_ab_a += np.einsum('mlbd,nlae,nmde->ab',t2_1_ab, t2_1_ab, eris_ooVV, optimize=True)

        M_ab_b -= np.einsum('mlbd,lnae,nmde->ab',t2_1_b, t2_1_b, eris_OOVV, optimize=True)
        M_ab_b += np.einsum('mlbd,lnae,nedm->ab',t2_1_b, t2_1_b, eris_OVVO, optimize=True)
        M_ab_b += np.einsum('mldb,nlea,nmde->ab',t2_1_ab, t2_1_ab, eris_oovv, optimize=True)
        M_ab_b -= np.einsum('mldb,nlea,nedm->ab',t2_1_ab, t2_1_ab, eris_ovvo, optimize=True)
        M_ab_b += np.einsum('mlbd,nlea,nedm->ab',t2_1_b, t2_1_ab,  eris_ovVO, optimize=True)
        M_ab_b -= np.einsum('mldb,lnae,nedm->ab',t2_1_ab, t2_1_b,  eris_OVvo, optimize=True)
        M_ab_b += np.einsum('lmdb,lnea,nmed->ab',t2_1_ab, t2_1_ab, eris_OOvv, optimize=True)

        M_ab_a += 0.5*np.einsum('lned,mled,nmab->ab',t2_1_a, t2_1_a, eris_oovv, optimize=True)
        M_ab_a -= 0.5*np.einsum('lned,mled,nbam->ab',t2_1_a, t2_1_a, eris_ovvo, optimize=True)
        M_ab_a -= np.einsum('nled,mled,nmab->ab',t2_1_ab, t2_1_ab, eris_oovv, optimize=True)
        M_ab_a += np.einsum('nled,mled,nbam->ab',t2_1_ab, t2_1_ab, eris_ovvo, optimize=True)
        M_ab_a += 0.5*np.einsum('lned,mled,nmab->ab',t2_1_b, t2_1_b, eris_OOvv, optimize=True)
        M_ab_a -= np.einsum('lned,lmed,nmab->ab',t2_1_ab, t2_1_ab, eris_OOvv, optimize=True)

        M_ab_b += 0.5*np.einsum('lned,mled,nmab->ab',t2_1_b, t2_1_b, eris_OOVV, optimize=True)
        M_ab_b -= 0.5*np.einsum('lned,mled,nbam->ab',t2_1_b, t2_1_b, eris_OVVO, optimize=True)
        M_ab_b -= np.einsum('lned,lmed,nmab->ab',t2_1_ab, t2_1_ab, eris_OOVV, optimize=True)
        M_ab_b += np.einsum('lned,lmed,nbam->ab',t2_1_ab, t2_1_ab, eris_OVVO, optimize=True)
        M_ab_b += 0.5*np.einsum('lned,mled,nmab->ab',t2_1_a, t2_1_a, eris_ooVV, optimize=True)
        M_ab_b -= np.einsum('nled,mled,nmab->ab',t2_1_ab, t2_1_ab, eris_ooVV, optimize=True)

        M_ab_a -= 0.25*np.einsum('mlbd,noad,nmol->ab',t2_1_a, t2_1_a, eris_oooo, optimize=True)
        M_ab_a += 0.25*np.einsum('mlbd,noad,nlom->ab',t2_1_a, t2_1_a, eris_oooo, optimize=True)
        M_ab_a -= np.einsum('mlbd,noad,nmol->ab',t2_1_ab, t2_1_ab, eris_ooOO, optimize=True)

        M_ab_b -= 0.25*np.einsum('mlbd,noad,nmol->ab',t2_1_b, t2_1_b, eris_OOOO, optimize=True)
        M_ab_b += 0.25*np.einsum('mlbd,noad,nlom->ab',t2_1_b, t2_1_b, eris_OOOO, optimize=True)
        M_ab_b -= np.einsum('lmdb,onda,olnm->ab',t2_1_ab, t2_1_ab, eris_ooOO, optimize=True)

        eris_vvvv = uadc_ao2mo.unpack_eri_2s(eris.vvvv, nvir_a)
        M_ab_a -= 0.25*np.einsum('mlef,mlbd,aedf->ab',t2_1_a, t2_1_a, eris_vvvv, optimize=True)
        M_ab_a += 0.25*np.einsum('mlef,mlbd,afde->ab',t2_1_a, t2_1_a, eris_vvvv, optimize=True)
        M_ab_a -= 0.25*np.einsum('mled,mlaf,ebdf->ab',t2_1_a, t2_1_a, eris_vvvv, optimize=True)
        M_ab_a += 0.25*np.einsum('mled,mlaf,efdb->ab',t2_1_a, t2_1_a, eris_vvvv, optimize=True)
        M_ab_a -= 0.5*np.einsum('mldf,mled,abef->ab',t2_1_a, t2_1_a, eris_vvvv, optimize=True)
        M_ab_a += 0.5*np.einsum('mldf,mled,afeb->ab',t2_1_a, t2_1_a, eris_vvvv, optimize=True)
        M_ab_a += np.einsum('mlfd,mled,abef->ab',t2_1_ab, t2_1_ab, eris_vvvv, optimize=True)
        M_ab_a -= np.einsum('mlfd,mled,afeb->ab',t2_1_ab, t2_1_ab, eris_vvvv, optimize=True)
        del eris_vvvv

        eris_VVVV = uadc_ao2mo.unpack_eri_2s(eris.VVVV, nvir_b)
        M_ab_b -= 0.25*np.einsum('mlef,mlbd,aedf->ab',t2_1_b, t2_1_b, eris_VVVV, optimize=True)
        M_ab_b += 0.25*np.einsum('mlef,mlbd,afde->ab',t2_1_b, t2_1_b, eris_VVVV, optimize=True)
        M_ab_b -= 0.25*np.einsum('mled,mlaf,ebdf->ab',t2_1_b, t2_1_b, eris_VVVV, optimize=True)
        M_ab_b += 0.25*np.einsum('mled,mlaf,efdb->ab',t2_1_b, t2_1_b, eris_VVVV, optimize=True)
        M_ab_b -= 0.5*np.einsum('mldf,mled,abef->ab',t2_1_b, t2_1_b, eris_VVVV, optimize=True)
        M_ab_b += 0.5*np.einsum('mldf,mled,afeb->ab',t2_1_b, t2_1_b, eris_VVVV, optimize=True)
        M_ab_b += np.einsum('mldf,mlde,abef->ab',t2_1_ab, t2_1_ab, eris_VVVV, optimize=True)
        M_ab_b -= np.einsum('mldf,mlde,afeb->ab',t2_1_ab, t2_1_ab, eris_VVVV, optimize=True)
        del eris_VVVV

        eris_vvVV = uadc_ao2mo.unpack_eri_2(eris.vvVV, nvir_a, nvir_b)
        M_ab_a -= np.einsum('mlef,mlbd,aedf->ab',t2_1_ab, t2_1_ab,   eris_vvVV, optimize=True)
        M_ab_a -= np.einsum('mled,mlaf,ebdf->ab',t2_1_ab, t2_1_ab,   eris_vvVV, optimize=True)
        M_ab_a -= 0.5*np.einsum('mldf,mled,abef->ab',t2_1_b, t2_1_b, eris_vvVV, optimize=True)
        M_ab_a += np.einsum('mldf,mlde,abef->ab',t2_1_ab, t2_1_ab,   eris_vvVV, optimize=True)

        M_ab_b -= np.einsum('mlef,mldb,deaf->ab',t2_1_ab, t2_1_ab,   eris_vvVV, optimize=True)
        M_ab_b -= np.einsum('mled,mlfa,efdb->ab',t2_1_ab, t2_1_ab,   eris_vvVV, optimize=True)
        M_ab_b -= 0.5*np.einsum('mldf,mled,efab->ab',t2_1_a, t2_1_a, eris_vvVV, optimize=True)
        M_ab_b += np.einsum('mlfd,mled,efab->ab',t2_1_ab, t2_1_ab,   eris_vvVV, optimize=True)
        del eris_vvVV

    M_ab = (M_ab_a, M_ab_b)

    return M_ab


def get_imds_ip(adc, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t1 = adc.t1
    t2 = adc.t2

    t1_2_a, t1_2_b = t1[0]
    t2_1_a, t2_1_ab, t2_1_b = t2[0]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    if eris is None:
        eris = uadc_ao2mo.transform_integrals_incore(adc)

    eris_ovov = eris.ovov
    eris_OVOV = eris.OVOV
    eris_ovOV = eris.ovOV
    eris_OVov = eris.OVov

    # i-j block
    # Zeroth-order terms

    M_ij_a = np.einsum('ij,j->ij', idn_occ_a ,e_occ_a)
    M_ij_b = np.einsum('ij,j->ij', idn_occ_b ,e_occ_b)

    # Second-order terms

    M_ij_a +=  np.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_a, t2_1_a)
    M_ij_a +=  np.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_ab, t2_1_ab)
    M_ij_a +=  np.einsum('d,iled,jled->ij',e_vir_b,t2_1_ab, t2_1_ab)

    M_ij_b +=  np.einsum('d,ilde,jlde->ij',e_vir_b,t2_1_b, t2_1_b)
    M_ij_b +=  np.einsum('d,lide,ljde->ij',e_vir_a,t2_1_ab, t2_1_ab)
    M_ij_b +=  np.einsum('d,lied,ljed->ij',e_vir_b,t2_1_ab, t2_1_ab)

    M_ij_a -= 0.5 *  np.einsum('l,ilde,jlde->ij',e_occ_a,t2_1_a, t2_1_a)
    M_ij_a -= 0.5*np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_1_ab)
    M_ij_a -= 0.5*np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_1_ab)

    M_ij_b -= 0.5 *  np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_b, t2_1_b)
    M_ij_b -= 0.5*np.einsum('l,lide,ljde->ij',e_occ_a,t2_1_ab, t2_1_ab)
    M_ij_b -= 0.5*np.einsum('l,lied,ljed->ij',e_occ_a,t2_1_ab, t2_1_ab)

    M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_a, t2_1_a)
    M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab)
    M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab)

    M_ij_b -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_b,t2_1_b, t2_1_b)
    M_ij_b -= 0.25 *  np.einsum('i,lied,ljed->ij',e_occ_b,t2_1_ab, t2_1_ab)
    M_ij_b -= 0.25 *  np.einsum('i,lide,ljde->ij',e_occ_b,t2_1_ab, t2_1_ab)

    M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_a, t2_1_a)
    M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab)
    M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab)

    M_ij_b -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_b,t2_1_b, t2_1_b)
    M_ij_b -= 0.25 *  np.einsum('j,lied,ljed->ij',e_occ_b,t2_1_ab, t2_1_ab)
    M_ij_b -= 0.25 *  np.einsum('j,lide,ljde->ij',e_occ_b,t2_1_ab, t2_1_ab)

    M_ij_a += 0.5 *  np.einsum('ilde,jdle->ij',t2_1_a, eris_ovov)
    M_ij_a -= 0.5 *  np.einsum('ilde,jeld->ij',t2_1_a, eris_ovov)
    M_ij_a += np.einsum('ilde,jdle->ij',t2_1_ab, eris_ovOV)

    M_ij_b += 0.5 *  np.einsum('ilde,jdle->ij',t2_1_b, eris_OVOV)
    M_ij_b -= 0.5 *  np.einsum('ilde,jeld->ij',t2_1_b, eris_OVOV)
    M_ij_b += np.einsum('lied,lejd->ij',t2_1_ab, eris_ovOV)

    M_ij_a += 0.5 *  np.einsum('jlde,idle->ij',t2_1_a, eris_ovov)
    M_ij_a -= 0.5 *  np.einsum('jlde,ldie->ij',t2_1_a, eris_ovov)
    M_ij_a += np.einsum('jlde,idle->ij',t2_1_ab, eris_ovOV)

    M_ij_b += 0.5 *  np.einsum('jlde,idle->ij',t2_1_b, eris_OVOV)
    M_ij_b -= 0.5 *  np.einsum('jlde,ldie->ij',t2_1_b, eris_OVOV)
    M_ij_b += np.einsum('ljed,leid->ij',t2_1_ab, eris_ovOV)

    # Third-order terms

    if (method == "adc(3)"):

        t2_2_a, t2_2_ab, t2_2_b = t2[1]
        eris_oovv = eris.oovv
        eris_OOVV = eris.OOVV
        eris_ooVV = eris.ooVV
        eris_OOvv = eris.OOvv
        eris_ovvo = eris.ovvo
        eris_OVVO = eris.OVVO
        eris_ovVO = eris.ovVO
        eris_OVvo = eris.OVvo
        eris_ovoo = eris.ovoo
        eris_OVOO = eris.OVOO
        eris_ovOO = eris.ovOO
        eris_OVoo = eris.OVoo
        eris_oooo = eris.oooo
        eris_OOOO = eris.OOOO
        eris_ooOO = eris.ooOO

        M_ij_a += np.einsum('ld,ldji->ij',t1_2_a, eris_ovoo)
        M_ij_a -= np.einsum('ld,jdli->ij',t1_2_a, eris_ovoo)
        M_ij_a += np.einsum('ld,ldji->ij',t1_2_b, eris_OVoo)

        M_ij_b += np.einsum('ld,ldji->ij',t1_2_b, eris_OVOO)
        M_ij_b -= np.einsum('ld,jdli->ij',t1_2_b, eris_OVOO)
        M_ij_b += np.einsum('ld,ldji->ij',t1_2_a, eris_ovOO)

        M_ij_a += np.einsum('ld,ldij->ij',t1_2_a, eris_ovoo)
        M_ij_a -= np.einsum('ld,idlj->ij',t1_2_a, eris_ovoo)
        M_ij_a += np.einsum('ld,ldij->ij',t1_2_b, eris_OVoo)

        M_ij_b += np.einsum('ld,ldij->ij',t1_2_b, eris_OVOO)
        M_ij_b -= np.einsum('ld,idlj->ij',t1_2_b, eris_OVOO)
        M_ij_b += np.einsum('ld,ldij->ij',t1_2_a, eris_ovOO)

        M_ij_a += 0.5* np.einsum('ilde,jdle->ij',t2_2_a, eris_ovov)
        M_ij_a -= 0.5* np.einsum('ilde,jeld->ij',t2_2_a, eris_ovov)
        M_ij_a += np.einsum('ilde,jdle->ij',t2_2_ab, eris_ovOV)

        M_ij_b += 0.5* np.einsum('ilde,jdle->ij',t2_2_b, eris_OVOV)
        M_ij_b -= 0.5* np.einsum('ilde,jeld->ij',t2_2_b, eris_OVOV)
        M_ij_b += np.einsum('lied,lejd->ij',t2_2_ab, eris_ovOV)

        M_ij_a += 0.5* np.einsum('jlde,leid->ij',t2_2_a, eris_ovov)
        M_ij_a -= 0.5* np.einsum('jlde,ield->ij',t2_2_a, eris_ovov)
        M_ij_a += np.einsum('jlde,leid->ij',t2_2_ab, eris_OVov)

        M_ij_b += 0.5* np.einsum('jlde,leid->ij',t2_2_b, eris_OVOV)
        M_ij_b -= 0.5* np.einsum('jlde,ield->ij',t2_2_b, eris_OVOV)
        M_ij_b += np.einsum('ljed,leid->ij',t2_2_ab, eris_ovOV)

        M_ij_a +=  np.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a +=  np.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a +=  np.einsum('d,iled,jled->ij',e_vir_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b +=  np.einsum('d,ilde,jlde->ij',e_vir_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b +=  np.einsum('d,lide,ljde->ij',e_vir_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b +=  np.einsum('d,lied,ljed->ij',e_vir_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a +=  np.einsum('d,jlde,ilde->ij',e_vir_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a +=  np.einsum('d,jlde,ilde->ij',e_vir_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a +=  np.einsum('d,jled,iled->ij',e_vir_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b +=  np.einsum('d,jlde,ilde->ij',e_vir_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b +=  np.einsum('d,ljde,lide->ij',e_vir_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b +=  np.einsum('d,ljed,lied->ij',e_vir_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.5 *  np.einsum('l,ilde,jlde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a -= 0.5*np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.5*np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b -= 0.5 *  np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.5*np.einsum('l,lied,ljed->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.5*np.einsum('l,lied,ljed->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.5 *  np.einsum('l,jlde,ilde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a -= 0.5*np.einsum('l,jlde,ilde->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.5*np.einsum('l,jlde,ilde->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b -= 0.5 *  np.einsum('l,jlde,ilde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.5*np.einsum('l,ljed,lied->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.5*np.einsum('l,ljed,lied->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.25 *  np.einsum('i,lied,ljed->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.25 *  np.einsum('i,lied,ljed->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.25 *  np.einsum('i,jlde,ilde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a -= 0.25 *  np.einsum('i,jlde,ilde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.25 *  np.einsum('i,jlde,ilde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b -= 0.25 *  np.einsum('i,jlde,ilde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.25 *  np.einsum('i,ljed,lied->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.25 *  np.einsum('i,ljed,lied->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.25 *  np.einsum('j,jlde,ilde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a -= 0.25 *  np.einsum('j,jlde,ilde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.25 *  np.einsum('j,jlde,ilde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b -= 0.25 *  np.einsum('j,jlde,ilde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.25 *  np.einsum('j,ljed,lied->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.25 *  np.einsum('j,ljed,lied->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.25 *  np.einsum('j,lied,ljed->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.25 *  np.einsum('j,lied,ljed->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= np.einsum('lmde,jldf,mefi->ij',t2_1_a, t2_1_a, eris_ovvo,optimize = True)
        M_ij_a += np.einsum('lmde,jldf,mife->ij',t2_1_a, t2_1_a, eris_oovv,optimize = True)
        M_ij_a += np.einsum('mled,jlfd,mefi->ij',t2_1_ab, t2_1_ab, eris_ovvo ,optimize = True)
        M_ij_a -= np.einsum('mled,jlfd,mife->ij',t2_1_ab, t2_1_ab, eris_oovv ,optimize = True)
        M_ij_a -= np.einsum('lmde,jldf,mefi->ij',t2_1_ab, t2_1_a, eris_OVvo,optimize = True)
        M_ij_a -= np.einsum('mlde,jldf,mife->ij',t2_1_ab, t2_1_ab, eris_ooVV ,optimize = True)
        M_ij_a += np.einsum('lmde,jlfd,mefi->ij',t2_1_b, t2_1_ab, eris_OVvo ,optimize = True)

        M_ij_b -= np.einsum('lmde,jldf,mefi->ij',t2_1_b, t2_1_b, eris_OVVO,optimize = True)
        M_ij_b += np.einsum('lmde,jldf,mife->ij',t2_1_b, t2_1_b, eris_OOVV,optimize = True)
        M_ij_b += np.einsum('lmde,ljdf,mefi->ij',t2_1_ab, t2_1_ab, eris_OVVO,optimize = True)
        M_ij_b -= np.einsum('lmde,ljdf,mife->ij',t2_1_ab, t2_1_ab, eris_OOVV,optimize = True)
        M_ij_b -= np.einsum('mled,jldf,mefi->ij',t2_1_ab, t2_1_b, eris_ovVO,optimize = True)
        M_ij_b -= np.einsum('lmed,ljfd,mife->ij',t2_1_ab, t2_1_ab, eris_OOvv ,optimize = True)
        M_ij_b += np.einsum('lmde,ljdf,mefi->ij',t2_1_a, t2_1_ab, eris_ovVO ,optimize = True)

        M_ij_a -= np.einsum('lmde,ildf,mefj->ij',t2_1_a, t2_1_a, eris_ovvo ,optimize = True)
        M_ij_a += np.einsum('lmde,ildf,mjfe->ij',t2_1_a, t2_1_a, eris_oovv ,optimize = True)
        M_ij_a += np.einsum('mled,ilfd,mefj->ij',t2_1_ab, t2_1_ab, eris_ovvo ,optimize = True)
        M_ij_a -= np.einsum('mled,ilfd,mjfe->ij',t2_1_ab, t2_1_ab, eris_oovv ,optimize = True)
        M_ij_a -= np.einsum('lmde,ildf,mefj->ij',t2_1_ab, t2_1_a, eris_OVvo,optimize = True)
        M_ij_a -= np.einsum('mlde,ildf,mjfe->ij',t2_1_ab, t2_1_ab, eris_ooVV ,optimize = True)
        M_ij_a += np.einsum('lmde,ilfd,mefj->ij',t2_1_b, t2_1_ab, eris_OVvo ,optimize = True)

        M_ij_b -= np.einsum('lmde,ildf,mefj->ij',t2_1_b, t2_1_b, eris_OVVO ,optimize = True)
        M_ij_b += np.einsum('lmde,ildf,mjfe->ij',t2_1_b, t2_1_b, eris_OOVV ,optimize = True)
        M_ij_b += np.einsum('lmde,lidf,mefj->ij',t2_1_ab, t2_1_ab, eris_OVVO ,optimize = True)
        M_ij_b -= np.einsum('lmde,lidf,mjfe->ij',t2_1_ab, t2_1_ab, eris_OOVV ,optimize = True)
        M_ij_b -= np.einsum('mled,ildf,mefj->ij',t2_1_ab, t2_1_b, eris_ovVO,optimize = True)
        M_ij_b -= np.einsum('lmed,lifd,mjfe->ij',t2_1_ab, t2_1_ab, eris_OOvv ,optimize = True)
        M_ij_b += np.einsum('lmde,lidf,mefj->ij',t2_1_a, t2_1_ab, eris_ovVO ,optimize = True)

        M_ij_a += 0.25*np.einsum('lmde,jnde,limn->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)
        M_ij_a -= 0.25*np.einsum('lmde,jnde,lnmi->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)
        M_ij_a += np.einsum('lmde,jnde,limn->ij',t2_1_ab ,t2_1_ab,eris_ooOO, optimize = True)

        M_ij_b += 0.25*np.einsum('lmde,jnde,limn->ij',t2_1_b, t2_1_b,eris_OOOO, optimize = True)
        M_ij_b -= 0.25*np.einsum('lmde,jnde,lnmi->ij',t2_1_b, t2_1_b,eris_OOOO, optimize = True)
        M_ij_b += np.einsum('mled,njed,mnli->ij',t2_1_ab ,t2_1_ab,eris_ooOO, optimize = True)

        eris_vvvv = uadc_ao2mo.unpack_eri_2s(eris.vvvv, nvir_a)
        M_ij_a += 0.25*np.einsum('ilde,jlgf,gdfe->ij',t2_1_a, t2_1_a, eris_vvvv, optimize = True)
        M_ij_a -= 0.25*np.einsum('ilde,jlgf,gefd->ij',t2_1_a, t2_1_a, eris_vvvv, optimize = True)
        del eris_vvvv

        eris_VVVV = uadc_ao2mo.unpack_eri_2s(eris.VVVV, nvir_b)
        M_ij_b += 0.25*np.einsum('ilde,jlgf,gdfe->ij',t2_1_b, t2_1_b, eris_VVVV, optimize = True)
        M_ij_b -= 0.25*np.einsum('ilde,jlgf,gefd->ij',t2_1_b, t2_1_b, eris_VVVV, optimize = True)
        del eris_VVVV

        eris_vvVV = uadc_ao2mo.unpack_eri_2(eris.vvVV, nvir_a, nvir_b)
        M_ij_a +=np.einsum('ilde,jlgf,gdfe->ij',t2_1_ab, t2_1_ab,eris_vvVV, optimize = True)
        M_ij_b +=np.einsum('lied,ljfg,fegd->ij',t2_1_ab, t2_1_ab,eris_vvVV, optimize = True)
        del eris_vvVV

        M_ij_a += 0.25*np.einsum('inde,lmde,jlnm->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)
        M_ij_a -= 0.25*np.einsum('inde,lmde,jmnl->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)
        M_ij_a +=np.einsum('inde,lmde,jlnm->ij',t2_1_ab, t2_1_ab,eris_ooOO, optimize = True)

        M_ij_b += 0.25*np.einsum('inde,lmde,jlnm->ij',t2_1_b, t2_1_b,eris_OOOO, optimize = True)
        M_ij_b -= 0.25*np.einsum('inde,lmde,jmnl->ij',t2_1_b, t2_1_b,eris_OOOO, optimize = True)
        M_ij_b +=np.einsum('nied,mled,nmjl->ij',t2_1_ab, t2_1_ab,eris_ooOO, optimize = True)

        M_ij_a += 0.5*np.einsum('lmdf,lmde,jief->ij',t2_1_a, t2_1_a, eris_oovv, optimize = True)
        M_ij_a -= 0.5*np.einsum('lmdf,lmde,jfei->ij',t2_1_a, t2_1_a, eris_ovvo, optimize = True)
        M_ij_a +=np.einsum('mlfd,mled,jief->ij',t2_1_ab, t2_1_ab, eris_oovv , optimize = True)
        M_ij_a -=np.einsum('mlfd,mled,jfei->ij',t2_1_ab, t2_1_ab, eris_ovvo , optimize = True)
        M_ij_a +=np.einsum('lmdf,lmde,jief->ij',t2_1_ab, t2_1_ab, eris_ooVV , optimize = True)
        M_ij_a +=0.5*np.einsum('lmdf,lmde,jief->ij',t2_1_b, t2_1_b, eris_ooVV , optimize = True)

        M_ij_b += 0.5*np.einsum('lmdf,lmde,jief->ij',t2_1_b, t2_1_b, eris_OOVV , optimize = True)
        M_ij_b -= 0.5*np.einsum('lmdf,lmde,jfei->ij',t2_1_b, t2_1_b, eris_OVVO , optimize = True)
        M_ij_b +=np.einsum('lmdf,lmde,jief->ij',t2_1_ab, t2_1_ab, eris_OOVV , optimize = True)
        M_ij_b -=np.einsum('lmdf,lmde,jfei->ij',t2_1_ab, t2_1_ab, eris_OVVO , optimize = True)
        M_ij_b +=np.einsum('lmfd,lmed,jief->ij',t2_1_ab, t2_1_ab, eris_OOvv , optimize = True)
        M_ij_b +=0.5*np.einsum('lmdf,lmde,jief->ij',t2_1_a, t2_1_a, eris_OOvv , optimize = True)

        M_ij_a -= np.einsum('ilde,jmdf,lmfe->ij',t2_1_a, t2_1_a, eris_oovv, optimize = True)
        M_ij_a += np.einsum('ilde,jmdf,lefm->ij',t2_1_a, t2_1_a, eris_ovvo, optimize = True)
        M_ij_a += np.einsum('ilde,jmdf,lefm->ij',t2_1_a, t2_1_ab, eris_ovVO, optimize = True)
        M_ij_a += np.einsum('ilde,jmdf,lefm->ij',t2_1_ab, t2_1_a, eris_OVvo, optimize = True)
        M_ij_a -= np.einsum('ilde,jmdf,lmfe->ij',t2_1_ab, t2_1_ab, eris_OOVV, optimize = True)
        M_ij_a += np.einsum('ilde,jmdf,lefm->ij',t2_1_ab, t2_1_ab, eris_OVVO, optimize = True)
        M_ij_a -= np.einsum('iled,jmfd,lmfe->ij',t2_1_ab, t2_1_ab, eris_OOvv, optimize = True)

        M_ij_b -= np.einsum('ilde,jmdf,lmfe->ij',t2_1_b, t2_1_b, eris_OOVV, optimize = True)
        M_ij_b += np.einsum('ilde,jmdf,lefm->ij',t2_1_b, t2_1_b, eris_OVVO, optimize = True)
        M_ij_b += np.einsum('ilde,mjfd,lefm->ij',t2_1_b, t2_1_ab, eris_OVvo, optimize = True)
        M_ij_b += np.einsum('lied,jmdf,lefm->ij',t2_1_ab, t2_1_b, eris_ovVO, optimize = True)
        M_ij_b -= np.einsum('lied,mjfd,lmfe->ij',t2_1_ab, t2_1_ab, eris_oovv, optimize = True)
        M_ij_b += np.einsum('lied,mjfd,lefm->ij',t2_1_ab, t2_1_ab, eris_ovvo, optimize = True)
        M_ij_b -= np.einsum('lide,mjdf,lmfe->ij',t2_1_ab, t2_1_ab, eris_ooVV, optimize = True)

        M_ij_a -= 0.5*np.einsum('lnde,lmde,jinm->ij',t2_1_a, t2_1_a, eris_oooo, optimize = True)
        M_ij_a += 0.5*np.einsum('lnde,lmde,jmni->ij',t2_1_a, t2_1_a, eris_oooo, optimize = True)
        M_ij_a -= np.einsum('nled,mled,jinm->ij',t2_1_ab, t2_1_ab, eris_oooo, optimize = True)
        M_ij_a += np.einsum('nled,mled,jmni->ij',t2_1_ab, t2_1_ab, eris_oooo, optimize = True)
        M_ij_a -= np.einsum('lnde,lmde,jinm->ij',t2_1_ab, t2_1_ab, eris_ooOO, optimize = True)
        M_ij_a -= 0.5 * np.einsum('lnde,lmde,jinm->ij',t2_1_b, t2_1_b, eris_ooOO, optimize = True)

        M_ij_b -= 0.5*np.einsum('lnde,lmde,jinm->ij',t2_1_b, t2_1_b, eris_OOOO, optimize = True)
        M_ij_b += 0.5*np.einsum('lnde,lmde,jmni->ij',t2_1_b, t2_1_b, eris_OOOO, optimize = True)
        M_ij_b -= np.einsum('lnde,lmde,jinm->ij',t2_1_ab, t2_1_ab, eris_OOOO, optimize = True)
        M_ij_b += np.einsum('lnde,lmde,jmni->ij',t2_1_ab, t2_1_ab, eris_OOOO, optimize = True)
        M_ij_b -= np.einsum('nled,mled,nmji->ij',t2_1_ab, t2_1_ab, eris_ooOO, optimize = True)
        M_ij_b -= 0.5 * np.einsum('lnde,lmde,nmji->ij',t2_1_a, t2_1_a, eris_ooOO, optimize = True)

    M_ij = (M_ij_a, M_ij_b)

    return M_ij


def ea_adc_diag(adc,M_ab=None):

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

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

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

    return diag


def ip_adc_diag(adc,M_ij=None):

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

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

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

    diag = -diag
    return diag


def ea_adc_matvec(adc, M_ab=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2_1_a, t2_1_ab, t2_1_b = adc.t2[0]
    t1_2_a, t1_2_b = adc.t1[0]

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

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    if eris is None:
        eris = uadc_ao2mo.transform_integrals_incore(adc)

    eris_ovov = eris.ovov
    eris_OVOV = eris.OVOV
    eris_ovOV = eris.ovOV
    eris_OVov = eris.OVov

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

        s = None
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

        s[s_a:f_a] = np.einsum('ab,b->a',M_ab_a,r_a)
        s[s_b:f_b] = np.einsum('ab,b->a',M_ab_b,r_b)

############# ADC(2) a - ibc and ibc - a coupling blocks #########################

        eris_ovvv = uadc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
        s[s_a:f_a] += 0.5*np.einsum('icab,ibc->a',eris_ovvv, r_aaa_, optimize = True)
        s[s_a:f_a] -= 0.5*np.einsum('ibac,ibc->a',eris_ovvv, r_aaa_, optimize = True)
        temp = np.einsum('icab,a->ibc', eris_ovvv, r_a, optimize = True)
        temp -= np.einsum('ibac,a->ibc', eris_ovvv, r_a, optimize = True)
        s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)
        del eris_ovvv

        eris_OVvv = uadc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
        s[s_a:f_a] += np.einsum('icab,ibc->a', eris_OVvv, r_bab, optimize = True)
        s[s_bab:f_bab] += np.einsum('icab,a->ibc', eris_OVvv, r_a, optimize = True).reshape(-1)
        del eris_OVvv

        eris_OVVV = uadc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
        s[s_b:f_b] += 0.5*np.einsum('icab,ibc->a',eris_OVVV, r_bbb_, optimize = True)
        s[s_b:f_b] -= 0.5*np.einsum('ibac,ibc->a',eris_OVVV, r_bbb_, optimize = True)
        temp = np.einsum('icab,a->ibc', eris_OVVV, r_b, optimize = True)
        temp -= np.einsum('ibac,a->ibc', eris_OVVV, r_b, optimize = True)
        s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)
        del eris_OVVV

        eris_ovVV = uadc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
        s[s_b:f_b] += np.einsum('icab,ibc->a', eris_ovVV, r_aba, optimize = True)
        s[s_aba:f_aba] += np.einsum('icab,a->ibc', eris_ovVV, r_b, optimize = True).reshape(-1)
        del eris_ovVV

################ ADC(2) iab - jcd block ############################

        s[s_aaa:f_aaa] += D_iab_a * r_aaa
        s[s_bab:f_bab] += D_iab_bab * r_bab.reshape(-1)
        s[s_aba:f_aba] += D_iab_aba * r_aba.reshape(-1)
        s[s_bbb:f_bbb] += D_iab_b * r_bbb

############### ADC(3) iab - jcd block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

               t2_2_a, t2_2_ab, t2_2_b = adc.t2[1]

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

               eris_vvvv = uadc_ao2mo.unpack_eri_2s(eris.vvvv, nvir_a)
               eris_vvvv = eris_vvvv.transpose(0,2,1,3)
               eris_vvvv = eris_vvvv.copy()[:].reshape(nvir_a*nvir_a,nvir_a*nvir_a)
               r_aaa_t = r_aaa_u.reshape(nocc_a,-1)
               temp_1 = np.dot(r_aaa_t,eris_vvvv.T).reshape(nocc_a,nvir_a,nvir_a)
               eris_vvvv = eris_vvvv[:].reshape(nvir_a,nvir_a,nvir_a,nvir_a)
               eris_vvvv = eris_vvvv.transpose(0,1,3,2)
               eris_vvvv = eris_vvvv.copy()[:].reshape(nvir_a*nvir_a,nvir_a*nvir_a)
               temp_1 -= np.dot(r_aaa_t,eris_vvvv.T).reshape(nocc_a,nvir_a,nvir_a)
               del eris_vvvv
               temp_1 = temp_1[:,ab_ind_a[0],ab_ind_a[1]]
               s[s_aaa:f_aaa] += 0.5*temp_1.reshape(-1)

               eris_VVVV = uadc_ao2mo.unpack_eri_2s(eris.VVVV, nvir_b)
               eris_VVVV = eris_VVVV.transpose(0,2,1,3)
               eris_VVVV = eris_VVVV.copy()[:].reshape(nvir_b*nvir_b,nvir_b*nvir_b)
               r_bbb_t = r_bbb_u.reshape(nocc_b,-1)
               temp_1 = np.dot(r_bbb_t,eris_VVVV.T).reshape(nocc_b,nvir_b,nvir_b)
               eris_VVVV = eris_VVVV[:].reshape(nvir_b,nvir_b,nvir_b,nvir_b)
               eris_VVVV = eris_VVVV.transpose(0,1,3,2)
               eris_VVVV = eris_VVVV.copy()[:].reshape(nvir_b*nvir_b,nvir_b*nvir_b)
               temp_1 -= np.dot(r_bbb_t,eris_VVVV.T).reshape(nocc_b,nvir_b,nvir_b)
               del eris_VVVV
               temp_1 = temp_1[:,ab_ind_b[0],ab_ind_b[1]]
               s[s_bbb:f_bbb] += 0.5*temp_1.reshape(-1)

               r_bab_t = r_bab.reshape(nocc_b,-1)
               r_aba_t = r_aba.transpose(0,2,1).reshape(nocc_a,-1)
               eris_vvVV = uadc_ao2mo.unpack_eri_2(eris.vvVV, nvir_a, nvir_b)
               eris_vvVV = eris_vvVV.transpose(0,2,1,3)
               eris_vvVV = eris_vvVV.copy()[:].reshape(nvir_a*nvir_b,nvir_a*nvir_b)
               s[s_bab:f_bab] += np.dot(r_bab_t,eris_vvVV.T).reshape(-1)
               temp_1 = np.dot(r_aba_t,eris_vvVV.T).reshape(nocc_a, nvir_a,nvir_b)
               del eris_vvVV
               s[s_aba:f_aba] += temp_1.transpose(0,2,1).copy().reshape(-1)

               temp = 0.5*np.einsum('jiyz,jzx->ixy',eris_oovv,r_aaa_u,optimize = True)
               temp -= 0.5*np.einsum('jzyi,jzx->ixy',eris_ovvo,r_aaa_u,optimize = True)
               temp +=0.5*np.einsum('jzyi,jxz->ixy',eris_OVvo,r_bab,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] -= 0.5*np.einsum('jzyi,jzx->ixy',eris_ovVO,r_aaa_u,optimize = True).reshape(-1)
               s[s_bab:f_bab] -= 0.5*np.einsum('jiyz,jxz->ixy',eris_OOVV,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] += 0.5*np.einsum('jzyi,jxz->ixy',eris_OVVO,r_bab,optimize = True).reshape(-1)

               temp = 0.5*np.einsum('jiyz,jzx->ixy',eris_OOVV,r_bbb_u,optimize = True)
               temp -= 0.5*np.einsum('jzyi,jzx->ixy',eris_OVVO,r_bbb_u,optimize = True)
               temp +=0.5* np.einsum('jzyi,jxz->ixy',eris_ovVO,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

               s[s_aba:f_aba] -= 0.5*np.einsum('jiyz,jxz->ixy',eris_oovv,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] += 0.5*np.einsum('jzyi,jxz->ixy',eris_ovvo,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= 0.5*np.einsum('jzyi,jzx->ixy',eris_OVvo,r_bbb_u,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('jixz,jzy->ixy',eris_oovv,r_aaa_u,optimize = True)
               temp += 0.5*np.einsum('jzxi,jzy->ixy',eris_ovvo,r_aaa_u,optimize = True)
               temp -= 0.5*np.einsum('jzxi,jyz->ixy',eris_OVvo,r_bab,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] -=  0.5*np.einsum('jixz,jzy->ixy',eris_OOvv,r_bab,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('jixz,jzy->ixy',eris_OOVV,r_bbb_u,optimize = True)
               temp += 0.5*np.einsum('jzxi,jzy->ixy',eris_OVVO,r_bbb_u,optimize = True)
               temp -= 0.5*np.einsum('jzxi,jyz->ixy',eris_ovVO,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

               s[s_aba:f_aba] -= 0.5*np.einsum('jixz,jzy->ixy',eris_ooVV,r_aba,optimize = True).reshape(-1)

               temp = 0.5*np.einsum('jixw,jyw->ixy',eris_oovv,r_aaa_u,optimize = True)
               temp -= 0.5*np.einsum('jwxi,jyw->ixy',eris_ovvo,r_aaa_u,optimize = True)
               temp -= 0.5*np.einsum('jwxi,jyw->ixy',eris_OVvo,r_bab,optimize = True)

               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] -= 0.5*np.einsum('jixw,jwy->ixy',eris_OOvv,r_bab,optimize = True).reshape(-1)

               temp = 0.5*np.einsum('jixw,jyw->ixy',eris_OOVV,r_bbb_u,optimize = True)
               temp -= 0.5*np.einsum('jwxi,jyw->ixy',eris_OVVO,r_bbb_u,optimize = True)
               temp -= 0.5*np.einsum('jwxi,jyw->ixy',eris_ovVO,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

               s[s_aba:f_aba] -= 0.5*np.einsum('jixw,jwy->ixy',eris_ooVV,r_aba,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('jiyw,jxw->ixy',eris_oovv,r_aaa_u,optimize = True)
               temp += 0.5*np.einsum('jwyi,jxw->ixy',eris_ovvo,r_aaa_u,optimize = True)
               temp += 0.5*np.einsum('jwyi,jxw->ixy',eris_OVvo,r_bab,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] -= 0.5*np.einsum('jiyw,jxw->ixy',eris_OOVV,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] += 0.5*np.einsum('jwyi,jxw->ixy',eris_OVVO,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] += 0.5*np.einsum('jwyi,jxw->ixy',eris_ovVO,r_aaa_u,optimize = True).reshape(-1)

               s[s_aba:f_aba] -= 0.5*np.einsum('jiyw,jxw->ixy',eris_oovv,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] += 0.5*np.einsum('jwyi,jxw->ixy',eris_ovvo,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] += 0.5*np.einsum('jwyi,jxw->ixy',eris_OVvo,r_bbb_u,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('jiyw,jxw->ixy',eris_OOVV,r_bbb_u,optimize = True)
               temp += 0.5*np.einsum('jwyi,jxw->ixy',eris_OVVO,r_bbb_u,optimize = True)
               temp += 0.5*np.einsum('jwyi,jxw->ixy',eris_ovVO,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

        if (method == "adc(3)"):

            #print("Calculating additional terms for adc(3)")

               eris_ovoo = eris.ovoo
               eris_OVOO = eris.OVOO
               eris_ovOO = eris.ovOO
               eris_OVoo = eris.OVoo

############### ADC(3) a - ibc block and ibc-a coupling blocks ########################

               #temp = -0.5*np.einsum('lmwz,lmaj->ajzw',t2_1_a,v2e_oovo_a)
               #temp = temp[:,:,ab_ind_a[0],ab_ind_a[1]]
               #r_aaa = r_aaa.reshape(nocc_a,-1)
               #s[s_a:f_a] += np.einsum('ajp,jp->a',temp, r_aaa, optimize=True)

               t2_1_a_t = t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]]
               r_aaa = r_aaa.reshape(nocc_a,-1)
               temp = 0.5*np.einsum('lmp,jp->lmj',t2_1_a_t,r_aaa)
               s[s_a:f_a] += np.einsum('lmj,lamj->a',temp, eris_ovoo, optimize=True)
               s[s_a:f_a] -= np.einsum('lmj,malj->a',temp, eris_ovoo, optimize=True)

               temp_1 = -np.einsum('lmzw,jzw->jlm',t2_1_ab,r_bab)
               s[s_a:f_a] -= np.einsum('jlm,lamj->a',temp_1, eris_ovOO, optimize=True)

               #temp = -0.5*np.einsum('lmwz,lmaj->ajzw',t2_1_b,v2e_oovo_b)
               #temp = temp[:,:,ab_ind_b[0],ab_ind_b[1]]
               #r_bbb = r_bbb.reshape(nocc_b,-1)
               #s[s_b:f_b] += np.einsum('ajp,jp->a',temp, r_bbb, optimize=True)

               t2_1_b_t = t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]]
               r_bbb = r_bbb.reshape(nocc_b,-1)
               temp = 0.5*np.einsum('lmp,jp->lmj',t2_1_b_t,r_bbb)
               s[s_b:f_b] += np.einsum('lmj,lamj->a',temp, eris_OVOO, optimize=True)
               s[s_b:f_b] -= np.einsum('lmj,malj->a',temp, eris_OVOO, optimize=True)

               temp_1 = -np.einsum('mlwz,jzw->jlm',t2_1_ab,r_aba)
               s[s_b:f_b] -= np.einsum('jlm,lamj->a',temp_1, eris_OVoo, optimize=True)

               r_aaa_u = np.zeros((nocc_a,nvir_a,nvir_a))
               r_aaa_u[:,ab_ind_a[0],ab_ind_a[1]]= r_aaa.copy()
               r_aaa_u[:,ab_ind_a[1],ab_ind_a[0]]= -r_aaa.copy()

               r_bbb_u = np.zeros((nocc_b,nvir_b,nvir_b))
               r_bbb_u[:,ab_ind_b[0],ab_ind_b[1]]= r_bbb.copy()
               r_bbb_u[:,ab_ind_b[1],ab_ind_b[0]]= -r_bbb.copy()

               r_bab = r_bab.reshape(nocc_b,nvir_a,nvir_b)
               r_aba = r_aba.reshape(nocc_a,nvir_b,nvir_a)

               temp_s_a = np.zeros_like(r_bab)
               temp_s_a = np.einsum('jlwd,jzw->lzd',t2_1_a,r_aaa_u,optimize=True)
               temp_s_a += np.einsum('ljdw,jzw->lzd',t2_1_ab,r_bab,optimize=True)

               temp_s_a_1 = np.zeros_like(r_bab)
               temp_s_a_1 = -np.einsum('jlzd,jwz->lwd',t2_1_a,r_aaa_u,optimize=True)
               temp_s_a_1 += -np.einsum('ljdz,jwz->lwd',t2_1_ab,r_bab,optimize=True)

               eris_ovvv = uadc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
               s[s_a:f_a] += 0.5*np.einsum('lzd,ldza->a',temp_s_a,eris_ovvv,optimize=True)
               s[s_a:f_a] -= 0.5*np.einsum('lzd,lazd->a',temp_s_a,eris_ovvv,optimize=True)
               s[s_a:f_a] -= 0.5*np.einsum('lwd,ldwa->a',temp_s_a_1,eris_ovvv,optimize=True)
               s[s_a:f_a] += 0.5*np.einsum('lwd,lawd->a',temp_s_a_1,eris_ovvv,optimize=True)

               temp_1_1 = np.einsum('ldxb,b->lxd', eris_ovvv,r_a,optimize=True)
               temp_1_1 -= np.einsum('lbxd,b->lxd', eris_ovvv,r_a,optimize=True)

               temp_1_2 = np.einsum('ldyb,b->lyd', eris_ovvv,r_a,optimize=True)
               temp_1_2 -= np.einsum('lbyd,b->lyd', eris_ovvv,r_a,optimize=True)
               del eris_ovvv

               temp_s_b = np.zeros_like(r_aba)
               temp_s_b = np.einsum('jlwd,jzw->lzd',t2_1_b,r_bbb_u,optimize=True)
               temp_s_b += np.einsum('jlwd,jzw->lzd',t2_1_ab,r_aba,optimize=True)

               temp_s_b_1 = np.zeros_like(r_aba)
               temp_s_b_1 = -np.einsum('jlzd,jwz->lwd',t2_1_b,r_bbb_u,optimize=True)
               temp_s_b_1 += -np.einsum('jlzd,jwz->lwd',t2_1_ab,r_aba,optimize=True)

               eris_OVVV = uadc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
               s[s_b:f_b] += 0.5*np.einsum('lzd,ldza->a',temp_s_b,eris_OVVV,optimize=True)
               s[s_b:f_b] -= 0.5*np.einsum('lzd,lazd->a',temp_s_b,eris_OVVV,optimize=True)
               s[s_b:f_b] -= 0.5*np.einsum('lwd,ldwa->a',temp_s_b_1,eris_OVVV,optimize=True)
               s[s_b:f_b] += 0.5*np.einsum('lwd,lawd->a',temp_s_b_1,eris_OVVV,optimize=True)

               temp_1_3 = np.einsum('ldxb,b->lxd', eris_OVVV,r_b,optimize=True)
               temp_1_3 -= np.einsum('lbxd,b->lxd', eris_OVVV,r_b,optimize=True)

               temp_1_4 = np.einsum('ldyb,b->lyd', eris_OVVV,r_b,optimize=True)
               temp_1_4 -= np.einsum('lbyd,b->lyd', eris_OVVV,r_b,optimize=True)
               del eris_OVVV

               eris_OVvv = uadc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
               temp_1 = np.zeros_like(r_bab)
               temp_1 = np.einsum('jlwd,jzw->lzd',t2_1_ab,r_aaa_u,optimize=True)
               temp_1 += np.einsum('jlwd,jzw->lzd',t2_1_b,r_bab,optimize=True)
               s[s_a:f_a] += 0.5*np.einsum('lzd,ldza->a',temp_1,eris_OVvv,optimize=True)

               temp_2 = np.einsum('jldw,jwz->lzd',t2_1_ab,r_aba,optimize=True)
               s[s_b:f_b] -= 0.5*np.einsum('lzd,lazd->a',temp_2,eris_OVvv,optimize=True)

               temp_1 = np.zeros_like(r_bab)
               temp_1 = -np.einsum('jlzd,jwz->lwd',t2_1_ab,r_aaa_u,optimize=True)
               temp_1 += -np.einsum('jlzd,jwz->lwd',t2_1_b,r_bab,optimize=True)
               s[s_a:f_a] -= 0.5*np.einsum('lwd,ldwa->a',temp_1,eris_OVvv,optimize=True)

               temp_2 = -np.einsum('jldz,jzw->lwd',t2_1_ab,r_aba,optimize=True)
               s[s_b:f_b] += 0.5*np.einsum('lwd,lawd->a',temp_2,eris_OVvv,optimize=True)

               temp_2_1 = np.einsum('ldxb,b->lxd', eris_OVvv,r_a,optimize=True)
               temp_2_2 = np.einsum('ldyb,b->lyd', eris_OVvv,r_a,optimize=True)

               temp  = -np.einsum('lbyd,b->lyd',eris_OVvv,r_b,optimize=True)
               temp_1= -np.einsum('lyd,ildx->ixy',temp,t2_1_ab,optimize=True)
               s[s_aba:f_aba] -= temp_1.reshape(-1)
               del eris_OVvv

               eris_ovVV = uadc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
               temp_a = t2_1_ab.transpose(0,3,1,2).copy()
               temp_b = temp_a.reshape(nocc_a*nvir_b,nocc_b*nvir_a)
               r_bab_t = r_bab.reshape(nocc_b*nvir_a,-1)
               temp_c = np.dot(temp_b,r_bab_t).reshape(nocc_a,nvir_b,nvir_b)
               temp_2 = temp_c.transpose(0,2,1).copy()
               s[s_a:f_a] -= 0.5*np.einsum('lzd,lazd->a',temp_2,eris_ovVV,optimize=True)

               temp_2 = -np.einsum('ljzd,jzw->lwd',t2_1_ab,r_bab,optimize=True)
               s[s_a:f_a] += 0.5*np.einsum('lwd,lawd->a',temp_2,eris_ovVV,optimize=True)

               temp_1 = np.zeros_like(r_aba)
               temp_1 = np.einsum('ljdw,jzw->lzd',t2_1_ab,r_bbb_u,optimize=True)
               temp_1 += np.einsum('jlwd,jzw->lzd',t2_1_a,r_aba,optimize=True)
               temp_a = temp_1.reshape(-1)
               eris_ovVV = eris_ovVV.transpose(0,2,1,3)
               eris_ovVV = eris_ovVV.copy()[:].reshape(nocc_a*nvir_b*nvir_a,-1)
               s[s_b:f_b] += 0.5*np.dot(temp_a,eris_ovVV)

               temp_1 = np.zeros_like(r_bab)
               temp_1 = -np.einsum('ljdz,jwz->lwd',t2_1_ab,r_bbb_u,optimize=True)
               temp_1 += -np.einsum('jlzd,jwz->lwd',t2_1_a,r_aba,optimize=True)
               temp_a = temp_1.reshape(-1)
               s[s_b:f_b] -= 0.5*np.dot(temp_a,eris_ovVV)

               eris_ovVV = eris_ovVV[:].reshape(nocc_a, nvir_b, nvir_a, nvir_b)
               eris_ovVV = eris_ovVV.transpose(0,2,1,3).copy()

               temp_2_3 = np.einsum('ldxb,b->lxd', eris_ovVV,r_b,optimize=True)
               temp_2_4 = np.einsum('ldyb,b->lyd', eris_ovVV,r_b,optimize=True)

               temp  = -np.einsum('lbyd,b->lyd',eris_ovVV,r_a,optimize=True)
               temp_1= -np.einsum('lyd,lixd->ixy',temp,t2_1_ab,optimize=True)
               s[s_bab:f_bab] -= temp_1.reshape(-1)
               del eris_ovVV

######################################################################################

               t2_1_a_t = t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]]
               temp = np.einsum('b,lbmi->lmi',r_a,eris_ovoo)
               temp -= np.einsum('b,mbli->lmi',r_a,eris_ovoo)
               s[s_aaa:f_aaa] += 0.5*np.einsum('lmi,lmp->ip',temp, t2_1_a_t, optimize=True).reshape(-1)

               temp_1 = np.einsum('b,lbmi->lmi',r_a,eris_ovOO)
               s[s_bab:f_bab] += np.einsum('lmi,lmxy->ixy',temp_1, t2_1_ab, optimize=True).reshape(-1)

               t2_1_b_t = t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]]
               temp = np.einsum('b,lbmi->lmi',r_b,eris_OVOO)
               temp -= np.einsum('b,mbli->lmi',r_b,eris_OVOO)
               s[s_bbb:f_bbb] += 0.5*np.einsum('lmi,lmp->ip',temp, t2_1_b_t, optimize=True).reshape(-1)

               temp_1 = np.einsum('b,lbmi->mli',r_b,eris_OVoo)
               s[s_aba:f_aba] += np.einsum('mli,mlyx->ixy',temp_1, t2_1_ab, optimize=True).reshape(-1)

               temp  = np.einsum('lxd,ilyd->ixy',temp_1_1,t2_1_a,optimize=True)
               temp += np.einsum('lxd,ilyd->ixy',temp_2_1,t2_1_ab,optimize=True)
               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1] ].reshape(-1)

               temp  = np.einsum('lyd,ilxd->ixy',temp_1_2,t2_1_a,optimize=True)
               temp += np.einsum('lyd,ilxd->ixy',temp_2_2,t2_1_ab,optimize=True)
               s[s_aaa:f_aaa] -= temp[:,ab_ind_a[0],ab_ind_a[1] ].reshape(-1)

               temp  = np.einsum('lxd,lidy->ixy',temp_1_1,t2_1_ab,optimize=True)
               temp  += np.einsum('lxd,ilyd->ixy',temp_2_1,t2_1_b,optimize=True)
               s[s_bab:f_bab] += temp.reshape(-1)

               temp  = np.einsum('lxd,ilyd->ixy',temp_1_3,t2_1_b,optimize=True)
               temp += np.einsum('lxd,lidy->ixy',temp_2_3,t2_1_ab,optimize=True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1] ].reshape(-1)

               temp  = np.einsum('lyd,ilxd->ixy',temp_1_4,t2_1_b,optimize=True)
               temp += np.einsum('lyd,lidx->ixy',temp_2_4,t2_1_ab,optimize=True)
               s[s_bbb:f_bbb] -= temp[:,ab_ind_b[0],ab_ind_b[1] ].reshape(-1)

               temp  = np.einsum('lxd,ilyd->ixy',temp_1_3,t2_1_ab,optimize=True)
               temp  += np.einsum('lxd,ilyd->ixy',temp_2_3,t2_1_a,optimize=True)
               s[s_aba:f_aba] += temp.reshape(-1)

        return s

    return sigma_


def ip_adc_matvec(adc, M_ij=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2_1_a, t2_1_ab, t2_1_b = adc.t2[0]
    t1_2_a, t1_2_b = adc.t1[0]

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

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    if eris is None:
        eris = uadc_ao2mo.transform_integrals_incore(adc)

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

        s = np.zeros((dim))

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

        #r_bab = r_bab.reshape(nvir_b,nocc_a,nocc_b)

        r_aba = r_aba.reshape(nvir_a,nocc_a,nocc_b)
        r_bab = r_bab.reshape(nvir_b,nocc_b,nocc_a)

        eris_ovoo = eris.ovoo
        eris_OVOO = eris.OVOO
        eris_OVoo = eris.OVoo
        eris_ovOO = eris.ovOO

############ ADC(2) ij block ############################

        s[s_a:f_a] = np.einsum('ij,j->i',M_ij_a,r_a)
        s[s_b:f_b] = np.einsum('ij,j->i',M_ij_b,r_b)

############ ADC(2) i - kja block #########################

        s[s_a:f_a] += 0.5*np.einsum('jaki,ajk->i', eris_ovoo, r_aaa_u, optimize = True)
        s[s_a:f_a] -= 0.5*np.einsum('kaji,ajk->i', eris_ovoo, r_aaa_u, optimize = True)
        s[s_a:f_a] += np.einsum('jaki,ajk->i', eris_OVoo, r_bab, optimize = True)

        s[s_b:f_b] += 0.5*np.einsum('jaki,ajk->i', eris_OVOO, r_bbb_u, optimize = True)
        s[s_b:f_b] -= 0.5*np.einsum('kaji,ajk->i', eris_OVOO, r_bbb_u, optimize = True)
        s[s_b:f_b] += np.einsum('jaki,ajk->i', eris_ovOO, r_aba, optimize = True)

################ ADC(2) ajk - i block ############################

        temp = np.einsum('jaki,i->ajk', eris_ovoo, r_a, optimize = True)
        temp -= np.einsum('kaji,i->ajk', eris_ovoo, r_a, optimize = True)
        s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
        s[s_bab:f_bab] += np.einsum('jaik,i->ajk', eris_OVoo, r_a, optimize = True).reshape(-1)
        s[s_aba:f_aba] += np.einsum('jaki,i->ajk', eris_ovOO, r_b, optimize = True).reshape(-1)
        temp = np.einsum('jaki,i->ajk', eris_OVOO, r_b, optimize = True)
        temp -= np.einsum('kaji,i->ajk', eris_OVOO, r_b, optimize = True)
        s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

################ ADC(2) ajk - bil block ############################

        r_aaa = r_aaa.reshape(-1)
        r_bbb = r_bbb.reshape(-1)

        s[s_aaa:f_aaa] += D_aij_a * r_aaa
        s[s_bab:f_bab] += D_aij_bab * r_bab.reshape(-1)
        s[s_aba:f_aba] += D_aij_aba * r_aba.reshape(-1)
        s[s_bbb:f_bbb] += D_aij_b * r_bbb

############### ADC(3) ajk - bil block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

               t2_2_a, t2_2_ab, t2_2_b = adc.t2[1]

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

               temp = 0.5*np.einsum('jlki,ail->ajk',eris_oooo,r_aaa_u ,optimize = True)
               temp -= 0.5*np.einsum('jikl,ail->ajk',eris_oooo,r_aaa_u ,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

               temp = 0.5*np.einsum('jlki,ail->ajk',eris_OOOO,r_bbb_u,optimize = True)
               temp -= 0.5*np.einsum('jikl,ail->ajk',eris_OOOO,r_bbb_u,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

               s[s_bab:f_bab] -= 0.5*np.einsum('kijl,ali->ajk',eris_ooOO,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] -= 0.5*np.einsum('klji,ail->ajk',eris_ooOO,r_bab,optimize = True).reshape(-1)

               s[s_aba:f_aba] -= 0.5*np.einsum('jlki,ali->ajk',eris_ooOO,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= 0.5*np.einsum('jikl,ail->ajk',eris_ooOO,r_aba,optimize = True).reshape(-1)

               temp = 0.5*np.einsum('klba,bjl->ajk',eris_oovv,r_aaa_u,optimize = True)
               temp -= 0.5*np.einsum('kabl,bjl->ajk',eris_ovvo,r_aaa_u,optimize = True)
               temp += 0.5* np.einsum('kabl,blj->ajk',eris_ovVO,r_bab,optimize = True)

               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] += 0.5*np.einsum('klba,bjl->ajk',eris_ooVV,r_bab,optimize = True).reshape(-1)

               temp_1 = 0.5*np.einsum('klba,bjl->ajk',eris_OOVV,r_bbb_u,optimize = True)
               temp_1 -= 0.5*np.einsum('kabl,bjl->ajk',eris_OVVO,r_bbb_u,optimize = True)
               temp_1 += 0.5*np.einsum('kabl,blj->ajk',eris_OVvo,r_aba,optimize = True)

               s[s_bbb:f_bbb] += temp_1[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

               s[s_aba:f_aba] += 0.5*np.einsum('klba,bjl->ajk',eris_OOvv,r_aba,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('jlba,bkl->ajk',eris_oovv,r_aaa_u,optimize = True)
               temp += 0.5*np.einsum('jabl,bkl->ajk',eris_ovvo,r_aaa_u,optimize = True)
               temp -= 0.5*np.einsum('jabl,blk->ajk',eris_ovVO,r_bab,optimize = True)

               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] +=  0.5*np.einsum('jabl,bkl->ajk',eris_OVvo,r_aaa_u,optimize = True).reshape(-1)
               s[s_bab:f_bab] +=  0.5*np.einsum('jlba,blk->ajk',eris_OOVV,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] -=  0.5*np.einsum('jabl,blk->ajk',eris_OVVO,r_bab,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('jlba,bkl->ajk',eris_OOVV,r_bbb_u,optimize = True)
               temp += 0.5*np.einsum('jabl,bkl->ajk',eris_OVVO,r_bbb_u,optimize = True)
               temp -= 0.5*np.einsum('jabl,blk->ajk',eris_OVvo,r_aba,optimize = True)

               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

               s[s_aba:f_aba] += 0.5*np.einsum('jlba,blk->ajk',eris_oovv,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= 0.5*np.einsum('jabl,blk->ajk',eris_ovvo,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] += 0.5*np.einsum('jabl,bkl->ajk',eris_ovVO,r_bbb_u,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('kiba,bij->ajk',eris_oovv,r_aaa_u,optimize = True)
               temp += 0.5*np.einsum('kabi,bij->ajk',eris_ovvo,r_aaa_u,optimize = True)
               temp += 0.5*np.einsum('kabi,bij->ajk',eris_ovVO,r_bab,optimize = True)

               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] += 0.5*np.einsum('kiba,bji->ajk',eris_ooVV,r_bab,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('kiba,bij->ajk',eris_OOVV,r_bbb_u,optimize = True)
               temp += 0.5*np.einsum('kabi,bij->ajk',eris_OVVO,r_bbb_u,optimize = True)
               temp += 0.5*np.einsum('kabi,bij->ajk',eris_OVvo,r_aba,optimize = True)

               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

               s[s_aba:f_aba] += 0.5*np.einsum('kiba,bji->ajk',eris_OOvv,r_aba,optimize = True).reshape(-1)

               temp = 0.5*np.einsum('jiba,bik->ajk',eris_oovv,r_aaa_u,optimize = True)
               temp -= 0.5*np.einsum('jabi,bik->ajk',eris_ovvo,r_aaa_u,optimize = True)
               temp -= 0.5*np.einsum('jabi,bik->ajk',eris_ovVO,r_bab,optimize = True)

               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] += 0.5*np.einsum('jiba,bik->ajk',eris_OOVV,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] -= 0.5*np.einsum('jabi,bik->ajk',eris_OVVO,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] -= 0.5*np.einsum('jabi,bik->ajk',eris_OVvo,r_aaa_u,optimize = True).reshape(-1)

               s[s_aba:f_aba] += 0.5*np.einsum('jiba,bik->ajk',eris_oovv,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= 0.5*np.einsum('jabi,bik->ajk',eris_ovvo,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= 0.5*np.einsum('jabi,bik->ajk',eris_ovVO,r_bbb_u,optimize = True).reshape(-1)

               temp = 0.5*np.einsum('jiba,bik->ajk',eris_OOVV,r_bbb_u,optimize = True)
               temp -= 0.5*np.einsum('jabi,bik->ajk',eris_OVVO,r_bbb_u,optimize = True)
               temp -= 0.5*np.einsum('jabi,bik->ajk',eris_OVvo,r_aba,optimize = True)

               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

        if (method == "adc(3)"):

               eris_ovoo = eris.ovoo
               eris_OVOO = eris.OVOO
               eris_ovOO = eris.ovOO
               eris_OVoo = eris.OVoo

################ ADC(3) i - kja block ############################

               #t2_1_a_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:]
               #temp = np.einsum('pbc,bcai->pai',t2_1_a_t,v2e_vvvo_a)
               #r_aaa = r_aaa.reshape(nvir_a,-1)
               #s[s_a:f_a] += 0.5*np.einsum('pai,ap->i',temp, r_aaa, optimize=True)

               eris_ovvv = uadc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
               r_aaa = r_aaa.reshape(nvir_a,-1)
               t2_1_a_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:].copy()
               temp = np.einsum('pbc,ap->abc',t2_1_a_t,r_aaa, optimize=True)
               s[s_a:f_a] += 0.5*np.einsum('abc,icab->i',temp, eris_ovvv, optimize=True)
               s[s_a:f_a] -= 0.5*np.einsum('abc,ibac->i',temp, eris_ovvv, optimize=True)
               del eris_ovvv

               eris_ovVV = uadc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
               temp_1 = np.einsum('kjcb,ajk->abc',t2_1_ab,r_bab, optimize=True)
               s[s_a:f_a] += np.einsum('abc,icab->i',temp_1, eris_ovVV, optimize=True)
               del eris_ovVV

               #t2_1_b_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:]
               #temp = np.einsum('pbc,bcai->pai',t2_1_b_t,v2e_vvvo_b)
               #r_bbb = r_bbb.reshape(nvir_b,-1)
               #s[s_b:f_b] += 0.5*np.einsum('pai,ap->i',temp, r_bbb, optimize=True)

               eris_OVVV = uadc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
               r_bbb = r_bbb.reshape(nvir_b,-1)
               t2_1_b_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:].copy()
               temp = np.einsum('pbc,ap->abc',t2_1_b_t,r_bbb, optimize=True)
               s[s_b:f_b] += 0.5*np.einsum('abc,icab->i',temp, eris_OVVV, optimize=True)
               s[s_b:f_b] -= 0.5*np.einsum('abc,ibac->i',temp, eris_OVVV, optimize=True)
               del eris_OVVV

               eris_OVvv = uadc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
               temp_1 = np.einsum('jkbc,ajk->abc',t2_1_ab,r_aba, optimize=True)
               s[s_b:f_b] += np.einsum('abc,icab->i',temp_1, eris_OVvv, optimize=True)
               del eris_OVvv

               r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
               r_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= r_aaa.copy()
               r_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -r_aaa.copy()

               r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))
               r_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= r_bbb.copy()
               r_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -r_bbb.copy()

               r_bab = r_bab.reshape(nvir_b,nocc_b,nocc_a)
               r_aba = r_aba.reshape(nvir_a,nocc_a,nocc_b)

               temp = np.zeros_like(r_bab)
               temp = np.einsum('jlab,ajk->blk',t2_1_a,r_aaa_u,optimize=True)
               temp += np.einsum('ljba,ajk->blk',t2_1_ab,r_bab,optimize=True)

               temp_1 = np.zeros_like(r_bab)
               temp_1 = np.einsum('jlab,ajk->blk',t2_1_ab,r_aaa_u,optimize=True)
               temp_1 += np.einsum('jlab,ajk->blk',t2_1_b,r_bab,optimize=True)

               temp_2 = np.einsum('jlba,akj->blk',t2_1_ab,r_bab, optimize=True)

               s[s_a:f_a] += 0.5*np.einsum('blk,lbik->i',temp,eris_ovoo,optimize=True)
               s[s_a:f_a] -= 0.5*np.einsum('blk,iblk->i',temp,eris_ovoo,optimize=True)
               s[s_a:f_a] += 0.5*np.einsum('blk,lbik->i',temp_1,eris_OVoo,optimize=True)
               s[s_a:f_a] -= 0.5*np.einsum('blk,iblk->i',temp_2,eris_ovOO,optimize=True)

               temp = np.zeros_like(r_aba)
               temp = np.einsum('jlab,ajk->blk',t2_1_b,r_bbb_u,optimize=True)
               temp += np.einsum('jlab,ajk->blk',t2_1_ab,r_aba,optimize=True)

               temp_1 = np.zeros_like(r_aba)
               temp_1 = np.einsum('ljba,ajk->blk',t2_1_ab,r_bbb_u,optimize=True)
               temp_1 += np.einsum('jlab,ajk->blk',t2_1_a,r_aba,optimize=True)

               temp_2 = np.einsum('ljab,akj->blk',t2_1_ab,r_aba,optimize=True)

               s[s_b:f_b] += 0.5*np.einsum('blk,lbik->i',temp,eris_OVOO,optimize=True)
               s[s_b:f_b] -= 0.5*np.einsum('blk,iblk->i',temp,eris_OVOO,optimize=True)
               s[s_b:f_b] += 0.5*np.einsum('blk,lbik->i',temp_1,eris_ovOO,optimize=True)
               s[s_b:f_b] -= 0.5*np.einsum('blk,iblk->i',temp_2,eris_OVoo,optimize=True)

               temp = np.zeros_like(r_bab)
               temp = -np.einsum('klab,akj->blj',t2_1_a,r_aaa_u,optimize=True)
               temp -= np.einsum('lkba,akj->blj',t2_1_ab,r_bab,optimize=True)

               temp_1 = np.zeros_like(r_bab)
               temp_1 = -np.einsum('klab,akj->blj',t2_1_ab,r_aaa_u,optimize=True)
               temp_1 -= np.einsum('klab,akj->blj',t2_1_b,r_bab,optimize=True)

               temp_2 = -np.einsum('klba,ajk->blj',t2_1_ab,r_bab,optimize=True)

               s[s_a:f_a] -= 0.5*np.einsum('blj,lbij->i',temp,eris_ovoo,optimize=True)
               s[s_a:f_a] += 0.5*np.einsum('blj,iblj->i',temp,eris_ovoo,optimize=True)
               s[s_a:f_a] -= 0.5*np.einsum('blj,lbij->i',temp_1,eris_OVoo,optimize=True)
               s[s_a:f_a] += 0.5*np.einsum('blj,iblj->i',temp_2,eris_ovOO,optimize=True)

               temp = np.zeros_like(r_aba)
               temp = -np.einsum('klab,akj->blj',t2_1_b,r_bbb_u,optimize=True)
               temp -= np.einsum('klab,akj->blj',t2_1_ab,r_aba,optimize=True)

               temp_1 = np.zeros_like(r_bab)
               temp_1 = -np.einsum('lkba,akj->blj',t2_1_ab,r_bbb_u,optimize=True)
               temp_1 -= np.einsum('klab,akj->blj',t2_1_a,r_aba,optimize=True)

               temp_2 = -np.einsum('lkab,ajk->blj',t2_1_ab,r_aba,optimize=True)

               s[s_b:f_b] -= 0.5*np.einsum('blj,lbij->i',temp,eris_OVOO,optimize=True)
               s[s_b:f_b] += 0.5*np.einsum('blj,iblj->i',temp,eris_OVOO,optimize=True)
               s[s_b:f_b] -= 0.5*np.einsum('blj,lbij->i',temp_1,eris_ovOO,optimize=True)
               s[s_b:f_b] += 0.5*np.einsum('blj,iblj->i',temp_2,eris_OVoo,optimize=True)

################ ADC(3) ajk - i block ############################
               #t2_1_a_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:]
               #temp = 0.5*np.einsum('pbc,bcai->api',t2_1_a_t,v2e_vvvo_a)
               #s[s_aaa:f_aaa] += np.einsum('api,i->ap',temp, r_a, optimize=True).reshape(-1)

               eris_ovvv = uadc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
               t2_1_a_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:].copy()
               temp = np.einsum('i,icab->bca',r_a,eris_ovvv,optimize=True)
               temp -= np.einsum('i,ibac->bca',r_a,eris_ovvv,optimize=True)
               s[s_aaa:f_aaa] += 0.5*np.einsum('bca,pbc->ap',temp,t2_1_a_t,optimize=True).reshape(-1)
               del eris_ovvv

               #temp_1 = np.einsum('kjcb,cbia->iajk',t2_1_ab,v2e_vvov_ab)
               #temp_1 = temp_1.reshape(nocc_a,-1)
               #s[s_bab:f_bab] += np.einsum('ip,i->p',temp_1, r_a, optimize=True).reshape(-1)

               eris_ovVV = uadc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
               temp_1 = np.einsum('i,icab->cba',r_a,eris_ovVV,optimize=True)
               s[s_bab:f_bab] += np.einsum('cba,kjcb->ajk',temp_1, t2_1_ab, optimize=True).reshape(-1)
               del eris_ovVV

               #t2_1_b_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:]
               #temp = 0.5*np.einsum('pbc,bcai->api',t2_1_b_t,v2e_vvvo_b)
               #s[s_bbb:f_bbb] += np.einsum('api,i->ap',temp, r_b, optimize=True).reshape(-1)

               eris_OVVV = uadc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
               t2_1_b_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:].copy()
               temp = np.einsum('i,icab->bca',r_b,eris_OVVV,optimize=True)
               temp -= np.einsum('i,ibac->bca',r_b,eris_OVVV,optimize=True)
               s[s_bbb:f_bbb] += 0.5*np.einsum('bca,pbc->ap',temp,t2_1_b_t,optimize=True).reshape(-1)
               del eris_OVVV

               #temp_1 = np.einsum('jkbc,bcai->iajk',t2_1_ab,v2e_vvvo_ab)
               #temp_1 = temp_1.reshape(nocc_b,-1)
               #s[s_aba:f_aba] += np.einsum('ip,i->p',temp_1, r_b, optimize=True).reshape(-1)

               eris_OVvv = uadc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
               temp_1 = np.einsum('i,icab->bca',r_b,eris_OVvv,optimize=True)
               s[s_aba:f_aba] += np.einsum('bca,jkbc->ajk',temp_1, t2_1_ab, optimize=True).reshape(-1)
               del eris_OVvv

               temp_1 = np.einsum('i,lbik->kbl',r_a, eris_ovoo)
               temp_1 -= np.einsum('i,iblk->kbl',r_a, eris_ovoo)
               temp_2 = np.einsum('i,lbik->kbl',r_a, eris_OVoo)

               temp  = np.einsum('kbl,jlab->ajk',temp_1,t2_1_a,optimize=True)
               temp += np.einsum('kbl,jlab->ajk',temp_2,t2_1_ab,optimize=True)
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1] ].reshape(-1)

               temp_1  = np.einsum('i,lbik->kbl',r_a,eris_ovoo)
               temp_1  -= np.einsum('i,iblk->kbl',r_a,eris_ovoo)
               temp_2  = np.einsum('i,lbik->kbl',r_a,eris_OVoo)

               temp  = np.einsum('kbl,ljba->ajk',temp_1,t2_1_ab,optimize=True)
               temp += np.einsum('kbl,jlab->ajk',temp_2,t2_1_b,optimize=True)
               s[s_bab:f_bab] += temp.reshape(-1)

               temp_1 = np.einsum('i,lbik->kbl',r_b, eris_OVOO)
               temp_1 -= np.einsum('i,iblk->kbl',r_b, eris_OVOO)
               temp_2 = np.einsum('i,lbik->kbl',r_b, eris_ovOO)

               temp  = np.einsum('kbl,jlab->ajk',temp_1,t2_1_b,optimize=True)
               temp += np.einsum('kbl,ljba->ajk',temp_2,t2_1_ab,optimize=True)
               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1] ].reshape(-1)

               temp_1  = np.einsum('i,lbik->kbl',r_b,eris_OVOO)
               temp_1  -= np.einsum('i,iblk->kbl',r_b,eris_OVOO)
               temp_2  = np.einsum('i,lbik->kbl',r_b,eris_ovOO)

               temp  = np.einsum('kbl,jlab->ajk',temp_1,t2_1_ab,optimize=True)
               temp += np.einsum('kbl,jlab->ajk',temp_2,t2_1_a,optimize=True)
               s[s_aba:f_aba] += temp.reshape(-1)

               temp_1 = np.einsum('i,lbij->jbl',r_a, eris_ovoo)
               temp_1 -= np.einsum('i,iblj->jbl',r_a, eris_ovoo)
               temp_2 = np.einsum('i,lbij->jbl',r_a, eris_OVoo)

               temp  = np.einsum('jbl,klab->ajk',temp_1,t2_1_a,optimize=True)
               temp += np.einsum('jbl,klab->ajk',temp_2,t2_1_ab,optimize=True)
               s[s_aaa:f_aaa] -= temp[:,ij_ind_a[0],ij_ind_a[1] ].reshape(-1)

               temp  = -np.einsum('i,iblj->jbl',r_a,eris_ovOO,optimize=True)
               temp_1 = -np.einsum('jbl,klba->ajk',temp,t2_1_ab,optimize=True)
               s[s_bab:f_bab] -= temp_1.reshape(-1)

               temp_1 = np.einsum('i,lbij->jbl',r_b, eris_OVOO)
               temp_1 -= np.einsum('i,iblj->jbl',r_b, eris_OVOO)
               temp_2 = np.einsum('i,lbij->jbl',r_b, eris_ovOO)

               temp  = np.einsum('jbl,klab->ajk',temp_1,t2_1_b,optimize=True)
               temp += np.einsum('jbl,lkba->ajk',temp_2,t2_1_ab,optimize=True)
               s[s_bbb:f_bbb] -= temp[:,ij_ind_b[0],ij_ind_b[1] ].reshape(-1)

               temp  = -np.einsum('i,iblj->jbl',r_b,eris_OVoo,optimize=True)
               temp_1 = -np.einsum('jbl,lkab->ajk',temp,t2_1_ab,optimize=True)
               s[s_aba:f_aba] -= temp_1.reshape(-1)

        s *= -1.0

        return s

    return sigma_


def ea_compute_trans_moments(adc, orb, spin="alpha"):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2_1_a, t2_1_ab, t2_1_b = adc.t2[0]
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

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
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
    if spin=="alpha":
######## ADC(2) part  ############################################

        if orb < nocc_a:

            T[s_a:f_a] = -t1_2_a[orb,:]

            t2_1_t = t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]].copy()
            t2_1_ab_t = -t2_1_ab.transpose(1,0,2,3).copy()

            T[s_aaa:f_aaa] += t2_1_t[:,orb,:].reshape(-1)
            T[s_bab:f_bab] += t2_1_ab_t[:,orb,:,:].reshape(-1)

        else :

            T[s_a:f_a] += idn_vir_a[(orb-nocc_a), :]
            T[s_a:f_a] -= 0.25*np.einsum('klc,klac->a',t2_1_a[:,:,(orb-nocc_a),:], t2_1_a, optimize = True)
            T[s_a:f_a] -= 0.25*np.einsum('klc,klac->a',t2_1_ab[:,:,(orb-nocc_a),:], t2_1_ab, optimize = True)
            T[s_a:f_a] -= 0.25*np.einsum('lkc,lkac->a',t2_1_ab[:,:,(orb-nocc_a),:], t2_1_ab, optimize = True)

######## ADC(3) 2p-1h  part  ############################################

        if(method=='adc(2)-x'or method=='adc(3)'):

            t2_2_a, t2_2_ab, t2_2_b = adc.t2[1]

            if orb < nocc_a:

                t2_2_t = t2_2_a[:,:,ab_ind_a[0],ab_ind_a[1]].copy()
                t2_2_ab_t = -t2_2_ab.transpose(1,0,2,3).copy()

                T[s_aaa:f_aaa] += t2_2_t[:,orb,:].reshape(-1)
                T[s_bab:f_bab] += t2_2_ab_t[:,orb,:,:].reshape(-1)

######### ADC(3) 1p part  ############################################

        if(method=='adc(3)'):

            t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_a:

                T[s_a:f_a] += 0.5*np.einsum('kac,ck->a',t2_1_a[:,orb,:,:], t1_2_a.T,optimize = True)
                T[s_a:f_a] -= 0.5*np.einsum('kac,ck->a',t2_1_ab[orb,:,:,:], t1_2_b.T,optimize = True)

                T[s_a:f_a] -= t1_3_a[orb,:]

            else:

                T[s_a:f_a] -= 0.25*np.einsum('klc,klac->a',t2_1_a[:,:,(orb-nocc_a),:], t2_2_a, optimize = True)
                T[s_a:f_a] -= 0.25*np.einsum('klc,klac->a',t2_1_ab[:,:,(orb-nocc_a),:], t2_2_ab, optimize = True)
                T[s_a:f_a] -= 0.25*np.einsum('lkc,lkac->a',t2_1_ab[:,:,(orb-nocc_a),:], t2_2_ab, optimize = True)

                T[s_a:f_a] -= 0.25*np.einsum('klac,klc->a',t2_1_a, t2_2_a[:,:,(orb-nocc_a),:],optimize = True)
                T[s_a:f_a] -= 0.25*np.einsum('klac,klc->a',t2_1_ab, t2_2_ab[:,:,(orb-nocc_a),:],optimize = True)
                T[s_a:f_a] -= 0.25*np.einsum('lkac,lkc->a',t2_1_ab, t2_2_ab[:,:,(orb-nocc_a),:],optimize = True)

######### spin = beta  ############################################
    else:
######## ADC(2) part  ############################################


        if orb < nocc_b:

            T[s_b:f_b] = -t1_2_b[orb,:]

            t2_1_t = t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]].copy()
            t2_1_ab_t = -t2_1_ab.transpose(0,1,3,2).copy()

            T[s_bbb:f_bbb] += t2_1_t[:,orb,:].reshape(-1)
            T[s_aba:f_aba] += t2_1_ab_t[:,orb,:,:].reshape(-1)

        else :

            T[s_b:f_b] += idn_vir_b[(orb-nocc_b), :]
            T[s_b:f_b] -= 0.25*np.einsum('klc,klac->a',t2_1_b[:,:,(orb-nocc_b),:], t2_1_b, optimize = True)
            T[s_b:f_b] -= 0.25*np.einsum('lkc,lkca->a',t2_1_ab[:,:,:,(orb-nocc_b)], t2_1_ab, optimize = True)
            T[s_b:f_b] -= 0.25*np.einsum('lkc,lkca->a',t2_1_ab[:,:,:,(orb-nocc_b)], t2_1_ab, optimize = True)

######### ADC(3) 2p-1h part  ############################################

        if(method=='adc(2)-x'or method=='adc(3)'):

            t2_2_a, t2_2_ab, t2_2_b = adc.t2[1]

            if orb < nocc_b:

                t2_2_t = t2_2_b[:,:,ab_ind_b[0],ab_ind_b[1]].copy()
                t2_2_ab_t = -t2_2_ab.transpose(0,1,3,2).copy()

                T[s_bbb:f_bbb] += t2_2_t[:,orb,:].reshape(-1)
                T[s_aba:f_aba] += t2_2_ab_t[:,orb,:,:].reshape(-1)

######### ADC(2) 1p part  ############################################

        if(method=='adc(3)'):

            t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_b:

                T[s_b:f_b] += 0.5*np.einsum('kac,ck->a',t2_1_b[:,orb,:,:], t1_2_b.T,optimize = True)
                T[s_b:f_b] -= 0.5*np.einsum('kca,ck->a',t2_1_ab[:,orb,:,:], t1_2_a.T,optimize = True)

                T[s_b:f_b] -= t1_3_b[orb,:]

            else:

                T[s_b:f_b] -= 0.25*np.einsum('klc,klac->a',t2_1_b[:,:,(orb-nocc_b),:], t2_2_b, optimize = True)
                T[s_b:f_b] -= 0.25*np.einsum('lkc,lkca->a',t2_1_ab[:,:,:,(orb-nocc_b)], t2_2_ab, optimize = True)
                T[s_b:f_b] -= 0.25*np.einsum('lkc,lkca->a',t2_1_ab[:,:,:,(orb-nocc_b)], t2_2_ab, optimize = True)

                T[s_b:f_b] -= 0.25*np.einsum('klac,klc->a',t2_1_b, t2_2_b[:,:,(orb-nocc_b),:],optimize = True)
                T[s_b:f_b] -= 0.25*np.einsum('lkca,lkc->a',t2_1_ab, t2_2_ab[:,:,:,(orb-nocc_b)],optimize = True)
                T[s_b:f_b] -= 0.25*np.einsum('klca,klc->a',t2_1_ab, t2_2_ab[:,:,:,(orb-nocc_b)],optimize = True)
    return T


def ip_compute_trans_moments(adc, orb, spin="alpha"):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2_1_a, t2_1_ab, t2_1_b = adc.t2[0]
    t1_2_a, t1_2_b = adc.t1[0]

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
    if spin=="alpha":
######## ADC(2) 1h part  ############################################

        if orb < nocc_a:
            T[s_a:f_a]  = idn_occ_a[orb, :]
            T[s_a:f_a] += 0.25*np.einsum('kdc,ikdc->i',t2_1_a[:,orb,:,:], t2_1_a, optimize = True)
            T[s_a:f_a] -= 0.25*np.einsum('kdc,ikdc->i',t2_1_ab[orb,:,:,:], t2_1_ab, optimize = True)
            T[s_a:f_a] -= 0.25*np.einsum('kcd,ikcd->i',t2_1_ab[orb,:,:,:], t2_1_ab, optimize = True)
        else :
            T[s_a:f_a] += t1_2_a[:,(orb-nocc_a)]

######## ADC(2) 2h-1p  part  ############################################

            t2_1_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:].copy()
            t2_1_t_a = t2_1_t.transpose(2,1,0).copy()
            t2_1_t_ab = t2_1_ab.transpose(2,3,1,0).copy()

            T[s_aaa:f_aaa] = t2_1_t_a[(orb-nocc_a),:,:].reshape(-1)
            T[s_bab:f_bab] = t2_1_t_ab[(orb-nocc_a),:,:,:].reshape(-1)

######## ADC(3) 2h-1p  part  ############################################

        if(method=='adc(2)-x'or method=='adc(3)'):

            t2_2_a, t2_2_ab, t2_2_b = adc.t2[1]

            if orb >= nocc_a:
                t2_2_t = t2_2_a[ij_ind_a[0],ij_ind_a[1],:,:].copy()
                t2_2_t_a = t2_2_t.transpose(2,1,0).copy()
                t2_2_t_ab = t2_2_ab.transpose(2,3,1,0).copy()

                T[s_aaa:f_aaa] += t2_2_t_a[(orb-nocc_a),:,:].reshape(-1)
                T[s_bab:f_bab] += t2_2_t_ab[(orb-nocc_a),:,:,:].reshape(-1)

######## ADC(3) 1h part  ############################################

        if(method=='adc(3)'):

            t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_a:
                T[s_a:f_a] += 0.25*np.einsum('kdc,ikdc->i',t2_1_a[:,orb,:,:], t2_2_a, optimize = True)
                T[s_a:f_a] -= 0.25*np.einsum('kdc,ikdc->i',t2_1_ab[orb,:,:,:], t2_2_ab, optimize = True)
                T[s_a:f_a] -= 0.25*np.einsum('kcd,ikcd->i',t2_1_ab[orb,:,:,:], t2_2_ab, optimize = True)

                T[s_a:f_a] += 0.25*np.einsum('ikdc,kdc->i',t2_1_a, t2_2_a[:,orb,:,:],optimize = True)
                T[s_a:f_a] -= 0.25*np.einsum('ikcd,kcd->i',t2_1_ab, t2_2_ab[orb,:,:,:],optimize = True)
                T[s_a:f_a] -= 0.25*np.einsum('ikdc,kdc->i',t2_1_ab, t2_2_ab[orb,:,:,:],optimize = True)
            else:
                T[s_a:f_a] += 0.5*np.einsum('ikc,kc->i',t2_1_a[:,:,(orb-nocc_a),:], t1_2_a,optimize = True)
                T[s_a:f_a] += 0.5*np.einsum('ikc,kc->i',t2_1_ab[:,:,(orb-nocc_a),:], t1_2_b,optimize = True)
                T[s_a:f_a] += t1_3_a[:,(orb-nocc_a)]

######## spin = beta  ############################################
    else:
######## ADC(2) 1h part  ############################################

        if orb < nocc_b:
            T[s_b:f_b] = idn_occ_b[orb, :]
            T[s_b:f_b]+= 0.25*np.einsum('kdc,ikdc->i',t2_1_b[:,orb,:,:], t2_1_b, optimize = True)
            T[s_b:f_b]-= 0.25*np.einsum('kdc,kidc->i',t2_1_ab[:,orb,:,:], t2_1_ab, optimize = True)
            T[s_b:f_b]-= 0.25*np.einsum('kcd,kicd->i',t2_1_ab[:,orb,:,:], t2_1_ab, optimize = True)
        else :
            T[s_b:f_b] += t1_2_b[:,(orb-nocc_b)]

######## ADC(2) 2h-1p part  ############################################

            t2_1_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:].copy()
            t2_1_t_b = t2_1_t.transpose(2,1,0).copy()
            t2_1_t_ab = t2_1_ab.transpose(2,3,0,1).copy()

            T[s_bbb:f_bbb] = t2_1_t_b[(orb-nocc_b),:,:].reshape(-1)
            T[s_aba:f_aba] = t2_1_t_ab[:,(orb-nocc_b),:,:].reshape(-1)

######## ADC(3) 2h-1p part  ############################################

        if(method=='adc(2)-x'or method=='adc(3)'):

            t2_2_a, t2_2_ab, t2_2_b = adc.t2[1]

            if orb >= nocc_b:
                t2_2_t = t2_2_b[ij_ind_b[0],ij_ind_b[1],:,:].copy()
                t2_2_t_b = t2_2_t.transpose(2,1,0).copy()

                t2_2_t_ab = t2_2_ab.transpose(2,3,0,1).copy()

                T[s_bbb:f_bbb] += t2_2_t_b[(orb-nocc_b),:,:].reshape(-1)
                T[s_aba:f_aba] += t2_2_t_ab[:,(orb-nocc_b),:,:].reshape(-1)

######## ADC(3) 1h part  ############################################

        if(method=='adc(3)'):

            t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_b:
                T[s_b:f_b] += 0.25*np.einsum('kdc,ikdc->i',t2_1_b[:,orb,:,:], t2_2_b, optimize = True)
                T[s_b:f_b] -= 0.25*np.einsum('kdc,kidc->i',t2_1_ab[:,orb,:,:], t2_2_ab, optimize = True)
                T[s_b:f_b] -= 0.25*np.einsum('kcd,kicd->i',t2_1_ab[:,orb,:,:], t2_2_ab, optimize = True)

                T[s_b:f_b] += 0.25*np.einsum('ikdc,kdc->i',t2_1_b, t2_2_b[:,orb,:,:],optimize = True)
                T[s_b:f_b] -= 0.25*np.einsum('kicd,kcd->i',t2_1_ab, t2_2_ab[:,orb,:,:],optimize = True)
                T[s_b:f_b] -= 0.25*np.einsum('kidc,kdc->i',t2_1_ab, t2_2_ab[:,orb,:,:],optimize = True)
            else:
                T[s_b:f_b] += 0.5*np.einsum('ikc,kc->i',t2_1_b[:,:,(orb-nocc_b),:], t1_2_b,optimize = True)
                T[s_b:f_b] += 0.5*np.einsum('kic,kc->i',t2_1_ab[:,:,:,(orb-nocc_b)], t1_2_a,optimize = True)
                T[s_b:f_b] += t1_3_b[:,(orb-nocc_b)]

    return T


def get_trans_moments(adc):

    nmo_a  = adc.nmo_a
    nmo_b  = adc.nmo_b

    T_a = []
    T_b = []

    for orb in range(nmo_a):

            T_aa = adc.compute_trans_moments(orb, spin = "alpha")
            T_a.append(T_aa)

    T_a = np.array(T_a)

    for orb in range(nmo_b):

            T_bb = adc.compute_trans_moments(orb, spin = "beta")
            T_b.append(T_bb)

    T_b = np.array(T_b)

    return (T_a, T_b)


def get_spec_factors(adc, T, U, nroots=1):

    nmo_a  = adc.nmo_a
    nmo_b  = adc.nmo_b

    T_a = T[0]
    T_b = T[1]

    X_a = np.dot(T_a, U.T).reshape(-1,nroots)
    X_b = np.dot(T_b, U.T).reshape(-1,nroots)

    P = np.einsum("pi,pi->i", X_a, X_a)
    P += np.einsum("pi,pi->i", X_b, X_b)

    return P


class UADCEA(UADC):
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
            EA energy (eigenvalue). For nroots = 1, it is a single float number. If nroots > 1, it is a list of floats for the lowest nroots eigenvalues.
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
        self.t1 = adc.t1
        self.t2 = adc.t2
        self.e_corr = adc.e_corr
        self.method = adc.method
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

        keys = set(('e_corr', 'method', 'mo_coeff', 'mo_energy_b', 'max_memory', 't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)

    kernel = kernel
    get_imds = get_imds_ea
    matvec = ea_adc_matvec
    get_diag = ea_adc_diag
    compute_trans_moments = ea_compute_trans_moments
    get_trans_moments = get_trans_moments
    get_spec_factors = get_spec_factors

    def get_init_guess(self, nroots=1, diag=None, ascending = True):
       if diag is None :
           diag = self.ea_adc_diag()
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
        if imds is None: imds = self.get_imds(eris)
        diag = self.get_diag(imds)
        matvec = self.matvec(imds, eris)
        #matvec = lambda x: self.matvec()
        return matvec, diag


class UADCIP(UADC):
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
            Convergence threshold for Davidson iterations.  Default is 1e-12.
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
            IP energy (eigenvalue). For nroots = 1, it is a single float number. If nroots > 1, it is a list of floats for the lowest nroots eigenvalues.
        v_ip : array
            Eigenvectors for each IP transition.
        p_ip : float
            Spectroscopic amplitudes for each IP transition.
    '''
    def __init__(self, adc):
        self.verbose = adc.verbose
        self.stdout = adc.stdout
        self.max_memory = adc.max_memory
        self.max_space = adc.max_space
        self.max_cycle = adc.max_cycle
        self.conv_tol  = adc.conv_tol
        self.t1 = adc.t1
        self.t2 = adc.t2
        self.e_corr = adc.e_corr
        self.method = adc.method
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

        keys = set(('e_corr', 'method', 'mo_coeff', 'mo_energy_b', 'max_memory', 't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)

    kernel = kernel
    get_imds = get_imds_ip
    get_diag = ip_adc_diag
    matvec = ip_adc_matvec
    compute_trans_moments = ip_compute_trans_moments
    get_trans_moments = get_trans_moments
    get_spec_factors = get_spec_factors

    def get_init_guess(self, nroots=1, diag=None, ascending = True):
        if diag is None :
            diag = self.ip_adc_diag()
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
        if imds is None: imds = self.get_imds(eris)
        diag = self.get_diag(imds)
        matvec = self.matvec(imds, eris)
        #matvec = lambda x: self.matvec()
        return matvec, diag

if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    from pyscf import adc

    r = 1.098
    mol = gto.Mole()
    mol.atom = [
        ['N', ( 0., 0.    , -r/2   )],
        ['N', ( 0., 0.    ,  r/2)],]
    mol.basis = {'N':'aug-cc-pvdz'}
    mol.verbose = 0
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    myadc = adc.ADC(mf)
    ecorr, t_amp1, t_amp2 = myadc.kernel()
    print(ecorr -  -0.32201692499346535)

    myadcip = UADCIP(myadc)
    e,v,p = kernel(myadcip,nroots=3)
    print("ADC(2) IP energies")
    print (e[0] - 0.5434389897908212)
    print (e[1] - 0.5434389942222756)
    print (e[2] - 0.6240296265084732)

    print("ADC(2) IP spectroscopic factors")
    print (p[0] - 0.884404855445607)
    print (p[1] - 0.8844048539643351)
    print (p[2] - 0.9096460559671828)

    myadcea = UADCEA(myadc)
    e,v,p = kernel(myadcea,nroots=3)
    print("ADC(2) EA energies")
    print (e[0] - 0.09617819143037348)
    print (e[1] - 0.09617819161265123)
    print (e[2] - 0.12583269048810924)

    print("ADC(2) EA spectroscopic factors")
    print (p[0] - 0.991642716974455)
    print (p[1] - 0.9916427170555298)
    print (p[2] - 0.9817184409336244)

    myadc = adc.ADC(mf)
    myadc.method = "adc(3)"
    ecorr, t_amp1, t_amp2 = myadc.kernel()
    print(ecorr - -0.31694173142858517)

    myadcip = UADCIP(myadc)
    e,v,p = kernel(myadcip,nroots=3)
    print("ADC(3) IP energies")
    print (e[0] - 0.5667526838174817)
    print (e[1] - 0.5667526888293601)
    print (e[2] - 0.6099995181296374)

    print("ADC(3) IP spectroscopic factors")
    print (p[0] - 0.9086596203469742)
    print (p[1] - 0.9086596190173993)
    print (p[2] - 0.9214613318791076)

    myadcea = UADCEA(myadc)
    e,v,p = kernel(myadcea,nroots=3)

    print("ADC(3) EA energies")
    print (e[0] - 0.09836545519235675)
    print (e[1] - 0.09836545535587536)
    print (e[2] - 0.12957093060942082)

    print("ADC(3) EA spectroscopic factors")
    print (p[0] - 0.9920495578633931)
    print (p[1] - 0.992049557938337)
    print (p[2] - 0.9819274864738444)

    myadc.method = "adc(2)-x"
    myadc.kernel()

    e,v,p = myadc.ip_adc(nroots=4)
    print("ADC(2)-x IP energies")
    print (e[0] - 0.5405255355249104)
    print (e[1] - 0.5405255399061982)
    print (e[2] - 0.62080267098272)
    print (e[3] - 0.620802670982715)

    e,v,p = myadc.ea_adc(nroots=4)
    print("ADC(2)-x EA energies")
    print (e[0] - 0.09530653292650725)
    print (e[1] - 0.09530653311305577)
    print (e[2] - 0.1238833077840878)
    print (e[3] - 0.12388330873739162)
