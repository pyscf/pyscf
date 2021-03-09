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
Restricted algebraic diagrammatic construction
'''
import time
import numpy as np
import pyscf.ao2mo as ao2mo
from pyscf import lib
from pyscf.lib import logger
from pyscf.adc import radc_ao2mo
from pyscf.adc import dfadc
from pyscf import __config__
from pyscf import df

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
        eris = adc.transform_integrals()

    imds = adc.get_imds(eris)
    matvec, diag = adc.gen_matvec(imds, eris)

    guess = adc.get_init_guess(nroots, diag, ascending = True)

    conv, E, U = lib.linalg_helper.davidson_nosym1(
        lambda xs : [matvec(x) for x in xs],
        guess, diag, nroots=nroots, verbose=log, tol=adc.conv_tol,
        max_cycle=adc.max_cycle, max_space=adc.max_space)

    U = np.array(U)

    T = adc.get_trans_moments()

    spec_factors = adc.get_spec_factors(T, U, nroots)

    nfalse = np.shape(conv)[0] - np.sum(conv)
    if nfalse >= 1:
        print ("*************************************************************")
        print (" WARNING : ", "Davidson iterations for ",nfalse, "root(s) not converged")
        print ("*************************************************************")

    if adc.verbose >= logger.INFO:
        if nroots == 1:
            logger.info(adc, '%s root %d    Energy (Eh) = %.10f    Energy (eV) = %.8f'
                        '    Spec factors = %.8f    conv = %s',
                        adc.method, 0, E, E*27.2114, spec_factors, conv)
        else:
            for n, en, pn, convn in zip(range(nroots), E, spec_factors, conv):
                logger.info(adc, '%s root %d    Energy (Eh) = %.10f    Energy (eV) = %.8f'
                            '    Spec factors = %.8f    conv = %s',
                            adc.method, n, en, en*27.2114, pn, convn)
        log.timer('ADC', *cput0)

    return E, U, spec_factors


def compute_amplitudes_energy(myadc, eris, verbose=None):

    t1, t2 = myadc.compute_amplitudes(eris)
    e_corr = myadc.compute_energy(t1, t2, eris)

    return e_corr, t1, t2


def compute_amplitudes(myadc, eris):

    if myadc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(myadc.method)

    nocc = myadc._nocc
    nvir = myadc._nvir

    eris_oooo = eris.oooo
    eris_ovoo = eris.ovoo
    eris_ovov = eris.ovov
    eris_oovv = eris.oovv
    eris_ovvo = eris.ovvo
    eris_ovvv = eris.ovvv

    e = myadc.mo_energy

    d_ij = e[:nocc][:,None] + e[:nocc]

    d_ab = e[nocc:][:,None] + e[nocc:]

    D2 = d_ij.reshape(-1,1) - d_ab.reshape(-1)

    D2 = D2.reshape((nocc,nocc,nvir,nvir))

    D1 = e[:nocc][:None].reshape(-1,1) - e[nocc:].reshape(-1)
    D1 = D1.reshape((nocc,nvir))

    # Compute first-order doubles t2 (tijab)

    v2e_oovv = eris_ovov[:].transpose(0,2,1,3).copy()

    t2_1 = v2e_oovv/D2

    # Compute second-order singles t1 (tij)

    t2_1_a = t2_1 - t2_1.transpose(1,0,2,3).copy()

    eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
    t1_2 = 0.5*lib.einsum('kdac,ikcd->ia',eris_ovvv,t2_1_a,optimize=True)
    t1_2 -= 0.5*lib.einsum('kcad,ikcd->ia',eris_ovvv,t2_1_a,optimize=True)
    t1_2 += lib.einsum('kdac,ikcd->ia',eris_ovvv,t2_1,optimize=True)
    del eris_ovvv
    t1_2 -= 0.5*lib.einsum('lcki,klac->ia',eris_ovoo,t2_1_a,optimize=True)
    t1_2 -= 0.5*lib.einsum('kcli,lkac->ia',eris_ovoo,t2_1_a,optimize=True)
    t1_2 -= lib.einsum('lcki,klac->ia',eris_ovoo,t2_1,optimize=True)

    t1_2 = t1_2/D1

    t2_2 = None
    t1_3 = None

    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):

        # Compute second-order doubles t2 (tijab)

        eris_oooo = eris.oooo
        eris_ovvo = eris.ovvo

        if isinstance(eris.vvvv, np.ndarray):
            eris_vvvv = eris.vvvv
            temp = t2_1.reshape(nocc*nocc,nvir*nvir)
            t2_2 = np.dot(temp,eris_vvvv.T).reshape(nocc,nocc,nvir,nvir)
        elif isinstance(eris.vvvv, list):
            t2_2 = contract_ladder(myadc,t2_1,eris.vvvv)
        else:
            t2_2 = contract_ladder(myadc,t2_1,eris.Lvv)

        t2_2 += lib.einsum('kilj,klab->ijab',eris_oooo,t2_1,optimize=True)
        t2_2 += lib.einsum('kcbj,kica->ijab',eris_ovvo,t2_1_a,optimize=True)
        t2_2 += lib.einsum('kcbj,ikac->ijab',eris_ovvo,t2_1,optimize=True)
        t2_2 -= lib.einsum('kjbc,ikac->ijab',eris_oovv,t2_1,optimize=True)
        t2_2 -= lib.einsum('kibc,kjac->ijab',eris_oovv,t2_1,optimize=True)
        t2_2 -= lib.einsum('kjac,ikcb->ijab',eris_oovv,t2_1,optimize=True)
        t2_2 += lib.einsum('kcai,kjcb->ijab',eris_ovvo,t2_1_a,optimize=True)
        t2_2 += lib.einsum('kcai,kjcb->ijab',eris_ovvo,t2_1,optimize=True)
        t2_2 -= lib.einsum('kiac,kjcb->ijab',eris_oovv,t2_1,optimize=True)

        t2_2 = t2_2/D2

    if (myadc.method == "adc(3)"):
        # Compute third-order singles (tij)

        t2_2_a = t2_2 - t2_2.transpose(1,0,2,3).copy()

        eris_ovoo = eris.ovoo

        t1_3 = lib.einsum('d,ilad,ld->ia',e[nocc:],t2_1_a,t1_2,optimize=True)
        t1_3 += lib.einsum('d,ilad,ld->ia',e[nocc:],t2_1,t1_2,optimize=True)

        t1_3 -= lib.einsum('l,ilad,ld->ia',e[:nocc],t2_1_a, t1_2,optimize=True)
        t1_3 -= lib.einsum('l,ilad,ld->ia',e[:nocc],t2_1,t1_2,optimize=True)

        t1_3 += 0.5*lib.einsum('a,ilad,ld->ia',e[nocc:],t2_1_a, t1_2,optimize=True)
        t1_3 += 0.5*lib.einsum('a,ilad,ld->ia',e[nocc:],t2_1,t1_2,optimize=True)

        t1_3 -= 0.5*lib.einsum('i,ilad,ld->ia',e[:nocc],t2_1_a, t1_2,optimize=True)
        t1_3 -= 0.5*lib.einsum('i,ilad,ld->ia',e[:nocc],t2_1,t1_2,optimize=True)

        t1_3 += lib.einsum('ld,iald->ia',t1_2,eris_ovov,optimize=True)
        t1_3 -= lib.einsum('ld,laid->ia',t1_2,eris_ovov,optimize=True)
        t1_3 += lib.einsum('ld,iald->ia',t1_2,eris_ovov,optimize=True)

        t1_3 += lib.einsum('ld,ldai->ia',t1_2,eris_ovvo ,optimize=True)
        t1_3 -= lib.einsum('ld,liad->ia',t1_2,eris_oovv ,optimize=True)
        t1_3 += lib.einsum('ld,ldai->ia',t1_2,eris_ovvo,optimize=True)

        t1_3 -= 0.5*lib.einsum('lmad,mdli->ia',t2_2_a,eris_ovoo,optimize=True)
        t1_3 += 0.5*lib.einsum('lmad,ldmi->ia',t2_2_a,eris_ovoo,optimize=True)
        t1_3 -=     lib.einsum('lmad,mdli->ia',t2_2,eris_ovoo,optimize=True)

        eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
        t1_3 += 0.5*lib.einsum('ilde,lead->ia',t2_2_a,eris_ovvv,optimize=True)
        t1_3 -= 0.5*lib.einsum('ilde,ldae->ia',t2_2_a,eris_ovvv,optimize=True)
        t1_3 -= lib.einsum('ildf,mefa,lmde->ia',t2_1_a, eris_ovvv,  t2_1_a ,optimize=True)
        t1_3 += lib.einsum('ildf,mafe,lmde->ia',t2_1_a, eris_ovvv,  t2_1_a ,optimize=True)
        t1_3 += lib.einsum('ilfd,mefa,mled->ia',t2_1,eris_ovvv, t2_1,optimize=True)
        t1_3 -= lib.einsum('ilfd,mafe,mled->ia',t2_1,eris_ovvv, t2_1,optimize=True)
        t1_3 += 0.5*lib.einsum('ilaf,mefd,lmde->ia',t2_1_a,eris_ovvv,t2_1_a,optimize=True)
        t1_3 -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',t2_1_a,eris_ovvv,t2_1_a,optimize=True)
        t1_3 += 0.5*lib.einsum('lmdf,iaef,lmde->ia',t2_1_a,eris_ovvv,t2_1_a,optimize=True)
        t1_3 -= 0.5*lib.einsum('lmdf,ifea,lmde->ia',t2_1_a,eris_ovvv,t2_1_a,optimize=True)
        t1_3 += lib.einsum('mlfd,iaef,mled->ia',t2_1,eris_ovvv,t2_1,optimize=True)
        t1_3 -= lib.einsum('mlfd,ifea,mled->ia',t2_1,eris_ovvv,t2_1,optimize=True)
        t1_3 -= 0.25*lib.einsum('lmef,iedf,lmad->ia',t2_1_a,eris_ovvv,t2_1_a,optimize=True)
        t1_3 += 0.25*lib.einsum('lmef,ifde,lmad->ia',t2_1_a,eris_ovvv,t2_1_a,optimize=True)

        t1_3 += 0.5*lib.einsum('ilaf,mefd,lmde->ia',t2_1,eris_ovvv,t2_1_a,optimize=True)
        t1_3 -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',t2_1,eris_ovvv,t2_1_a,optimize=True)

        t1_3 -= lib.einsum('ildf,mafe,mlde->ia',t2_1,eris_ovvv,t2_1,optimize=True)
        t1_3 += lib.einsum('ilaf,mefd,mled->ia',t2_1,eris_ovvv,t2_1,optimize=True)
        t1_3 += 0.5*lib.einsum('lmdf,iaef,lmde->ia',t2_1_a,eris_ovvv,t2_1_a,optimize=True)
        t1_3 += lib.einsum('lmdf,iaef,lmde->ia',t2_1,eris_ovvv,t2_1,optimize=True)
        t1_3 -= lib.einsum('lmef,iedf,lmad->ia',t2_1,eris_ovvv,t2_1,optimize=True)

        t1_3 += lib.einsum('ilde,lead->ia',t2_2,eris_ovvv,optimize=True)
        t1_3 -= lib.einsum('ildf,mefa,lmde->ia',t2_1_a,eris_ovvv, t2_1,optimize=True)
        t1_3 += lib.einsum('ilfd,mefa,lmde->ia',t2_1,eris_ovvv,t2_1_a ,optimize=True)
        t1_3 += lib.einsum('ilaf,mefd,lmde->ia',t2_1_a,eris_ovvv,t2_1,optimize=True)
        del eris_ovvv

        t1_3 += 0.25*lib.einsum('inde,lamn,lmde->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3 -= 0.25*lib.einsum('inde,maln,lmde->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3 += lib.einsum('inde,lamn,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)

        t1_3 += 0.5*lib.einsum('inad,lemn,lmde->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3 -= 0.5*lib.einsum('inad,meln,lmde->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3 -= 0.5 * lib.einsum('inad,lemn,mlde->ia',t2_1_a,eris_ovoo,t2_1,optimize=True)
        t1_3 -= 0.5 * lib.einsum('inad,meln,lmde->ia',t2_1_a,eris_ovoo,t2_1,optimize=True)
        t1_3 -= 0.5 *lib.einsum('inad,lemn,lmed->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= 0.5*lib.einsum('inad,meln,mled->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += 0.5*lib.einsum('inad,lemn,lmde->ia',t2_1,eris_ovoo,t2_1_a,optimize=True)
        t1_3 -= 0.5*lib.einsum('inad,meln,lmde->ia',t2_1,eris_ovoo,t2_1_a,optimize=True)

        t1_3 -= 0.5*lib.einsum('lnde,ianm,lmde->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3 += 0.5*lib.einsum('lnde,naim,lmde->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3 -= lib.einsum('nled,ianm,mled->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += lib.einsum('nled,naim,mled->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= 0.5*lib.einsum('lnde,ianm,lmde->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3 -= lib.einsum('lnde,ianm,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)

        t1_3 -= lib.einsum('lnde,ienm,lmad->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3 += lib.einsum('lnde,neim,lmad->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3 += lib.einsum('lnde,neim,lmad->ia',t2_1,eris_ovoo,t2_1_a,optimize=True)
        t1_3 += lib.einsum('nled,ienm,mlad->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= lib.einsum('nled,neim,mlad->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += lib.einsum('lned,ienm,lmad->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= lib.einsum('lnde,neim,mlad->ia',t2_1_a,eris_ovoo,t2_1,optimize=True)

        t1_3 = t1_3/D1

    t1 = (t1_2, t1_3)
    t2 = (t2_1, t2_2)

    return t1, t2


def compute_energy(myadc, t1, t2, eris):

    if myadc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(myadc.method)

    nocc = myadc._nocc
    nvir = myadc._nvir

    eris_ovov = eris.ovov

    t2_1  = t2[0]

    #Compute MP2 correlation energy

    e_mp2 = 0.5 * lib.einsum('ijab,iajb', t2_1, eris_ovov,optimize=True)
    e_mp2 -= 0.5 * lib.einsum('ijab,ibja', t2_1, eris_ovov,optimize=True)
    e_mp2 -= 0.5 * lib.einsum('jiab,iajb', t2_1, eris_ovov,optimize=True)
    e_mp2 += 0.5 * lib.einsum('jiab,ibja', t2_1, eris_ovov,optimize=True)
    e_mp2 += lib.einsum('ijab,iajb', t2_1, eris_ovov,optimize=True)

    e_corr = e_mp2

    if (myadc.method == "adc(3)"):

        t2_1_a = t2_1 - t2_1.transpose(1,0,2,3).copy()

        #Compute MP3 correlation energy
        eris_oovv = eris.oovv
        eris_ovvo = eris.ovvo
        eris_oooo = eris.oooo

        temp_t2_a = None
        temp_t2_ab = None
        temp_t2_t2 = None
        temp_t2a_t2a = None
        temp_t2_a_vvvv = None
        temp_t2_ab_vvvv = None

        eris_vvvv = eris.vvvv

        if isinstance(eris.vvvv, np.ndarray):
            temp_t2_a = t2_1_a.reshape(nocc*nocc,nvir*nvir)
            temp_t2_a_vvvv = np.dot(temp_t2_a,eris_vvvv.T).reshape(nocc,nocc,nvir,nvir)
        elif isinstance(eris.vvvv, list):
            temp_t2_a_vvvv = contract_ladder(myadc,t2_1_a,eris.vvvv)
        else:
            temp_t2_a_vvvv = contract_ladder(myadc,t2_1_a,eris.Lvv)

        if isinstance(eris.vvvv, np.ndarray):
            temp_t2_ab = t2_1.reshape(nocc*nocc,nvir*nvir)
            temp_t2_ab_vvvv = np.dot(temp_t2_ab,eris_vvvv.T).reshape(nocc,nocc,nvir,nvir)
        elif isinstance(eris.vvvv, list):
            temp_t2_ab_vvvv = contract_ladder(myadc,t2_1,eris.vvvv)
        else:
            temp_t2_ab_vvvv = contract_ladder(myadc,t2_1,eris.Lvv)


        e_mp3 =  lib.einsum('ijcd,ijcd',temp_t2_ab_vvvv, t2_1,optimize=True)
        del temp_t2_ab_vvvv


        if isinstance(eris.vvvv, np.ndarray):
            temp_t2_a = temp_t2_a.reshape(nocc,nocc,nvir,nvir)
            temp_t2_a = np.ascontiguousarray(temp_t2_a.transpose(0,1,3,2))
            temp_t2_a = temp_t2_a.reshape(nocc*nocc,nvir*nvir)
            temp_t2_a_vvvv -= np.dot(temp_t2_a,eris_vvvv.T).reshape(nocc,nocc,nvir,nvir)
        elif isinstance(eris.vvvv, list):
            t2_1_a_t = np.ascontiguousarray(t2_1_a.transpose(0,1,3,2))
            temp_t2_a_vvvv_n = contract_ladder(myadc,t2_1_a_t,eris.vvvv)
            temp_t2_a_vvvv -= temp_t2_a_vvvv_n
        else:
            t2_1_a_t = np.ascontiguousarray(t2_1_a.transpose(0,1,3,2))
            temp_t2_a_vvvv_n = contract_ladder(myadc,t2_1_a_t,eris.Lvv)
            temp_t2_a_vvvv -= temp_t2_a_vvvv_n

        e_mp3 += 0.25 * lib.einsum('ijcd,ijcd',temp_t2_a_vvvv, t2_1_a,optimize=True)

        temp_t2a_t2a =  lib.einsum('ijab,klab', t2_1_a, t2_1_a,optimize=True)
        e_mp3 += 0.25 * lib.einsum('ijkl,ikjl',temp_t2a_t2a, eris_oooo,optimize=True)
        e_mp3 -= 0.25 * lib.einsum('ijkl,iljk',temp_t2a_t2a, eris_oooo,optimize=True)
        del temp_t2a_t2a

        temp_t2_t2 =  lib.einsum('ijab,klab', t2_1, t2_1,optimize=True)
        e_mp3 +=  lib.einsum('ijkl,ikjl',temp_t2_t2, eris_oooo,optimize=True)
        del temp_t2_t2

        temp_t2_t2 = lib.einsum('ijab,ikcb->akcj', t2_1_a, t2_1_a,optimize=True)
        temp_t2_t2 += lib.einsum('jiab,kicb->akcj', t2_1, t2_1,optimize=True)
        e_mp3 -= 2 * lib.einsum('akcj,kjac',temp_t2_t2, eris_oovv,optimize=True)
        e_mp3 += 2 * lib.einsum('akcj,kcaj',temp_t2_t2, eris_ovvo,optimize=True)
        del temp_t2_t2

        temp_t2_t2 = lib.einsum('ijab,ikcb->akcj', t2_1, t2_1,optimize=True)
        e_mp3 -= lib.einsum('akcj,kjac',temp_t2_t2, eris_oovv,optimize=True)
        del temp_t2_t2

        temp_t2_t2 = lib.einsum('jiba,kibc->akcj', t2_1, t2_1,optimize=True)
        e_mp3 -= lib.einsum('akcj,kjac',temp_t2_t2, eris_oovv,optimize=True)
        del temp_t2_t2

        temp_t2_t2 = -lib.einsum('ijab,ikbc->akcj', t2_1_a, t2_1,optimize=True)
        temp_t2_t2 -= lib.einsum('jiab,ikcb->akcj', t2_1, t2_1_a,optimize=True)
        e_mp3 += lib.einsum('akcj,kcaj',temp_t2_t2, eris_ovvo,optimize=True)
        del temp_t2_t2

        temp_t2_t2 = -lib.einsum('ijba,ikcb->akcj', t2_1, t2_1_a,optimize=True)
        temp_t2_t2 -= lib.einsum('ijab,kicb->akcj', t2_1_a, t2_1,optimize=True)
        e_mp3 += lib.einsum('akcj,kcaj',temp_t2_t2, eris_ovvo,optimize=True)
        del temp_t2_t2

        e_corr += e_mp3

    return e_corr


def contract_ladder(myadc,t_amp,vvvv):

    nocc = myadc._nocc
    nvir = myadc._nvir

    t_amp_t = np.ascontiguousarray(t_amp.reshape(nocc*nocc,nvir*nvir).T)
    t = np.zeros((nvir,nvir, nocc*nocc))
    chnk_size = radc_ao2mo.calculate_chunk_size(myadc)

    a = 0
    if isinstance(vvvv, list):
        for dataset in vvvv:
            k = dataset.shape[0]
            dataset = dataset[:].reshape(-1,nvir*nvir)
            t[a:a+k] = np.dot(dataset,t_amp_t).reshape(-1,nvir,nocc*nocc)
            a += k
    elif getattr(myadc, 'with_df', None):
        for p in range(0,nvir,chnk_size):
            vvvv_p = dfadc.get_vvvv_df(myadc, vvvv, p, chnk_size)
            k = vvvv_p.shape[0]
            vvvv_p = vvvv_p.reshape(-1,nvir*nvir)
            t[a:a+k] = np.dot(vvvv_p,t_amp_t).reshape(-1,nvir,nocc*nocc)
            del (vvvv_p)
            a += k
    else:
        raise Exception("Unknown vvvv type")

    t = np.ascontiguousarray(t.transpose(2,0,1)).reshape(nocc, nocc, nvir, nvir)

    return t


def density_matrix(myadc, T=None):

    if T is None:
        T = RADCIP(myadc).get_trans_moments()

    nocc = myadc._nocc
    nvir = myadc._nvir

    n_singles = nocc
    n_doubles = nvir * nocc * nocc

    s1 = 0
    f1 = n_singles

    T_doubles = T[:,n_singles:]
    T_doubles = T_doubles.reshape(-1,nvir,nocc,nocc)
    T_doubles_transpose = T_doubles.transpose(0,1,3,2).copy()
    T_bab = (2/3)*T_doubles + (1/3)*T_doubles_transpose

    T_aaa = T_bab - T_bab.transpose(0,1,3,2)

    T_a = T[:,s1:f1]
    T_bab = T_bab.reshape(-1,n_doubles)
    T_aaa = T_aaa.reshape(-1,n_doubles)

    dm = 2 * np.dot(T_a,T_a.T) + np.dot(T_aaa, T_aaa.T) + 2 * np.dot(T_bab, T_bab.T)

    return dm


class RADC(lib.StreamObject):
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
            >>> myadc = adc.RADC(mf).run()

    Saved results

        e_corr : float
            MPn correlation correction
        e_tot : float
            Total energy (HF + correlation)
        t1, t2 :
            T amplitudes t1[i,a], t2[i,j,a,b]  (i,j in occ, a,b in virt)
    '''
    incore_complete = getattr(__config__, 'adc_radc_RADC_incore_complete', False)
    async_io = getattr(__config__, 'adc_radc_RADC_async_io', True)
    blkmin = getattr(__config__, 'adc_radc_RADC_blkmin', 4)
    memorymin = getattr(__config__, 'adc_radc_RADC_memorymin', 2000)

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        from pyscf import gto

        if 'dft' in str(mf.__module__):
            raise NotImplementedError('DFT reference for UADC')

        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if mo_occ is None: mo_occ = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.max_space = getattr(__config__, 'adc_radc_RADC_max_space', 12)
        self.max_cycle = getattr(__config__, 'adc_radc_RADC_max_cycle', 50)
        self.conv_tol = getattr(__config__, 'adc_radc_RADC_conv_tol', 1e-12)
        self.scf_energy = mf.e_tot

        self.frozen = frozen
        self.incore_complete = self.incore_complete or self.mol.incore_anyway

        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.e_corr = None
        self.e_tot = None
        self.t1 = None
        self.t2 = None
        self._nocc = mf.mol.nelectron//2
        self._nmo = mo_coeff.shape[1]
        self._nvir = self._nmo - self._nocc
        self.mo_energy = mf.mo_energy
        self.chkfile = mf.chkfile
        self.method = "adc(2)"
        self.method_type = "ip"
        self.with_df = None

        keys = set(('conv_tol', 'e_corr', 'method', 'mo_coeff', 'mol',
                    'mo_energy', 'max_memory', 'incore_complete',
                    'scf_energy', 'e_tot', 't1', 'frozen', 'chkfile',
                    'max_space', 't2', 'mo_occ', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)

    compute_amplitudes = compute_amplitudes
    compute_energy = compute_energy
    transform_integrals = radc_ao2mo.transform_integrals_incore
    make_rdm1 = density_matrix

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

    def kernel_gs(self):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)

        self.method = self.method.lower()
        if self.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
            raise NotImplementedError(self.method)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags_gs()

        nmo = self._nmo
        nao = self.mo_coeff.shape[0]
        nmo_pair = nmo * (nmo+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]

        if getattr(self, 'with_df', None) or getattr(self._scf, 'with_df', None):
            if getattr(self, 'with_df', None):
                self.with_df = self.with_df
            else:
                self.with_df = self._scf.with_df

            def df_transform():
                return radc_ao2mo.transform_integrals_df(self)
            self.transform_integrals = df_transform
        elif (self._scf._eri is None or
              (mem_incore+mem_now >= self.max_memory and not self.incore_complete)):
            def outcore_transform():
                return radc_ao2mo.transform_integrals_outcore(self)
            self.transform_integrals = outcore_transform

        eris = self.transform_integrals()

        self.e_corr, self.t1, self.t2 = compute_amplitudes_energy(self, eris=eris, verbose=self.verbose)
        self.e_tot = self.scf_energy + self.e_corr

        self._finalize()

        return self.e_corr, self.t1, self.t2

    def kernel(self, nroots=1, guess=None, eris=None):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)

        self.method = self.method.lower()
        if self.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
            raise NotImplementedError(self.method)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags_gs()

        nmo = self._nmo
        nao = self.mo_coeff.shape[0]
        nmo_pair = nmo * (nmo+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]

        if getattr(self, 'with_df', None) or getattr(self._scf, 'with_df', None):
            if getattr(self, 'with_df', None):
                self.with_df = self.with_df
            else:
                self.with_df = self._scf.with_df

            def df_transform():
                return radc_ao2mo.transform_integrals_df(self)
            self.transform_integrals = df_transform
        elif (self._scf._eri is None or
              (mem_incore+mem_now >= self.max_memory and not self.incore_complete)):
            def outcore_transform():
                return radc_ao2mo.transform_integrals_outcore(self)
            self.transform_integrals = outcore_transform

        eris = self.transform_integrals()

        self.e_corr, self.t1, self.t2 = compute_amplitudes_energy(self, eris=eris, verbose=self.verbose)
        self.e_tot = self.scf_energy + self.e_corr

        self._finalize()

        self.method_type = self.method_type.lower()
        if(self.method_type == "ea"):
            e_exc, v_exc, spec_fac = self.ea_adc(nroots=nroots, guess=guess, eris=eris)

        elif(self.method_type == "ip"):
            e_exc, v_exc, spec_fac = self.ip_adc(nroots=nroots, guess=guess, eris=eris)

        else:
            raise NotImplementedError(self.method_type)

        return e_exc, v_exc, spec_fac

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        logger.note(self, 'E_corr = %.8f  E_tot = %.8f',
                    self.e_corr, self.e_tot)
        return self

    def ea_adc(self, nroots=1, guess=None, eris=None):
        return RADCEA(self).kernel(nroots, guess, eris)

    def ip_adc(self, nroots=1, guess=None, eris=None):
        return RADCIP(self).kernel(nroots, guess, eris)

    def density_fit(self, auxbasis=None, with_df = None):
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


def get_imds_ea(adc, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t1 = adc.t1
    t2 = adc.t2

    t1_2 = t1[0]
    t2_1 = t2[0]

    t2_1_a = t2_1 - t2_1.transpose(1,0,2,3).copy()

    nocc = adc._nocc
    nvir = adc._nvir

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    idn_vir = np.identity(nvir)

    if eris is None:
        eris = adc.transform_integrals()

    eris_ovov = eris.ovov

    # a-b block
    # Zeroth-order terms

    M_ab = lib.einsum('ab,a->ab', idn_vir, e_vir)

    # Second-order terms

    M_ab +=  lib.einsum('l,lmad,lmbd->ab',e_occ ,t2_1_a, t2_1_a,optimize=True)
    M_ab +=  lib.einsum('l,lmad,lmbd->ab',e_occ,t2_1, t2_1,optimize=True)
    M_ab +=  lib.einsum('l,mlad,mlbd->ab',e_occ,t2_1, t2_1,optimize=True)

    M_ab -= 0.5 *  lib.einsum('d,lmad,lmbd->ab',e_vir,t2_1_a, t2_1_a,optimize=True)
    M_ab -= 0.5 *  lib.einsum('d,lmad,lmbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab -= 0.5 *  lib.einsum('d,mlad,mlbd->ab',e_vir,t2_1, t2_1,optimize=True)

    M_ab -= 0.25 *  lib.einsum('a,lmad,lmbd->ab',e_vir,t2_1_a, t2_1_a,optimize=True)
    M_ab -= 0.25 *  lib.einsum('a,lmad,lmbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab -= 0.25 *  lib.einsum('a,mlad,mlbd->ab',e_vir,t2_1, t2_1,optimize=True)

    M_ab -= 0.25 *  lib.einsum('b,lmad,lmbd->ab',e_vir,t2_1_a, t2_1_a,optimize=True)
    M_ab -= 0.25 *  lib.einsum('b,lmad,lmbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab -= 0.25 *  lib.einsum('b,mlad,mlbd->ab',e_vir,t2_1, t2_1,optimize=True)

    M_ab -= 0.5 *  lib.einsum('lmad,lbmd->ab',t2_1_a, eris_ovov,optimize=True)
    M_ab += 0.5 *  lib.einsum('lmad,ldmb->ab',t2_1_a, eris_ovov,optimize=True)
    M_ab -=        lib.einsum('lmad,lbmd->ab',t2_1, eris_ovov,optimize=True)

    M_ab -= 0.5 *  lib.einsum('lmbd,lamd->ab',t2_1_a, eris_ovov,optimize=True)
    M_ab += 0.5 *  lib.einsum('lmbd,ldma->ab',t2_1_a, eris_ovov,optimize=True)
    M_ab -=        lib.einsum('lmbd,lamd->ab',t2_1, eris_ovov,optimize=True)

    #Third-order terms

    if(method =='adc(3)'):

        t2_2 = t2[1]
        t2_2_a = t2_2 - t2_2.transpose(1,0,2,3).copy()

        eris_oovv = eris.oovv
        eris_ovvo = eris.ovvo
        eris_oooo = eris.oooo

        eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
        M_ab += 4. * lib.einsum('ld,ldab->ab',t1_2, eris_ovvv,optimize=True)
        M_ab -=  lib.einsum('ld,lbad->ab',t1_2, eris_ovvv,optimize=True)
        M_ab -= lib.einsum('ld,ladb->ab',t1_2, eris_ovvv,optimize=True)
        del eris_ovvv

        M_ab -= 0.5 *  lib.einsum('lmad,lbmd->ab',t2_2_a, eris_ovov,optimize=True)
        M_ab += 0.5 *  lib.einsum('lmad,ldmb->ab',t2_2_a, eris_ovov,optimize=True)
        M_ab -=        lib.einsum('lmad,lbmd->ab',t2_2, eris_ovov,optimize=True)

        M_ab -= 0.5 * lib.einsum('lmbd,lamd->ab',t2_2_a,eris_ovov,optimize=True)
        M_ab += 0.5 * lib.einsum('lmbd,ldma->ab',t2_2_a, eris_ovov,optimize=True)
        M_ab -=       lib.einsum('lmbd,lamd->ab',t2_2,eris_ovov,optimize=True)

        M_ab += lib.einsum('l,lmbd,lmad->ab',e_occ, t2_1_a, t2_2_a, optimize=True)
        M_ab += lib.einsum('l,lmbd,lmad->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab += lib.einsum('l,mlbd,mlad->ab',e_occ, t2_1, t2_2, optimize=True)

        M_ab += lib.einsum('l,lmad,lmbd->ab',e_occ, t2_1_a, t2_2_a, optimize=True)
        M_ab += lib.einsum('l,lmad,lmbd->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab += lib.einsum('l,mlad,mlbd->ab',e_occ, t2_1, t2_2, optimize=True)

        M_ab -= 0.5*lib.einsum('d,lmbd,lmad->ab', e_vir, t2_1_a ,t2_2_a, optimize=True)
        M_ab -= 0.5*lib.einsum('d,lmbd,lmad->ab', e_vir, t2_1 ,t2_2, optimize=True)
        M_ab -= 0.5*lib.einsum('d,mlbd,mlad->ab', e_vir, t2_1 ,t2_2, optimize=True)

        M_ab -= 0.5*lib.einsum('d,lmad,lmbd->ab', e_vir, t2_1_a, t2_2_a, optimize=True)
        M_ab -= 0.5*lib.einsum('d,lmad,lmbd->ab', e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.5*lib.einsum('d,mlad,mlbd->ab', e_vir, t2_1, t2_2, optimize=True)

        M_ab -= 0.25*lib.einsum('a,lmbd,lmad->ab',e_vir, t2_1_a, t2_2_a, optimize=True)
        M_ab -= 0.25*lib.einsum('a,lmbd,lmad->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.25*lib.einsum('a,mlbd,mlad->ab',e_vir, t2_1, t2_2, optimize=True)

        M_ab -= 0.25*lib.einsum('a,lmad,lmbd->ab',e_vir, t2_1_a, t2_2_a, optimize=True)
        M_ab -= 0.25*lib.einsum('a,lmad,lmbd->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.25*lib.einsum('a,mlad,mlbd->ab',e_vir, t2_1, t2_2, optimize=True)

        M_ab -= 0.25*lib.einsum('b,lmbd,lmad->ab',e_vir, t2_1_a, t2_2_a, optimize=True)
        M_ab -= 0.25*lib.einsum('b,lmbd,lmad->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.25*lib.einsum('b,mlbd,mlad->ab',e_vir, t2_1, t2_2, optimize=True)

        M_ab -= 0.25*lib.einsum('b,lmad,lmbd->ab',e_vir, t2_1_a, t2_2_a, optimize=True)
        M_ab -= 0.25*lib.einsum('b,lmad,lmbd->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.25*lib.einsum('b,mlad,mlbd->ab',e_vir, t2_1, t2_2, optimize=True)

        M_ab -= lib.einsum('lned,mlbd,nmae->ab',t2_1_a, t2_1_a, eris_oovv, optimize=True)
        M_ab += lib.einsum('lned,mlbd,mane->ab',t2_1_a, t2_1_a, eris_ovov, optimize=True)
        M_ab += lib.einsum('nled,mlbd,nmae->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab -= lib.einsum('nled,mlbd,mane->ab',t2_1, t2_1, eris_ovov, optimize=True)
        M_ab -= lib.einsum('lnde,mlbd,neam->ab',t2_1, t2_1_a, eris_ovvo, optimize=True)
        M_ab += lib.einsum('lned,mlbd,neam->ab',t2_1_a, t2_1, eris_ovvo, optimize=True)
        M_ab += lib.einsum('lned,lmbd,nmae->ab',t2_1, t2_1, eris_oovv, optimize=True)

        M_ab -= lib.einsum('mled,lnad,nmeb->ab',t2_1_a, t2_1_a, eris_oovv, optimize=True)
        M_ab += lib.einsum('mled,lnad,nbem->ab',t2_1_a, t2_1_a, eris_ovvo, optimize=True)
        M_ab += lib.einsum('mled,nlad,nmeb->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab -= lib.einsum('mled,nlad,nbem->ab',t2_1, t2_1, eris_ovvo, optimize=True)
        M_ab += lib.einsum('lmed,lnad,nmeb->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab -= lib.einsum('mled,nlad,nbem->ab',t2_1_a, t2_1, eris_ovvo, optimize=True)
        M_ab += lib.einsum('lmde,lnad,nbem->ab',t2_1, t2_1_a, eris_ovvo, optimize=True)

        M_ab -= lib.einsum('mlbd,lnae,nmde->ab',t2_1_a, t2_1_a,   eris_oovv, optimize=True)
        M_ab += lib.einsum('mlbd,lnae,nedm->ab',t2_1_a, t2_1_a,   eris_ovvo, optimize=True)
        M_ab += lib.einsum('lmbd,lnae,nmde->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab -= lib.einsum('lmbd,lnae,nedm->ab',t2_1, t2_1, eris_ovvo, optimize=True)
        M_ab += lib.einsum('mlbd,lnae,nedm->ab',t2_1_a, t2_1,  eris_ovvo, optimize=True)
        M_ab -= lib.einsum('lmbd,lnae,nedm->ab',t2_1, t2_1_a,  eris_ovvo, optimize=True)
        M_ab += lib.einsum('mlbd,nlae,nmde->ab',t2_1, t2_1, eris_oovv, optimize=True)

        M_ab += 0.5*lib.einsum('lned,mled,nmab->ab',t2_1_a, t2_1_a, eris_oovv, optimize=True)
        M_ab -= 0.5*lib.einsum('lned,mled,nbam->ab',t2_1_a, t2_1_a, eris_ovvo, optimize=True)
        M_ab -= lib.einsum('nled,mled,nmab->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab += lib.einsum('nled,mled,nbam->ab',t2_1, t2_1, eris_ovvo, optimize=True)
        M_ab += 0.5*lib.einsum('lned,mled,nmab->ab',t2_1_a, t2_1_a, eris_oovv, optimize=True)
        M_ab -= lib.einsum('lned,lmed,nmab->ab',t2_1, t2_1, eris_oovv, optimize=True)

        M_ab -= 0.25*lib.einsum('mlbd,noad,nmol->ab',t2_1_a, t2_1_a, eris_oooo, optimize=True)
        M_ab += 0.25*lib.einsum('mlbd,noad,nlom->ab',t2_1_a, t2_1_a, eris_oooo, optimize=True)
        M_ab -= lib.einsum('mlbd,noad,nmol->ab',t2_1, t2_1, eris_oooo, optimize=True)


        if isinstance(eris.vvvv, np.ndarray):
            t2_1_a_r = t2_1_a.reshape(nocc*nocc,nvir*nvir)
            t2_1_r = t2_1.reshape(nocc*nocc,nvir*nvir)
            eris_vvvv = eris.vvvv
            temp_t2a = np.dot(t2_1_a_r,eris_vvvv)
            temp_t2 = np.dot(t2_1_r,eris_vvvv)
            temp_t2a = temp_t2a.reshape(nocc,nocc,nvir,nvir)
            temp_t2 = temp_t2.reshape(nocc,nocc,nvir,nvir)
            M_ab -= 0.25*lib.einsum('mlaf,mlbf->ab',t2_1_a, temp_t2a, optimize=True)
            M_ab += 0.25*lib.einsum('mlaf,mlfb->ab',t2_1_a, temp_t2a, optimize=True)
            M_ab -= lib.einsum('mlaf,mlbf->ab',t2_1, temp_t2, optimize=True)
            temp_vvvv_t2a = np.dot(eris_vvvv,t2_1_a_r.T)
            temp_vvvv_t2a = temp_vvvv_t2a.reshape(nvir,nvir,nocc,nocc)
            M_ab += 0.25*lib.einsum('adlm,mlbd->ab',temp_vvvv_t2a, t2_1_a, optimize=True)
        else:
            if isinstance(eris.vvvv, list):
                temp_t2a_vvvv = contract_ladder(adc,t2_1_a,eris.vvvv)
                temp_t2_vvvv = contract_ladder(adc,t2_1,eris.vvvv)
            else:
                temp_t2a_vvvv = contract_ladder(adc,t2_1_a,eris.Lvv)
                temp_t2_vvvv = contract_ladder(adc,t2_1,eris.Lvv)

            M_ab -= 0.25*lib.einsum('mlaf,mlbf->ab',t2_1_a, temp_t2a_vvvv, optimize=True)
            M_ab += 0.25*lib.einsum('mlaf,mlfb->ab',t2_1_a, temp_t2a_vvvv, optimize=True)
            M_ab -= lib.einsum('mlaf,mlbf->ab',t2_1, temp_t2_vvvv, optimize=True)
            M_ab += 0.25*lib.einsum('lmad,mlbd->ab',temp_t2a_vvvv, t2_1_a, optimize=True)


        if isinstance(eris.vvvv, np.ndarray):
            t2_1_a_temp = t2_1_a.reshape(nocc*nocc,nvir*nvir)
            t2_1_temp = t2_1.reshape(nocc*nocc,nvir*nvir)
            temp_1 = np.dot(t2_1_temp,eris_vvvv.T).reshape(nocc,nocc,nvir,nvir)
            temp_1_a = np.dot(t2_1_a_temp,eris_vvvv.T).reshape(nocc,nocc,nvir,nvir)

            M_ab -= 0.25*lib.einsum('mlad,mlbd->ab', temp_1_a, t2_1_a, optimize=True)
            M_ab -= lib.einsum('mlad,mlbd->ab', temp_1, t2_1, optimize=True)

            eris_vvvv = eris_vvvv.reshape(nvir,nvir,nvir,nvir)
            M_ab -= lib.einsum('mldf,mled,aebf->ab',t2_1_a, t2_1_a, eris_vvvv, optimize=True)
            M_ab += 0.5*lib.einsum('mldf,mled,aefb->ab',t2_1_a, t2_1_a, eris_vvvv, optimize=True)
            M_ab += 2.*lib.einsum('mlfd,mled,aebf->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab -= lib.einsum('mlfd,mled,aefb->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            eris_vvvv = eris_vvvv.reshape(nvir*nvir,nvir*nvir)
        else:
            if isinstance(eris.vvvv, list):
                temp_t2_a_vvvv = contract_ladder(adc,t2_1_a,eris.vvvv)
                temp_t2_vvvv = contract_ladder(adc,t2_1,eris.vvvv)
            else:
                temp_t2_a_vvvv = contract_ladder(adc,t2_1_a,eris.Lvv)
                temp_t2_vvvv = contract_ladder(adc,t2_1,eris.Lvv)
            M_ab -= 0.25*lib.einsum('mlad,mlbd->ab', temp_t2_a_vvvv, t2_1_a, optimize=True)
            M_ab -= lib.einsum('mlad,mlbd->ab', temp_t2_vvvv, t2_1, optimize=True)

            chnk_size = radc_ao2mo.calculate_chunk_size(adc)
            a = 0
            temp = np.zeros((nvir,nvir))

            if isinstance(eris.vvvv, list):
                for dataset in eris.vvvv:
                    k = dataset.shape[0]
                    eris_vvvv = dataset[:].reshape(-1,nvir,nvir,nvir)
                    temp[a:a+k] -= lib.einsum('mldf,mled,aebf->ab',t2_1_a, t2_1_a,  eris_vvvv, optimize=True)
                    temp[a:a+k] += 0.5*lib.einsum('mldf,mled,aefb->ab',t2_1_a, t2_1_a,  eris_vvvv, optimize=True)
                    temp[a:a+k] += 2.*lib.einsum('mlfd,mled,aebf->ab',t2_1, t2_1, eris_vvvv, optimize=True)
                    temp[a:a+k] -= lib.einsum('mlfd,mled,aefb->ab',t2_1, t2_1, eris_vvvv, optimize=True)
                    del eris_vvvv
                    a += k
            else:

                for p in range(0,nvir,chnk_size):

                    vvvv = dfadc.get_vvvv_df(adc, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
                    k = vvvv.shape[0]
                    temp[a:a+k] -= lib.einsum('mldf,mled,aebf->ab',t2_1_a, t2_1_a,  vvvv, optimize=True)
                    temp[a:a+k] += 0.5*lib.einsum('mldf,mled,aefb->ab',t2_1_a, t2_1_a,  vvvv, optimize=True)
                    temp[a:a+k] += 2.*lib.einsum('mlfd,mled,aebf->ab',t2_1, t2_1, vvvv, optimize=True)
                    temp[a:a+k] -= lib.einsum('mlfd,mled,aefb->ab',t2_1, t2_1, vvvv, optimize=True)
                    del vvvv
                    a += k

            M_ab += temp


    return M_ab


def get_imds_ip(adc, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t1 = adc.t1
    t2 = adc.t2

    t1_2 = t1[0]
    t2_1 = t2[0]

    t2_1_a = t2_1 - t2_1.transpose(1,0,2,3).copy()

    nocc = adc._nocc
    nvir = adc._nvir

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    idn_occ = np.identity(nocc)

    if eris is None:
        eris = adc.transform_integrals()

    eris_ovov = eris.ovov

    # i-j block
    # Zeroth-order terms

    M_ij = lib.einsum('ij,j->ij', idn_occ ,e_occ)

    # Second-order terms

    M_ij +=  lib.einsum('d,ilde,jlde->ij',e_vir,t2_1_a, t2_1_a, optimize=True)
    M_ij +=  lib.einsum('d,ilde,jlde->ij',e_vir,t2_1, t2_1, optimize=True)
    M_ij +=  lib.einsum('d,iled,jled->ij',e_vir,t2_1, t2_1, optimize=True)

    M_ij -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ,t2_1_a, t2_1_a, optimize=True)
    M_ij -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ,t2_1, t2_1, optimize=True)

    M_ij -= 0.25 *  lib.einsum('i,ilde,jlde->ij',e_occ,t2_1_a, t2_1_a, optimize=True)
    M_ij -= 0.25 *  lib.einsum('i,ilde,jlde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij -= 0.25 *  lib.einsum('i,ilde,jlde->ij',e_occ,t2_1, t2_1, optimize=True)

    M_ij -= 0.25 *  lib.einsum('j,ilde,jlde->ij',e_occ,t2_1_a, t2_1_a, optimize=True)
    M_ij -= 0.25 *  lib.einsum('j,ilde,jlde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij -= 0.25 *  lib.einsum('j,ilde,jlde->ij',e_occ,t2_1, t2_1, optimize=True)

    M_ij += 0.5 *  lib.einsum('ilde,jdle->ij',t2_1_a, eris_ovov,optimize=True)
    M_ij -= 0.5 *  lib.einsum('ilde,jeld->ij',t2_1_a, eris_ovov,optimize=True)
    M_ij += lib.einsum('ilde,jdle->ij',t2_1, eris_ovov,optimize=True)

    M_ij += 0.5 *  lib.einsum('jlde,idle->ij',t2_1_a, eris_ovov,optimize=True)
    M_ij -= 0.5 *  lib.einsum('jlde,ldie->ij',t2_1_a, eris_ovov,optimize=True)
    M_ij += lib.einsum('jlde,idle->ij',t2_1, eris_ovov,optimize=True)

    # Third-order terms

    if (method == "adc(3)"):

        t2_2 = t2[1]
        t2_2_a = t2_2 - t2_2.transpose(1,0,2,3).copy()

        eris_oovv = eris.oovv
        eris_ovvo = eris.ovvo
        eris_ovoo = eris.ovoo
        eris_oooo = eris.oooo

        M_ij += lib.einsum('ld,ldji->ij',t1_2, eris_ovoo,optimize=True)
        M_ij -= lib.einsum('ld,jdli->ij',t1_2, eris_ovoo,optimize=True)
        M_ij += lib.einsum('ld,ldji->ij',t1_2, eris_ovoo,optimize=True)

        M_ij += lib.einsum('ld,ldij->ij',t1_2, eris_ovoo,optimize=True)
        M_ij -= lib.einsum('ld,idlj->ij',t1_2, eris_ovoo,optimize=True)
        M_ij += lib.einsum('ld,ldij->ij',t1_2, eris_ovoo,optimize=True)

        M_ij += 0.5* lib.einsum('ilde,jdle->ij',t2_2_a, eris_ovov,optimize=True)
        M_ij -= 0.5* lib.einsum('ilde,jeld->ij',t2_2_a, eris_ovov,optimize=True)
        M_ij += lib.einsum('ilde,jdle->ij',t2_2, eris_ovov,optimize=True)

        M_ij += 0.5* lib.einsum('jlde,leid->ij',t2_2_a, eris_ovov,optimize=True)
        M_ij -= 0.5* lib.einsum('jlde,ield->ij',t2_2_a, eris_ovov,optimize=True)
        M_ij += lib.einsum('jlde,leid->ij',t2_2, eris_ovov,optimize=True)

        M_ij +=  lib.einsum('d,ilde,jlde->ij',e_vir,t2_1_a, t2_2_a,optimize=True)
        M_ij +=  lib.einsum('d,ilde,jlde->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij +=  lib.einsum('d,iled,jled->ij',e_vir,t2_1, t2_2,optimize=True)

        M_ij +=  lib.einsum('d,jlde,ilde->ij',e_vir,t2_1_a, t2_2_a,optimize=True)
        M_ij +=  lib.einsum('d,jlde,ilde->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij +=  lib.einsum('d,jled,iled->ij',e_vir,t2_1, t2_2,optimize=True)

        M_ij -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ,t2_1_a, t2_2_a,optimize=True)
        M_ij -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ,t2_1, t2_2,optimize=True)

        M_ij -= 0.5 *  lib.einsum('l,jlde,ilde->ij',e_occ,t2_1_a, t2_2_a,optimize=True)
        M_ij -= 0.5*lib.einsum('l,jlde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.5*lib.einsum('l,jlde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)

        M_ij -= 0.25 *  lib.einsum('i,ilde,jlde->ij',e_occ,t2_1_a, t2_2_a,optimize=True)
        M_ij -= 0.25 *  lib.einsum('i,ilde,jlde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.25 *  lib.einsum('i,ilde,jlde->ij',e_occ,t2_1, t2_2,optimize=True)

        M_ij -= 0.25 *  lib.einsum('i,jlde,ilde->ij',e_occ,t2_1_a, t2_2_a,optimize=True)
        M_ij -= 0.25 *  lib.einsum('i,jlde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.25 *  lib.einsum('i,jlde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)

        M_ij -= 0.25 *  lib.einsum('j,jlde,ilde->ij',e_occ,t2_1_a, t2_2_a,optimize=True)
        M_ij -= 0.25 *  lib.einsum('j,jlde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.25 *  lib.einsum('j,jlde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)

        M_ij -= 0.25 *  lib.einsum('j,ilde,jlde->ij',e_occ,t2_1_a, t2_2_a,optimize=True)
        M_ij -= 0.25 *  lib.einsum('j,ilde,jlde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.25 *  lib.einsum('j,ilde,jlde->ij',e_occ,t2_1, t2_2,optimize=True)

        M_ij -= lib.einsum('lmde,jldf,mefi->ij',t2_1_a, t2_1_a, eris_ovvo,optimize = True)
        M_ij += lib.einsum('lmde,jldf,mife->ij',t2_1_a, t2_1_a, eris_oovv,optimize = True)
        M_ij += lib.einsum('mled,jlfd,mefi->ij',t2_1, t2_1, eris_ovvo ,optimize = True)
        M_ij -= lib.einsum('mled,jlfd,mife->ij',t2_1, t2_1, eris_oovv ,optimize = True)
        M_ij -= lib.einsum('lmde,jldf,mefi->ij',t2_1, t2_1_a, eris_ovvo,optimize = True)
        M_ij -= lib.einsum('mlde,jldf,mife->ij',t2_1, t2_1, eris_oovv ,optimize = True)
        M_ij += lib.einsum('lmde,jlfd,mefi->ij',t2_1_a, t2_1, eris_ovvo ,optimize = True)

        M_ij -= lib.einsum('lmde,ildf,mefj->ij',t2_1_a, t2_1_a, eris_ovvo ,optimize = True)
        M_ij += lib.einsum('lmde,ildf,mjfe->ij',t2_1_a, t2_1_a, eris_oovv ,optimize = True)
        M_ij += lib.einsum('mled,ilfd,mefj->ij',t2_1, t2_1, eris_ovvo ,optimize = True)
        M_ij -= lib.einsum('mled,ilfd,mjfe->ij',t2_1, t2_1, eris_oovv ,optimize = True)
        M_ij -= lib.einsum('lmde,ildf,mefj->ij',t2_1, t2_1_a, eris_ovvo,optimize = True)
        M_ij -= lib.einsum('mlde,ildf,mjfe->ij',t2_1, t2_1, eris_oovv ,optimize = True)
        M_ij += lib.einsum('lmde,ilfd,mefj->ij',t2_1_a, t2_1, eris_ovvo ,optimize = True)

        M_ij += 0.25*lib.einsum('lmde,jnde,limn->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)
        M_ij -= 0.25*lib.einsum('lmde,jnde,lnmi->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)
        M_ij += lib.einsum('lmde,jnde,limn->ij',t2_1 ,t2_1, eris_oooo, optimize = True)

        if isinstance(eris.vvvv, np.ndarray):
            eris_vvvv = eris.vvvv
            t2_1_a_r = t2_1_a.reshape(nocc*nocc,nvir*nvir)
            t2_1_r = t2_1.reshape(nocc*nocc,nvir*nvir)
            temp_t2a_vvvv = np.dot(t2_1_a_r,eris_vvvv)
            temp_t2_vvvv = np.dot(t2_1_r,eris_vvvv)
            temp_t2a_vvvv = temp_t2a_vvvv.reshape(nocc,nocc,nvir,nvir)
            temp_t2_vvvv = temp_t2_vvvv.reshape(nocc,nocc,nvir,nvir)
        elif isinstance(eris.vvvv, list):
            temp_t2a_vvvv = contract_ladder(adc,t2_1_a,eris.vvvv)
            temp_t2_vvvv = contract_ladder(adc,t2_1,eris.vvvv)
        else:
            temp_t2a_vvvv = contract_ladder(adc,t2_1_a,eris.Lvv)
            temp_t2_vvvv = contract_ladder(adc,t2_1,eris.Lvv)

        M_ij += 0.25*lib.einsum('ilde,jlde->ij',t2_1_a, temp_t2a_vvvv, optimize = True)
        M_ij -= 0.25*lib.einsum('ilde,jled->ij',t2_1_a, temp_t2a_vvvv, optimize = True)
        M_ij +=lib.einsum('ilde,jlde->ij',t2_1, temp_t2_vvvv, optimize = True)

        M_ij += 0.25*lib.einsum('inde,lmde,jlnm->ij',t2_1_a, t2_1_a, eris_oooo, optimize = True)
        M_ij -= 0.25*lib.einsum('inde,lmde,jmnl->ij',t2_1_a, t2_1_a, eris_oooo, optimize = True)
        M_ij +=lib.einsum('inde,lmde,jlnm->ij',t2_1, t2_1, eris_oooo, optimize = True)

        M_ij += 0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_a, t2_1_a, eris_oovv, optimize = True)
        M_ij -= 0.5*lib.einsum('lmdf,lmde,jfei->ij',t2_1_a, t2_1_a, eris_ovvo, optimize = True)
        M_ij +=lib.einsum('mlfd,mled,jief->ij',t2_1, t2_1, eris_oovv , optimize = True)
        M_ij -=lib.einsum('mlfd,mled,jfei->ij',t2_1, t2_1, eris_ovvo , optimize = True)
        M_ij +=lib.einsum('lmdf,lmde,jief->ij',t2_1, t2_1, eris_oovv , optimize = True)
        M_ij +=0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_a, t2_1_a, eris_oovv , optimize = True)

        M_ij -= lib.einsum('ilde,jmdf,lmfe->ij',t2_1_a, t2_1_a, eris_oovv, optimize = True)
        M_ij += lib.einsum('ilde,jmdf,lefm->ij',t2_1_a, t2_1_a, eris_ovvo, optimize = True)
        M_ij += lib.einsum('ilde,jmdf,lefm->ij',t2_1_a, t2_1, eris_ovvo, optimize = True)
        M_ij += lib.einsum('ilde,jmdf,lefm->ij',t2_1, t2_1_a, eris_ovvo, optimize = True)
        M_ij -= lib.einsum('ilde,jmdf,lmfe->ij',t2_1, t2_1, eris_oovv, optimize = True)
        M_ij += lib.einsum('ilde,jmdf,lefm->ij',t2_1, t2_1, eris_ovvo, optimize = True)
        M_ij -= lib.einsum('iled,jmfd,lmfe->ij',t2_1, t2_1, eris_oovv, optimize = True)

        M_ij -= 0.5*lib.einsum('lnde,lmde,jinm->ij',t2_1_a, t2_1_a, eris_oooo, optimize = True)
        M_ij += 0.5*lib.einsum('lnde,lmde,jmni->ij',t2_1_a, t2_1_a, eris_oooo, optimize = True)
        M_ij -= lib.einsum('nled,mled,jinm->ij',t2_1, t2_1, eris_oooo, optimize = True)
        M_ij += lib.einsum('nled,mled,jmni->ij',t2_1, t2_1, eris_oooo, optimize = True)
        M_ij -= lib.einsum('lnde,lmde,jinm->ij',t2_1, t2_1, eris_oooo, optimize = True)
        M_ij -= 0.5 * lib.einsum('lnde,lmde,jinm->ij',t2_1_a, t2_1_a, eris_oooo, optimize = True)

    return M_ij


def ea_adc_diag(adc,M_ab=None,eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    if M_ab is None:
        M_ab = adc.get_imds()

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nvir
    n_doubles = nocc * nvir * nvir

    dim = n_singles + n_doubles

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    d_ab = e_vir[:,None] + e_vir
    d_i = e_occ[:,None]
    D_n = -d_i + d_ab.reshape(-1)
    D_iab = D_n.reshape(-1)

    diag = np.zeros(dim)

    # Compute precond in p1-p1 block

    M_ab_diag = np.diagonal(M_ab)

    diag[s1:f1] = M_ab_diag.copy()

    # Compute precond in 2p1h-2p1h block

    diag[s2:f2] = D_iab

    ###### Additional terms for the preconditioner ####

    if (method == "adc(2)-x" or method == "adc(3)"):

        if eris is None:
            eris = adc.transform_integrals()

            if isinstance(eris.vvvv, np.ndarray):

                eris_oovv = eris.oovv
                eris_vvvv = eris.vvvv

                temp = np.zeros((nocc, eris_vvvv.shape[0]))
                temp[:] += np.diag(eris_vvvv)
                diag[s2:f2] += temp.reshape(-1)

                eris_ovov_p = np.ascontiguousarray(eris_oovv[:].transpose(0,2,1,3))
                eris_ovov_p = eris_ovov_p.reshape(nocc*nvir, nocc*nvir)

                temp = np.zeros((nvir, nocc, nvir))
                temp[:] += np.diagonal(eris_ovov_p).reshape(nocc, nvir)
                temp = np.ascontiguousarray(temp.transpose(1,0,2))
                diag[s2:f2] += -temp.reshape(-1)

                eris_ovov_p = np.ascontiguousarray(eris_oovv[:].transpose(0,2,1,3))
                eris_ovov_p = eris_ovov_p.reshape(nocc*nvir, nocc*nvir)

                temp = np.zeros((nvir, nocc, nvir))
                temp[:] += np.diagonal(eris_ovov_p).reshape(nocc, nvir)
                temp = np.ascontiguousarray(temp.transpose(1,2,0))
                diag[s2:f2] += -temp.reshape(-1)

    return diag


def ip_adc_diag(adc,M_ij=None,eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    if M_ij is None:
        M_ij = adc.get_imds()

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc
    n_doubles = nvir * nocc * nocc

    dim = n_singles + n_doubles

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    d_ij = e_occ[:,None] + e_occ
    d_a = e_vir[:,None]
    D_n = -d_a + d_ij.reshape(-1)
    D_aij = D_n.reshape(-1)

    diag = np.zeros(dim)

    # Compute precond in h1-h1 block
    M_ij_diag = np.diagonal(M_ij)

    diag[s1:f1] = M_ij_diag.copy()

    # Compute precond in 2p1h-2p1h block

    diag[s2:f2] = D_aij.copy()

    ###### Additional terms for the preconditioner ####
    if (method == "adc(2)-x" or method == "adc(3)"):

        if eris is None:
            eris = adc.transform_integrals()

            if isinstance(eris.vvvv, np.ndarray):

                eris_oooo = eris.oooo
                eris_oovv = eris.oovv

                eris_oooo_p = np.ascontiguousarray(eris_oooo.transpose(0,2,1,3))
                eris_oooo_p = eris_oooo_p.reshape(nocc*nocc, nocc*nocc)

                temp = np.zeros((nvir, eris_oooo_p.shape[0]))
                temp[:] += np.diag(eris_oooo_p)
                diag[s2:f2] += -temp.reshape(-1)

                eris_ovov_p = np.ascontiguousarray(eris_oovv.transpose(0,2,1,3))
                eris_ovov_p = eris_ovov_p.reshape(nocc*nvir, nocc*nvir)

                temp = np.zeros((nocc, nocc, nvir))
                temp[:] += np.diagonal(eris_ovov_p).reshape(nocc, nvir)
                temp = np.ascontiguousarray(temp.transpose(2,1,0))
                diag[s2:f2] += temp.reshape(-1)

                eris_ovov_p = np.ascontiguousarray(eris_oovv.transpose(0,2,1,3))
                eris_ovov_p = eris_ovov_p.reshape(nocc*nvir, nocc*nvir)

                temp = np.zeros((nocc, nocc, nvir))
                temp[:] += np.diagonal(eris_ovov_p).reshape(nocc, nvir)
                temp = np.ascontiguousarray(temp.transpose(2,0,1))
                diag[s2:f2] += temp.reshape(-1)

    diag = -diag

    return diag

def ea_contract_r_vvvv(myadc,r2,vvvv):

    nocc = myadc._nocc
    nvir = myadc._nvir

    r2_vvvv = np.zeros((nocc,nvir,nvir))
    r2 = np.ascontiguousarray(r2.reshape(nocc,-1))
    chnk_size = radc_ao2mo.calculate_chunk_size(myadc)

    a = 0
    if isinstance(vvvv, list):
        for dataset in vvvv:
            k = dataset.shape[0]
            dataset = dataset[:].reshape(-1,nvir*nvir)
            r2_vvvv[:,a:a+k] = np.dot(r2,dataset.T).reshape(nocc,-1,nvir)
            del (dataset)
            a += k
    elif getattr(myadc, 'with_df', None):
        for p in range(0,nvir,chnk_size):
            vvvv_p = dfadc.get_vvvv_df(myadc, vvvv, p, chnk_size)
            k = vvvv_p.shape[0]
            vvvv_p = vvvv_p.reshape(-1,nvir*nvir)
            r2_vvvv[:,a:a+k] = np.dot(r2,vvvv_p.T).reshape(nocc,-1,nvir)
            del (vvvv_p)
            a += k
    else:
        raise Exception("Unknown vvvv type")

    r2_vvvv = r2_vvvv.reshape(-1)

    return r2_vvvv


def ea_adc_matvec(adc, M_ab=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2_1 = adc.t2[0]

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nvir
    n_doubles = nocc * nvir * nvir

    dim = n_singles + n_doubles

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    if eris is None:
        eris = adc.transform_integrals()

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    d_ab = e_vir[:,None] + e_vir
    d_i = e_occ[:,None]
    D_n = -d_i + d_ab.reshape(-1)
    D_iab = D_n.reshape(-1)

    if M_ab is None:
        M_ab = adc.get_imds()

    #Calculate sigma vector
    def sigma_(r):

        s = np.zeros((dim))

        r1 = r[s1:f1]
        r2 = r[s2:f2]

        r2 = r2.reshape(nocc,nvir,nvir)

############ ADC(2) ab block ############################

        s[s1:f1] = lib.einsum('ab,b->a',M_ab,r1)

############# ADC(2) a - ibc and ibc - a coupling blocks #########################

        eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)

        s[s1:f1] +=  2. * lib.einsum('icab,ibc->a', eris_ovvv, r2, optimize = True)
        s[s1:f1] -=  lib.einsum('ibac,ibc->a',   eris_ovvv, r2, optimize = True)

        temp = lib.einsum('icab,a->ibc', eris_ovvv, r1, optimize = True)
        s[s2:f2] +=  temp.reshape(-1)
        del eris_ovvv

################ ADC(2) iab - jcd block ############################

        s[s2:f2] +=  D_iab * r2.reshape(-1)

############### ADC(3) iab - jcd block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

            r2_a = r2 - r2.transpose(0,2,1).copy()

            eris_oovv = eris.oovv
            eris_ovvo = eris.ovvo

            r2 = r2.reshape(nocc, nvir, nvir)

            if isinstance(eris.vvvv, np.ndarray):
                r_bab_t = r2.reshape(nocc,-1)
                eris_vvvv = eris.vvvv
                s[s2:f2] += np.dot(r_bab_t,eris_vvvv.T).reshape(-1)
            elif isinstance(eris.vvvv, list):
                s[s2:f2] += ea_contract_r_vvvv(adc,r2,eris.vvvv)
            else:
                s[s2:f2] += ea_contract_r_vvvv(adc,r2,eris.Lvv)

            s[s2:f2] -= 0.5*lib.einsum('jzyi,jzx->ixy',eris_ovvo,r2_a,optimize = True).reshape(-1)
            s[s2:f2] -= 0.5*lib.einsum('jiyz,jxz->ixy',eris_oovv,r2,optimize = True).reshape(-1)
            s[s2:f2] += 0.5*lib.einsum('jzyi,jxz->ixy',eris_ovvo,r2,optimize = True).reshape(-1)
            s[s2:f2] -=  0.5*lib.einsum('jixz,jzy->ixy',eris_oovv,r2,optimize = True).reshape(-1)
            s[s2:f2] -=  0.5*lib.einsum('jixw,jwy->ixy',eris_oovv,r2,optimize = True).reshape(-1)
            s[s2:f2] -= 0.5*lib.einsum('jiyw,jxw->ixy',eris_oovv,r2,optimize = True).reshape(-1)
            s[s2:f2] += 0.5*lib.einsum('jwyi,jxw->ixy',eris_ovvo,r2,optimize = True).reshape(-1)
            s[s2:f2] += 0.5*lib.einsum('jwyi,jxw->ixy',eris_ovvo,r2_a,optimize = True).reshape(-1)

            #print("Calculating additional terms for adc(3)")
        if (method == "adc(3)"):

            eris_ovoo = eris.ovoo

############### ADC(3) a - ibc block and ibc-a coupling blocks ########################

            t2_1_a = t2_1 - t2_1.transpose(1,0,2,3).copy()
            t2_1_a_t = t2_1_a.reshape(nocc,nocc,-1)
            r2_a = r2_a.reshape(nocc,-1)
            temp =  0.25 * lib.einsum('lmp,jp->lmj',t2_1_a_t,r2_a)
            s[s1:f1] += lib.einsum('lmj,lamj->a',temp, eris_ovoo, optimize=True)
            s[s1:f1] -= lib.einsum('lmj,malj->a',temp, eris_ovoo, optimize=True)

            temp_1 = -lib.einsum('lmzw,jzw->jlm',t2_1,r2)
            s[s1:f1] -= lib.einsum('jlm,lamj->a',temp_1, eris_ovoo, optimize=True)

            r2_a = r2_a.reshape(nocc,nvir,nvir)
            temp_s_a = np.zeros_like(r2)
            temp_s_a = lib.einsum('jlwd,jzw->lzd',t2_1_a,r2_a,optimize=True)
            temp_s_a += lib.einsum('ljdw,jzw->lzd',t2_1,r2,optimize=True)

            temp_s_a_1 = np.zeros_like(r2)
            temp_s_a_1 = -lib.einsum('jlzd,jwz->lwd',t2_1_a,r2_a,optimize=True)
            temp_s_a_1 += -lib.einsum('ljdz,jwz->lwd',t2_1,r2,optimize=True)

            eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)

            temp_1_1 = lib.einsum('ldxb,b->lxd', eris_ovvv,r1,optimize=True)
            temp_1_1 -= lib.einsum('lbxd,b->lxd', eris_ovvv,r1,optimize=True)
            temp_2_1 = lib.einsum('ldxb,b->lxd', eris_ovvv,r1,optimize=True)

            s[s1:f1] += 0.5*lib.einsum('lzd,ldza->a',temp_s_a,eris_ovvv,optimize=True)
            s[s1:f1] -= 0.5*lib.einsum('lzd,lazd->a',temp_s_a,eris_ovvv,optimize=True)
            s[s1:f1] -= 0.5*lib.einsum('lwd,ldwa->a',temp_s_a_1,eris_ovvv,optimize=True)
            s[s1:f1] += 0.5*lib.einsum('lwd,lawd->a',temp_s_a_1,eris_ovvv,optimize=True)

            temp_1 = np.zeros_like(r2)
            temp_1 = lib.einsum('jlwd,jzw->lzd',t2_1,r2_a,optimize=True)
            temp_1 += lib.einsum('jlwd,jzw->lzd',t2_1_a,r2,optimize=True)
            s[s1:f1] += 0.5*lib.einsum('lzd,ldza->a',temp_1,eris_ovvv,optimize=True)

            temp_1 = np.zeros_like(r2)
            temp_1 = -lib.einsum('jlzd,jwz->lwd',t2_1,r2_a,optimize=True)
            temp_1 += -lib.einsum('jlzd,jwz->lwd',t2_1_a,r2,optimize=True)
            s[s1:f1] -= 0.5*lib.einsum('lwd,ldwa->a',temp_1,eris_ovvv,optimize=True)

            temp_2 = -lib.einsum('ljzd,jzw->lwd',t2_1,r2,optimize=True)
            s[s1:f1] += 0.5*lib.einsum('lwd,lawd->a',temp_2,eris_ovvv,optimize=True)

            temp_a = t2_1.transpose(0,3,1,2).copy()
            temp_b = temp_a.reshape(nocc*nvir,nocc*nvir)
            r2_t = r2.reshape(nocc*nvir,-1)
            temp_c = np.dot(temp_b,r2_t).reshape(nocc,nvir,nvir)
            temp_2 = temp_c.transpose(0,2,1).copy()
            s[s1:f1] -= 0.5*lib.einsum('lzd,lazd->a',temp_2,eris_ovvv,optimize=True)

            temp  = -lib.einsum('lbyd,b->lyd',eris_ovvv,r1,optimize=True)
            temp_1= -lib.einsum('lyd,lixd->ixy',temp,t2_1,optimize=True)
            s[s2:f2] -= temp_1.reshape(-1)
            del eris_ovvv

            temp_1 = lib.einsum('b,lbmi->lmi',r1,eris_ovoo)
            s[s2:f2] += lib.einsum('lmi,lmxy->ixy',temp_1, t2_1, optimize=True).reshape(-1)

            temp  = lib.einsum('lxd,lidy->ixy',temp_1_1,t2_1,optimize=True)
            temp  += lib.einsum('lxd,ilyd->ixy',temp_2_1,t2_1_a,optimize=True)
            s[s2:f2] += temp.reshape(-1)

        return s

    return sigma_


def ip_adc_matvec(adc, M_ij=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2_1 = adc.t2[0]

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc
    n_doubles = nvir * nocc * nocc

    dim = n_singles + n_doubles

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    if eris is None:
        eris = adc.transform_integrals()

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    d_ij = e_occ[:,None] + e_occ
    d_a = e_vir[:,None]
    D_n = -d_a + d_ij.reshape(-1)
    D_aij = D_n.reshape(-1)

    if M_ij is None:
        M_ij = adc.get_imds()

    #Calculate sigma vector
    def sigma_(r):

        s = np.zeros((dim))

        r1 = r[s1:f1]
        r2 = r[s2:f2]

        r2 = r2.reshape(nvir,nocc,nocc)

        eris_ovoo = eris.ovoo

############ ADC(2) ij block ############################

        s[s1:f1] = lib.einsum('ij,j->i',M_ij,r1)

############ ADC(2) i - kja block #########################

        s[s1:f1] += 2. * lib.einsum('jaki,ajk->i', eris_ovoo, r2, optimize = True)
        s[s1:f1] -= lib.einsum('kaji,ajk->i', eris_ovoo, r2, optimize = True)

############## ADC(2) ajk - i block ############################

        temp = lib.einsum('jaki,i->ajk', eris_ovoo, r1, optimize = True).reshape(-1)
        s[s2:f2] += temp.reshape(-1)

################ ADC(2) ajk - bil block ############################

        s[s2:f2] += D_aij * r2.reshape(-1)

############### ADC(3) ajk - bil block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

            r2_a = r2 - r2.transpose(0,2,1).copy()

            eris_oooo = eris.oooo
            eris_oovv = eris.oovv
            eris_ovvo = eris.ovvo

            s[s2:f2] -= 0.5*lib.einsum('kijl,ali->ajk',eris_oooo, r2, optimize = True).reshape(-1)
            s[s2:f2] -= 0.5*lib.einsum('klji,ail->ajk',eris_oooo ,r2, optimize = True).reshape(-1)

            s[s2:f2] += 0.5*lib.einsum('klba,bjl->ajk',eris_oovv,r2,optimize = True).reshape(-1)

            s[s2:f2] +=  0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo,r2_a,optimize = True).reshape(-1)
            s[s2:f2] +=  0.5*lib.einsum('jlba,blk->ajk',eris_oovv,r2,optimize = True).reshape(-1)
            s[s2:f2] -=  0.5*lib.einsum('jabl,blk->ajk',eris_ovvo,r2,optimize = True).reshape(-1)

            s[s2:f2] += 0.5*lib.einsum('kiba,bji->ajk',eris_oovv,r2,optimize = True).reshape(-1)

            s[s2:f2] += 0.5*lib.einsum('jiba,bik->ajk',eris_oovv,r2,optimize = True).reshape(-1)
            s[s2:f2] -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
            s[s2:f2] -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo,r2_a,optimize = True).reshape(-1)

        if (method == "adc(3)"):

            eris_ovoo = eris.ovoo

################ ADC(3) i - kja block and ajk - i ############################

            eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
            t2_1_a = t2_1 - t2_1.transpose(1,0,2,3).copy()
            r2_a = r2_a.reshape(nvir,-1)
            t2_1_a_t = t2_1_a.reshape(-1,nvir,nvir)
            temp = 0.25 * lib.einsum('pbc,ap->abc',t2_1_a_t,r2_a, optimize=True)
            s[s1:f1] += lib.einsum('abc,icab->i',temp, eris_ovvv, optimize=True)
            s[s1:f1] -= lib.einsum('abc,ibac->i',temp, eris_ovvv, optimize=True)
            temp_1 = lib.einsum('kjcb,ajk->abc',t2_1,r2, optimize=True)
            s[s1:f1] += lib.einsum('abc,icab->i',temp_1, eris_ovvv, optimize=True)

            temp_1 = lib.einsum('i,icab->cba',r1,eris_ovvv,optimize=True)
            s[s2:f2] += lib.einsum('cba,kjcb->ajk',temp_1, t2_1, optimize=True).reshape(-1)
            del eris_ovvv

            r2_a = r2_a.reshape(nvir,nocc,nocc)

            temp = np.zeros_like(r2)
            temp = lib.einsum('jlab,ajk->blk',t2_1_a,r2_a,optimize=True)
            temp += lib.einsum('ljba,ajk->blk',t2_1,r2,optimize=True)

            temp_1 = np.zeros_like(r2)
            temp_1 = lib.einsum('jlab,ajk->blk',t2_1,r2_a,optimize=True)
            temp_1 += lib.einsum('jlab,ajk->blk',t2_1_a,r2,optimize=True)

            temp_2 = lib.einsum('jlba,akj->blk',t2_1,r2, optimize=True)

            s[s1:f1] += 0.5*lib.einsum('blk,lbik->i',temp,eris_ovoo,optimize=True)
            s[s1:f1] -= 0.5*lib.einsum('blk,iblk->i',temp,eris_ovoo,optimize=True)
            s[s1:f1] += 0.5*lib.einsum('blk,lbik->i',temp_1,eris_ovoo,optimize=True)
            s[s1:f1] -= 0.5*lib.einsum('blk,iblk->i',temp_2,eris_ovoo,optimize=True)

            temp = np.zeros_like(r2)
            temp = -lib.einsum('klab,akj->blj',t2_1_a,r2_a,optimize=True)
            temp -= lib.einsum('lkba,akj->blj',t2_1,r2,optimize=True)

            temp_1 = np.zeros_like(r2)
            temp_1 = -lib.einsum('klab,akj->blj',t2_1,r2_a,optimize=True)
            temp_1 -= lib.einsum('klab,akj->blj',t2_1_a,r2,optimize=True)

            temp_2 = -lib.einsum('klba,ajk->blj',t2_1,r2,optimize=True)

            s[s1:f1] -= 0.5*lib.einsum('blj,lbij->i',temp,eris_ovoo,optimize=True)
            s[s1:f1] += 0.5*lib.einsum('blj,iblj->i',temp,eris_ovoo,optimize=True)
            s[s1:f1] -= 0.5*lib.einsum('blj,lbij->i',temp_1,eris_ovoo,optimize=True)
            s[s1:f1] += 0.5*lib.einsum('blj,iblj->i',temp_2,eris_ovoo,optimize=True)

            temp_1  = lib.einsum('i,lbik->kbl',r1,eris_ovoo)
            temp_1  -= lib.einsum('i,iblk->kbl',r1,eris_ovoo)
            temp_2  = lib.einsum('i,lbik->kbl',r1,eris_ovoo)

            temp  = lib.einsum('kbl,ljba->ajk',temp_1,t2_1,optimize=True)
            temp += lib.einsum('kbl,jlab->ajk',temp_2,t2_1_a,optimize=True)
            s[s2:f2] += temp.reshape(-1)

            temp  = -lib.einsum('i,iblj->jbl',r1,eris_ovoo,optimize=True)
            temp_1 = -lib.einsum('jbl,klba->ajk',temp,t2_1,optimize=True)
            s[s2:f2] -= temp_1.reshape(-1)

        s *= -1.0

        return s

    return sigma_


def ea_compute_trans_moments(adc, orb):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2_1 = adc.t2[0]
    t1_2 = adc.t1[0]
    t2_1_a = t2_1 - t2_1.transpose(1,0,2,3).copy()

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nvir
    n_doubles = nocc * nvir * nvir

    dim = n_singles + n_doubles

    idn_vir = np.identity(nvir)

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    T = np.zeros((dim))

######## ADC(2) part  ############################################

    if orb < nocc:

        T[s1:f1] = -t1_2[orb,:]

        t2_1_t = -t2_1.transpose(1,0,2,3).copy()

        T[s2:f2] += t2_1_t[:,orb,:,:].reshape(-1)

    else:

        T[s1:f1] += idn_vir[(orb-nocc), :]
        T[s1:f1] -= 0.25*lib.einsum('klc,klac->a',t2_1_a[:,:,(orb-nocc),:], t2_1_a, optimize = True)
        T[s1:f1] -= 0.25*lib.einsum('klc,klac->a',t2_1[:,:,(orb-nocc),:], t2_1, optimize = True)
        T[s1:f1] -= 0.25*lib.einsum('lkc,lkac->a',t2_1[:,:,(orb-nocc),:], t2_1, optimize = True)

######## ADC(3) 2p-1h  part  ############################################

    if(method=="adc(2)-x"or adc.method=="adc(3)"):

        t2_2 = adc.t2[1]
        t2_2_a = t2_2 - t2_2.transpose(1,0,2,3).copy()

        if orb < nocc:

            t2_2_t = -t2_2.transpose(1,0,2,3).copy()

            T[s2:f2] += t2_2_t[:,orb,:,:].reshape(-1)

########## ADC(3) 1p part  ############################################

    if(adc.method=="adc(3)"):

        t1_3 = adc.t1[1]

        if orb < nocc:

            T[s1:f1] += 0.5*lib.einsum('kac,ck->a',t2_1_a[:,orb,:,:], t1_2.T,optimize = True)
            T[s1:f1] -= 0.5*lib.einsum('kac,ck->a',t2_1[orb,:,:,:], t1_2.T,optimize = True)

            T[s1:f1] -= t1_3[orb,:]

        else:

            T[s1:f1] -= 0.25*lib.einsum('klc,klac->a',t2_1_a[:,:,(orb-nocc),:], t2_2_a, optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('klc,klac->a',t2_1[:,:,(orb-nocc),:], t2_2, optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('lkc,lkac->a',t2_1[:,:,(orb-nocc),:], t2_2, optimize = True)

            T[s1:f1] -= 0.25*lib.einsum('klac,klc->a',t2_1_a, t2_2_a[:,:,(orb-nocc),:],optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('klac,klc->a',t2_1, t2_2[:,:,(orb-nocc),:],optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('lkac,lkc->a',t2_1, t2_2[:,:,(orb-nocc),:],optimize = True)

    T_aaa = T[n_singles:].reshape(nocc,nvir,nvir).copy()
    T_aaa = T_aaa - T_aaa.transpose(0,2,1)
    T[n_singles:] += T_aaa.reshape(-1)

    return T


def ip_compute_trans_moments(adc, orb):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2_1 = adc.t2[0]
    t1_2 = adc.t1[0]
    t2_1_a = t2_1 - t2_1.transpose(1,0,2,3).copy()

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc
    n_doubles = nvir * nocc * nocc

    dim = n_singles + n_doubles

    idn_occ = np.identity(nocc)

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    T = np.zeros((dim))

######## ADC(2) 1h part  ############################################

    if orb < nocc:
        T[s1:f1]  = idn_occ[orb, :]
        T[s1:f1] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_a[:,orb,:,:], t2_1_a, optimize = True)
        T[s1:f1] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1[orb,:,:,:], t2_1, optimize = True)
        T[s1:f1] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1[orb,:,:,:], t2_1, optimize = True)
    else:
        T[s1:f1] += t1_2[:,(orb-nocc)]

######## ADC(2) 2h-1p  part  ############################################

        t2_1_t = t2_1.transpose(2,3,1,0).copy()

        T[s2:f2] = t2_1_t[(orb-nocc),:,:,:].reshape(-1)

######## ADC(3) 2h-1p  part  ############################################

    if(method=='adc(2)-x'or method=='adc(3)'):

        t2_2 = adc.t2[1]
        t2_2_a = t2_2 - t2_2.transpose(1,0,2,3).copy()

        if orb >= nocc:
            t2_2_t = t2_2.transpose(2,3,1,0).copy()

            T[s2:f2] += t2_2_t[(orb-nocc),:,:,:].reshape(-1)

######## ADC(3) 1h part  ############################################

    if(method=='adc(3)'):

        t1_3 = adc.t1[1]

        if orb < nocc:
            T[s1:f1] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_a[:,orb,:,:], t2_2_a, optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1[orb,:,:,:], t2_2, optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1[orb,:,:,:], t2_2, optimize = True)

            T[s1:f1] += 0.25*lib.einsum('ikdc,kdc->i',t2_1_a, t2_2_a[:,orb,:,:],optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('ikcd,kcd->i',t2_1, t2_2[orb,:,:,:],optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('ikdc,kdc->i',t2_1, t2_2[orb,:,:,:],optimize = True)
        else:
            T[s1:f1] += 0.5*lib.einsum('ikc,kc->i',t2_1_a[:,:,(orb-nocc),:], t1_2,optimize = True)
            T[s1:f1] += 0.5*lib.einsum('ikc,kc->i',t2_1[:,:,(orb-nocc),:], t1_2,optimize = True)
            T[s1:f1] += t1_3[:,(orb-nocc)]

    T_aaa = T[n_singles:].reshape(nvir,nocc,nocc).copy()
    T_aaa = T_aaa - T_aaa.transpose(0,2,1)
    T[n_singles:] += T_aaa.reshape(-1)

    return T


def get_trans_moments(adc):

    nmo  = adc.nmo

    T = []

    for orb in range(nmo):

        T_a = adc.compute_trans_moments(orb)
        T.append(T_a)

    T = np.array(T)
    return T


def get_spec_factors_ea(adc, T, U, nroots=1):

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nvir

    U = U.reshape(nroots,-1)

    for I in range(U.shape[0]):
        U1 = U[I, :n_singles]
        U2 = U[I, n_singles:].reshape(nocc,nvir,nvir)
        UdotU = np.dot(U1, U1) + 2.*np.dot(U2.ravel(), U2.ravel()) - np.dot(U2.ravel(), U2.transpose(0,2,1).ravel())
        U[I,:] /= np.sqrt(UdotU)

    X = np.dot(T, U.T).reshape(-1, nroots)

    P = 2.0*lib.einsum("pi,pi->i", X, X)

    return P

def get_spec_factors_ip(adc, T, U, nroots=1):

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc

    U = U.reshape(nroots,-1)

    for I in range(U.shape[0]):
        U1 = U[I, :n_singles]
        U2 = U[I, n_singles:].reshape(nvir,nocc,nocc)
        UdotU = np.dot(U1, U1) + 2.*np.dot(U2.ravel(), U2.ravel()) - np.dot(U2.ravel(), U2.transpose(0,2,1).ravel())
        U[I,:] /= np.sqrt(UdotU)

    X = np.dot(T, U.T).reshape(-1, nroots)

    P = 2.0*lib.einsum("pi,pi->i", X, X)

    return P


class RADCEA(RADC):
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
            Space size to hold trial vectors for Davidson iterative
            diagonalization.  Default is 12.

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested. Default value is 1.

            >>> myadc = adc.RADC(mf).run()
            >>> myadcea = adc.RADC(myadc).run()

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
        self.mol = adc.mol
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
        self.method_type = adc.method_type
        self._scf = adc._scf
        self._nocc = adc._nocc
        self._nvir = adc._nvir
        self._nmo = adc._nmo
        self.mo_coeff = adc.mo_coeff
        self.mo_energy = adc.mo_energy
        self.nmo = adc._nmo
        self.transform_integrals = adc.transform_integrals
        self.with_df = adc.with_df

        keys = set(('conv_tol', 'e_corr', 'method', 'mo_coeff', 'mo_energy',
                    'max_memory', 't1', 'max_space', 't2', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)

    kernel = kernel
    get_imds = get_imds_ea
    matvec = ea_adc_matvec
    get_diag = ea_adc_diag
    compute_trans_moments = ea_compute_trans_moments
    get_trans_moments = get_trans_moments
    get_spec_factors = get_spec_factors_ea

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
        diag = self.get_diag(imds, eris)
        matvec = self.matvec(imds, eris)
        return matvec, diag


class RADCIP(RADC):
    '''restricted ADC for IP energies and spectroscopic amplitudes

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

        e_ip : float or list of floats
            IP energy (eigenvalue). For nroots = 1, it is a single float
            number. If nroots > 1, it is a list of floats for the lowest
            nroots eigenvalues.
        v_ip : array
            Eigenvectors for each IP transition.
        p_ip : float
            Spectroscopic amplitudes for each IP transition.
    '''
    def __init__(self, adc):
        self.mol = adc.mol
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
        self.method_type = adc.method_type
        self._scf = adc._scf
        self._nocc = adc._nocc
        self._nvir = adc._nvir
        self._nmo = adc._nmo
        self.mo_coeff = adc.mo_coeff
        self.mo_energy = adc.mo_energy
        self.nmo = adc._nmo
        self.transform_integrals = adc.transform_integrals
        self.with_df = adc.with_df

        keys = set(('conv_tol', 'e_corr', 'method', 'mo_coeff', 'mo_energy_b',
                    'max_memory', 't1', 'mo_energy_a', 'max_space', 't2',
                    'max_cycle'))
        self._keys = set(self.__dict__.keys()).union(keys)

    kernel = kernel
    get_imds = get_imds_ip
    get_diag = ip_adc_diag
    matvec = ip_adc_matvec
    compute_trans_moments = ip_compute_trans_moments
    get_trans_moments = get_trans_moments
    get_spec_factors = get_spec_factors_ip

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
        diag = self.get_diag(imds, eris)
        matvec = self.matvec(imds, eris)
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
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    myadc = adc.ADC(mf)
    ecorr, t_amp1, t_amp2 = myadc.kernel_gs()
    print(ecorr -  -0.3220169236051954)

    myadcip = RADCIP(myadc)
    e,v,p = kernel(myadcip,nroots=3)
    print("ADC(2) IP energies")
    print (e[0] - 0.5434389910483670)
    print (e[1] - 0.6240296243595950)
    print (e[2] - 0.6240296243595956)

    print("ADC(2) IP spectroscopic factors")
    print (p[0] - 1.7688097076459075)
    print (p[1] - 1.8192921131700284)
    print (p[2] - 1.8192921131700293)

    myadcea = RADCEA(myadc)
    e,v,p = kernel(myadcea,nroots=3)
    print("ADC(2) EA energies")
    print (e[0] - 0.0961781923822576)
    print (e[1] - 0.1258326916409743)
    print (e[2] - 0.1380779405750178)

    print("ADC(2) EA spectroscopic factors")
    print (p[0] - 1.9832854445007961)
    print (p[1] - 1.9634368668786559)
    print (p[2] - 1.9783719593912672)

    myadc = adc.ADC(mf)
    myadc.method = "adc(3)"
    ecorr, t_amp1, t_amp2 = myadc.kernel_gs()
    print(ecorr - -0.31694173142858517)

    myadcip = RADCIP(myadc)
    e,v,p = kernel(myadcip,nroots=3)
    print("ADC(3) IP energies")
    print (e[0] - 0.5667526829981027)
    print (e[1] - 0.6099995170092525)
    print (e[2] - 0.6099995170092529)

    print("ADC(3) IP spectroscopic factors")
    print (p[0] - 1.8173191958988848)
    print (p[1] - 1.8429224413853840)
    print (p[2] - 1.8429224413853851)

    myadcea = RADCEA(myadc)
    e,v,p = kernel(myadcea,nroots=3)

    print("ADC(3) EA energies")
    print (e[0] - 0.0936790850738445)
    print (e[1] - 0.0983654552141278)
    print (e[2] - 0.1295709313652367)

    print("ADC(3) EA spectroscopic factors")
    print (p[0] - 1.8324175318668088)
    print (p[1] - 1.9840991060607487)
    print (p[2] - 1.9638550014980212)

    myadc.method = "adc(2)-x"
    e,v,p = myadc.kernel(nroots=4)
    print("ADC(2)-x IP energies")
    print (e[0] - 0.5405255360673724)
    print (e[1] - 0.6208026698756577)
    print (e[2] - 0.6208026698756582)
    print (e[3] - 0.6465332771967947)

    myadc.method_type = "ea"
    e,v,p = myadc.kernel(nroots=4)
    print("ADC(2)-x EA energies")
    print (e[0] - 0.0953065329985665)
    print (e[1] - 0.1238833070823509)
    print (e[2] - 0.1365693811939308)
    print (e[3] - 0.1365693811939316)
