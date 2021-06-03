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
import numpy as np
import pyscf.ao2mo as ao2mo
from pyscf import lib
from pyscf.lib import logger
from pyscf.adc import radc_ao2mo
from pyscf.adc import dfadc
from pyscf import __config__
from pyscf import df
from pyscf import symm

def kernel(adc, nroots=1, guess=None, eris=None, verbose=None):

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

    imds = adc.get_imds(eris)
    matvec, diag = adc.gen_matvec(imds, eris)

    guess = adc.get_init_guess(nroots, diag, ascending = True)

    conv, adc.E, U = lib.linalg_helper.davidson_nosym1(
        lambda xs : [matvec(x) for x in xs],
        guess, diag, nroots=nroots, verbose=log, tol=adc.conv_tol,
        max_cycle=adc.max_cycle, max_space=adc.max_space, tol_residual=adc.tol_residual)

    adc.U = np.array(U).T.copy()

    if adc.compute_properties:
        adc.P,adc.X = adc.get_properties(nroots)

    nfalse = np.shape(conv)[0] - np.sum(conv)

    header = ("\n*************************************************************"
              "\n            ADC calculation summary"
              "\n*************************************************************")
    logger.info(adc, header)

    if nfalse >= 1:
        logger.warn(adc, "Davidson iterations for " + str(nfalse) + " root(s) not converged\n")

    for n in range(nroots):
        print_string = ('%s root %d  |  Energy (Eh) = %14.10f  |  Energy (eV) = %12.8f  ' %
                        (adc.method, n, adc.E[n], adc.E[n]*27.2114))
        if adc.compute_properties:
            print_string += ("|  Spec factors = %10.8f  " % adc.P[n])
        print_string += ("|  conv = %s" % conv[n])
        logger.info(adc, print_string)

    log.timer('ADC', *cput0)

    return adc.E, adc.U, adc.P, adc.X


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
    if not isinstance(eris.oooo, np.ndarray):
        t2_1 = radc_ao2mo.write_dataset(t2_1)

    del v2e_oovv
    del D2

    cput0 = log.timer_debug1("Completed t2_1 amplitude calculation", *cput0)

    # Compute second-order singles t1 (tij)

    if isinstance(eris.ovvv, type(None)):
        chnk_size = radc_ao2mo.calculate_chunk_size(myadc)
    else:
        chnk_size = nocc
    a = 0
    t1_2 = np.zeros((nocc,nvir))

    for p in range(0,nocc,chnk_size):
        if getattr(myadc, 'with_df', None):
            eris_ovvv = dfadc.get_ovvv_df(myadc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
        else:
            eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
        k = eris_ovvv.shape[0]

        t1_2 += 0.5*lib.einsum('kdac,ikcd->ia',eris_ovvv,t2_1[:,a:a+k],optimize=True)
        t1_2 -= 0.5*lib.einsum('kdac,kicd->ia',eris_ovvv,t2_1[a:a+k,:],optimize=True)
        t1_2 -= 0.5*lib.einsum('kcad,ikcd->ia',eris_ovvv,t2_1[:,a:a+k],optimize=True)
        t1_2 += 0.5*lib.einsum('kcad,kicd->ia',eris_ovvv,t2_1[a:a+k,:],optimize=True)

        t1_2 += lib.einsum('kdac,ikcd->ia',eris_ovvv,t2_1[:,a:a+k],optimize=True)
        del eris_ovvv
        a += k

    t1_2 -= 0.5*lib.einsum('lcki,klac->ia',eris_ovoo,t2_1[:],optimize=True)
    t1_2 += 0.5*lib.einsum('lcki,lkac->ia',eris_ovoo,t2_1[:],optimize=True)
    t1_2 -= 0.5*lib.einsum('kcli,lkac->ia',eris_ovoo,t2_1[:],optimize=True)
    t1_2 += 0.5*lib.einsum('kcli,klac->ia',eris_ovoo,t2_1[:],optimize=True)
    t1_2 -= lib.einsum('lcki,klac->ia',eris_ovoo,t2_1[:],optimize=True)

    t1_2 = t1_2/D1

    cput0 = log.timer_debug1("Completed t1_2 amplitude calculation", *cput0)

    t2_2 = None
    t1_3 = None
    t2_1_vvvv = None

    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):

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
            t2_1_vvvv = radc_ao2mo.write_dataset(t2_1_vvvv)

        t2_2 = t2_1_vvvv[:].copy()

        t2_2 += lib.einsum('kilj,klab->ijab',eris_oooo,t2_1[:],optimize=True)
        t2_2 += lib.einsum('kcbj,kica->ijab',eris_ovvo,t2_1[:],optimize=True)
        t2_2 -= lib.einsum('kcbj,ikca->ijab',eris_ovvo,t2_1[:],optimize=True)
        t2_2 += lib.einsum('kcbj,ikac->ijab',eris_ovvo,t2_1[:],optimize=True)
        t2_2 -= lib.einsum('kjbc,ikac->ijab',eris_oovv,t2_1[:],optimize=True)
        t2_2 -= lib.einsum('kibc,kjac->ijab',eris_oovv,t2_1[:],optimize=True)
        t2_2 -= lib.einsum('kjac,ikcb->ijab',eris_oovv,t2_1[:],optimize=True)
        t2_2 += lib.einsum('kcai,kjcb->ijab',eris_ovvo,t2_1[:],optimize=True)
        t2_2 -= lib.einsum('kcai,jkcb->ijab',eris_ovvo,t2_1[:],optimize=True)
        t2_2 += lib.einsum('kcai,kjcb->ijab',eris_ovvo,t2_1[:],optimize=True)
        t2_2 -= lib.einsum('kiac,kjcb->ijab',eris_oovv,t2_1[:],optimize=True)

        D2 = d_ij.reshape(-1,1) - d_ab.reshape(-1)
        D2 = D2.reshape((nocc,nocc,nvir,nvir))

        t2_2 = t2_2/D2
        if not isinstance(eris.oooo, np.ndarray):
            t2_2 = radc_ao2mo.write_dataset(t2_2)
        del D2

    cput0 = log.timer_debug1("Completed t2_2 amplitude calculation", *cput0)

    if (myadc.method == "adc(3)"):

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
        else:
            chnk_size = nocc
        a = 0

        for p in range(0,nocc,chnk_size):
            if getattr(myadc, 'with_df', None):
                eris_ovvv = dfadc.get_ovvv_df(myadc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
            else:
                eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
            k = eris_ovvv.shape[0]

            t1_3 += 0.5*lib.einsum('ilde,lead->ia', t2_2[:,a:a+k],eris_ovvv,optimize=True)
            t1_3 -= 0.5*lib.einsum('lide,lead->ia', t2_2[a:a+k],eris_ovvv,optimize=True)

            t1_3 -= 0.5*lib.einsum('ilde,ldae->ia', t2_2[:,a:a+k],eris_ovvv,optimize=True)
            t1_3 += 0.5*lib.einsum('lide,ldae->ia', t2_2[a:a+k],eris_ovvv,optimize=True)

            t1_3 -= lib.einsum('ildf,mefa,lmde->ia',t2_1[:], eris_ovvv,  t2_1[:,a:a+k] ,optimize=True)
            t1_3 += lib.einsum('ildf,mefa,mlde->ia',t2_1[:], eris_ovvv,  t2_1[a:a+k] ,optimize=True)
            t1_3 += lib.einsum('lidf,mefa,lmde->ia',t2_1[:], eris_ovvv,  t2_1[:,a:a+k] ,optimize=True)
            t1_3 -= lib.einsum('lidf,mefa,mlde->ia',t2_1[:], eris_ovvv,  t2_1[a:a+k] ,optimize=True)

            t1_3 += lib.einsum('ildf,mafe,lmde->ia',t2_1[:], eris_ovvv,  t2_1[:,a:a+k] ,optimize=True)
            t1_3 -= lib.einsum('ildf,mafe,mlde->ia',t2_1[:], eris_ovvv,  t2_1[a:a+k] ,optimize=True)
            t1_3 -= lib.einsum('lidf,mafe,lmde->ia',t2_1[:], eris_ovvv,  t2_1[:,a:a+k] ,optimize=True)
            t1_3 += lib.einsum('lidf,mafe,mlde->ia',t2_1[:], eris_ovvv,  t2_1[a:a+k] ,optimize=True)

            t1_3 += lib.einsum('ilfd,mefa,mled->ia',  t2_1[:],eris_ovvv, t2_1[a:a+k],optimize=True)
            t1_3 -= lib.einsum('ilfd,mafe,mled->ia',  t2_1[:],eris_ovvv, t2_1[a:a+k],optimize=True)

            t1_3 += 0.5*lib.einsum('ilaf,mefd,lmde->ia',t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
            t1_3 -= 0.5*lib.einsum('ilaf,mefd,mlde->ia',t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)
            t1_3 -= 0.5*lib.einsum('liaf,mefd,lmde->ia',t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
            t1_3 += 0.5*lib.einsum('liaf,mefd,mlde->ia',t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)

            t1_3 -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
            t1_3 += 0.5*lib.einsum('ilaf,mdfe,mlde->ia',t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)
            t1_3 += 0.5*lib.einsum('liaf,mdfe,lmde->ia',t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
            t1_3 -= 0.5*lib.einsum('liaf,mdfe,mlde->ia',t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)

            t1_3[a:a+k] += 0.5*lib.einsum('lmdf,iaef,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3[a:a+k] -= 0.5*lib.einsum('lmdf,iaef,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3[a:a+k] -= 0.5*lib.einsum('mldf,iaef,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3[a:a+k] += 0.5*lib.einsum('mldf,iaef,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)

            t1_3[a:a+k] -= 0.5*lib.einsum('lmdf,ifea,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3[a:a+k] += 0.5*lib.einsum('lmdf,ifea,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3[a:a+k] += 0.5*lib.einsum('mldf,ifea,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3[a:a+k] -= 0.5*lib.einsum('mldf,ifea,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)

            t1_3[a:a+k] += lib.einsum('mlfd,iaef,mled->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3[a:a+k] -= lib.einsum('mlfd,ifea,mled->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)

            t1_3[a:a+k] -= 0.25*lib.einsum('lmef,iedf,lmad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3[a:a+k] += 0.25*lib.einsum('lmef,iedf,mlad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3[a:a+k] += 0.25*lib.einsum('mlef,iedf,lmad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3[a:a+k] -= 0.25*lib.einsum('mlef,iedf,mlad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)

            t1_3[a:a+k] += 0.25*lib.einsum('lmef,ifde,lmad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3[a:a+k] -= 0.25*lib.einsum('lmef,ifde,mlad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3[a:a+k] -= 0.25*lib.einsum('mlef,ifde,lmad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3[a:a+k] += 0.25*lib.einsum('mlef,ifde,mlad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)

            t1_3 += 0.5*lib.einsum('ilaf,mefd,lmde->ia',t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
            t1_3 -= 0.5*lib.einsum('ilaf,mefd,mlde->ia',t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)

            t1_3 -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
            t1_3 += 0.5*lib.einsum('ilaf,mdfe,mlde->ia',t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)

            t1_3 -= lib.einsum('ildf,mafe,mlde->ia',t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)
            t1_3 += lib.einsum('ilaf,mefd,mled->ia',t2_1[:],eris_ovvv,t2_1[a:a+k],optimize=True)

            t1_3[a:a+k] += 0.5*lib.einsum('lmdf,iaef,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3[a:a+k] -= 0.5*lib.einsum('lmdf,iaef,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3[a:a+k] -= 0.5*lib.einsum('mldf,iaef,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3[a:a+k] += 0.5*lib.einsum('mldf,iaef,mlde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)

            t1_3[a:a+k] += lib.einsum('lmdf,iaef,lmde->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)
            t1_3[a:a+k] -= lib.einsum('lmef,iedf,lmad->ia',t2_1[:],eris_ovvv,t2_1[:],optimize=True)

            t1_3 += lib.einsum('ilde,lead->ia',t2_2[:,a:a+k],eris_ovvv,optimize=True)

            t1_3 -= lib.einsum('ildf,mefa,lmde->ia',t2_1[:],eris_ovvv, t2_1[:,a:a+k],optimize=True)
            t1_3 += lib.einsum('lidf,mefa,lmde->ia',t2_1[:],eris_ovvv, t2_1[:,a:a+k],optimize=True)

            t1_3 += lib.einsum('ilfd,mefa,lmde->ia',t2_1[:],eris_ovvv,t2_1[:,a:a+k] ,optimize=True)
            t1_3 -= lib.einsum('ilfd,mefa,mlde->ia',t2_1[:],eris_ovvv,t2_1[a:a+k] ,optimize=True)

            t1_3 += lib.einsum('ilaf,mefd,lmde->ia',t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)
            t1_3 -= lib.einsum('liaf,mefd,lmde->ia',t2_1[:],eris_ovvv,t2_1[:,a:a+k],optimize=True)

            del eris_ovvv
            a += k

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

    #Compute MP2 correlation energy

    e_mp = 0.5 * lib.einsum('ijab,iabj', t2_new, eris_ovvo,optimize=True)
    e_mp -= 0.5 * lib.einsum('ijab,ibaj', t2_new, eris_ovvo,optimize=True)
    e_mp -= 0.5 * lib.einsum('jiab,iabj', t2_new, eris_ovvo,optimize=True)
    e_mp += 0.5 * lib.einsum('jiab,ibaj', t2_new, eris_ovvo,optimize=True)
    e_mp += lib.einsum('ijab,iabj', t2_new, eris_ovvo,optimize=True)

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

    X = myadc.X

    if X is None:
        nroots = myadc.U.shape[1]
        P,X = myadc.get_properties(nroots)

    nroots = X.shape[1]
    dyson_mo = np.dot(myadc.mo_coeff,X)

    return dyson_mo


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
        self.tol_residual = getattr(__config__, 'adc_radc_RADC_tol_res', 1e-6)
        self.scf_energy = mf.e_tot

        self.frozen = frozen
        self.incore_complete = self.incore_complete or self.mol.incore_anyway

        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.e_corr = None
        self.t1 = None
        self.t2 = None
        self.imds = lambda:None
        self._nocc = mf.mol.nelectron//2
        self._nmo = mo_coeff.shape[1]
        self._nvir = self._nmo - self._nocc
        self.mo_energy = mf.mo_energy
        self.chkfile = mf.chkfile
        self.method = "adc(2)"
        self.method_type = "ip"
        self.with_df = None
        self.compute_properties = True
        self.evec_print_tol = 0.1
        self.spec_factor_print_tol = 0.1

        self.E = None
        self.U = None
        self.P = None
        self.X = None

        keys = set(('tol_residual','conv_tol', 'e_corr', 'method', 'mo_coeff',
                    'mol', 'mo_energy', 'max_memory', 'incore_complete',
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
        self._finalize()

        self.method_type = self.method_type.lower()
        if(self.method_type == "ea"):
            e_exc, v_exc, spec_fac, x, adc_es = self.ea_adc(nroots=nroots, guess=guess, eris=eris)

        elif(self.method_type == "ip"):
            e_exc, v_exc, spec_fac, x, adc_es = self.ip_adc(nroots=nroots, guess=guess, eris=eris)

        else:
            raise NotImplementedError(self.method_type)
        self._adc_es = adc_es
        return e_exc, v_exc, spec_fac, x

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        logger.note(self, 'E_corr = %.8f',
                    self.e_corr)
        return self

    def ea_adc(self, nroots=1, guess=None, eris=None):
        adc_es = RADCEA(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris)
        return e_exc, v_exc, spec_fac, x, adc_es

    def ip_adc(self, nroots=1, guess=None, eris=None):
        adc_es = RADCIP(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris)
        return e_exc, v_exc, spec_fac, x, adc_es

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

    def analyze(self):
        self._adc_es.analyze()

    def compute_dyson_mo(self):
        return self._adc_es.compute_dyson_mo()


def get_imds_ea(adc, eris=None):

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t1 = adc.t1
    t2 = adc.t2

    t1_2 = t1[0]

    eris_ovvo = eris.ovvo
    nocc = adc._nocc
    nvir = adc._nvir

    e_occ = adc.mo_energy[:nocc].copy()
    e_vir = adc.mo_energy[nocc:].copy()

    idn_vir = np.identity(nvir)

    if eris is None:
        eris = adc.transform_integrals()

    # a-b block
    # Zeroth-order terms

    M_ab = lib.einsum('ab,a->ab', idn_vir, e_vir)

    # Second-order terms
    t2_1 = t2[0][:]

    M_ab +=  lib.einsum('l,lmad,lmbd->ab',e_occ ,t2_1, t2_1,optimize=True)
    M_ab -=  lib.einsum('l,lmad,mlbd->ab',e_occ ,t2_1, t2_1,optimize=True)
    M_ab -=  lib.einsum('l,mlad,lmbd->ab',e_occ ,t2_1, t2_1,optimize=True)
    M_ab +=  lib.einsum('l,mlad,mlbd->ab',e_occ ,t2_1, t2_1,optimize=True)
    M_ab +=  lib.einsum('l,lmad,lmbd->ab',e_occ,t2_1, t2_1,optimize=True)
    M_ab +=  lib.einsum('l,mlad,mlbd->ab',e_occ,t2_1, t2_1,optimize=True)

    M_ab -= 0.5 *  lib.einsum('d,lmad,lmbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab += 0.5 *  lib.einsum('d,lmad,mlbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab += 0.5 *  lib.einsum('d,mlad,lmbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab -= 0.5 *  lib.einsum('d,mlad,mlbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab -= 0.5 *  lib.einsum('d,lmad,lmbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab -= 0.5 *  lib.einsum('d,mlad,mlbd->ab',e_vir,t2_1, t2_1,optimize=True)

    M_ab_t = lib.einsum('lmad,lmbd->ab', t2_1,t2_1, optimize=True)
    M_ab -= 1 *  lib.einsum('a,ab->ab',e_vir,M_ab_t,optimize=True)
    M_ab -= 1 *  lib.einsum('b,ab->ab',e_vir,M_ab_t,optimize=True)

    M_ab_t = lib.einsum('lmad,mlbd->ab', t2_1,t2_1, optimize=True)
    M_ab += 0.5 *  lib.einsum('a,ab->ab',e_vir,M_ab_t,optimize=True)
    M_ab += 0.5 *  lib.einsum('b,ab->ab',e_vir,M_ab_t,optimize=True)
    del M_ab_t

    M_ab -= 0.5 *  lib.einsum('lmad,lbdm->ab',t2_1, eris_ovvo,optimize=True)
    M_ab += 0.5 *  lib.einsum('mlad,lbdm->ab',t2_1, eris_ovvo,optimize=True)
    M_ab += 0.5 *  lib.einsum('lmad,ldbm->ab',t2_1, eris_ovvo,optimize=True)
    M_ab -= 0.5 *  lib.einsum('mlad,ldbm->ab',t2_1, eris_ovvo,optimize=True)
    M_ab -=        lib.einsum('lmad,lbdm->ab',t2_1, eris_ovvo,optimize=True)

    M_ab -= 0.5 *  lib.einsum('lmbd,ladm->ab',t2_1, eris_ovvo,optimize=True)
    M_ab += 0.5 *  lib.einsum('mlbd,ladm->ab',t2_1, eris_ovvo,optimize=True)
    M_ab += 0.5 *  lib.einsum('lmbd,ldam->ab',t2_1, eris_ovvo,optimize=True)
    M_ab -= 0.5 *  lib.einsum('mlbd,ldam->ab',t2_1, eris_ovvo,optimize=True)
    M_ab -=        lib.einsum('lmbd,ladm->ab',t2_1, eris_ovvo,optimize=True)

    del t2_1
    cput0 = log.timer_debug1("Completed M_ab second-order terms ADC(2) calculation", *cput0)

    #Third-order terms

    if(method =='adc(3)'):

        eris_oovv = eris.oovv
        eris_oooo = eris.oooo

        if isinstance(eris.ovvv, type(None)):
            chnk_size = radc_ao2mo.calculate_chunk_size(adc)
        else:
            chnk_size = nocc
        a = 0
        for p in range(0,nocc,chnk_size):
            if getattr(adc, 'with_df', None):
                eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
            else:
                eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
            k = eris_ovvv.shape[0]
            M_ab += 4. * lib.einsum('ld,ldab->ab',t1_2[a:a+k], eris_ovvv,optimize=True)
            M_ab -=  lib.einsum('ld,lbad->ab',t1_2[a:a+k], eris_ovvv,optimize=True)
            M_ab -= lib.einsum('ld,ladb->ab',t1_2[a:a+k], eris_ovvv,optimize=True)
            del eris_ovvv
            a += k

        cput0 = log.timer_debug1("Completed M_ab ovvv ADC(3) calculation", *cput0)
        t2_2 = t2[1][:]

        M_ab -= 0.5 *  lib.einsum('lmad,lbdm->ab',t2_2, eris_ovvo,optimize=True)
        M_ab += 0.5 *  lib.einsum('mlad,lbdm->ab',t2_2, eris_ovvo,optimize=True)
        M_ab += 0.5 *  lib.einsum('lmad,ldbm->ab',t2_2, eris_ovvo,optimize=True)
        M_ab -= 0.5 *  lib.einsum('mlad,ldbm->ab',t2_2, eris_ovvo,optimize=True)
        M_ab -=        lib.einsum('lmad,lbdm->ab',t2_2, eris_ovvo,optimize=True)

        M_ab -= 0.5 * lib.einsum('lmbd,ladm->ab',t2_2,eris_ovvo,optimize=True)
        M_ab += 0.5 * lib.einsum('mlbd,ladm->ab',t2_2,eris_ovvo,optimize=True)
        M_ab += 0.5 * lib.einsum('lmbd,ldam->ab',t2_2, eris_ovvo,optimize=True)
        M_ab -= 0.5 * lib.einsum('mlbd,ldam->ab',t2_2, eris_ovvo,optimize=True)
        M_ab -=       lib.einsum('lmbd,ladm->ab',t2_2,eris_ovvo,optimize=True)
        t2_1 = t2[0][:]

        M_ab += lib.einsum('l,lmbd,lmad->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab -= lib.einsum('l,lmbd,mlad->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab -= lib.einsum('l,mlbd,lmad->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab += lib.einsum('l,mlbd,mlad->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab += lib.einsum('l,lmbd,lmad->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab += lib.einsum('l,mlbd,mlad->ab',e_occ, t2_1, t2_2, optimize=True)

        M_ab += lib.einsum('l,lmad,lmbd->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab -= lib.einsum('l,lmad,mlbd->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab -= lib.einsum('l,mlad,lmbd->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab += lib.einsum('l,mlad,mlbd->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab += lib.einsum('l,lmad,lmbd->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab += lib.einsum('l,mlad,mlbd->ab',e_occ, t2_1, t2_2, optimize=True)

        M_ab -= 0.5*lib.einsum('d,lmbd,lmad->ab', e_vir, t2_1 ,t2_2, optimize=True)
        M_ab += 0.5*lib.einsum('d,lmbd,mlad->ab', e_vir, t2_1 ,t2_2, optimize=True)
        M_ab += 0.5*lib.einsum('d,mlbd,lmad->ab', e_vir, t2_1 ,t2_2, optimize=True)
        M_ab -= 0.5*lib.einsum('d,mlbd,mlad->ab', e_vir, t2_1 ,t2_2, optimize=True)
        M_ab -= 0.5*lib.einsum('d,lmbd,lmad->ab', e_vir, t2_1 ,t2_2, optimize=True)
        M_ab -= 0.5*lib.einsum('d,mlbd,mlad->ab', e_vir, t2_1 ,t2_2, optimize=True)

        M_ab -= 0.5*lib.einsum('d,lmad,lmbd->ab', e_vir, t2_1, t2_2, optimize=True)
        M_ab += 0.5*lib.einsum('d,lmad,mlbd->ab', e_vir, t2_1, t2_2, optimize=True)
        M_ab += 0.5*lib.einsum('d,mlad,lmbd->ab', e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.5*lib.einsum('d,mlad,mlbd->ab', e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.5*lib.einsum('d,lmad,lmbd->ab', e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.5*lib.einsum('d,mlad,mlbd->ab', e_vir, t2_1, t2_2, optimize=True)

        M_ab_t = lib.einsum('lmbd,lmad->ab', t2_1,t2_2, optimize=True)
        M_ab -= 1. * lib.einsum('a,ab->ab',e_vir, M_ab_t, optimize=True)
        M_ab -= 1. * lib.einsum('a,ba->ab',e_vir, M_ab_t, optimize=True)
        M_ab -= 1. * lib.einsum('b,ab->ab',e_vir, M_ab_t, optimize=True)
        M_ab -= 1. * lib.einsum('b,ba->ab',e_vir, M_ab_t, optimize=True)
        del M_ab_t

        M_ab_t_1 = lib.einsum('lmbd,mlad->ab', t2_1,t2_2, optimize=True)
        del t2_2
        M_ab += 0.5 * lib.einsum('a,ab->ab',e_vir, M_ab_t_1, optimize=True)
        M_ab += 0.5 * lib.einsum('a,ba->ab',e_vir, M_ab_t_1, optimize=True)
        M_ab += 0.5 * lib.einsum('b,ab->ab',e_vir, M_ab_t_1, optimize=True)
        M_ab += 0.5 * lib.einsum('b,ba->ab',e_vir, M_ab_t_1, optimize=True)
        del M_ab_t_1

        log.timer_debug1("Starting the small integrals  calculation")
        temp_t2_v_1 = lib.einsum('lned,mlbd->nemb',t2_1, t2_1,optimize=True)
        M_ab -= lib.einsum('nemb,nmae->ab',temp_t2_v_1, eris_oovv, optimize=True)
        M_ab -= lib.einsum('mbne,nmae->ab',temp_t2_v_1, eris_oovv, optimize=True)
        M_ab += lib.einsum('nemb,maen->ab',temp_t2_v_1, eris_ovvo, optimize=True)
        M_ab += lib.einsum('mbne,maen->ab',temp_t2_v_1, eris_ovvo, optimize=True)
        M_ab += lib.einsum('nemb,neam->ab',temp_t2_v_1, eris_ovvo, optimize=True)
        M_ab -= lib.einsum('name,nmeb->ab',temp_t2_v_1, eris_oovv, optimize=True)
        M_ab -= lib.einsum('mena,nmeb->ab',temp_t2_v_1, eris_oovv, optimize=True)
        M_ab += 2. * lib.einsum('name,nbem->ab',temp_t2_v_1, eris_ovvo, optimize=True)
        M_ab += 2. * lib.einsum('mena,nbem->ab',temp_t2_v_1, eris_ovvo, optimize=True)
        M_ab += lib.einsum('nbme,mean->ab',temp_t2_v_1, eris_ovvo, optimize=True)
        del temp_t2_v_1

        temp_t2_v_2 = lib.einsum('nled,mlbd->nemb',t2_1, t2_1,optimize=True)
        M_ab += 2. * lib.einsum('nemb,nmae->ab',temp_t2_v_2, eris_oovv, optimize=True)
        M_ab -= 2. * lib.einsum('nemb,maen->ab',temp_t2_v_2, eris_ovvo, optimize=True)
        M_ab -= lib.einsum('nemb,neam->ab',temp_t2_v_2, eris_ovvo, optimize=True)
        M_ab += 2. * lib.einsum('mena,nmeb->ab',temp_t2_v_2, eris_oovv, optimize=True)
        M_ab -= 4. * lib.einsum('mena,nbem->ab',temp_t2_v_2, eris_ovvo, optimize=True)
        M_ab -= lib.einsum('nemb,neam->ab',temp_t2_v_2, eris_ovvo, optimize=True)
        del temp_t2_v_2

        temp_t2_v_3 = lib.einsum('lned,lmbd->nemb',t2_1, t2_1,optimize=True)
        M_ab -= lib.einsum('nemb,maen->ab',temp_t2_v_3, eris_ovvo, optimize=True)
        M_ab += 2. * lib.einsum('nemb,nmae->ab',temp_t2_v_3, eris_oovv, optimize=True)
        M_ab += 2. * lib.einsum('mena,nmeb->ab',temp_t2_v_3, eris_oovv, optimize=True)
        M_ab -= lib.einsum('mena,nbem->ab',temp_t2_v_3, eris_ovvo, optimize=True)
        del temp_t2_v_3

        temp_t2_v_4 = lib.einsum('lnae,nmde->lmad',t2_1, eris_oovv,optimize=True)
        M_ab -= lib.einsum('mlbd,lmad->ab',t2_1, temp_t2_v_4,optimize=True)
        M_ab += 2. * lib.einsum('lmbd,lmad->ab',t2_1, temp_t2_v_4,optimize=True)
        del temp_t2_v_4

        temp_t2_v_5 = lib.einsum('nlae,nmde->lamd',t2_1, eris_oovv,optimize=True)
        M_ab += 2. * lib.einsum('mlbd,lamd->ab',t2_1, temp_t2_v_5, optimize=True)
        M_ab -= lib.einsum('lmbd,lamd->ab',t2_1, temp_t2_v_5, optimize=True)
        del temp_t2_v_5

        temp_t2_v_6 = lib.einsum('lnae,nedm->ladm',t2_1, eris_ovvo,optimize=True)
        M_ab += 2. * lib.einsum('mlbd,ladm->ab',t2_1, temp_t2_v_6, optimize=True)
        M_ab -= 4. * lib.einsum('lmbd,ladm->ab',t2_1, temp_t2_v_6, optimize=True)
        del temp_t2_v_6

        temp_t2_v_7 = lib.einsum('nlae,nedm->ladm',t2_1, eris_ovvo,optimize=True)
        M_ab -= lib.einsum('mlbd,ladm->ab',t2_1, temp_t2_v_7, optimize=True)
        M_ab += 2. * lib.einsum('lmbd,ladm->ab',t2_1, temp_t2_v_7, optimize=True)
        del temp_t2_v_7

        temp_t2_v_8 = lib.einsum('lned,mled->mn',t2_1, t2_1,optimize=True)
        M_ab += 2.* lib.einsum('mn,nmab->ab',temp_t2_v_8, eris_oovv, optimize=True)
        M_ab -= lib.einsum('mn,nbam->ab', temp_t2_v_8, eris_ovvo, optimize=True)
        del temp_t2_v_8

        temp_t2_v_9 = lib.einsum('nled,mled->mn',t2_1, t2_1,optimize=True)
        M_ab -= 4.* lib.einsum('mn,nmab->ab',temp_t2_v_9, eris_oovv, optimize=True)
        M_ab += 2. * lib.einsum('mn,nbam->ab',temp_t2_v_9, eris_ovvo, optimize=True)
        del temp_t2_v_9

        temp_t2_v_10 = lib.einsum('noad,nmol->mlad',t2_1, eris_oooo,optimize=True)
        M_ab -= 0.25*lib.einsum('mlbd,mlad->ab',t2_1, temp_t2_v_10, optimize=True)
        M_ab += 0.25*lib.einsum('lmbd,mlad->ab',t2_1, temp_t2_v_10, optimize=True)
        M_ab += 0.25*lib.einsum('mlbd,lmad->ab',t2_1, temp_t2_v_10, optimize=True)
        M_ab -= 0.25*lib.einsum('lmbd,lmad->ab',t2_1, temp_t2_v_10, optimize=True)
        M_ab -= lib.einsum('mlbd,mlad->ab',t2_1, temp_t2_v_10, optimize=True)
        del temp_t2_v_10

        temp_t2_v_11 = lib.einsum('onad,nmol->mlad',t2_1, eris_oooo,optimize=True)
        M_ab += 0.25*lib.einsum('mlbd,mlad->ab',t2_1, temp_t2_v_11, optimize=True)
        M_ab -= 0.25*lib.einsum('lmbd,mlad->ab',t2_1, temp_t2_v_11, optimize=True)
        M_ab -= 0.25*lib.einsum('mlbd,lmad->ab',t2_1, temp_t2_v_11, optimize=True)
        M_ab += 0.25*lib.einsum('lmbd,lmad->ab',t2_1, temp_t2_v_11, optimize=True)
        del temp_t2_v_11
        log.timer_debug1("Completed M_ab ADC(3) small integrals calculation")

        log.timer_debug1("Starting M_ab vvvv ADC(3) calculation")

        if isinstance(eris.vvvv, np.ndarray):
            temp_t2 = adc.imds.t2_1_vvvv
            M_ab -= 0.25*lib.einsum('mlaf,mlbf->ab',t2_1, temp_t2, optimize=True)
            M_ab += 0.25*lib.einsum('mlaf,lmbf->ab',t2_1, temp_t2, optimize=True)
            M_ab += 0.25*lib.einsum('lmaf,mlbf->ab',t2_1, temp_t2, optimize=True)
            M_ab -= 0.25*lib.einsum('lmaf,lmbf->ab',t2_1, temp_t2, optimize=True)
            M_ab += 0.25*lib.einsum('mlaf,mlfb->ab',t2_1, temp_t2, optimize=True)
            M_ab -= 0.25*lib.einsum('mlaf,lmfb->ab',t2_1, temp_t2, optimize=True)
            M_ab -= 0.25*lib.einsum('lmaf,mlfb->ab',t2_1, temp_t2, optimize=True)
            M_ab += 0.25*lib.einsum('lmaf,lmfb->ab',t2_1, temp_t2, optimize=True)
            M_ab -= lib.einsum('mlaf,mlbf->ab',t2_1, temp_t2, optimize=True)

            M_ab -= 0.25*lib.einsum('mlad,mlbd->ab', temp_t2, t2_1, optimize=True)
            M_ab += 0.25*lib.einsum('mlad,lmbd->ab', temp_t2, t2_1, optimize=True)
            M_ab += 0.25*lib.einsum('lmad,mlbd->ab', temp_t2, t2_1, optimize=True)
            M_ab -= 0.25*lib.einsum('lmad,lmbd->ab', temp_t2, t2_1, optimize=True)
            M_ab -= lib.einsum('mlad,mlbd->ab', temp_t2, t2_1, optimize=True)

            M_ab += 0.25*lib.einsum('lmad,mlbd->ab',temp_t2, t2_1, optimize=True)
            M_ab -= 0.25*lib.einsum('lmad,lmbd->ab',temp_t2, t2_1, optimize=True)
            M_ab -= 0.25*lib.einsum('mlad,mlbd->ab',temp_t2, t2_1, optimize=True)
            M_ab += 0.25*lib.einsum('mlad,lmbd->ab',temp_t2, t2_1, optimize=True)
            del temp_t2

            eris_vvvv =  eris.vvvv
            eris_vvvv = eris_vvvv.reshape(nvir,nvir,nvir,nvir)
            M_ab -= lib.einsum('mldf,mled,aebf->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab += lib.einsum('mldf,lmed,aebf->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab += lib.einsum('lmdf,mled,aebf->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab -= lib.einsum('lmdf,lmed,aebf->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab += 0.5*lib.einsum('mldf,mled,aefb->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab -= 0.5*lib.einsum('mldf,lmed,aefb->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab -= 0.5*lib.einsum('lmdf,mled,aefb->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab += 0.5*lib.einsum('lmdf,lmed,aefb->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab += 2.*lib.einsum('mlfd,mled,aebf->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab -= lib.einsum('mlfd,mled,aefb->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            eris_vvvv = eris_vvvv.reshape(nvir*nvir,nvir*nvir)

        else:
            temp_t2_vvvv = adc.imds.t2_1_vvvv[:]
            M_ab -= 0.25*lib.einsum('mlaf,mlbf->ab',t2_1, temp_t2_vvvv, optimize=True)
            M_ab += 0.25*lib.einsum('mlaf,lmbf->ab',t2_1, temp_t2_vvvv, optimize=True)
            M_ab += 0.25*lib.einsum('lmaf,mlbf->ab',t2_1, temp_t2_vvvv, optimize=True)
            M_ab -= 0.25*lib.einsum('lmaf,lmbf->ab',t2_1, temp_t2_vvvv, optimize=True)

            M_ab += 0.25*lib.einsum('mlaf,mlfb->ab',t2_1, temp_t2_vvvv, optimize=True)
            M_ab -= 0.25*lib.einsum('mlaf,lmfb->ab',t2_1, temp_t2_vvvv, optimize=True)
            M_ab -= 0.25*lib.einsum('lmaf,mlfb->ab',t2_1, temp_t2_vvvv, optimize=True)
            M_ab += 0.25*lib.einsum('lmaf,lmfb->ab',t2_1, temp_t2_vvvv, optimize=True)

            M_ab -= lib.einsum('mlaf,mlbf->ab',t2_1, temp_t2_vvvv, optimize=True)

            M_ab += 0.25*lib.einsum('lmad,mlbd->ab',temp_t2_vvvv, t2_1, optimize=True)
            M_ab -= 0.25*lib.einsum('lmad,lmbd->ab',temp_t2_vvvv, t2_1, optimize=True)
            M_ab -= 0.25*lib.einsum('mlad,mlbd->ab',temp_t2_vvvv, t2_1, optimize=True)
            M_ab += 0.25*lib.einsum('mlad,lmbd->ab',temp_t2_vvvv, t2_1, optimize=True)

            M_ab -= 0.25*lib.einsum('mlad,mlbd->ab', temp_t2_vvvv, t2_1, optimize=True)
            M_ab += 0.25*lib.einsum('mlad,lmbd->ab', temp_t2_vvvv, t2_1, optimize=True)
            M_ab += 0.25*lib.einsum('lmad,mlbd->ab', temp_t2_vvvv, t2_1, optimize=True)
            M_ab -= 0.25*lib.einsum('lmad,lmbd->ab', temp_t2_vvvv, t2_1, optimize=True)
            M_ab -= lib.einsum('mlad,mlbd->ab', temp_t2_vvvv, t2_1, optimize=True)
            del temp_t2_vvvv

            chnk_size = radc_ao2mo.calculate_chunk_size(adc)
            a = 0
            temp = np.zeros((nvir,nvir))

            if isinstance(eris.vvvv, list):
                for dataset in eris.vvvv:
                    k = dataset.shape[0]
                    eris_vvvv = dataset[:].reshape(-1,nvir,nvir,nvir)
                    temp[a:a+k] -= lib.einsum('mldf,mled,aebf->ab',t2_1, t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] += lib.einsum('mldf,lmed,aebf->ab',t2_1, t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] += lib.einsum('lmdf,mled,aebf->ab',t2_1, t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] -= lib.einsum('lmdf,lmed,aebf->ab',t2_1, t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] += 0.5*lib.einsum('mldf,mled,aefb->ab',t2_1, t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] -= 0.5*lib.einsum('mldf,lmed,aefb->ab',t2_1, t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] -= 0.5*lib.einsum('lmdf,mled,aefb->ab',t2_1, t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] += 0.5*lib.einsum('lmdf,lmed,aefb->ab',t2_1, t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] += 2.*lib.einsum('mlfd,mled,aebf->ab',t2_1, t2_1, eris_vvvv, optimize=True)
                    temp[a:a+k] -= lib.einsum('mlfd,mled,aefb->ab',t2_1, t2_1, eris_vvvv, optimize=True)
                    del eris_vvvv
                    a += k
            else:
                for p in range(0,nvir,chnk_size):

                    vvvv = dfadc.get_vvvv_df(adc, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
                    k = vvvv.shape[0]
                    temp[a:a+k] -= lib.einsum('mldf,mled,aebf->ab',t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] += lib.einsum('mldf,lmed,aebf->ab',t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] += lib.einsum('lmdf,mled,aebf->ab',t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] -= lib.einsum('lmdf,lmed,aebf->ab',t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] += 0.5*lib.einsum('mldf,mled,aefb->ab',t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] -= 0.5*lib.einsum('mldf,lmed,aefb->ab',t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] -= 0.5*lib.einsum('lmdf,mled,aefb->ab',t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] += 0.5*lib.einsum('lmdf,lmed,aefb->ab',t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] += 2.*lib.einsum('mlfd,mled,aebf->ab',t2_1, t2_1, vvvv, optimize=True)
                    temp[a:a+k] -= lib.einsum('mlfd,mled,aefb->ab',t2_1, t2_1, vvvv, optimize=True)
                    del vvvv
                    a += k

            M_ab += temp
            del temp
            del t2_1

    cput0 = log.timer_debug1("Completed M_ab ADC(3) calculation", *cput0)
    return M_ab


def get_imds_ip(adc, eris=None):

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t1 = adc.t1
    t2 = adc.t2

    t1_2 = t1[0]

    nocc = adc._nocc

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    idn_occ = np.identity(nocc)

    if eris is None:
        eris = adc.transform_integrals()

    eris_ovvo = eris.ovvo

    # i-j block
    # Zeroth-order terms

    M_ij = lib.einsum('ij,j->ij', idn_occ ,e_occ)

    # Second-order terms
    t2_1 = t2[0][:]

    M_ij +=  lib.einsum('d,ilde,jlde->ij',e_vir,t2_1, t2_1, optimize=True)
    M_ij -=  lib.einsum('d,ilde,ljde->ij',e_vir,t2_1, t2_1, optimize=True)
    M_ij -=  lib.einsum('d,lide,jlde->ij',e_vir,t2_1, t2_1, optimize=True)
    M_ij +=  lib.einsum('d,lide,ljde->ij',e_vir,t2_1, t2_1, optimize=True)
    M_ij +=  lib.einsum('d,ilde,jlde->ij',e_vir,t2_1, t2_1, optimize=True)
    M_ij +=  lib.einsum('d,iled,jled->ij',e_vir,t2_1, t2_1, optimize=True)

    M_ij -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij += 0.5 *  lib.einsum('l,ilde,ljde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij += 0.5 *  lib.einsum('l,lide,jlde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij -= 0.5 *  lib.einsum('l,lide,ljde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ,t2_1, t2_1, optimize=True)

    M_ij_t = lib.einsum('ilde,jlde->ij', t2_1,t2_1, optimize=True)
    M_ij -= lib.einsum('i,ij->ij',e_occ,M_ij_t, optimize=True)
    M_ij -= lib.einsum('j,ij->ij',e_occ,M_ij_t, optimize=True)

    M_ij_t = lib.einsum('ilde,ljde->ij', t2_1,t2_1, optimize=True)
    M_ij += 0.5 * lib.einsum('i,ij->ij',e_occ,M_ij_t, optimize=True)
    M_ij += 0.5 * lib.einsum('j,ij->ij',e_occ,M_ij_t, optimize=True)
    del M_ij_t

    M_ij += 0.5 *  lib.einsum('ilde,jdel->ij',t2_1, eris_ovvo,optimize=True)
    M_ij -= 0.5 *  lib.einsum('lide,jdel->ij',t2_1, eris_ovvo,optimize=True)
    M_ij -= 0.5 *  lib.einsum('ilde,jedl->ij',t2_1, eris_ovvo,optimize=True)
    M_ij += 0.5 *  lib.einsum('lide,jedl->ij',t2_1, eris_ovvo,optimize=True)
    M_ij += lib.einsum('ilde,jdel->ij',t2_1, eris_ovvo,optimize=True)

    M_ij += 0.5 *  lib.einsum('jlde,idel->ij',t2_1, eris_ovvo,optimize=True)
    M_ij -= 0.5 *  lib.einsum('ljde,idel->ij',t2_1, eris_ovvo,optimize=True)
    M_ij -= 0.5 *  lib.einsum('jlde,ldei->ij',t2_1, eris_ovvo,optimize=True)
    M_ij += 0.5 *  lib.einsum('ljde,ldei->ij',t2_1, eris_ovvo,optimize=True)
    M_ij += lib.einsum('jlde,idel->ij',t2_1, eris_ovvo,optimize=True)

    del t2_1
    cput0 = log.timer_debug1("Completed M_ij second-order terms ADC(2) calculation", *cput0)
    # Third-order terms

    if (method == "adc(3)"):

        eris_oovv = eris.oovv
        eris_ovoo = eris.ovoo
        eris_oooo = eris.oooo

        M_ij += lib.einsum('ld,ldji->ij',t1_2, eris_ovoo,optimize=True)
        M_ij -= lib.einsum('ld,jdli->ij',t1_2, eris_ovoo,optimize=True)
        M_ij += lib.einsum('ld,ldji->ij',t1_2, eris_ovoo,optimize=True)

        M_ij += lib.einsum('ld,ldij->ij',t1_2, eris_ovoo,optimize=True)
        M_ij -= lib.einsum('ld,idlj->ij',t1_2, eris_ovoo,optimize=True)
        M_ij += lib.einsum('ld,ldij->ij',t1_2, eris_ovoo,optimize=True)
        t2_2 = t2[1][:]

        M_ij += 0.5* lib.einsum('ilde,jdel->ij',t2_2, eris_ovvo,optimize=True)
        M_ij -= 0.5* lib.einsum('lide,jdel->ij',t2_2, eris_ovvo,optimize=True)
        M_ij -= 0.5* lib.einsum('ilde,jedl->ij',t2_2, eris_ovvo,optimize=True)
        M_ij += 0.5* lib.einsum('lide,jedl->ij',t2_2, eris_ovvo,optimize=True)
        M_ij += lib.einsum('ilde,jdel->ij',t2_2, eris_ovvo,optimize=True)

        M_ij += 0.5* lib.einsum('jlde,ledi->ij',t2_2, eris_ovvo,optimize=True)
        M_ij -= 0.5* lib.einsum('ljde,ledi->ij',t2_2, eris_ovvo,optimize=True)
        M_ij -= 0.5* lib.einsum('jlde,iedl->ij',t2_2, eris_ovvo,optimize=True)
        M_ij += 0.5* lib.einsum('ljde,iedl->ij',t2_2, eris_ovvo,optimize=True)
        M_ij += lib.einsum('jlde,ledi->ij',t2_2, eris_ovvo,optimize=True)
        t2_1 = t2[0][:]

        M_ij +=  lib.einsum('d,ilde,jlde->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij -=  lib.einsum('d,ilde,ljde->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij -=  lib.einsum('d,lide,jlde->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij +=  lib.einsum('d,lide,ljde->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij +=  lib.einsum('d,ilde,jlde->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij +=  lib.einsum('d,iled,jled->ij',e_vir,t2_1, t2_2,optimize=True)

        M_ij +=  lib.einsum('d,jlde,ilde->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij -=  lib.einsum('d,jlde,lide->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij -=  lib.einsum('d,ljde,ilde->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij +=  lib.einsum('d,ljde,lide->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij +=  lib.einsum('d,jlde,ilde->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij +=  lib.einsum('d,jled,iled->ij',e_vir,t2_1, t2_2,optimize=True)

        M_ij -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij += 0.5 *  lib.einsum('l,ilde,ljde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij += 0.5 *  lib.einsum('l,lide,jlde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.5 *  lib.einsum('l,lide,ljde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ,t2_1, t2_2,optimize=True)

        M_ij -= 0.5 *  lib.einsum('l,jlde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij += 0.5 *  lib.einsum('l,jlde,lide->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij += 0.5 *  lib.einsum('l,ljde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.5 *  lib.einsum('l,ljde,lide->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.5*lib.einsum('l,jlde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.5*lib.einsum('l,jlde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)

        M_ij_t = lib.einsum('ilde,jlde->ij', t2_1,t2_2, optimize=True)
        M_ij -= 1. * lib.einsum('i,ij->ij',e_occ, M_ij_t, optimize=True)
        M_ij -= 1. * lib.einsum('i,ji->ij',e_occ, M_ij_t, optimize=True)
        M_ij -= 1. * lib.einsum('j,ij->ij',e_occ, M_ij_t, optimize=True)
        M_ij -= 1. * lib.einsum('j,ji->ij',e_occ, M_ij_t, optimize=True)
        del M_ij_t

        M_ij_t_1 = lib.einsum('ilde,ljde->ij', t2_1,t2_2, optimize=True)
        del t2_2
        M_ij += 0.5 * lib.einsum('i,ij->ij',e_occ, M_ij_t_1, optimize=True)
        M_ij += 0.5 * lib.einsum('i,ji->ij',e_occ, M_ij_t_1, optimize=True)
        M_ij += 0.5 * lib.einsum('j,ij->ij',e_occ, M_ij_t_1, optimize=True)
        M_ij += 0.5 * lib.einsum('j,ji->ij',e_occ, M_ij_t_1, optimize=True)
        del M_ij_t_1

        temp_t2_vvvv = adc.imds.t2_1_vvvv[:]
        M_ij += 0.25*lib.einsum('ilde,jlde->ij',t2_1, temp_t2_vvvv, optimize = True)
        M_ij -= 0.25*lib.einsum('ilde,ljde->ij',t2_1, temp_t2_vvvv, optimize = True)
        M_ij -= 0.25*lib.einsum('lide,jlde->ij',t2_1, temp_t2_vvvv, optimize = True)
        M_ij += 0.25*lib.einsum('lide,ljde->ij',t2_1, temp_t2_vvvv, optimize = True)
        M_ij -= 0.25*lib.einsum('ilde,jled->ij',t2_1, temp_t2_vvvv, optimize = True)
        M_ij += 0.25*lib.einsum('ilde,ljed->ij',t2_1, temp_t2_vvvv, optimize = True)
        M_ij += 0.25*lib.einsum('lide,jled->ij',t2_1, temp_t2_vvvv, optimize = True)
        M_ij -= 0.25*lib.einsum('lide,ljed->ij',t2_1, temp_t2_vvvv, optimize = True)
        M_ij +=lib.einsum('ilde,jlde->ij',t2_1, temp_t2_vvvv, optimize = True)
        del temp_t2_vvvv

        log.timer_debug1("Starting the small integrals  calculation")
        temp_t2_v_1 = lib.einsum('lmde,jldf->mejf',t2_1, t2_1,optimize=True)
        M_ij -=  2 * lib.einsum('mejf,mefi->ij',temp_t2_v_1, eris_ovvo,optimize = True)
        M_ij -=  2 * lib.einsum('jfme,mefi->ij',temp_t2_v_1, eris_ovvo,optimize = True)
        M_ij +=  lib.einsum('mejf,mife->ij',temp_t2_v_1, eris_oovv,optimize = True)
        M_ij +=  lib.einsum('jfme,mife->ij',temp_t2_v_1, eris_oovv,optimize = True)
        M_ij -=  2 * lib.einsum('meif,mefj->ij',temp_t2_v_1, eris_ovvo ,optimize = True)
        M_ij -=  2 * lib.einsum('ifme,mefj->ij',temp_t2_v_1, eris_ovvo ,optimize = True)
        M_ij +=  lib.einsum('meif,mjfe->ij',temp_t2_v_1, eris_oovv ,optimize = True)
        M_ij +=  lib.einsum('ifme,mjfe->ij',temp_t2_v_1, eris_oovv ,optimize = True)
        del temp_t2_v_1

        temp_t2_v_2 = lib.einsum('lmde,ljdf->mejf',t2_1, t2_1,optimize=True)
        M_ij +=  4 * lib.einsum('mejf,mefi->ij',temp_t2_v_2, eris_ovvo,optimize = True)
        M_ij +=  4 * lib.einsum('meif,mefj->ij',temp_t2_v_2, eris_ovvo,optimize = True)
        M_ij -=  2 * lib.einsum('meif,mjfe->ij',temp_t2_v_2, eris_oovv,optimize = True)
        M_ij -=  2 * lib.einsum('mejf,mife->ij',temp_t2_v_2, eris_oovv,optimize = True)
        del temp_t2_v_2

        temp_t2_v_3 = lib.einsum('mlde,jldf->mejf',t2_1, t2_1,optimize=True)
        M_ij += lib.einsum('mejf,mefi->ij',temp_t2_v_3, eris_ovvo,optimize = True)
        M_ij += lib.einsum('meif,mefj->ij',temp_t2_v_3, eris_ovvo,optimize = True)
        M_ij -= 2 *lib.einsum('meif,mjfe->ij',temp_t2_v_3, eris_oovv,optimize = True)
        M_ij -= 2 * lib.einsum('mejf,mife->ij',temp_t2_v_3, eris_oovv,optimize = True)
        del temp_t2_v_3

        temp_t2_v_4 = lib.einsum('ilde,lmfe->idmf',t2_1, eris_oovv,optimize=True)
        M_ij -= 2 * lib.einsum('idmf,jmdf->ij',temp_t2_v_4, t2_1, optimize = True)
        M_ij += lib.einsum('idmf,mjdf->ij',temp_t2_v_4, t2_1, optimize = True)
        del temp_t2_v_4

        temp_t2_v_5 = lib.einsum('lide,lmfe->idmf',t2_1, eris_oovv,optimize=True)
        M_ij += lib.einsum('idmf,jmdf->ij',temp_t2_v_5, t2_1, optimize = True)
        M_ij -= 2 * lib.einsum('idmf,mjdf->ij',temp_t2_v_5, t2_1, optimize = True)
        del temp_t2_v_5

        temp_t2_v_6 = lib.einsum('ilde,lefm->idfm',t2_1, eris_ovvo,optimize=True)
        M_ij += 4 * lib.einsum('idfm,jmdf->ij',temp_t2_v_6, t2_1,optimize = True)
        M_ij -= 2 * lib.einsum('idfm,mjdf->ij',temp_t2_v_6, t2_1,optimize = True)
        del temp_t2_v_6

        temp_t2_v_7 = lib.einsum('lide,lefm->idfm',t2_1, eris_ovvo,optimize=True)
        M_ij -= 2 * lib.einsum('idfm,jmdf->ij',temp_t2_v_7, t2_1,optimize = True)
        M_ij += lib.einsum('idfm,mjdf->ij',temp_t2_v_7, t2_1,optimize = True)
        del temp_t2_v_7

        temp_t2_v_8 = lib.einsum('lmdf,lmde->fe',t2_1, t2_1,optimize=True)
        M_ij += 3 *lib.einsum('fe,jief->ij',temp_t2_v_8, eris_oovv, optimize = True)
        M_ij -= 1.5 *lib.einsum('fe,jfei->ij',temp_t2_v_8, eris_ovvo, optimize = True)
        M_ij +=   lib.einsum('ef,jief->ij',temp_t2_v_8, eris_oovv, optimize = True)
        M_ij -= 0.5 * lib.einsum('ef,jfei->ij',temp_t2_v_8, eris_ovvo, optimize = True)
        del temp_t2_v_8

        temp_t2_v_9 = lib.einsum('lmdf,mlde->fe',t2_1, t2_1,optimize=True)
        M_ij -= 1.0 * lib.einsum('fe,jief->ij',temp_t2_v_9, eris_oovv, optimize = True)
        M_ij -= 1.0 * lib.einsum('ef,jief->ij',temp_t2_v_9, eris_oovv, optimize = True)
        M_ij += 0.5 * lib.einsum('fe,jfei->ij',temp_t2_v_9, eris_ovvo, optimize = True)
        M_ij += 0.5 * lib.einsum('ef,jfei->ij',temp_t2_v_9, eris_ovvo, optimize = True)
        del temp_t2_v_9

        temp_t2_v_10 = lib.einsum('lnde,lmde->nm',t2_1, t2_1,optimize=True)
        M_ij -= 3.0 * lib.einsum('nm,jinm->ij',temp_t2_v_10, eris_oooo, optimize = True)
        M_ij -= 1.0 * lib.einsum('mn,jinm->ij',temp_t2_v_10, eris_oooo, optimize = True)
        M_ij += 1.5 * lib.einsum('nm,jmni->ij',temp_t2_v_10, eris_oooo, optimize = True)
        M_ij += 0.5 * lib.einsum('mn,jmni->ij',temp_t2_v_10, eris_oooo, optimize = True)
        del temp_t2_v_10

        temp_t2_v_11 = lib.einsum('lnde,mlde->nm',t2_1, t2_1,optimize=True)
        M_ij += 1.0 * lib.einsum('nm,jinm->ij',temp_t2_v_11, eris_oooo, optimize = True)
        M_ij -= 0.5 * lib.einsum('nm,jmni->ij',temp_t2_v_11, eris_oooo, optimize = True)
        M_ij -= 0.5 * lib.einsum('mn,jmni->ij',temp_t2_v_11, eris_oooo, optimize = True)
        M_ij += 1.0 * lib.einsum('mn,jinm->ij',temp_t2_v_11, eris_oooo, optimize = True)
        del temp_t2_v_11

        temp_t2_v_12 = lib.einsum('inde,lmde->inlm',t2_1, t2_1,optimize=True)
        M_ij += 1.25 * lib.einsum('inlm,jlnm->ij',temp_t2_v_12, eris_oooo, optimize = True)
        M_ij += 0.25 * lib.einsum('lmin,jlnm->ij',temp_t2_v_12, eris_oooo, optimize = True)
        M_ij -= 0.25 * lib.einsum('inlm,jmnl->ij',temp_t2_v_12, eris_oooo, optimize = True)
        M_ij -= 0.25 * lib.einsum('lmin,jmnl->ij',temp_t2_v_12, eris_oooo, optimize = True)

        M_ij += 0.25 * lib.einsum('inlm,jlnm->ji',temp_t2_v_12, eris_oooo, optimize = True)
        M_ij -= 0.25 * lib.einsum('inlm,lnmj->ji',temp_t2_v_12, eris_oooo, optimize = True)
        M_ij += 1.00 * lib.einsum('inlm,ljmn->ji',temp_t2_v_12, eris_oooo, optimize = True)
        M_ij -= 0.25 * lib.einsum('lmin,lnmj->ji',temp_t2_v_12, eris_oooo, optimize = True)
        M_ij += 0.25 * lib.einsum('lmin,ljmn->ji',temp_t2_v_12, eris_oooo, optimize = True)
        del temp_t2_v_12

        temp_t2_v_13 = lib.einsum('inde,mlde->inml',t2_1, t2_1,optimize=True)
        M_ij -= 0.25 * lib.einsum('inml,jlnm->ij',temp_t2_v_13, eris_oooo, optimize = True)
        M_ij -= 0.25 * lib.einsum('mlin,jlnm->ij',temp_t2_v_13, eris_oooo, optimize = True)
        M_ij += 0.25 * lib.einsum('inml,jmnl->ij',temp_t2_v_13, eris_oooo, optimize = True)
        M_ij += 0.25 * lib.einsum('mlin,jmnl->ij',temp_t2_v_13, eris_oooo, optimize = True)

        M_ij -= 0.25 * lib.einsum('inml,jlnm->ji',temp_t2_v_13, eris_oooo, optimize = True)
        M_ij += 0.25 * lib.einsum('inml,lnmj->ji',temp_t2_v_13, eris_oooo, optimize = True)

        M_ij -= 0.25 * lib.einsum('inml,ljmn->ji',temp_t2_v_13, eris_oooo, optimize = True)
        M_ij += 0.25 * lib.einsum('inml,lnmj->ji',temp_t2_v_13, eris_oooo, optimize = True)
        del temp_t2_v_13
        del t2_1

    cput0 = log.timer_debug1("Completed M_ij ADC(n) calculation", *cput0)
    return M_ij


def ea_adc_diag(adc,M_ab=None,eris=None):

    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

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
    del D_iab

#    ###### Additional terms for the preconditioner ####
#
#    if (method == "adc(2)-x" or method == "adc(3)"):
#
#        if eris is None:
#            eris = adc.transform_integrals()
#
#        #TODO Implement this for out-of-core and density-fitted algorithms
#        if isinstance(eris.vvvv, np.ndarray):
#
#            eris_oovv = eris.oovv
#            eris_ovvo = eris.ovvo
#            eris_vvvv = eris.vvvv
#
#            temp = np.zeros((nocc, eris_vvvv.shape[0]))
#            temp[:] += np.diag(eris_vvvv)
#            diag[s2:f2] += temp.reshape(-1)
#
#            eris_ovov_p = np.ascontiguousarray(eris_oovv[:].transpose(0,2,1,3))
#            eris_ovov_p = eris_ovov_p.reshape(nocc*nvir, nocc*nvir)
#
#            temp = np.zeros((nvir, nocc, nvir))
#            temp[:] += np.diagonal(eris_ovov_p).reshape(nocc, nvir)
#            temp = np.ascontiguousarray(temp.transpose(1,0,2))
#            diag[s2:f2] += -temp.reshape(-1)
#
#            eris_ovov_p = np.ascontiguousarray(eris_oovv[:].transpose(0,2,1,3))
#            eris_ovov_p = eris_ovov_p.reshape(nocc*nvir, nocc*nvir)
#
#            temp = np.zeros((nvir, nocc, nvir))
#            temp[:] += np.diagonal(eris_ovov_p).reshape(nocc, nvir)
#            temp = np.ascontiguousarray(temp.transpose(1,2,0))
#            diag[s2:f2] += -temp.reshape(-1)
#        else:
#           raise Exception("Precond not available for out-of-core and density-fitted algo")

    log.timer_debug1("Completed ea_diag calculation")
    return diag


def ip_adc_diag(adc,M_ij=None,eris=None):

    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

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

#    ###### Additional terms for the preconditioner ####
#    if (method == "adc(2)-x" or method == "adc(3)"):
#
#        if eris is None:
#            eris = adc.transform_integrals()
#
#        if isinstance(eris.vvvv, np.ndarray):
#
#            eris_oooo = eris.oooo
#            eris_oovv = eris.oovv
#            eris_ovvo = eris.ovvo
#
#            eris_oooo_p = np.ascontiguousarray(eris_oooo.transpose(0,2,1,3))
#            eris_oooo_p = eris_oooo_p.reshape(nocc*nocc, nocc*nocc)
#
#            temp = np.zeros((nvir, eris_oooo_p.shape[0]))
#            temp[:] += np.diag(eris_oooo_p)
#            diag[s2:f2] += -temp.reshape(-1)
#
#            eris_ovov_p = np.ascontiguousarray(eris_oovv.transpose(0,2,1,3))
#            eris_ovov_p = eris_ovov_p.reshape(nocc*nvir, nocc*nvir)
#
#            temp = np.zeros((nocc, nocc, nvir))
#            temp[:] += np.diagonal(eris_ovov_p).reshape(nocc, nvir)
#            temp = np.ascontiguousarray(temp.transpose(2,1,0))
#            diag[s2:f2] += temp.reshape(-1)
#
#            eris_ovov_p = np.ascontiguousarray(eris_oovv.transpose(0,2,1,3))
#            eris_ovov_p = eris_ovov_p.reshape(nocc*nvir, nocc*nvir)
#
#            temp = np.zeros((nocc, nocc, nvir))
#            temp[:] += np.diagonal(eris_ovov_p).reshape(nocc, nvir)
#            temp = np.ascontiguousarray(temp.transpose(2,0,1))
#            diag[s2:f2] += temp.reshape(-1)
#        else:
#            raise Exception("Precond not available for out-of-core and density-fitted algo")

    diag = -diag
    log.timer_debug1("Completed ea_diag calculation")

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
            del dataset
            a += k
    elif getattr(myadc, 'with_df', None):
        for p in range(0,nvir,chnk_size):
            vvvv_p = dfadc.get_vvvv_df(myadc, vvvv, p, chnk_size)
            k = vvvv_p.shape[0]
            vvvv_p = vvvv_p.reshape(-1,nvir*nvir)
            r2_vvvv[:,a:a+k] = np.dot(r2,vvvv_p.T).reshape(nocc,-1,nvir)
            del vvvv_p
            a += k
    else:
        raise Exception("Unknown vvvv type")

    r2_vvvv = r2_vvvv.reshape(-1)

    return r2_vvvv


def ea_adc_matvec(adc, M_ab=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method


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
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(adc.stdout, adc.verbose)

        s = np.zeros((dim))

        r1 = r[s1:f1]
        r2 = r[s2:f2]

        r2 = r2.reshape(nocc,nvir,nvir)

############ ADC(2) ab block ############################

        s[s1:f1] = lib.einsum('ab,b->a',M_ab,r1)

############# ADC(2) a - ibc and ibc - a coupling blocks #########################

        if isinstance(eris.ovvv, type(None)):
            chnk_size = radc_ao2mo.calculate_chunk_size(adc)
        else:
            chnk_size = nocc
        a = 0
        temp_doubles = np.zeros((nocc,nvir,nvir))
        for p in range(0,nocc,chnk_size):
            if getattr(adc, 'with_df', None):
                eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
            else:
                eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
            k = eris_ovvv.shape[0]

            s[s1:f1] +=  2. * lib.einsum('icab,ibc->a', eris_ovvv, r2[a:a+k], optimize = True)
            s[s1:f1] -=  lib.einsum('ibac,ibc->a',   eris_ovvv, r2[a:a+k], optimize = True)

            temp_doubles[a:a+k] += lib.einsum('icab,a->ibc', eris_ovvv, r1, optimize = True)
            del eris_ovvv
            a += k

        s[s2:f2] +=  temp_doubles.reshape(-1)
################ ADC(2) iab - jcd block ############################

        s[s2:f2] +=  D_iab * r2.reshape(-1)

############### ADC(3) iab - jcd block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

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

            s[s2:f2] -= 0.5*lib.einsum('jzyi,jzx->ixy',eris_ovvo,r2,optimize = True).reshape(-1)
            s[s2:f2] += 0.5*lib.einsum('jzyi,jxz->ixy',eris_ovvo,r2,optimize = True).reshape(-1)

            s[s2:f2] -= 0.5*lib.einsum('jiyz,jxz->ixy',eris_oovv,r2,optimize = True).reshape(-1)
            s[s2:f2] += 0.5*lib.einsum('jzyi,jxz->ixy',eris_ovvo,r2,optimize = True).reshape(-1)
            s[s2:f2] -=  0.5*lib.einsum('jixz,jzy->ixy',eris_oovv,r2,optimize = True).reshape(-1)
            s[s2:f2] -=  0.5*lib.einsum('jixw,jwy->ixy',eris_oovv,r2,optimize = True).reshape(-1)
            s[s2:f2] -= 0.5*lib.einsum('jiyw,jxw->ixy',eris_oovv,r2,optimize = True).reshape(-1)
            s[s2:f2] += 0.5*lib.einsum('jwyi,jxw->ixy',eris_ovvo,r2,optimize = True).reshape(-1)
            s[s2:f2] += 0.5*lib.einsum('jwyi,jxw->ixy',eris_ovvo,r2,optimize = True).reshape(-1)
            s[s2:f2] -= 0.5*lib.einsum('jwyi,jwx->ixy',eris_ovvo,r2,optimize = True).reshape(-1)

            #print("Calculating additional terms for adc(3)")

        if (method == "adc(3)"):

            eris_ovoo = eris.ovoo

############### ADC(3) a - ibc block and ibc-a coupling blocks ########################

            t2_1 = adc.t2[0][:]

            temp =   0.25 * lib.einsum('lmab,jab->lmj',t2_1,r2)
            temp -=  0.25 * lib.einsum('lmab,jba->lmj',t2_1,r2)
            temp -=  0.25 * lib.einsum('mlab,jab->lmj',t2_1,r2)
            temp +=  0.25 * lib.einsum('mlab,jba->lmj',t2_1,r2)

            s[s1:f1] += lib.einsum('lmj,lamj->a',temp, eris_ovoo, optimize=True)
            s[s1:f1] -= lib.einsum('lmj,malj->a',temp, eris_ovoo, optimize=True)
            del temp

            temp_1 = -lib.einsum('lmzw,jzw->jlm',t2_1,r2)
            s[s1:f1] -= lib.einsum('jlm,lamj->a',temp_1, eris_ovoo, optimize=True)

            temp_s_a = lib.einsum('jlwd,jzw->lzd',t2_1,r2,optimize=True)
            temp_s_a -= lib.einsum('jlwd,jwz->lzd',t2_1,r2,optimize=True)
            temp_s_a -= lib.einsum('ljwd,jzw->lzd',t2_1,r2,optimize=True)
            temp_s_a += lib.einsum('ljwd,jwz->lzd',t2_1,r2,optimize=True)
            temp_s_a += lib.einsum('ljdw,jzw->lzd',t2_1,r2,optimize=True)

            temp_s_a_1 = -lib.einsum('jlzd,jwz->lwd',t2_1,r2,optimize=True)
            temp_s_a_1 += lib.einsum('jlzd,jzw->lwd',t2_1,r2,optimize=True)
            temp_s_a_1 += lib.einsum('ljzd,jwz->lwd',t2_1,r2,optimize=True)
            temp_s_a_1 -= lib.einsum('ljzd,jzw->lwd',t2_1,r2,optimize=True)
            temp_s_a_1 += -lib.einsum('ljdz,jwz->lwd',t2_1,r2,optimize=True)

            temp_t2_r2_1 = lib.einsum('jlwd,jzw->lzd',t2_1,r2,optimize=True)
            temp_t2_r2_1 -= lib.einsum('jlwd,jwz->lzd',t2_1,r2,optimize=True)
            temp_t2_r2_1 += lib.einsum('jlwd,jzw->lzd',t2_1,r2,optimize=True)
            temp_t2_r2_1 -= lib.einsum('ljwd,jzw->lzd',t2_1,r2,optimize=True)

            temp_t2_r2_2 = -lib.einsum('jlzd,jwz->lwd',t2_1,r2,optimize=True)
            temp_t2_r2_2 += lib.einsum('jlzd,jzw->lwd',t2_1,r2,optimize=True)
            temp_t2_r2_2 -= lib.einsum('jlzd,jwz->lwd',t2_1,r2,optimize=True)
            temp_t2_r2_2 += lib.einsum('ljzd,jwz->lwd',t2_1,r2,optimize=True)

            temp_t2_r2_3 = -lib.einsum('ljzd,jzw->lwd',t2_1,r2,optimize=True)

            temp_a = t2_1.transpose(0,3,1,2).copy()
            temp_b = temp_a.reshape(nocc*nvir,nocc*nvir)
            r2_t = r2.reshape(nocc*nvir,-1)
            temp_c = np.dot(temp_b,r2_t).reshape(nocc,nvir,nvir)
            temp_t2_r2_4 = temp_c.transpose(0,2,1).copy()

            del t2_1

            if isinstance(eris.ovvv, type(None)):
                chnk_size = radc_ao2mo.calculate_chunk_size(adc)
            else:
                chnk_size = nocc
            a = 0
            temp = np.zeros((nocc,nvir,nvir))
            temp_1_1 = np.zeros((nocc,nvir,nvir))
            temp_2_1 = np.zeros((nocc,nvir,nvir))
            for p in range(0,nocc,chnk_size):
                if getattr(adc, 'with_df', None):
                    eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
                else:
                    eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
                k = eris_ovvv.shape[0]

                temp_1_1[a:a+k] = lib.einsum('ldxb,b->lxd', eris_ovvv,r1,optimize=True)
                temp_1_1[a:a+k] -= lib.einsum('lbxd,b->lxd', eris_ovvv,r1,optimize=True)
                temp_2_1[a:a+k] = lib.einsum('ldxb,b->lxd', eris_ovvv,r1,optimize=True)

                s[s1:f1] += 0.5*lib.einsum('lzd,ldza->a',temp_s_a[a:a+k],eris_ovvv,optimize=True)
                s[s1:f1] -= 0.5*lib.einsum('lzd,lazd->a',temp_s_a[a:a+k],eris_ovvv,optimize=True)
                s[s1:f1] -= 0.5*lib.einsum('lwd,ldwa->a',temp_s_a_1[a:a+k],eris_ovvv,optimize=True)
                s[s1:f1] += 0.5*lib.einsum('lwd,lawd->a',temp_s_a_1[a:a+k],eris_ovvv,optimize=True)

                s[s1:f1] += 0.5*lib.einsum('lzd,ldza->a',temp_t2_r2_1[a:a+k],eris_ovvv,optimize=True)

                s[s1:f1] -= 0.5*lib.einsum('lwd,ldwa->a',temp_t2_r2_2[a:a+k],eris_ovvv,optimize=True)

                s[s1:f1] += 0.5*lib.einsum('lwd,lawd->a',temp_t2_r2_3[a:a+k],eris_ovvv,optimize=True)

                s[s1:f1] -= 0.5*lib.einsum('lzd,lazd->a',temp_t2_r2_4[a:a+k],eris_ovvv,optimize=True)

                temp[a:a+k]  -= lib.einsum('lbyd,b->lyd',eris_ovvv,r1,optimize=True)

                del eris_ovvv
                a += k

            t2_1 = adc.t2[0][:]
            temp_1 = -lib.einsum('lyd,lixd->ixy',temp,t2_1,optimize=True)
            s[s2:f2] -= temp_1.reshape(-1)

            del temp_s_a
            del temp_s_a_1
            del temp_t2_r2_1
            del temp_t2_r2_2
            del temp_t2_r2_3
            del temp_t2_r2_4

            temp_1 = lib.einsum('b,lbmi->lmi',r1,eris_ovoo)
            s[s2:f2] += lib.einsum('lmi,lmxy->ixy',temp_1, t2_1, optimize=True).reshape(-1)

            temp  = lib.einsum('lxd,lidy->ixy',temp_1_1,t2_1,optimize=True)
            temp  += lib.einsum('lxd,ilyd->ixy',temp_2_1,t2_1,optimize=True)
            temp  -= lib.einsum('lxd,ildy->ixy',temp_2_1,t2_1,optimize=True)
            s[s2:f2] += temp.reshape(-1)

            del t2_1
            del temp
            del temp_1
            del temp_1_1
            del temp_2_1

        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)
        return s

    return sigma_


def ip_adc_matvec(adc, M_ij=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method


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
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(adc.stdout, adc.verbose)

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

            eris_oooo = eris.oooo
            eris_oovv = eris.oovv
            eris_ovvo = eris.ovvo

            s[s2:f2] -= 0.5*lib.einsum('kijl,ali->ajk',eris_oooo, r2, optimize = True).reshape(-1)
            s[s2:f2] -= 0.5*lib.einsum('klji,ail->ajk',eris_oooo ,r2, optimize = True).reshape(-1)

            s[s2:f2] += 0.5*lib.einsum('klba,bjl->ajk',eris_oovv,r2,optimize = True).reshape(-1)

            s[s2:f2] +=  0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
            s[s2:f2] -=  0.5*lib.einsum('jabl,blk->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
            s[s2:f2] +=  0.5*lib.einsum('jlba,blk->ajk',eris_oovv,r2,optimize = True).reshape(-1)
            s[s2:f2] -=  0.5*lib.einsum('jabl,blk->ajk',eris_ovvo,r2,optimize = True).reshape(-1)

            s[s2:f2] += 0.5*lib.einsum('kiba,bji->ajk',eris_oovv,r2,optimize = True).reshape(-1)

            s[s2:f2] += 0.5*lib.einsum('jiba,bik->ajk',eris_oovv,r2,optimize = True).reshape(-1)
            s[s2:f2] -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
            s[s2:f2] -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
            s[s2:f2] += 0.5*lib.einsum('jabi,bki->ajk',eris_ovvo,r2,optimize = True).reshape(-1)

        if (method == "adc(3)"):

            eris_ovoo = eris.ovoo
            t2_1 = adc.t2[0]

################ ADC(3) i - kja block and ajk - i ############################

            temp =  0.25 * lib.einsum('ijbc,aij->abc',t2_1, r2, optimize=True)
            temp -= 0.25 * lib.einsum('ijbc,aji->abc',t2_1, r2, optimize=True)
            temp -= 0.25 * lib.einsum('jibc,aij->abc',t2_1, r2, optimize=True)
            temp += 0.25 * lib.einsum('jibc,aji->abc',t2_1, r2, optimize=True)

            temp_1 = lib.einsum('kjcb,ajk->abc',t2_1,r2, optimize=True)

            if isinstance(eris.ovvv, type(None)):
                chnk_size = radc_ao2mo.calculate_chunk_size(adc)
            else:
                chnk_size = nocc
            a = 0
            temp_singles = np.zeros((nocc))
            temp_doubles = np.zeros((nvir,nvir,nvir))
            for p in range(0,nocc,chnk_size):
                if getattr(adc, 'with_df', None):
                    eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
                else:
                    eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
                k = eris_ovvv.shape[0]

                temp_singles[a:a+k] += lib.einsum('abc,icab->i',temp, eris_ovvv, optimize=True)
                temp_singles[a:a+k] -= lib.einsum('abc,ibac->i',temp, eris_ovvv, optimize=True)
                temp_singles[a:a+k] += lib.einsum('abc,icab->i',temp_1, eris_ovvv, optimize=True)
                temp_doubles = lib.einsum('i,icab->cba',r1[a:a+k],eris_ovvv,optimize=True)
                s[s2:f2] += lib.einsum('cba,kjcb->ajk',temp_doubles, t2_1, optimize=True).reshape(-1)
                del eris_ovvv
                del temp_doubles
                a += k

            s[s1:f1] += temp_singles
            temp = np.zeros_like(r2)
            temp =  lib.einsum('jlab,ajk->blk',t2_1,r2,optimize=True)
            temp -= lib.einsum('jlab,akj->blk',t2_1,r2,optimize=True)
            temp -= lib.einsum('ljab,ajk->blk',t2_1,r2,optimize=True)
            temp += lib.einsum('ljab,akj->blk',t2_1,r2,optimize=True)
            temp += lib.einsum('ljba,ajk->blk',t2_1,r2,optimize=True)

            temp_1 = np.zeros_like(r2)
            temp_1 =  lib.einsum('jlab,ajk->blk',t2_1,r2,optimize=True)
            temp_1 -= lib.einsum('jlab,akj->blk',t2_1,r2,optimize=True)
            temp_1 += lib.einsum('jlab,ajk->blk',t2_1,r2,optimize=True)
            temp_1 -= lib.einsum('ljab,ajk->blk',t2_1,r2,optimize=True)

            temp_2 = lib.einsum('jlba,akj->blk',t2_1,r2, optimize=True)

            s[s1:f1] += 0.5*lib.einsum('blk,lbik->i',temp,eris_ovoo,optimize=True)
            s[s1:f1] -= 0.5*lib.einsum('blk,iblk->i',temp,eris_ovoo,optimize=True)
            s[s1:f1] += 0.5*lib.einsum('blk,lbik->i',temp_1,eris_ovoo,optimize=True)
            s[s1:f1] -= 0.5*lib.einsum('blk,iblk->i',temp_2,eris_ovoo,optimize=True)
            del temp
            del temp_1
            del temp_2

            temp = np.zeros_like(r2)
            temp = -lib.einsum('klab,akj->blj',t2_1,r2,optimize=True)
            temp += lib.einsum('klab,ajk->blj',t2_1,r2,optimize=True)
            temp += lib.einsum('lkab,akj->blj',t2_1,r2,optimize=True)
            temp -= lib.einsum('lkab,ajk->blj',t2_1,r2,optimize=True)
            temp -= lib.einsum('lkba,akj->blj',t2_1,r2,optimize=True)

            temp_1 = np.zeros_like(r2)
            temp_1  = -lib.einsum('klab,akj->blj',t2_1,r2,optimize=True)
            temp_1 += lib.einsum('klab,ajk->blj',t2_1,r2,optimize=True)
            temp_1 -= lib.einsum('klab,akj->blj',t2_1,r2,optimize=True)
            temp_1 += lib.einsum('lkab,akj->blj',t2_1,r2,optimize=True)

            temp_2 = -lib.einsum('klba,ajk->blj',t2_1,r2,optimize=True)

            s[s1:f1] -= 0.5*lib.einsum('blj,lbij->i',temp,eris_ovoo,optimize=True)
            s[s1:f1] += 0.5*lib.einsum('blj,iblj->i',temp,eris_ovoo,optimize=True)
            s[s1:f1] -= 0.5*lib.einsum('blj,lbij->i',temp_1,eris_ovoo,optimize=True)
            s[s1:f1] += 0.5*lib.einsum('blj,iblj->i',temp_2,eris_ovoo,optimize=True)

            del temp
            del temp_1
            del temp_2

            temp_1  = lib.einsum('i,lbik->kbl',r1,eris_ovoo)
            temp_1  -= lib.einsum('i,iblk->kbl',r1,eris_ovoo)
            temp_2  = lib.einsum('i,lbik->kbl',r1,eris_ovoo)

            temp  = lib.einsum('kbl,ljba->ajk',temp_1,t2_1,optimize=True)
            temp += lib.einsum('kbl,jlab->ajk',temp_2,t2_1,optimize=True)
            temp -= lib.einsum('kbl,ljab->ajk',temp_2,t2_1,optimize=True)
            s[s2:f2] += temp.reshape(-1)

            temp  = -lib.einsum('i,iblj->jbl',r1,eris_ovoo,optimize=True)
            temp_1 = -lib.einsum('jbl,klba->ajk',temp,t2_1,optimize=True)
            s[s2:f2] -= temp_1.reshape(-1)

            del temp
            del temp_1
            del temp_2
            del t2_1

        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)
        s *= -1.0

        return s

    return sigma_


def ea_compute_trans_moments(adc, orb):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2_1 = adc.t2[0][:]
    t1_2 = adc.t1[0][:]

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

        t2_1_t = -t2_1.transpose(1,0,2,3)

        T[s2:f2] += t2_1_t[:,orb,:,:].reshape(-1)

    else:

        T[s1:f1] += idn_vir[(orb-nocc), :]
        T[s1:f1] -= 0.25*lib.einsum('klc,klac->a',t2_1[:,:,(orb-nocc),:], t2_1, optimize = True)
        T[s1:f1] -= 0.25*lib.einsum('lkc,lkac->a',t2_1[:,:,(orb-nocc),:], t2_1, optimize = True)

        T[s1:f1] -= 0.25*lib.einsum('klc,klac->a',t2_1[:,:,(orb-nocc),:], t2_1, optimize = True)
        T[s1:f1] += 0.25*lib.einsum('lkc,klac->a',t2_1[:,:,(orb-nocc),:], t2_1, optimize = True)
        T[s1:f1] += 0.25*lib.einsum('klc,lkac->a',t2_1[:,:,(orb-nocc),:], t2_1, optimize = True)
        T[s1:f1] -= 0.25*lib.einsum('lkc,lkac->a',t2_1[:,:,(orb-nocc),:], t2_1, optimize = True)

######### ADC(3) 2p-1h  part  ############################################

    if(method=="adc(2)-x"or adc.method=="adc(3)"):

        t2_2 = adc.t2[1][:]

        if orb < nocc:

            t2_2_t = -t2_2.transpose(1,0,2,3)

            T[s2:f2] += t2_2_t[:,orb,:,:].reshape(-1)

########### ADC(3) 1p part  ############################################

    if(adc.method=="adc(3)"):

        t1_3 = adc.t1[1]

        if orb < nocc:
            T[s1:f1] += 0.5*lib.einsum('kac,ck->a',t2_1[:,orb,:,:], t1_2.T,optimize = True)
            T[s1:f1] -= 0.5*lib.einsum('kac,ck->a',t2_1[orb,:,:,:], t1_2.T,optimize = True)
            T[s1:f1] -= 0.5*lib.einsum('kac,ck->a',t2_1[orb,:,:,:], t1_2.T,optimize = True)
            T[s1:f1] -= t1_3[orb,:]

        else:

            T[s1:f1] -= 0.25*lib.einsum('klc,klac->a',t2_1[:,:,(orb-nocc),:], t2_2, optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('lkc,lkac->a',t2_1[:,:,(orb-nocc),:], t2_2, optimize = True)

            T[s1:f1] -= 0.25*lib.einsum('klac,klc->a',t2_1, t2_2[:,:,(orb-nocc),:],optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('lkac,lkc->a',t2_1, t2_2[:,:,(orb-nocc),:],optimize = True)

            T[s1:f1] -= 0.25*lib.einsum('klc,klac->a',t2_1[:,:,(orb-nocc),:], t2_2, optimize = True)
            T[s1:f1] += 0.25*lib.einsum('klc,lkac->a',t2_1[:,:,(orb-nocc),:], t2_2, optimize = True)
            T[s1:f1] += 0.25*lib.einsum('lkc,klac->a',t2_1[:,:,(orb-nocc),:], t2_2, optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('lkc,lkac->a',t2_1[:,:,(orb-nocc),:], t2_2, optimize = True)

            T[s1:f1] -= 0.25*lib.einsum('klac,klc->a',t2_1, t2_2[:,:,(orb-nocc),:],optimize = True)
            T[s1:f1] += 0.25*lib.einsum('klac,lkc->a',t2_1, t2_2[:,:,(orb-nocc),:],optimize = True)
            T[s1:f1] += 0.25*lib.einsum('lkac,klc->a',t2_1, t2_2[:,:,(orb-nocc),:],optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('lkac,lkc->a',t2_1, t2_2[:,:,(orb-nocc),:],optimize = True)

        del t2_2
    del t2_1

    T_aaa = T[n_singles:].reshape(nocc,nvir,nvir).copy()
    T_aaa = T_aaa - T_aaa.transpose(0,2,1)
    T[n_singles:] += T_aaa.reshape(-1)

    return T


def ip_compute_trans_moments(adc, orb):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2_1 = adc.t2[0][:]
    t1_2 = adc.t1[0][:]

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
        T[s1:f1] += 0.25*lib.einsum('kdc,ikdc->i',t2_1[:,orb,:,:], t2_1, optimize = True)
        T[s1:f1] -= 0.25*lib.einsum('kcd,ikdc->i',t2_1[:,orb,:,:], t2_1, optimize = True)
        T[s1:f1] -= 0.25*lib.einsum('kdc,ikcd->i',t2_1[:,orb,:,:], t2_1, optimize = True)
        T[s1:f1] += 0.25*lib.einsum('kcd,ikcd->i',t2_1[:,orb,:,:], t2_1, optimize = True)
        T[s1:f1] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1[orb,:,:,:], t2_1, optimize = True)
        T[s1:f1] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1[orb,:,:,:], t2_1, optimize = True)
    else:
        T[s1:f1] += t1_2[:,(orb-nocc)]

######## ADC(2) 2h-1p  part  ############################################

        t2_1_t = t2_1.transpose(2,3,1,0)

        T[s2:f2] = t2_1_t[(orb-nocc),:,:,:].reshape(-1)

######## ADC(3) 2h-1p  part  ############################################

    if(method=='adc(2)-x'or method=='adc(3)'):

        t2_2 = adc.t2[1][:]

        if orb >= nocc:
            t2_2_t = t2_2.transpose(2,3,1,0)

            T[s2:f2] += t2_2_t[(orb-nocc),:,:,:].reshape(-1)

######### ADC(3) 1h part  ############################################

    if(method=='adc(3)'):

        t1_3 = adc.t1[1]

        if orb < nocc:
            T[s1:f1] += 0.25*lib.einsum('kdc,ikdc->i',t2_1[:,orb,:,:], t2_2, optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('kcd,ikdc->i',t2_1[:,orb,:,:], t2_2, optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('kdc,ikcd->i',t2_1[:,orb,:,:], t2_2, optimize = True)
            T[s1:f1] += 0.25*lib.einsum('kcd,ikcd->i',t2_1[:,orb,:,:], t2_2, optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1[orb,:,:,:], t2_2, optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1[orb,:,:,:], t2_2, optimize = True)

            T[s1:f1] += 0.25*lib.einsum('ikdc,kdc->i',t2_1, t2_2[:,orb,:,:],optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('ikcd,kdc->i',t2_1, t2_2[:,orb,:,:],optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('ikdc,kcd->i',t2_1, t2_2[:,orb,:,:],optimize = True)
            T[s1:f1] += 0.25*lib.einsum('ikcd,kcd->i',t2_1, t2_2[:,orb,:,:],optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('ikcd,kcd->i',t2_1, t2_2[orb,:,:,:],optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('ikdc,kdc->i',t2_1, t2_2[orb,:,:,:],optimize = True)
        else:
            T[s1:f1] += 0.5*lib.einsum('ikc,kc->i',t2_1[:,:,(orb-nocc),:], t1_2,optimize = True)
            T[s1:f1] -= 0.5*lib.einsum('kic,kc->i',t2_1[:,:,(orb-nocc),:], t1_2,optimize = True)
            T[s1:f1] += 0.5*lib.einsum('ikc,kc->i',t2_1[:,:,(orb-nocc),:], t1_2,optimize = True)
            T[s1:f1] += t1_3[:,(orb-nocc)]

        del t2_2
    del t2_1

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


def analyze_eigenvector_ea(adc):

    nocc = adc._nocc
    nvir = adc._nvir
    evec_print_tol = adc.evec_print_tol

    logger.info(adc, "Number of occupied orbitals = %d", nocc)
    logger.info(adc, "Number of virtual orbitals =  %d", nvir)
    logger.info(adc, "Print eigenvector elements > %f\n", evec_print_tol)

    n_singles = nvir
    U = adc.U

    for I in range(U.shape[1]):
        U1 = U[:n_singles,I]
        U2 = U[n_singles:,I].reshape(nocc,nvir,nvir)
        U1dotU1 = np.dot(U1, U1)
        U2dotU2 =  2.*np.dot(U2.ravel(), U2.ravel()) - np.dot(U2.ravel(), U2.transpose(0,2,1).ravel())

        U_sq = U[:,I].copy()**2
        ind_idx = np.argsort(-U_sq)
        U_sq = U_sq[ind_idx]
        U_sorted = U[ind_idx,I].copy()

        U_sorted = U_sorted[U_sq > evec_print_tol**2]
        ind_idx = ind_idx[U_sq > evec_print_tol**2]

        singles_idx = []
        doubles_idx = []
        singles_val = []
        doubles_val = []
        iter_num = 0

        for orb_idx in ind_idx:

            if orb_idx < n_singles:
                a_idx = orb_idx + 1 + nocc
                singles_idx.append(a_idx)
                singles_val.append(U_sorted[iter_num])

            if orb_idx >= n_singles:
                iab_idx = orb_idx - n_singles
                ab_rem = iab_idx % (nvir*nvir)
                i_idx = iab_idx //(nvir*nvir)
                a_idx = ab_rem//nvir
                b_idx = ab_rem % nvir
                doubles_idx.append((i_idx + 1, a_idx + 1 + nocc, b_idx + 1 + nocc))
                doubles_val.append(U_sorted[iter_num])

            iter_num += 1

        logger.info(adc, '%s | root %d | norm(1p)  = %6.4f | norm(1h2p) = %6.4f ',
                    adc.method ,I, U1dotU1, U2dotU2)

        if singles_val:
            logger.info(adc, "\n1p block: ")
            logger.info(adc, "     a     U(a)")
            logger.info(adc, "------------------")
            for idx, print_singles in enumerate(singles_idx):
                logger.info(adc, '  %4d   %7.4f', print_singles, singles_val[idx])

        if doubles_val:
            logger.info(adc, "\n1h2p block: ")
            logger.info(adc, "     i     a     b     U(i,a,b)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f',
                            print_doubles[0], print_doubles[1], print_doubles[2], doubles_val[idx])

        logger.info(adc, "\n*************************************************************\n")


def analyze_eigenvector_ip(adc):

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc
    evec_print_tol = adc.evec_print_tol
    U = adc.U

    logger.info(adc, "Number of occupied orbitals = %d", nocc)
    logger.info(adc, "Number of virtual orbitals =  %d", nvir)
    logger.info(adc, "Print eigenvector elements > %f\n", evec_print_tol)

    for I in range(U.shape[1]):
        U1 = U[:n_singles,I]
        U2 = U[n_singles:,I].reshape(nvir,nocc,nocc)
        U1dotU1 = np.dot(U1, U1)
        U2dotU2 =  2.*np.dot(U2.ravel(), U2.ravel()) - np.dot(U2.ravel(), U2.transpose(0,2,1).ravel())

        U_sq = U[:,I].copy()**2
        ind_idx = np.argsort(-U_sq)
        U_sq = U_sq[ind_idx]
        U_sorted = U[ind_idx,I].copy()

        U_sorted = U_sorted[U_sq > evec_print_tol**2]
        ind_idx = ind_idx[U_sq > evec_print_tol**2]

        singles_idx = []
        doubles_idx = []
        singles_val = []
        doubles_val = []
        iter_num = 0

        for orb_idx in ind_idx:

            if orb_idx < n_singles:
                i_idx = orb_idx + 1
                singles_idx.append(i_idx)
                singles_val.append(U_sorted[iter_num])

            if orb_idx >= n_singles:
                aij_idx = orb_idx - n_singles
                ij_rem = aij_idx % (nocc*nocc)
                a_idx = aij_idx//(nocc*nocc)
                i_idx = ij_rem//nocc
                j_idx = ij_rem % nocc
                doubles_idx.append((a_idx + 1 + n_singles, i_idx + 1, j_idx + 1))
                doubles_val.append(U_sorted[iter_num])

            iter_num += 1

        logger.info(adc, '%s | root %d | norm(1h)  = %6.4f | norm(2h1p) = %6.4f ',
                    adc.method ,I, U1dotU1, U2dotU2)

        if singles_val:
            logger.info(adc, "\n1h block: ")
            logger.info(adc, "     i     U(i)")
            logger.info(adc, "------------------")
            for idx, print_singles in enumerate(singles_idx):
                logger.info(adc, '  %4d   %7.4f', print_singles, singles_val[idx])

        if doubles_val:
            logger.info(adc, "\n2h1p block: ")
            logger.info(adc, "     i     j     a     U(i,j,a)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f',
                            print_doubles[1], print_doubles[2], print_doubles[0], doubles_val[idx])

        logger.info(adc, "\n*************************************************************\n")


def analyze_spec_factor(adc):

    X = adc.X
    X_2 = (X.copy()**2)*2
    thresh = adc.spec_factor_print_tol

    logger.info(adc, "Print spectroscopic factors > %E\n", adc.spec_factor_print_tol)

    for i in range(X_2.shape[1]):

        sort = np.argsort(-X_2[:,i])
        X_2_row = X_2[:,i]
        X_2_row = X_2_row[sort]

        if not adc.mol.symmetry:
            sym = np.repeat(['A'], X_2_row.shape[0])
        else:
            sym = [symm.irrep_id2name(adc.mol.groupname, x) for x in adc._scf.mo_coeff.orbsym]
            sym = np.array(sym)

            sym = sym[sort]

        spec_Contribution = X_2_row[X_2_row > thresh]
        index_mo = sort[X_2_row > thresh]+1

        if np.sum(spec_Contribution) == 0.0:
            continue

        logger.info(adc,'%s | root %d \n',adc.method ,i)
        logger.info(adc, "     HF MO     Spec. Contribution     Orbital symmetry")
        logger.info(adc, "-----------------------------------------------------------")

        for c in range(index_mo.shape[0]):
            logger.info(adc, '     %3.d          %10.8f                %s', index_mo[c], spec_Contribution[c], sym[c])

        logger.info(adc, '\nPartial spec. factor sum = %10.8f', np.sum(spec_Contribution))
        logger.info(adc, "\n*************************************************************\n")


def renormalize_eigenvectors_ea(adc, nroots=1):

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nvir

    U = adc.U

    for I in range(U.shape[1]):
        U1 = U[:n_singles,I]
        U2 = U[n_singles:,I].reshape(nocc,nvir,nvir)
        UdotU = np.dot(U1, U1) + 2.*np.dot(U2.ravel(), U2.ravel()) - np.dot(U2.ravel(), U2.transpose(0,2,1).ravel())
        U[:,I] /= np.sqrt(UdotU)

    return U


def renormalize_eigenvectors_ip(adc, nroots=1):

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc

    U = adc.U

    for I in range(U.shape[1]):
        U1 = U[:n_singles,I]
        U2 = U[n_singles:,I].reshape(nvir,nocc,nocc)
        UdotU = np.dot(U1, U1) + 2.*np.dot(U2.ravel(), U2.ravel()) - np.dot(U2.ravel(), U2.transpose(0,2,1).ravel())
        U[:,I] /= np.sqrt(UdotU)

    return U


def get_properties(adc, nroots=1):

    #Transition moments
    T = adc.get_trans_moments()

    #Spectroscopic amplitudes
    U = adc.renormalize_eigenvectors(nroots)
    X = np.dot(T, U).reshape(-1, nroots)

    #Spectroscopic factors
    P = 2.0*lib.einsum("pi,pi->i", X, X)

    return P,X


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
        self._nmo = adc._nmo
        self.mo_coeff = adc.mo_coeff
        self.mo_energy = adc.mo_energy
        self.nmo = adc._nmo
        self.transform_integrals = adc.transform_integrals
        self.with_df = adc.with_df
        self.compute_properties = adc.compute_properties
        self.E = None
        self.U = None
        self.P = None
        self.X = None
        self.evec_print_tol = adc.evec_print_tol
        self.spec_factor_print_tol = adc.spec_factor_print_tol

        keys = set(('tol_residual','conv_tol', 'e_corr', 'method', 'mo_coeff',
                    'mo_energy', 'max_memory', 't1', 'max_space', 't2',
                    'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)

    kernel = kernel
    get_imds = get_imds_ea
    matvec = ea_adc_matvec
    get_diag = ea_adc_diag
    compute_trans_moments = ea_compute_trans_moments
    get_trans_moments = get_trans_moments
    renormalize_eigenvectors = renormalize_eigenvectors_ea
    get_properties = get_properties
    analyze_spec_factor = analyze_spec_factor
    analyze_eigenvector = analyze_eigenvector_ea
    analyze = analyze
    compute_dyson_mo = compute_dyson_mo

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
        self._nmo = adc._nmo
        self.mo_coeff = adc.mo_coeff
        self.mo_energy = adc.mo_energy
        self.nmo = adc._nmo
        self.transform_integrals = adc.transform_integrals
        self.with_df = adc.with_df
        self.compute_properties = adc.compute_properties
        self.E = None
        self.U = None
        self.P = None
        self.X = None
        self.evec_print_tol = adc.evec_print_tol
        self.spec_factor_print_tol = adc.spec_factor_print_tol

        keys = set(('tol_residual','conv_tol', 'e_corr', 'method', 'mo_coeff',
                    'mo_energy_b', 'max_memory', 't1', 'mo_energy_a',
                    'max_space', 't2', 'max_cycle'))
        self._keys = set(self.__dict__.keys()).union(keys)

    kernel = kernel
    get_imds = get_imds_ip
    get_diag = ip_adc_diag
    matvec = ip_adc_matvec
    compute_trans_moments = ip_compute_trans_moments
    get_trans_moments = get_trans_moments
    renormalize_eigenvectors = renormalize_eigenvectors_ip
    get_properties = get_properties
    analyze_spec_factor = analyze_spec_factor
    analyze_eigenvector = analyze_eigenvector_ip
    analyze = analyze
    compute_dyson_mo = compute_dyson_mo

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
