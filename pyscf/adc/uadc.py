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
        eris = uadc_ao2mo.transform_integrals(adc)

    imds = adc.get_imds(eris)
    matvec, diag = adc.gen_matvec(imds, eris)

    guess = adc.get_init_guess(nroots, diag, ascending = True)

    E, U = lib.linalg_helper.davidson(matvec, guess, diag, nroots=nroots, verbose=log, max_cycle=adc.max_cycle, max_space=adc.max_space)

    T_a, T_b = adc.get_trans_moments(nroots, eris)

    spec_factors = adc.get_spec_factors(nroots, (T_a,T_b), U)

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

    t2_2 = (None,)
    t1_3 = (None,)
    nocc_a = myadc._nocc[0]
    nocc_b = myadc._nocc[1]  
    nvir_a = myadc._nvir[0] 
    nvir_b = myadc._nvir[1]

    v2e_oovv_a,v2e_oovv_ab,v2e_oovv_b  = eris.oovv
    v2e_vvvv_a,v2e_vvvv_ab,v2e_vvvv_b  = eris.vvvv
    v2e_oooo_a,v2e_oooo_ab,v2e_oooo_b  = eris.oooo
    v2e_voov_a,v2e_voov_ab,v2e_voov_b  = eris.voov
    v2e_ooov_a,v2e_ooov_ab,v2e_ooov_b  = eris.ooov
    v2e_vovv_a,v2e_vovv_ab,v2e_vovv_b  = eris.vovv
    v2e_vvoo_a,v2e_vvoo_ab,v2e_vvoo_b  = eris.vvoo
    v2e_oovo_a,v2e_oovo_ab,v2e_oovo_b  = eris.oovo
    v2e_ovov_a,v2e_ovov_ab,v2e_ovov_b  = eris.ovov
    v2e_vovo_a,v2e_vovo_ab,v2e_vovo_b  = eris.vovo
    v2e_vvvo_a,v2e_vvvo_ab,v2e_vvvo_b  = eris.vvvo
    v2e_vvov_a,v2e_vvov_ab,v2e_vvov_b  = eris.vvov
    v2e_vooo_a,v2e_vooo_ab,v2e_vooo_b  = eris.vooo
    v2e_ovoo_a,v2e_ovoo_ab,v2e_ovoo_b  = eris.ovoo
    v2e_ovvv_a,v2e_ovvv_ab,v2e_ovvv_b  = eris.ovvv
    v2e_oovo_a,v2e_oovo_ab,v2e_oovo_b  = eris.oovo
    v2e_ovvo_a,v2e_ovvo_ab,v2e_ovvo_b  = eris.ovvo

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

    t2_1_a = v2e_oovv_a/D2_a
    t2_1_b = v2e_oovv_b/D2_b
    t2_1_ab = v2e_oovv_ab/D2_ab

    t2_1 = (t2_1_a , t2_1_ab, t2_1_b)

    # Compute second-order singles t1 (tij) 

    t1_2_a = 0.5*np.einsum('akcd,ikcd->ia',v2e_vovv_a,t2_1_a)
    t1_2_a -= 0.5*np.einsum('klic,klac->ia',v2e_ooov_a,t2_1_a)
    t1_2_a += np.einsum('akcd,ikcd->ia',v2e_vovv_ab,t2_1_ab)
    t1_2_a -= np.einsum('klic,klac->ia',v2e_ooov_ab,t2_1_ab)

    t1_2_b = 0.5*np.einsum('akcd,ikcd->ia',v2e_vovv_b,t2_1_b)
    t1_2_b -= 0.5*np.einsum('klic,klac->ia',v2e_ooov_b,t2_1_b)
    t1_2_b += np.einsum('kadc,kidc->ia',v2e_ovvv_ab,t2_1_ab)
    t1_2_b -= np.einsum('lkci,lkca->ia',v2e_oovo_ab,t2_1_ab)

    t1_2_a = t1_2_a/D1_a
    t1_2_b = t1_2_b/D1_b

    t1_2 = (t1_2_a , t1_2_b)

    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):

    # Compute second-order doubles t2 (tijab) 

        temp = t2_1_a.reshape(nocc_a*nocc_a,nvir_a*nvir_a)
        temp_1 = v2e_vvvv_a[:].reshape(nvir_a*nvir_a,nvir_a*nvir_a)
        t2_2_a = 0.5*np.dot(temp,temp_1.T).reshape(nocc_a,nocc_a,nvir_a,nvir_a)
        del temp_1
        t2_2_a += 0.5*np.einsum('klij,klab->ijab',v2e_oooo_a,t2_1_a,optimize=True)
 
        temp = np.einsum('bkjc,kica->ijab',v2e_voov_a,t2_1_a,optimize=True)
        temp_1 = np.einsum('bkjc,ikac->ijab',v2e_voov_ab,t2_1_ab,optimize=True)
 
        t2_2_a += temp - temp.transpose(1,0,2,3) - temp.transpose(0,1,3,2) + temp.transpose(1,0,3,2)
        t2_2_a += temp_1 - temp_1.transpose(1,0,2,3) - temp_1.transpose(0,1,3,2) + temp_1.transpose(1,0,3,2)
 
        temp = t2_1_b.reshape(nocc_b*nocc_b,nvir_b*nvir_b)
        temp_1 = v2e_vvvv_b[:].reshape(nvir_b*nvir_b,nvir_b*nvir_b)
        t2_2_b = 0.5*np.dot(temp,temp_1.T).reshape(nocc_b,nocc_b,nvir_b,nvir_b)
        del temp_1
        t2_2_b += 0.5*np.einsum('klij,klab->ijab',v2e_oooo_b,t2_1_b,optimize=True)
 
        temp = np.einsum('bkjc,kica->ijab',v2e_voov_b,t2_1_b,optimize=True)
        temp_1 = np.einsum('kbcj,kica->ijab',v2e_ovvo_ab,t2_1_ab,optimize=True)
 
        t2_2_b += temp - temp.transpose(1,0,2,3) - temp.transpose(0,1,3,2) + temp.transpose(1,0,3,2)
        t2_2_b += temp_1 - temp_1.transpose(1,0,2,3) - temp_1.transpose(0,1,3,2) + temp_1.transpose(1,0,3,2)
 
        temp = t2_1_ab.reshape(nocc_a*nocc_b,nvir_a*nvir_b)
        temp_1 = v2e_vvvv_ab[:].reshape(nvir_a*nvir_b,nvir_a*nvir_b)
        t2_2_ab = np.dot(temp,temp_1.T).reshape(nocc_a,nocc_b,nvir_a,nvir_b)
        del temp_1
        t2_2_ab += np.einsum('klij,klab->ijab',v2e_oooo_ab,t2_1_ab,optimize=True)
        t2_2_ab += np.einsum('kbcj,kica->ijab',v2e_ovvo_ab,t2_1_a,optimize=True)
        t2_2_ab += np.einsum('bkjc,ikac->ijab',v2e_voov_b,t2_1_ab,optimize=True)
        t2_2_ab -= np.einsum('kbic,kjac->ijab',v2e_ovov_ab,t2_1_ab,optimize=True)
        t2_2_ab -= np.einsum('akcj,ikcb->ijab',v2e_vovo_ab,t2_1_ab,optimize=True)
        t2_2_ab += np.einsum('akic,kjcb->ijab',v2e_voov_ab,t2_1_b,optimize=True)
        t2_2_ab += np.einsum('akic,kjcb->ijab',v2e_voov_a,t2_1_ab,optimize=True)
 
        t2_2_a = t2_2_a/D2_a
        t2_2_b = t2_2_b/D2_b
        t2_2_ab = t2_2_ab/D2_ab
 
        t2_2 = (t2_2_a , t2_2_ab, t2_2_b)

    if (myadc.method == "adc(3)"):
    # Compute third-order singles (tij)

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
 
        t1_3_a += np.einsum('ld,adil->ia',t1_2_a,v2e_vvoo_a ,optimize=True)
        t1_3_a += np.einsum('ld,adil->ia',t1_2_b,v2e_vvoo_ab,optimize=True)
 
        t1_3_b += np.einsum('ld,adil->ia',t1_2_b,v2e_vvoo_b ,optimize=True)
        t1_3_b += np.einsum('ld,dali->ia',t1_2_a,v2e_vvoo_ab,optimize=True)
 
        t1_3_a += np.einsum('ld,alid->ia',t1_2_a,v2e_voov_a ,optimize=True)
        t1_3_a += np.einsum('ld,alid->ia',t1_2_b,v2e_voov_ab,optimize=True)
 
        t1_3_b += np.einsum('ld,alid->ia',t1_2_b,v2e_voov_b ,optimize=True)
        t1_3_b += np.einsum('ld,ladi->ia',t1_2_a,v2e_ovvo_ab,optimize=True)
 
        t1_3_a -= 0.5*np.einsum('lmad,lmid->ia',t2_2_a,v2e_ooov_a,optimize=True)
        t1_3_a -=     np.einsum('lmad,lmid->ia',t2_2_ab,v2e_ooov_ab,optimize=True)
 
        t1_3_b -= 0.5*np.einsum('lmad,lmid->ia',t2_2_b,v2e_ooov_b,optimize=True)
        t1_3_b -=     np.einsum('mlda,mldi->ia',t2_2_ab,v2e_oovo_ab,optimize=True)
 
        t1_3_a += 0.5*np.einsum('ilde,alde->ia',t2_2_a,v2e_vovv_a,optimize=True)
        t1_3_a += np.einsum('ilde,alde->ia',t2_2_ab,v2e_vovv_ab,optimize=True)
 
        t1_3_b += 0.5*np.einsum('ilde,alde->ia',t2_2_b,v2e_vovv_b,optimize=True)
        t1_3_b += np.einsum('lied,laed->ia',t2_2_ab,v2e_ovvv_ab,optimize=True)
 
        t1_3_a -= np.einsum('ildf,aefm,lmde->ia',t2_1_a,v2e_vvvo_a,  t2_1_a ,optimize=True)
        t1_3_a += np.einsum('ilfd,aefm,mled->ia',t2_1_ab,v2e_vvvo_a, t2_1_ab,optimize=True)
        t1_3_a -= np.einsum('ildf,aefm,lmde->ia',t2_1_a,v2e_vvvo_ab, t2_1_ab,optimize=True)
        t1_3_a += np.einsum('ilfd,aefm,lmde->ia',t2_1_ab,v2e_vvvo_ab,t2_1_b ,optimize=True)
        t1_3_a -= np.einsum('ildf,aemf,mlde->ia',t2_1_ab,v2e_vvov_ab,t2_1_ab,optimize=True)
 
        t1_3_b -= np.einsum('ildf,aefm,lmde->ia',t2_1_b,v2e_vvvo_b,t2_1_b,optimize=True)
        t1_3_b += np.einsum('lidf,aefm,lmde->ia',t2_1_ab,v2e_vvvo_b,t2_1_ab,optimize=True)
        t1_3_b -= np.einsum('ildf,eamf,mled->ia',t2_1_b,v2e_vvov_ab,t2_1_ab,optimize=True)
        t1_3_b += np.einsum('lidf,eamf,lmde->ia',t2_1_ab,v2e_vvov_ab,t2_1_a,optimize=True)
        t1_3_b -= np.einsum('lifd,eafm,lmed->ia',t2_1_ab,v2e_vvvo_ab,t2_1_ab,optimize=True)
 
        t1_3_a += 0.5*np.einsum('ilaf,defm,lmde->ia',t2_1_a,v2e_vvvo_a,t2_1_a,optimize=True)
        t1_3_a += 0.5*np.einsum('ilaf,defm,lmde->ia',t2_1_ab,v2e_vvvo_b,t2_1_b,optimize=True)
        t1_3_a += np.einsum('ilaf,edmf,mled->ia',t2_1_ab,v2e_vvov_ab,t2_1_ab,optimize=True)
        t1_3_a += np.einsum('ilaf,defm,lmde->ia',t2_1_a,v2e_vvvo_ab,t2_1_ab,optimize=True)
 
        t1_3_b += 0.5*np.einsum('ilaf,defm,lmde->ia',t2_1_b,v2e_vvvo_b,t2_1_b,optimize=True)
        t1_3_b += 0.5*np.einsum('lifa,defm,lmde->ia',t2_1_ab,v2e_vvvo_a,t2_1_a,optimize=True)
        t1_3_b += np.einsum('lifa,defm,lmde->ia',t2_1_ab,v2e_vvvo_ab,t2_1_ab,optimize=True)
        t1_3_b += np.einsum('ilaf,edmf,mled->ia',t2_1_b,v2e_vvov_ab,t2_1_ab,optimize=True)
 
        t1_3_a += 0.25*np.einsum('inde,anlm,lmde->ia',t2_1_a,v2e_vooo_a,t2_1_a,optimize=True)
        t1_3_a += np.einsum('inde,anlm,lmde->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab,optimize=True)
 
        t1_3_b += 0.25*np.einsum('inde,anlm,lmde->ia',t2_1_b,v2e_vooo_b,t2_1_b,optimize=True)
        t1_3_b += np.einsum('nied,naml,mled->ia',t2_1_ab,v2e_ovoo_ab,t2_1_ab,optimize=True)
 
        t1_3_a += 0.5*np.einsum('inad,enlm,lmde->ia',t2_1_a,v2e_vooo_a,t2_1_a,optimize=True)
        t1_3_a -= 0.5 * np.einsum('inad,neml,mlde->ia',t2_1_a,v2e_ovoo_ab,t2_1_ab,optimize=True)
        t1_3_a -= 0.5 * np.einsum('inad,nelm,lmde->ia',t2_1_a,v2e_ovoo_ab,t2_1_ab,optimize=True)
        t1_3_a -= 0.5 *np.einsum('inad,enlm,lmed->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab,optimize=True)
        t1_3_a -= 0.5*np.einsum('inad,enml,mled->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab,optimize=True)
        t1_3_a += 0.5*np.einsum('inad,enlm,lmde->ia',t2_1_ab,v2e_vooo_b,t2_1_b,optimize=True)
 
        t1_3_b += 0.5*np.einsum('inad,enlm,lmde->ia',t2_1_b,v2e_vooo_b,t2_1_b,optimize=True)
        t1_3_b -= 0.5 * np.einsum('inad,enml,mled->ia',t2_1_b,v2e_vooo_ab,t2_1_ab,optimize=True)
        t1_3_b -= 0.5 * np.einsum('inad,enlm,lmed->ia',t2_1_b,v2e_vooo_ab,t2_1_ab,optimize=True)
        t1_3_b -= 0.5 *np.einsum('nida,nelm,lmde->ia',t2_1_ab,v2e_ovoo_ab,t2_1_ab,optimize=True)
        t1_3_b -= 0.5*np.einsum('nida,neml,mlde->ia',t2_1_ab,v2e_ovoo_ab,t2_1_ab,optimize=True)
        t1_3_b += 0.5*np.einsum('nida,enlm,lmde->ia',t2_1_ab,v2e_vooo_a,t2_1_a,optimize=True)
 
        t1_3_a -= 0.5*np.einsum('lnde,amin,lmde->ia',t2_1_a,v2e_vooo_a,t2_1_a,optimize=True)
        t1_3_a -= np.einsum('nled,amin,mled->ia',t2_1_ab,v2e_vooo_a,t2_1_ab,optimize=True)
        t1_3_a -= 0.5*np.einsum('lnde,amin,lmde->ia',t2_1_b,v2e_vooo_ab,t2_1_b,optimize=True)
        t1_3_a -= np.einsum('lnde,amin,lmde->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab,optimize=True)
 
        t1_3_b -= 0.5*np.einsum('lnde,amin,lmde->ia',t2_1_b,v2e_vooo_b,t2_1_b,optimize=True)
        t1_3_b -= np.einsum('lnde,amin,lmde->ia',t2_1_ab,v2e_vooo_b,t2_1_ab,optimize=True)
        t1_3_b -= 0.5*np.einsum('lnde,mani,lmde->ia',t2_1_a,v2e_ovoo_ab,t2_1_a,optimize=True)
        t1_3_b -= np.einsum('nled,mani,mled->ia',t2_1_ab,v2e_ovoo_ab,t2_1_ab,optimize=True)
 
        t1_3_a += 0.5*np.einsum('lmdf,afie,lmde->ia',t2_1_a,v2e_vvov_a,t2_1_a,optimize=True)
        t1_3_a += np.einsum('mlfd,afie,mled->ia',t2_1_ab,v2e_vvov_a,t2_1_ab,optimize=True)
        t1_3_a += 0.5*np.einsum('lmdf,afie,lmde->ia',t2_1_b,v2e_vvov_ab,t2_1_b,optimize=True)
        t1_3_a += np.einsum('lmdf,afie,lmde->ia',t2_1_ab,v2e_vvov_ab,t2_1_ab,optimize=True)
 
        t1_3_b += 0.5*np.einsum('lmdf,afie,lmde->ia',t2_1_b,v2e_vvov_b,t2_1_b,optimize=True)
        t1_3_b += np.einsum('lmdf,afie,lmde->ia',t2_1_ab,v2e_vvov_b,t2_1_ab,optimize=True)
        t1_3_b += 0.5*np.einsum('lmdf,faei,lmde->ia',t2_1_a,v2e_vvvo_ab,t2_1_a,optimize=True)
        t1_3_b += np.einsum('mlfd,faei,mled->ia',t2_1_ab,v2e_vvvo_ab,t2_1_ab,optimize=True)
 
        t1_3_a -= np.einsum('lnde,emin,lmad->ia',t2_1_a,v2e_vooo_a,t2_1_a,optimize=True)
        t1_3_a += np.einsum('lnde,mein,lmad->ia',t2_1_ab,v2e_ovoo_ab,t2_1_a,optimize=True)
        t1_3_a += np.einsum('nled,emin,mlad->ia',t2_1_ab,v2e_vooo_a,t2_1_ab,optimize=True)
        t1_3_a += np.einsum('lned,emin,lmad->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab,optimize=True)
        t1_3_a -= np.einsum('lnde,mein,mlad->ia',t2_1_b,v2e_ovoo_ab,t2_1_ab,optimize=True)
 
        t1_3_b -= np.einsum('lnde,emin,lmad->ia',t2_1_b,v2e_vooo_b,t2_1_b,optimize=True)
        t1_3_b += np.einsum('nled,emni,lmad->ia',t2_1_ab,v2e_vooo_ab,t2_1_b,optimize=True)
        t1_3_b += np.einsum('lnde,emin,lmda->ia',t2_1_ab,v2e_vooo_b,t2_1_ab,optimize=True)
        t1_3_b += np.einsum('nlde,meni,mlda->ia',t2_1_ab,v2e_ovoo_ab,t2_1_ab,optimize=True)
        t1_3_b -= np.einsum('lnde,emni,lmda->ia',t2_1_a,v2e_vooo_ab,t2_1_ab,optimize=True)
 
        t1_3_a -= 0.25*np.einsum('lmef,efid,lmad->ia',t2_1_a,v2e_vvov_a,t2_1_a,optimize=True)
        t1_3_a -= np.einsum('lmef,efid,lmad->ia',t2_1_ab,v2e_vvov_ab,t2_1_ab,optimize=True)
 
        t1_3_b -= 0.25*np.einsum('lmef,efid,lmad->ia',t2_1_b,v2e_vvov_b,t2_1_b,optimize=True)
        temp = t2_1_ab.reshape(nocc_a*nocc_b,-1)
        temp_1 = v2e_vvvo_ab[:].reshape(nvir_a*nvir_b,-1)
        temp_2 = t2_1_ab.reshape(nocc_a*nocc_b*nvir_a,-1)
        int_1 = np.dot(temp,temp_1).reshape(nocc_a*nocc_b*nvir_a,-1)
        t1_3_b -= np.dot(int_1.T,temp_2).reshape(nocc_b,nvir_b)
        del temp_1
        t1_3_a = t1_3_a/D1_a
        t1_3_b = t1_3_b/D1_b
 
        t1_3 = (t1_3_a, t1_3_b)

    t1 = (t1_2, t1_3)
    t2 = (t2_1, t2_2)

    return t1, t2

def compute_energy(myadc, t1, t2, eris):

    if myadc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(myadc.method)

    v2e_oovv_a, v2e_oovv_ab, v2e_oovv_b = eris.oovv

    t2_1_a, t2_1_ab, t2_1_b  = t2[0]

    #Compute MP2 correlation energy

    e_mp2 = 0.25 * np.einsum('ijab,ijab', t2_1_a, v2e_oovv_a)
    e_mp2 += np.einsum('ijab,ijab', t2_1_ab, v2e_oovv_ab)
    e_mp2 += 0.25 * np.einsum('ijab,ijab', t2_1_b, v2e_oovv_b)
    
    e_corr = e_mp2 

    if (myadc.method == "adc(3)"):

        #Compute MP3 correlation energy

        v2e_oooo_a, v2e_oooo_ab, v2e_oooo_b = eris.oooo
        v2e_vvvv_a, v2e_vvvv_ab, v2e_vvvv_b = eris.vvvv
        v2e_voov_a, v2e_voov_ab, v2e_voov_b = eris.voov
        v2e_ovvo_a, v2e_ovvo_ab, v2e_ovvo_b = eris.ovvo
        v2e_ovov_a, v2e_ovov_ab, v2e_ovov_b = eris.ovov
        v2e_vovo_a, v2e_vovo_ab, v2e_vovo_b = eris.vovo

        temp_1_a =  np.einsum('ijab,ijcd', t2_1_a, t2_1_a)
        temp_1_b =  np.einsum('ijab,ijcd', t2_1_b, t2_1_b)
        temp_1_ab_1 =  np.einsum('ijab,ijcd', t2_1_ab, t2_1_ab)

        temp_2_a =  np.einsum('ijab,klab', t2_1_a, t2_1_a)
        temp_2_b =  np.einsum('ijab,klab', t2_1_b, t2_1_b)
        temp_2_ab_1 =  np.einsum('ijab,klab', t2_1_ab, t2_1_ab)

        temp_3_a = np.einsum('ijab,ikcb->akcj', t2_1_a, t2_1_a)
        temp_3_a += np.einsum('jiab,kicb->akcj', t2_1_ab, t2_1_ab)
        temp_3_b = np.einsum('ijab,ikcb->akcj', t2_1_b, t2_1_b)
        temp_3_b += np.einsum('ijba,ikbc->akcj', t2_1_ab, t2_1_ab)

        temp_3_ab_1 = np.einsum('ijab,ikcb->akcj', t2_1_ab, t2_1_ab)
        temp_3_ab_2 = np.einsum('jiba,kibc->akcj', t2_1_ab, t2_1_ab)
        temp_3_ab_3 = -np.einsum('ijab,ikbc->akcj', t2_1_a, t2_1_ab)
        temp_3_ab_3 -= np.einsum('jiab,ikcb->akcj', t2_1_ab, t2_1_b)
        temp_3_ab_4 = -np.einsum('ijba,ikcb->akcj', t2_1_ab, t2_1_a)
        temp_3_ab_4 -= np.einsum('ijab,kicb->akcj', t2_1_b, t2_1_ab)

        e_mp3 = 0.125 * np.einsum('abcd,abcd',temp_1_a, v2e_vvvv_a)
        e_mp3 += 0.125 * np.einsum('abcd,abcd',temp_1_b, v2e_vvvv_b)
        e_mp3 +=  np.einsum('abcd,abcd',temp_1_ab_1, v2e_vvvv_ab)

        e_mp3 += 0.125 * np.einsum('ijkl,ijkl',temp_2_a, v2e_oooo_a)
        e_mp3 += 0.125 * np.einsum('ijkl,ijkl',temp_2_b, v2e_oooo_b)
        e_mp3 +=  np.einsum('ijkl,ijkl',temp_2_ab_1, v2e_oooo_ab)

        e_mp3 -= np.einsum('akcj,akcj',temp_3_a, v2e_vovo_a)
        e_mp3 -= np.einsum('akcj,akcj',temp_3_b, v2e_vovo_b)
        e_mp3 -= np.einsum('akcj,akcj',temp_3_ab_1, v2e_vovo_ab)
        e_mp3 -= np.einsum('akcj,kajc',temp_3_ab_2, v2e_ovov_ab)
        e_mp3 += np.einsum('akcj,akjc',temp_3_ab_3, v2e_voov_ab)
        e_mp3 += np.einsum('akcj,kacj',temp_3_ab_4, v2e_ovvo_ab)
    
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
    
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        from pyscf import gto
        
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
    
        eris = uadc_ao2mo.transform_integrals(self)
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
        eris = uadc_ao2mo.transform_integrals(adc)

    v2e_oovv_a,v2e_oovv_ab,v2e_oovv_b = eris.oovv
    v2e_ooov_a,v2e_ooov_ab,v2e_ooov_b = eris.ooov
    v2e_oooo_a,v2e_oooo_ab,v2e_oooo_b = eris.oooo
    v2e_ovoo_a,v2e_ovoo_ab,v2e_ovoo_b = eris.ovoo
    v2e_ovov_a,v2e_ovov_ab,v2e_ovov_b = eris.ovov
    v2e_vvoo_a,v2e_vvoo_ab,v2e_vvoo_b = eris.vvoo
    v2e_vvvv_a,v2e_vvvv_ab,v2e_vvvv_b = eris.vvvv
    v2e_voov_a,v2e_voov_ab,v2e_voov_b = eris.voov
    v2e_ovvo_a,v2e_ovvo_ab,v2e_ovvo_b = eris.ovvo
    v2e_vovo_a,v2e_vovo_ab,v2e_vovo_b = eris.vovo
    v2e_vvvo_a,v2e_vvvo_ab,v2e_vvvo_b = eris.vvvo
    v2e_vovv_a,v2e_vovv_ab,v2e_vovv_b = eris.vovv
    v2e_oovo_a,v2e_oovo_ab,v2e_oovo_b = eris.oovo
    v2e_ovvv_a,v2e_ovvv_ab,v2e_ovvv_b = eris.ovvv
    v2e_vvov_a,v2e_vvov_ab,v2e_vvov_b = eris.vvov

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

    M_ab_a -= 0.5 *  np.einsum('lmad,lmbd->ab',t2_1_a, v2e_oovv_a)
    M_ab_a -=        np.einsum('lmad,lmbd->ab',t2_1_ab, v2e_oovv_ab)

    M_ab_b -= 0.5 *  np.einsum('lmad,lmbd->ab',t2_1_b, v2e_oovv_b)
    M_ab_b -=        np.einsum('mlda,mldb->ab',t2_1_ab, v2e_oovv_ab)

    M_ab_a -= 0.5 *  np.einsum('lmbd,lmad->ab',t2_1_a, v2e_oovv_a)
    M_ab_a -=        np.einsum('lmbd,lmad->ab',t2_1_ab, v2e_oovv_ab)

    M_ab_b -= 0.5 *  np.einsum('lmbd,lmad->ab',t2_1_b, v2e_oovv_b)
    M_ab_b -=        np.einsum('mldb,mlda->ab',t2_1_ab, v2e_oovv_ab)


    #Third-order terms

    if(method =='adc(3)'):

        t2_2_a, t2_2_ab, t2_2_b = t2[1]

        M_ab_a +=  np.einsum('ld,albd->ab',t1_2_a, v2e_vovv_a)
        M_ab_a +=  np.einsum('ld,albd->ab',t1_2_b, v2e_vovv_ab)

        M_ab_b +=  np.einsum('ld,albd->ab',t1_2_b, v2e_vovv_b)
        M_ab_b +=  np.einsum('ld,ladb->ab',t1_2_a, v2e_ovvv_ab)

        M_ab_a += np.einsum('ld,adbl->ab',t1_2_a, v2e_vvvo_a)
        M_ab_a += np.einsum('ld,adbl->ab',t1_2_b, v2e_vvvo_ab)

        M_ab_b += np.einsum('ld,adbl->ab',t1_2_b, v2e_vvvo_b)
        M_ab_b += np.einsum('ld,dalb->ab',t1_2_a, v2e_vvov_ab)

        M_ab_a -=0.5* np.einsum('lmbd,lmad->ab',t2_2_a,v2e_oovv_a)
        M_ab_a -= np.einsum('lmbd,lmad->ab',t2_2_ab,v2e_oovv_ab)

        M_ab_b -=0.5* np.einsum('lmbd,lmad->ab',t2_2_b,v2e_oovv_b)
        M_ab_b -= np.einsum('mldb,mlda->ab',t2_2_ab,v2e_oovv_ab)

        M_ab_a -=0.5* np.einsum('lmad,lmbd->ab',t2_2_a,v2e_oovv_a)
        M_ab_a -= np.einsum('lmad,lmbd->ab',t2_2_ab,v2e_oovv_ab)

        M_ab_b -=0.5* np.einsum('lmad,lmbd->ab',t2_2_b,v2e_oovv_b)
        M_ab_b -= np.einsum('mlda,mldb->ab',t2_2_ab,v2e_oovv_ab)

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

        M_ab_a -= np.einsum('lned,mlbd,anem->ab',t2_1_a, t2_1_a, v2e_vovo_a, optimize=True)
        M_ab_a += np.einsum('nled,mlbd,anem->ab',t2_1_ab, t2_1_ab, v2e_vovo_a, optimize=True)
        M_ab_a -= np.einsum('lnde,mlbd,anme->ab',t2_1_ab, t2_1_a, v2e_voov_ab, optimize=True)
        M_ab_a += np.einsum('lned,mlbd,anme->ab',t2_1_b, t2_1_ab, v2e_voov_ab, optimize=True)
        M_ab_a += np.einsum('lned,lmbd,anem->ab',t2_1_ab, t2_1_ab, v2e_vovo_ab, optimize=True)

        M_ab_b -= np.einsum('lned,mlbd,anem->ab',t2_1_b, t2_1_b, v2e_vovo_b, optimize=True)
        M_ab_b += np.einsum('lnde,lmdb,anem->ab',t2_1_ab, t2_1_ab, v2e_vovo_b, optimize=True)
        M_ab_b -= np.einsum('nled,mlbd,naem->ab',t2_1_ab, t2_1_b, v2e_ovvo_ab, optimize=True)
        M_ab_b += np.einsum('lned,lmdb,naem->ab',t2_1_a, t2_1_ab, v2e_ovvo_ab, optimize=True)
        M_ab_b += np.einsum('nlde,mldb,name->ab',t2_1_ab, t2_1_ab, v2e_ovov_ab, optimize=True)

        M_ab_a -= np.einsum('mled,lnad,enbm->ab',t2_1_a, t2_1_a, v2e_vovo_a, optimize=True)
        M_ab_a -= np.einsum('mled,nlad,nebm->ab',t2_1_b, t2_1_ab, v2e_ovvo_ab, optimize=True)
        M_ab_a += np.einsum('mled,nlad,enbm->ab',t2_1_ab, t2_1_ab, v2e_vovo_a, optimize=True)
        M_ab_a += np.einsum('lmde,lnad,nebm->ab',t2_1_ab, t2_1_a, v2e_ovvo_ab, optimize=True)
        M_ab_a += np.einsum('lmed,lnad,enbm->ab',t2_1_ab, t2_1_ab, v2e_vovo_ab, optimize=True)

        M_ab_b -= np.einsum('mled,lnad,enbm->ab',t2_1_b, t2_1_b, v2e_vovo_b, optimize=True)
        M_ab_b -= np.einsum('mled,lnda,enmb->ab',t2_1_a, t2_1_ab, v2e_voov_ab, optimize=True)
        M_ab_b += np.einsum('lmde,lnda,enbm->ab',t2_1_ab, t2_1_ab, v2e_vovo_b, optimize=True)
        M_ab_b += np.einsum('mled,lnad,enmb->ab',t2_1_ab, t2_1_b, v2e_voov_ab, optimize=True)
        M_ab_b += np.einsum('mlde,nlda,nemb->ab',t2_1_ab, t2_1_ab, v2e_ovov_ab, optimize=True)

        M_ab_a -= np.einsum('mlbd,lnae,dnem->ab',t2_1_a, t2_1_a, v2e_vovo_a, optimize=True)
        M_ab_a += np.einsum('lmbd,lnae,dnem->ab',t2_1_ab, t2_1_ab, v2e_vovo_b, optimize=True)
        M_ab_a += np.einsum('mlbd,lnae,dnme->ab',t2_1_a, t2_1_ab, v2e_voov_ab, optimize=True)
        M_ab_a -= np.einsum('lmbd,lnae,ndem->ab',t2_1_ab, t2_1_a, v2e_ovvo_ab, optimize=True)
        M_ab_a += np.einsum('mlbd,nlae,ndme->ab',t2_1_ab, t2_1_ab, v2e_ovov_ab, optimize=True)

        M_ab_b -= np.einsum('mlbd,lnae,dnem->ab',t2_1_b, t2_1_b, v2e_vovo_b, optimize=True)
        M_ab_b += np.einsum('mldb,nlea,dnem->ab',t2_1_ab, t2_1_ab, v2e_vovo_a, optimize=True)
        M_ab_b += np.einsum('mlbd,nlea,ndem->ab',t2_1_b, t2_1_ab, v2e_ovvo_ab, optimize=True)
        M_ab_b -= np.einsum('mldb,lnae,dnme->ab',t2_1_ab, t2_1_b, v2e_voov_ab, optimize=True)
        M_ab_b += np.einsum('lmdb,lnea,dnem->ab',t2_1_ab, t2_1_ab, v2e_vovo_ab, optimize=True)

        M_ab_a -= 0.25*np.einsum('mlef,mlbd,adef->ab',t2_1_a, t2_1_a, v2e_vvvv_a, optimize=True)
        M_ab_a -= np.einsum('mlef,mlbd,adef->ab',t2_1_ab, t2_1_ab, v2e_vvvv_ab, optimize=True)

        M_ab_b -= 0.25*np.einsum('mlef,mlbd,adef->ab',t2_1_b, t2_1_b, v2e_vvvv_b, optimize=True)
        M_ab_b -= np.einsum('mlef,mldb,daef->ab',t2_1_ab, t2_1_ab, v2e_vvvv_ab, optimize=True)

        M_ab_a -= 0.25*np.einsum('mled,mlaf,edbf->ab',t2_1_a, t2_1_a, v2e_vvvv_a, optimize=True)
        M_ab_a -= np.einsum('mled,mlaf,edbf->ab',t2_1_ab, t2_1_ab, v2e_vvvv_ab, optimize=True)

        M_ab_b -= 0.25*np.einsum('mled,mlaf,edbf->ab',t2_1_b, t2_1_b, v2e_vvvv_b, optimize=True)
        M_ab_b -= np.einsum('mled,mlfa,edfb->ab',t2_1_ab, t2_1_ab, v2e_vvvv_ab, optimize=True)

        M_ab_a -= 0.25*np.einsum('mlbd,noad,noml->ab',t2_1_a, t2_1_a, v2e_oooo_a, optimize=True)
        M_ab_a -= np.einsum('mlbd,noad,noml->ab',t2_1_ab, t2_1_ab, v2e_oooo_ab, optimize=True)

        M_ab_b -= 0.25*np.einsum('mlbd,noad,noml->ab',t2_1_b, t2_1_b, v2e_oooo_b, optimize=True)
        M_ab_b -= np.einsum('lmdb,onda,onlm->ab',t2_1_ab, t2_1_ab, v2e_oooo_ab, optimize=True)

        M_ab_a += 0.5*np.einsum('lned,mled,anbm->ab',t2_1_a, t2_1_a, v2e_vovo_a, optimize=True)
        M_ab_a += 0.5*np.einsum('lned,mled,anbm->ab',t2_1_b, t2_1_b, v2e_vovo_ab, optimize=True)
        M_ab_a -= np.einsum('lned,lmed,anbm->ab',t2_1_ab, t2_1_ab, v2e_vovo_ab, optimize=True)
        M_ab_a -= np.einsum('nled,mled,anbm->ab',t2_1_ab, t2_1_ab, v2e_vovo_a, optimize=True)

        M_ab_b += 0.5*np.einsum('lned,mled,anbm->ab',t2_1_b, t2_1_b, v2e_vovo_b, optimize=True)
        M_ab_b += 0.5*np.einsum('lned,mled,namb->ab',t2_1_a, t2_1_a, v2e_ovov_ab, optimize=True)
        M_ab_b -= np.einsum('nled,mled,namb->ab',t2_1_ab, t2_1_ab, v2e_ovov_ab, optimize=True)
        M_ab_b -= np.einsum('lned,lmed,anbm->ab',t2_1_ab, t2_1_ab, v2e_vovo_b, optimize=True)

        M_ab_a -= 0.5*np.einsum('mldf,mled,aebf->ab',t2_1_a, t2_1_a, v2e_vvvv_a, optimize=True)
        M_ab_a -= 0.5*np.einsum('mldf,mled,aebf->ab',t2_1_b, t2_1_b, v2e_vvvv_ab, optimize=True)
        M_ab_a += np.einsum('mldf,mlde,aebf->ab',t2_1_ab, t2_1_ab, v2e_vvvv_ab, optimize=True)
        M_ab_a += np.einsum('mlfd,mled,aebf->ab',t2_1_ab, t2_1_ab, v2e_vvvv_a, optimize=True)

        M_ab_b -= 0.5*np.einsum('mldf,mled,aebf->ab',t2_1_b, t2_1_b, v2e_vvvv_b, optimize=True)
        M_ab_b -= 0.5*np.einsum('mldf,mled,eafb->ab',t2_1_a, t2_1_a, v2e_vvvv_ab, optimize=True)
        M_ab_b += np.einsum('mlfd,mled,eafb->ab',t2_1_ab, t2_1_ab, v2e_vvvv_ab, optimize=True)
        M_ab_b += np.einsum('mldf,mlde,aebf->ab',t2_1_ab, t2_1_ab, v2e_vvvv_b, optimize=True)

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
        eris = uadc_ao2mo.transform_integrals(adc)

    v2e_oovv_a,v2e_oovv_ab,v2e_oovv_b = eris.oovv
    v2e_vvoo_a,v2e_vvoo_ab,v2e_vvoo_b = eris.vvoo
    v2e_ooov_a,v2e_ooov_ab,v2e_ooov_b = eris.ooov
    v2e_ovoo_a,v2e_ovoo_ab,v2e_ovoo_b = eris.ovoo
    v2e_ovov_a,v2e_ovov_ab,v2e_ovov_b = eris.ovov
    v2e_vovo_a,v2e_vovo_ab,v2e_vovo_b = eris.vovo
    v2e_oooo_a,v2e_oooo_ab,v2e_oooo_b = eris.oooo
    v2e_ovvo_a,v2e_ovvo_ab,v2e_ovvo_b = eris.ovvo
    v2e_vvvv_a,v2e_vvvv_ab,v2e_vvvv_b = eris.vvvv
    v2e_voov_a,v2e_voov_ab,v2e_voov_b = eris.voov
    v2e_oovo_a,v2e_oovo_ab,v2e_oovo_b = eris.oovo
    v2e_vooo_a,v2e_vooo_ab,v2e_vooo_b = eris.vooo

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

    M_ij_a += 0.5 *  np.einsum('ilde,jlde->ij',t2_1_a, v2e_oovv_a)
    M_ij_a += np.einsum('ilde,jlde->ij',t2_1_ab, v2e_oovv_ab)

    M_ij_b += 0.5 *  np.einsum('ilde,jlde->ij',t2_1_b, v2e_oovv_b)
    M_ij_b += np.einsum('lied,ljed->ij',t2_1_ab, v2e_oovv_ab)

    M_ij_a += 0.5 *  np.einsum('jlde,deil->ij',t2_1_a, v2e_vvoo_a)
    M_ij_a += np.einsum('jlde,deil->ij',t2_1_ab, v2e_vvoo_ab)

    M_ij_b += 0.5 *  np.einsum('jlde,deil->ij',t2_1_b, v2e_vvoo_b)
    M_ij_b += np.einsum('ljed,edli->ij',t2_1_ab, v2e_vvoo_ab)

    # Third-order terms

    if (method == "adc(3)"):

        t2_2_a, t2_2_ab, t2_2_b = t2[1]

        M_ij_a += np.einsum('ld,jlid->ij',t1_2_a, v2e_ooov_a)
        M_ij_a += np.einsum('ld,jlid->ij',t1_2_b, v2e_ooov_ab)

        M_ij_b += np.einsum('ld,jlid->ij',t1_2_b, v2e_ooov_b)
        M_ij_b += np.einsum('ld,ljdi->ij',t1_2_a, v2e_oovo_ab)

        M_ij_a += np.einsum('ld,jdil->ij',t1_2_a, v2e_ovoo_a)
        M_ij_a += np.einsum('ld,jdil->ij',t1_2_b, v2e_ovoo_ab)

        M_ij_b += np.einsum('ld,jdil->ij',t1_2_b, v2e_ovoo_b)
        M_ij_b += np.einsum('ld,djli->ij',t1_2_a, v2e_vooo_ab)

        M_ij_a += 0.5* np.einsum('ilde,jlde->ij',t2_2_a, v2e_oovv_a)
        M_ij_a += np.einsum('ilde,jlde->ij',t2_2_ab, v2e_oovv_ab)

        M_ij_b += 0.5* np.einsum('ilde,jlde->ij',t2_2_b, v2e_oovv_b)
        M_ij_b += np.einsum('lied,ljed->ij',t2_2_ab, v2e_oovv_ab)

        M_ij_a += 0.5* np.einsum('jlde,deil->ij',t2_2_a, v2e_vvoo_a)
        M_ij_a += np.einsum('jlde,deil->ij',t2_2_ab, v2e_vvoo_ab)

        M_ij_b += 0.5* np.einsum('jlde,deil->ij',t2_2_b, v2e_vvoo_b)
        M_ij_b += np.einsum('ljed,edli->ij',t2_2_ab, v2e_vvoo_ab)

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

        M_ij_a -= np.einsum('lmde,jldf,fmie->ij',t2_1_a, t2_1_a, v2e_voov_a ,optimize = True)
        M_ij_a += np.einsum('mled,jlfd,fmie->ij',t2_1_ab, t2_1_ab, v2e_voov_a ,optimize = True)
        M_ij_a -= np.einsum('lmde,jldf,fmie->ij',t2_1_ab, t2_1_a, v2e_voov_ab,optimize = True)
        M_ij_a -= np.einsum('mlde,jldf,mfie->ij',t2_1_ab, t2_1_ab, v2e_ovov_ab ,optimize = True)
        M_ij_a += np.einsum('lmde,jlfd,fmie->ij',t2_1_b, t2_1_ab, v2e_voov_ab ,optimize = True)

        M_ij_b -= np.einsum('lmde,jldf,fmie->ij',t2_1_b, t2_1_b, v2e_voov_b ,optimize = True)
        M_ij_b += np.einsum('lmde,ljdf,fmie->ij',t2_1_ab, t2_1_ab, v2e_voov_b ,optimize = True)
        M_ij_b -= np.einsum('mled,jldf,mfei->ij',t2_1_ab, t2_1_b, v2e_ovvo_ab,optimize = True)
        M_ij_b -= np.einsum('lmed,ljfd,fmei->ij',t2_1_ab, t2_1_ab, v2e_vovo_ab ,optimize = True)
        M_ij_b += np.einsum('lmde,ljdf,mfei->ij',t2_1_a, t2_1_ab, v2e_ovvo_ab ,optimize = True)

        M_ij_a -= np.einsum('lmde,ildf,fmje->ij',t2_1_a, t2_1_a, v2e_voov_a ,optimize = True)
        M_ij_a += np.einsum('mled,ilfd,fmje->ij',t2_1_ab, t2_1_ab, v2e_voov_a ,optimize = True)
        M_ij_a -= np.einsum('lmde,ildf,fmje->ij',t2_1_ab, t2_1_a, v2e_voov_ab,optimize = True)
        M_ij_a -= np.einsum('mlde,ildf,mfje->ij',t2_1_ab, t2_1_ab, v2e_ovov_ab ,optimize = True)
        M_ij_a += np.einsum('lmde,ilfd,fmje->ij',t2_1_b, t2_1_ab, v2e_voov_ab ,optimize = True)

        M_ij_b -= np.einsum('lmde,ildf,fmje->ij',t2_1_b, t2_1_b, v2e_voov_b ,optimize = True)
        M_ij_b += np.einsum('lmde,lidf,fmje->ij',t2_1_ab, t2_1_ab, v2e_voov_b ,optimize = True)
        M_ij_b -= np.einsum('mled,ildf,mfej->ij',t2_1_ab, t2_1_b, v2e_ovvo_ab,optimize = True)
        M_ij_b -= np.einsum('lmed,lifd,fmej->ij',t2_1_ab, t2_1_ab, v2e_vovo_ab ,optimize = True)
        M_ij_b += np.einsum('lmde,lidf,mfej->ij',t2_1_a, t2_1_ab, v2e_ovvo_ab ,optimize = True)

        M_ij_a += 0.25*np.einsum('lmde,jnde,lmin->ij',t2_1_a, t2_1_a,v2e_oooo_a, optimize = True)
        M_ij_a += np.einsum('lmde,jnde,lmin->ij',t2_1_ab ,t2_1_ab,v2e_oooo_ab, optimize = True)

        M_ij_b += 0.25*np.einsum('lmde,jnde,lmin->ij',t2_1_b, t2_1_b,v2e_oooo_b, optimize = True)
        M_ij_b += np.einsum('mled,njed,mlni->ij',t2_1_ab ,t2_1_ab,v2e_oooo_ab, optimize = True)

        M_ij_a += 0.25*np.einsum('ilde,jlgf,gfde->ij',t2_1_a, t2_1_a,v2e_vvvv_a, optimize = True)
        M_ij_a +=np.einsum('ilde,jlgf,gfde->ij',t2_1_ab, t2_1_ab,v2e_vvvv_ab, optimize = True)

        M_ij_b += 0.25*np.einsum('ilde,jlgf,gfde->ij',t2_1_b, t2_1_b,v2e_vvvv_b, optimize = True)
        M_ij_b +=np.einsum('lied,ljfg,fged->ij',t2_1_ab, t2_1_ab,v2e_vvvv_ab, optimize = True)

        M_ij_a += 0.25*np.einsum('inde,lmde,jnlm->ij',t2_1_a, t2_1_a,v2e_oooo_a, optimize = True)
        M_ij_a +=np.einsum('inde,lmde,jnlm->ij',t2_1_ab, t2_1_ab,v2e_oooo_ab, optimize = True)

        M_ij_b += 0.25*np.einsum('inde,lmde,jnlm->ij',t2_1_b, t2_1_b,v2e_oooo_b, optimize = True)
        M_ij_b +=np.einsum('nied,mled,njml->ij',t2_1_ab, t2_1_ab,v2e_oooo_ab, optimize = True)

        M_ij_a += 0.5*np.einsum('lmdf,lmde,jeif->ij',t2_1_a, t2_1_a, v2e_ovov_a , optimize = True)
        M_ij_a +=np.einsum('mlfd,mled,jeif->ij',t2_1_ab, t2_1_ab, v2e_ovov_a , optimize = True)
        M_ij_a +=np.einsum('lmdf,lmde,jeif->ij',t2_1_ab, t2_1_ab, v2e_ovov_ab , optimize = True)
        M_ij_a +=0.5*np.einsum('lmdf,lmde,jeif->ij',t2_1_b, t2_1_b, v2e_ovov_ab , optimize = True)

        M_ij_b += 0.5*np.einsum('lmdf,lmde,jeif->ij',t2_1_b, t2_1_b, v2e_ovov_b , optimize = True)
        M_ij_b +=np.einsum('lmdf,lmde,jeif->ij',t2_1_ab, t2_1_ab, v2e_ovov_b , optimize = True)
        M_ij_b +=np.einsum('lmfd,lmed,ejfi->ij',t2_1_ab, t2_1_ab, v2e_vovo_ab , optimize = True)
        M_ij_b +=0.5*np.einsum('lmdf,lmde,ejfi->ij',t2_1_a, t2_1_a, v2e_vovo_ab , optimize = True)

        M_ij_a -= np.einsum('ilde,jmdf,flem->ij',t2_1_a, t2_1_a, v2e_vovo_a, optimize = True)
        M_ij_a += np.einsum('ilde,jmdf,lfem->ij',t2_1_a, t2_1_ab, v2e_ovvo_ab, optimize = True)
        M_ij_a += np.einsum('ilde,jmdf,flme->ij',t2_1_ab, t2_1_a, v2e_voov_ab, optimize = True)
        M_ij_a -= np.einsum('ilde,jmdf,flem->ij',t2_1_ab, t2_1_ab, v2e_vovo_b, optimize = True)
        M_ij_a -= np.einsum('iled,jmfd,flem->ij',t2_1_ab, t2_1_ab, v2e_vovo_ab, optimize = True)

        M_ij_b -= np.einsum('ilde,jmdf,flem->ij',t2_1_b, t2_1_b, v2e_vovo_b, optimize = True)
        M_ij_b += np.einsum('ilde,mjfd,flme->ij',t2_1_b, t2_1_ab, v2e_voov_ab, optimize = True)
        M_ij_b += np.einsum('lied,jmdf,lfem->ij',t2_1_ab, t2_1_b, v2e_ovvo_ab, optimize = True)
        M_ij_b -= np.einsum('lied,mjfd,flem->ij',t2_1_ab, t2_1_ab, v2e_vovo_a, optimize = True)
        M_ij_b -= np.einsum('lide,mjdf,lfme->ij',t2_1_ab, t2_1_ab, v2e_ovov_ab, optimize = True)

        M_ij_a -= 0.5*np.einsum('lnde,lmde,jnim->ij',t2_1_a, t2_1_a, v2e_oooo_a, optimize = True)
        M_ij_a -= np.einsum('nled,mled,jnim->ij',t2_1_ab, t2_1_ab, v2e_oooo_a, optimize = True)
        M_ij_a -= np.einsum('lnde,lmde,jnim->ij',t2_1_ab, t2_1_ab, v2e_oooo_ab, optimize = True)
        M_ij_a -= 0.5 * np.einsum('lnde,lmde,jnim->ij',t2_1_b, t2_1_b, v2e_oooo_ab, optimize = True)

        M_ij_b -= 0.5*np.einsum('lnde,lmde,jnim->ij',t2_1_b, t2_1_b, v2e_oooo_b, optimize = True)
        M_ij_b -= np.einsum('lnde,lmde,jnim->ij',t2_1_ab, t2_1_ab, v2e_oooo_b, optimize = True)
        M_ij_b -= np.einsum('nled,mled,njmi->ij',t2_1_ab, t2_1_ab, v2e_oooo_ab, optimize = True)
        M_ij_b -= 0.5 * np.einsum('lnde,lmde,njmi->ij',t2_1_a, t2_1_a, v2e_oooo_ab, optimize = True)

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
        eris = uadc_ao2mo.transform_integrals(adc)

    v2e_oovv_a,v2e_oovv_ab,v2e_oovv_b = eris.oovv
    v2e_ooov_a,v2e_ooov_ab,v2e_ooov_b = eris.ooov
    v2e_oooo_a,v2e_oooo_ab,v2e_oooo_b = eris.oooo
    v2e_ovoo_a,v2e_ovoo_ab,v2e_ovoo_b = eris.ovoo
    v2e_ovov_a,v2e_ovov_ab,v2e_ovov_b = eris.ovov
    v2e_vvoo_a,v2e_vvoo_ab,v2e_vvoo_b = eris.vvoo
    v2e_vvvv_a,v2e_vvvv_ab,v2e_vvvv_b = eris.vvvv
    v2e_voov_a,v2e_voov_ab,v2e_voov_b = eris.voov
    v2e_ovvo_a,v2e_ovvo_ab,v2e_ovvo_b = eris.ovvo
    v2e_vovo_a,v2e_vovo_ab,v2e_vovo_b = eris.vovo
    v2e_vvvo_a,v2e_vvvo_ab,v2e_vvvo_b = eris.vvvo
    v2e_vovv_a,v2e_vovv_ab,v2e_vovv_b = eris.vovv
    v2e_oovo_a,v2e_oovo_ab,v2e_oovo_b = eris.oovo
    v2e_ovvv_a,v2e_ovvv_ab,v2e_ovvv_b = eris.ovvv

    v2e_vovv_1_a = v2e_vovv_a[:][:,:,ab_ind_a[0],ab_ind_a[1]].reshape(nvir_a,-1)
    v2e_vovv_1_b = v2e_vovv_b[:][:,:,ab_ind_b[0],ab_ind_b[1]].reshape(nvir_b,-1)

    v2e_vovv_2_a = v2e_vovv_a[:][:,:,ab_ind_a[0],ab_ind_a[1]]
    v2e_vovv_2_b = v2e_vovv_b[:][:,:,ab_ind_b[0],ab_ind_b[1]]

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

        r_aba = r_aba.reshape(nocc_a,nvir_b,nvir_a)
        r_bab = r_bab.reshape(nocc_b,nvir_a,nvir_b)

############ ADC(2) ab block ############################

        s[s_a:f_a] = np.einsum('ab,b->a',M_ab_a,r_a)
        s[s_b:f_b] = np.einsum('ab,b->a',M_ab_b,r_b)

############ ADC(2) a - ibc block #########################

        s[s_a:f_a] += np.einsum('ap,p->a',v2e_vovv_1_a, r_aaa, optimize = True)
        s[s_a:f_a] += np.einsum('aibc,ibc->a', v2e_vovv_ab, r_bab, optimize = True)

        s[s_b:f_b] += np.einsum('ap,p->a', v2e_vovv_1_b, r_bbb, optimize = True)
        s[s_b:f_b] += np.einsum('iacb,ibc->a', v2e_ovvv_ab, r_aba, optimize = True)

############### ADC(2) ibc - a block ############################

        s[s_aaa:f_aaa] += np.einsum('aip,a->ip', v2e_vovv_2_a, r_a, optimize = True).reshape(-1)
        s[s_bab:f_bab] += np.einsum('aibc,a->ibc', v2e_vovv_ab, r_a, optimize = True).reshape(-1)
        s[s_aba:f_aba] += np.einsum('iacb,a->ibc', v2e_ovvv_ab, r_b, optimize = True).reshape(-1)
        s[s_bbb:f_bbb] += np.einsum('aip,a->ip', v2e_vovv_2_b, r_b, optimize = True).reshape(-1)

################ ADC(2) iab - jcd block ############################

        s[s_aaa:f_aaa] += D_iab_a * r_aaa
        s[s_bab:f_bab] += D_iab_bab * r_bab.reshape(-1)
        s[s_aba:f_aba] += D_iab_aba * r_aba.reshape(-1)
        s[s_bbb:f_bbb] += D_iab_b * r_bbb

############### ADC(3) iab - jcd block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

               t2_2_a, t2_2_ab, t2_2_b = adc.t2[1]

               r_aaa = r_aaa.reshape(nocc_a,-1)
               r_bbb = r_bbb.reshape(nocc_b,-1)

               r_aaa_u = np.zeros((nocc_a,nvir_a,nvir_a))
               r_aaa_u[:,ab_ind_a[0],ab_ind_a[1]]= r_aaa.copy()
               r_aaa_u[:,ab_ind_a[1],ab_ind_a[0]]= -r_aaa.copy()

               r_bbb_u = None
               r_bbb_u = np.zeros((nocc_b,nvir_b,nvir_b))
               r_bbb_u[:,ab_ind_b[0],ab_ind_b[1]]= r_bbb.copy()
               r_bbb_u[:,ab_ind_b[1],ab_ind_b[0]]= -r_bbb.copy()

               #temp = 0.5*np.einsum('yxwz,izw->ixy',v2e_vvvv_a,r_aaa_u ,optimize = True)
               #####temp = -0.5*np.einsum('yxzw,izw->ixy',v2e_vvvv_a,r_aaa_u )
               #s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)
               #temp = v2e_vvvv_a[ab_ind_a[0],ab_ind_a[1],:,:]
               #temp = temp.reshape(-1,nvir_a*nvir_a)
               #r_aaa_t = r_aaa_u.reshape(nocc_a,-1)
               #s[s_aaa:f_aaa] += 0.5*np.dot(r_aaa_t,temp.T).reshape(-1)

               temp = v2e_vvvv_a[:].reshape(nvir_a*nvir_a,nvir_a*nvir_a)
               r_aaa_t = r_aaa_u.reshape(nocc_a,-1)
               temp_1 = np.dot(r_aaa_t,temp.T).reshape(nocc_a,nvir_a,nvir_a)
               del temp
               temp_1 = temp_1[:,ab_ind_a[0],ab_ind_a[1]]
               s[s_aaa:f_aaa] += 0.5*temp_1.reshape(-1)

               temp = v2e_vvvv_b[:].reshape(nvir_b*nvir_b,nvir_b*nvir_b)
               r_bbb_t = r_bbb_u.reshape(nocc_b,-1)
               temp_1 = np.dot(r_bbb_t,temp.T).reshape(nocc_b,nvir_b,nvir_b)
               del temp
               temp_1 = temp_1[:,ab_ind_b[0],ab_ind_b[1]]
               s[s_bbb:f_bbb] += 0.5*temp_1.reshape(-1)

               #temp = v2e_vvvv_b[ab_ind_b[0],ab_ind_b[1],:,:]
               #temp = temp.reshape(-1,nvir_b*nvir_b)
               #r_bbb_t = r_bbb_u.reshape(nocc_b,-1)
               #s[s_bbb:f_bbb] += 0.5*np.dot(r_bbb_t,temp.T).reshape(-1)

               #temp = 0.5*np.einsum('yxwz,izw->ixy',v2e_vvvv_b,r_bbb_u,optimize = True)
               ########temp = -0.5*np.einsum('yxzw,izw->ixy',v2e_vvvv_b,r_bbb_u)
               #s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

               #s[s_bab:f_bab] += np.einsum('xyzw,izw->ixy',v2e_vvvv_ab,r_bab,optimize = True).reshape(-1)
               #s[s_bab:f_bab] += np.einsum('xyzw,izw->ixy',v2e_vvvv_ab,r_bab).reshape(-1)
               temp = v2e_vvvv_ab[:].reshape(nvir_a*nvir_b,nvir_a*nvir_b)
               r_bab_t = r_bab.reshape(nocc_b,-1)
               s[s_bab:f_bab] += np.dot(r_bab_t,temp.T).reshape(-1)
               del temp

               #s[s_aba:f_aba] += np.einsum('yxwz,izw->ixy',v2e_vvvv_ab,r_aba,optimize = True).reshape(-1)
               #temp = v2e_vvvv_ab.transpose(3,2,1,0)
               #temp = temp.reshape(nvir_a*nvir_b,nvir_a*nvir_b)
               #r_aba_t = r_aba.reshape(nocc_a,-1)
               #s[s_aba:f_aba] += np.dot(r_aba_t,temp).reshape(-1)

               temp = v2e_vvvv_ab[:].reshape(nvir_a*nvir_b,nvir_a*nvir_b)
               r_aba_t = r_aba.transpose(0,2,1).reshape(nocc_a,-1)
               temp_1 = np.dot(r_aba_t,temp.T).reshape(nocc_a, nvir_a,nvir_b)
               s[s_aba:f_aba] += temp_1.transpose(0,2,1).copy().reshape(-1)
               del temp

               temp = 0.5*np.einsum('yjzi,jzx->ixy',v2e_vovo_a,r_aaa_u,optimize = True)
               temp +=0.5*np.einsum('yjiz,jxz->ixy',v2e_voov_ab,r_bab,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] -= 0.5*np.einsum('jyzi,jzx->ixy',v2e_ovvo_ab,r_aaa_u,optimize = True).reshape(-1)
               s[s_bab:f_bab] -= 0.5*np.einsum('yjzi,jxz->ixy',v2e_vovo_b,r_bab,optimize = True).reshape(-1)

               temp = 0.5*np.einsum('yjzi,jzx->ixy',v2e_vovo_b,r_bbb_u,optimize = True)
               temp +=0.5* np.einsum('jyzi,jxz->ixy',v2e_ovvo_ab,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

               s[s_aba:f_aba] -= 0.5*np.einsum('yjzi,jxz->ixy',v2e_vovo_a,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= 0.5*np.einsum('yjiz,jzx->ixy',v2e_voov_ab,r_bbb_u,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('xjzi,jzy->ixy',v2e_vovo_a,r_aaa_u,optimize = True)
               temp -= 0.5*np.einsum('xjiz,jyz->ixy',v2e_voov_ab,r_bab,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] -=  0.5*np.einsum('xjzi,jzy->ixy',v2e_vovo_ab,r_bab,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('xjzi,jzy->ixy',v2e_vovo_b,r_bbb_u,optimize = True)
               temp -= 0.5*np.einsum('jxzi,jyz->ixy',v2e_ovvo_ab,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

               s[s_aba:f_aba] -= 0.5*np.einsum('jxiz,jzy->ixy',v2e_ovov_ab,r_aba,optimize = True).reshape(-1)

               temp = 0.5*np.einsum('xjwi,jyw->ixy',v2e_vovo_a,r_aaa_u,optimize = True)
               temp -= 0.5*np.einsum('xjiw,jyw->ixy',v2e_voov_ab,r_bab,optimize = True)

               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] -= 0.5*np.einsum('xjwi,jwy->ixy',v2e_vovo_ab,r_bab,optimize = True).reshape(-1)

               temp = 0.5*np.einsum('xjwi,jyw->ixy',v2e_vovo_b,r_bbb_u,optimize = True)
               temp -= 0.5*np.einsum('jxwi,jyw->ixy',v2e_ovvo_ab,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

               s[s_aba:f_aba] -= 0.5*np.einsum('jxiw,jwy->ixy',v2e_ovov_ab,r_aba,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('yjwi,jxw->ixy',v2e_vovo_a,r_aaa_u,optimize = True)
               temp += 0.5*np.einsum('yjiw,jxw->ixy',v2e_voov_ab,r_bab,optimize = True)

               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] -= 0.5*np.einsum('yjwi,jxw->ixy',v2e_vovo_b,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] += 0.5*np.einsum('jywi,jxw->ixy',v2e_ovvo_ab,r_aaa_u,optimize = True).reshape(-1)

               s[s_aba:f_aba] -= 0.5*np.einsum('yjwi,jxw->ixy',v2e_vovo_a,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] += 0.5*np.einsum('yjiw,jxw->ixy',v2e_voov_ab,r_bbb_u,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('yjwi,jxw->ixy',v2e_vovo_b,r_bbb_u,optimize = True)
               temp += 0.5*np.einsum('jywi,jxw->ixy',v2e_ovvo_ab,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

        if (method == "adc(3)"):

            #print("Calculating additional terms for adc(3)")

############### ADC(3) a - ibc block ############################

               #temp = -0.5*np.einsum('lmwz,lmaj->ajzw',t2_1_a,v2e_oovo_a)
               #temp = temp[:,:,ab_ind_a[0],ab_ind_a[1]]
               #r_aaa = r_aaa.reshape(nocc_a,-1)
               #s[s_a:f_a] += np.einsum('ajp,jp->a',temp, r_aaa, optimize=True)

               t2_1_a_t = t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]]
               r_aaa = r_aaa.reshape(nocc_a,-1)
               temp = 0.5*np.einsum('lmp,jp->lmj',t2_1_a_t,r_aaa)
               s[s_a:f_a] += np.einsum('lmj,lmaj->a',temp, v2e_oovo_a, optimize=True)

               temp_1 = -np.einsum('lmzw,jzw->jlm',t2_1_ab,r_bab)
               s[s_a:f_a] -= np.einsum('jlm,lmaj->a',temp_1, v2e_oovo_ab, optimize=True)

               #temp = -0.5*np.einsum('lmwz,lmaj->ajzw',t2_1_b,v2e_oovo_b)
               #temp = temp[:,:,ab_ind_b[0],ab_ind_b[1]]
               #r_bbb = r_bbb.reshape(nocc_b,-1)
               #s[s_b:f_b] += np.einsum('ajp,jp->a',temp, r_bbb, optimize=True)

               t2_1_b_t = t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]]
               r_bbb = r_bbb.reshape(nocc_b,-1)
               temp = 0.5*np.einsum('lmp,jp->lmj',t2_1_b_t,r_bbb)
               s[s_b:f_b] += np.einsum('lmj,lmaj->a',temp, v2e_oovo_b, optimize=True)

               temp_1 = -np.einsum('mlwz,jzw->jlm',t2_1_ab,r_aba)
               s[s_b:f_b] -= np.einsum('jlm,mlja->a',temp_1, v2e_ooov_ab, optimize=True)

               r_aaa_u = np.zeros((nocc_a,nvir_a,nvir_a))
               r_aaa_u[:,ab_ind_a[0],ab_ind_a[1]]= r_aaa.copy()
               r_aaa_u[:,ab_ind_a[1],ab_ind_a[0]]= -r_aaa.copy()

               r_bbb_u = np.zeros((nocc_b,nvir_b,nvir_b))
               r_bbb_u[:,ab_ind_b[0],ab_ind_b[1]]= r_bbb.copy()
               r_bbb_u[:,ab_ind_b[1],ab_ind_b[0]]= -r_bbb.copy()

               r_bab = r_bab.reshape(nocc_b,nvir_a,nvir_b)
               r_aba = r_aba.reshape(nocc_a,nvir_b,nvir_a)

               temp = np.zeros_like(r_bab)

               temp = np.einsum('jlwd,jzw->lzd',t2_1_a,r_aaa_u,optimize=True)
               temp += np.einsum('ljdw,jzw->lzd',t2_1_ab,r_bab,optimize=True)

               temp_1 = np.zeros_like(r_bab)

               temp_1 = np.einsum('jlwd,jzw->lzd',t2_1_ab,r_aaa_u,optimize=True)
               temp_1 += np.einsum('jlwd,jzw->lzd',t2_1_b,r_bab,optimize=True)

               #temp_2 = np.einsum('ljwd,jwz->lzd',t2_1_ab,r_bab)

               temp_a = t2_1_ab.transpose(0,3,1,2).copy()
               temp_b = temp_a.reshape(nocc_a*nvir_b,nocc_b*nvir_a)
               r_bab_t = r_bab.reshape(nocc_b*nvir_a,-1)
               temp_c = np.dot(temp_b,r_bab_t).reshape(nocc_a,nvir_b,nvir_b)
               temp_2 = temp_c.transpose(0,2,1).copy()

               s[s_a:f_a] += 0.5*np.einsum('lzd,zlad->a',temp,v2e_vovv_a,optimize=True)
               s[s_a:f_a] += 0.5*np.einsum('lzd,zlad->a',temp_1,v2e_vovv_ab,optimize=True)
               s[s_a:f_a] -= 0.5*np.einsum('lzd,lzad->a',temp_2,v2e_ovvv_ab,optimize=True)

               temp = np.zeros_like(r_aba)
               temp = np.einsum('jlwd,jzw->lzd',t2_1_b,r_bbb_u,optimize=True)
               temp += np.einsum('jlwd,jzw->lzd',t2_1_ab,r_aba,optimize=True)

               temp_1 = np.zeros_like(r_aba)
               temp_1 = np.einsum('ljdw,jzw->lzd',t2_1_ab,r_bbb_u,optimize=True)
               temp_1 += np.einsum('jlwd,jzw->lzd',t2_1_a,r_aba,optimize=True)

               temp_2 = np.einsum('jldw,jwz->lzd',t2_1_ab,r_aba,optimize=True)

               s[s_b:f_b] += 0.5*np.einsum('lzd,zlad->a',temp,v2e_vovv_b,optimize=True)
               #s[s_b:f_b] += 0.5*np.einsum('lzd,lzda->a',temp_1,v2e_ovvv_ab,optimize=True)
               temp_a = temp_1.reshape(-1)
               temp_b = v2e_ovvv_ab[:].reshape(nocc_a*nvir_b*nvir_a,-1)
               s[s_b:f_b] += 0.5*np.dot(temp_a,temp_b)
               del temp_b
               s[s_b:f_b] -= 0.5*np.einsum('lzd,zlda->a',temp_2,v2e_vovv_ab,optimize=True)
               temp = np.zeros_like(r_bab)
               temp = -np.einsum('jlzd,jwz->lwd',t2_1_a,r_aaa_u,optimize=True)
               temp += -np.einsum('ljdz,jwz->lwd',t2_1_ab,r_bab,optimize=True)

               temp_1 = np.zeros_like(r_bab)
               temp_1 = -np.einsum('jlzd,jwz->lwd',t2_1_ab,r_aaa_u,optimize=True)
               temp_1 += -np.einsum('jlzd,jwz->lwd',t2_1_b,r_bab,optimize=True)

               temp_2 = -np.einsum('ljzd,jzw->lwd',t2_1_ab,r_bab,optimize=True)

               s[s_a:f_a] -= 0.5*np.einsum('lwd,wlad->a',temp,v2e_vovv_a,optimize=True)
               s[s_a:f_a] -= 0.5*np.einsum('lwd,wlad->a',temp_1,v2e_vovv_ab,optimize=True)
               s[s_a:f_a] += 0.5*np.einsum('lwd,lwad->a',temp_2,v2e_ovvv_ab,optimize=True)

               temp = np.zeros_like(r_aba)
               temp = -np.einsum('jlzd,jwz->lwd',t2_1_b,r_bbb_u,optimize=True)
               temp += -np.einsum('jlzd,jwz->lwd',t2_1_ab,r_aba,optimize=True)

               temp_1 = np.zeros_like(r_bab)
               temp_1 = -np.einsum('ljdz,jwz->lwd',t2_1_ab,r_bbb_u,optimize=True)
               temp_1 += -np.einsum('jlzd,jwz->lwd',t2_1_a,r_aba,optimize=True)

               temp_2 = -np.einsum('jldz,jzw->lwd',t2_1_ab,r_aba,optimize=True)

               s[s_b:f_b] -= 0.5*np.einsum('lwd,wlad->a',temp,v2e_vovv_b,optimize=True)
               #s[s_b:f_b] -= 0.5*np.einsum('lwd,lwda->a',temp_1,v2e_ovvv_ab,optimize=True)
               temp_a = temp_1.reshape(-1)
               temp_b = v2e_ovvv_ab[:].reshape(nocc_a*nvir_b*nvir_a,-1)
               s[s_b:f_b] -= 0.5*np.dot(temp_a,temp_b)
               del temp_b
               s[s_b:f_b] += 0.5*np.einsum('lwd,wlda->a',temp_2,v2e_vovv_ab,optimize=True)

################ ADC(3) ibc - a block ############################

               #t2_1_a_t = t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]]
               #temp = np.einsum('lmp,lmbi->bip',t2_1_a_t,v2e_oovo_a)
               #s[s_aaa:f_aaa] += 0.5*np.einsum('bip,b->ip',temp, r_a, optimize=True).reshape(-1)

               t2_1_a_t = t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]]
               temp = np.einsum('b,lmbi->lmi',r_a,v2e_oovo_a)
               s[s_aaa:f_aaa] += 0.5*np.einsum('lmi,lmp->ip',temp, t2_1_a_t, optimize=True).reshape(-1)

               #temp_1 = np.einsum('lmxy,lmbi->bixy',t2_1_ab,v2e_oovo_ab)
               #s[s_bab:f_bab] += np.einsum('bixy,b->ixy',temp_1, r_a, optimize=True).reshape(-1)

               temp_1 = np.einsum('b,lmbi->lmi',r_a,v2e_oovo_ab)
               s[s_bab:f_bab] += np.einsum('lmi,lmxy->ixy',temp_1, t2_1_ab, optimize=True).reshape(-1)

               #t2_1_b_t = t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]]
               #temp = np.einsum('lmp,lmbi->bip',t2_1_b_t,v2e_oovo_b)
               #s[s_bbb:f_bbb] += 0.5*np.einsum('bip,b->ip',temp, r_b, optimize=True).reshape(-1)

               t2_1_b_t = t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]]
               temp = np.einsum('b,lmbi->lmi',r_b,v2e_oovo_b)
               s[s_bbb:f_bbb] += 0.5*np.einsum('lmi,lmp->ip',temp, t2_1_b_t, optimize=True).reshape(-1)

               #temp_1 = np.einsum('mlyx,mlib->bixy',t2_1_ab,v2e_ooov_ab)
               #s[s_aba:f_aba] += np.einsum('bixy,b->ixy',temp_1, r_b, optimize=True).reshape(-1)

               temp_1 = np.einsum('b,mlib->mli',r_b,v2e_ooov_ab)
               s[s_aba:f_aba] += np.einsum('mli,mlyx->ixy',temp_1, t2_1_ab, optimize=True).reshape(-1)

               temp_1 = np.einsum('xlbd,b->lxd', v2e_vovv_a,r_a,optimize=True)
               temp_2 = np.einsum('xlbd,b->lxd', v2e_vovv_ab,r_a,optimize=True)

               temp  = np.einsum('lxd,ilyd->ixy',temp_1,t2_1_a,optimize=True)
               temp += np.einsum('lxd,ilyd->ixy',temp_2,t2_1_ab,optimize=True)
               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1] ].reshape(-1)

               temp  = np.einsum('lxd,lidy->ixy',temp_1,t2_1_ab,optimize=True)
               temp  += np.einsum('lxd,ilyd->ixy',temp_2,t2_1_b,optimize=True)
               s[s_bab:f_bab] += temp.reshape(-1)

               temp_1 = np.einsum('xlbd,b->lxd', v2e_vovv_b,r_b,optimize=True)
               temp_2 = np.einsum('lxdb,b->lxd', v2e_ovvv_ab,r_b,optimize=True)

               temp  = np.einsum('lxd,ilyd->ixy',temp_1,t2_1_b,optimize=True)
               temp += np.einsum('lxd,lidy->ixy',temp_2,t2_1_ab,optimize=True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1] ].reshape(-1)

               temp  = np.einsum('lxd,ilyd->ixy',temp_1,t2_1_ab,optimize=True)
               temp  += np.einsum('lxd,ilyd->ixy',temp_2,t2_1_a,optimize=True)
               s[s_aba:f_aba] += temp.reshape(-1)

               temp_1 = np.einsum('ylbd,b->lyd', v2e_vovv_a,r_a,optimize=True)
               temp_2 = np.einsum('ylbd,b->lyd', v2e_vovv_ab,r_a,optimize=True)

               temp  = np.einsum('lyd,ilxd->ixy',temp_1,t2_1_a,optimize=True)
               temp += np.einsum('lyd,ilxd->ixy',temp_2,t2_1_ab,optimize=True)
               s[s_aaa:f_aaa] -= temp[:,ab_ind_a[0],ab_ind_a[1] ].reshape(-1)

               temp  = -np.einsum('lybd,b->lyd',v2e_ovvv_ab,r_a,optimize=True)
               temp_1= -np.einsum('lyd,lixd->ixy',temp,t2_1_ab,optimize=True)
               s[s_bab:f_bab] -= temp_1.reshape(-1)

               temp_1 = np.einsum('ylbd,b->lyd', v2e_vovv_b,r_b,optimize=True)
               temp_2 = np.einsum('lydb,b->lyd', v2e_ovvv_ab,r_b,optimize=True)

               temp  = np.einsum('lyd,ilxd->ixy',temp_1,t2_1_b,optimize=True)
               temp += np.einsum('lyd,lidx->ixy',temp_2,t2_1_ab,optimize=True)
               s[s_bbb:f_bbb] -= temp[:,ab_ind_b[0],ab_ind_b[1] ].reshape(-1)

               temp  = -np.einsum('yldb,b->lyd',v2e_vovv_ab,r_b,optimize=True)
               temp_1= -np.einsum('lyd,ildx->ixy',temp,t2_1_ab,optimize=True)
               s[s_aba:f_aba] -= temp_1.reshape(-1)

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
        eris = uadc_ao2mo.transform_integrals(adc)

    v2e_oovv_a,v2e_oovv_ab,v2e_oovv_b = eris.oovv
    v2e_vooo_a,v2e_vooo_ab,v2e_vooo_b = eris.vooo
    v2e_oovo_a,v2e_oovo_ab,v2e_oovo_b = eris.oovo
    v2e_vvoo_a,v2e_vvoo_ab,v2e_vvoo_b = eris.vvoo
    v2e_ooov_a,v2e_ooov_ab,v2e_ooov_b = eris.ooov
    v2e_ovoo_a,v2e_ovoo_ab,v2e_ovoo_b = eris.ovoo
    v2e_vovv_a,v2e_vovv_ab,v2e_vovv_b = eris.vovv
    v2e_vovo_a,v2e_vovo_ab,v2e_vovo_b = eris.vovo
    v2e_oooo_a,v2e_oooo_ab,v2e_oooo_b = eris.oooo
    v2e_vvvo_a,v2e_vvvo_ab,v2e_vvvo_b = eris.vvvo
    v2e_ovov_a,v2e_ovov_ab,v2e_ovov_b = eris.ovov
    v2e_ovvv_a,v2e_ovvv_ab,v2e_ovvv_b = eris.ovvv
    v2e_vvov_a,v2e_vvov_ab,v2e_vvov_b = eris.vvov
    v2e_ovvo_a,v2e_ovvo_ab,v2e_ovvo_b = eris.ovvo
    v2e_voov_a,v2e_voov_ab,v2e_voov_b = eris.voov

    v2e_vooo_1_a = v2e_vooo_a[:,:,ij_ind_a[0],ij_ind_a[1]].transpose(1,0,2).reshape(nocc_a,-1)
    v2e_vooo_1_b = v2e_vooo_b[:,:,ij_ind_b[0],ij_ind_b[1]].transpose(1,0,2).reshape(nocc_b,-1)

    v2e_vooo_1_ab_a = -v2e_ovoo_ab.transpose(0,1,3,2).reshape(nocc_a, -1)
    v2e_vooo_1_ab_b = -v2e_vooo_ab.transpose(1,0,2,3).reshape(nocc_b, -1)

    v2e_oovo_1_a = v2e_oovo_a[ij_ind_a[0],ij_ind_a[1],:,:].transpose(1,0,2)
    v2e_oovo_1_b = v2e_oovo_b[ij_ind_b[0],ij_ind_b[1],:,:].transpose(1,0,2)
    v2e_oovo_1_ab = -v2e_ovoo_ab.transpose(1,3,2,0)
    v2e_oovo_2_ab = -v2e_vooo_ab.transpose(0,2,3,1)

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

        #r_bab = r_bab.reshape(nvir_b,nocc_a,nocc_b)

############ ADC(2) ij block ############################

        s[s_a:f_a] = np.einsum('ij,j->i',M_ij_a,r_a)
        s[s_b:f_b] = np.einsum('ij,j->i',M_ij_b,r_b)

############ ADC(2) i - kja block #########################

        s[s_a:f_a] += np.einsum('ip,p->i', v2e_vooo_1_a, r_aaa, optimize = True)
        s[s_a:f_a] -= np.einsum('ip,p->i', v2e_vooo_1_ab_a, r_bab, optimize = True)

        s[s_b:f_b] += np.einsum('ip,p->i', v2e_vooo_1_b, r_bbb, optimize = True)
        s[s_b:f_b] -= np.einsum('ip,p->i', v2e_vooo_1_ab_b, r_aba, optimize = True)

################ ADC(2) ajk - i block ############################

        s[s_aaa:f_aaa] += np.einsum('api,i->ap', v2e_oovo_1_a, r_a, optimize = True).reshape(-1)
        s[s_bab:f_bab] -= np.einsum('ajki,i->ajk', v2e_oovo_1_ab, r_a, optimize = True).reshape(-1)
        s[s_aba:f_aba] -= np.einsum('ajki,i->ajk', v2e_oovo_2_ab, r_b, optimize = True).reshape(-1)
        s[s_bbb:f_bbb] += np.einsum('api,i->ap', v2e_oovo_1_b, r_b, optimize = True).reshape(-1)

################ ADC(2) ajk - bil block ############################

        s[s_aaa:f_aaa] += D_aij_a * r_aaa
        s[s_bab:f_bab] += D_aij_bab * r_bab.reshape(-1)
        s[s_aba:f_aba] += D_aij_aba * r_aba.reshape(-1)
        s[s_bbb:f_bbb] += D_aij_b * r_bbb

############### ADC(3) ajk - bil block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

               t2_2_a, t2_2_ab, t2_2_b = adc.t2[1]

              #print("Calculating additional terms for adc(2)-e")

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

               temp = 0.5*np.einsum('jkli,ail->ajk',v2e_oooo_a,r_aaa_u ,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

               temp = 0.5*np.einsum('jkli,ail->ajk',v2e_oooo_b,r_bbb_u,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

               s[s_bab:f_bab] -= 0.5*np.einsum('kjil,ali->ajk',v2e_oooo_ab,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] -= 0.5*np.einsum('kjli,ail->ajk',v2e_oooo_ab,r_bab,optimize = True).reshape(-1)

               s[s_aba:f_aba] -= 0.5*np.einsum('jkli,ali->ajk',v2e_oooo_ab,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= 0.5*np.einsum('jkil,ail->ajk',v2e_oooo_ab,r_aba,optimize = True).reshape(-1)

               temp = 0.5*np.einsum('bkal,bjl->ajk',v2e_vovo_a,r_aaa_u,optimize = True)
               temp += 0.5* np.einsum('kbal,blj->ajk',v2e_ovvo_ab,r_bab,optimize = True)

               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] += 0.5*np.einsum('kbla,bjl->ajk',v2e_ovov_ab,r_bab,optimize = True).reshape(-1)

               temp_1 = 0.5*np.einsum('bkal,bjl->ajk',v2e_vovo_b,r_bbb_u,optimize = True)
               temp_1 += 0.5*np.einsum('bkla,blj->ajk',v2e_voov_ab,r_aba,optimize = True)

               s[s_bbb:f_bbb] += temp_1[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

               s[s_aba:f_aba] += 0.5*np.einsum('bkal,bjl->ajk',v2e_vovo_ab,r_aba,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('bjal,bkl->ajk',v2e_vovo_a,r_aaa_u,optimize = True)
               temp -= 0.5*np.einsum('jbal,blk->ajk',v2e_ovvo_ab,r_bab,optimize = True)

               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] +=  0.5*np.einsum('bjla,bkl->ajk',v2e_voov_ab,r_aaa_u,optimize = True).reshape(-1)
               s[s_bab:f_bab] +=  0.5*np.einsum('bjal,blk->ajk',v2e_vovo_b,r_bab,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('bjal,bkl->ajk',v2e_vovo_b,r_bbb_u,optimize = True)
               temp -= 0.5*np.einsum('bjla,blk->ajk',v2e_voov_ab,r_aba,optimize = True)

               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

               s[s_aba:f_aba] += 0.5*np.einsum('bjal,blk->ajk',v2e_vovo_a,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] += 0.5*np.einsum('jbal,bkl->ajk',v2e_ovvo_ab,r_bbb_u,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('bkai,bij->ajk',v2e_vovo_a,r_aaa_u,optimize = True)
               temp += 0.5*np.einsum('kbai,bij->ajk',v2e_ovvo_ab,r_bab,optimize = True)

               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] += 0.5*np.einsum('kbia,bji->ajk',v2e_ovov_ab,r_bab,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('bkai,bij->ajk',v2e_vovo_b,r_bbb_u,optimize = True)
               temp += 0.5*np.einsum('bkia,bij->ajk',v2e_voov_ab,r_aba,optimize = True)

               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

               s[s_aba:f_aba] += 0.5*np.einsum('bkai,bji->ajk',v2e_vovo_ab,r_aba,optimize = True).reshape(-1)

               temp = 0.5*np.einsum('bjai,bik->ajk',v2e_vovo_a,r_aaa_u,optimize = True)
               temp -= 0.5*np.einsum('jbai,bik->ajk',v2e_ovvo_ab,r_bab,optimize = True)

               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] += 0.5*np.einsum('bjai,bik->ajk',v2e_vovo_b,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] -= 0.5*np.einsum('bjia,bik->ajk',v2e_voov_ab,r_aaa_u,optimize = True).reshape(-1)

               s[s_aba:f_aba] += 0.5*np.einsum('bjai,bik->ajk',v2e_vovo_a,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= 0.5*np.einsum('jbai,bik->ajk',v2e_ovvo_ab,r_bbb_u,optimize = True).reshape(-1)

               temp = 0.5*np.einsum('bjai,bik->ajk',v2e_vovo_b,r_bbb_u,optimize = True)
               temp -= 0.5*np.einsum('bjia,bik->ajk',v2e_voov_ab,r_aba,optimize = True)

               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

        if (method == "adc(3)"):

           #print("Calculating additional terms for adc(3)")

################ ADC(3) i - kja block ############################

               #t2_1_a_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:]
               #temp = np.einsum('pbc,bcai->pai',t2_1_a_t,v2e_vvvo_a)
               #r_aaa = r_aaa.reshape(nvir_a,-1)
               #s[s_a:f_a] += 0.5*np.einsum('pai,ap->i',temp, r_aaa, optimize=True)

               r_aaa = r_aaa.reshape(nvir_a,-1)
               t2_1_a_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:].copy()
               temp = np.einsum('pbc,ap->abc',t2_1_a_t,r_aaa, optimize=True)
               s[s_a:f_a] += 0.5*np.einsum('abc,bcai->i',temp, v2e_vvvo_a, optimize=True)

               temp_1 = np.einsum('kjcb,ajk->abc',t2_1_ab,r_bab, optimize=True)
               s[s_a:f_a] += np.einsum('abc,cbia->i',temp_1, v2e_vvov_ab, optimize=True)

               #t2_1_b_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:]
               #temp = np.einsum('pbc,bcai->pai',t2_1_b_t,v2e_vvvo_b)
               #r_bbb = r_bbb.reshape(nvir_b,-1)
               #s[s_b:f_b] += 0.5*np.einsum('pai,ap->i',temp, r_bbb, optimize=True)

               r_bbb = r_bbb.reshape(nvir_b,-1)
               t2_1_b_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:].copy()
               temp = np.einsum('pbc,ap->abc',t2_1_b_t,r_bbb, optimize=True)
               s[s_b:f_b] += 0.5*np.einsum('abc,bcai->i',temp, v2e_vvvo_b, optimize=True)

               temp_1 = np.einsum('jkbc,ajk->abc',t2_1_ab,r_aba, optimize=True)
               s[s_b:f_b] += np.einsum('abc,bcai->i',temp_1, v2e_vvvo_ab, optimize=True)

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

               s[s_a:f_a] += 0.5*np.einsum('blk,ilkb->i',temp,v2e_ooov_a,optimize=True)
               s[s_a:f_a] += 0.5*np.einsum('blk,ilkb->i',temp_1,v2e_ooov_ab,optimize=True)
               s[s_a:f_a] -= 0.5*np.einsum('blk,ilbk->i',temp_2,v2e_oovo_ab,optimize=True)

               temp = np.zeros_like(r_aba)
               temp = np.einsum('jlab,ajk->blk',t2_1_b,r_bbb_u,optimize=True)
               temp += np.einsum('jlab,ajk->blk',t2_1_ab,r_aba,optimize=True)

               temp_1 = np.zeros_like(r_aba)
               temp_1 = np.einsum('ljba,ajk->blk',t2_1_ab,r_bbb_u,optimize=True)
               temp_1 += np.einsum('jlab,ajk->blk',t2_1_a,r_aba,optimize=True)

               temp_2 = np.einsum('ljab,akj->blk',t2_1_ab,r_aba,optimize=True)

               s[s_b:f_b] += 0.5*np.einsum('blk,ilkb->i',temp,v2e_ooov_b,optimize=True)
               s[s_b:f_b] += 0.5*np.einsum('blk,libk->i',temp_1,v2e_oovo_ab,optimize=True)
               s[s_b:f_b] -= 0.5*np.einsum('blk,likb->i',temp_2,v2e_ooov_ab,optimize=True)

               temp = np.zeros_like(r_bab)
               temp = -np.einsum('klab,akj->blj',t2_1_a,r_aaa_u,optimize=True)
               temp -= np.einsum('lkba,akj->blj',t2_1_ab,r_bab,optimize=True)

               temp_1 = np.zeros_like(r_bab)
               temp_1 = -np.einsum('klab,akj->blj',t2_1_ab,r_aaa_u,optimize=True)
               temp_1 -= np.einsum('klab,akj->blj',t2_1_b,r_bab,optimize=True)

               temp_2 = -np.einsum('klba,ajk->blj',t2_1_ab,r_bab,optimize=True)

               s[s_a:f_a] -= 0.5*np.einsum('blj,iljb->i',temp,v2e_ooov_a,optimize=True)
               s[s_a:f_a] -= 0.5*np.einsum('blj,iljb->i',temp_1,v2e_ooov_ab,optimize=True)
               s[s_a:f_a] += 0.5*np.einsum('blj,ilbj->i',temp_2,v2e_oovo_ab,optimize=True)

               temp = np.zeros_like(r_aba)
               temp = -np.einsum('klab,akj->blj',t2_1_b,r_bbb_u,optimize=True)
               temp -= np.einsum('klab,akj->blj',t2_1_ab,r_aba,optimize=True)

               temp_1 = np.zeros_like(r_bab)
               temp_1 = -np.einsum('lkba,akj->blj',t2_1_ab,r_bbb_u,optimize=True)
               temp_1 -= np.einsum('klab,akj->blj',t2_1_a,r_aba,optimize=True)

               temp_2 = -np.einsum('lkab,ajk->blj',t2_1_ab,r_aba,optimize=True)

               s[s_b:f_b] -= 0.5*np.einsum('blj,iljb->i',temp,v2e_ooov_b,optimize=True)
               s[s_b:f_b] -= 0.5*np.einsum('blj,libj->i',temp_1,v2e_oovo_ab,optimize=True)
               s[s_b:f_b] += 0.5*np.einsum('blj,lijb->i',temp_2,v2e_ooov_ab,optimize=True)

################ ADC(3) ajk - i block ############################
               #t2_1_a_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:]
               #temp = 0.5*np.einsum('pbc,bcai->api',t2_1_a_t,v2e_vvvo_a)
               #s[s_aaa:f_aaa] += np.einsum('api,i->ap',temp, r_a, optimize=True).reshape(-1)

               t2_1_a_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:].copy()
               temp = np.einsum('i,bcai->bca',r_a,v2e_vvvo_a,optimize=True)
               s[s_aaa:f_aaa] += 0.5*np.einsum('bca,pbc->ap',temp,t2_1_a_t,optimize=True).reshape(-1)

               #temp_1 = np.einsum('kjcb,cbia->iajk',t2_1_ab,v2e_vvov_ab)
               #temp_1 = temp_1.reshape(nocc_a,-1)
               #s[s_bab:f_bab] += np.einsum('ip,i->p',temp_1, r_a, optimize=True).reshape(-1)

               temp_1 = np.einsum('i,cbia->cba',r_a,v2e_vvov_ab,optimize=True)
               s[s_bab:f_bab] += np.einsum('cba,kjcb->ajk',temp_1, t2_1_ab, optimize=True).reshape(-1)

               #t2_1_b_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:]
               #temp = 0.5*np.einsum('pbc,bcai->api',t2_1_b_t,v2e_vvvo_b)
               #s[s_bbb:f_bbb] += np.einsum('api,i->ap',temp, r_b, optimize=True).reshape(-1)

               t2_1_b_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:].copy()
               temp = np.einsum('i,bcai->bca',r_b,v2e_vvvo_b,optimize=True)
               s[s_bbb:f_bbb] += 0.5*np.einsum('bca,pbc->ap',temp,t2_1_b_t,optimize=True).reshape(-1)

               #temp_1 = np.einsum('jkbc,bcai->iajk',t2_1_ab,v2e_vvvo_ab)
               #temp_1 = temp_1.reshape(nocc_b,-1)
               #s[s_aba:f_aba] += np.einsum('ip,i->p',temp_1, r_b, optimize=True).reshape(-1)

               temp_1 = np.einsum('i,bcai->bca',r_b,v2e_vvvo_ab,optimize=True)
               s[s_aba:f_aba] += np.einsum('bca,jkbc->ajk',temp_1, t2_1_ab, optimize=True).reshape(-1)

               temp_1 = np.einsum('i,kbil->kbl',r_a, v2e_ovoo_a)
               temp_2 = np.einsum('i,kbil->kbl',r_a, v2e_ovoo_ab)

               temp  = np.einsum('kbl,jlab->ajk',temp_1,t2_1_a,optimize=True)
               temp += np.einsum('kbl,jlab->ajk',temp_2,t2_1_ab,optimize=True)
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1] ].reshape(-1)

               temp_1  = np.einsum('i,kbil->kbl',r_a,v2e_ovoo_a)
               temp_2  = np.einsum('i,kbil->kbl',r_a,v2e_ovoo_ab)

               temp  = np.einsum('kbl,ljba->ajk',temp_1,t2_1_ab,optimize=True)
               temp += np.einsum('kbl,jlab->ajk',temp_2,t2_1_b,optimize=True)
               s[s_bab:f_bab] += temp.reshape(-1)

               temp_1 = np.einsum('i,kbil->kbl',r_b, v2e_ovoo_b)
               temp_2 = np.einsum('i,bkli->kbl',r_b, v2e_vooo_ab)

               temp  = np.einsum('kbl,jlab->ajk',temp_1,t2_1_b,optimize=True)
               temp += np.einsum('kbl,ljba->ajk',temp_2,t2_1_ab,optimize=True)
               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1] ].reshape(-1)

               temp_1  = np.einsum('i,kbil->kbl',r_b,v2e_ovoo_b)
               temp_2  = np.einsum('i,bkli->kbl',r_b,v2e_vooo_ab)

               temp  = np.einsum('kbl,jlab->ajk',temp_1,t2_1_ab,optimize=True)
               temp += np.einsum('kbl,jlab->ajk',temp_2,t2_1_a,optimize=True)
               s[s_aba:f_aba] += temp.reshape(-1)

               temp_1 = np.einsum('i,jbil->jbl',r_a, v2e_ovoo_a)
               temp_2 = np.einsum('i,jbil->jbl',r_a, v2e_ovoo_ab)

               temp  = np.einsum('jbl,klab->ajk',temp_1,t2_1_a,optimize=True)
               temp += np.einsum('jbl,klab->ajk',temp_2,t2_1_ab,optimize=True)
               s[s_aaa:f_aaa] -= temp[:,ij_ind_a[0],ij_ind_a[1] ].reshape(-1)

               temp  = -np.einsum('i,bjil->jbl',r_a,v2e_vooo_ab,optimize=True)
               temp_1 = -np.einsum('jbl,klba->ajk',temp,t2_1_ab,optimize=True)
               s[s_bab:f_bab] -= temp_1.reshape(-1)

               temp_1 = np.einsum('i,jbil->jbl',r_b, v2e_ovoo_b)
               temp_2 = np.einsum('i,bjli->jbl',r_b, v2e_vooo_ab)

               temp  = np.einsum('jbl,klab->ajk',temp_1,t2_1_b,optimize=True)
               temp += np.einsum('jbl,lkba->ajk',temp_2,t2_1_ab,optimize=True)
               s[s_bbb:f_bbb] -= temp[:,ij_ind_b[0],ij_ind_b[1] ].reshape(-1)

               temp  = -np.einsum('i,jbli->jbl',r_b,v2e_ovoo_ab,optimize=True)
               temp_1 = -np.einsum('jbl,lkab->ajk',temp,t2_1_ab,optimize=True)
               s[s_aba:f_aba] -= temp_1.reshape(-1)

        s *= -1.0

        return s

    return sigma_

def ea_compute_trans_moments(adc, orb, eris=None, spin="alpha"):

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

    if eris is None:
        eris = uadc_ao2mo.transform_integrals(adc)

    v2e_oovv_a , v2e_oovv_ab, v2e_oovv_b = eris.oovv
    v2e_vvvo_a , v2e_vvvo_ab, v2e_vvvo_b = eris.vvvo
    v2e_ovoo_a , v2e_ovoo_ab, v2e_ovoo_b = eris.ovoo
    v2e_voov_a , v2e_voov_ab, v2e_voov_b = eris.voov
    v2e_ovov_a , v2e_ovov_ab, v2e_ovov_b = eris.ovov
    v2e_vovv_a , v2e_vovv_ab, v2e_vovv_b = eris.vovv
    v2e_ooov_a , v2e_ooov_ab, v2e_ooov_b = eris.ooov

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

def ip_compute_trans_moments(adc, orb, eris=None, spin="alpha"):

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

    if eris is None:
        eris = uadc_ao2mo.transform_integrals(adc)

    v2e_oovv_a , v2e_oovv_ab, v2e_oovv_b = eris.oovv
    v2e_vvvo_a , v2e_vvvo_ab, v2e_vvvo_b = eris.vvvo
    v2e_ovoo_a , v2e_ovoo_ab, v2e_ovoo_b = eris.ovoo
    v2e_voov_a , v2e_voov_ab, v2e_voov_b = eris.voov
    v2e_ovov_a , v2e_ovov_ab, v2e_ovov_b = eris.ovov
    v2e_vovv_a , v2e_vovv_ab, v2e_vovv_b = eris.vovv
    v2e_ooov_a , v2e_ooov_ab, v2e_ooov_b = eris.ooov

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

######## ADC(2) 1h part  ############################################

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
    
    def get_trans_moments(self, nroots=1, eris = None):
    
        nmo_a  = self.nmo_a
        nmo_b  = self.nmo_b
    
        T_a = []
        T_b = []
    
        for orb in range(nmo_a):
    
                T_aa = self.compute_trans_moments(orb, eris = eris, spin = "alpha")
                T_a.append(T_aa)
    
        for orb in range(nmo_b):
    
                T_bb = self.compute_trans_moments(orb, eris = eris, spin = "beta")
                T_b.append(T_bb) 
        
        return (T_a, T_b)

    def get_spec_factors(self, nroots=1, T=(None,None), U=None):
    
        nmo_a  = self.nmo_a
        nmo_b  = self.nmo_b
    
        P = np.zeros((nroots))
    
        T_a = T[0]
        T_b = T[1]    
        U = np.array(U)
    
        for orb in range(nmo_a):
    
            T_aa = T_a[orb]
            T_aa = np.dot(T_aa, U.T)
            if nroots == 1:
                P += np.square(np.absolute(T_aa))
            else :    
                for i in range(nroots):
                    P[i] += np.square(np.absolute(T_aa[i]))
    
        for orb in range(nmo_b):
    
            T_bb = T_b[orb]
            T_bb = np.dot(T_bb, U.T)
            if nroots == 1:
                P += np.square(np.absolute(T_bb))
            else :    
                for i in range(nroots):
                    P[i] += np.square(np.absolute(T_bb[i]))
    
        return P

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

    def get_trans_moments(self, nroots=1, eris=None):

        nmo_a  = self.nmo_a
        nmo_b  = self.nmo_b

        T_a = []
        T_b = []

        for orb in range(nmo_a):
    
                T_aa = self.compute_trans_moments(orb, eris, spin = "alpha")
                T_a.append(T_aa)

        for orb in range(nmo_b):
    
                T_bb = self.compute_trans_moments(orb, eris, spin = "beta")
                T_b.append(T_bb) 
        
        return (T_a, T_b)

    def get_spec_factors(self, nroots=1, T=(None,None), U=None):
    
        nmo_a  = self.nmo_a
        nmo_b  = self.nmo_b
    
        P = np.zeros((nroots))
   
        T_a = T[0]
        T_b = T[1]    
        U = np.array(U)

        for orb in range(nmo_a):

            T_aa = T_a[orb]
            T_aa = np.dot(T_aa, U.T)
            if nroots == 1:
                P += np.square(np.absolute(T_aa))
            else :    
                for i in range(nroots):
                    P[i] += np.square(np.absolute(T_aa[i]))
   
        for orb in range(nmo_b):

            T_bb = T_b[orb]
            T_bb = np.dot(T_bb, U.T)
            if nroots == 1:
                P += np.square(np.absolute(T_bb))
            else :    
                for i in range(nroots):
                    P[i] += np.square(np.absolute(T_bb[i]))

        return P


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
