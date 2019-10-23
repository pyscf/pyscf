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
# Author: Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

'''
Unrestricted algebraic diagrammatic construction
'''

import time
#import ctypes
#from functools import reduce
import numpy as np
#from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.adc import uadc_ao2mo
#from pyscf.ao2mo import _ao2mo
#from pyscf.cc import _ccsd
#from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, _mo_without_core
from pyscf import __config__
#
#BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
#MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

def kernel(myadc, eris, verbose=None):

    log = logger.new_logger(myadc, verbose)
    if eris is None:
        #eris = mycc.ao2mo(mycc.mo_coeff)
        # TODO: transform integrals if they are not provided
        raise NotImplementedError('Integrals for UADC amplitudes')

    cput0 = (time.clock(), time.time())

    t1, t2 = myadc.compute_amplitudes(eris)
    e_corr = myadc.energy(t1, t2, eris)

    log.info('E(corr) = %.15g', e_corr)
    #log.timer('ADC ground-state energy', *cput0)
    return e_corr, t1, t2

def compute_amplitudes(myadc, eris):

    t1 = (None,) 
    t2 = (None,) 

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
 
    t1_3 = (t1_3_a , t1_3_b)

    t1 = (t1_2, t1_3)
    t2 = (t2_1, t2_2)

    return t1, t2

def energy(myadc, t1, t2, eris):

    v2e_oovv_a, v2e_oovv_ab, v2e_oovv_b = eris.oovv

    t2_1_a, t2_1_ab, t2_1_b  = t2[0]

    e_mp2 = 0.25 * np.einsum('ijab,ijab', t2_1_a, v2e_oovv_a)
    e_mp2 += np.einsum('ijab,ijab', t2_1_ab, v2e_oovv_ab)
    e_mp2 += 0.25 * np.einsum('ijab,ijab', t2_1_b, v2e_oovv_b)

    e_corr = e_mp2
    return e_corr

class UADC(lib.StreamObject):

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
        self.scf_energy = mf.scf()

        self.frozen = frozen
        self.incore_complete = self.incore_complete or self.mol.incore_anyway

        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.e_corr = None
        self.t1 = None
        self.t2 = None
        self._nocc = mf.nelec
        self._nmo = (mo_coeff[0].shape[1], mo_coeff[1].shape[1])
        self._nvir = (self._nmo[0] - self._nocc[0], self._nmo[1] - self._nocc[1])
        self.mo_energy_a = mf.mo_energy[0]
        self.mo_energy_b = mf.mo_energy[1]
        self.chkfile = mf.chkfile

    compute_amplitudes = compute_amplitudes
    energy = energy

    def kernel(self):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        # TODO: Implement
        #self.dump_flags()

        # TODO: ao2mo transformation if eris is None
        eris = uadc_ao2mo.transform_integrals(self)
        

        self.e_corr, self.t1, self.t2 = kernel(self, eris, verbose=self.verbose)

        # TODO: Implement
        #self._finalize()
        return self.e_corr, self.t1, self.t2

# TODO: add a test main section
