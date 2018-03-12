#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

'''
Intermediates for restricted CCSD.  Complex integrals are supported.
'''

import numpy as np
from pyscf import lib
from pyscf import ao2mo

# This is restricted (R)CCSD
# Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)

### Eqs. (37)-(39) "kappa"

def cc_Foo(t1, t2, eris):
    nocc, nvir = t1.shape
    foo = eris.fock[:nocc,:nocc]
    eris_ovov = np.asarray(eris.ovov)
    Fki  = 2*lib.einsum('kcld,ilcd->ki', eris_ovov, t2)
    Fki -=   lib.einsum('kdlc,ilcd->ki', eris_ovov, t2)
    Fki += 2*lib.einsum('kcld,ic,ld->ki', eris_ovov, t1, t1)
    Fki -=   lib.einsum('kdlc,ic,ld->ki', eris_ovov, t1, t1)
    Fki += foo
    return Fki

def cc_Fvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fvv = eris.fock[nocc:,nocc:]
    eris_ovov = np.asarray(eris.ovov)
    Fac  =-2*lib.einsum('kcld,klad->ac', eris_ovov, t2)
    Fac +=   lib.einsum('kdlc,klad->ac', eris_ovov, t2)
    Fac -= 2*lib.einsum('kcld,ka,ld->ac', eris_ovov, t1, t1)
    Fac +=   lib.einsum('kdlc,ka,ld->ac', eris_ovov, t1, t1)
    Fac += fvv
    return Fac

def cc_Fov(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    eris_ovov = np.asarray(eris.ovov)
    Fkc  = 2*np.einsum('kcld,ld->kc', eris_ovov, t1)
    Fkc -=   np.einsum('kdlc,ld->kc', eris_ovov, t1)
    Fkc += fov
    return Fkc

### Eqs. (40)-(41) "lambda"

def Loo(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Lki = cc_Foo(t1, t2, eris) + np.einsum('kc,ic->ki',fov, t1)
    eris_ovoo = np.asarray(eris.ovoo)
    Lki += 2*np.einsum('lcki,lc->ki', eris_ovoo, t1)
    Lki -=   np.einsum('kcli,lc->ki', eris_ovoo, t1)
    return Lki

def Lvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Lac = cc_Fvv(t1, t2, eris) - np.einsum('kc,ka->ac',fov, t1)
    eris_ovvv = np.asarray(eris.get_ovvv())
    Lac += 2*np.einsum('kdac,kd->ac', eris_ovvv, t1)
    Lac -=   np.einsum('kcad,kd->ac', eris_ovvv, t1)
    return Lac

### Eqs. (42)-(45) "chi"

def cc_Woooo(t1, t2, eris):
    eris_ovoo = np.asarray(eris.ovoo)
    Wklij  = lib.einsum('lcki,jc->klij', eris_ovoo, t1)
    Wklij += lib.einsum('kclj,ic->klij', eris_ovoo, t1)
    eris_ovov = np.asarray(eris.ovov)
    Wklij += lib.einsum('kcld,ijcd->klij', eris_ovov, t2)
    Wklij += lib.einsum('kcld,ic,jd->klij', eris_ovov, t1, t1)
    Wklij += np.asarray(eris.oooo).transpose(0,2,1,3)
    return Wklij

def cc_Wvvvv(t1, t2, eris):
    # Incore 
    eris_ovvv = np.asarray(eris.get_ovvv())
    Wabcd  = lib.einsum('kdac,kb->abcd', eris_ovvv,-t1)
    Wabcd -= lib.einsum('kcbd,ka->abcd', eris_ovvv, t1)
    Wabcd += np.asarray(_get_vvvv(eris)).transpose(0,2,1,3)
    return Wabcd

def cc_Wvoov(t1, t2, eris):
    eris_ovvv = np.asarray(eris.get_ovvv())
    eris_ovoo = np.asarray(eris.ovoo)
    Wakic  = lib.einsum('kcad,id->akic', eris_ovvv, t1)
    Wakic -= lib.einsum('kcli,la->akic', eris_ovoo, t1)
    Wakic += np.asarray(eris.ovvo).transpose(2,0,3,1)
    eris_ovov = np.asarray(eris.ovov)
    Wakic -= 0.5*lib.einsum('ldkc,ilda->akic', eris_ovov, t2)
    Wakic -= 0.5*lib.einsum('lckd,ilad->akic', eris_ovov, t2)
    Wakic -= lib.einsum('ldkc,id,la->akic', eris_ovov, t1, t1)
    Wakic += lib.einsum('ldkc,ilad->akic', eris_ovov, t2)
    return Wakic

def cc_Wvovo(t1, t2, eris):
    eris_ovvv = np.asarray(eris.get_ovvv())
    eris_ovoo = np.asarray(eris.ovoo)
    Wakci  = lib.einsum('kdac,id->akci', eris_ovvv, t1)
    Wakci -= lib.einsum('lcki,la->akci', eris_ovoo, t1)
    Wakci += np.asarray(eris.oovv).transpose(2,0,3,1)
    eris_ovov = np.asarray(eris.ovov)
    Wakci -= 0.5*lib.einsum('lckd,ilda->akci', eris_ovov, t2)
    Wakci -= lib.einsum('lckd,id,la->akci', eris_ovov, t1, t1)
    return Wakci

def Wooov(t1, t2, eris):
    eris_ovov = np.asarray(eris.ovov)
    Wklid  = lib.einsum('ic,kcld->klid', t1, eris_ovov)
    Wklid += np.asarray(eris.ovoo).transpose(2,0,3,1)
    return Wklid

def Wvovv(t1, t2, eris):
    eris_ovov = np.asarray(eris.ovov)
    Walcd  = lib.einsum('ka,kcld->alcd',-t1, eris_ovov)
    Walcd += np.asarray(eris.get_ovvv()).transpose(2,0,3,1)
    return Walcd

def W1ovvo(t1, t2, eris):
    eris_ovov = np.asarray(eris.ovov)
    Wkaci  = 2*lib.einsum('kcld,ilad->kaci', eris_ovov, t2)
    Wkaci +=  -lib.einsum('kcld,liad->kaci', eris_ovov, t2)
    Wkaci +=  -lib.einsum('kdlc,ilad->kaci', eris_ovov, t2)
    Wkaci += np.asarray(eris.ovvo).transpose(0,2,1,3)
    return Wkaci

def W2ovvo(t1, t2, eris):
    Wkaci = lib.einsum('la,lkic->kaci',-t1, Wooov(t1, t2, eris))
    eris_ovvv = np.asarray(eris.get_ovvv())
    Wkaci += lib.einsum('kcad,id->kaci', eris_ovvv, t1)
    return Wkaci

def Wovvo(t1, t2, eris):
    Wkaci = W1ovvo(t1, t2, eris) + W2ovvo(t1, t2, eris)
    return Wkaci

def W1ovov(t1, t2, eris):
    eris_ovov = np.asarray(eris.ovov)
    Wkbid = -lib.einsum('kcld,ilcb->kbid', eris_ovov, t2)
    Wkbid += np.asarray(eris.oovv).transpose(0,2,1,3)
    return Wkbid

def W2ovov(t1, t2, eris):
    Wkbid = lib.einsum('klid,lb->kbid', Wooov(t1, t2, eris),-t1)
    eris_ovvv = np.asarray(eris.get_ovvv())
    Wkbid += lib.einsum('kcbd,ic->kbid', eris_ovvv, t1)
    return Wkbid

def Wovov(t1, t2, eris):
    return W1ovov(t1, t2, eris) + W2ovov(t1, t2, eris)

def Woooo(t1, t2, eris):
    eris_ovov = np.asarray(eris.ovov)
    Wklij  = lib.einsum('kcld,ijcd->klij', eris_ovov, t2)
    Wklij += lib.einsum('kcld,ic,jd->klij', eris_ovov, t1, t1)
    eris_ovoo = np.asarray(eris.ovoo)
    Wklij += lib.einsum('ldki,jd->klij', eris_ovoo, t1)
    Wklij += lib.einsum('kclj,ic->klij', eris_ovoo, t1)
    Wklij += np.asarray(eris.oooo).transpose(0,2,1,3)
    return Wklij

def Wvvvv(t1, t2, eris):
    eris_ovov = np.asarray(eris.ovov)
    Wabcd  = lib.einsum('kcld,klab->abcd', eris_ovov, t2)
    Wabcd += lib.einsum('kcld,ka,lb->abcd', eris_ovov, t1, t1)
    Wabcd += np.asarray(_get_vvvv(eris)).transpose(0,2,1,3)
    eris_ovvv = np.asarray(eris.get_ovvv())
    Wabcd -= lib.einsum('ldac,lb->abcd', eris_ovvv, t1)
    Wabcd -= lib.einsum('kcbd,ka->abcd', eris_ovvv, t1)
    return Wabcd

def Wvvvo(t1, t2, eris, _Wvvvv=None):
    nocc,nvir = t1.shape
    eris_ovvv = np.asarray(eris.get_ovvv())
    # Check if t1=0 (HF+MBPT(2))
    # don't make vvvv if you can avoid it!
    Wabcj  =  -lib.einsum('alcj,lb->abcj', W1ovov(t1, t2, eris).transpose(1,0,3,2), t1)
    Wabcj +=  -lib.einsum('kbcj,ka->abcj', W1ovvo(t1, t2, eris), t1)
    Wabcj += 2*lib.einsum('ldac,ljdb->abcj', eris_ovvv, t2)
    Wabcj +=  -lib.einsum('ldac,ljbd->abcj', eris_ovvv, t2)
    Wabcj +=  -lib.einsum('lcad,ljdb->abcj', eris_ovvv, t2)
    Wabcj +=  -lib.einsum('kcbd,jkda->abcj', eris_ovvv, t2)
    eris_ovoo = np.asarray(eris.ovoo)
    Wabcj +=   lib.einsum('kclj,lkba->abcj', eris_ovoo, t2)
    Wabcj +=   lib.einsum('kclj,lb,ka->abcj', eris_ovoo, t1, t1)
    Wabcj +=  -lib.einsum('kc,kjab->abcj', cc_Fov(t1, t2, eris), t2)
    Wabcj += np.asarray(eris_ovvv).transpose(3,1,2,0).conj()
    if np.any(t1):
        if _Wvvvv is None:
            _Wvvvv = Wvvvv(t1, t2, eris)
        Wabcj += lib.einsum('abcd,jd->abcj', _Wvvvv, t1)
    return Wabcj

def Wovoo(t1, t2, eris):
    eris_ovoo = np.asarray(eris.ovoo)
    eris_ovvv = np.asarray(eris.get_ovvv())
    Wkbij  =   lib.einsum('kbid,jd->kbij', W1ovov(t1, t2, eris), t1)
    Wkbij +=  -lib.einsum('klij,lb->kbij', Woooo(t1, t2, eris), t1)
    Wkbij +=   lib.einsum('kbcj,ic->kbij', W1ovvo(t1, t2, eris), t1)
    Wkbij += 2*lib.einsum('ldki,ljdb->kbij', eris_ovoo, t2)
    Wkbij +=  -lib.einsum('ldki,jldb->kbij', eris_ovoo, t2)
    Wkbij +=  -lib.einsum('kdli,ljdb->kbij', eris_ovoo, t2)
    Wkbij +=   lib.einsum('kcbd,jidc->kbij', eris_ovvv, t2)
    Wkbij +=   lib.einsum('kcbd,jd,ic->kbij', eris_ovvv, t1, t1)
    Wkbij +=  -lib.einsum('kclj,libc->kbij', eris_ovoo, t2)
    Wkbij +=   lib.einsum('kc,ijcb->kbij', cc_Fov(t1, t2, eris), t2)
    Wkbij += np.asarray(eris_ovoo).transpose(3,1,2,0).conj()
    return Wkbij

def _get_vvvv(eris):
    if eris.vvvv is None and hasattr(eris, 'vvL'):  # DF eris
        vvL = np.asarray(eris.vvL)
        nvir = int(np.sqrt(eris.vvL.shape[0]*2))
        return ao2mo.restore(1, lib.dot(vvL, vvL.T), nvir)
    elif len(eris.vvvv.shape) == 2:  # DO not use .ndim here for h5py library
                                     # backward compatbility
        nvir = int(np.sqrt(eris.vvvv.shape[0]*2))
        return ao2mo.restore(1, np.asarray(eris.vvvv), nvir)
    else:
        return eris.vvvv
