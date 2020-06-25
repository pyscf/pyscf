#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Author: Yang Gao <younggao1994@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>

from symtensor.sym_ctf import einsum

# This is restricted (R)CCSD
# Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)

### Eqs. (37)-(39) "kappa"

def cc_Foo(t1, t2, eris):
    Fki  = 2*einsum('kcld,ilcd->ki', eris.ovov, t2)
    Fki -= einsum('kdlc,ilcd->ki', eris.ovov, t2)
    tmp  = einsum('kcld,ld->kc', eris.ovov, t1)
    Fki += 2*einsum('kc,ic->ki', tmp, t1)
    tmp  = einsum('kdlc,ld->kc', eris.ovov, t1)
    Fki -= einsum('kc,ic->ki', tmp, t1)
    Fki += eris.foo
    return Fki

def cc_Fvv(t1, t2, eris):
    Fac  =-2*einsum('kcld,klad->ac', eris.ovov, t2)
    Fac +=   einsum('kdlc,klad->ac', eris.ovov, t2)
    tmp  =   einsum('kcld,ld->kc', eris.ovov, t1)
    Fac -= 2*einsum('kc,ka->ac', tmp, t1)
    tmp  =   einsum('kdlc,ld->kc', eris.ovov, t1)
    Fac +=   einsum('kc,ka->ac', tmp, t1)
    Fac +=   eris.fvv
    return Fac

def cc_Fov(t1, t2, eris):
    Fkc  = 2*einsum('kcld,ld->kc', eris.ovov, t1)
    Fkc -=   einsum('kdlc,ld->kc', eris.ovov, t1)
    Fkc +=   eris.fov
    return Fkc

### Eqs. (40)-(41) "lambda"

def Loo(t1, t2, eris):
    Lki = cc_Foo(t1, t2, eris) + einsum('kc,ic->ki',eris.fov, t1)
    Lki += 2*einsum('kilc,lc->ki', eris.ooov, t1)
    Lki -=   einsum('likc,lc->ki', eris.ooov, t1)
    return Lki

def Lvv(t1, t2, eris):
    Lac = cc_Fvv(t1, t2, eris) - einsum('kc,ka->ac', eris.fov, t1)
    Lac += 2*einsum('kdac,kd->ac', eris.ovvv, t1)
    Lac -=   einsum('kcad,kd->ac', eris.ovvv, t1)
    return Lac

### Eqs. (42)-(45) "chi"

def cc_Woooo(t1, t2, eris):
    Wklij = einsum('kilc,jc->klij', eris.ooov, t1)
    Wklij += einsum('ljkc,ic->klij', eris.ooov, t1)
    Wklij += einsum('kcld,ijcd->klij', eris.ovov, t2)
    tmp    = einsum('kcld,ic->kild', eris.ovov, t1)
    Wklij += einsum('kild,jd->klij', tmp, t1)
    Wklij += eris.oooo.transpose(0,2,1,3)
    return Wklij

def cc_Wvvvv(t1, t2, eris):
    Wabcd  = einsum('kdac,kb->abcd', eris.ovvv,-t1)
    Wabcd -= einsum('kcbd,ka->abcd', eris.ovvv, t1)
    Wabcd += eris.vvvv.transpose(0,2,1,3)
    return Wabcd

def cc_Wvoov(t1, t2, eris):
    Wakic  = einsum('kcad,id->akic', eris.ovvv, t1)
    Wakic -= einsum('likc,la->akic', eris.ooov, t1)
    Wakic += eris.ovvo.transpose(2,0,3,1)

    Wakic -= 0.5*einsum('ldkc,ilda->akic', eris.ovov, t2)
    Wakic -= 0.5*einsum('lckd,ilad->akic', eris.ovov, t2)
    tmp    = einsum('ldkc,id->likc', eris.ovov, t1)
    Wakic -= einsum('likc,la->akic', tmp, t1)
    Wakic += einsum('ldkc,ilad->akic', eris.ovov, t2)
    return Wakic

def cc_Wvovo(t1, t2, eris):
    Wakci  = einsum('kdac,id->akci', eris.ovvv, t1)
    Wakci -= einsum('kilc,la->akci', eris.ooov, t1)
    Wakci += eris.oovv.transpose(2,0,3,1)
    Wakci -= 0.5*einsum('lckd,ilda->akci', eris.ovov, t2)
    tmp    = einsum('lckd,la->ackd', eris.ovov, t1)
    Wakci -= einsum('ackd,id->akci', tmp, t1)
    return Wakci

def Wooov(t1, t2, eris):
    Wklid  = einsum('ic,kcld->klid', t1, eris.ovov)
    Wklid += eris.ooov.transpose(0,2,1,3)
    return Wklid

def Wvovv(t1, t2, eris):
    Walcd  = einsum('ka,kcld->alcd',-t1, eris.ovov)
    Walcd += eris.ovvv.transpose(2,0,3,1)
    return Walcd

def W1ovvo(t1, t2, eris):
    Wkaci  = 2*einsum('kcld,ilad->kaci', eris.ovov, t2)
    Wkaci +=  -einsum('kcld,liad->kaci', eris.ovov, t2)
    Wkaci +=  -einsum('kdlc,ilad->kaci', eris.ovov, t2)
    Wkaci += eris.ovvo.transpose(0,2,1,3)
    return Wkaci

def W2ovvo(t1, t2, eris):
    Wkaci = einsum('la,lkic->kaci',-t1, Wooov(t1, t2, eris))
    Wkaci += einsum('kcad,id->kaci', eris.ovvv, t1)
    return Wkaci

def Wovvo(t1, t2, eris):
    Wkaci = W1ovvo(t1, t2, eris) + W2ovvo(t1, t2, eris)
    return Wkaci

def W1ovov(t1, t2, eris):
    Wkbid = -einsum('kcld,ilcb->kbid', eris.ovov, t2)
    Wkbid += eris.oovv.transpose(0,2,1,3)
    return Wkbid

def W2ovov(t1, t2, eris):
    Wkbid = einsum('klid,lb->kbid', Wooov(t1, t2, eris),- t1)
    Wkbid += einsum('kcbd,ic->kbid', eris.ovvv, t1)
    return Wkbid

def Wovov(t1, t2, eris):
    return W1ovov(t1, t2, eris) + W2ovov(t1, t2, eris)

def Woooo(t1, t2, eris):
    Wklij  = einsum('kcld,ijcd->klij', eris.ovov, t2)
    tmp    = einsum('kcld,ic->kild', eris.ovov, t1)
    Wklij += einsum('kild,jd->klij', tmp, t1)
    Wklij += einsum('kild,jd->klij', eris.ooov, t1)
    Wklij += einsum('ljkc,ic->klij', eris.ooov, t1)
    Wklij += eris.oooo.transpose(0,2,1,3)
    return Wklij

def Wvvvv(t1, t2, eris):
    Wabcd  = einsum('kcld,klab->abcd', eris.ovov, t2)
    tmp    = einsum('kcld,ka->acld', eris.ovov, t1)
    Wabcd += einsum('acld,lb->abcd', tmp, t1)
    Wabcd += eris.vvvv.transpose(0,2,1,3)
    Wabcd -= einsum('ldac,lb->abcd', eris.ovvv, t1)
    Wabcd -= einsum('kcbd,ka->abcd', eris.ovvv, t1)
    return Wabcd

def Wvvvo(t1, t2, eris, Wvvvv=None):
    Wabcj  =  -einsum('alcj,lb->abcj', W1ovov(t1, t2, eris).transpose(1,0,3,2), t1)
    Wabcj +=  -einsum('kbcj,ka->abcj', W1ovvo(t1, t2, eris), t1)
    Wabcj += 2*einsum('ldac,ljdb->abcj', eris.ovvv, t2)
    Wabcj +=  -einsum('ldac,ljbd->abcj', eris.ovvv, t2)
    Wabcj +=  -einsum('lcad,ljdb->abcj', eris.ovvv, t2)
    Wabcj +=  -einsum('kcbd,jkda->abcj', eris.ovvv, t2)
    Wabcj +=   einsum('ljkc,lkba->abcj', eris.ooov, t2)
    tmp    =   einsum('ljkc,lb->kcbj', eris.ooov, t1)
    Wabcj +=   einsum('kcbj,ka->abcj', tmp, t1)
    Wabcj +=  -einsum('kc,kjab->abcj', cc_Fov(t1, t2, eris), t2)
    Wabcj += eris.ovvv.transpose(3,1,2,0).conj()
    if Wvvvv is None:
        Wvvvv = Wvvvv(t1, t2, eris)
    Wabcj += einsum('abcd,jd->abcj', Wvvvv, t1)
    return Wabcj

def Wovoo(t1, t2, eris):
    Wkbij  =   einsum('kbid,jd->kbij', W1ovov(t1, t2, eris), t1)
    Wkbij +=  -einsum('klij,lb->kbij', Woooo(t1, t2, eris), t1)
    Wkbij +=   einsum('kbcj,ic->kbij', W1ovvo(t1, t2, eris), t1)
    Wkbij += 2*einsum('kild,ljdb->kbij', eris.ooov, t2)
    Wkbij +=  -einsum('kild,jldb->kbij', eris.ooov, t2)
    Wkbij +=  -einsum('likd,ljdb->kbij', eris.ooov, t2)
    Wkbij +=   einsum('kcbd,jidc->kbij', eris.ovvv, t2)
    tmp    =   einsum('kcbd,ic->kibd', eris.ovvv, t1)
    Wkbij +=   einsum('kibd,jd->kbij', tmp, t1)
    Wkbij +=  -einsum('ljkc,libc->kbij', eris.ooov, t2)
    Wkbij +=   einsum('kc,ijcb->kbij', cc_Fov(t1, t2, eris), t2)
    Wkbij += eris.ooov.transpose(1,3,0,2).conj()
    return Wkbij
