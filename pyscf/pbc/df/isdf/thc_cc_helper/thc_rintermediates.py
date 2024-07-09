

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
# Author: Ning Zhang <ningzhang1024@gmail.com>
#

import pyscf.pbc.df.isdf.thc_cc_helper._einsum_holder as einsum_holder

einsum = einsum_holder.thc_einsum_sybolic

# This is restricted (R)CCSD
# Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)

### Eqs. (37)-(39) "kappa"

def cc_Foo(t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None):
    foo = einsum_holder._expr_foo()
    if eris is None:
        eris_ovov = einsum_holder._thc_eri_ovov()
    else:
        eris_ovov = eris.ovov
    Fki  = 2*einsum('kcld,ilcd->ki', eris_ovov, t2, cached=True)
    Fki -=   einsum('kdlc,ilcd->ki', eris_ovov, t2, cached=True)
    if t1 is not None: 
        Fki += 2*einsum('kcld,ic,ld->ki', eris_ovov, t1, t1, cached=True)
        Fki -=   einsum('kdlc,ic,ld->ki', eris_ovov, t1, t1, cached=True)
    Fki += foo
    Fki.name = "FOO"
    Fki.cached = True
    return Fki

def cc_Fvv(t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None):
    fvv = einsum_holder._expr_fvv()
    if eris is None:
        eris_ovov = einsum_holder._thc_eri_ovov()
    else:
        eris_ovov = eris.ovov
    Fac  =-2*einsum('kcld,klad->ac', eris_ovov, t2, cached=True)
    Fac +=   einsum('kdlc,klad->ac', eris_ovov, t2, cached=True)
    if t1 is not None: 
        Fac -= 2*einsum('kcld,ka,ld->ac', eris_ovov, t1, t1, cached=True)
        Fac +=   einsum('kdlc,ka,ld->ac', eris_ovov, t1, t1, cached=True)
    Fac += fvv
    Fac.name = "FVV"
    Fac.cached = True
    return Fac

def cc_Fov(t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None):
    fov = einsum_holder._expr_fov()
    if eris is None:
        eris_ovov = einsum_holder._thc_eri_ovov()
    else:
        eris_ovov = eris.ovov
    Fkc = fov
    if t1 is not None: 
        Fkc += 2*einsum('kcld,ld->kc', eris_ovov, t1, cached=True)
        Fkc -=   einsum('kdlc,ld->kc', eris_ovov, t1, cached=True)
    Fkc = einsum_holder.to_expr_holder(Fkc)
    Fkc.name = "FOV"
    Fkc.cached = True
    return Fkc

### Eqs. (40)-(41) "lambda" ###

def Loo(t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None):
    fov = einsum_holder._expr_fov()
    if t1 is not None:
        Lki = cc_Foo(t1, t2, eris) + einsum('kc,ic->ki',fov, t1, cached=True)
    else:
        Lki = cc_Foo(t1, t2, eris)
    if eris is None:
        eris_ovoo = einsum_holder._thc_eri_ovoo()
    else:
        eris_ovoo = eris.ovoo
    if t1 is not None:
        Lki += 2*einsum('lcki,lc->ki', eris_ovoo, t1, cached=True)
        Lki -=   einsum('kcli,lc->ki', eris_ovoo, t1, cached=True)
    Lki.name = "LOO"
    Lki.cached = True
    return Lki

def Lvv(t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None):
    fov = einsum_holder._expr_fov()
    if eris is None:
        eris_ovvv = einsum_holder._thc_eri_ovvv()
    else:
        eris_ovvv = eris.ovvv
    if t1 is not None:
        Lac = cc_Fvv(t1, t2, eris) - einsum('kc,ka->ac',fov, t1, cached=True)
        Lac += 2*einsum('kdac,kd->ac', eris_ovvv, t1, cached=True)
        Lac -=   einsum('kcad,kd->ac', eris_ovvv, t1, cached=True)
    else:
        Lac = cc_Fvv(t1, t2, eris)
    Lac.name = "LVV"
    Lac.cached = True
    return Lac

### Eqs. (42)-(45) "chi"

def cc_Woooo(t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None):
    if eris is None:
        eris_ovoo = einsum_holder._thc_eri_ovoo()
        eris_ovov = einsum_holder._thc_eri_ovov()
        eris_oooo = einsum_holder._thc_eri_oooo()
    else:
        eris_ovoo = eris.ovoo
        eris_ovov = eris.ovov
        eris_oooo = eris.oooo
    Wklij  = einsum('kcld,ijcd->klij', eris_ovov, t2)
    Wklij += eris_oooo.transpose((0,2,1,3))
    if t1 is not None:
        Wklij += einsum('lcki,jc->klij', eris_ovoo, t1)
        Wklij += einsum('kclj,ic->klij', eris_ovoo, t1)
        Wklij += einsum('kcld,ic,jd->klij', eris_ovov, t1, t1)
    return Wklij

def cc_Wvvvv(t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None):
    if eris is None:
        eris_ovvv = einsum_holder._thc_eri_ovvv()
        eris_vvvv = einsum_holder._thc_eri_vvvv()
    else:
        eris_ovvv = eris.ovvv
        eris_vvvv = eris.vvvv
    Wabcd = eris_vvvv.transpose((0,2,1,3))
    if t1 is not None:
        Wabcd += einsum('kdac,kb->abcd', eris_ovvv,-t1)
        Wabcd -= einsum('kcbd,ka->abcd', eris_ovvv, t1)
    return Wabcd

def cc_Wvoov(t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None):
    if eris is None:
        eris_ovvv = einsum_holder._thc_eri_ovvv()
        eris_ovoo = einsum_holder._thc_eri_ovoo()
        eris_ovvo = einsum_holder._thc_eri_ovvo()
        eris_ovov = einsum_holder._thc_eri_ovov()
    else:
        eris_ovvv = eris.ovvv
        eris_ovoo = eris.ovoo
        eris_ovvo = eris.ovvo
        eris_ovov = eris.ovov
    Wakic = eris_ovvo.transpose((2,0,3,1))
    if t1 is not None:
        Wakic += einsum('kcad,id->akic', eris_ovvv, t1)
        Wakic -= einsum('kcli,la->akic', eris_ovoo, t1)
        Wakic -= einsum('ldkc,id,la->akic', eris_ovov, t1, t1)
    Wakic -= 0.5*einsum('ldkc,ilda->akic', eris_ovov, t2)
    Wakic -= 0.5*einsum('lckd,ilad->akic', eris_ovov, t2)
    Wakic += einsum('ldkc,ilad->akic', eris_ovov, t2)
    return Wakic

def cc_Wvovo(t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None):
    if eris is None:
        eris_ovvv = einsum_holder._thc_eri_ovvv()
        eris_ovoo = einsum_holder._thc_eri_ovoo()
        eris_oovv = einsum_holder._thc_eri_oovv()
        eris_ovov = einsum_holder._thc_eri_ovov()
    else:
        eris_ovvv = eris.ovvv
        eris_ovoo = eris.ovoo
        eris_oovv = eris.oovv
        eris_ovov = eris.ovov
    Wakci = eris_oovv.transpose((2,0,3,1))
    if t1 is not None:
        Wakci += einsum('kdac,id->akci', eris_ovvv, t1)
        Wakci -= einsum('lcki,la->akci', eris_ovoo, t1)
        Wakci -= einsum('lckd,id,la->akci', eris_ovov, t1, t1)
    Wakci -= 0.5*einsum('lckd,ilda->akci', eris_ovov, t2)
    return Wakci

############# EOM Intermediates #############

def Wooov(t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None):
    if eris is None:
        eris_ovoo = einsum_holder._thc_eri_ovoo()
        eris_ovov = einsum_holder._thc_eri_ovov()
    else:
        eris_ovoo = eris.ovoo
        eris_ovov = eris.ovov
    Wklid  = eris_ovoo.transpose((2,0,3,1))
    if t1 is not None:
        Wklid  += einsum('ic,kcld->klid', t1, eris_ovov)
    return Wklid

def Wvovv(t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None):
    if eris is None:
        eris_ovov = einsum_holder._thc_eri_ovvv()
        eris_ovvv = einsum_holder._thc_eri_ovvv()
    else:
        eris_ovov = eris.ovov
        eris_ovvv = eris.ovvv
    Walcd  = eris_ovvv.transpose((2,0,3,1))
    if t1 is not None:
        Walcd  += einsum('ka,kcld->alcd',-t1, eris_ovov)
    return Walcd

def W1ovvo(t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None):
    if eris is None:
        eris_ovov = einsum_holder._thc_eri_ovov()
        eris_ovvo = einsum_holder._thc_eri_ovvo()
    else:
        eris_ovov = eris.ovov
        eris_ovvo = eris.ovvo
    Wkaci  = 2*einsum('kcld,ilad->kaci', eris_ovov, t2)
    Wkaci +=  -einsum('kcld,liad->kaci', eris_ovov, t2)
    Wkaci +=  -einsum('kdlc,ilad->kaci', eris_ovov, t2)
    Wkaci += eris_ovvo.transpose((0,2,1,3))
    return Wkaci

def W2ovvo(t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None):
    if t1 is None:
        return None  # 或者其他默认值
    if eris is None:
        eris_ovvv = einsum_holder._thc_eri_ovvv()
    else:
        eris_ovvv = eris.ovvv
    Wkaci = einsum('la,lkic->kaci',-t1, Wooov(t1, t2, eris))
    Wkaci += einsum('kcad,id->kaci', eris_ovvv, t1)
    return Wkaci

def Wovvo(t1, t2, eris):
    if t1 is None:
        Wkaci = W1ovvo(None, t2, eris)
    else:
        Wkaci = W1ovvo(t1, t2, eris) + W2ovvo(t1, t2, eris)
    return Wkaci

def W1ovov(t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None):
    if eris is None:
        eris_ovov = einsum_holder._thc_eri_ovov()
        eris_oovv = einsum_holder._thc_eri_oovv()
    else:
        eris_ovov = eris.ovov
        eris_oovv = eris.oovv
    Wkbid = -einsum('kcld,ilcb->kbid', eris_ovov, t2)
    Wkbid += eris_oovv.transpose((0,2,1,3))
    return Wkbid

def W2ovov(t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None):
    if t1 is None:
        return None  # 或者其他默认值
    if eris is None:
        eris_ovvv = einsum_holder._thc_eri_ovvv()
    else:
        eris_ovvv = eris.ovvv
    Wkbid = einsum('klid,lb->kbid', Wooov(t1, t2, eris),-t1)
    Wkbid += einsum('kcbd,ic->kbid', eris_ovvv, t1)
    return Wkbid

def Wovov(t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None):
    if t1 is None:
        return W1ovov(t1, t2, eris)
    return W1ovov(t1, t2, eris) + W2ovov(t1, t2, eris)

def Woooo(t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None):
    if eris is None:
        eris_ovov = einsum_holder._thc_eri_ovov()
        eris_ovoo = einsum_holder._thc_eri_ovoo()
        eris_oooo = einsum_holder._thc_eri_oooo()
    else:
        eris_ovov = eris.ovov
        eris_ovoo = eris.ovoo
        eris_oooo = eris.oooo
    Wklij  = einsum('kcld,ijcd->klij', eris_ovov, t2)
    if t1 is not None:
        Wklij += einsum('kcld,ic,jd->klij', eris_ovov, t1, t1)
        Wklij += einsum('ldki,jd->klij', eris_ovoo, t1)
        Wklij += einsum('kclj,ic->klij', eris_ovoo, t1)
    Wklij += eris_oooo.transpose((0,2,1,3))
    return Wklij

def Wvvvv(t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None):
    if eris is None:
        eris_ovov = einsum_holder._thc_eri_ovov()
        eris_ovvv = einsum_holder._thc_eri_ovvv()
        eris_vvvv = einsum_holder._thc_eri_vvvv()
    else:
        eris_ovov = eris.ovov
        eris_ovvv = eris.ovvv
        eris_vvvv = eris.vvvv
    Wabcd  = einsum('kcld,klab->abcd', eris_ovov, t2)
    if t1 is not None:
        Wabcd += einsum('kcld,ka,lb->abcd', eris_ovov, t1, t1)
        Wabcd -= einsum('ldac,lb->abcd', eris_ovvv, t1)
        Wabcd -= einsum('kcbd,ka->abcd', eris_ovvv, t1)
    Wabcd += eris_vvvv.transpose((0,2,1,3))
    return Wabcd

def Wvvvo(t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None, _Wvvvv=None):
    if eris is None:
        eris_ovvv = einsum_holder._thc_eri_ovvv()
        eris_ovoo = einsum_holder._thc_eri_ovoo()
        eris_vvvv = einsum_holder._thc_eri_vvvv()
    else:
        eris_ovvv = eris.ovvv
        eris_ovoo = eris.ovoo
        eris_vvvv = eris.vvvv
    
    Wabcj  = 2*einsum('ldac,ljdb->abcj', eris_ovvv, t2)
    Wabcj +=  -einsum('ldac,ljbd->abcj', eris_ovvv, t2)
    Wabcj +=  -einsum('lcad,ljdb->abcj', eris_ovvv, t2)
    Wabcj +=  -einsum('kcbd,jkda->abcj', eris_ovvv, t2)
    Wabcj +=   einsum('kclj,lkba->abcj', eris_ovoo, t2)
    Wabcj +=  -einsum('kc,kjab->abcj', cc_Fov(t1, t2, eris), t2)
    
    if t1 is not None:
        Wabcj +=  -einsum('alcj,lb->abcj', W1ovov(t1, t2, eris).transpose((1,0,3,2)), t1)
        Wabcj +=  -einsum('kbcj,ka->abcj', W1ovvo(t1, t2, eris), t1)
        Wabcj +=   einsum('kclj,lb,ka->abcj', eris_ovoo, t1, t1)
        Wabcj +=   einsum('abcd,jd->abcj', Wvvvv(t1, t2, eris), t1)
        
    Wabcj += eris_ovvv.transpose((3,1,2,0)).conj()
    
    return Wabcj

def Wovoo(t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None):
    if eris is None:
        eris_ovoo = einsum_holder._thc_eri_ovoo()
        eris_ovvv = einsum_holder._thc_eri_ovvv()
    else:
        eris_ovoo = eris.ovoo
        eris_ovvv = eris.ovvv
        
    Wkbij  = 2*einsum('ldki,ljdb->kbij', eris_ovoo, t2)
    Wkbij +=  -einsum('ldki,jldb->kbij', eris_ovoo, t2)
    Wkbij +=  -einsum('kdli,ljdb->kbij', eris_ovoo, t2)
    Wkbij +=   einsum('kcbd,jidc->kbij', eris_ovvv, t2)
    Wkbij +=   eris_ovoo.transpose((3,1,2,0)).conj()
    Wkbij +=  -einsum('kclj,libc->kbij', eris_ovoo, t2)
    Wkbij +=   einsum('kc,ijcb->kbij', cc_Fov(t1, t2, eris), t2)    
    
    if t1 is not None:
        Wkbij +=   einsum('kbid,jd->kbij', W1ovov(t1, t2, eris), t1)
        Wkbij +=  -einsum('klij,lb->kbij', Woooo(t1, t2, eris), t1)
        Wkbij +=   einsum('kbcj,ic->kbij', W1ovvo(t1, t2, eris), t1)
        Wkbij +=   einsum('kcbd,jd,ic->kbij', eris_ovvv, t1, t1)
        
    return Wkbij

