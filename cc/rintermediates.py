import numpy as np
from pyscf import lib
from pyscf.pbc import lib as pbclib
from pyscf.cc.ccsd import _cp

#einsum = np.einsum
einsum = pbclib.einsum

#################################################
# FOLLOWING:                                    #
# S. Hirata, ..., R. J. Bartlett                #
# J. Chem. Phys. 120, 2581 (2004)               #
#################################################

### Eqs. (37)-(39) "kappa"

def cc_Foo(t1,t2,eris):
    nocc, nvir = t1.shape
    foo = eris.fock[:nocc,:nocc]
    Fki = foo.copy()
    Fki += einsum('klcd,ilcd->ki',2*eris.oovv,t2)
    Fki += einsum('kldc,ilcd->ki', -eris.oovv,t2)
    Fki += einsum('klcd,ic,ld->ki',2*eris.oovv,t1,t1)
    Fki += einsum('kldc,ic,ld->ki', -eris.oovv,t1,t1)
    return Fki

def cc_Fvv(t1,t2,eris):
    nocc, nvir = t1.shape
    fvv = eris.fock[nocc:,nocc:]
    Fac = fvv.copy()
    Fac += -einsum('klcd,klad->ac',2*eris.oovv,t2)
    Fac += -einsum('kldc,klad->ac', -eris.oovv,t2)
    Fac += -einsum('klcd,ka,ld->ac',2*eris.oovv,t1,t1)
    Fac += -einsum('kldc,ka,ld->ac', -eris.oovv,t1,t1)
    return Fac

def cc_Fov(t1,t2,eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Fkc = fov.copy()
    Fkc += einsum('klcd,ld->kc',2*eris.oovv,t1)
    Fkc += einsum('kldc,ld->kc', -eris.oovv,t1)
    return Fkc

### Eqs. (40)-(41) "lambda"

def Loo(t1,t2,eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Lki = cc_Foo(t1,t2,eris) + einsum('kc,ic->ki',fov,t1)
    Lki += einsum('klic,lc->ki',2*eris.ooov,t1)
    Lki += einsum('klci,lc->ki', -eris.oovo,t1)
    return Lki

def Lvv(t1,t2,eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Lac = cc_Fvv(t1,t2,eris) - einsum('kc,ka->ac',fov,t1)
    Lac += einsum('akcd,kd->ac',2*eris.vovv,t1)
    Lac += einsum('akdc,kd->ac', -eris.vovv,t1)
    return Lac

### Eqs. (42)-(45) "chi"

def cc_Woooo(t1,t2,eris):
    Wklij = np.array(eris.oooo, copy=True)
    Wklij += einsum('klic,jc->klij',eris.ooov,t1)
    Wklij += einsum('klcj,ic->klij',eris.oovo,t1)
    Wklij += einsum('klcd,ijcd->klij',eris.oovv,t2)
    Wklij += einsum('klcd,ic,jd->klij',eris.oovv,t1,t1)
    return Wklij

def cc_Wvvvv(t1,t2,eris):
    ## Slow:
    Wabcd = np.array(eris.vvvv, copy=True)
    Wabcd += einsum('akcd,kb->abcd',eris.vovv,-t1)
    Wabcd += einsum('kbcd,ka->abcd',eris.ovvv,-t1)

    ## Fast
    #nocc,nvir = t1.shape
    #Wabcd = np.empty((nvir,)*4)
    #for a in range(nvir):
    #    Wabcd[a,:] = einsum('kcd,kb->bcd',eris.vovv[a],-t1)
    ##Wabcd += einsum('kbcd,ka->abcd',eris.ovvv,-t1)
    #Wabcd += lib.dot(-t1.T,eris.ovvv.reshape(nocc,-1)).reshape((nvir,)*4)
    #Wabcd += np.asarray(eris.vvvv)

    return Wabcd

def cc_Wvoov(t1,t2,eris):
    Wakic = np.array(eris.voov, copy=True)
    Wakic -= einsum('lkic,la->akic',eris.ooov,t1)
    Wakic += einsum('akdc,id->akic',eris.vovv,t1)
    Wakic -= 0.5*einsum('lkdc,ilda->akic',eris.oovv,t2)
    Wakic -= einsum('lkdc,id,la->akic',eris.oovv,t1,t1)
    Wakic += 0.5*einsum('lkdc,ilad->akic',2*eris.oovv,t2)
    Wakic += 0.5*einsum('lkcd,ilad->akic', -eris.oovv,t2)
    return Wakic

def cc_Wvovo(t1,t2,eris):
    Wakci = np.array(eris.vovo, copy=True)
    Wakci -= einsum('lkci,la->akci',eris.oovo,t1)
    Wakci += einsum('akcd,id->akci',eris.vovv,t1)
    Wakci -= 0.5*einsum('lkcd,ilda->akci',eris.oovv,t2)
    Wakci -= einsum('lkcd,id,la->akci',eris.oovv,t1,t1)
    return Wakci

# Indices in the following can be safely permuted.

def Wooov(t1,t2,eris):
    Wklid = eris.ooov + einsum('ic,klcd->klid',t1,eris.oovv)
    return Wklid

def Wvovv(t1,t2,eris):
    eris_vovv = np.array(eris.ovvv).transpose(1,0,3,2)
    Walcd = eris_vovv - einsum('ka,klcd->alcd',t1,eris.oovv)
    return Walcd

def W1ovvo(t1,t2,eris):
    Wkaci = np.array(eris.voov).transpose(1,0,3,2)
    Wkaci += 2.*einsum('klcd,ilad->kaci',eris.oovv,t2)
    Wkaci +=   -einsum('klcd,liad->kaci',eris.oovv,t2)
    Wkaci +=   -einsum('kldc,ilad->kaci',eris.oovv,t2)
    return Wkaci

def W2ovvo(t1,t2,eris):
    Wkaci = einsum('la,lkic->kaci',-t1,Wooov(t1,t2,eris))
    Wkaci += einsum('akdc,id->kaci',eris.vovv,t1)
    return Wkaci

def Wovvo(t1,t2,eris):
    Wkaci = W1ovvo(t1,t2,eris) + W2ovvo(t1,t2,eris)
    return Wkaci

def W1ovov(t1,t2,eris):
    Wkbid = np.array(eris.ovov, copy=True)
    Wkbid += -einsum('klcd,ilcb->kbid',eris.oovv,t2)
    return Wkbid

def W2ovov(t1,t2,eris):
    Wkbid = einsum('klid,lb->kbid',Wooov(t1,t2,eris),-t1)
    Wkbid += einsum('bkdc,ic->kbid',eris.vovv,t1)
    return Wkbid

def Wovov(t1,t2,eris):
    return W1ovov(t1,t2,eris) + W2ovov(t1,t2,eris)

def Woooo(t1,t2,eris):
    Wklij = np.array(eris.oooo, copy=True)
    Wklij += einsum('klcd,ijcd->klij',eris.oovv,t2)
    Wklij += einsum('klcd,ic,jd->klij',eris.oovv,t1,t1)
    Wklij += einsum('klid,jd->klij',eris.ooov,t1)
    Wklij += einsum('lkjc,ic->klij',eris.ooov,t1)
    return Wklij

def Wvvvv(t1,t2,eris):
    Wabcd = np.array(eris.vvvv, copy=True)
    Wabcd += einsum('klcd,klab->abcd',eris.oovv,t2)
    Wabcd += einsum('klcd,ka,lb->abcd',eris.oovv,t1,t1)
    Wabcd += einsum('alcd,lb->abcd',eris.vovv,-t1)
    Wabcd += einsum('bkdc,ka->abcd',eris.vovv,-t1)
    return Wabcd

def Wvvvo(t1,t2,eris):
    Wabcj = np.array(eris.vovv).transpose(2,3,0,1).conj()
    Wabcj += einsum('abcd,jd->abcj',Wvvvv(t1,t2,eris),t1)
    Wabcj += einsum('alcj,lb->abcj',W1ovov(t1,t2,eris).transpose(1,0,3,2),-t1)
    Wabcj += einsum('kbcj,ka->abcj',W1ovvo(t1,t2,eris),-t1)
    Wabcj += einsum('alcd,ljdb->abcj',eris.vovv,2.*t2)
    Wabcj += einsum('alcd,ljbd->abcj',eris.vovv,  -t2)
    Wabcj += einsum('aldc,ljdb->abcj',eris.vovv,  -t2)
    Wabcj += einsum('bkdc,jkda->abcj',eris.vovv,  -t2)
    Wabcj += einsum('lkjc,lkba->abcj',eris.ooov,t2)
    Wabcj += einsum('lkjc,lb,ka->abcj',eris.ooov,t1,t1)
    Wabcj += einsum('kc,kjab->abcj',-cc_Fov(t1,t2,eris),t2)
    return Wabcj

def Wovoo(t1,t2,eris):
    Wkbij = np.array(eris.oovo).transpose(3,2,1,0).conj()
    Wkbij += einsum('kbid,jd->kbij',W1ovov(t1,t2,eris), t1)
    Wkbij += einsum('klij,lb->kbij',Woooo(t1,t2,eris),-t1)
    Wkbij += einsum('kbcj,ic->kbij',W1ovvo(t1,t2,eris),t1)
    Wkbij += einsum('klid,ljdb->kbij', 2.*eris.ooov,t2)
    Wkbij += einsum('klid,jldb->kbij',   -eris.ooov,t2)
    Wkbij += einsum('lkid,ljdb->kbij',   -eris.ooov,t2)
    Wkbij += einsum('bkdc,jidc->kbij',eris.vovv,t2)
    Wkbij += einsum('bkdc,jd,ic->kbij',eris.vovv,t1,t1)
    Wkbij += einsum('lkjc,libc->kbij',   -eris.ooov,t2)
    Wkbij += einsum('kc,ijcb->kbij',cc_Fov(t1,t2,eris),t2)
    return Wkbij
