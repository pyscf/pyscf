import tempfile
import numpy as np
from pyscf import lib

#einsum = np.einsum
einsum = lib.einsum

# This is restricted (R)CCSD
# Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)

### Eqs. (37)-(39) "kappa"

def cc_Foo(t1, t2, eris):
    nocc, nvir = t1.shape
    foo = eris.fock[:nocc,:nocc]
    Fki = foo.copy()
    eris_ovov = eris.vovo.conj().transpose(1,0,3,2)
    Fki += 2*einsum('kcld,ilcd->ki', eris_ovov, t2)
    Fki -=   einsum('kdlc,ilcd->ki', eris_ovov, t2)
    Fki += 2*einsum('kcld,ic,ld->ki', eris_ovov, t1, t1)
    Fki -=   einsum('kdlc,ic,ld->ki', eris_ovov, t1, t1)
    return Fki

def cc_Fvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fvv = eris.fock[nocc:,nocc:]
    Fac = fvv.copy()
    eris_ovov = eris.vovo.conj().transpose(1,0,3,2)
    Fac -= 2*einsum('kcld,klad->ac', eris_ovov, t2)
    Fac +=   einsum('kdlc,klad->ac', eris_ovov, t2)
    Fac -= 2*einsum('kcld,ka,ld->ac', eris_ovov, t1, t1)
    Fac +=   einsum('kdlc,ka,ld->ac', eris_ovov, t1, t1)
    return Fac

def cc_Fov(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Fkc = fov.copy()
    eris_ovov = eris.vovo.conj().transpose(1,0,3,2)
    Fkc += 2*einsum('kcld,ld->kc', eris_ovov, t1)
    Fkc -=   einsum('kdlc,ld->kc', eris_ovov, t1)
    return Fkc

### Eqs. (40)-(41) "lambda"

def Loo(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Lki = cc_Foo(t1, t2, eris) + einsum('kc,ic->ki',fov, t1)
    Lki += 2*einsum('kilc,lc->ki', eris.ooov, t1)
    Lki -=   einsum('likc,lc->ki', eris.ooov, t1)
    return Lki

def Lvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Lac = cc_Fvv(t1, t2, eris) - einsum('kc,ka->ac',fov, t1)
    eris_ovvv = np.asarray(eris.vovv).conj().transpose(1,0,3,2)
    Lac += 2*einsum('kdac,kd->ac', eris_ovvv, t1)
    Lac -=   einsum('kcad,kd->ac', eris_ovvv, t1)
    return Lac

### Eqs. (42)-(45) "chi"

def cc_Woooo(t1, t2, eris):
    Wklij = np.asarray(eris.oooo).transpose(0,2,1,3).copy()
    Wklij += einsum('kilc,jc->klij', eris.ooov, t1)
    Wklij += einsum('ljkc,ic->klij', eris.ooov, t1)
    eris_ovov = eris.vovo.conj().transpose(1,0,3,2)
    Wklij += einsum('kcld,ijcd->klij', eris_ovov, t2)
    Wklij += einsum('kcld,ic,jd->klij', eris_ovov, t1, t1)
    return Wklij

def cc_Wvvvv(t1, t2, eris):
    # Incore 
    Wabcd = np.asarray(eris.vvvv).transpose(0,2,1,3).copy()
    eris_ovvv = np.asarray(eris.vovv).conj().transpose(1,0,3,2)
    Wabcd -= einsum('kdac,kb->abcd', eris_ovvv, t1)
    Wabcd -= einsum('kcbd,ka->abcd', eris_ovvv, t1)
    return Wabcd

def cc_Wvoov(t1, t2, eris):
    Wakic = np.asarray(eris.voov).transpose(0,2,1,3).copy()
    eris_ovvv = np.asarray(eris.vovv).conj().transpose(1,0,3,2)
    Wakic -= einsum('likc,la->akic', eris.ooov, t1)
    Wakic += einsum('kcad,id->akic', eris_ovvv, t1)
    eris_ovov = eris.vovo.conj().transpose(1,0,3,2)
    Wakic -= 0.5*einsum('ldkc,ilda->akic', eris_ovov, t2)
    Wakic -= 0.5*einsum('lckd,ilad->akic', eris_ovov, t2)
    Wakic -= einsum('ldkc,id,la->akic', eris_ovov, t1, t1)
    Wakic += einsum('ldkc,ilad->akic', eris_ovov, t2)
    return Wakic

def cc_Wvovo(t1, t2, eris):
    Wakci = np.asarray(eris.vvoo).transpose(0,2,1,3).copy()
    eris_ovvv = np.asarray(eris.vovv).conj().transpose(1,0,3,2)
    Wakci -= einsum('kilc,la->akci', eris.ooov, t1)
    Wakci += einsum('kdac,id->akci', eris_ovvv, t1)
    eris_ovov = eris.vovo.conj().transpose(1,0,3,2)
    Wakci -= 0.5*einsum('lckd,ilda->akci', eris_ovov, t2)
    Wakci -= einsum('lckd,id,la->akci', eris_ovov, t1, t1)
    return Wakci

def Wooov(t1, t2, eris):
    Wklid = np.asarray(eris.ooov).transpose(0,2,1,3).copy()
    eris_ovov = eris.vovo.conj().transpose(1,0,3,2)
    Wklid += einsum('ic,kcld->klid', t1, eris_ovov)
    return Wklid

def Wvovv(t1, t2, eris):
    eris_ovov = eris.vovo.conj().transpose(1,0,3,2)
    Walcd = np.asarray(eris.vovv).conj().transpose(3,1,2,0).copy()
    Walcd -= einsum('ka,kcld->alcd', t1, eris_ovov)
    return Walcd

def W1ovvo(t1, t2, eris):
    Wkaci = np.asarray(eris.voov).transpose(2,0,3,1).copy()
    eris_ovov = eris.vovo.conj().transpose(1,0,3,2)
    Wkaci += 2*einsum('kcld,ilad->kaci', eris_ovov, t2)
    Wkaci +=  -einsum('kcld,liad->kaci', eris_ovov, t2)
    Wkaci +=  -einsum('kdlc,ilad->kaci', eris_ovov, t2)
    return Wkaci

def W2ovvo(t1, t2, eris):
    Wkaci = einsum('la,lkic->kaci',-t1, Wooov(t1, t2, eris))
    eris_ovvv = np.asarray(eris.vovv).conj().transpose(1,0,3,2)
    Wkaci += einsum('kcad,id->kaci', eris_ovvv, t1)
    return Wkaci

def Wovvo(t1, t2, eris):
    Wkaci = W1ovvo(t1, t2, eris) + W2ovvo(t1, t2, eris)
    return Wkaci

def W1ovov(t1, t2, eris):
    Wkbid = np.asarray(eris.vvoo).transpose(2,0,3,1).copy()
    eris_ovov = eris.vovo.conj().transpose(1,0,3,2)
    Wkbid += -einsum('kcld,ilcb->kbid', eris_ovov, t2)
    return Wkbid

def W2ovov(t1, t2, eris):
    Wkbid = einsum('klid,lb->kbid', Wooov(t1, t2, eris),-t1)
    eris_ovvv = np.asarray(eris.vovv).conj().transpose(1,0,3,2)
    Wkbid += einsum('kcbd,ic->kbid', eris_ovvv, t1)
    return Wkbid

def Wovov(t1, t2, eris):
    return W1ovov(t1, t2, eris) + W2ovov(t1, t2, eris)

def Woooo(t1, t2, eris):
    Wklij = np.asarray(eris.oooo).transpose(0,2,1,3).copy()
    eris_ovov = eris.vovo.conj().transpose(1,0,3,2)
    Wklij += einsum('kcld,ijcd->klij', eris_ovov, t2)
    Wklij += einsum('kcld,ic,jd->klij', eris_ovov, t1, t1)
    Wklij += einsum('kild,jd->klij', eris.ooov, t1)
    Wklij += einsum('ljkc,ic->klij', eris.ooov, t1)
    return Wklij

def Wvvvv(t1, t2, eris):
    Wabcd = np.asarray(eris.vvvv).transpose(0,2,1,3).copy()
    eris_ovov = eris.vovo.conj().transpose(1,0,3,2)
    Wabcd += einsum('kcld,klab->abcd', eris_ovov, t2)
    Wabcd += einsum('kcld,ka,lb->abcd', eris_ovov, t1, t1)
    eris_ovvv = np.asarray(eris.vovv).conj().transpose(1,0,3,2)
    Wabcd += -einsum('ldac,lb->abcd', eris_ovvv, t1)
    Wabcd += -einsum('kcbd,ka->abcd', eris_ovvv, t1)
    return Wabcd

def Wvvvo(t1, t2, eris, _Wvvvv=None):
    nocc,nvir = t1.shape
    eris_ovvv = np.asarray(eris.vovv).conj().transpose(1,0,3,2)
    Wabcj = np.asarray(eris_ovvv).transpose(3,1,2,0).conj().copy()
    # Check if t1=0 (HF+MBPT(2))
    # einsum will check, but don't make vvvv if you can avoid it!
    if np.any(t1):
        if _Wvvvv is None:
            _Wvvvv = Wvvvv(t1, t2, eris)
        Wabcj += einsum('abcd,jd->abcj', _Wvvvv, t1)
    Wabcj +=  -einsum('alcj,lb->abcj', W1ovov(t1, t2, eris).transpose(1,0,3,2), t1)
    Wabcj +=  -einsum('kbcj,ka->abcj', W1ovvo(t1, t2, eris), t1)
    Wabcj += 2*einsum('ldac,ljdb->abcj', eris_ovvv, t2)
    Wabcj +=  -einsum('ldac,ljbd->abcj', eris_ovvv, t2)
    Wabcj +=  -einsum('lcad,ljdb->abcj', eris_ovvv, t2)
    Wabcj +=  -einsum('kcbd,jkda->abcj', eris_ovvv, t2)
    Wabcj +=   einsum('ljkc,lkba->abcj', eris.ooov, t2)
    Wabcj +=   einsum('ljkc,lb,ka->abcj', eris.ooov, t1, t1)
    Wabcj +=  -einsum('kc,kjab->abcj', cc_Fov(t1, t2, eris), t2)
    return Wabcj

def Wovoo(t1, t2, eris):
    Wkbij = np.asarray(eris.ooov).transpose(1,3,0,2).conj().copy()
    eris_ovvv = np.asarray(eris.vovv).conj().transpose(1,0,3,2)
    Wkbij +=   einsum('kbid,jd->kbij', W1ovov(t1, t2, eris), t1)
    Wkbij +=  -einsum('klij,lb->kbij', Woooo(t1, t2, eris), t1)
    Wkbij +=   einsum('kbcj,ic->kbij', W1ovvo(t1, t2, eris), t1)
    Wkbij += 2*einsum('kild,ljdb->kbij', eris.ooov, t2)
    Wkbij +=  -einsum('kild,jldb->kbij', eris.ooov, t2)
    Wkbij +=  -einsum('likd,ljdb->kbij', eris.ooov, t2)
    Wkbij +=   einsum('kcbd,jidc->kbij', eris_ovvv, t2)
    Wkbij +=   einsum('kcbd,jd,ic->kbij', eris_ovvv, t1, t1)
    Wkbij +=  -einsum('ljkc,libc->kbij', eris.ooov, t2)
    Wkbij +=   einsum('kc,ijcb->kbij', cc_Fov(t1, t2, eris), t2)
    return Wkbij
