import numpy as np
from pyscf import lib
from pyscf import ao2mo

# This is restricted (R)CCSD
# Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)

### Eqs. (37)-(39) "kappa"

def cc_Foo(t1, t2, eris):
    nocc, nvir = t1.shape
    foo = eris.fock[:nocc,:nocc]
    Fki = foo.copy()
    eris_ovov = _get_ovov(eris)
    Fki += 2*lib.einsum('kcld,ilcd->ki', eris_ovov, t2)
    Fki -=   lib.einsum('kdlc,ilcd->ki', eris_ovov, t2)
    Fki += 2*lib.einsum('kcld,ic,ld->ki', eris_ovov, t1, t1)
    Fki -=   lib.einsum('kdlc,ic,ld->ki', eris_ovov, t1, t1)
    return Fki

def cc_Fvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fvv = eris.fock[nocc:,nocc:]
    Fac = fvv.copy()
    eris_ovov = _get_ovov(eris)
    Fac -= 2*lib.einsum('kcld,klad->ac', eris_ovov, t2)
    Fac +=   lib.einsum('kdlc,klad->ac', eris_ovov, t2)
    Fac -= 2*lib.einsum('kcld,ka,ld->ac', eris_ovov, t1, t1)
    Fac +=   lib.einsum('kdlc,ka,ld->ac', eris_ovov, t1, t1)
    return Fac

def cc_Fov(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Fkc = fov.copy()
    eris_ovov = _get_ovov(eris)
    Fkc += 2*np.einsum('kcld,ld->kc', eris_ovov, t1)
    Fkc -=   np.einsum('kdlc,ld->kc', eris_ovov, t1)
    return Fkc

### Eqs. (40)-(41) "lambda"

def Loo(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:].copy()
    Lki = cc_Foo(t1, t2, eris) + np.einsum('kc,ic->ki',fov, t1)
    eris_ooov = np.asarray(eris.vooo).conj().transpose(3,2,1,0)
    Lki += 2*np.einsum('kilc,lc->ki', eris_ooov, t1)
    Lki -=   np.einsum('likc,lc->ki', eris_ooov, t1)
    return Lki

def Lvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:].copy()
    Lac = cc_Fvv(t1, t2, eris) - np.einsum('kc,ka->ac',fov, t1)
    eris_ovvv = _get_ovvv(eris)
    Lac += 2*np.einsum('kdac,kd->ac', eris_ovvv, t1)
    Lac -=   np.einsum('kcad,kd->ac', eris_ovvv, t1)
    return Lac

### Eqs. (42)-(45) "chi"

def cc_Woooo(t1, t2, eris):
    Wklij = np.asarray(eris.oooo).transpose(0,2,1,3).copy()
    eris_ooov = np.asarray(eris.vooo).conj().transpose(3,2,1,0)
    Wklij += lib.einsum('kilc,jc->klij', eris_ooov, t1)
    Wklij += lib.einsum('ljkc,ic->klij', eris_ooov, t1)
    eris_ovov = _get_ovov(eris)
    Wklij += lib.einsum('kcld,ijcd->klij', eris_ovov, t2)
    Wklij += lib.einsum('kcld,ic,jd->klij', eris_ovov, t1, t1)
    return Wklij

def cc_Wvvvv(t1, t2, eris):
    # Incore 
    Wabcd = _get_vvvv(eris).transpose(0,2,1,3).copy()
    eris_ovvv = _get_ovvv(eris)
    Wabcd -= lib.einsum('kdac,kb->abcd', eris_ovvv, t1)
    Wabcd -= lib.einsum('kcbd,ka->abcd', eris_ovvv, t1)
    return Wabcd

def cc_Wvoov(t1, t2, eris):
    Wakic = np.asarray(eris.voov).transpose(0,2,1,3).copy()
    eris_ovvv = _get_ovvv(eris)
    eris_ooov = np.asarray(eris.vooo).conj().transpose(3,2,1,0)
    Wakic -= lib.einsum('likc,la->akic', eris_ooov, t1)
    Wakic += lib.einsum('kcad,id->akic', eris_ovvv, t1)
    eris_ovov = _get_ovov(eris)
    Wakic -= 0.5*lib.einsum('ldkc,ilda->akic', eris_ovov, t2)
    Wakic -= 0.5*lib.einsum('lckd,ilad->akic', eris_ovov, t2)
    Wakic -= lib.einsum('ldkc,id,la->akic', eris_ovov, t1, t1)
    Wakic += lib.einsum('ldkc,ilad->akic', eris_ovov, t2)
    return Wakic

def cc_Wvovo(t1, t2, eris):
    Wakci = np.asarray(eris.vvoo).transpose(0,2,1,3).copy()
    eris_ovvv = _get_ovvv(eris)
    eris_ooov = np.asarray(eris.vooo).conj().transpose(3,2,1,0)
    Wakci -= lib.einsum('kilc,la->akci', eris_ooov, t1)
    Wakci += lib.einsum('kdac,id->akci', eris_ovvv, t1)
    eris_ovov = _get_ovov(eris)
    Wakci -= 0.5*lib.einsum('lckd,ilda->akci', eris_ovov, t2)
    Wakci -= lib.einsum('lckd,id,la->akci', eris_ovov, t1, t1)
    return Wakci

def Wooov(t1, t2, eris):
    Wklid = np.asarray(eris.vooo).conj().transpose(3,1,2,0).copy()
    eris_ovov = _get_ovov(eris)
    Wklid += lib.einsum('ic,kcld->klid', t1, eris_ovov)
    return Wklid

def Wvovv(t1, t2, eris):
    eris_ovov = _get_ovov(eris)
    Walcd = _get_ovvv(eris).transpose(2,0,3,1).copy()
    Walcd -= lib.einsum('ka,kcld->alcd', t1, eris_ovov)
    return Walcd

def W1ovvo(t1, t2, eris):
    Wkaci = np.asarray(eris.voov).transpose(2,0,3,1).copy()
    eris_ovov = _get_ovov(eris)
    Wkaci += 2*lib.einsum('kcld,ilad->kaci', eris_ovov, t2)
    Wkaci +=  -lib.einsum('kcld,liad->kaci', eris_ovov, t2)
    Wkaci +=  -lib.einsum('kdlc,ilad->kaci', eris_ovov, t2)
    return Wkaci

def W2ovvo(t1, t2, eris):
    Wkaci = lib.einsum('la,lkic->kaci',-t1, Wooov(t1, t2, eris))
    eris_ovvv = _get_ovvv(eris)
    Wkaci += lib.einsum('kcad,id->kaci', eris_ovvv, t1)
    return Wkaci

def Wovvo(t1, t2, eris):
    Wkaci = W1ovvo(t1, t2, eris) + W2ovvo(t1, t2, eris)
    return Wkaci

def W1ovov(t1, t2, eris):
    Wkbid = np.asarray(eris.vvoo).transpose(2,0,3,1).copy()
    eris_ovov = _get_ovov(eris)
    Wkbid += -lib.einsum('kcld,ilcb->kbid', eris_ovov, t2)
    return Wkbid

def W2ovov(t1, t2, eris):
    Wkbid = lib.einsum('klid,lb->kbid', Wooov(t1, t2, eris),-t1)
    eris_ovvv = _get_ovvv(eris)
    Wkbid += lib.einsum('kcbd,ic->kbid', eris_ovvv, t1)
    return Wkbid

def Wovov(t1, t2, eris):
    return W1ovov(t1, t2, eris) + W2ovov(t1, t2, eris)

def Woooo(t1, t2, eris):
    Wklij = np.asarray(eris.oooo).transpose(0,2,1,3).copy()
    eris_ovov = _get_ovov(eris)
    Wklij += lib.einsum('kcld,ijcd->klij', eris_ovov, t2)
    Wklij += lib.einsum('kcld,ic,jd->klij', eris_ovov, t1, t1)
    eris_ooov = np.asarray(eris.vooo).conj().transpose(3,2,1,0)
    Wklij += lib.einsum('kild,jd->klij', eris_ooov, t1)
    Wklij += lib.einsum('ljkc,ic->klij', eris_ooov, t1)
    return Wklij

def Wvvvv(t1, t2, eris):
    Wabcd = _get_vvvv(eris).transpose(0,2,1,3).copy()
    eris_ovov = _get_ovov(eris)
    Wabcd += lib.einsum('kcld,klab->abcd', eris_ovov, t2)
    Wabcd += lib.einsum('kcld,ka,lb->abcd', eris_ovov, t1, t1)
    eris_ovvv = _get_ovvv(eris)
    Wabcd += -lib.einsum('ldac,lb->abcd', eris_ovvv, t1)
    Wabcd += -lib.einsum('kcbd,ka->abcd', eris_ovvv, t1)
    return Wabcd

def Wvvvo(t1, t2, eris, _Wvvvv=None):
    nocc,nvir = t1.shape
    eris_ovvv = _get_ovvv(eris)
    Wabcj = np.asarray(eris_ovvv).transpose(3,1,2,0).conj().copy()
    # Check if t1=0 (HF+MBPT(2))
    # don't make vvvv if you can avoid it!
    if np.any(t1):
        if _Wvvvv is None:
            _Wvvvv = Wvvvv(t1, t2, eris)
        Wabcj += lib.einsum('abcd,jd->abcj', _Wvvvv, t1)
    Wabcj +=  -lib.einsum('alcj,lb->abcj', W1ovov(t1, t2, eris).transpose(1,0,3,2), t1)
    Wabcj +=  -lib.einsum('kbcj,ka->abcj', W1ovvo(t1, t2, eris), t1)
    Wabcj += 2*lib.einsum('ldac,ljdb->abcj', eris_ovvv, t2)
    Wabcj +=  -lib.einsum('ldac,ljbd->abcj', eris_ovvv, t2)
    Wabcj +=  -lib.einsum('lcad,ljdb->abcj', eris_ovvv, t2)
    Wabcj +=  -lib.einsum('kcbd,jkda->abcj', eris_ovvv, t2)
    eris_ooov = np.asarray(eris.vooo).conj().transpose(3,2,1,0)
    Wabcj +=   lib.einsum('ljkc,lkba->abcj', eris_ooov, t2)
    Wabcj +=   lib.einsum('ljkc,lb,ka->abcj', eris_ooov, t1, t1)
    Wabcj +=  -lib.einsum('kc,kjab->abcj', cc_Fov(t1, t2, eris), t2)
    return Wabcj

def Wovoo(t1, t2, eris):
    eris_ooov = np.asarray(eris.vooo).conj().transpose(3,2,1,0)
    Wkbij = np.asarray(eris_ooov).transpose(1,3,0,2).conj().copy()
    eris_ovvv = _get_ovvv(eris)
    Wkbij +=   lib.einsum('kbid,jd->kbij', W1ovov(t1, t2, eris), t1)
    Wkbij +=  -lib.einsum('klij,lb->kbij', Woooo(t1, t2, eris), t1)
    Wkbij +=   lib.einsum('kbcj,ic->kbij', W1ovvo(t1, t2, eris), t1)
    Wkbij += 2*lib.einsum('kild,ljdb->kbij', eris_ooov, t2)
    Wkbij +=  -lib.einsum('kild,jldb->kbij', eris_ooov, t2)
    Wkbij +=  -lib.einsum('likd,ljdb->kbij', eris_ooov, t2)
    Wkbij +=   lib.einsum('kcbd,jidc->kbij', eris_ovvv, t2)
    Wkbij +=   lib.einsum('kcbd,jd,ic->kbij', eris_ovvv, t1, t1)
    Wkbij +=  -lib.einsum('ljkc,libc->kbij', eris_ooov, t2)
    Wkbij +=   lib.einsum('kc,ijcb->kbij', cc_Fov(t1, t2, eris), t2)
    return Wkbij

def _get_ovov(eris):
    if hasattr(eris, 'vovo') and eris.vovo is not None:
        return np.asarray(eris.vovo).conj().transpose(1,0,3,2)
    else:
        return np.asarray(eris.voov).transpose(1,0,2,3)

def _get_ovvv(eris, *slices):
    if eris.vovv.ndim == 3:
        vow = np.asarray(eris.vovv[slices])
        nvir, nocc, nvir_pair = vow.shape
        vovv = lib.unpack_tril(vow.reshape(nvir*nocc,nvir_pair))
        return vovv.reshape(nvir,nocc,nvir,nvir).transpose(1,0,3,2)
    else:
        return np.asarray(eris.vovv[slices]).conj().transpose(1,0,3,2)

def _get_vvvv(eris):
    if eris.vvvv is None and hasattr(eris, 'vvL'):  # DF eris
        vvL = np.asarray(eris.vvL)
        nvir = int(np.sqrt(eris.vvL.shape[0]*2))
        return ao2mo.restore(1, lib.dot(vvL, vvL.T), nvir)
    elif eris.vvvv.ndim == 2:
        nvir = int(np.sqrt(eris.vvvv.shape[0]*2))
        return ao2mo.restore(1, np.asarray(eris.vvvv), nvir)
    else:
        return np.asarray(eris.vvvv)
