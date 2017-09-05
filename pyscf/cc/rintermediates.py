import tempfile
import numpy as np
import h5py
from pyscf import lib

#einsum = np.einsum
einsum = lib.einsum

# This is restricted (R)CCSD
# Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)

### Eqs. (37)-(39) "kappa"

def cc_Foo(t1,t2,eris):
    nocc, nvir = t1.shape
    foo = eris.fock[:nocc,:nocc]
    Fki = foo.copy()
    Fki += 2*einsum('kcld,ilcd->ki',eris.ovov,t2)
    Fki +=  -einsum('kdlc,ilcd->ki',eris.ovov,t2)
    Fki += 2*einsum('kcld,ic,ld->ki',eris.ovov,t1,t1)
    Fki +=  -einsum('kdlc,ic,ld->ki',eris.ovov,t1,t1)
    return Fki

def cc_Fvv(t1,t2,eris):
    nocc, nvir = t1.shape
    fvv = eris.fock[nocc:,nocc:]
    Fac = fvv.copy()
    Fac += -2*einsum('kcld,klad->ac',eris.ovov,t2)
    Fac +=    einsum('kdlc,klad->ac',eris.ovov,t2)
    Fac += -2*einsum('kcld,ka,ld->ac',eris.ovov,t1,t1)
    Fac +=    einsum('kdlc,ka,ld->ac',eris.ovov,t1,t1)
    return Fac

def cc_Fov(t1,t2,eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Fkc = fov.copy()
    Fkc += 2*einsum('kcld,ld->kc',eris.ovov,t1)
    Fkc +=  -einsum('kdlc,ld->kc',eris.ovov,t1)
    return Fkc

### Eqs. (40)-(41) "lambda"

def Loo(t1,t2,eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Lki = cc_Foo(t1,t2,eris) + einsum('kc,ic->ki',fov,t1)
    Lki += 2*einsum('kilc,lc->ki',eris.ooov,t1)
    Lki +=  -einsum('likc,lc->ki',eris.ooov,t1)
    return Lki

def Lvv(t1,t2,eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Lac = cc_Fvv(t1,t2,eris) - einsum('kc,ka->ac',fov,t1)
    eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,-1)).reshape(nocc,nvir,nvir,nvir)
    Lac += 2*einsum('kdac,kd->ac',eris_ovvv,t1)
    Lac +=  -einsum('kcad,kd->ac',eris_ovvv,t1)
    return Lac

### Eqs. (42)-(45) "chi"

def cc_Woooo(t1,t2,eris):
    Wklij = np.array(eris.oooo).transpose(0,2,1,3).copy()
    Wklij += einsum('kilc,jc->klij',eris.ooov,t1)
    Wklij += einsum('ljkc,ic->klij',eris.ooov,t1)
    Wklij += einsum('kcld,ijcd->klij',eris.ovov,t2)
    Wklij += einsum('kcld,ic,jd->klij',eris.ovov,t1,t1)
    return Wklij

def cc_Wvvvv(t1,t2,eris):
    ## Incore 
    #Wabcd = np.array(eris.vvvv).transpose(0,2,1,3)
    #Wabcd += -einsum('kdac,kb->abcd',eris.ovvv,t1)
    #Wabcd += -einsum('kcbd,ka->abcd',eris.ovvv,t1)

    ## HDF5
    if t1.dtype == np.complex: ds_type = 'c16'
    else: ds_type = 'f8'
    _tmpfile1 = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    fimd = h5py.File(_tmpfile1.name)
    nocc,nvir = t1.shape
    Wabcd = fimd.create_dataset('vvvv', (nvir,nvir,nvir,nvir), ds_type)
    # avoid transpose inside loop
    eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,-1)).reshape(nocc,nvir,nvir,nvir)
    ovvv = np.array(eris_ovvv).transpose(0,2,1,3)
    for a in range(nvir):
#        Wabcd[a] = eris.vvvv[a].transpose(1,0,2)
#        Wabcd[a] += -einsum('kdc,kb->bcd',eris_ovvv[:,:,a,:],t1)
#        #Wabcd[a] += -einsum('kcbd,k->bcd',eris_ovvv,t1[:,a])
#        Wabcd[a] += -einsum('k,kbcd->bcd',t1[:,a],ovvv)
        w_vvv  = einsum('kdc,kb->bcd',eris_ovvv[:,:,a,:],-t1)
        w_vvv -= einsum('k,kbcd->bcd',t1[:,a],ovvv)
        a0 = a*(a+1)//2
        w_vvv[:,:a+1] += lib.unpack_tril(eris.vvvv[a0:a0+a+1]).transpose(1,0,2)
        for i in range(a+1,nvir):
            w_vvv[:,i] += lib.unpack_tril(eris.vvvv[i*(i+1)//2+a])
        Wabcd[a] = w_vvv
    return Wabcd

def cc_Wvoov(t1,t2,eris):
    nocc, nvir = t1.shape
    eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,-1)).reshape(nocc,nvir,nvir,nvir)
    Wakic = np.array(eris.ovvo).transpose(1,3,0,2)
    Wakic -= einsum('likc,la->akic',eris.ooov,t1)
    Wakic += einsum('kcad,id->akic',eris_ovvv,t1)
    Wakic -= 0.5*einsum('ldkc,ilda->akic',eris.ovov,t2)
    Wakic -= einsum('ldkc,id,la->akic',eris.ovov,t1,t1)
    Wakic += einsum('ldkc,ilad->akic',eris.ovov,t2)
    Wakic += -0.5*einsum('lckd,ilad->akic',eris.ovov,t2)
    return Wakic

def cc_Wvovo(t1,t2,eris):
    nocc, nvir = t1.shape
    eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,-1)).reshape(nocc,nvir,nvir,nvir)
    Wakci = np.array(eris.oovv).transpose(2,0,3,1)
    Wakci -= einsum('kilc,la->akci',eris.ooov,t1)
    Wakci += einsum('kdac,id->akci',eris_ovvv,t1)
    Wakci -= 0.5*einsum('lckd,ilda->akci',eris.ovov,t2)
    Wakci -= einsum('lckd,id,la->akci',eris.ovov,t1,t1)
    return Wakci

def Wooov(t1,t2,eris):
    Wklid = np.asarray(eris.ooov).transpose(0,2,1,3) + einsum('ic,kcld->klid',t1,eris.ovov)
    return Wklid

def Wvovv(t1,t2,eris):
    nocc, nvir = t1.shape
    eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,-1)).reshape(nocc,nvir,nvir,nvir)
    Walcd = np.asarray(eris_ovvv).transpose(2,0,3,1) - einsum('ka,kcld->alcd',t1,eris.ovov)
    return Walcd

def W1ovvo(t1,t2,eris):
    Wkaci = np.array(eris.ovvo).transpose(3,1,2,0)
    Wkaci += 2*einsum('kcld,ilad->kaci',eris.ovov,t2)
    Wkaci +=  -einsum('kcld,liad->kaci',eris.ovov,t2)
    Wkaci +=  -einsum('kdlc,ilad->kaci',eris.ovov,t2)
    return Wkaci

def W2ovvo(t1,t2,eris):
    nocc, nvir = t1.shape
    eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,-1)).reshape(nocc,nvir,nvir,nvir)
    Wkaci = einsum('la,lkic->kaci',-t1,Wooov(t1,t2,eris))
    Wkaci += einsum('kcad,id->kaci',eris_ovvv,t1)
    return Wkaci

def Wovvo(t1,t2,eris):
    Wkaci = W1ovvo(t1,t2,eris) + W2ovvo(t1,t2,eris)
    return Wkaci

def W1ovov(t1,t2,eris):
    Wkbid = np.array(eris.oovv).transpose(0,2,1,3)
    Wkbid += -einsum('kcld,ilcb->kbid',eris.ovov,t2)
    return Wkbid

def W2ovov(t1,t2,eris):
    nocc, nvir = t1.shape
    eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,-1)).reshape(nocc,nvir,nvir,nvir)
    Wkbid = einsum('klid,lb->kbid',Wooov(t1,t2,eris),-t1)
    Wkbid += einsum('kcbd,ic->kbid',eris_ovvv,t1)
    return Wkbid

def Wovov(t1,t2,eris):
    return W1ovov(t1,t2,eris) + W2ovov(t1,t2,eris)

def Woooo(t1,t2,eris):
    Wklij = np.array(eris.oooo).transpose(0,2,1,3).copy()
    Wklij += einsum('kcld,ijcd->klij',eris.ovov,t2)
    Wklij += einsum('kcld,ic,jd->klij',eris.ovov,t1,t1)
    Wklij += einsum('kild,jd->klij',eris.ooov,t1)
    Wklij += einsum('ljkc,ic->klij',eris.ooov,t1)
    return Wklij

def Wvvvv(t1,t2,eris):
    ## Incore 
    #Wabcd = np.array(eris.vvvv).transpose(0,2,1,3)
    #Wabcd += einsum('kcld,klab->abcd',eris.ovov,t2)
    #Wabcd += einsum('kcld,ka,lb->abcd',eris.ovov,t1,t1)
    #Wabcd += -einsum('ldac,lb->abcd',eris.ovvv,t1)
    #Wabcd += -einsum('kcbd,ka->abcd',eris.ovvv,t1)

    ## HDF5
    if t1.dtype == np.complex: ds_type = 'c16'
    else: ds_type = 'f8'
    _tmpfile1 = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    fimd = h5py.File(_tmpfile1.name)
    nocc,nvir = t1.shape
    Wabcd = fimd.create_dataset('vvvv', (nvir,nvir,nvir,nvir), ds_type)
    eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,-1)).reshape(nocc,nvir,nvir,nvir)
    for a in range(nvir):
        #Wabcd[a] = eris.vvvv[a].transpose(1,0,2)
        #Wabcd[a] += -einsum('ldc,lb->bcd',eris_ovvv[:,:,a,:],t1)
        #Wabcd[a] += -einsum('kcbd,k->bcd',eris_ovvv,t1[:,a])
        #Wabcd[a] += einsum('kcld,klb->bcd',eris.ovov,t2[:,:,a,:])
        #Wabcd[a] += einsum('kcld,k,lb->bcd',eris.ovov,t1[:,a],t1)
        w_vvv  = einsum('ldc,lb->bcd',eris_ovvv[:,:,a,:],-t1)
        w_vvv += einsum('kcbd,k->bcd',eris_ovvv,-t1[:,a])
        w_vvv += einsum('kcld,klb->bcd',eris.ovov,t2[:,:,a,:])
        w_vvv += einsum('kcld,k,lb->bcd',eris.ovov,t1[:,a],t1)
        a0 = a*(a+1)//2
        w_vvv[:,:a+1] += lib.unpack_tril(eris.vvvv[a0:a0+a+1]).transpose(1,0,2)
        for i in range(a+1,nvir):
            w_vvv[:,i] += lib.unpack_tril(eris.vvvv[i*(i+1)//2+a])
        Wabcd[a] = w_vvv
    return Wabcd

def Wvvvo(t1,t2,eris,_Wvvvv=None):
    nocc, nvir = t1.shape
    nocc,nvir = t1.shape
    eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,-1)).reshape(nocc,nvir,nvir,nvir)
    Wabcj = np.array(eris_ovvv).transpose(3,1,2,0).conj()
    # Check if t1=0 (HF+MBPT(2))
    # einsum will check, but don't make vvvv if you can avoid it!
    if np.any(t1):
        if _Wvvvv is None:
            _Wvvvv = Wvvvv(t1,t2,eris)
        for a in range(nvir):
            Wabcj[a] += einsum('bcd,jd->bcj',_Wvvvv[a],t1)
    Wabcj +=  -einsum('alcj,lb->abcj',W1ovov(t1,t2,eris).transpose(1,0,3,2),t1)
    Wabcj +=  -einsum('kbcj,ka->abcj',W1ovvo(t1,t2,eris),t1)
    Wabcj += 2*einsum('ldac,ljdb->abcj',eris_ovvv,t2)
    Wabcj +=  -einsum('ldac,ljbd->abcj',eris_ovvv,t2)
    Wabcj +=  -einsum('lcad,ljdb->abcj',eris_ovvv,t2)
    Wabcj +=  -einsum('kcbd,jkda->abcj',eris_ovvv,t2)
    Wabcj +=   einsum('ljkc,lkba->abcj',eris.ooov,t2)
    Wabcj +=   einsum('ljkc,lb,ka->abcj',eris.ooov,t1,t1)
    Wabcj +=  -einsum('kc,kjab->abcj',cc_Fov(t1,t2,eris),t2)
    return Wabcj

def Wovoo(t1,t2,eris):
    nocc, nvir = t1.shape
    eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,-1)).reshape(nocc,nvir,nvir,nvir)
    Wkbij = np.array(eris.ooov).transpose(1,3,0,2).conj()
    Wkbij +=   einsum('kbid,jd->kbij',W1ovov(t1,t2,eris),t1)
    Wkbij +=  -einsum('klij,lb->kbij',Woooo(t1,t2,eris),t1)
    Wkbij +=   einsum('kbcj,ic->kbij',W1ovvo(t1,t2,eris),t1)
    Wkbij += 2*einsum('kild,ljdb->kbij',eris.ooov,t2)
    Wkbij +=  -einsum('kild,jldb->kbij',eris.ooov,t2)
    Wkbij +=  -einsum('likd,ljdb->kbij',eris.ooov,t2)
    Wkbij +=   einsum('kcbd,jidc->kbij',eris_ovvv,t2)
    Wkbij +=   einsum('kcbd,jd,ic->kbij',eris_ovvv,t1,t1)
    Wkbij +=  -einsum('ljkc,libc->kbij',eris.ooov,t2)
    Wkbij +=   einsum('kc,ijcb->kbij',cc_Fov(t1,t2,eris),t2)
    return Wkbij
