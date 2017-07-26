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
    Fki += 2*einsum('klcd,ilcd->ki',eris.oovv,t2)
    Fki +=  -einsum('kldc,ilcd->ki',eris.oovv,t2)
    Fki += 2*einsum('klcd,ic,ld->ki',eris.oovv,t1,t1)
    Fki +=  -einsum('kldc,ic,ld->ki',eris.oovv,t1,t1)
    return Fki

def cc_Fvv(t1,t2,eris):
    nocc, nvir = t1.shape
    fvv = eris.fock[nocc:,nocc:]
    Fac = fvv.copy()
    Fac += -2*einsum('klcd,klad->ac',eris.oovv,t2)
    Fac +=    einsum('kldc,klad->ac',eris.oovv,t2)
    Fac += -2*einsum('klcd,ka,ld->ac',eris.oovv,t1,t1)
    Fac +=    einsum('kldc,ka,ld->ac',eris.oovv,t1,t1)
    return Fac

def cc_Fov(t1,t2,eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Fkc = fov.copy()
    Fkc += 2*einsum('klcd,ld->kc',eris.oovv,t1)
    Fkc +=  -einsum('kldc,ld->kc',eris.oovv,t1)
    return Fkc

### Eqs. (40)-(41) "lambda"

def Loo(t1,t2,eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Lki = cc_Foo(t1,t2,eris) + einsum('kc,ic->ki',fov,t1)
    Lki += 2*einsum('klic,lc->ki',eris.ooov,t1)
    Lki +=  -einsum('lkic,lc->ki',eris.ooov,t1)
    return Lki

def Lvv(t1,t2,eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Lac = cc_Fvv(t1,t2,eris) - einsum('kc,ka->ac',fov,t1)
    Lac += 2*einsum('akcd,kd->ac',eris.vovv,t1)
    Lac +=  -einsum('akdc,kd->ac',eris.vovv,t1)
    return Lac

### Eqs. (42)-(45) "chi"

def cc_Woooo(t1,t2,eris):
    Wklij = np.array(eris.oooo)
    Wklij += einsum('klic,jc->klij',eris.ooov,t1)
    Wklij += einsum('lkjc,ic->klij',eris.ooov,t1)
    Wklij += einsum('klcd,ijcd->klij',eris.oovv,t2)
    Wklij += einsum('klcd,ic,jd->klij',eris.oovv,t1,t1)
    return Wklij

def cc_Wvvvv(t1,t2,eris):
    ## Incore 
    #Wabcd = np.array(eris.vvvv)
    #Wabcd += -einsum('akcd,kb->abcd',eris.vovv,t1)
    ##Wabcd += -einsum('kbcd,ka->abcd',eris.ovvv,t1)
    #Wabcd += -einsum('bkdc,ka->abcd',eris.vovv,t1)

    ## HDF5
    if t1.dtype == np.complex: ds_type = 'c16'
    else: ds_type = 'f8'
    _tmpfile1 = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    fimd = h5py.File(_tmpfile1.name)
    nocc,nvir = t1.shape
    Wabcd = fimd.create_dataset('vvvv', (nvir,nvir,nvir,nvir), ds_type)
    # avoid transpose inside loop
    ovvv = np.array(eris.vovv).transpose(1,0,3,2)
    for a in range(nvir):
        Wabcd[a] = eris.vvvv[a]
        Wabcd[a] += -einsum('kcd,kb->bcd',eris.vovv[a],t1)
        #Wabcd[a] += -einsum('bkdc,k->bcd',eris.vovv,t1[:,a])
        Wabcd[a] += -einsum('k,kbcd->bcd',t1[:,a],ovvv)
    return Wabcd

def cc_Wvoov(t1,t2,eris):
    Wakic = np.array(eris.voov)
    Wakic -= einsum('lkic,la->akic',eris.ooov,t1)
    Wakic += einsum('akdc,id->akic',eris.vovv,t1)
    Wakic -= 0.5*einsum('lkdc,ilda->akic',eris.oovv,t2)
    Wakic -= einsum('lkdc,id,la->akic',eris.oovv,t1,t1)
    Wakic += einsum('lkdc,ilad->akic',eris.oovv,t2)
    Wakic += -0.5*einsum('lkcd,ilad->akic',eris.oovv,t2)
    return Wakic

def cc_Wvovo(t1,t2,eris):
    Wakci = np.array(eris.ovov).transpose(1,0,3,2)
    Wakci -= einsum('klic,la->akci',eris.ooov,t1)
    Wakci += einsum('akcd,id->akci',eris.vovv,t1)
    Wakci -= 0.5*einsum('lkcd,ilda->akci',eris.oovv,t2)
    Wakci -= einsum('lkcd,id,la->akci',eris.oovv,t1,t1)
    return Wakci

def Wooov(t1,t2,eris):
    Wklid = eris.ooov + einsum('ic,klcd->klid',t1,eris.oovv)
    return Wklid

def Wvovv(t1,t2,eris):
    Walcd = eris.vovv - einsum('ka,klcd->alcd',t1,eris.oovv)
    return Walcd

def W1ovvo(t1,t2,eris):
    Wkaci = np.array(eris.voov).transpose(1,0,3,2)
    Wkaci += 2*einsum('klcd,ilad->kaci',eris.oovv,t2)
    Wkaci +=  -einsum('klcd,liad->kaci',eris.oovv,t2)
    Wkaci +=  -einsum('kldc,ilad->kaci',eris.oovv,t2)
    return Wkaci

def W2ovvo(t1,t2,eris):
    Wkaci = einsum('la,lkic->kaci',-t1,Wooov(t1,t2,eris))
    Wkaci += einsum('akdc,id->kaci',eris.vovv,t1)
    return Wkaci

def Wovvo(t1,t2,eris):
    Wkaci = W1ovvo(t1,t2,eris) + W2ovvo(t1,t2,eris)
    return Wkaci

def W1ovov(t1,t2,eris):
    Wkbid = np.array(eris.ovov)
    Wkbid += -einsum('klcd,ilcb->kbid',eris.oovv,t2)
    return Wkbid

def W2ovov(t1,t2,eris):
    Wkbid = einsum('klid,lb->kbid',Wooov(t1,t2,eris),-t1)
    Wkbid += einsum('bkdc,ic->kbid',eris.vovv,t1)
    return Wkbid

def Wovov(t1,t2,eris):
    return W1ovov(t1,t2,eris) + W2ovov(t1,t2,eris)

def Woooo(t1,t2,eris):
    Wklij = np.array(eris.oooo)
    Wklij += einsum('klcd,ijcd->klij',eris.oovv,t2)
    Wklij += einsum('klcd,ic,jd->klij',eris.oovv,t1,t1)
    Wklij += einsum('klid,jd->klij',eris.ooov,t1)
    Wklij += einsum('lkjc,ic->klij',eris.ooov,t1)
    return Wklij

def Wvvvv(t1,t2,eris):
    ## Incore 
    #Wabcd = np.array(eris.vvvv)
    #Wabcd += einsum('klcd,klab->abcd',eris.oovv,t2)
    #Wabcd += einsum('klcd,ka,lb->abcd',eris.oovv,t1,t1)
    #Wabcd += -einsum('alcd,lb->abcd',eris.vovv,t1)
    #Wabcd += -einsum('bkdc,ka->abcd',eris.vovv,t1)

    ## HDF5
    if t1.dtype == np.complex: ds_type = 'c16'
    else: ds_type = 'f8'
    _tmpfile1 = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    fimd = h5py.File(_tmpfile1.name)
    nocc,nvir = t1.shape
    Wabcd = fimd.create_dataset('vvvv', (nvir,nvir,nvir,nvir), ds_type)
    for a in range(nvir):
        Wabcd[a] = eris.vvvv[a]
        Wabcd[a] += -einsum('lcd,lb->bcd',eris.vovv[a],t1)
        Wabcd[a] += -einsum('bkdc,k->bcd',eris.vovv,t1[:,a])
        Wabcd[a] += einsum('klcd,klb->bcd',eris.oovv,t2[:,:,a,:])
        Wabcd[a] += einsum('klcd,k,lb->bcd',eris.oovv,t1[:,a],t1)
    return Wabcd

def Wvvvo(t1,t2,eris,_Wvvvv=None):
    nocc,nvir = t1.shape
    Wabcj = np.array(eris.vovv).transpose(2,3,0,1).conj()
    # Check if t1=0 (HF+MBPT(2))
    # einsum will check, but don't make vvvv if you can avoid it!
    if np.any(t1):
        if _Wvvvv is None:
            _Wvvvv = Wvvvv(t1,t2,eris)
        for a in range(nvir):
            Wabcj[a] += einsum('bcd,jd->bcj',_Wvvvv[a],t1)
    Wabcj +=  -einsum('alcj,lb->abcj',W1ovov(t1,t2,eris).transpose(1,0,3,2),t1)
    Wabcj +=  -einsum('kbcj,ka->abcj',W1ovvo(t1,t2,eris),t1)
    Wabcj += 2*einsum('alcd,ljdb->abcj',eris.vovv,t2)
    Wabcj +=  -einsum('alcd,ljbd->abcj',eris.vovv,t2)
    Wabcj +=  -einsum('aldc,ljdb->abcj',eris.vovv,t2)
    Wabcj +=  -einsum('bkdc,jkda->abcj',eris.vovv,t2)
    Wabcj +=   einsum('lkjc,lkba->abcj',eris.ooov,t2)
    Wabcj +=   einsum('lkjc,lb,ka->abcj',eris.ooov,t1,t1)
    Wabcj +=  -einsum('kc,kjab->abcj',cc_Fov(t1,t2,eris),t2)
    return Wabcj

def Wovoo(t1,t2,eris):
    Wkbij = np.array(eris.ooov).transpose(2,3,0,1).conj()
    Wkbij +=   einsum('kbid,jd->kbij',W1ovov(t1,t2,eris),t1)
    Wkbij +=  -einsum('klij,lb->kbij',Woooo(t1,t2,eris),t1)
    Wkbij +=   einsum('kbcj,ic->kbij',W1ovvo(t1,t2,eris),t1)
    Wkbij += 2*einsum('klid,ljdb->kbij',eris.ooov,t2)
    Wkbij +=  -einsum('klid,jldb->kbij',eris.ooov,t2)
    Wkbij +=  -einsum('lkid,ljdb->kbij',eris.ooov,t2)
    Wkbij +=   einsum('bkdc,jidc->kbij',eris.vovv,t2)
    Wkbij +=   einsum('bkdc,jd,ic->kbij',eris.vovv,t1,t1)
    Wkbij +=  -einsum('lkjc,libc->kbij',eris.ooov,t2)
    Wkbij +=   einsum('kc,ijcb->kbij',cc_Fov(t1,t2,eris),t2)
    return Wkbij
