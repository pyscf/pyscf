import tempfile
import h5py
import numpy as np
from pyscf import lib

#einsum = np.einsum
einsum = lib.einsum

# Ref: Gauss and Stanton, J. Chem. Phys. 103, 3561 (1995) Table III

# Section (a)

def make_tau(t2, t1a, t1b, fac=1, out=None):
    #:tmp = einsum('ia,jb->ijab',t1a,t1b)
    #:t1t1 = tmp - tmp.transpose(1,0,2,3) - tmp.transpose(0,1,3,2) + tmp.transpose(1,0,3,2)
    #:tau1 = t2 + fac*0.50*t1t1
    tau1  = np.einsum('ia,jb->ijab', t1a, t1b)
    tau1 -= np.einsum('ia,jb->jiab', t1a, t1b)
    tau1 = tau1 - tau1.transpose(0,1,3,2)
    tau1 *= fac * .5
    tau1 += t2
    return tau1

def cc_Fvv(t1,t2,eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    fvv = eris.fock[nocc:,nocc:]
    Fae = fvv.copy()
    Fae -= 0.5*einsum('me,ma->ae',fov,t1)
    for i in range(nocc):
        vvv = eris.ovvv[i]
        Fae += np.einsum('f,fae->ae', t1[i], vvv)
        Fae -= np.einsum('f,eaf->ae', t1[i], vvv)
    tau_tilde = make_tau(t2,t1,t1,fac=0.5)
    eris_ovov = np.asarray(eris.ovov)
    Fae -= 0.5*einsum('mnaf,menf->ae',tau_tilde,eris_ovov)
    Fae -= 0.5*einsum('mnaf,menf->ae',tau_tilde,eris_ovov)
    return Fae

def cc_Foo(t1,t2,eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    foo = eris.fock[:nocc,:nocc]
    tau_tilde = make_tau(t2,t1,t1,fac=0.5)
    Fmi = foo + 0.5*einsum('me,ie->mi',fov,t1)
    eris_ooov = np.asarray(eris.ooov)
    Fmi += np.einsum('ne,mine->mi',t1,eris_ooov)
    Fmi -= np.einsum('ne,nime->mi',t1,eris_ooov)
    eris_ovov = np.asarray(eris.ovov)
    Fmi += 0.5*einsum('inef,menf->mi',tau_tilde,eris_ovov)
    Fmi -= 0.5*einsum('inef,nemf->mi',tau_tilde,eris_ovov)
    return Fmi

def cc_Fov(t1,t2,eris):
    nocc, nvir = t1.shape
    eris_ovov = np.asarray(eris.ovov)
    fov = eris.fock[:nocc,nocc:]
    Fme = fov.copy()
    Fme += np.einsum('nf,menf->me',t1,eris_ovov)
    Fme -= np.einsum('nf,mfne->me',t1,eris_ovov)
    return Fme

def cc_Woooo(t1,t2,eris):
    tau = make_tau(t2,t1,t1)
    eris_ooov = np.asarray(eris.ooov)
    tmp  = einsum('je,mine->mnij', t1, eris_ooov)
    tmp -= einsum('je,nime->mnij', t1, eris_ooov)
    tmp += np.asarray(eris.oooo).transpose(0,2,1,3)
    tmp += 0.25*einsum('ijef,menf->mnij', tau, eris.ovov)
    Wmnij = tmp - tmp.transpose(0,1,3,2)
    return Wmnij

def cc_Wovvo(t1,t2,eris):
    nocc, nvir = t1.shape
    eris_ooov = np.asarray(eris.ooov)
    Wmbej  = einsum('nb,mjne->mbej',t1,eris_ooov)
    Wmbej -= einsum('nb,njme->mbej',t1,eris_ooov)
    eris_ooov = None
    for i in range(nocc):
        vvv = eris.ovvv[i]
        Wmbej[i] += einsum('jf,ebf->bej', t1, vvv)
        Wmbej[i] -= einsum('jf,fbe->bej', t1, vvv)
    Wmbej += np.asarray(eris.ovvo).transpose(0,2,1,3)
    Wmbej -= np.asarray(eris.oovv).transpose(0,2,3,1)
    eris_ovov = np.asarray(eris.ovov)
    Wmbej -= 0.5*einsum('jnfb,menf->mbej',t2,eris_ovov)
    Wmbej += 0.5*einsum('jnfb,mfne->mbej',t2,eris_ovov)
    tmp = einsum('jf,menf->mnej', t1, eris_ovov)
    tmp = tmp - tmp.transpose(1,0,2,3)
    eris_ovov = None
    Wmbej -= einsum('nb,mnej->mbej', t1, tmp)
    return Wmbej

### Section (b)

def Fvv(t1,t2,eris):
    ccFov = cc_Fov(t1,t2,eris)
    Fae = cc_Fvv(t1,t2,eris) - 0.5*einsum('ma,me->ae',t1,ccFov)
    return Fae

def Foo(t1,t2,eris):
    ccFov = cc_Fov(t1,t2,eris)
    Fmi = cc_Foo(t1,t2,eris) + 0.5*einsum('ie,me->mi',t1,ccFov)
    return Fmi

def Fov(t1,t2,eris):
    Fme = cc_Fov(t1,t2,eris)
    return Fme

def Woooo(t1,t2,eris):
    tau = make_tau(t2,t1,t1)
    eris_ooov = np.asarray(eris.ooov)
    tmp  = einsum('je,mine->mnij', t1, eris_ooov)
    tmp -= einsum('je,nime->mnij', t1, eris_ooov)
    tmp += np.asarray(eris.oooo).transpose(0,2,1,3)
    tmp += 0.5*einsum('ijef,menf->mnij', tau, eris.ovov)
    Wmnij = tmp - tmp.transpose(0,1,3,2)
    return Wmnij

def Wvvvv(t1,t2,eris):
    tau = make_tau(t2,t1,t1)
    #Wabef = cc_Wvvvv(t1,t2,eris) + 0.25*einsum('mnab,mnef->abef',tau,eris.oovv)
    ds_type = t1.dtype.char
    nocc, nvir = t1.shape
    Wabef = eris.feri.create_dataset('Wvvvv', (nvir,nvir,nvir,nvir), ds_type)
    #_cc_Wvvvv = cc_Wvvvv(t1,t2,eris)
    eris_ovvv = np.asarray(eris.ovvv)
    eris_ovov = np.asarray(eris.ovov)
    for a in range(nvir):
        vvv = einsum('mb,mef->bef', t1, eris_ovvv[:,:,a,:])
        vvv -= einsum('m,mebf->bef', t1[:,a], eris_ovvv)
        vvv += 0.5*einsum('mnb,menf->bef', tau[:,:,a,:], eris_ovov)
        vvv += eris.vvvv[a].transpose(1,0,2)
        Wabef[a] = vvv - vvv.transpose(0,2,1)
    return Wabef

def Wovvo(t1,t2,eris):
    Wmbej = cc_Wovvo(t1,t2,eris)
    eris_ovov = np.asarray(eris.ovov)
    Wmbej -= 0.5*einsum('jnfb,menf->mbej',t2,eris_ovov)
    Wmbej += 0.5*einsum('jnfb,mfne->mbej',t2,eris_ovov)
    return Wmbej

def Wooov(t1,t2,eris):
    tmp = np.asarray(eris.ooov).transpose(0,2,1,3) + einsum('if,mfne->mnie',t1,eris.ovov)
    Wmnie = tmp - tmp.transpose(1,0,2,3)
    return Wmnie

def Wvovv(t1,t2,eris):
    tmp = -np.array(eris.ovvv).transpose(0,2,1,3) - einsum('na,nemf->maef',t1,eris.ovov)
    Wamef = tmp - tmp.transpose(0,1,3,2)
    return Wamef.transpose(1,0,2,3)

def Wovoo(t1,t2,eris):
    nocc, nvir = t1.shape
    eris_ooov = np.asarray(eris.ooov)
    Wmbij  = einsum('mine,jnbe->mbij', eris_ooov, t2)
    Wmbij -= einsum('nime,jnbe->mbij', eris_ooov, t2)
    eris_ooov = None
    eris_ovvo = np.array(eris.ovvo)
    Wmbij += einsum('ie,mebj->mbij',t1, eris_ovvo)
    eris_oovv = np.array(eris.oovv)
    Wmbij -= einsum('ie,mjbe->mbij',t1, eris_oovv)
    eris_oovv = eris_ovvo = None
    eris_ovov = np.asarray(eris.ovov)
    tmp  = einsum('njbf,menf->mbej', t2, eris_ovov)
    tmp -= einsum('njbf,nemf->mbej', t2, eris_ovov)
    eris_ovov = None
    Wmbij -= einsum('ie,mbej->mbij', t1, tmp)
    Wmbij += np.asarray(eris.oovo).transpose(0,2,1,3)
    Wmbij = Wmbij - Wmbij.transpose(0,1,3,2)
    FFov = Fov(t1,t2,eris)
    WWoooo = Woooo(t1,t2,eris)
    Wmbij -= einsum('me,ijbe->mbij', FFov, t2)
    Wmbij -= einsum('nb,mnij->mbij', t1, WWoooo)
    tau = make_tau(t2,t1,t1)
    for i in range(nocc):
        vvv = eris.ovvv[i]
        Wmbij[i] += 0.5 * einsum('ebf,ijef->bij', vvv, tau)
        Wmbij[i] -= 0.5 * einsum('fbe,ijef->bij', vvv, tau)
    return Wmbij

def Wvvvo(t1,t2,eris,_Wvvvv=None):
    nocc,nvir = t1.shape
    Wabei = np.asarray(eris.ovvv).transpose(1,3,2,0).conj()
    Wabei = Wabei.transpose(1,0,2,3) - Wabei
    FFov = Fov(t1,t2,eris)
    Wabei += -einsum('me,miab->abei',FFov,t2)
    tau = make_tau(t2,t1,t1)
    eris_ooov = np.asarray(eris.ooov)
    Wabei -= 0.5 * einsum('mine,mnab->abei',eris_ooov,tau)
    Wabei += 0.5 * einsum('nime,mnab->abei',eris_ooov,tau)
    eris_ooov = None
    eris_ovov = np.asarray(eris.ovov)
    tmp2  = einsum('nibf,menf->mbei', t2, eris_ovov)
    tmp2 -= einsum('nibf,nemf->mbei', t2, eris_ovov)
    tmp2  = einsum('ma,mbei->abei', -t1, tmp2)
    eris_ovov = None
    eris_ovvo = np.asarray(eris.ovvo)
    eris_oovv = np.asarray(eris.oovv)
    tmp2 += einsum('ma,mebi->abei', t1, eris_ovvo)
    tmp2 -= einsum('ma,mibe->abei', t1, eris_oovv)
    eris_ovov = eris_ovvo = None
    Wabei += -tmp2 + tmp2.transpose(1,0,2,3)
    tmp1 = np.zeros((nvir,nvir,nvir,nocc), dtype=t2.dtype)
    for i in range(nocc):
        vvv = eris.ovvv[i]
        tmp1 += einsum('ebf,iaf->abei', vvv, t2[i])
        tmp1 -= einsum('fbe,iaf->abei', vvv, t2[i])
    Wabei += -tmp1 + tmp1.transpose(1,0,2,3)
    if _Wvvvv is None:
        _Wvvvv = Wvvvv(t1,t2,eris)
    for a in range(nvir):
        Wabei[a] += einsum('if,bef->bei',t1,_Wvvvv[a])
    return Wabei

