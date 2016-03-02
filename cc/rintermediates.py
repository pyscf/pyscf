import numpy as np
from pyscf import lib
from pyscf.pbc import lib as pbclib

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

#def cc_Woooo(t1,t2,eris):
#    tau = make_tau(t2,t1,t1)
#    tmp = einsum('je,mnie->mnij',t1,eris.ooov)
#    Wmnij = eris.oooo + tmp - tmp.transpose(0,1,3,2)
#    Wmnij += 0.25*einsum('ijef,mnef->mnij',tau,eris.oovv)
#    return Wmnij
#
#def cc_Wvvvv(t1,t2,eris):
#    eris_vovv = _cp(eris.ovvv).transpose(1,0,3,2)
#    tau = make_tau(t2,t1,t1)
#    tmp = einsum('mb,amef->abef',t1,eris_vovv)
#    Wabef = eris.vvvv - tmp + tmp.transpose(1,0,2,3)
#    Wabef += 0.25*einsum('mnab,mnef->abef',tau,eris.oovv)
#    return Wabef
#
#def cc_Wovvo(t1,t2,eris):
#    eris_ovvo = - _cp(eris.ovov).transpose(0,1,3,2)
#    eris_oovo = - _cp(eris.ooov).transpose(0,1,3,2)
#    Wmbej = eris_ovvo.copy()
#    Wmbej +=  einsum('jf,mbef->mbej',t1,eris.ovvv)
#    Wmbej += -einsum('nb,mnej->mbej',t1,eris_oovo)
#    Wmbej += -0.5*einsum('jnfb,mnef->mbej',t2,eris.oovv)
#    Wmbej += -einsum('jf,nb,mnef->mbej',t1,t1,eris.oovv)
#    return Wmbej
#
#def Woooo(t1,t2,eris):
#    tau = make_tau(t2,t1,t1)
#    Wmnij = cc_Woooo(t1,t2,eris) + 0.25*einsum('ijef,mnef->mnij',tau,eris.oovv)
#    return Wmnij
#
#def Wvvvv(t1,t2,eris):
#    tau = make_tau(t2,t1,t1)
#    Wabef = cc_Wvvvv(t1,t2,eris) + 0.25*einsum('mnab,mnef->abef',tau,eris.oovv)
#    return Wabef
#
#def Wovvo(t1,t2,eris):
#    Wmbej = cc_Wovvo(t1,t2,eris) - 0.5*einsum('jnfb,mnef->mbej',t2,eris.oovv)
#    return Wmbej
#
## Indices in the following can be safely permuted.
#
#def Wooov(t1,t2,eris):
#    Wmnie = eris.ooov + einsum('if,mnfe->mnie',t1,eris.oovv)
#    return Wmnie
#
#def Wvovv(t1,t2,eris):
#    eris_vovv = - _cp(eris.ovvv).transpose(1,0,2,3)
#    Wamef = eris_vovv - einsum('na,nmef->amef',t1,eris.oovv)
#    return Wamef
#
#def Wovoo(t1,t2,eris):
#    eris_ovvo = - _cp(eris.ovov).transpose(0,1,3,2)
#    tmp1 = einsum('mnie,jnbe->mbij',eris.ooov,t2)
#    tmp2 = ( einsum('ie,mbej->mbij',t1,eris_ovvo)
#            - einsum('ie,njbf,mnef->mbij',t1,t2,eris.oovv) )
#    FFov = Fov(t1,t2,eris)
#    WWoooo = Woooo(t1,t2,eris)
#    tau = make_tau(t2,t1,t1)
#    Wmbij = ( eris.ovoo - einsum('me,ijbe->mbij',FFov,t2)
#              - einsum('nb,mnij->mbij',t1,WWoooo)
#              + 0.5 * einsum('mbef,ijef->mbij',eris.ovvv,tau)
#              + tmp1 - tmp1.transpose(0,1,3,2)
#              + tmp2 - tmp2.transpose(0,1,3,2) )
#    return Wmbij
#
#def Wvvvo(t1,t2,eris):
#    eris_ovvo = - _cp(eris.ovov).transpose(0,1,3,2)
#    eris_vvvo = - _cp(eris.ovvv).transpose(2,3,1,0).conj()
#    eris_oovo = - _cp(eris.ooov).transpose(0,1,3,2)
#    tmp1 = einsum('mbef,miaf->abei',eris.ovvv,t2)
#    tmp2 = ( einsum('ma,mbei->abei',t1,eris_ovvo)
#            - einsum('ma,nibf,mnef->abei',t1,t2,eris.oovv) )
#    FFov = Fov(t1,t2,eris)
#    WWvvvv = Wvvvv(t1,t2,eris)
#    tau = make_tau(t2,t1,t1)
#    Wabei = ( eris_vvvo - einsum('me,miab->abei',FFov,t2)
#                    + einsum('if,abef->abei',t1,WWvvvv)
#                    + 0.5 * einsum('mnei,mnab->abei',eris_oovo,tau)
#                    - tmp1 + tmp1.transpose(1,0,2,3)
#                    - tmp2 + tmp2.transpose(1,0,2,3) )
#    return Wabei
