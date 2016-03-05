import numpy as np
import pyscf.pbc.tools as tools
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

def cc_Foo(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = tools.get_kconserv(cc._scf.cell, cc._kpts)
    Fki = np.empty((nkpts,nocc,nocc),dtype=t2.dtype)
    for ki in range(nkpts):
        kk = ki
        Fki[ki] = eris.fock[ki,:nocc,:nocc].copy()
        for kl in range(nkpts):
            for kc in range(nkpts):
                kd = kconserv[kk,kc,kl]
                Fki[ki] += einsum('klcd,ilcd->ki',2*eris.oovv[kk,kl,kc],t2[ki,kl,kc])
                Fki[ki] += einsum('kldc,ilcd->ki', -eris.oovv[kk,kl,kd],t2[ki,kl,kc])
            #if ki == kc:
            kd = kconserv[kk,ki,kl]
            Fki[ki] += einsum('klcd,ic,ld->ki',2*eris.oovv[kk,kl,ki],t1[ki],t1[kl])
            Fki[ki] += einsum('kldc,ic,ld->ki', -eris.oovv[kk,kl,kd],t1[ki],t1[kl])
    return Fki

def cc_Fvv(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = tools.get_kconserv(cc._scf.cell, cc._kpts)
    Fac = np.empty((nkpts,nvir,nvir),dtype=t2.dtype)
    for ka in range(nkpts):
        kc = ka
        Fac[ka] = eris.fock[ka,nocc:,nocc:].copy()
        for kl in range(nkpts):
            for kk in range(nkpts):
                kd = kconserv[kk,kc,kl]
                Fac[ka] += -einsum('klcd,klad->ac',2*eris.oovv[kk,kl,kc],t2[kk,kl,ka])
                Fac[ka] += -einsum('kldc,klad->ac', -eris.oovv[kk,kl,kd],t2[kk,kl,ka])
            #if kk == ka
            kd = kconserv[ka,kc,kl]
            Fac[ka] += -einsum('klcd,ka,ld->ac',2*eris.oovv[ka,kl,kc],t1[ka],t1[kl])
            Fac[ka] += -einsum('kldc,ka,ld->ac', -eris.oovv[ka,kl,kd],t1[ka],t1[kl])
    return Fac

def cc_Fov(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:]
    Fkc = fov.copy()
    for kk in range(nkpts):
        for kl in range(nkpts):
            Fkc[kk] += einsum('klcd,ld->kc',2*eris.oovv[kk,kl,kk],t1[kl])
            Fkc[kk] += einsum('kldc,ld->kc', -eris.oovv[kk,kl,kl],t1[kl])
    return Fkc

### Eqs. (40)-(41) "lambda"

def Loo(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:]
    Lki = cc_Foo(cc,t1,t2,eris)
    for ki in range(nkpts):
        kk = ki
        Lki[ki] += einsum('kc,ic->ki',fov[kk],t1[ki])
        for kl in range(nkpts):
            Lki[ki] += einsum('klic,lc->ki',2*eris.ooov[kk,kl,ki],t1[kl])
            Lki[ki] += einsum('klci,lc->ki', -eris.oovo[kk,kl,kl],t1[kl])
    return Lki

def Lvv(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:]
    Lac = cc_Fvv(cc,t1,t2,eris)
    for ka in range(nkpts):
        kc = ka
        Lac[ka] += -einsum('kc,ka->ac',fov[kc],t1[ka])
        for kk in range(nkpts):
            Lac[ka] += einsum('akcd,kd->ac',2*eris.vovv[ka,kk,kc],t1[kk])
            Lac[ka] += einsum('akdc,kd->ac', -eris.vovv[ka,kk,kk],t1[kk])
    return Lac

### Eqs. (42)-(45) "chi"

def cc_Woooo(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = tools.get_kconserv(cc._scf.cell, cc._kpts)
    Wklij = np.array(eris.oooo, copy=True)
    for kk in range(nkpts):
        for kl in range(nkpts):
            for ki in range(nkpts):
                kj = kconserv[kk,ki,kl]
                Wklij[kk,kl,ki] += einsum('klic,jc->klij',eris.ooov[kk,kl,ki],t1[kj])
                Wklij[kk,kl,ki] += einsum('klcj,ic->klij',eris.oovo[kk,kl,ki],t1[ki])
                for kc in range(nkpts):
                    Wklij[kk,kl,ki] += einsum('klcd,ijcd->klij',eris.oovv[kk,kl,kc],t2[ki,kj,kc])
                    if kc == ki:
                        Wklij[kk,kl,ki] += einsum('klcd,ic,jd->klij',eris.oovv[kk,kl,kc],t1[ki],t1[kj])
    return Wklij

def cc_Wvvvv(cc,t1,t2,eris):
    ## Slow:
    nkpts, nocc, nvir = t1.shape
    Wabcd = np.array(eris.vvvv, copy=True)
    for ka in range(nkpts):
        for kb in range(nkpts):
            for kc in range(nkpts):
                Wabcd[ka,kb,kc] += einsum('akcd,kb->abcd',eris.vovv[ka,kb,kc],-t1[kb])
                Wabcd[ka,kb,kc] += einsum('kbcd,ka->abcd',eris.ovvv[ka,kb,kc],-t1[ka])

    ## Fast
    #nocc,nvir = t1.shape
    #Wabcd = np.empty((nvir,)*4)
    #for a in range(nvir):
    #    Wabcd[a,:] = einsum('kcd,kb->bcd',eris.vovv[a],-t1)
    ##Wabcd += einsum('kbcd,ka->abcd',eris.ovvv,-t1)
    #Wabcd += lib.dot(-t1.T,eris.ovvv.reshape(nocc,-1)).reshape((nvir,)*4)
    #Wabcd += np.asarray(eris.vvvv)

    return Wabcd

def cc_Wvoov(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = tools.get_kconserv(cc._scf.cell, cc._kpts)
    Wakic = np.array(eris.voov, copy=True)
    for ka in range(nkpts):
        for kk in range(nkpts):
            for ki in range(nkpts):
                kc = kconserv[ka,ki,kk]
                Wakic[ka,kk,ki] -= einsum('lkic,la->akic',eris.ooov[ka,kk,ki],t1[ka])
                Wakic[ka,kk,ki] += einsum('akdc,id->akic',eris.vovv[ka,kk,ki],t1[ki])
                for kl in range(nkpts):
                    # kl - kd + kk = kc
                    # => kd = kl - kc + kk
                    kd = kconserv[kl,kc,kk]
                    Wakic[ka,kk,ki] -= 0.5*einsum('lkdc,ilda->akic',eris.oovv[kl,kk,kd],t2[ki,kl,kd])
                    if kl == ka:
                        Wakic[ka,kk,ki] -= einsum('lkdc,id,la->akic',eris.oovv[ka,kk,ki],t1[ki],t1[ka])
                    Wakic[ka,kk,ki] += 0.5*einsum('lkdc,ilad->akic',2*eris.oovv[kl,kk,kd],t2[ki,kl,ka])
                    Wakic[ka,kk,ki] += 0.5*einsum('lkcd,ilad->akic', -eris.oovv[kl,kk,kc],t2[ki,kl,ka])
    return Wakic

def cc_Wvovo(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = tools.get_kconserv(cc._scf.cell, cc._kpts)
    Wakci = np.array(eris.vovo, copy=True)
    for ka in range(nkpts):
        for kk in range(nkpts):
            for kc in range(nkpts):
                ki = kconserv[ka,kc,kk]
                Wakci[ka,kk,kc] -= einsum('lkci,la->akci',eris.oovo[ka,kk,kc],t1[ka])
                Wakci[ka,kk,kc] += einsum('akcd,id->akci',eris.vovv[ka,kk,kc],t1[ki])
                for kl in range(nkpts):
                    kd = kconserv[kl,kc,kk]
                    Wakci[ka,kk,kc] -= 0.5*einsum('lkcd,ilda->akci',eris.oovv[kl,kk,kc],t2[ki,kl,kd])
                    if kl == ka:
                        Wakci[ka,kk,kc] -= einsum('lkcd,id,la->akci',eris.oovv[kl,kk,kc],t1[ki],t1[ka])
    return Wakci

#def cc_Woooo(cc,t1,t2,eris):
#    tau = make_tau(t2,t1,t1)
#    tmp = einsum('je,mnie->mnij',t1,eris.ooov)
#    Wmnij = eris.oooo + tmp - tmp.transpose(0,1,3,2)
#    Wmnij += 0.25*einsum('ijef,mnef->mnij',tau,eris.oovv)
#    return Wmnij
#
#def cc_Wvvvv(cc,t1,t2,eris):
#    eris_vovv = _cp(eris.ovvv).transpose(1,0,3,2)
#    tau = make_tau(t2,t1,t1)
#    tmp = einsum('mb,amef->abef',t1,eris_vovv)
#    Wabef = eris.vvvv - tmp + tmp.transpose(1,0,2,3)
#    Wabef += 0.25*einsum('mnab,mnef->abef',tau,eris.oovv)
#    return Wabef
#
#def cc_Wovvo(cc,t1,t2,eris):
#    eris_ovvo = - _cp(eris.ovov).transpose(0,1,3,2)
#    eris_oovo = - _cp(eris.ooov).transpose(0,1,3,2)
#    Wmbej = eris_ovvo.copy()
#    Wmbej +=  einsum('jf,mbef->mbej',t1,eris.ovvv)
#    Wmbej += -einsum('nb,mnej->mbej',t1,eris_oovo)
#    Wmbej += -0.5*einsum('jnfb,mnef->mbej',t2,eris.oovv)
#    Wmbej += -einsum('jf,nb,mnef->mbej',t1,t1,eris.oovv)
#    return Wmbej
#
#def Woooo(cc,t1,t2,eris):
#    tau = make_tau(t2,t1,t1)
#    Wmnij = cc_Woooo(cc,t1,t2,eris) + 0.25*einsum('ijef,mnef->mnij',tau,eris.oovv)
#    return Wmnij
#
#def Wvvvv(cc,t1,t2,eris):
#    tau = make_tau(t2,t1,t1)
#    Wabef = cc_Wvvvv(cc,t1,t2,eris) + 0.25*einsum('mnab,mnef->abef',tau,eris.oovv)
#    return Wabef
#
#def Wovvo(cc,t1,t2,eris):
#    Wmbej = cc_Wovvo(cc,t1,t2,eris) - 0.5*einsum('jnfb,mnef->mbej',t2,eris.oovv)
#    return Wmbej
#
## Indices in the following can be safely permuted.
#
#def Wooov(cc,t1,t2,eris):
#    Wmnie = eris.ooov + einsum('if,mnfe->mnie',t1,eris.oovv)
#    return Wmnie
#
#def Wvovv(cc,t1,t2,eris):
#    eris_vovv = - _cp(eris.ovvv).transpose(1,0,2,3)
#    Wamef = eris_vovv - einsum('na,nmef->amef',t1,eris.oovv)
#    return Wamef
#
#def Wovoo(cc,t1,t2,eris):
#    eris_ovvo = - _cp(eris.ovov).transpose(0,1,3,2)
#    tmp1 = einsum('mnie,jnbe->mbij',eris.ooov,t2)
#    tmp2 = ( einsum('ie,mbej->mbij',t1,eris_ovvo)
#            - einsum('ie,njbf,mnef->mbij',t1,t2,eris.oovv) )
#    FFov = Fov(cc,t1,t2,eris)
#    WWoooo = Woooo(cc,t1,t2,eris)
#    tau = make_tau(t2,t1,t1)
#    Wmbij = ( eris.ovoo - einsum('me,ijbe->mbij',FFov,t2)
#              - einsum('nb,mnij->mbij',t1,WWoooo)
#              + 0.5 * einsum('mbef,ijef->mbij',eris.ovvv,tau)
#              + tmp1 - tmp1.transpose(0,1,3,2)
#              + tmp2 - tmp2.transpose(0,1,3,2) )
#    return Wmbij
#
#def Wvvvo(cc,t1,t2,eris):
#    eris_ovvo = - _cp(eris.ovov).transpose(0,1,3,2)
#    eris_vvvo = - _cp(eris.ovvv).transpose(2,3,1,0).conj()
#    eris_oovo = - _cp(eris.ooov).transpose(0,1,3,2)
#    tmp1 = einsum('mbef,miaf->abei',eris.ovvv,t2)
#    tmp2 = ( einsum('ma,mbei->abei',t1,eris_ovvo)
#            - einsum('ma,nibf,mnef->abei',t1,t2,eris.oovv) )
#    FFov = Fov(cc,t1,t2,eris)
#    WWvvvv = Wvvvv(cc,t1,t2,eris)
#    tau = make_tau(t2,t1,t1)
#    Wabei = ( eris_vvvo - einsum('me,miab->abei',FFov,t2)
#                    + einsum('if,abef->abei',t1,WWvvvv)
#                    + 0.5 * einsum('mnei,mnab->abei',eris_oovo,tau)
#                    - tmp1 + tmp1.transpose(1,0,2,3)
#                    - tmp2 + tmp2.transpose(1,0,2,3) )
#    return Wabei
