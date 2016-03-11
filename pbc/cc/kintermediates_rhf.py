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
    Fkc = np.empty((nkpts,nocc,nvir),dtype=t2.dtype)
    Fkc[:] = eris.fock[:,:nocc,nocc:].copy()
    for kk in range(nkpts):
        for kl in range(nkpts):
            Fkc[kk] += 2*einsum('klcd,ld->kc',eris.oovv[kk,kl,kk],t1[kl])
            Fkc[kk] +=  -einsum('kldc,ld->kc',eris.oovv[kk,kl,kl],t1[kl])
    return Fkc

### Eqs. (40)-(41) "lambda"

def Loo(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:]
    Lki = cc_Foo(cc,t1,t2,eris)
    for ki in range(nkpts):
        Lki[ki] += einsum('kc,ic->ki',fov[ki],t1[ki])
        for kl in range(nkpts):
            Lki[ki] += einsum('klic,lc->ki',2*eris.ooov[ki,kl,ki],t1[kl])
            Lki[ki] += einsum('klci,lc->ki', -eris.oovo[ki,kl,kl],t1[kl])
    return Lki

def Lvv(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:]
    Lac = cc_Fvv(cc,t1,t2,eris)
    for ka in range(nkpts):
        Lac[ka] += -einsum('kc,ka->ac',fov[ka],t1[ka])
        for kk in range(nkpts):
            Lac[ka] += einsum('akcd,kd->ac',2*eris.vovv[ka,kk,ka],t1[kk])
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


########################################################
#        EOM Intermediates w/ k-points                 #
########################################################

# Indices in the following can be safely permuted.

def Wooov(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = tools.get_kconserv(cc._scf.cell, cc._kpts)

    Wklid = np.array(eris.ooov, copy=True)
    for kk in range(nkpts):
        for kl in range(nkpts):
            for ki in range(nkpts):
                kd = kconserv[kk,ki,kl]
                Wklid[kk,kl,ki] += einsum('ic,klcd->klid',t1[ki],eris.oovv[kk,kl,ki])
    return Wklid

def Wvovv(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = tools.get_kconserv(cc._scf.cell, cc._kpts)

    Walcd = np.empty((nkpts,nkpts,nkpts,nvir,nocc,nvir,nvir),dtype=t1.dtype)
    for ka in range(nkpts):
        for kl in range(nkpts):
            for kc in range(nkpts):
                kd = kconserv[ka,kc,kl]
                # vovv[ka,kl,kc,kd] <= ovvv[kl,ka,kd,kc].transpose(1,0,3,2)
                Walcd[ka,kl,kc] = np.array(eris.ovvv[kl,ka,kd]).transpose(1,0,3,2)
                Walcd[ka,kl,kc] += -einsum('ka,klcd->alcd',t1[ka],eris.oovv[ka,kl,kc])
    return Walcd

def W1ovvo(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = tools.get_kconserv(cc._scf.cell, cc._kpts)

    Wkaci = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc),dtype=t1.dtype)
    for kk in range(nkpts):
        for ka in range(nkpts):
            for kc in range(nkpts):
                ki = kconserv[kk,kc,ka]
                # ovvo[kk,ka,kc,ki] => voov[ka,kk,ki,kc]
                Wkaci[kk,ka,kc] = np.array(eris.voov[ka,kk,ki]).transpose(1,0,3,2)
                for kl in range(nkpts):
                    kd = kconserv[ki,ka,kl]
                    Wkaci[kk,ka,kc] += 2.*einsum('klcd,ilad->kaci',eris.oovv[kk,kl,kc],t2[ki,kl,ka])
                    Wkaci[kk,ka,kc] +=   -einsum('klcd,liad->kaci',eris.oovv[kk,kl,kc],t2[kl,ki,ka])
                    Wkaci[kk,ka,kc] +=   -einsum('kldc,ilad->kaci',eris.oovv[kk,kl,kd],t2[ki,kl,ka])
    return Wkaci

def W2ovvo(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = tools.get_kconserv(cc._scf.cell, cc._kpts)

    Wkaci = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc),dtype=t1.dtype)
    WWooov = Wooov(cc,t1,t2,eris)
    for kk in range(nkpts):
        for ka in range(nkpts):
            for kc in range(nkpts):
                ki = kconserv[kk,kc,ka]
                Wkaci[kk,ka,kc] = einsum('la,lkic->kaci',-t1[ka],WWooov[ka,kk,ki])
                Wkaci[kk,ka,kc] += einsum('akdc,id->kaci',eris.vovv[ka,kk,ki],t1[ki])
    return Wkaci

def Wovvo(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = tools.get_kconserv(cc._scf.cell, cc._kpts)

    return W1ovvo(cc,t1,t2,eris) + W2ovvo(cc,t1,t2,eris)

def W1ovov(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = tools.get_kconserv(cc._scf.cell, cc._kpts)

    Wkbid = np.array(eris.ovov, copy=True)
    for kk in range(nkpts):
        for kb in range(nkpts):
            for ki in range(nkpts):
                kd = kconserv[kk,ki,kb]
                #   kk + kl - kc - kd = 0
                # => kc = kk - kd + kl
                for kl in range(nkpts):
                    kc = kconserv[kk,kd,kl]
                    Wkbid[kk,kb,ki] += -einsum('klcd,ilcb->kbid',eris.oovv[kk,kl,kc],t2[ki,kl,kc])
    return Wkbid

def W2ovov(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = tools.get_kconserv(cc._scf.cell, cc._kpts)

    Wkbid = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir),dtype=t1.dtype)
    WWooov = Wooov(cc,t1,t2,eris)
    for kk in range(nkpts):
        for kb in range(nkpts):
            for ki in range(nkpts):
                kd = kconserv[kk,ki,kb]
                Wkbid[kk,kb,ki] = einsum('klid,lb->kbid',WWooov[kk,kb,ki],-t1[kb])
                Wkbid[kk,kb,ki] += einsum('bkdc,ic->kbid',eris.vovv[kb,kk,kd],t1[ki])
    return Wkbid

def Wovov(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = tools.get_kconserv(cc._scf.cell, cc._kpts)
    return W1ovov(cc,t1,t2,eris) + W2ovov(cc,t1,t2,eris)

def Woooo(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = tools.get_kconserv(cc._scf.cell, cc._kpts)

    Wklij = np.array(eris.oooo, copy=True)
    for kk in range(nkpts):
        for kl in range(nkpts):
            for ki in range(nkpts):
                kj = kconserv[kk,ki,kl]
                for kc in range(nkpts):
                    kd = kconserv[kk,kc,kl]
                    Wklij[kk,kl,ki] += einsum('klcd,ijcd->klij',eris.oovv[kk,kl,kc],t2[ki,kj,kc])
                    if ki == kc and kj == kd:
                        Wklij[kk,kl,ki] += einsum('klcd,ic,jd->klij',eris.oovv[kk,kl,kc],t1[ki],t1[kj])
                Wklij[kk,kl,ki] += einsum('klid,jd->klij',eris.ooov[kk,kl,ki],t1[kj])
                Wklij[kk,kl,ki] += einsum('lkjc,ic->klij',eris.ooov[kl,kk,kj],t1[ki])
    return Wklij

def Wvvvv(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = tools.get_kconserv(cc._scf.cell, cc._kpts)

    Wabcd = np.array(eris.vvvv, copy=True)
    for ka in range(nkpts):
        for kb in range(nkpts):
            for kc in range(nkpts):
                kd = kconserv[ka,kc,kb]
                for kk in range(nkpts):
                    # kk + kl - kc - kd = 0
                    # => kl = kc - kk + kd
                    kl = kconserv[kc,kk,kd]
                    Wabcd[ka,kb,kc] += einsum('klcd,klab->abcd',eris.oovv[kk,kl,kc],t2[kk,kl,ka])
                    if kl == kb and kk == ka:
                        Wabcd[ka,kb,kc] += einsum('klcd,ka,lb->abcd',eris.oovv[kk,kl,kc],t1[ka],t1[kb])
                Wabcd[ka,kb,kc] += einsum('alcd,lb->abcd',eris.vovv[ka,kb,kc],-t1[kb])
                Wabcd[ka,kb,kc] += einsum('bkdc,ka->abcd',eris.vovv[kb,ka,kd],-t1[ka])
    return Wabcd

def Wvvvo(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = tools.get_kconserv(cc._scf.cell, cc._kpts)

    Wabcj = np.empty((nkpts,nkpts,nkpts,nvir,nvir,nvir,nocc),dtype=t1.dtype)
    WWvvvv = Wvvvv(cc,t1,t2,eris)
    WW1ovov = W1ovov(cc,t1,t2,eris)
    WW1ovvo = W1ovvo(cc,t1,t2,eris)
    FFov = cc_Fov(cc,t1,t2,eris)
    for ka in range(nkpts):
        for kb in range(nkpts):
            for kc in range(nkpts):
                kj = kconserv[ka,kc,kb]
                # vvvo[ka,kb,kc,kj] <= vovv[kc,kj,ka,kb].transpose(2,3,0,1).conj()
                Wabcj[ka,kb,kc] = np.array(eris.vovv[kc,kj,ka]).transpose(2,3,0,1).conj()
                Wabcj[ka,kb,kc] += einsum('abcd,jd->abcj',WWvvvv[ka,kb,kc],t1[kj])
                # Wvovo[ka,kl,kc,kj] <= Wovov[kl,ka,kj,kc].transpose(1,0,3,2)
                Wabcj[ka,kb,kc] += einsum('alcj,lb->abcj',WW1ovov[kb,ka,kj].transpose(1,0,3,2),-t1[kb])
                Wabcj[ka,kb,kc] += einsum('kbcj,ka->abcj',WW1ovvo[ka,kb,kc],-t1[ka])

                for kl in range(nkpts):
                    # ka + kl - kc - kd = 0
                    # => kd = ka - kc + kl
                    kd = kconserv[ka,kc,kl]
                    Wabcj[ka,kb,kc] += einsum('alcd,ljdb->abcj',eris.vovv[ka,kl,kc],2.*t2[kl,kj,kd])
                    Wabcj[ka,kb,kc] += einsum('alcd,ljbd->abcj',eris.vovv[ka,kl,kc],  -t2[kl,kj,kb])
                    Wabcj[ka,kb,kc] += einsum('aldc,ljdb->abcj',eris.vovv[ka,kl,kd],  -t2[kl,kj,kd])
                    # kb - kc + kl = kd
                    kd = kconserv[kb,kc,kl]
                    Wabcj[ka,kb,kc] += einsum('bldc,jlda->abcj',eris.vovv[kb,kl,kd],  -t2[kj,kl,kd])

                    # kl + kk - kb - ka = 0
                    # => kk = kb + ka - kl
                    kk = kconserv[kb,kl,ka]
                    Wabcj[ka,kb,kc] += einsum('lkjc,lkba->abcj',eris.ooov[kl,kk,kj],t2[kl,kk,kb])
                    if kk == ka and kl == kb:
                        Wabcj[ka,kb,kc] += einsum('lkjc,lb,ka->abcj',eris.ooov[kb,ka,kj],t1[kb],t1[ka])
                Wabcj[ka,kb,kc] += einsum('lc,ljab->abcj',-FFov[kc],t2[kc,kj,ka])
    return Wabcj

def Wovoo(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = tools.get_kconserv(cc._scf.cell, cc._kpts)

    WW1ovov = W1ovov(cc,t1,t2,eris)
    WWoooo = Woooo(cc,t1,t2,eris)
    WW1ovvo = W1ovvo(cc,t1,t2,eris)
    FFov = cc_Fov(cc,t1,t2,eris)

    Wkbij = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc),dtype=t1.dtype)
    for kk in range(nkpts):
        for kb in range(nkpts):
            for ki in range(nkpts):
                kj = kconserv[kk,ki,kb]
                # ovoo[kk,kb,ki,kj] <= oovo[kj,ki,kb,kk].transpose(3,2,1,0).conj()
                Wkbij[kk,kb,ki] = np.array(eris.oovo[kj,ki,kb]).transpose(3,2,1,0).conj()
                Wkbij[kk,kb,ki] += einsum('kbid,jd->kbij',WW1ovov[kk,kb,ki], t1[kj])
                Wkbij[kk,kb,ki] += einsum('klij,lb->kbij',WWoooo[kk,kb,ki],-t1[kb])
                Wkbij[kk,kb,ki] += einsum('kbcj,ic->kbij',WW1ovvo[kk,kb,ki],t1[ki])

                for kd in range(nkpts):
                    # kk + kl - ki - kd = 0
                    # => kl = ki - kk + kd
                    kl = kconserv[ki,kk,kd]
                    Wkbij[kk,kb,ki] += einsum('klid,ljdb->kbij', 2.*eris.ooov[kk,kl,ki],t2[kl,kj,kd])
                    Wkbij[kk,kb,ki] += einsum('klid,jldb->kbij',   -eris.ooov[kk,kl,ki],t2[kj,kl,kd])
                    Wkbij[kk,kb,ki] += einsum('lkid,ljdb->kbij',   -eris.ooov[kl,kk,ki],t2[kl,kj,kd])
                    kl = kconserv[kb,ki,kd]
                    Wkbij[kk,kb,ki] += einsum('lkjd,libd->kbij',   -eris.ooov[kl,kk,kj],t2[kl,ki,kb])

                    # kb + kk - kd = kc
                    kc = kconserv[kb,kd,kk]
                    Wkbij[kk,kb,ki] += einsum('bkdc,jidc->kbij',eris.vovv[kb,kk,kd],t2[kj,ki,kd])
                    if ki == kc and kj == kd:
                        Wkbij[kk,kb,ki] += einsum('bkdc,jd,ic->kbij',eris.vovv[kb,kk,kd],t1[kj],t1[ki])
                Wkbij[kk,kb,ki] += einsum('kc,ijcb->kbij',FFov[kk],t2[ki,kj,kk])
    return Wkbij
