#!/usr/bin/env python
#
# Authors: James D. McClain <jmcclain@princeton.edu>
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import tempfile
import numpy as np
import h5py
from pyscf import lib

#einsum = np.einsum
einsum = lib.einsum

# This is restricted (R)CCSD
# Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)

### Eqs. (37)-(39) "kappa"

def cc_Foo(t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    Fki = np.empty((nkpts,nocc,nocc),dtype=t2.dtype)
    for ki in range(nkpts):
        kk = ki
        Fki[ki] = eris.fock[ki,:nocc,:nocc].copy()
        for kl in range(nkpts):
            for kc in range(nkpts):
                kd = kconserv[kk,kc,kl]
                Soovv = 2*eris.oovv[kk,kl,kc] - eris.oovv[kk,kl,kd].transpose(0,1,3,2)
                Fki[ki] += einsum('klcd,ilcd->ki',Soovv,t2[ki,kl,kc])
            #if ki == kc:
            kd = kconserv[kk,ki,kl]
            Soovv = 2*eris.oovv[kk,kl,ki] - eris.oovv[kk,kl,kd].transpose(0,1,3,2)
            Fki[ki] += einsum('klcd,ic,ld->ki',Soovv,t1[ki],t1[kl])
    return Fki

def cc_Fvv(t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    Fac = np.empty((nkpts,nvir,nvir),dtype=t2.dtype)
    for ka in range(nkpts):
        kc = ka
        Fac[ka] = eris.fock[ka,nocc:,nocc:].copy()
        for kl in range(nkpts):
            for kk in range(nkpts):
                kd = kconserv[kk,kc,kl]
                Soovv = 2*eris.oovv[kk,kl,kc] - eris.oovv[kk,kl,kd].transpose(0,1,3,2)
                Fac[ka] += -einsum('klcd,klad->ac',Soovv,t2[kk,kl,ka])
            #if kk == ka
            kd = kconserv[ka,kc,kl]
            Soovv = 2*eris.oovv[ka,kl,kc] - eris.oovv[ka,kl,kd].transpose(0,1,3,2)
            Fac[ka] += -einsum('klcd,ka,ld->ac',Soovv,t1[ka],t1[kl])
    return Fac

def cc_Fov(t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    Fkc = np.empty((nkpts,nocc,nvir),dtype=t2.dtype)
    Fkc[:] = eris.fock[:,:nocc,nocc:].copy()
    for kk in range(nkpts):
        for kl in range(nkpts):
            Soovv = 2.*eris.oovv[kk,kl,kk] - eris.oovv[kk,kl,kl].transpose(0,1,3,2)
            Fkc[kk] += einsum('klcd,ld->kc',Soovv,t1[kl])
    return Fkc

### Eqs. (40)-(41) "lambda"

def Loo(t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:]
    Lki = cc_Foo(t1,t2,eris,kconserv)
    for ki in range(nkpts):
        Lki[ki] += einsum('kc,ic->ki',fov[ki],t1[ki])
        for kl in range(nkpts):
            Lki[ki] += 2*einsum('klic,lc->ki',eris.ooov[ki,kl,ki],t1[kl])
            Lki[ki] +=  -einsum('lkic,lc->ki',eris.ooov[kl,ki,ki],t1[kl])
    return Lki

def Lvv(t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:]
    Lac = cc_Fvv(t1,t2,eris,kconserv)
    for ka in range(nkpts):
        Lac[ka] += -einsum('kc,ka->ac',fov[ka],t1[ka])
        for kk in range(nkpts):
            Svovv = 2*eris.vovv[ka,kk,ka] - eris.vovv[ka,kk,kk].transpose(0,1,3,2)
            Lac[ka] += einsum('akcd,kd->ac',Svovv,t1[kk])
    return Lac

### Eqs. (42)-(45) "chi"

def cc_Woooo(t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape

    Wklij = np.array(eris.oooo, copy=True)
    for kk in range(nkpts):
        for kl in range(kk+1):
            for ki in range(nkpts):
                kj = kconserv[kk,ki,kl]
                Wklij[kk,kl,ki] += einsum('klic,jc->klij',eris.ooov[kk,kl,ki],t1[kj])
                Wklij[kk,kl,ki] += einsum('lkjc,ic->klij',eris.ooov[kl,kk,kj],t1[ki])

                # ==== Beginning of change ====
                #
                #for kc in range(nkpts):
                #    Wklij[kk,kl,ki] += einsum('klcd,ijcd->klij',eris.oovv[kk,kl,kc],t2[ki,kj,kc])
                #Wklij[kk,kl,ki] += einsum('klcd,ic,jd->klij',eris.oovv[kk,kl,ki],t1[ki],t1[kj])
                vvoo = eris.oovv[kk,kl].transpose(0,3,4,1,2).reshape(nkpts*nvir,nvir,nocc,nocc)
                t2t  = t2[ki,kj].copy().transpose(0,3,4,1,2)
                #for kc in range(nkpts):
                #    kd = kconserv[ki,kc,kj]
                #    if kc == ki and kj == kd:
                #        t2t[kc] += einsum('ic,jd->cdij',t1[ki],t1[kj])
                t2t[ki] += einsum('ic,jd->cdij',t1[ki],t1[kj])
                t2t = t2t.reshape(nkpts*nvir,nvir,nocc,nocc)
                Wklij[kk,kl,ki] += einsum('cdkl,cdij->klij',vvoo,t2t)
                # =====   End of change  = ====

        # Be careful about making this term only after all the others are created
        for kl in range(kk+1):
            for ki in range(nkpts):
                kj = kconserv[kk,ki,kl]
                Wklij[kl,kk,kj] = Wklij[kk,kl,ki].transpose(1,0,3,2)
    return Wklij

def cc_Wvvvv(t1,t2,eris,kconserv):
    # Incore:
    #nkpts, nocc, nvir = t1.shape
    #Wabcd = np.array(eris.vvvv, copy=True)
    #for ka in range(nkpts):
    #    for kb in range(ka+1):
    #        for kc in range(nkpts):
    #            Wabcd[ka,kb,kc] += einsum('akcd,kb->abcd',eris.vovv[ka,kb,kc],-t1[kb])
    #            Wabcd[ka,kb,kc] += einsum('kbcd,ka->abcd',eris.ovvv[ka,kb,kc],-t1[ka])
    #
    #    # Be careful about making this term only after all the others are created
    #    for kb in range(ka+1):
    #        for kc in range(nkpts):
    #            kd = kconserv[ka,kc,kb]
    #            Wabcd[kb,ka,kd] = Wabcd[ka,kb,kc].transpose(1,0,3,2)

    ## HDF5
    if t1.dtype == np.complex: ds_type = 'c16'
    else: ds_type = 'f8'
    _tmpfile1 = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    fimd = h5py.File(_tmpfile1.name)
    nkpts, nocc, nvir = t1.shape
    Wabcd = fimd.create_dataset('vvvv', (nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), ds_type) 
    for ka in range(nkpts):
        for kb in range(ka+1):
            for kc in range(nkpts):
                kd = kconserv[ka,kc,kb]
                # avoid transpose in loop
                Wabcd[ka,kb,kc] = eris.vvvv[ka,kb,kc]
                Wabcd[ka,kb,kc] += -einsum('akcd,kb->abcd',eris.vovv[ka,kb,kc],t1[kb])
                Wabcd[ka,kb,kc] += -einsum('bkdc,ka->abcd',eris.vovv[kb,ka,kd],t1[ka])

        # Be careful about making this term only after all the others are created
        for kb in range(ka+1):
            for kc in range(nkpts):
                kd = kconserv[ka,kc,kb]
                Wabcd[kb,ka,kd] = Wabcd[ka,kb,kc].transpose(1,0,3,2)

    return Wabcd

def cc_Wvoov(t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    Wakic = np.array(eris.voov, copy=True)
    for ka in range(nkpts):
        for kk in range(nkpts):
            for ki in range(nkpts):
                kc = kconserv[ka,ki,kk]
                Wakic[ka,kk,ki] -= einsum('lkic,la->akic',eris.ooov[ka,kk,ki],t1[ka])
                Wakic[ka,kk,ki] += einsum('akdc,id->akic',eris.vovv[ka,kk,ki],t1[ki])
                # ==== Beginning of change ====
                #
                #for kl in range(nkpts):
                #    # kl - kd + kk = kc
                #    # => kd = kl - kc + kk
                #    kd = kconserv[kl,kc,kk]
                #    Soovv = 2*eris.oovv[kl,kk,kd] - eris.oovv[kl,kk,kc].transpose(0,1,3,2)
                #    Wakic[ka,kk,ki] += 0.5*einsum('lkdc,ilad->akic',Soovv,t2[ki,kl,ka])
                #    Wakic[ka,kk,ki] -= 0.5*einsum('lkdc,ilda->akic',eris.oovv[kl,kk,kd],t2[ki,kl,kd])
                #Wakic[ka,kk,ki] -= einsum('lkdc,id,la->akic',eris.oovv[ka,kk,ki],t1[ki],t1[ka])

                #
                # Making various intermediates...
                #
                Soovv = np.empty((nkpts,nocc,nocc,nvir,nvir),dtype=t1.dtype)
                oovvf = np.empty((nkpts,nocc,nocc,nvir,nvir),dtype=t1.dtype)
                t2f_1  = t2[:,ki,ka].copy()   # This is a tau-like term
                for kl in range(nkpts):
                    # kl - kd + kk = kc
                    # => kd = kl - kc + kk
                    kd = kconserv[kl,kc,kk]
                    Soovv[kl] = 2*eris.oovv[kl,kk,kd] - eris.oovv[kl,kk,kc].transpose(0,1,3,2)
                    oovvf[kl] = eris.oovv[kl,kk,kd]
                    #if ki == kd and kl == ka:
                    #    t2f_1[kl] += 2*einsum('id,la->liad',t1[ki],t1[ka])
                kd = kconserv[ka,kc,kk]
                t2f_1[ka] += 2*einsum('id,la->liad',t1[kd],t1[ka])
                t2f_1  = t2f_1.reshape(nkpts*nocc,nocc,nvir,nvir)
                oovvf  = oovvf.reshape(nkpts*nocc,nocc,nvir,nvir)
                Soovvf = Soovv.reshape(nkpts*nocc,nocc,nvir,nvir)
                t2f    = t2[ki,:,ka].transpose(0,2,1,3,4).reshape(nkpts*nocc,nocc,nvir,nvir)

                Wakic[ka,kk,ki] += 0.5*einsum('lkdc,liad->akic',Soovvf,t2f)
                Wakic[ka,kk,ki] -= 0.5*einsum('lkdc,liad->akic',oovvf,t2f_1)
                # =====   End of change  = ====
    return Wakic

def cc_Wvovo(t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    Wakci = np.empty((nkpts,nkpts,nkpts,nvir,nocc,nvir,nocc),dtype=t1.dtype)
    for ka in range(nkpts):
        for kk in range(nkpts):
            for kc in range(nkpts):
                ki = kconserv[ka,kc,kk]
                Wakci[ka,kk,kc] = np.array(eris.ovov[kk,ka,ki]).transpose(1,0,3,2)
                Wakci[ka,kk,kc] -= einsum('klic,la->akci',eris.ooov[kk,ka,ki],t1[ka])
                Wakci[ka,kk,kc] += einsum('akcd,id->akci',eris.vovv[ka,kk,kc],t1[ki])
                # ==== Beginning of change ====
                #
                #for kl in range(nkpts):
                #    kd = kconserv[kl,kc,kk]
                #    Wakci[ka,kk,kc] -= 0.5*einsum('lkcd,ilda->akci',eris.oovv[kl,kk,kc],t2[ki,kl,kd])
                #Wakci[ka,kk,kc] -= einsum('lkcd,id,la->akci',eris.oovv[ka,kk,kc],t1[ki],t1[ka])
                oovvf = eris.oovv[:,kk,kc].reshape(nkpts*nocc,nocc,nvir,nvir)
                t2f   = t2[:,ki,ka].copy() #This is a tau like term
                #for kl in range(nkpts):
                #    kd = kconserv[kl,kc,kk]
                #    if ki == kd and kl == ka:
                #        t2f[kl] += 2*einsum('id,la->liad',t1[ki],t1[ka])
                kd = kconserv[ka,kc,kk]
                t2f[ka] += 2*einsum('id,la->liad',t1[kd],t1[ka])
                t2f = t2f.reshape(nkpts*nocc,nocc,nvir,nvir)

                Wakci[ka,kk,kc] -= 0.5*einsum('lkcd,liad->akci',oovvf,t2f)
                # =====   End of change  = ====

    return Wakci

def Wooov(t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape

    Wklid = np.array(eris.ooov)
    for kk in range(nkpts):
        for kl in range(nkpts):
            for ki in range(nkpts):
                Wklid[kk,kl,ki] += einsum('ic,klcd->klid',t1[ki],eris.oovv[kk,kl,ki])
    return Wklid

def Wvovv(t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape

    Walcd = np.array(eris.vovv)
    for ka in range(nkpts):
        for kl in range(nkpts):
            for kc in range(nkpts):
                Walcd[ka,kl,kc] += -einsum('ka,klcd->alcd',t1[ka],eris.oovv[ka,kl,kc])
    return Walcd

def W1ovvo(t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape

    Wkaci = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc),dtype=t1.dtype)
    for kk in range(nkpts):
        for ka in range(nkpts):
            for kc in range(nkpts):
                ki = kconserv[kk,kc,ka]
                # ovvo[kk,ka,kc,ki] => voov[ka,kk,ki,kc]
                Wkaci[kk,ka,kc] = np.array(eris.voov[ka,kk,ki]).transpose(1,0,3,2)
                for kl in range(nkpts):
                    kd = kconserv[ki,ka,kl]
                    St2 = 2.*t2[ki,kl,ka] - t2[kl,ki,ka].transpose(1,0,2,3)
                    Wkaci[kk,ka,kc] +=  einsum('klcd,ilad->kaci',eris.oovv[kk,kl,kc],St2)
                    Wkaci[kk,ka,kc] += -einsum('kldc,ilad->kaci',eris.oovv[kk,kl,kd],t2[ki,kl,ka])
    return Wkaci

def W2ovvo(t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape

    Wkaci = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc),dtype=t1.dtype)
    WWooov = Wooov(t1,t2,eris,kconserv)
    for kk in range(nkpts):
        for ka in range(nkpts):
            for kc in range(nkpts):
                ki = kconserv[kk,kc,ka]
                Wkaci[kk,ka,kc] =  einsum('la,lkic->kaci',-t1[ka],WWooov[ka,kk,ki])
                Wkaci[kk,ka,kc] += einsum('akdc,id->kaci',eris.vovv[ka,kk,ki],t1[ki])
    return Wkaci

def Wovvo(t1,t2,eris,kconserv):
    return W1ovvo(t1,t2,eris,kconserv) + W2ovvo(t1,t2,eris,kconserv)

def W1ovov(t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape

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

def W2ovov(t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape

    Wkbid = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir),dtype=t1.dtype)
    WWooov = Wooov(t1,t2,eris,kconserv)
    for kk in range(nkpts):
        for kb in range(nkpts):
            for ki in range(nkpts):
                kd = kconserv[kk,ki,kb]
                Wkbid[kk,kb,ki] = einsum('klid,lb->kbid',WWooov[kk,kb,ki],-t1[kb])
                Wkbid[kk,kb,ki] += einsum('bkdc,ic->kbid',eris.vovv[kb,kk,kd],t1[ki])
    return Wkbid

def Wovov(t1,t2,eris,kconserv):
    return W1ovov(t1,t2,eris,kconserv) + W2ovov(t1,t2,eris,kconserv)

def Woooo(t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape

    Wklij = np.array(eris.oooo, copy=True)
    for kk in range(nkpts):
        for kl in range(nkpts):
            for ki in range(nkpts):
                kj = kconserv[kk,ki,kl]
                for kc in range(nkpts):
                    #kd = kconserv[kk,kc,kl]
                    Wklij[kk,kl,ki] += einsum('klcd,ijcd->klij',eris.oovv[kk,kl,kc],t2[ki,kj,kc])
                Wklij[kk,kl,ki] += einsum('klcd,ic,jd->klij',eris.oovv[kk,kl,ki],t1[ki],t1[kj])
                Wklij[kk,kl,ki] += einsum('klid,jd->klij',eris.ooov[kk,kl,ki],t1[kj])
                Wklij[kk,kl,ki] += einsum('lkjc,ic->klij',eris.ooov[kl,kk,kj],t1[ki])
    return Wklij

def Wvvvv(t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape

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
                Wabcd[ka,kb,kc] += einsum('klcd,ka,lb->abcd',eris.oovv[ka,kb,kc],t1[ka],t1[kb])
                Wabcd[ka,kb,kc] += einsum('alcd,lb->abcd',eris.vovv[ka,kb,kc],-t1[kb])
                Wabcd[ka,kb,kc] += einsum('bkdc,ka->abcd',eris.vovv[kb,ka,kd],-t1[ka])
    return Wabcd

def Wvvvo(t1,t2,eris,kconserv,_Wvvvv=None):
    nkpts, nocc, nvir = t1.shape

    Wabcj = np.empty((nkpts,nkpts,nkpts,nvir,nvir,nvir,nocc),dtype=t1.dtype)
    WW1ovov = W1ovov(t1,t2,eris,kconserv)
    WW1ovvo = W1ovvo(t1,t2,eris,kconserv)
    FFov = cc_Fov(t1,t2,eris,kconserv)
    for ka in range(nkpts):
        for kb in range(nkpts):
            for kc in range(nkpts):
                kj = kconserv[ka,kc,kb]
                # vvvo[ka,kb,kc,kj] <= vovv[kc,kj,ka,kb].transpose(2,3,0,1).conj()
                Wabcj[ka,kb,kc] = np.array(eris.vovv[kc,kj,ka]).transpose(2,3,0,1).conj()
                # Wvovo[ka,kl,kc,kj] <= Wovov[kl,ka,kj,kc].transpose(1,0,3,2)
                Wabcj[ka,kb,kc] += einsum('alcj,lb->abcj',WW1ovov[kb,ka,kj].transpose(1,0,3,2),-t1[kb])
                Wabcj[ka,kb,kc] += einsum('kbcj,ka->abcj',WW1ovvo[ka,kb,kc],-t1[ka])

                for kl in range(nkpts):
                    # ka + kl - kc - kd = 0
                    # => kd = ka - kc + kl
                    kd = kconserv[ka,kc,kl]
                    St2 = 2.*t2[kl,kj,kd] - t2[kl,kj,kb].transpose(0,1,3,2)
                    Wabcj[ka,kb,kc] += einsum('alcd,ljdb->abcj',eris.vovv[ka,kl,kc], St2)
                    Wabcj[ka,kb,kc] += einsum('aldc,ljdb->abcj',eris.vovv[ka,kl,kd], -t2[kl,kj,kd])
                    # kb - kc + kl = kd
                    kd = kconserv[kb,kc,kl]
                    Wabcj[ka,kb,kc] += einsum('bldc,jlda->abcj',eris.vovv[kb,kl,kd], -t2[kj,kl,kd])

                    # kl + kk - kb - ka = 0
                    # => kk = kb + ka - kl
                    kk = kconserv[kb,kl,ka]
                    Wabcj[ka,kb,kc] += einsum('lkjc,lkba->abcj',eris.ooov[kl,kk,kj],t2[kl,kk,kb])
                Wabcj[ka,kb,kc] += einsum('lkjc,lb,ka->abcj',eris.ooov[kb,ka,kj],t1[kb],t1[ka])
                Wabcj[ka,kb,kc] += einsum('lc,ljab->abcj',-FFov[kc],t2[kc,kj,ka])
    # Check if t1=0 (HF+MBPT(2))
    # einsum will check, but don't make vvvv if you can avoid it!
    if np.any(t1):
        if _Wvvvv is None:
            _Wvvvv = Wvvvv(t1,t2,eris,kconserv)
        for ka in range(nkpts):
            for kb in range(nkpts):
                for kc in range(nkpts):
                    kj = kconserv[ka,kc,kb]
                    for a in range(nvir):
                        Wabcj[ka,kb,kc,a] += einsum('bcd,jd->bcj',_Wvvvv[ka,kb,kc,a],t1[kj])
    return Wabcj

def Wovoo(t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape

    WW1ovov = W1ovov(t1,t2,eris,kconserv)
    WWoooo = Woooo(t1,t2,eris,kconserv)
    WW1ovvo = W1ovvo(t1,t2,eris,kconserv)
    FFov = cc_Fov(t1,t2,eris,kconserv)

    Wkbij = np.empty((nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc),dtype=t1.dtype)
    for kk in range(nkpts):
        for kb in range(nkpts):
            for ki in range(nkpts):
                kj = kconserv[kk,ki,kb]
                Wkbij[kk,kb,ki] = np.array(eris.ooov[ki,kj,kk]).transpose(2,3,0,1).conj()
                Wkbij[kk,kb,ki] += einsum('kbid,jd->kbij',WW1ovov[kk,kb,ki], t1[kj])
                Wkbij[kk,kb,ki] += einsum('klij,lb->kbij',WWoooo[kk,kb,ki],-t1[kb])
                Wkbij[kk,kb,ki] += einsum('kbcj,ic->kbij',WW1ovvo[kk,kb,ki],t1[ki])

                for kd in range(nkpts):
                    # kk + kl - ki - kd = 0
                    # => kl = ki - kk + kd
                    kl = kconserv[ki,kk,kd]
                    St2 = 2.*t2[kl,kj,kd] - t2[kj,kl,kd].transpose(1,0,2,3)
                    Wkbij[kk,kb,ki] += einsum('klid,ljdb->kbij',  eris.ooov[kk,kl,ki], St2)
                    Wkbij[kk,kb,ki] += einsum('lkid,ljdb->kbij', -eris.ooov[kl,kk,ki],t2[kl,kj,kd])
                    kl = kconserv[kb,ki,kd]
                    Wkbij[kk,kb,ki] += einsum('lkjd,libd->kbij', -eris.ooov[kl,kk,kj],t2[kl,ki,kb])

                    # kb + kk - kd = kc
                    #kc = kconserv[kb,kd,kk]
                    Wkbij[kk,kb,ki] += einsum('bkdc,jidc->kbij',eris.vovv[kb,kk,kd],t2[kj,ki,kd])
                Wkbij[kk,kb,ki] += einsum('bkdc,jd,ic->kbij',eris.vovv[kb,kk,kj],t1[kj],t1[ki])
                Wkbij[kk,kb,ki] += einsum('kc,ijcb->kbij',FFov[kk],t2[ki,kj,kk])
    return Wkbij
