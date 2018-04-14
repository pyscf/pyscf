#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: James D. McClain
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

def cc_Woooo(t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape

    Wklij = _new(eris.oooo.shape, t1.dtype, out)
    for kk in range(nkpts):
        for kl in range(kk+1):
            for ki in range(nkpts):
                kj = kconserv[kk,ki,kl]
                oooo  = einsum('klic,jc->klij',eris.ooov[kk,kl,ki],t1[kj])
                oooo += einsum('lkjc,ic->klij',eris.ooov[kl,kk,kj],t1[ki])
                oooo += eris.oooo[kk,kl,ki]

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
                oooo += einsum('cdkl,cdij->klij',vvoo,t2t)
                Wklij[kk,kl,ki] = oooo
                # =====   End of change  = ====

        # Be careful about making this term only after all the others are created
        for kl in range(kk+1):
            for ki in range(nkpts):
                kj = kconserv[kk,ki,kl]
                Wklij[kl,kk,kj] = Wklij[kk,kl,ki].transpose(1,0,3,2)
    return Wklij

def cc_Wvvvv(t1, t2, eris, kconserv, out=None):
    Wabcd = _new(eris.vvvv.shape, t1.dtype, out)
    nkpts, nocc, nvir = t1.shape
    for ka in range(nkpts):
        for kb in range(ka+1):
            for kc in range(nkpts):
                kd = kconserv[ka,kc,kb]
                # avoid transpose in loop
                vvvv  = einsum('akcd,kb->abcd', eris.vovv[ka,kb,kc], -t1[kb])
                vvvv += einsum('bkdc,ka->abcd', eris.vovv[kb,ka,kd], -t1[ka])
                vvvv += eris.vvvv[ka,kb,kc]
                Wabcd[ka,kb,kc] = vvvv

        # Be careful: making this term only after all the others are created
        for kb in range(ka+1):
            for kc in range(nkpts):
                kd = kconserv[ka,kc,kb]
                Wabcd[kb,ka,kd] = Wabcd[ka,kb,kc].transpose(1,0,3,2)

    return Wabcd

def cc_Wvoov(t1, t2, eris, kconserv, out=None):
    Wakic = _new(eris.voov.shape, t1.dtype, out)
    nkpts, nocc, nvir = t1.shape
    for ka in range(nkpts):
        for kk in range(nkpts):
            for ki in range(nkpts):
                kc = kconserv[ka,ki,kk]
                voov  = einsum('akdc,id->akic',eris.vovv[ka,kk,ki],t1[ki])
                voov -= einsum('lkic,la->akic',eris.ooov[ka,kk,ki],t1[ka])
                voov += eris.voov[ka,kk,ki]
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

                voov += 0.5*einsum('lkdc,liad->akic',Soovvf,t2f)
                voov -= 0.5*einsum('lkdc,liad->akic',oovvf,t2f_1)
                Wakic[ka,kk,ki] = voov
                # =====   End of change  = ====
    return Wakic

def cc_Wvovo(t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape
    Wakci = _new((nkpts,nkpts,nkpts,nvir,nocc,nvir,nocc), t1.dtype, out)

    for ka in range(nkpts):
        for kk in range(nkpts):
            for kc in range(nkpts):
                ki = kconserv[ka,kc,kk]
                vovo  = einsum('akcd,id->akci',eris.vovv[ka,kk,kc],t1[ki])
                vovo -= einsum('klic,la->akci',eris.ooov[kk,ka,ki],t1[ka])
                vovo += np.asarray(eris.ovov[kk,ka,ki]).transpose(1,0,3,2)
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

                vovo -= 0.5*einsum('lkcd,liad->akci',oovvf,t2f)
                Wakci[ka,kk,kc] = vovo
                # =====   End of change  = ====
    return Wakci

def Wooov(t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape
    Wklid = _new(eris.ooov.shape, t1.dtype, out)
    for kk in range(nkpts):
        for kl in range(nkpts):
            for ki in range(nkpts):
                ooov = einsum('ic,klcd->klid',t1[ki],eris.oovv[kk,kl,ki])
                ooov += eris.ooov[kk,kl,ki]
                Wklid[kk,kl,ki] = ooov
    return Wklid

def Wvovv(t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape
    Walcd = _new(eris.vovv.shape, t1.dtype, out)
    for ka in range(nkpts):
        for kl in range(nkpts):
            for kc in range(nkpts):
                vovv = einsum('ka,klcd->alcd', -t1[ka], eris.oovv[ka,kl,kc])
                vovv += eris.vovv[ka,kl,kc]
                Walcd[ka,kl,kc] = vovv
    return Walcd

def W1ovvo(t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape
    Wkaci = _new((nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), t1.dtype, out)
    for kk in range(nkpts):
        for ka in range(nkpts):
            for kc in range(nkpts):
                ki = kconserv[kk,kc,ka]
                # ovvo[kk,ka,kc,ki] => voov[ka,kk,ki,kc]
                ovvo = np.asarray(eris.voov[ka,kk,ki]).transpose(1,0,3,2).copy()
                for kl in range(nkpts):
                    kd = kconserv[ki,ka,kl]
                    St2 = 2.*t2[ki,kl,ka] - t2[kl,ki,ka].transpose(1,0,2,3)
                    ovvo +=  einsum('klcd,ilad->kaci',eris.oovv[kk,kl,kc],St2)
                    ovvo += -einsum('kldc,ilad->kaci',eris.oovv[kk,kl,kd],t2[ki,kl,ka])
                Wkaci[kk,ka,kc] = ovvo
    return Wkaci

def W2ovvo(t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape
    Wkaci = _new((nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), t1.dtype, out)
    WWooov = Wooov(t1,t2,eris,kconserv)
    for kk in range(nkpts):
        for ka in range(nkpts):
            for kc in range(nkpts):
                ki = kconserv[kk,kc,ka]
                ovvo =  einsum('la,lkic->kaci',-t1[ka],WWooov[ka,kk,ki])
                ovvo += einsum('akdc,id->kaci',eris.vovv[ka,kk,ki],t1[ki])
                Wkaci[kk,ka,kc] = ovvo
    return Wkaci

def Wovvo(t1, t2, eris, kconserv, out=None):
    Wovvo = W1ovvo(t1, t2, eris, kconserv, out)
    for k, w2 in enumerate(W2ovvo(t1, t2, eris, kconserv)):
        Wovvo[k] = Wovvo[k] + w2
    return Wovvo

def W1ovov(t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape
    Wkbid = _new(eris.ovov.shape, t1.dtype, out)
    for kk in range(nkpts):
        for kb in range(nkpts):
            for ki in range(nkpts):
                kd = kconserv[kk,ki,kb]
                #   kk + kl - kc - kd = 0
                # => kc = kk - kd + kl
                ovov = eris.ovov[kk,kb,ki].copy()
                for kl in range(nkpts):
                    kc = kconserv[kk,kd,kl]
                    ovov -= einsum('klcd,ilcb->kbid',eris.oovv[kk,kl,kc],t2[ki,kl,kc])
                Wkbid[kk,kb,ki] = ovov
    return Wkbid

def W2ovov(t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape
    Wkbid = _new((nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), t1.dtype, out)
    WWooov = Wooov(t1,t2,eris,kconserv)
    for kk in range(nkpts):
        for kb in range(nkpts):
            for ki in range(nkpts):
                kd = kconserv[kk,ki,kb]
                ovov = einsum('klid,lb->kbid',WWooov[kk,kb,ki],-t1[kb])
                ovov += einsum('bkdc,ic->kbid',eris.vovv[kb,kk,kd],t1[ki])
                Wkbid[kk,kb,ki] = ovov
    return Wkbid

def Wovov(t1, t2, eris, kconserv, out=None):
    Wovov = W1ovov(t1, t2, eris, kconserv, out)
    for k, w2 in enumerate(W2ovov(t1, t2, eris, kconserv)):
        Wovov[k] = Wovov[k] + w2
    return Wovov

def Woooo(t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape
    Wklij = _new(eris.oooo.shape, t1.dtype, out)
    for kk in range(nkpts):
        for kl in range(nkpts):
            for ki in range(nkpts):
                kj = kconserv[kk,ki,kl]
                oooo  = einsum('klcd,ic,jd->klij',eris.oovv[kk,kl,ki],t1[ki],t1[kj])
                oooo += einsum('klid,jd->klij',eris.ooov[kk,kl,ki],t1[kj])
                oooo += einsum('lkjc,ic->klij',eris.ooov[kl,kk,kj],t1[ki])
                oooo += eris.oooo[kk,kl,ki]
                for kc in range(nkpts):
                    #kd = kconserv[kk,kc,kl]
                    oooo += einsum('klcd,ijcd->klij',eris.oovv[kk,kl,kc],t2[ki,kj,kc])
                Wklij[kk,kl,ki] = oooo
    return Wklij

def Wvvvv(t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape
    Wabcd = _new(eris.vvvv.shape, t1.dtype, out)

    for ka in range(nkpts):
        for kb in range(nkpts):
            for kc in range(nkpts):
                kd = kconserv[ka,kc,kb]
                vvvv  = einsum('klcd,ka,lb->abcd',eris.oovv[ka,kb,kc],t1[ka],t1[kb])
                vvvv += einsum('alcd,lb->abcd',eris.vovv[ka,kb,kc],-t1[kb])
                vvvv += einsum('bkdc,ka->abcd',eris.vovv[kb,ka,kd],-t1[ka])
                vvvv += eris.vvvv[ka,kb,kc]
                for kk in range(nkpts):
                    # kk + kl - kc - kd = 0
                    # => kl = kc - kk + kd
                    kl = kconserv[kc,kk,kd]
                    vvvv += einsum('klcd,klab->abcd',eris.oovv[kk,kl,kc],t2[kk,kl,ka])
                Wabcd[ka,kb,kc] = vvvv
    return Wabcd

def Wvvvo(t1, t2, eris, kconserv, _Wvvvv=None, out=None):
    nkpts, nocc, nvir = t1.shape
    Wabcj = _new((nkpts,nkpts,nkpts,nvir,nvir,nvir,nocc), t1.dtype, out)
    WW1ovov = W1ovov(t1,t2,eris,kconserv)
    WW1ovvo = W1ovvo(t1,t2,eris,kconserv)
    FFov = cc_Fov(t1,t2,eris,kconserv)
    for ka in range(nkpts):
        for kb in range(nkpts):
            for kc in range(nkpts):
                kj = kconserv[ka,kc,kb]
                # Wvovo[ka,kl,kc,kj] <= Wovov[kl,ka,kj,kc].transpose(1,0,3,2)
                vvvo  = einsum('alcj,lb->abcj',WW1ovov[kb,ka,kj].transpose(1,0,3,2),-t1[kb])
                vvvo += einsum('kbcj,ka->abcj',WW1ovvo[ka,kb,kc],-t1[ka])
                # vvvo[ka,kb,kc,kj] <= vovv[kc,kj,ka,kb].transpose(2,3,0,1).conj()
                vvvo += np.asarray(eris.vovv[kc,kj,ka]).transpose(2,3,0,1).conj()

                for kl in range(nkpts):
                    # ka + kl - kc - kd = 0
                    # => kd = ka - kc + kl
                    kd = kconserv[ka,kc,kl]
                    St2 = 2.*t2[kl,kj,kd] - t2[kl,kj,kb].transpose(0,1,3,2)
                    vvvo += einsum('alcd,ljdb->abcj',eris.vovv[ka,kl,kc], St2)
                    vvvo += einsum('aldc,ljdb->abcj',eris.vovv[ka,kl,kd], -t2[kl,kj,kd])
                    # kb - kc + kl = kd
                    kd = kconserv[kb,kc,kl]
                    vvvo += einsum('bldc,jlda->abcj',eris.vovv[kb,kl,kd], -t2[kj,kl,kd])

                    # kl + kk - kb - ka = 0
                    # => kk = kb + ka - kl
                    kk = kconserv[kb,kl,ka]
                    vvvo += einsum('lkjc,lkba->abcj',eris.ooov[kl,kk,kj],t2[kl,kk,kb])
                vvvo += einsum('lkjc,lb,ka->abcj',eris.ooov[kb,ka,kj],t1[kb],t1[ka])
                vvvo += einsum('lc,ljab->abcj',-FFov[kc],t2[kc,kj,ka])
                Wabcj[ka,kb,kc] = vvvo
    # Check if t1=0 (HF+MBPT(2))
    # einsum will check, but don't make vvvv if you can avoid it!
    if np.any(t1 != 0):
        if _Wvvvv is None:
            _Wvvvv = Wvvvv(t1,t2,eris,kconserv)
        for ka in range(nkpts):
            for kb in range(nkpts):
                for kc in range(nkpts):
                    kj = kconserv[ka,kc,kb]
                    Wabcj[ka,kb,kc] = (Wabcj[ka,kb,kc] +
                                       einsum('abcd,jd->abcj',_Wvvvv[ka,kb,kc],t1[kj]))
    return Wabcj

def Wovoo(t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape

    WW1ovov = W1ovov(t1,t2,eris,kconserv)
    WWoooo = Woooo(t1,t2,eris,kconserv)
    WW1ovvo = W1ovvo(t1,t2,eris,kconserv)
    FFov = cc_Fov(t1,t2,eris,kconserv)

    Wkbij = _new((nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc), t1.dtype, out)
    for kk in range(nkpts):
        for kb in range(nkpts):
            for ki in range(nkpts):
                kj = kconserv[kk,ki,kb]
                ovoo  = einsum('kbid,jd->kbij',WW1ovov[kk,kb,ki], t1[kj])
                ovoo += einsum('klij,lb->kbij',WWoooo[kk,kb,ki],-t1[kb])
                ovoo += einsum('kbcj,ic->kbij',WW1ovvo[kk,kb,ki],t1[ki])
                ovoo += np.array(eris.ooov[ki,kj,kk]).transpose(2,3,0,1).conj()

                for kd in range(nkpts):
                    # kk + kl - ki - kd = 0
                    # => kl = ki - kk + kd
                    kl = kconserv[ki,kk,kd]
                    St2 = 2.*t2[kl,kj,kd] - t2[kj,kl,kd].transpose(1,0,2,3)
                    ovoo += einsum('klid,ljdb->kbij',  eris.ooov[kk,kl,ki], St2)
                    ovoo += einsum('lkid,ljdb->kbij', -eris.ooov[kl,kk,ki],t2[kl,kj,kd])
                    kl = kconserv[kb,ki,kd]
                    ovoo += einsum('lkjd,libd->kbij', -eris.ooov[kl,kk,kj],t2[kl,ki,kb])

                    # kb + kk - kd = kc
                    #kc = kconserv[kb,kd,kk]
                    ovoo += einsum('bkdc,jidc->kbij',eris.vovv[kb,kk,kd],t2[kj,ki,kd])
                ovoo += einsum('bkdc,jd,ic->kbij',eris.vovv[kb,kk,kj],t1[kj],t1[ki])
                ovoo += einsum('kc,ijcb->kbij',FFov[kk],t2[ki,kj,kk])
                Wkbij[kk,kb,ki] = ovoo
    return Wkbij

def _new(shape, dtype, out):
    if out is None: # Incore:
        out = np.empty(shape, dtype=dtype)
    else:
        assert(out.shape == shape)
        assert(out.dtype == dtype)
    return out
