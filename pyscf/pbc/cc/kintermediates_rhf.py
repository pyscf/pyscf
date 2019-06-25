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

import numpy as np
import time
from itertools import product
from pyscf import lib
from pyscf.lib import logger
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)

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
            voov_i  = einsum('xakdc,xid->xakic',eris.vovv[ka,kk,:],t1[:])
            voov_i -= einsum('xlkic,la->xakic',eris.ooov[ka,kk,:],t1[ka])
            voov_i += eris.voov[ka,kk,:]
            for ki in range(nkpts):
                kc = kconserv[ka,ki,kk]

                #for kl in range(nkpts):
                #    # kl - kd + kk = kc
                #    # => kd = kl - kc + kk
                #    kd = kconserv[kl,kc,kk]
                #    Soovv = 2*eris.oovv[kl,kk,kd] - eris.oovv[kl,kk,kc].transpose(0,1,3,2)
                #    Wakic[ka,kk,ki] += 0.5*einsum('lkdc,ilad->akic',Soovv,t2[ki,kl,ka])
                #    Wakic[ka,kk,ki] -= 0.5*einsum('lkdc,ilda->akic',eris.oovv[kl,kk,kd],t2[ki,kl,kd])
                #Wakic[ka,kk,ki] -= einsum('lkdc,id,la->akic',eris.oovv[ka,kk,ki],t1[ki],t1[ka])

                kd = kconserv[ka,kc,kk]
                tau = t2[:,ki,ka].copy()
                tau[ka] += 2*einsum('id,la->liad',t1[kd],t1[ka])
                oovv_tmp = np.array(eris.oovv[kk,:,kc])
                voov_i[ki] -= 0.5*einsum('xklcd,xliad->akic',oovv_tmp,tau)

                Soovv_tmp = 2*oovv_tmp - eris.oovv[:,kk,kc].transpose(0,2,1,3,4)
                voov_i[ki] += 0.5*einsum('xklcd,xilad->akic',Soovv_tmp,t2[ki,:,ka])

            Wakic[ka,kk,:] = voov_i[:]
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
    Wabcd = _new((nkpts,nkpts,nkpts,nvir,nvir,nvir,nvir), t2.dtype, out)
    for ka in range(nkpts):
        for kb in range(nkpts):
            for kc in range(nkpts):
                Wabcd[ka,kb,kc] = get_Wvvvv(t1, t2, eris, kconserv, ka, kb, kc)
    return Wabcd

def get_Wvvvv(t1, t2, eris, kconserv, ka, kb, kc):
    kd = kconserv[ka, kc, kb]
    nkpts, nocc, nvir = t1.shape
    if getattr(eris, 'Lpv', None) is not None:
        # Using GDF to generate Wvvvv on the fly
        Lpv = eris.Lpv
        Lac = (Lpv[ka,kc][:,nocc:] -
               einsum('Lkc,ka->Lac', Lpv[ka,kc][:,:nocc], t1[ka]))
        Lbd = (Lpv[kb,kd][:,nocc:] -
               einsum('Lkd,kb->Lbd', Lpv[kb,kd][:,:nocc], t1[kb]))
        vvvv = einsum('Lac,Lbd->abcd', Lac, Lbd)
        vvvv *= (1. / nkpts)
    else:
        vvvv  = einsum('klcd,ka,lb->abcd',eris.oovv[ka,kb,kc],t1[ka],t1[kb])
        vvvv += einsum('alcd,lb->abcd',eris.vovv[ka,kb,kc],-t1[kb])
        vvvv += einsum('bkdc,ka->abcd',eris.vovv[kb,ka,kd],-t1[ka])
        vvvv += eris.vvvv[ka,kb,kc]

    for kk in range(nkpts):
        kl = kconserv[kc,kk,kd]
        vvvv += einsum('klcd,klab->abcd', eris.oovv[kk,kl,kc], t2[kk,kl,ka])
    return vvvv

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
        for ka in range(nkpts):
            for kb in range(nkpts):
                for kc in range(nkpts):
                    kj = kconserv[ka,kc,kb]
                    if _Wvvvv is None:
                        Wvvvv = get_Wvvvv(t1, t2, eris, kconserv, ka, kb, kc)
                    else:
                        Wvvvv = _Wvvvv[ka, kb, kc]
                    Wabcj[ka,kb,kc] = (Wabcj[ka,kb,kc] +
                                       einsum('abcd,jd->abcj', Wvvvv, t1[kj]))
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

def get_t3p2_imds_slow(cc, t1, t2, eris=None, t3p2_ip_out=None, t3p2_ea_out=None):
    """For a description of arguments, see `get_t3p2_imds_slow` in
    the corresponding `kintermediates.py`.
    """
    from pyscf.pbc.cc.kccsd_t_rhf import _get_epqr
    if eris is None:
        eris = cc.ao2mo()
    fock = eris.fock
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.khelper.kconserv
    dtype = np.result_type(t1, t2)

    fov = fock[:, :nocc, nocc:]
    foo = [fock[ikpt, :nocc, :nocc].diagonal() for ikpt in range(nkpts)]
    fvv = [fock[ikpt, nocc:, nocc:].diagonal() for ikpt in range(nkpts)]
    mo_energy_occ = np.array([eris.mo_energy[ki][:nocc] for ki in range(nkpts)])
    mo_energy_vir = np.array([eris.mo_energy[ki][nocc:] for ki in range(nkpts)])

    mo_e_o = mo_energy_occ
    mo_e_v = mo_energy_vir

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(cc, kind="split")

    ccsd_energy = cc.energy(t1, t2, eris)

    if t3p2_ip_out is None:
        t3p2_ip_out = np.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc),dtype=dtype)
    Wmcik = t3p2_ip_out

    if t3p2_ea_out is None:
        t3p2_ea_out = np.zeros((nkpts,nkpts,nkpts,nvir,nvir,nvir,nocc),dtype=dtype)
    Wacek = t3p2_ea_out

    from itertools import product
    tmp_t3 = np.empty((nkpts, nkpts, nkpts, nkpts, nkpts, nocc, nocc, nocc, nvir, nvir, nvir),
                      dtype = t2.dtype)

    def get_w(ki, kj, kk, ka, kb, kc):
        kf = kconserv[ka,ki,kb]
        ret = lib.einsum('fiba,kjcf->ijkabc', eris.vovv[kf, ki, kb].conj(), t2[kk, kj, kc])
        km = kconserv[kc,kk,kb]
        ret -= lib.einsum('jima,mkbc->ijkabc', eris.ooov[kj, ki, km].conj(), t2[km, kk, kb])
        return ret

    for ki, kj, kk, ka, kb in product(range(nkpts), repeat=5):
        kc = kpts_helper.get_kconserv3(cc._scf.cell, cc.kpts,
                                       [ki, kj, kk, ka, kb])
        tmp_t3[ki, kj, kk, ka, kb] = get_w(ki, kj, kk, ka, kb, kc)
        tmp_t3[ki, kj, kk, ka, kb] += get_w(ki, kk, kj, ka, kc, kb).transpose(0, 2, 1, 3, 5, 4)
        tmp_t3[ki, kj, kk, ka, kb] += get_w(kj, ki, kk, kb, ka, kc).transpose(1, 0, 2, 4, 3, 5)
        tmp_t3[ki, kj, kk, ka, kb] += get_w(kj, kk, ki, kb, kc, ka).transpose(2, 0, 1, 5, 3, 4)
        tmp_t3[ki, kj, kk, ka, kb] += get_w(kk, ki, kj, kc, ka, kb).transpose(1, 2, 0, 4, 5, 3)
        tmp_t3[ki, kj, kk, ka, kb] += get_w(kk, kj, ki, kc, kb, ka).transpose(2, 1, 0, 5, 4, 3)

        eijk = _get_epqr([0,nocc,ki,mo_e_o,nonzero_opadding],
                         [0,nocc,kj,mo_e_o,nonzero_opadding],
                         [0,nocc,kk,mo_e_o,nonzero_opadding])
        eabc = _get_epqr([0,nvir,ka,mo_e_v,nonzero_vpadding],
                         [0,nvir,kb,mo_e_v,nonzero_vpadding],
                         [0,nvir,kc,mo_e_v,nonzero_vpadding],
                         fac=[-1.,-1.,-1.])
        eijkabc = eijk[:, :, :, None, None, None] + eabc[None, None, None, :, :, :]
        tmp_t3[ki, kj, kk, ka, kb] /= eijkabc

    pt1 = np.zeros((nkpts, nocc, nvir), dtype=t2.dtype)
    for ki in range(nkpts):
        for km, kn, ke in product(range(nkpts), repeat=3):
            kf = kconserv[km, ke, kn]
            Soovv = 2. * eris.oovv[km, kn, ke] - eris.oovv[km, kn, kf].transpose(0, 1, 3, 2)
            St3 = (tmp_t3[ki, km, kn, ki, ke] -
                   tmp_t3[ki, km, kn, ke, ki].transpose(0, 1, 2, 4, 3, 5))
            pt1[ki] += lib.einsum('mnef,imnaef->ia', Soovv, St3)

    pt2 = np.zeros((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=t2.dtype)
    for ki, kj, ka in product(range(nkpts), repeat=3):
        kb = kconserv[ki, ka, kj]
        for km in range(nkpts):
            for kn in range(nkpts):
                # (ia,jb) -> (ia,jb)
                ke = kconserv[km, kj, kn]
                pt2[ki, kj, ka] += - 2. * lib.einsum('imnabe,mnje->ijab',
                                                     tmp_t3[ki, km, kn, ka, kb],
                                                     eris.ooov[km, kn, kj])
                pt2[ki, kj, ka] += lib.einsum('imnabe,nmje->ijab',
                                              tmp_t3[ki, km, kn, ka, kb],
                                              eris.ooov[kn, km, kj])
                pt2[ki, kj, ka] += lib.einsum('inmeab,mnje->ijab',
                                              tmp_t3[ki, kn, km, ke, ka],
                                              eris.ooov[km, kn, kj])

                # (ia,jb) -> (jb,ia)
                ke = kconserv[km, ki, kn]
                pt2[ki, kj, ka] += - 2. * lib.einsum('jmnbae,mnie->ijab',
                                                     tmp_t3[kj, km, kn, kb, ka],
                                                     eris.ooov[km, kn, ki])
                pt2[ki, kj, ka] += lib.einsum('jmnbae,nmie->ijab',
                                              tmp_t3[kj, km, kn, kb, ka],
                                              eris.ooov[kn, km, ki])
                pt2[ki, kj, ka] += lib.einsum('jnmeba,mnie->ijab',
                                              tmp_t3[kj, kn, km, ke, kb],
                                              eris.ooov[km, kn, ki])

            # (ia,jb) -> (ia,jb)
            pt2[ki, kj, ka] += lib.einsum('ijmabe,me->ijab',
                                          tmp_t3[ki, kj, km, ka, kb],
                                          fov[km])
            pt2[ki, kj, ka] -= lib.einsum('ijmaeb,me->ijab',
                                          tmp_t3[ki, kj, km, ka, km],
                                          fov[km])

            # (ia,jb) -> (jb,ia)
            pt2[ki, kj, ka] += lib.einsum('jimbae,me->ijab',
                                          tmp_t3[kj, ki, km, kb, ka],
                                          fov[km])
            pt2[ki, kj, ka] -= lib.einsum('jimbea,me->ijab',
                                          tmp_t3[kj, ki, km, kb, km],
                                          fov[km])

            for ke in range(nkpts):
                # (ia,jb) -> (ia,jb)
                kf = kconserv[km, ke, kb]
                pt2[ki, kj, ka] += 2. * lib.einsum('ijmaef,bmef->ijab',
                                                   tmp_t3[ki, kj, km, ka, ke],
                                                   eris.vovv[kb, km, ke])
                pt2[ki, kj, ka] -= lib.einsum('ijmaef,bmfe->ijab',
                                              tmp_t3[ki, kj, km, ka, ke],
                                              eris.vovv[kb, km, kf])
                pt2[ki, kj, ka] -= lib.einsum('imjfae,bmef->ijab',
                                              tmp_t3[ki, km, kj, kf, ka],
                                              eris.vovv[kb, km, ke])

                # (ia,jb) -> (jb,ia)
                kf = kconserv[km, ke, ka]
                pt2[ki, kj, ka] += 2. * lib.einsum('jimbef,amef->ijab',
                                                   tmp_t3[kj, ki, km, kb, ke],
                                                   eris.vovv[ka, km, ke])
                pt2[ki, kj, ka] -= lib.einsum('jimbef,amfe->ijab',
                                              tmp_t3[kj, ki, km, kb, ke],
                                              eris.vovv[ka, km, kf])
                pt2[ki, kj, ka] -= lib.einsum('jmifbe,amef->ijab',
                                              tmp_t3[kj, km, ki, kf, kb],
                                              eris.vovv[ka, km, ke])

    for ki in range(nkpts):
        ka = ki
        eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
        n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
        eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]
        pt1[ki] /= eia

    for ki, ka in product(range(nkpts), repeat=2):
        eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
        n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
        eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]
        for kj in range(nkpts):
            kb = kconserv[ki, ka, kj]
            ejb = LARGE_DENOM * np.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
            n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
            ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]
            eijab = eia[:, None, :, None] + ejb[:, None, :]
            pt2[ki, kj, ka] /= eijab

    pt1 += t1
    pt2 += t2

    for ki, kj, kk, ka, kb in product(range(nkpts), repeat=5):
        kc = kpts_helper.get_kconserv3(cc._scf.cell, cc.kpts,
                                       [ki, kj, kk, ka, kb])
        km = kconserv[kc, ki, ka]

        _oovv = eris.oovv[km, ki, kc]
        Wmcik[km, kb, kk] += 2. * lib.einsum('ijkabc,mica->mbkj', tmp_t3[ki, kj, kk, ka, kb], _oovv)
        Wmcik[km, kb, kk] -=      lib.einsum('jikabc,mica->mbkj', tmp_t3[kj, ki, kk, ka, kb], _oovv)
        Wmcik[km, kb, kk] -=      lib.einsum('kjiabc,mica->mbkj', tmp_t3[kk, kj, ki, ka, kb], _oovv)

    for ki, kj, kk, ka, kb in product(range(nkpts), repeat=5):
        kc = kpts_helper.get_kconserv3(cc._scf.cell, cc.kpts,
                                       [ki, kj, kk, ka, kb])
        ke = kconserv[ki, ka, kk]

        _oovv = eris.oovv[ki, kk, ka]
        Wacek[kc, kb, ke] -= 2. * lib.einsum('ijkabc,ikae->cbej', tmp_t3[ki, kj, kk, ka, kb], _oovv)
        Wacek[kc, kb, ke] +=      lib.einsum('jikabc,ikae->cbej', tmp_t3[kj, ki, kk, ka, kb], _oovv)
        Wacek[kc, kb, ke] +=      lib.einsum('kjiabc,ikae->cbej', tmp_t3[kk, kj, ki, ka, kb], _oovv)

    delta_ccsd_energy = cc.energy(pt1, pt2, eris) - ccsd_energy
    lib.logger.info(cc, 'CCSD energy T3[2] correction : %16.12e', delta_ccsd_energy)

    return delta_ccsd_energy, pt1, pt2, Wmcik, Wacek


def _add_pt2(pt2, nkpts, kconserv, kpt_indices, orb_indices, val):
    '''Adds term P(ia|jb)[tmp] to pt2.

        P(ia|jb)(tmp[i,j,a,b]) = tmp[i,j,a,b] + tmp[j,i,b,a]

    or equivalently for each i,j,a,b, pt2 is defined as

        pt2[i,j,a,b] += tmp[i,j,a,b]
        pt2[j,i,b,a] += tmp[i,j,a,b].transpose(1,0,3,2)

    If pt2 is lower-triangular, only adds the RHS term that contributes
    to the lower-triangular pt2.

    Args:
        pt2 (ndarray or HDF5 dataset):
            Full or lower triangular T2 array to which one is adding to.
        kpt_indices (array-like):
            K-point indices ki, kj, ka.
        orb_indices (array-like):
            Array-like of four tuples describing the range for i,j,a,b.  An
            element of None will convert to slice(None,None).
        val (ndarray):
            Values to be added to pt2.
    '''
    assert(len(orb_indices) == 4)
    ki, kj, ka = kpt_indices
    kb = kconserv[ki,ka,kj]
    idxi, idxj, idxa, idxb = [slice(None, None)
                              if x is None else slice(x[0],x[1])
                              for x in orb_indices]
    if len(pt2.shape) == 7 and pt2.shape[:2] == (nkpts, nkpts):
        pt2[ki,kj,ka,idxi,idxj,idxa,idxb] += val
        pt2[kj,ki,kb,idxj,idxi,idxb,idxa] += val.transpose(1,0,3,2)
    elif len(pt2.shape) == 6 and pt2.shape[:2] == (nkpts*(nkpts+1)//2, nkpts):
        if ki <= kj:  # Add tmp[i,j,a,b] to pt2[i,j,a,b]
            idx = (kj*(kj+1))//2 + ki
            pt2[idx,ka,idxi,idxj,idxa,idxb] += val
            if ki == kj:
                pt2[idx,kb,idxj,idxi,idxb,idxa] += val.transpose(1,0,3,2)
        else:  # pt2[i,a,j,b] += tmp[j,i,a,b].transpose(1,0,3,2)
            idx = (ki*(ki+1))//2 + kj
            pt2[idx,kb,idxj,idxi,idxb,idxa] += val.transpose(1,0,3,2)
    else:
        raise ValueError('No known conversion for t2 shape %s' % t2.shape)


def get_t3p2_imds(mycc, t1, t2, eris=None, t3p2_ip_out=None, t3p2_ea_out=None):
    """For a description of arguments, see `get_t3p2_imds_slow` in
    the corresponding `kintermediates.py`.
    """
    from pyscf.pbc.cc.kccsd_t_rhf import _get_epqr
    cpu1 = cpu0 = (time.clock(), time.time())
    if eris is None:
        eris = mycc.ao2mo()
    fock = eris.fock
    nkpts, nocc, nvir = t1.shape
    cell = mycc._scf.cell
    kpts = mycc.kpts
    kconserv = mycc.khelper.kconserv
    dtype = np.result_type(t1, t2)

    fov = fock[:, :nocc, nocc:]
    foo = np.asarray([fock[ikpt, :nocc, :nocc].diagonal() for ikpt in range(nkpts)])
    fvv = np.asarray([fock[ikpt, nocc:, nocc:].diagonal() for ikpt in range(nkpts)])
    mo_energy_occ = np.array([eris.mo_energy[ki][:nocc] for ki in range(nkpts)])
    mo_energy_vir = np.array([eris.mo_energy[ki][nocc:] for ki in range(nkpts)])

    mo_e_o = mo_energy_occ
    mo_e_v = mo_energy_vir

    ccsd_energy = mycc.energy(t1, t2, eris)

    if t3p2_ip_out is None:
        t3p2_ip_out = np.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc),dtype=dtype)
    Wmcik = t3p2_ip_out

    if t3p2_ea_out is None:
        t3p2_ea_out = np.zeros((nkpts,nkpts,nkpts,nvir,nvir,nvir,nocc),dtype=dtype)
    Wacek = t3p2_ea_out

    # Create necessary temporary eris for fast read
    from pyscf.pbc.cc.kccsd_t_rhf import create_t3_eris, get_data_slices
    feri_tmp, t2T, eris_vvop, eris_vooo_C = create_t3_eris(mycc, kconserv, [eris.vovv, eris.oovv, eris.ooov, t2])
    t1T = np.array([x.T for x in t1], dtype=np.complex, order='C')
    fvo = np.array([x.T for x in fov], dtype=np.complex, order='C')
    cpu1 = logger.timer_debug1(mycc, 'CCSD(T) tmp eri creation', *cpu1)

    def get_w(ki, kj, kk, ka, kb, kc, a0, a1, b0, b1, c0, c1):
        '''Wijkabc intermediate as described in Scuseria paper before Pijkabc acts

        Function copied for `kccsd_t_rhf.py`'''
        km = kconserv[kc, kk, kb]
        kf = kconserv[kk, kc, kj]
        out = einsum('cfjk,abif->abcijk', t2T[kc,kf,kj,c0:c1,:,:,:], eris_vvop[ka,kb,ki,a0:a1,b0:b1,:,nocc:])
        out = out - einsum('cbmk,aijm->abcijk', t2T[kc,kb,km,c0:c1,b0:b1,:,:], eris_vooo_C[ka,ki,kj,a0:a1,:,:,:])
        return out

    def get_permuted_w(ki, kj, kk, ka, kb, kc, orb_indices):
        '''Pijkabc operating on Wijkabc intermediate as described in Scuseria paper

        Function copied for `kccsd_t_rhf.py`'''
        a0, a1, b0, b1, c0, c1 = orb_indices
        out = get_w(ki, kj, kk, ka, kb, kc, a0, a1, b0, b1, c0, c1)
        out = out + get_w(kj, kk, ki, kb, kc, ka, b0, b1, c0, c1, a0, a1).transpose(2,0,1,5,3,4)
        out = out + get_w(kk, ki, kj, kc, ka, kb, c0, c1, a0, a1, b0, b1).transpose(1,2,0,4,5,3)
        out = out + get_w(ki, kk, kj, ka, kc, kb, a0, a1, c0, c1, b0, b1).transpose(0,2,1,3,5,4)
        out = out + get_w(kk, kj, ki, kc, kb, ka, c0, c1, b0, b1, a0, a1).transpose(2,1,0,5,4,3)
        out = out + get_w(kj, ki, kk, kb, ka, kc, b0, b1, a0, a1, c0, c1).transpose(1,0,2,4,3,5)
        return out

    def get_data(kpt_indices):
        idx_args = get_data_slices(kpt_indices, task, kconserv)
        vvop_indices, vooo_indices, t2T_vvop_indices, t2T_vooo_indices = idx_args
        vvop_data = [eris_vvop[tuple(x)] for x in vvop_indices]
        vooo_data = [eris_vooo_C[tuple(x)] for x in vooo_indices]
        t2T_vvop_data = [t2T[tuple(x)] for x in t2T_vvop_indices]
        t2T_vooo_data = [t2T[tuple(x)] for x in t2T_vooo_indices]
        data = [vvop_data, vooo_data, t2T_vvop_data, t2T_vooo_data]
        return data

    def add_and_permute(kpt_indices, orb_indices, data):
        '''Performs permutation and addition of t3 temporary arrays.'''
        ki, kj, kk, ka, kb, kc = kpt_indices
        a0, a1, b0, b1, c0, c1 = orb_indices
        tmp_t3Tv_ijk = np.asarray(data[0], dtype=dtype, order='C')
        tmp_t3Tv_jik = np.asarray(data[1], dtype=dtype, order='C')
        tmp_t3Tv_kji = np.asarray(data[2], dtype=dtype, order='C')
        out_ijk = np.empty(data[0].shape, dtype=dtype, order='C')

        #drv = _ccsd.libcc.MPICCadd_and_permute_t3T
        #drv(ctypes.c_int(nocc), ctypes.c_int(nvir),
        #    ctypes.c_int(0),
        #    out_ijk.ctypes.data_as(ctypes.c_void_p),
        #    tmp_t3Tv_ijk.ctypes.data_as(ctypes.c_void_p),
        #    tmp_t3Tv_jik.ctypes.data_as(ctypes.c_void_p),
        #    tmp_t3Tv_kji.ctypes.data_as(ctypes.c_void_p),
        #    mo_offset.ctypes.data_as(ctypes.c_void_p),
        #    slices.ctypes.data_as(ctypes.c_void_p))
        return (2.*tmp_t3Tv_ijk -
                   tmp_t3Tv_jik.transpose(0,1,2,4,3,5) -
                   tmp_t3Tv_kji.transpose(0,1,2,5,4,3))
        #return out_ijk

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mycc, kind="split")

    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)
    blkmin = 4
    # temporary t3 array is size:  nkpts**3 * blksize**3 * nocc**3 * 16
    vir_blksize = min(nvir, max(blkmin, int((max_memory*.9e6/16/nocc**3/nkpts**3)**(1./3))))
    tasks = []
    logger.debug(mycc, 'max_memory %d MB (%d MB in use)', max_memory, mem_now)
    logger.debug(mycc, 'virtual blksize = %d (nvir = %d)', vir_blksize, nvir)
    for a0, a1 in lib.prange(0, nvir, vir_blksize):
        for b0, b1 in lib.prange(0, nvir, vir_blksize):
            for c0, c1 in lib.prange(0, nvir, vir_blksize):
                tasks.append((a0,a1,b0,b1,c0,c1))

    eaa = []
    for ka in range(nkpts):
        eaa.append(mo_e_o[ka][:, None] - mo_e_v[ka][None, :])

    pt1 = np.zeros((nkpts,nocc,nvir), dtype=dtype)
    pt2 = np.zeros((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=dtype)
    for ka, kb in product(range(nkpts), repeat=2):
        for task_id, task in enumerate(tasks):
            cput2 = (time.clock(), time.time())
            a0,a1,b0,b1,c0,c1 = task
            my_permuted_w = np.zeros((nkpts,)*3 + (a1-a0,b1-b0,c1-c0) + (nocc,)*3, dtype=dtype)

            for ki, kj, kk in product(range(nkpts), repeat=3):
                # Find momentum conservation condition for triples
                # amplitude t3ijkabc
                kc = kpts_helper.get_kconserv3(cell, kpts, [ki, kj, kk, ka, kb])

                kpt_indices = [ki,kj,kk,ka,kb,kc]
                data = get_data(kpt_indices)
                my_permuted_w[ki,kj,kk] = get_permuted_w(ki,kj,kk,ka,kb,kc,task)

            for ki, kj, kk in product(range(nkpts), repeat=3):
                # eigenvalue denominator: e(i) + e(j) + e(k)
                eijk = _get_epqr([0,nocc,ki,mo_e_o,nonzero_opadding],
                                 [0,nocc,kj,mo_e_o,nonzero_opadding],
                                 [0,nocc,kk,mo_e_o,nonzero_opadding])

                # Find momentum conservation condition for triples
                # amplitude t3ijkabc
                kc = kpts_helper.get_kconserv3(cell, kpts, [ki, kj, kk, ka, kb])
                eabc = _get_epqr([a0,a1,ka,mo_e_v,nonzero_vpadding],
                                 [b0,b1,kb,mo_e_v,nonzero_vpadding],
                                 [c0,c1,kc,mo_e_v,nonzero_vpadding],
                                 fac=[-1.,-1.,-1.])

                kpt_indices = [ki,kj,kk,ka,kb,kc]
                eabcijk = (eijk[None,None,None,:,:,:] + eabc[:,:,:,None,None,None])

                tmp_t3Tv_ijk = my_permuted_w[ki,kj,kk]
                tmp_t3Tv_jik = my_permuted_w[kj,ki,kk]
                tmp_t3Tv_kji = my_permuted_w[kk,kj,ki]
                Ptmp_t3Tv = add_and_permute(kpt_indices, task,
                                (tmp_t3Tv_ijk,tmp_t3Tv_jik,tmp_t3Tv_kji))
                Ptmp_t3Tv /= eabcijk

                # Contribution to T1 amplitudes
                if ki == ka and kc == kconserv[kj, kb, kk]:
                    eris_Soovv = (2.*eris.oovv[kj,kk,kb,:,:,b0:b1,c0:c1] -
                                     eris.oovv[kj,kk,kc,:,:,c0:c1,b0:b1].transpose(0,1,3,2))
                    pt1[ka,:,a0:a1] += 0.5*einsum('abcijk,jkbc->ia', Ptmp_t3Tv,
                                                  eris_Soovv)

                # Contribution to T2 amplitudes
                if ki == ka and kc == kconserv[kj, kb, kk]:
                    tmp = einsum('abcijk,ia->jkbc', Ptmp_t3Tv, 0.5*fov[ki,:,a0:a1])
                    _add_pt2(pt2, nkpts, kconserv, [kj,kk,kb], [None,None,(b0,b1),(c0,c1)], tmp)

                kd = kconserv[ka,ki,kb]
                eris_vovv = eris.vovv[kd,ki,kb,:,:,b0:b1,a0:a1]
                tmp = einsum('abcijk,diba->jkdc', Ptmp_t3Tv, eris_vovv)
                _add_pt2(pt2, nkpts, kconserv, [kj,kk,kd], [None,None,None,(c0,c1)], tmp)

                km = kconserv[kc, kk, kb]
                eris_ooov = eris.ooov[kj,ki,km,:,:,:,a0:a1]
                tmp = einsum('abcijk,jima->mkbc', Ptmp_t3Tv, eris_ooov)
                _add_pt2(pt2, nkpts, kconserv, [km,kk,kb], [None,None,(b0,b1),(c0,c1)], -1.*tmp)

                # Contribution to Wovoo array
                km = kconserv[ka,ki,kc]
                eris_oovv = eris.oovv[km,ki,kc,:,:,c0:c1,a0:a1]
                tmp = einsum('abcijk,mica->mbkj', Ptmp_t3Tv, eris_oovv)
                Wmcik[km,kb,kk,:,b0:b1,:,:] += tmp

                # Contribution to Wvvoo array
                ke = kconserv[ki,ka,kk]
                eris_oovv = eris.oovv[ki,kk,ka,:,:,a0:a1,:]
                tmp = einsum('abcijk,ikae->cbej', Ptmp_t3Tv, eris_oovv)
                Wacek[kc,kb,ke,c0:c1,b0:b1,:,:] -= tmp

            logger.timer_debug1(mycc, 'EOM-CCSD T3[2] ka,kb,vir=(%d,%d,%d/%d) [total=%d]'%
                                (ka,kb,task_id,len(tasks),nkpts**5), *cput2)

    for ki in range(nkpts):
        ka = ki
        eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
        n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
        eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]
        pt1[ki] /= eia

    for ki, ka in product(range(nkpts), repeat=2):
        eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
        n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
        eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]
        for kj in range(nkpts):
            kb = kconserv[ki, ka, kj]
            ejb = LARGE_DENOM * np.ones((nocc, nvir), dtype=eris.mo_energy[0].dtype)
            n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
            ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]
            eijab = eia[:, None, :, None] + ejb[:, None, :]
            pt2[ki, kj, ka] /= eijab


    pt1 += t1
    pt2 += t2

    logger.timer(mycc, 'EOM-CCSD(T) imds', *cpu0)

    delta_ccsd_energy = mycc.energy(pt1, pt2, eris) - ccsd_energy
    logger.info(mycc, 'CCSD energy T3[2] correction : %16.12e', delta_ccsd_energy)

    return delta_ccsd_energy, pt1, pt2, Wmcik, Wacek
