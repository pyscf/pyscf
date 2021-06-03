#!/usr/bin/env python
# Copyright 2017-2021 The PySCF Developers. All Rights Reserved.
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

import itertools
import numpy as np
from pyscf import lib
from pyscf.pbc.lib import kpts_helper

einsum = lib.einsum

#FIXME: the dtype of each intermediates. When the khf is at gamma point, the
# dtype is inconsistent between intermediates and t amplitudes

def make_tau(cc, t2, t1, t1p, fac=1.):
    t2aa, t2ab, t2bb = t2
    nkpts = len(t2aa)

    tauaa = t2aa.copy()
    tauab = t2ab.copy()
    taubb = t2bb.copy()
    for ki in range(nkpts):
        for kj in range(nkpts):
            tauaa[ki,kj,ki] += einsum('ia,jb->ijab', fac*.5*t1[0][ki], t1p[0][kj])
            tauaa[ki,kj,kj] -= einsum('ib,ja->ijab', fac*.5*t1[0][ki], t1p[0][kj])
            tauaa[ki,kj,kj] -= einsum('ja,ib->ijab', fac*.5*t1[0][kj], t1p[0][ki])
            tauaa[ki,kj,ki] += einsum('jb,ia->ijab', fac*.5*t1[0][kj], t1p[0][ki])

            taubb[ki,kj,ki] += einsum('ia,jb->ijab', fac*.5*t1[1][ki], t1p[1][kj])
            taubb[ki,kj,kj] -= einsum('ib,ja->ijab', fac*.5*t1[1][ki], t1p[1][kj])
            taubb[ki,kj,kj] -= einsum('ja,ib->ijab', fac*.5*t1[1][kj], t1p[1][ki])
            taubb[ki,kj,ki] += einsum('jb,ia->ijab', fac*.5*t1[1][kj], t1p[1][ki])

            tauab[ki,kj,ki] += einsum('ia,jb->ijab', fac*.5*t1[0][ki], t1p[1][kj])
            tauab[ki,kj,ki] += einsum('jb,ia->ijab', fac*.5*t1[1][kj], t1p[0][ki])
    return tauaa, tauab, taubb

def make_tau2(cc, t2, t1, t1p, fac=1.):
    t2aa, t2ab, t2bb = t2
    nkpts = len(t2aa)

    tauaa = t2aa.copy()
    tauab = t2ab.copy()
    taubb = t2bb.copy()
    for ki in range(nkpts):
        for kj in range(nkpts):
            tauaa[ki,kj,ki] += einsum('ia,jb->ijab', fac*.5*t1[0][ki], t1p[0][kj])
            tauaa[ki,kj,ki] += einsum('jb,ia->ijab', fac*.5*t1[0][kj], t1p[0][ki])

            taubb[ki,kj,ki] += einsum('ia,jb->ijab', fac*.5*t1[1][ki], t1p[1][kj])
            taubb[ki,kj,ki] += einsum('jb,ia->ijab', fac*.5*t1[1][kj], t1p[1][ki])

            tauab[ki,kj,ki] += einsum('ia,jb->ijab', fac*.5*t1[0][ki], t1p[1][kj])
            tauab[ki,kj,ki] += einsum('jb,ia->ijab', fac*.5*t1[1][kj], t1p[0][ki])
    return tauaa, tauab, taubb

def cc_Fvv(cc, t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocc_a, nvir_a = t1a.shape
    nocc_b, nvir_b = t1b.shape[1:]

    kconserv = cc.khelper.kconserv

    fa = np.zeros((nkpts,nvir_a,nvir_a), dtype=np.complex128)
    fb = np.zeros((nkpts,nvir_b,nvir_b), dtype=np.complex128)

    tau_tildeaa,tau_tildeab,tau_tildebb=make_tau(cc,t2,t1,t1,fac=0.5)

    fov = eris.fock[0][:,:nocc_a,nocc_a:]
    fOV = eris.fock[1][:,:nocc_b,nocc_b:]
    fvv = eris.fock[0][:,nocc_a:,nocc_a:]
    fVV = eris.fock[1][:,nocc_b:,nocc_b:]

    for ka in range(nkpts):
        fa[ka]+=fvv[ka]
        fb[ka]+=fVV[ka]
        fa[ka]-=0.5*einsum('me,ma->ae',fov[ka],t1a[ka])
        fb[ka]-=0.5*einsum('me,ma->ae',fOV[ka],t1b[ka])
        for km in range(nkpts):
            fa[ka]+=einsum('mf,fmea->ae',t1a[km], eris.vovv[km,km,ka].conj())
            fa[ka]-=einsum('mf,emfa->ae',t1a[km], eris.vovv[ka,km,km].conj())
            fa[ka]+=einsum('mf,fmea->ae',t1b[km], eris.VOvv[km,km,ka].conj())

            fb[ka]+=einsum('mf,fmea->ae',t1b[km], eris.VOVV[km,km,ka].conj())
            fb[ka]-=einsum('mf,emfa->ae',t1b[km], eris.VOVV[ka,km,km].conj())
            fb[ka]+=einsum('mf,fmea->ae',t1a[km], eris.voVV[km,km,ka].conj())

            for kn in range(nkpts):
                kf = kconserv[km,ka,kn]
                tmp = eris.ovov[km,ka,kn] - eris.ovov[km,kf,kn].transpose(0,3,2,1)
                fa[ka] -= einsum('mnaf,menf->ae', tau_tildeaa[km,kn,ka], tmp) * .5
                fa[ka] -= einsum('mNaF,meNF->ae', tau_tildeab[km,kn,ka], eris.ovOV[km,ka,kn])

                tmp = eris.OVOV[km,ka,kn] - eris.OVOV[km,kf,kn].transpose(0,3,2,1)
                fb[ka] -= einsum('mnaf,menf->ae', tau_tildebb[km,kn,ka], tmp) * .5
                fb[ka] -= einsum('MnFa,MFne->ae', tau_tildeab[km,kn,kf], eris.ovOV[km,kf,kn])

    return fa,fb


def cc_Foo(cc, t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocc_a, nvir_a = t1a.shape
    nocc_b, nvir_b = t1b.shape[1:]

    kconserv = cc.khelper.kconserv

    fa = np.zeros((nkpts,nocc_a,nocc_a), dtype=np.complex128)
    fb = np.zeros((nkpts,nocc_b,nocc_b), dtype=np.complex128)

    tau_tildeaa,tau_tildeab,tau_tildebb=make_tau(cc,t2,t1,t1,fac=0.5)

    fov = eris.fock[0][:,:nocc_a,nocc_a:]
    fOV = eris.fock[1][:,:nocc_b,nocc_b:]
    foo = eris.fock[0][:,:nocc_a,:nocc_a]
    fOO = eris.fock[1][:,:nocc_b,:nocc_b]

    for ka in range(nkpts):
        fa[ka]+=foo[ka]
        fb[ka]+=fOO[ka]
        fa[ka]+=0.5*einsum('me,ne->mn',fov[ka],t1a[ka])
        fb[ka]+=0.5*einsum('me,ne->mn',fOV[ka],t1b[ka])
        for km in range(nkpts):
            fa[ka]+=einsum('oa,mnoa->mn',t1a[km],eris.ooov[ka,ka,km])
            fa[ka]+=einsum('oa,mnoa->mn',t1b[km],eris.ooOV[ka,ka,km])
            fa[ka]-=einsum('oa,onma->mn',t1a[km],eris.ooov[km,ka,ka])

            fb[ka]+=einsum('oa,mnoa->mn',t1b[km],eris.OOOV[ka,ka,km])
            fb[ka]+=einsum('oa,mnoa->mn',t1a[km],eris.OOov[ka,ka,km])
            fb[ka]-=einsum('oa,onma->mn',t1b[km],eris.OOOV[km,ka,ka])

    for km in range(nkpts):
        for kn in range(nkpts):
            for ke in range(nkpts):
                kf = kconserv[km,ke,kn]
                tmp = eris.ovov[km,ke,kn] - eris.ovov[km,kf,kn].transpose(0,3,2,1)
                fa[km] += einsum('inef,menf->mi', tau_tildeaa[km,kn,ke], tmp) * .5
                fa[km] += einsum('iNeF,meNF->mi',tau_tildeab[km,kn,ke],eris.ovOV[km,ke,kn])

                tmp = eris.OVOV[km,ke,kn] - eris.OVOV[km,kf,kn].transpose(0,3,2,1)
                fb[km] += einsum('INEF,MENF->MI',tau_tildebb[km,kn,ke], tmp) * .5
                fb[km] += einsum('nIeF,neMF->MI',tau_tildeab[kn,km,ke],eris.ovOV[kn,ke,km])

    return fa,fb


def cc_Fov(cc, t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocc_a, nvir_a = t1a.shape
    nocc_b, nvir_b = t1b.shape[1:]

    fov = eris.fock[0][:,:nocc_a,nocc_a:]
    fOV = eris.fock[1][:,:nocc_b,nocc_b:]

    fa = np.zeros((nkpts,nocc_a,nvir_a), dtype=np.complex128)
    fb = np.zeros((nkpts,nocc_b,nvir_b), dtype=np.complex128)

    for km in range(nkpts):
        fa[km]+=fov[km]
        fb[km]+=fOV[km]
        for kn in range(nkpts):
            fa[km]+=einsum('nf,menf->me',t1a[kn],eris.ovov[km,km,kn])
            fa[km]+=einsum('nf,menf->me',t1b[kn],eris.ovOV[km,km,kn])
            fa[km]-=einsum('nf,mfne->me',t1a[kn],eris.ovov[km,kn,kn])
            fb[km]+=einsum('nf,menf->me',t1b[kn],eris.OVOV[km,km,kn])
            fb[km]+=einsum('nf,nfme->me',t1a[kn],eris.ovOV[kn,kn,km])
            fb[km]-=einsum('nf,mfne->me',t1b[kn],eris.OVOV[km,kn,kn])

    return fa,fb

def cc_Woooo(cc, t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape[1:]
    dtype = np.result_type(t1a, t1b, t2aa, t2ab, t2bb)

    Woooo = np.zeros(eris.oooo.shape, dtype=dtype)
    WooOO = np.zeros(eris.ooOO.shape, dtype=dtype)
    WOOOO = np.zeros(eris.OOOO.shape, dtype=dtype)

    kconserv = cc.khelper.kconserv
    tau_aa, tau_ab, tau_bb = make_tau(cc, t2, t1, t1)
    for km in range(nkpts):
        for kn in range(nkpts):
            tmp_aaaaJ = einsum('xje, ymine->yxminj', t1a, eris.ooov[km,:,kn])
            tmp_aaaaJ -= tmp_aaaaJ.transpose((1,0,2,5,4,3))
            tmp_bbbbJ = einsum('xje, ymine->yxminj', t1b, eris.OOOV[km,:,kn])
            tmp_bbbbJ -= tmp_bbbbJ.transpose((1,0,2,5,4,3))
            tmp_aabbJ = einsum('xje, ymine->yxminj', t1b, eris.ooOV[km,:,kn])
            tmp_baabJ = -einsum('yie,xmjne->yxminj', t1a, eris.OOov[km,:,kn])

            Woooo[km,:,kn] += eris.oooo[km,:,kn]
            WooOO[km,:,kn] += eris.ooOO[km,:,kn]
            WOOOO[km,:,kn] += eris.OOOO[km,:,kn]

            ki = range(nkpts)
            kj = kconserv[km,ki,kn]
            Woooo[km,ki,kn] += tmp_aaaaJ[ki,kj]
            WOOOO[km,ki,kn] += tmp_bbbbJ[ki,kj]
            WooOO[km,ki,kn] += tmp_aabbJ[ki,kj]
            WooOO[kn,ki,km] -= tmp_baabJ[ki,kj].transpose(0,3,2,1,4)
            Woooo[km,ki,kn] += 0.25*einsum('yxijef,xmenf->yminj', tau_aa[ki,kj], eris.ovov[km,ki,kn])
            WOOOO[km,ki,kn] += 0.25*einsum('yxijef,xmenf->yminj', tau_bb[ki,kj], eris.OVOV[km,ki,kn])
            WooOO[km,ki,kn] += 0.5*einsum('yxijef,xmenf->yminj', tau_ab[ki,kj], eris.ovOV[km,ki,kn])

    Woooo = Woooo - Woooo.transpose(2,1,0,5,4,3,6)
    WOOOO = WOOOO - WOOOO.transpose(2,1,0,5,4,3,6)
    return Woooo, WooOO, WOOOO


def cc_Wvvvv(cc, t1, t2, eris):
    t1a, t1b = t1
    kconserv = cc.khelper.kconserv

    #:wvvvv = eris.vvvv.copy()
    #:Wvvvv += np.einsum('ymb,zyxemfa,zxyw->wzyaebf', t1a, eris.vovv.conj(), P)
    #:Wvvvv -= np.einsum('ymb,xyzfmea,xzyw->wzyaebf', t1a, eris.vovv.conj(), P)
    #:Wvvvv = Wvvvv - Wvvvv.transpose(2,1,0,5,4,3,6)
    Wvvvv = np.zeros_like(eris.vvvv)
    for ka, kb, ke in kpts_helper.loop_kkk(cc.nkpts):
        kf = kconserv[ka,ke,kb]
        aebf = eris.vvvv[ka,ke,kb].copy()
        aebf += einsum('mb,emfa->aebf', t1a[kb], eris.vovv[ke,kb,kf].conj())
        aebf -= einsum('mb,fmea->aebf', t1a[kb], eris.vovv[kf,kb,ke].conj())
        Wvvvv[ka,ke,kb] += aebf
        Wvvvv[kb,ke,ka] -= aebf.transpose(2,1,0,3)

    #:WvvVV = eris.vvVV.copy()
    #:WvvVV -= np.einsum('xma,zxwemFB,zwxy->xzyaeBF', t1a, eris.voVV.conj(), P)
    #:WvvVV -= np.einsum('yMB,wyzFMea,wzyx->xzyaeBF', t1b, eris.VOvv.conj(), P)
    WvvVV = np.empty_like(eris.vvVV)
    for ka, kb, ke in kpts_helper.loop_kkk(cc.nkpts):
        kf = kconserv[ka,ke,kb]
        aebf = eris.vvVV[ka,ke,kb].copy()
        aebf -= einsum('ma,emfb->aebf', t1a[ka], eris.voVV[ke,ka,kf].conj())
        aebf -= einsum('mb,fmea->aebf', t1b[kb], eris.VOvv[kf,kb,ke].conj())
        WvvVV[ka,ke,kb] = aebf

    #:WVVVV = eris.VVVV.copy()
    #:WVVVV += np.einsum('ymb,zyxemfa,zxyw->wzyaebf', t1b, eris.VOVV.conj(), P)
    #:WVVVV -= np.einsum('ymb,xyzfmea,xzyw->wzyaebf', t1b, eris.VOVV.conj(), P)
    #:WVVVV = WVVVV - WVVVV.transpose(2,1,0,5,4,3,6)
    WVVVV = np.zeros_like(eris.VVVV)
    for ka, kb, ke in kpts_helper.loop_kkk(cc.nkpts):
        kf = kconserv[ka,ke,kb]
        aebf = eris.VVVV[ka,ke,kb].copy()
        aebf += einsum('mb,emfa->aebf', t1b[kb], eris.VOVV[ke,kb,kf].conj())
        aebf -= einsum('mb,fmea->aebf', t1b[kb], eris.VOVV[kf,kb,ke].conj())
        WVVVV[ka,ke,kb] += aebf
        WVVVV[kb,ke,ka] -= aebf.transpose(2,1,0,3)
    return Wvvvv, WvvVV, WVVVV


#TODO: merge cc_Wvvvv_half and cc_Wvvvv
def cc_Wvvvv_half(cc, t1, t2, eris):
    '''Similar to cc_Wvvvv, without anti-symmetrization'''
    t1a, t1b = t1
    kconserv = cc.khelper.kconserv

    #:wvvvv = eris.vvvv.copy()
    #:Wvvvv += np.einsum('ymb,zyxemfa,zxyw->wzyaebf', t1a, eris.vovv.conj(), P)
    #:Wvvvv -= np.einsum('ymb,xyzfmea,xzyw->wzyaebf', t1a, eris.vovv.conj(), P)
    #:Wvvvv = Wvvvv - Wvvvv.transpose(2,1,0,5,4,3,6)
    Wvvvv = np.zeros_like(eris.vvvv)
    for ka, kb, ke in kpts_helper.loop_kkk(cc.nkpts):
        kf = kconserv[ka,ke,kb]
        aebf = eris.vvvv[ka,ke,kb].copy()
        aebf += einsum('mb,emfa->aebf', t1a[kb], eris.vovv[ke,kb,kf].conj())
        aebf -= einsum('mb,fmea->aebf', t1a[kb], eris.vovv[kf,kb,ke].conj())
        Wvvvv[ka,ke,kb] += aebf

    #:WvvVV = eris.vvVV.copy()
    #:WvvVV -= np.einsum('xma,zxwemFB,zwxy->xzyaeBF', t1a, eris.voVV.conj(), P)
    #:WvvVV -= np.einsum('yMB,wyzFMea,wzyx->xzyaeBF', t1b, eris.VOvv.conj(), P)
    WvvVV = np.empty_like(eris.vvVV)
    for ka, kb, ke in kpts_helper.loop_kkk(cc.nkpts):
        kf = kconserv[ka,ke,kb]
        aebf = eris.vvVV[ka,ke,kb].copy()
        aebf -= einsum('ma,emfb->aebf', t1a[ka], eris.voVV[ke,ka,kf].conj())
        aebf -= einsum('mb,fmea->aebf', t1b[kb], eris.VOvv[kf,kb,ke].conj())
        WvvVV[ka,ke,kb] = aebf

    #:WVVVV = eris.VVVV.copy()
    #:WVVVV += np.einsum('ymb,zyxemfa,zxyw->wzyaebf', t1b, eris.VOVV.conj(), P)
    #:WVVVV -= np.einsum('ymb,xyzfmea,xzyw->wzyaebf', t1b, eris.VOVV.conj(), P)
    #:WVVVV = WVVVV - WVVVV.transpose(2,1,0,5,4,3,6)
    WVVVV = np.zeros_like(eris.VVVV)
    for ka, kb, ke in kpts_helper.loop_kkk(cc.nkpts):
        kf = kconserv[ka,ke,kb]
        aebf = eris.VVVV[ka,ke,kb].copy()
        aebf += einsum('mb,emfa->aebf', t1b[kb], eris.VOVV[ke,kb,kf].conj())
        aebf -= einsum('mb,fmea->aebf', t1b[kb], eris.VOVV[kf,kb,ke].conj())
        WVVVV[ka,ke,kb] += aebf
    return Wvvvv, WvvVV, WVVVV

def Wvvvv(cc, t1, t2, eris):
    nkpts = cc.nkpts
    kconserv = cc.khelper.kconserv

    tauaa, tauab, taubb = make_tau(cc, t2, t1, t1)
    Wvvvv, WvvVV, WVVVV = cc_Wvvvv(cc, t1, t2, eris)
    for ka, kb, ke in kpts_helper.loop_kkk(cc.nkpts):
        for km in range(nkpts):
            kn = kconserv[ka,km,kb]
            Wvvvv[ka,ke,kb] += einsum('mnab,menf->aebf', tauaa[km,kn,ka], eris.ovov[km,ke,kn])
            WvvVV[ka,ke,kb] += einsum('mNaB,meNF->aeBF', tauab[km,kn,ka], eris.ovOV[km,ke,kn])
            WVVVV[ka,ke,kb] += einsum('mnab,menf->aebf', taubb[km,kn,ka], eris.OVOV[km,ke,kn])
    return Wvvvv, WvvVV, WVVVV

def get_Wvvvv(cc, t1, t2, eris, ka, kb, kc):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb = cc.nocc
    nkpts = cc.nkpts
    kconserv = cc.khelper.kconserv
    kd = kconserv[ka, kc, kb]

    if getattr(eris, 'Lpv', None) is not None:
        # Using GDF to generate Wvvvv on the fly
        Lpv = eris.Lpv
        LPV = eris.LPV
        Lac = (Lpv[ka,kc][:,nocca:] -
               einsum('Lkc,ka->Lac', Lpv[ka,kc][:,:nocca], t1a[ka]))
        Lbd = (Lpv[kb,kd][:,nocca:] -
               einsum('Lkd,kb->Lbd', Lpv[kb,kd][:,:nocca], t1a[kb]))
        Lbc = (Lpv[kb,kc][:,nocca:] -
               einsum('Lkc,ka->Lac', Lpv[kb,kc][:,:nocca], t1a[kb]))
        Lad = (Lpv[ka,kd][:,nocca:] -
               einsum('Lkd,kb->Lbd', Lpv[ka,kd][:,:nocca], t1a[ka]))
        LAC = (LPV[ka,kc][:,noccb:] -
               einsum('Lkd,kb->Lbd', LPV[ka,kc][:,:noccb], t1b[ka]))
        LBD = (LPV[kb,kd][:,noccb:] -
               einsum('Lkd,kb->Lbd', LPV[kb,kd][:,:noccb], t1b[kb]))
        LBC = (LPV[kb,kc][:,noccb:] -
               einsum('Lkc,ka->Lac', LPV[kb,kc][:,:noccb], t1b[kb]))
        LAD = (LPV[ka,kd][:,noccb:] -
               einsum('Lkd,kb->Lbd', LPV[ka,kd][:,:noccb], t1b[ka]))
        vvvv = einsum('Lac,Lbd->acbd', Lac, Lbd)
        vvvv-= einsum('Lbc,Lad->acbd', Lbc, Lad)
        vvVV = einsum('Lac,Lbd->acbd', Lac, LBD)
        VVVV = einsum('Lac,Lbd->acbd', LAC, LBD)
        VVVV-= einsum('Lbc,Lad->acbd', LBC, LAD)
        vvvv *= (1./nkpts)
        vvVV *= (1./nkpts)
        VVVV *= (1./nkpts)
    else:
        vvvv  = einsum('emfa,mb->aebf', eris.vovv[kc,kb,kd].conj(), t1a[kb])
        vvvv -= einsum('fmea,mb->aebf', eris.vovv[kd,kb,kc].conj(), t1a[kb])
        vvvv -= einsum('emfb,ma->aebf', eris.vovv[kc,ka,kd].conj(), t1a[ka])
        vvvv += einsum('fmeb,ma->aebf', eris.vovv[kd,ka,kc].conj(), t1a[ka])
        vvvv += eris.vvvv[ka,kc,kb]
        vvvv -= eris.vvvv[kb,kc,ka].transpose(2,1,0,3)
        vvvv += einsum('mcnf,ma,nb->acbf', eris.ovov[ka,kc,kb], t1a[ka], t1a[kb])
        vvvv -= einsum('mcnf,mb,na->acbf', eris.ovov[kb,kc,ka], t1a[kb], t1a[ka])

        vvVV  = einsum('emfb,ma->aebf', eris.voVV[kc,ka,kd].conj(),-t1a[ka])
        vvVV += einsum('fmea,mb->aebf', eris.VOvv[kd,kb,kc].conj(),-t1b[kb])
        vvVV += einsum('mcnf,ma,nb->acbf', eris.ovOV[ka,kc,kb], t1a[ka], t1b[kb])
        vvVV += eris.vvVV[ka,kc,kb]

        VVVV  = einsum('emfa,mb->aebf', eris.VOVV[kc,kb,kd].conj(), t1b[kb])
        VVVV -= einsum('fmea,mb->aebf', eris.VOVV[kd,kb,kc].conj(), t1b[kb])
        VVVV -= einsum('emfb,ma->aebf', eris.VOVV[kc,ka,kd].conj(), t1b[ka])
        VVVV += einsum('fmeb,ma->aebf', eris.VOVV[kd,ka,kc].conj(), t1b[ka])
        VVVV += eris.VVVV[ka,kc,kb]
        VVVV -= eris.VVVV[kb,kc,ka].transpose(2,1,0,3)
        VVVV += einsum('mcnf,ma,nb->acbf', eris.OVOV[ka,kc,kb], t1b[ka], t1b[kb])
        VVVV -= einsum('mcnf,mb,na->acbf', eris.OVOV[kb,kc,ka], t1b[kb], t1b[ka])

    for km in range(nkpts):
        kn = kconserv[ka,km,kb]
        vvvv += einsum('mnab,mcnf->acbf', t2aa[km,kn,ka], eris.ovov[km,kc,kn])
        vvVV += einsum('mNaB,mcNF->acBF', t2ab[km,kn,ka], eris.ovOV[km,kc,kn])
        VVVV += einsum('mnab,mcnf->acbf', t2bb[km,kn,ka], eris.OVOV[km,kc,kn])
    return vvvv, vvVV, VVVV

def cc_Wovvo(cc, t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape[1:]
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)

    dtype = np.result_type(*t2)
    Wovvo = np.zeros((nkpts,nkpts,nkpts,nocca,nvira,nvira,nocca), dtype)
    WovVO = np.zeros((nkpts,nkpts,nkpts,nocca,nvira,nvirb,noccb), dtype)
    WOVvo = np.zeros((nkpts,nkpts,nkpts,noccb,nvirb,nvira,nocca), dtype)
    WOVVO = np.zeros((nkpts,nkpts,nkpts,noccb,nvirb,nvirb,noccb), dtype)
    WoVVo = np.zeros((nkpts,nkpts,nkpts,nocca,nvirb,nvirb,nocca), dtype)
    WOvvO = np.zeros((nkpts,nkpts,nkpts,noccb,nvira,nvira,noccb), dtype)

    for ka, ki, kj in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ka,ki,kj]
        Wovvo[ki,ka,kb] += eris.voov[ka,ki,kj].conj().transpose(1,0,3,2)
        WovVO[ki,ka,kb] += eris.voOV[ka,ki,kj].conj().transpose(1,0,3,2)
        WOVvo[ki,ka,kb] += eris.voOV[kb,kj,ki].transpose(2,3,0,1)
        WOVVO[ki,ka,kb] += eris.VOOV[ka,ki,kj].conj().transpose(1,0,3,2)

        kb = kconserv[ki,kj,ka]
        Wovvo[ki,kb,ka] -= eris.oovv[ki,kj,ka].transpose(0,3,2,1)
        WOVVO[ki,kb,ka] -= eris.OOVV[ki,kj,ka].transpose(0,3,2,1)
        WoVVo[ki,kb,ka] -= eris.ooVV[ki,kj,ka].transpose(0,3,2,1)
        WOvvO[ki,kb,ka] -= eris.OOvv[ki,kj,ka].transpose(0,3,2,1)

    tauaa, tauab, taubb = make_tau2(cc, t2, t1, t1,fac=2.0)
    for km in range(nkpts):
        for kb in range(nkpts):
            for ke in range(nkpts):
                kj = kconserv[km,ke,kb]
                vovv = eris.vovv[ke,km,kj].conj()
                VOVV = eris.VOVV[ke,km,kj].conj()
                voVV = eris.voVV[ke,km,kj].conj()
                VOvv = eris.VOvv[ke,km,kj].conj()

                Wovvo[km,ke,kb] += einsum('jf, emfb->mebj', t1a[kj], vovv)
                WOVVO[km,ke,kb] += einsum('jf, emfb->mebj', t1b[kj], VOVV)
                WovVO[km,ke,kb] += einsum('jf, emfb->mebj', t1b[kj], voVV)
                WOVvo[km,ke,kb] += einsum('jf, emfb->mebj', t1a[kj], VOvv)
                ##### warnings for Ks
                Wovvo[km,kj,kb] -= einsum('je, emfb->mfbj', t1a[ke], vovv)
                WOVVO[km,kj,kb] -= einsum('je, emfb->mfbj', t1b[ke], VOVV)
                WOvvO[km,kj,kb] -= einsum('je, emfb->mfbj', t1b[ke], VOvv)
                WoVVo[km,kj,kb] -= einsum('je, emfb->mfbj', t1a[ke], voVV)

                WOVvo[km,ke,kb] -= einsum('nb, njme->mebj', t1a[kb], eris.ooOV[kb,kj,km])
                WovVO[km,ke,kb] -= einsum('nb, njme->mebj', t1b[kb], eris.OOov[kb,kj,km])

                WOvvO[km,ke,kb] += einsum('nb, mjne->mebj', t1a[kb], eris.OOov[km,kj,kb])
                WoVVo[km,ke,kb] += einsum('nb, mjne->mebj', t1b[kb], eris.ooOV[km,kj,kb])

                ooov_temp = eris.ooov[kb,kj,km] - eris.ooov[km,kj,kb].transpose((2,1,0,3))
                Wovvo[km,ke,kb] -= einsum('nb, njme->mebj', t1a[kb], ooov_temp)
                ooov_temp = None
                OOOV_temp = eris.OOOV[kb,kj,km] - eris.OOOV[km,kj,kb].transpose((2,1,0,3))
                WOVVO[km,ke,kb] -= einsum('nb, njme->mebj', t1b[kb], OOOV_temp)
                OOOV_temp = None

                Wovvo[km,ke,kb] += 0.5*einsum('xjnbf,xmenf->mebj', t2ab[kj,:,kb], eris.ovOV[km,ke,:])
                WOvvO[km,ke,kb] += 0.5*einsum('xnjbf,xnemf->mebj', tauab[:,kj,kb], eris.ovOV[:,ke,km])
                WovVO[km,ke,kb] -= 0.5*einsum('xnjbf,xmenf->mebj', taubb[:,kj,kb], eris.ovOV[km,ke,:])

                temp_ovOV_1 = np.zeros([nkpts, nocca, nvira, noccb, nvirb], dtype=dtype)
                temp_ovOV_2 = np.zeros([nkpts, nocca, nvira, noccb, nvirb], dtype=dtype)
                for kn in range(nkpts):
                    kf = kconserv[km,ke,kn]
                    temp_ovOV_1[kn] += eris.ovOV[kn,kf,km].copy()
                    temp_ovOV_2[kn] += eris.ovOV[km,kf,kn].copy()

                kn = range(nkpts)
                kf = kconserv[km,ke][kn]
                WOVVO[km,ke,kb] += 0.5*einsum('xnjfb,xnfme->mebj', t2ab[kn,kj,kf], temp_ovOV_1)
                WOVvo[km,ke,kb] -= 0.5*einsum('xnjbf,xnfme->mebj', tauaa[:,kj,kb], temp_ovOV_1)
                WoVVo[km,ke,kb] += 0.5*einsum('xjnfb,xmfne->mebj', tauab[kj,kn,kf], temp_ovOV_2)

                temp_OVOV = eris.OVOV[km,ke,:] - eris.OVOV[:,ke,km].transpose((0,3,2,1,4))
                WOVVO[km,ke,kb] -= 0.5*einsum('xnjbf,xmenf->mebj', taubb[:,kj,kb], temp_OVOV)
                WOVvo[km,ke,kb] += 0.5*einsum('xjnbf,xmenf->mebj', t2ab[kj,:,kb], temp_OVOV)
                temp_OVOV = None

                temp_ovov = eris.ovov[:,ke,km] - eris.ovov[km,ke,:].transpose((0,3,2,1,4))
                Wovvo[km,ke,kb] += 0.5*einsum('xnjbf,xnemf->mebj', tauaa[:,kj,kb], temp_ovov)
                WovVO[km,ke,kb] -= 0.5*einsum('xnjfb,xnemf->mebj', t2ab[kn,kj,kf], temp_ovov)
                temp_ovov = None

    return Wovvo, WovVO, WOVvo, WOVVO, WoVVo, WOvvO

def _cc_Wovvo_k0k2(cc, t1, t2, eris, k0, k2):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape[1:]
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)

    dtype = np.result_type(*t2)
    Wovvo = np.zeros((nkpts,nocca,nvira,nvira,nocca), dtype)
    WovVO = np.zeros((nkpts,nocca,nvira,nvirb,noccb), dtype)
    WOVvo = np.zeros((nkpts,noccb,nvirb,nvira,nocca), dtype)
    WOVVO = np.zeros((nkpts,noccb,nvirb,nvirb,noccb), dtype)
    WoVVo = np.zeros((nkpts,nocca,nvirb,nvirb,nocca), dtype)
    WOvvO = np.zeros((nkpts,noccb,nvira,nvira,noccb), dtype)

    #:P = kconserv_mat(cc.nkpts, kconserv)
    #:Wovvo = np.einsum('xyzaijb,xzyw->yxwiabj', eris.voov, P).conj()
    #:WovVO = np.einsum('xyzaijb,xzyw->yxwiabj', eris.voOV, P).conj()
    #:WOVvo = np.einsum('wzybjia,xzyw->yxwiabj', eris.voOV, P)
    #:WOVVO = np.einsum('xyzaijb,xzyw->yxwiabj', eris.VOOV, P).conj()
    #:Wovvo-= np.einsum('xyzijab,xzyw->xwzibaj', eris.oovv, P)
    #:WOVVO-= np.einsum('xyzijab,xzyw->xwzibaj', eris.OOVV, P)
    #:WoVVo = np.einsum('xyzijab,xzyw->xwzibaj', eris.ooVV, -P)
    #:WOvvO = np.einsum('xyzijab,xzyw->xwzibaj', eris.OOvv, -P)
    for kj in range(nkpts):
        ka = kconserv[k2,kj,k0]
        Wovvo[ka] += eris.voov[ka,k0,kj].conj().transpose(1,0,3,2)
        WovVO[ka] += eris.voOV[ka,k0,kj].conj().transpose(1,0,3,2)
        WOVvo[ka] += eris.voOV[k2,kj,k0].transpose(2,3,0,1)
        WOVVO[ka] += eris.VOOV[ka,k0,kj].conj().transpose(1,0,3,2)

    for kj in range(nkpts):
        kb = kconserv[k0,kj,k2]
        Wovvo[kb] -= eris.oovv[k0,kj,k2].transpose(0,3,2,1)
        WOVVO[kb] -= eris.OOVV[k0,kj,k2].transpose(0,3,2,1)
        WoVVo[kb] -= eris.ooVV[k0,kj,k2].transpose(0,3,2,1)
        WOvvO[kb] -= eris.OOvv[k0,kj,k2].transpose(0,3,2,1)

    for ke in range(nkpts):
        kj = kconserv[k0,ke,k2]
        vovv = eris.vovv[ke,k0,kj].conj()
        VOVV = eris.VOVV[ke,k0,kj].conj()
        voVV = eris.voVV[ke,k0,kj].conj()
        VOvv = eris.VOvv[ke,k0,kj].conj()

        Wovvo[ke] += einsum('jf, emfb->mebj', t1a[kj], vovv)
        WOVVO[ke] += einsum('jf, emfb->mebj', t1b[kj], VOVV)
        WovVO[ke] += einsum('jf, emfb->mebj', t1b[kj], voVV)
        WOVvo[ke] += einsum('jf, emfb->mebj', t1a[kj], VOvv)

        Wovvo[kj] -= einsum('je, emfb->mfbj', t1a[ke], vovv)
        WOVVO[kj] -= einsum('je, emfb->mfbj', t1b[ke], VOVV)
        WOvvO[kj] -= einsum('je, emfb->mfbj', t1b[ke], VOvv)
        WoVVo[kj] -= einsum('je, emfb->mfbj', t1a[ke], voVV)

        Wovvo[ke] -= einsum('nb, njme->mebj', t1a[k2], eris.ooov[k2,kj,k0])
        WOVvo[ke] -= einsum('nb, njme->mebj', t1a[k2], eris.ooOV[k2,kj,k0])
        WOVVO[ke] -= einsum('nb, njme->mebj', t1b[k2], eris.OOOV[k2,kj,k0])
        WovVO[ke] -= einsum('nb, njme->mebj', t1b[k2], eris.OOov[k2,kj,k0])

        Wovvo[ke] += einsum('nb, mjne->mebj', t1a[k2], eris.ooov[k0,kj,k2])
        WOVVO[ke] += einsum('nb, mjne->mebj', t1b[k2], eris.OOOV[k0,kj,k2])
        WoVVo[ke] += einsum('nb, mjne->mebj', t1b[k2], eris.ooOV[k0,kj,k2])
        WOvvO[ke] += einsum('nb, mjne->mebj', t1a[k2], eris.OOov[k0,kj,k2])

        for kn in range(nkpts):
            kf = kconserv[k0,ke,kn]

            tmp = eris.ovov[k0,ke,kn] - eris.ovov[kn,ke,k0].transpose(2,1,0,3)
            Wovvo[ke] -= 0.5*einsum('jnfb,menf->mebj', t2aa[kj,kn,kf], tmp)
            Wovvo[ke] += 0.5*einsum('jnbf,menf->mebj', t2ab[kj,kn,k2], eris.ovOV[k0,ke,kn])
            tmp = eris.OVOV[k0,ke,kn] - eris.OVOV[kn,ke,k0].transpose(2,1,0,3)
            WOVVO[ke] -= 0.5*einsum('jnfb,menf->mebj', t2bb[kj,kn,kf], tmp)
            WOVVO[ke] += 0.5*einsum('njfb,nfme->mebj', t2ab[kn,kj,kf], eris.ovOV[kn,kf,k0])
            tmp = eris.ovov[k0,ke,kn] - eris.ovov[kn,ke,k0].transpose(2,1,0,3)
            WovVO[ke] += 0.5*einsum('njfb,menf->mebj', t2ab[kn,kj,kf], tmp)
            WovVO[ke] -= 0.5*einsum('jnfb,menf->mebj', t2bb[kj,kn,kf], eris.ovOV[k0,ke,kn])
            tmp = eris.OVOV[k0,ke,kn] - eris.OVOV[kn,ke,k0].transpose(2,1,0,3)
            WOVvo[ke] += 0.5*einsum('jnbf,menf->mebj', t2ab[kj,kn,k2], tmp)
            WOVvo[ke] -= 0.5*einsum('jnfb,nfme->mebj', t2aa[kj,kn,kf], eris.ovOV[kn,kf,k0])
            WoVVo[ke] += 0.5*einsum('jnfb,mfne->mebj', t2ab[kj,kn,kf], eris.ovOV[k0,kf,kn])
            WOvvO[ke] += 0.5*einsum('njbf,nemf->mebj', t2ab[kn,kj,k2], eris.ovOV[kn,ke,k0])

            if kn == k2 and kf == kj:
                tmp = einsum('menf,jf->menj', eris.ovov[k0,ke,kn], t1a[kj])
                tmp-= einsum('nemf,jf->menj', eris.ovov[kn,ke,k0], t1a[kj])
                Wovvo[ke] -= einsum('nb,menj->mebj', t1a[kn], tmp)
                tmp = einsum('menf,jf->menj', eris.OVOV[k0,ke,kn], t1b[kj])
                tmp-= einsum('nemf,jf->menj', eris.OVOV[kn,ke,k0], t1b[kj])
                WOVVO[ke] -= einsum('nb,menj->mebj', t1b[kn], tmp)

                WovVO[ke] -= einsum('jf,nb,menf->mebj',t1b[kj],t1b[kn], eris.ovOV[k0,ke,kn])
                WOVvo[ke] -= einsum('jf,nb,nfme->mebj',t1a[kj],t1a[kn], eris.ovOV[kn,kf,k0])
                WoVVo[ke] += einsum('jf,nb,mfne->mebj',t1a[kj],t1b[kn], eris.ovOV[k0,kf,kn])
                WOvvO[ke] += einsum('jf,nb,nemf->mebj',t1b[kj],t1a[kn], eris.ovOV[kn,ke,k0])

    return Wovvo, WovVO, WOVvo, WOVVO, WoVVo, WOvvO


def kconserv_mat(nkpts, kconserv):
    P = np.zeros((nkpts,nkpts,nkpts,nkpts))
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                P[ki,kj,ka,kb] = 1
    return P

def Foo(cc,t1,t2,eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocca, noccb, nvira, nvirb = t2ab.shape[2:]

    Fova, Fovb = cc_Fov(cc,t1,t2,eris)
    Fooa, Foob = cc_Foo(cc,t1,t2,eris)
    for ki in range(nkpts):
        Fooa[ki] += 0.5*einsum('ie,me->mi',t1a[ki],Fova[ki])
        Foob[ki] += 0.5*einsum('ie,me->mi',t1b[ki],Fovb[ki])
    return Fooa, Foob

def Fvv(cc,t1,t2,eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocca, noccb, nvira, nvirb = t2ab.shape[2:]

    Fova, Fovb = cc_Fov(cc,t1,t2,eris)
    Fvva, Fvvb = cc_Fvv(cc,t1,t2,eris)
    for ka in range(nkpts):
        Fvva[ka] -= 0.5*lib.einsum('me,ma->ae', Fova[ka], t1a[ka])
        Fvvb[ka] -= 0.5*lib.einsum('me,ma->ae', Fovb[ka], t1b[ka])
    return Fvva, Fvvb

def Fov(cc,t1,t2,eris):
    Fme = cc_Fov(cc,t1,t2,eris)
    return Fme

def Wvvov(cc,t1,t2,eris):
    kconserv = cc.khelper.kconserv
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocca, noccb, nvira, nvirb = t2ab.shape[2:]

    Wvvov = np.zeros((nkpts,nkpts,nkpts,nvira,nvira,nocca,nvira),dtype=t1a.dtype)
    WvvOV = np.zeros((nkpts,nkpts,nkpts,nvira,nvira,noccb,nvirb),dtype=t1a.dtype)
    WVVov = np.zeros((nkpts,nkpts,nkpts,nvirb,nvirb,nocca,nvira),dtype=t1a.dtype)
    WVVOV = np.zeros((nkpts,nkpts,nkpts,nvirb,nvirb,noccb,nvirb),dtype=t1a.dtype)

    for kn, km, ke in itertools.product(range(nkpts),repeat=3):
        kf = kconserv[kn, ke, km]
        ka = kn
        Wvvov[ka,ke,km] += eris.vovv[kf,km,ke].transpose(3,2,1,0).conj() - eris.vovv[ke,km,kf].transpose(3,0,1,2).conj()
        WVVov[ka,ke,km] += eris.voVV[kf,km,ke].transpose(3,2,1,0).conj()
        WvvOV[ka,ke,km] += eris.VOvv[kf,km,ke].transpose(3,2,1,0).conj()
        WVVOV[ka,ke,km] += eris.VOVV[kf,km,ke].transpose(3,2,1,0).conj() - eris.VOVV[ke,km,kf].transpose(3,0,1,2).conj()

        ovov = eris.ovov[kn, ke, km] - eris.ovov[kn, kf, km].transpose(0,3,2,1)
        OVOV = eris.OVOV[kn, ke, km] - eris.OVOV[kn, kf, km].transpose(0,3,2,1)

        Wvvov[ka,ke,km] += -lib.einsum('na,nemf->aemf',t1a[kn],ovov)
        WvvOV[ka,ke,km] += -lib.einsum('na,neMF->aeMF',t1a[kn],eris.ovOV[kn,ke,km])
        WVVov[ka,ke,km] += -lib.einsum('NA,NEmf->AEmf',t1b[kn],eris.OVov[kn,ke,km])
        WVVOV[ka,ke,km] += -lib.einsum('NA,NEMF->AEMF',t1b[kn],OVOV)

    return Wvvov, WvvOV, WVVov, WVVOV

def Wvvvo(cc,t1,t2,eris):
    kconserv = cc.khelper.kconserv
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocca, noccb, nvira, nvirb = t2ab.shape[2:]

    fova, fovb = cc_Fov(cc, t1, t2, eris)
    tauaa, tauab, taubb = make_tau(cc, t2, t1, t1)

    Wvvvo = np.zeros((nkpts,nkpts,nkpts,nvira,nvira,nvira,nocca),dtype=t1a.dtype)
    WvvVO = np.zeros((nkpts,nkpts,nkpts,nvira,nvira,nvirb,noccb),dtype=t1a.dtype)
    WVVvo = np.zeros((nkpts,nkpts,nkpts,nvirb,nvirb,nvira,nocca),dtype=t1a.dtype)
    WVVVO = np.zeros((nkpts,nkpts,nkpts,nvirb,nvirb,nvirb,noccb),dtype=t1a.dtype)

    for ka, ke, kb in itertools.product(range(nkpts),repeat=3):
        ki = kconserv[ka, ke, kb]
        # - <mb||ef> t2(miaf)
        for km in range(nkpts):
            kf = kconserv[km,ke,kb]
            ovvv = eris.vovv[ke,km,kf].transpose(1,0,3,2).conj() - eris.vovv[kf,km,ke].transpose(1,2,3,0).conj()
            OVvv = eris.VOvv[ke,km,kf].transpose(1,0,3,2).conj()
            ovVV = eris.voVV[ke,km,kf].transpose(1,0,3,2).conj()
            OVVV = eris.VOVV[ke,km,kf].transpose(1,0,3,2).conj() - eris.VOVV[kf,km,ke].transpose(1,2,3,0).conj()

            aebi = lib.einsum('mebf,miaf->aebi',ovvv,t2aa[km,ki,ka])
            aebi += lib.einsum('MFbe,iMaF->aebi',eris.VOvv[kf,km,ke].transpose(1,0,3,2).conj(),t2ab[ki,km,ka])
            Wvvvo[ka,ke,kb] -= aebi
            # P(ab) for all alpha spin
            Wvvvo[kb,ke,ka] += aebi.transpose(2,1,0,3)

            WVVvo[ka,ke,kb] -= lib.einsum('MEbf,iMfA->AEbi',OVvv,t2ab[ki,km,kf])
            WvvVO[ka,ke,kb] -= lib.einsum('meBF,mIaF->aeBI',ovVV,t2ab[km,ki,ka])

            AEBI = lib.einsum('MEBF,MIAF->AEBI',OVVV,t2bb[km,ki,ka])
            AEBI += lib.einsum('mfBE,mIfA->AEBI',eris.voVV[kf,km,ke].transpose(1,0,3,2).conj(),t2ab[km,ki,kf])
            WVVVO[ka,ke,kb] -= AEBI
            # P(ab) for all beta spin
            WVVVO[kb,ke,ka] += AEBI.transpose(2,1,0,3)

        # - t1(ma) (<mb||ei> - t2(nibf) <mn||ef>)
        km = ka
        ovvo = eris.voov[ke,km,ki].transpose(1,0,3,2).conj() - eris.oovv[km,ki,kb].transpose(0,3,2,1)
        OVvo = eris.VOov[ke,km,ki].transpose(1,0,3,2).conj()
        ovVO = eris.voOV[ke,km,ki].transpose(1,0,3,2).conj()
        OVVO = eris.VOOV[ke,km,ki].transpose(1,0,3,2).conj() - eris.OOVV[km,ki,kb].transpose(0,3,2,1)

        tmp1aa = np.zeros((nocca, nvira, nvira, nocca),dtype=t1a.dtype)
        tmp1ab = np.zeros((nocca, nvira, nvirb, noccb),dtype=t1a.dtype)
        tmp1ba = np.zeros((noccb, nvirb, nvira, nocca),dtype=t1a.dtype)
        tmp1bb = np.zeros((noccb, nvirb, nvirb, noccb),dtype=t1a.dtype)

        for kn in range(nkpts):
            kf = kconserv[km,ke,kn]
            ovov = eris.ovov[km,ke,kn] - eris.ovov[km,kf,kn].transpose(0,3,2,1)
            OVov = eris.OVov[km,ke,kn]
            ovOV = eris.ovOV[km,ke,kn]
            OVOV = eris.OVOV[km,ke,kn] - eris.OVOV[km,kf,kn].transpose(0,3,2,1)

            tmp1aa -= einsum('nibf,menf->mebi',t2aa[kn,ki,kb], ovov)
            tmp1aa += einsum('iNbF,meNF->mebi',t2ab[ki,kn,kb], ovOV)

            tmp1ab += einsum('nIfB,menf->meBI',t2ab[kn,ki,kf], ovov)
            tmp1ab -= einsum('NIBF,meNF->meBI',t2bb[kn,ki,kb], ovOV)

            tmp1ba += einsum('iNbF,MENF->MEbi',t2ab[ki,kn,kb], OVOV)
            tmp1ba -= einsum('nibf,MEnf->MEbi',t2aa[kn,ki,kb], OVov)

            tmp1bb -= einsum('NIBF,MENF->MEBI',t2bb[kn,ki,kb], OVOV)
            tmp1bb += einsum('nIfB,MEnf->MEBI',t2ab[kn,ki,kf], OVov)

        aebi = einsum('ma,mebi->aebi',t1a[km],(ovvo+tmp1aa))
        Wvvvo[ka,ke,kb] -= aebi

        WVVvo[ka,ke,kb] -= einsum('MA,MEbi->AEbi',t1b[km],OVvo+tmp1ba)
        WvvVO[ka,ke,kb] -= einsum('ma,meBI->aeBI',t1a[km],ovVO+tmp1ab)

        AEBI = einsum('MA,MEBI->AEBI',t1b[km],(OVVO+tmp1bb))
        WVVVO[ka,ke,kb] -= AEBI


    for ka, ke, kb in itertools.product(range(nkpts),repeat=3):
        ki = kconserv[ka, ke, kb]
        # P(ab) <mb||ef> t2(miaf) (alpha alpha beta beta) and (beta beta alpha alpha)
        for km in range(nkpts):
            kf = kconserv[km,ke,ka]
            OVVV = (eris.VOVV[ke,km,kf].transpose(1,0,3,2).conj() -
                    eris.VOVV[kf,km,ke].transpose(1,2,3,0).conj())
            ovvv = (eris.vovv[ke,km,kf].transpose(1,0,3,2).conj() -
                    eris.vovv[kf,km,ke].transpose(1,2,3,0).conj())

            WVVvo[ka,ke,kb] -= lib.einsum('mfAE,mibf->AEbi',
                                          eris.voVV[kf,km,ke].transpose(1,0,3,2).conj(), t2aa[km,ki,kb])
            WVVvo[ka,ke,kb] -= lib.einsum('MEAF,iMbF->AEbi', OVVV, t2ab[ki,km,kb])

            WvvVO[ka,ke,kb] -= lib.einsum('MFae,MIBF->aeBI',
                                          eris.VOvv[kf,km,ke].transpose(1,0,3,2).conj(), t2bb[km,ki,kb])
            WvvVO[ka,ke,kb] -= lib.einsum('meaf,mIfB->aeBI', ovvv, t2ab[km,ki,kf])

        # P(ab) -t1(ma) (<mb||ei> - t2(nibf) <mn||ef>) for all spin configurations
        km = kb
        ovvo = (eris.voov[ke,km,ki].transpose(1,0,3,2).conj() -
                eris.oovv[km,ki,ka].transpose(0,3,2,1))
        OVVO = (eris.VOOV[ke,km,ki].transpose(1,0,3,2).conj() -
                eris.OOVV[km,ki,ka].transpose(0,3,2,1))

        tmp1aa = np.zeros((nocca, nvira, nvira, nocca),dtype=t1a.dtype)
        tmp1ab = np.zeros((noccb, nvira, nvira, noccb),dtype=t1a.dtype)
        tmp1ba = np.zeros((nocca, nvirb, nvirb, nocca),dtype=t1a.dtype)
        tmp1bb = np.zeros((noccb, nvirb, nvirb, noccb),dtype=t1a.dtype)

        for kn in range(nkpts):
            kf = kconserv[km,ke,kn]
            ovov = eris.ovov[km,ke,kn] - eris.ovov[km,kf,kn].transpose(0,3,2,1)
            OVov = eris.OVov[km,ke,kn]
            ovOV = eris.ovOV[km,ke,kn]
            OVOV = eris.OVOV[km,ke,kn] - eris.OVOV[km,kf,kn].transpose(0,3,2,1)

            tmp1aa -= einsum('niaf,menf->meai',t2aa[kn,ki,ka], ovov)
            tmp1aa += einsum('iNaF,meNF->meai',t2ab[ki,kn,ka], ovOV)

            tmp1ab += einsum('nIaF,MFne->MeaI',t2ab[kn,ki,ka], eris.OVov[km,kf,kn])

            tmp1ba += einsum('iNfA,mfNE->mEAi',t2ab[ki,kn,kf], eris.ovOV[km,kf,kn])

            tmp1bb -= einsum('NIAF,MENF->MEAI',t2bb[kn,ki,ka], OVOV)
            tmp1bb += einsum('nIfA,MEnf->MEAI',t2ab[kn,ki,kf], OVov)

        aebi = einsum('mb,meai->aebi',t1a[km],(ovvo+tmp1aa))
        Wvvvo[ka,ke,kb] += aebi

        WVVvo[ka,ke,kb] += einsum('mb,mEAi->AEbi',t1a[km], -eris.ooVV[km,ki,ka].transpose(0,3,2,1)+tmp1ba)
        WvvVO[ka,ke,kb] += einsum('MB,MeaI->aeBI',t1b[km], -eris.OOvv[km,ki,ka].transpose(0,3,2,1)+tmp1ab)

        AEBI = einsum('MB,MEAI->AEBI',t1b[km],(OVVO+tmp1bb))
        WVVVO[ka,ke,kb] += AEBI

    # Remaining terms
    for ka, ke, kb in itertools.product(range(nkpts),repeat=3):
        ki = kf = kconserv[ka, ke, kb]
        Wvvvo[ka,ke,kb] += eris.vovv[kb,ki,ka].transpose(2,3,0,1) - eris.vovv[ka,ki,kb].transpose(0,3,2,1)
        WVVvo[ka,ke,kb] += eris.voVV[kb,ki,ka].transpose(2,3,0,1)
        WvvVO[ka,ke,kb] += eris.VOvv[kb,ki,ka].transpose(2,3,0,1)
        WVVVO[ka,ke,kb] += eris.VOVV[kb,ki,ka].transpose(2,3,0,1) - eris.VOVV[ka,ki,kb].transpose(0,3,2,1)

        Wvvvo[ka,ke,kb] -= lib.einsum('me,miab->aebi',fova[ke],t2aa[ke,ki,ka])
        WVVvo[ka,ke,kb] -= lib.einsum('ME,iMbA->AEbi',fovb[ke],t2ab[ki,ke,kb])
        WvvVO[ka,ke,kb] -= lib.einsum('me,mIaB->aeBI',fova[ke],t2ab[ke,ki,ka])
        WVVVO[ka,ke,kb] -= lib.einsum('ME,MIAB->AEBI',fovb[ke],t2bb[ke,ki,ka])

        Wvvvv, WvvVV, WVVVV = get_Wvvvv(cc, t1, t2, eris, ka, kb, ke)
        Wvvvo[ka,ke,kb] += lib.einsum('if,aebf->aebi', t1a[ki], Wvvvv)
        WVVvo[kb,kf,ka] += lib.einsum('ie,aeBF->BFai', t1a[ke], WvvVV)
        WvvVO[ka,ke,kb] += lib.einsum('IF,aeBF->aeBI', t1b[ki], WvvVV)
        WVVVO[ka,ke,kb] += lib.einsum('IF,AEBF->AEBI', t1b[ki], WVVVV)

        for km in range(nkpts):
            kn = kconserv[ka,km,kb]
            ovoo = eris.ooov[kn,ki,km].transpose(2,3,0,1) - eris.ooov[km,ki,kn].transpose(0,3,2,1)
            ovOO = eris.OOov[kn,ki,km].transpose(2,3,0,1)

            ooOV = eris.ooOV[km,ki,kn]
            OOov = eris.OOov[km,ki,kn]

            OVoo = eris.ooOV[kn,ki,km].transpose(2,3,0,1)
            OVOO = eris.OOOV[kn,ki,km].transpose(2,3,0,1) - eris.OOOV[km,ki,kn].transpose(0,3,2,1)

            Wvvvo[ka,ke,kb] += 0.5*lib.einsum('meni,mnab->aebi',ovoo,tauaa[km,kn,ka])

            WVVvo[ka,ke,kb] += 0.5*lib.einsum('MEni,nMbA->AEbi',OVoo,tauab[kn,km,kb])
            WVVvo[ka,ke,kb] += 0.5*lib.einsum('miNE,mNbA->AEbi',ooOV,tauab[km,kn,kb])

            WvvVO[ka,ke,kb] += 0.5*lib.einsum('meNI,mNaB->aeBI',ovOO,tauab[km,kn,ka])
            WvvVO[ka,ke,kb] += 0.5*lib.einsum('MIne,nMaB->aeBI',OOov,tauab[kn,km,ka])

            WVVVO[ka,ke,kb] += 0.5*lib.einsum('MENI,MNAB->AEBI',OVOO,taubb[km,kn,ka])

    return Wvvvo, WvvVO, WVVvo, WVVVO


def Woooo(cc,t1,t2,eris):
    kconserv = cc.khelper.kconserv
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    _, _, nkpts, nocca, noccb, nvira, nvirb = t2ab.shape

    dtype = np.result_type(*t2)
    Woooo = np.zeros(eris.oooo.shape, dtype=dtype)
    WooOO = np.zeros(eris.ooOO.shape, dtype=dtype)
    WOOOO = np.zeros(eris.OOOO.shape, dtype=dtype)

    tau_aa, tau_ab, tau_bb = make_tau(cc, t2, t1, t1)
    for km in range(nkpts):
        for kn in range(nkpts):
            tmp_aaaaJ = einsum('xje, ymine->yxminj', t1a, eris.ooov[km,:,kn])
            tmp_aaaaJ-= einsum('yie, xmjne->yxminj', t1a, eris.ooov[km,:,kn])
            tmp_bbbbJ = einsum('xje, ymine->yxminj', t1b, eris.OOOV[km,:,kn])
            tmp_bbbbJ-= einsum('yie, xmjne->yxminj', t1b, eris.OOOV[km,:,kn])
            #tmp_aabbJ = einsum('xje, ymine->yxminj', t1b, eris.ooOV[km,:,kn])
            #tmp_bbaaJ = einsum('xje, ymine->yxminj', t1a, eris.OOov[km,:,kn])
            #tmp_abbaJ = -einsum('yie,xmjne->yxminj', t1b, eris.ooOV[km,:,kn])
            tmp_baabJ = -einsum('yie,xmjne->yxminj', t1a, eris.OOov[km,:,kn])
            tmp_aabbJ = einsum('xje, ymine->yxminj', t1b, eris.ooOV[km,:,kn])

            for ki in range(nkpts):
                kj = kconserv[km,ki,kn]
                Woooo[km,ki,kn] += tmp_aaaaJ[ki,kj]
                WOOOO[km,ki,kn] += tmp_bbbbJ[ki,kj]
                WooOO[km,ki,kn] += tmp_aabbJ[ki,kj]
                WooOO[kn,ki,km] -= tmp_baabJ[ki,kj].transpose(2,1,0,3)
                Woooo[km,ki,kn] += eris.oooo[km,ki,kn]
                WooOO[km,ki,kn] += eris.ooOO[km,ki,kn]
                WOOOO[km,ki,kn] += eris.OOOO[km,ki,kn]

    Woooo = Woooo - Woooo.transpose(2,1,0,5,4,3,6)
    WOOOO = WOOOO - WOOOO.transpose(2,1,0,5,4,3,6)

    for km, ki, kn in itertools.product(range(nkpts), repeat=3):
        kj = kconserv[km, ki, kn]

        for ke in range(nkpts):
            kf = kconserv[km, ke, kn]

            ovov = eris.ovov[km, ke, kn] - eris.ovov[km, kf, kn].transpose(0,3,2,1)
            OVOV = eris.OVOV[km, ke, kn] - eris.OVOV[km, kf, kn].transpose(0,3,2,1)

            Woooo[km, ki, kn] += 0.5*lib.einsum('ijef,menf->minj', tau_aa[ki, kj, ke],      ovov)
            WOOOO[km, ki, kn] += 0.5*lib.einsum('IJEF,MENF->MINJ', tau_bb[ki, kj, ke],      OVOV)
            WooOO[km, ki, kn] +=     lib.einsum('iJeF,meNF->miNJ', tau_ab[ki, kj, ke], eris.ovOV[km, ke, kn])

    WOOoo = None
    return Woooo, WooOO, WOOoo, WOOOO

def Woovo(cc,t1,t2,eris):
    kconserv = cc.khelper.kconserv
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocca, noccb, nvira, nvirb = t2ab.shape[2:]

    dtype = np.result_type(*t2)
    Woovo = np.zeros((nkpts, nkpts, nkpts, nocca, nocca, nvira, nocca), dtype=dtype)
    WooVO = np.zeros((nkpts, nkpts, nkpts, nocca, nocca, nvirb, noccb), dtype=dtype)
    WOOvo = np.zeros((nkpts, nkpts, nkpts, noccb, noccb, nvira, nocca), dtype=dtype)
    WOOVO = np.zeros((nkpts, nkpts, nkpts, noccb, noccb, nvirb, noccb), dtype=dtype)

    for km, kb, ki in kpts_helper.loop_kkk(nkpts):
        kj = kconserv[km, ki, kb]
        Woovo[km,ki,kb] += eris.ooov[ki,km,kj].transpose(1,0,3,2).conj()
        Woovo[km,ki,kb] -= eris.ooov[kj,km,ki].transpose(1,2,3,0).conj()
        WooVO[km,ki,kb] += eris.ooOV[ki,km,kj].transpose(1,0,3,2).conj()
        WOOvo[km,ki,kb] += eris.OOov[ki,km,kj].transpose(1,0,3,2).conj()
        WOOVO[km,ki,kb] += eris.OOOV[ki,km,kj].transpose(1,0,3,2).conj()
        WOOVO[km,ki,kb] -= eris.OOOV[kj,km,ki].transpose(1,2,3,0).conj()
        for kn in range(nkpts):
            ke = kconserv[km,ki,kn]
            ooov = eris.ooov[km,ki,kn] - eris.ooov[kn,ki,km].transpose(2,1,0,3)
            OOOV = eris.OOOV[km,ki,kn] - eris.OOOV[kn,ki,km].transpose(2,1,0,3)

            Woovo[km,ki,kb] += einsum('mine,jnbe->mibj', ooov, t2aa[kj,kn,kb])
            Woovo[km,ki,kb] += einsum('miNE,jNbE->mibj', eris.ooOV[km,ki,kn], t2ab[kj,kn,kb])
            WooVO[km,ki,kb] += einsum('mine,nJeB->miBJ', ooov, t2ab[kn,kj,ke])
            WooVO[km,ki,kb] += einsum('miNE,JNBE->miBJ', eris.ooOV[km,ki,kn], t2bb[kj,kn,kb])
            WOOvo[km,ki,kb] += einsum('MINE,jNbE->MIbj', OOOV, t2ab[kj,kn,kb])
            WOOvo[km,ki,kb] += einsum('MIne,jnbe->MIbj', eris.OOov[km,ki,kn], t2aa[kj,kn,kb])
            WOOVO[km,ki,kb] += einsum('MINE,JNBE->MIBJ', OOOV, t2bb[kj,kn,kb])
            WOOVO[km,ki,kb] += einsum('MIne,nJeB->MIBJ', eris.OOov[km,ki,kn], t2ab[kn,kj,ke])
            # P(ij)
            ke = kconserv[km,kj,kn]
            ooov = eris.ooov[km,kj,kn] - eris.ooov[kn,kj,km].transpose(2,1,0,3)
            OOOV = eris.OOOV[km,kj,kn] - eris.OOOV[kn,kj,km].transpose(2,1,0,3)

            Woovo[km,ki,kb] -= einsum('mjne,inbe->mibj', ooov, t2aa[ki,kn,kb])
            Woovo[km,ki,kb] -= einsum('mjNE,iNbE->mibj', eris.ooOV[km,kj,kn], t2ab[ki,kn,kb])
            WooVO[km,ki,kb] -= einsum('NJme,iNeB->miBJ', eris.OOov[kn,kj,km], t2ab[ki,kn,ke])
            WOOvo[km,ki,kb] -= einsum('njME,nIbE->MIbj', eris.ooOV[kn,kj,km], t2ab[kn,ki,kb])
            WOOVO[km,ki,kb] -= einsum('MJNE,INBE->MIBJ', OOOV, t2bb[ki,kn,kb])
            WOOVO[km,ki,kb] -= einsum('MJne,nIeB->MIBJ', eris.OOov[km,kj,kn], t2ab[kn,ki,ke])

        ovvo = eris.voov[ki,km,kj].transpose(1,0,3,2).conj() - eris.oovv[km,kj,kb].transpose(0,3,2,1)
        OVVO = eris.VOOV[ki,km,kj].transpose(1,0,3,2).conj() - eris.OOVV[km,kj,kb].transpose(0,3,2,1)
        ovVO = eris.voOV[ki,km,kj].transpose(1,0,3,2).conj()
        OVvo = eris.VOov[ki,km,kj].transpose(1,0,3,2).conj()
        Woovo[km,ki,kb] += einsum('ie,mebj->mibj', t1a[ki], ovvo)
        WooVO[km,ki,kb] += einsum('ie,meBJ->miBJ', t1a[ki], ovVO)
        WOOvo[km,ki,kb] += einsum('IE,MEbj->MIbj', t1b[ki], OVvo)
        WOOVO[km,ki,kb] += einsum('IE,MEBJ->MIBJ', t1b[ki], OVVO)
        #P(ij)
        ovvo = eris.voov[kj,km,ki].transpose(1,0,3,2).conj() - eris.oovv[km,ki,kb].transpose(0,3,2,1)
        OVVO = eris.VOOV[kj,km,ki].transpose(1,0,3,2).conj() - eris.OOVV[km,ki,kb].transpose(0,3,2,1)
        Woovo[km,ki,kb] -= einsum('je,mebi->mibj', t1a[kj], ovvo)
        WooVO[km,ki,kb] -= -einsum('JE,miBE->miBJ', t1b[kj], eris.ooVV[km,ki,kb])
        WOOvo[km,ki,kb] -= -einsum('je,MIbe->MIbj', t1a[kj], eris.OOvv[km,ki,kb])
        WOOVO[km,ki,kb] -= einsum('JE,MEBI->MIBJ', t1b[kj], OVVO)


        for kf in range(nkpts):
            kn = kconserv[kb, kj, kf]
            ovov = eris.ovov[km,ki,kn] - eris.ovov[km,kf,kn].transpose(0,3,2,1)
            OVOV = eris.OVOV[km,ki,kn] - eris.OVOV[km,kf,kn].transpose(0,3,2,1)
            Woovo[km,ki,kb] -= (einsum('ie,njbf,menf->mibj', t1a[ki], t2aa[kn,kj,kb], ovov) -
                                einsum('ie,jNbF,meNF->mibj', t1a[ki], t2ab[kj,kn,kb], eris.ovOV[km,ki,kn]))
            WooVO[km,ki,kb] -= (-einsum('ie,nJfB,menf->miBJ', t1a[ki], t2ab[kn,kj,kf], ovov) +
                                einsum('ie,NJBF,meNF->miBJ', t1a[ki], t2bb[kn,kj,kb], eris.ovOV[km,ki,kn]))
            WOOvo[km,ki,kb] -= (-einsum('IE,jNbF,MENF->MIbj', t1b[ki], t2ab[kj,kn,kb], OVOV) +
                                einsum('IE,njbf,MEnf->MIbj', t1b[ki], t2aa[kn,kj,kb], eris.OVov[km,ki,kn]))
            WOOVO[km,ki,kb] -= (einsum('IE,NJBF,MENF->MIBJ', t1b[ki], t2bb[kn,kj,kb], OVOV) -
                                einsum('IE,nJfB,MEnf->MIBJ', t1b[ki], t2ab[kn,kj,kf], eris.OVov[km,ki,kn]))
            #P(ij)
            kn = kconserv[kb, ki, kf]
            ovov = eris.ovov[km,kj,kn] - eris.ovov[km,kf,kn].transpose(0,3,2,1)
            OVOV = eris.OVOV[km,kj,kn] - eris.OVOV[km,kf,kn].transpose(0,3,2,1)
            Woovo[km,ki,kb] += (einsum('je,nibf,menf->mibj', t1a[kj], t2aa[kn,ki,kb], ovov) -
                                einsum('je,iNbF,meNF->mibj', t1a[kj], t2ab[ki,kn,kb], eris.ovOV[km,kj,kn]))
            WooVO[km,ki,kb] += -einsum('JE,iNfB,mfNE->miBJ', t1b[kj], t2ab[ki,kn,kf], eris.ovOV[km, kf, kn])
            WOOvo[km,ki,kb] += -einsum('je,nIbF,MFne->MIbj', t1a[kj], t2ab[kn,ki,kb], eris.OVov[km, kf, kn])
            WOOVO[km,ki,kb] += (einsum('JE,NIBF,MENF->MIBJ', t1b[kj], t2bb[kn,ki,kb], OVOV) -
                                einsum('JE,nIfB,MEnf->MIBJ', t1b[kj], t2ab[kn,ki,kf], eris.OVov[km,kj,kn]))

    Fme, FME = Fov(cc, t1, t2, eris)
    Wminj, WmiNJ, WMInj, WMINJ = Woooo(cc,t1,t2,eris)
    tauaa, tauab, taubb = make_tau(cc, t2, t1, t1, fac=1.)
    for km, kb, ki in kpts_helper.loop_kkk(nkpts):
        kj = kconserv[km, ki, kb]

        Woovo[km,ki,kb] -= einsum('me,ijbe->mibj', Fme[km], t2aa[ki,kj,kb])
        WooVO[km,ki,kb] -= -einsum('me,iJeB->miBJ', Fme[km], t2ab[ki,kj,km])
        WOOvo[km,ki,kb] -= -einsum('ME,jIbE->MIbj', FME[km], t2ab[kj,ki,kb])
        WOOVO[km,ki,kb] -= einsum('ME,IJBE->MIBJ', FME[km], t2bb[ki,kj,kb])

        Woovo[km,ki,kb] -= einsum('nb, minj->mibj', t1a[kb], Wminj[km, ki, kb])
        WooVO[km,ki,kb] -= einsum('NB, miNJ->miBJ', t1b[kb], WmiNJ[km, ki, kb])
        WOOvo[km,ki,kb] -= einsum('nb, njMI->MIbj', t1a[kb], WmiNJ[kb, kj, km])
        WOOVO[km,ki,kb] -= einsum('NB, MINJ->MIBJ', t1b[kb], WMINJ[km, ki, kb])

    for km, kb, ki in kpts_helper.loop_kkk(nkpts):
        kj = kconserv[km, ki, kb]
        for ke in range(nkpts):
            kf = kconserv[km,ke,kb]
            ovvv = eris.vovv[ke,km,kf].transpose(1,0,3,2).conj() - eris.vovv[kf,km,ke].transpose(1,2,3,0).conj()
            OVVV = eris.VOVV[ke,km,kf].transpose(1,0,3,2).conj() - eris.VOVV[kf,km,ke].transpose(1,2,3,0).conj()
            ovVV = eris.voVV[ke,km,kf].transpose(1,0,3,2).conj()
            OVvv = eris.VOvv[ke,km,kf].transpose(1,0,3,2).conj()
            Woovo[km,ki,kb] += 0.5 * einsum('mebf,ijef->mibj', ovvv, tauaa[ki,kj,ke])
            WooVO[km,ki,kb] += einsum('meBF,iJeF->miBJ', ovVV, tauab[ki,kj,ke])
            WOOvo[km,ki,kb] += einsum('MEbf,jIfE->MIbj', OVvv, tauab[kj,ki,kf])
            WOOVO[km,ki,kb] += 0.5 * einsum('MEBF,IJEF->MIBJ', OVVV, taubb[ki,kj,ke])

    return Woovo, WooVO, WOOvo, WOOVO


def Wooov(cc, t1, t2, eris, kconserv):
    t1a, t1b = t1

    Wooov = eris.ooov - np.asarray(eris.ooov).transpose(2,1,0,5,4,3,6)
    WooOV = np.array(eris.ooOV, copy=True)
    WOOov = np.array(eris.OOov, copy=True)
    WOOOV = eris.OOOV - np.asarray(eris.OOOV).transpose(2,1,0,5,4,3,6)

    Wooov += einsum('yif,xyzmfne->xyzmine', t1a, eris.ovov) - einsum('yif, zyxnfme->xyzmine', t1a, eris.ovov)
    WooOV += einsum('yif,xyzmfNE->xyzmiNE', t1a, eris.ovOV)
    WOOov += einsum('yIF,xyzMFne->xyzMIne', t1b, eris.OVov)
    WOOOV += einsum('yIF,xyzMFNE->xyzMINE', t1b, eris.OVOV) - einsum('yIF, zyxNFME->xyzMINE', t1b, eris.OVOV)

    return Wooov, WooOV, WOOov, WOOOV

def Wovvo(cc, t1, t2, eris):
    kconserv = cc.khelper.kconserv
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocca, noccb, nvira, nvirb = t2ab.shape[2:]

    Wovvo, WovVO, WOVvo, WOVVO, WoVVo, WOvvO = cc_Wovvo(cc,t1,t2,eris)
    for km, kb, ke in kpts_helper.loop_kkk(nkpts):
        kj = kconserv[km, ke, kb]
        for kn in range(nkpts):
            kf = kconserv[km, ke, kn]
            Wovvo[km,ke,kb] += 0.5 * einsum('jnbf,menf->mebj', t2aa[kj,kn,kb], eris.ovov[km,ke,kn])
            Wovvo[km,ke,kb] -= 0.5 * einsum('jnbf,mfne->mebj', t2aa[kj,kn,kb], eris.ovov[km,kf,kn])
            Wovvo[km,ke,kb] += 0.5 * einsum('jNbF,meNF->mebj', t2ab[kj,kn,kb], eris.ovOV[km,ke,kn])

            WOVvo[km,ke,kb] += 0.5 * einsum('jNbF,MENF->MEbj', t2ab[kj,kn,kb], eris.OVOV[km,ke,kn])
            WOVvo[km,ke,kb] -= 0.5 * einsum('jNbF,MFNE->MEbj', t2ab[kj,kn,kb], eris.OVOV[km,kf,kn])
            WOVvo[km,ke,kb] += 0.5 * einsum('jnbf,MEnf->MEbj', t2aa[kj,kn,kb], eris.OVov[km,ke,kn])

            WovVO[km,ke,kb] += 0.5 * einsum('nJfB,menf->meBJ', t2ab[kn,kj,kf], eris.ovov[km,ke,kn])
            WovVO[km,ke,kb] -= 0.5 * einsum('nJfB,mfne->meBJ', t2ab[kn,kj,kf], eris.ovov[km,kf,kn])
            WovVO[km,ke,kb] += 0.5 * einsum('JNBF,meNF->meBJ', t2bb[kj,kn,kb], eris.ovOV[km,ke,kn])

            WOVVO[km,ke,kb] += 0.5 * einsum('JNBF,MENF->MEBJ', t2bb[kj,kn,kb], eris.OVOV[km,ke,kn])
            WOVVO[km,ke,kb] -= 0.5 * einsum('JNBF,MFNE->MEBJ', t2bb[kj,kn,kb], eris.OVOV[km,kf,kn])
            WOVVO[km,ke,kb] += 0.5 * einsum('nJfB,MEnf->MEBJ', t2ab[kn,kj,kf], eris.OVov[km,ke,kn])

    return Wovvo, WovVO, WOVvo, WOVVO

def W1oovv(cc, t1, t2, eris):
    kconserv = cc.khelper.kconserv
    dtype = np.result_type(*t1)
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocc, nvir = t1a.shape
    Woovv = np.zeros(eris.oovv.shape, dtype=dtype)
    WooVV = np.zeros(eris.ooVV.shape, dtype=dtype)
    WOOvv = np.zeros(eris.OOvv.shape, dtype=dtype)
    WOOVV = np.zeros(eris.OOVV.shape, dtype=dtype)
    for kk in range(nkpts):
        for ki in range(nkpts):
            for kb in range(nkpts):
                kd = kconserv[kk,ki,kb]
                Woovv[kk,ki,kb] += eris.oovv[kk,ki,kb]
                Woovv[kk,ki,kb] -= eris.voov[kb,ki,kk].transpose(2,1,0,3)
                WooVV[kk,ki,kb] += eris.ooVV[kk,ki,kb]
                WOOvv[kk,ki,kb] += eris.OOvv[kk,ki,kb]
                WOOVV[kk,ki,kb] += eris.OOVV[kk,ki,kb]
                WOOVV[kk,ki,kb] -= eris.VOOV[kb,ki,kk].transpose(2,1,0,3)

                for kl in range(nkpts):
                    kc = kconserv[ki,kb,kl]
                    Woovv[kk,ki,kb] -= einsum('lckd,ilbc->kibd', eris.ovov[kl,kc,kk], t2aa[ki,kl,kb])
                    Woovv[kk,ki,kb] += einsum('ldkc,ilbc->kibd', eris.ovov[kl,kd,kk], t2aa[ki,kl,kb])
                    Woovv[kk,ki,kb] -= einsum('LCkd,iLbC->kibd', eris.OVov[kl,kc,kk], t2ab[ki,kl,kb])

                    WooVV[kk,ki,kb] -= einsum('kcLD,iLcB->kiBD', eris.ovOV[kk,kc,kl], t2ab[ki,kl,kc])
                    WOOvv[kk,ki,kb] -= einsum('KCld,lIbC->KIbd', eris.OVov[kk,kc,kl], t2ab[kl,ki,kb])

                    WOOVV[kk,ki,kb] -= einsum('LCKD,ILBC->KIBD', eris.OVOV[kl,kc,kk], t2bb[ki,kl,kb])
                    WOOVV[kk,ki,kb] += einsum('LDKC,ILBC->KIBD', eris.OVOV[kl,kd,kk], t2bb[ki,kl,kb])
                    WOOVV[kk,ki,kb] -= einsum('lcKD,lIcB->KIBD', eris.ovOV[kl,kc,kk], t2ab[kl,ki,kc])

    return Woovv, WooVV, WOOvv, WOOVV

def W2oovv(cc, t1, t2, eris):
    kconserv = cc.khelper.kconserv
    dtype = np.result_type(*t1)
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocc, nvir = t1a.shape
    Woovv = np.zeros(eris.oovv.shape, dtype=dtype)
    WooVV = np.zeros(eris.ooVV.shape, dtype=dtype)
    WOOvv = np.zeros(eris.OOvv.shape, dtype=dtype)
    WOOVV = np.zeros(eris.OOVV.shape, dtype=dtype)
    WWooov, WWooOV, WWOOov, WWOOOV = Wooov(cc, t1, t2, eris, kconserv)
    for kk in range(nkpts):
        for ki in range(nkpts):
            for kb in range(nkpts):
                kd = kconserv[kk,ki,kb]
                Woovv[kk,ki,kb] += einsum('kild,lb->kibd',WWooov[kk,ki,kb],-t1a[kb])
                WooVV[kk,ki,kb] += einsum('kiLD,LB->kiBD',WWooOV[kk,ki,kb],-t1b[kb])
                WOOvv[kk,ki,kb] += einsum('KIld,lb->KIbd',WWOOov[kk,ki,kb],-t1a[kb])
                WOOVV[kk,ki,kb] += einsum('KILD,LB->KIBD',WWOOOV[kk,ki,kb],-t1b[kb])

                Woovv[kk,ki,kb] += einsum('ckdb,ic->kibd', eris.vovv[ki,kk,kd].conj(), t1a[ki])
                Woovv[kk,ki,kb] -= einsum('dkcb,ic->kibd', eris.vovv[kd,kk,ki].conj(), t1a[ki])

                WooVV[kk,ki,kb] += einsum('ckDB,ic->kiBD', eris.voVV[ki,kk,kd].conj(), t1a[ki])
                WOOvv[kk,ki,kb] += einsum('CKdb,IC->KIbd', eris.VOvv[ki,kk,kd].conj(), t1b[ki])

                WOOVV[kk,ki,kb] += einsum('CKDB,IC->KIBD', eris.VOVV[ki,kk,kd].conj(), t1b[ki])
                WOOVV[kk,ki,kb] -= einsum('DKCB,IC->KIBD', eris.VOVV[kd,kk,ki].conj(), t1b[ki])

    return Woovv, WooVV, WOOvv, WOOVV

def Woovv(cc, t1, t2, eris):
    t1a, t1b = t1
    nkpts, nocc, nvir = t1a.shape
    Woovv, WooVV, WOOvv, WOOVV = W1oovv(cc, t1, t2, eris)
    WWoovv, WWooVV, WWOOvv, WWOOVV = W2oovv(cc, t1, t2, eris)
    for kk,ki,kb in itertools.product(range(nkpts), repeat=3):
        Woovv[kk,ki,kb] = Woovv[kk,ki,kb] + WWoovv[kk,ki,kb]
        WooVV[kk,ki,kb] = WooVV[kk,ki,kb] + WWooVV[kk,ki,kb]
        WOOvv[kk,ki,kb] = WOOvv[kk,ki,kb] + WWOOvv[kk,ki,kb]
        WOOVV[kk,ki,kb] = WOOVV[kk,ki,kb] + WWOOVV[kk,ki,kb]
    return Woovv, WooVV, WOOvv, WOOVV

# vvvv is a string, ('oooo', 'ooov', ..., 'vvvv')
# orbspin can be accessed through general spin-orbital kintermediates eris
# orbspin = eris.mo_coeff.orbspin
def _eri_spin2spatial(chemist_eri_spin, vvvv, eris, nocc, orbspin, cross_ab=False):
    nocc_a, nocc_b = nocc
    nocc = nocc_a + nocc_b
    nkpts = len(orbspin)
    idxoa = [np.where(orbspin[k][:nocc] == 0)[0] for k in range(nkpts)]
    idxob = [np.where(orbspin[k][:nocc] == 1)[0] for k in range(nkpts)]
    idxva = [np.where(orbspin[k][nocc:] == 0)[0] for k in range(nkpts)]
    idxvb = [np.where(orbspin[k][nocc:] == 1)[0] for k in range(nkpts)]

    def select_idx(s):
        if s.lower() == 'o':
            return idxoa, idxob
        else:
            return idxva, idxvb

    if len(vvvv) == 2:
        idx1a, idx1b = select_idx(vvvv[0])
        idx2a, idx2b = select_idx(vvvv[1])

        fa = np.zeros((nkpts,len(idx1a[0]),len(idx2a[0])), dtype=np.complex128)
        fb = np.zeros((nkpts,len(idx1b[0]),len(idx2b[0])), dtype=np.complex128)
        for k in range(nkpts):
            fa[k] = chemist_eri_spin[k, idx1a[k][:,None],idx2a[k]]
            fb[k] = chemist_eri_spin[k, idx1b[k][:,None],idx2b[k]]
        return fa, fb

    idx1a, idx1b = select_idx(vvvv[0])
    idx2a, idx2b = select_idx(vvvv[1])
    idx3a, idx3b = select_idx(vvvv[2])
    idx4a, idx4b = select_idx(vvvv[3])

    eri_aaaa = np.zeros((nkpts,nkpts,nkpts,len(idx1a[0]),len(idx2a[0]),len(idx3a[0]),len(idx4a[0])), dtype=np.complex128)  # noqa: E501
    eri_aabb = np.zeros((nkpts,nkpts,nkpts,len(idx1a[0]),len(idx2a[0]),len(idx3b[0]),len(idx4b[0])), dtype=np.complex128)  # noqa: E501
    eri_bbaa = np.zeros((nkpts,nkpts,nkpts,len(idx1b[0]),len(idx2b[0]),len(idx3a[0]),len(idx4a[0])), dtype=np.complex128)  # noqa: E501
    eri_bbbb = np.zeros((nkpts,nkpts,nkpts,len(idx1b[0]),len(idx2b[0]),len(idx3b[0]),len(idx4b[0])), dtype=np.complex128)  # noqa: E501
    if cross_ab:
        eri_abba = np.zeros((nkpts,nkpts,nkpts,len(idx1a[0]),len(idx2b[0]),len(idx3b[0]),len(idx4a[0])), dtype=np.complex128)  # noqa: E501
        eri_baab = np.zeros((nkpts,nkpts,nkpts,len(idx1b[0]),len(idx2a[0]),len(idx3a[0]),len(idx4b[0])), dtype=np.complex128)  # noqa: E501
    kconserv = kpts_helper.get_kconserv(eris.cell, eris.kpts)
    for ki, kj, kk in kpts_helper.loop_kkk(nkpts):
        kl = kconserv[ki, kj, kk]
        eri_aaaa[ki,kj,kk] = chemist_eri_spin[ki,kj,kk, idx1a[ki][:,None,None,None],idx2a[kj][:,None,None],idx3a[kk][:,None],idx4a[kl]]  # noqa: E501
        eri_aabb[ki,kj,kk] = chemist_eri_spin[ki,kj,kk, idx1a[ki][:,None,None,None],idx2a[kj][:,None,None],idx3b[kk][:,None],idx4b[kl]]  # noqa: E501
        eri_bbaa[ki,kj,kk] = chemist_eri_spin[ki,kj,kk, idx1b[ki][:,None,None,None],idx2b[kj][:,None,None],idx3a[kk][:,None],idx4a[kl]]  # noqa: E501
        eri_bbbb[ki,kj,kk] = chemist_eri_spin[ki,kj,kk, idx1b[ki][:,None,None,None],idx2b[kj][:,None,None],idx3b[kk][:,None],idx4b[kl]]  # noqa: E501
        if cross_ab:
            eri_abba[ki,kj,kk] = chemist_eri_spin[ki,kj,kk, idx1a[ki][:,None,None,None],idx2b[kj][:,None,None],idx3b[kk][:,None],idx4a[kl]]  # noqa: E501
            eri_baab[ki,kj,kk] = chemist_eri_spin[ki,kj,kk, idx1b[ki][:,None,None,None],idx2a[kj][:,None,None],idx3a[kk][:,None],idx4b[kl]]  # noqa: E501
    if cross_ab:
        return eri_aaaa, eri_aabb, eri_bbaa, eri_bbbb, eri_abba, eri_baab
    else:
        return eri_aaaa, eri_aabb, eri_bbaa, eri_bbbb

def _eri_spatial2spin(eri_aa_ab_ba_bb, vvvv, eris, orbspin, cross_ab=False):
    nocc_a, nocc_b = eris.nocc
    nocc = nocc_a + nocc_b
    nkpts = len(orbspin)
    idxoa = [np.where(orbspin[k][:nocc] == 0)[0] for k in range(nkpts)]
    idxob = [np.where(orbspin[k][:nocc] == 1)[0] for k in range(nkpts)]
    idxva = [np.where(orbspin[k][nocc:] == 0)[0] for k in range(nkpts)]
    idxvb = [np.where(orbspin[k][nocc:] == 1)[0] for k in range(nkpts)]

    def select_idx(s):
        if s.lower() == 'o':
            return idxoa, idxob
        else:
            return idxva, idxvb

    if len(vvvv) == 2:
        idx1a, idx1b = select_idx(vvvv[0])
        idx2a, idx2b = select_idx(vvvv[1])

        fa, fb = eri_aa_ab_ba_bb
        f = np.zeros((nkpts, len(idx1a[0])+len(idx1b[0]),
                      len(idx2a[0])+len(idx2b[0])), dtype=np.complex128)
        for k in range(nkpts):
            f[k, idx1a[k][:,None],idx2a[k]] = fa[k]
            f[k, idx1b[k][:,None],idx2b[k]] = fb[k]
        return f

    idx1a, idx1b = select_idx(vvvv[0])
    idx2a, idx2b = select_idx(vvvv[1])
    idx3a, idx3b = select_idx(vvvv[2])
    idx4a, idx4b = select_idx(vvvv[3])

    if cross_ab:
        eri_aaaa, eri_aabb, eri_bbaa, eri_bbbb, eri_abba, eri_baab = eri_aa_ab_ba_bb
    else:
        eri_aaaa, eri_aabb, eri_bbaa, eri_bbbb = eri_aa_ab_ba_bb
    eri = np.zeros((nkpts,nkpts,nkpts, len(idx1a[0])+len(idx1b[0]),
                    len(idx2a[0])+len(idx2b[0]),
                    len(idx3a[0])+len(idx3b[0]),
                    len(idx4a[0])+len(idx4b[0])), dtype=np.complex128)
    kconserv = kpts_helper.get_kconserv(eris.cell, eris.kpts)
    for ki, kj, kk in kpts_helper.loop_kkk(nkpts):
        kl = kconserv[ki, kj, kk]
        eri[ki,kj,kk, idx1a[ki][:,None,None,None],idx2a[kj][:,None,None],idx3a[kk][:,None],idx4a[kl]] = eri_aaaa[ki,kj,kk]  # noqa: E501
        eri[ki,kj,kk, idx1a[ki][:,None,None,None],idx2a[kj][:,None,None],idx3b[kk][:,None],idx4b[kl]] = eri_aabb[ki,kj,kk]  # noqa: E501
        eri[ki,kj,kk, idx1b[ki][:,None,None,None],idx2b[kj][:,None,None],idx3a[kk][:,None],idx4a[kl]] = eri_bbaa[ki,kj,kk]  # noqa: E501
        eri[ki,kj,kk, idx1b[ki][:,None,None,None],idx2b[kj][:,None,None],idx3b[kk][:,None],idx4b[kl]] = eri_bbbb[ki,kj,kk]  # noqa: E501
        if cross_ab:
            eri[ki,kj,kk, idx1a[ki][:,None,None,None],idx2b[kj][:,None,None],idx3b[kk][:,None],idx4a[kl]] = eri_abba[ki,kj,kk]  # noqa: E501
            eri[ki,kj,kk, idx1b[ki][:,None,None,None],idx2a[kj][:,None,None],idx3a[kk][:,None],idx4b[kl]] = eri_baab[ki,kj,kk]  # noqa: E501
    return eri
