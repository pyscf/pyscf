import itertools
import numpy as np
from functools import reduce

from pyscf import lib
from pyscf.pbc.lib import kpts_helper

einsum = lib.einsum

def make_tau(cc, t2, t1, t1p, fac=1.):
    t2aa, t2ab, t2bb = t2
    nkpts = len(t2aa)

    tauaa = t2aa.copy()
    tauab = t2ab.copy()
    taubb = t2bb.copy()
    for ki in range(nkpts):
        for kj in range(nkpts):
            tauaa[ki,kj,ki] += np.einsum('ia,jb->ijab', fac*.5*t1[0][ki], t1p[0][kj])
            tauaa[ki,kj,kj] -= np.einsum('ib,ja->ijab', fac*.5*t1[0][ki], t1p[0][kj])
            tauaa[ki,kj,kj] -= np.einsum('ja,ib->ijab', fac*.5*t1[0][kj], t1p[0][ki])
            tauaa[ki,kj,ki] += np.einsum('jb,ia->ijab', fac*.5*t1[0][kj], t1p[0][ki])

            taubb[ki,kj,ki] += np.einsum('ia,jb->ijab', fac*.5*t1[1][ki], t1p[1][kj])
            taubb[ki,kj,kj] -= np.einsum('ib,ja->ijab', fac*.5*t1[1][ki], t1p[1][kj])
            taubb[ki,kj,kj] -= np.einsum('ja,ib->ijab', fac*.5*t1[1][kj], t1p[1][ki])
            taubb[ki,kj,ki] += np.einsum('jb,ia->ijab', fac*.5*t1[1][kj], t1p[1][ki])

            tauab[ki,kj,ki] += np.einsum('ia,jb->ijab', fac*.5*t1[0][ki], t1p[1][kj])
            tauab[ki,kj,ki] += np.einsum('jb,ia->ijab', fac*.5*t1[1][kj], t1p[0][ki])
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
                fa[ka]-=0.5*einsum('mnaf,menf->ae',tau_tildeaa[km,kn,ka],eris.ovov[km,ka,kn]) # v
                fa[ka]-=0.5*einsum('mNaF,meNF->ae',tau_tildeab[km,kn,ka],eris.ovOV[km,ka,kn]) # v
                kf=kconserv[km,ka,kn]
                fa[ka]+=0.5*einsum('mnaf,mfne->ae',tau_tildeaa[km,kn,ka],eris.ovov[km,kf,kn]) # v
                fa[ka]-=0.5*einsum('nMaF,neMF->ae',tau_tildeab[kn,km,ka],eris.ovOV[kn,ka,km]) # c

                fb[ka]-=0.5*einsum('mnaf,menf->ae',tau_tildebb[km,kn,ka],eris.OVOV[km,ka,kn]) # v
                fb[ka]-=0.5*einsum('nmfa,nfme->ae',tau_tildeab[kn,km,kf],eris.ovOV[kn,kf,km]) # v
                kf=kconserv[km,ka,kn]
                fb[ka]+=0.5*einsum('mnaf,mfne->ae',tau_tildebb[km,kn,ka],eris.OVOV[km,kf,kn]) # v
                fb[ka]-=0.5*einsum('MnFa,MFne->ae',tau_tildeab[km,kn,kf],eris.ovOV[km,kf,kn]) # c

    return fa,fb #Fvv, FVV


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
                fa[km]+=0.5*einsum('inef,menf->mi',tau_tildeaa[km,kn,ke],eris.ovov[km,ke,kn]) # v
                fa[km]+=0.5*einsum('iNeF,meNF->mi',tau_tildeab[km,kn,ke],eris.ovOV[km,ke,kn]) # v
                kf=kconserv[km,ke,kn]
                fa[km]-=0.5*einsum('inef,mfne->mi',tau_tildeaa[km,kn,ke],eris.ovov[km,kf,kn]) # v
                fb[km]+=0.5*einsum('NiEf,NEmf->mi',tau_tildeab[kn,km,ke],eris.ovOV[kn,ke,km]) # c

                fb[km]+=0.5*einsum('INEF,MENF->MI',tau_tildebb[km,kn,ke],eris.OVOV[km,ke,kn]) # v
                fb[km]+=0.5*einsum('nIfE,nfME->MI',tau_tildeab[kn,km,kf],eris.ovOV[kn,kf,km]) # v
                kf=kconserv[km,ke,kn]
                fb[km]-=0.5*einsum('INEF,MFNE->MI',tau_tildebb[km,kn,ke],eris.OVOV[km,kf,kn]) # v
                fa[km]+=0.5*einsum('InFe,MFne->MI',tau_tildeab[km,kn,kf],eris.ovOV[km,kf,kn]) # c

    return fa,fb #Foo, FOO


def cc_Fov(cc, t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocc_a, nvir_a = t1a.shape
    nocc_b, nvir_b = t1b.shape[1:]

    kconserv = cc.khelper.kconserv

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

    Woooo = eris.oooo.copy()
    WooOO = eris.ooOO.copy()
    WOOOO = eris.OOOO.copy()

    kconserv = cc.khelper.kconserv
    P = kconserv_mat(cc.nkpts, cc.khelper.kconserv)
    tau_aa, tau_ab, tau_bb = make_tau(cc, t2, t1, t1)
    for km in range(nkpts):
        for kn in range(nkpts):
            tmp_aaaaJ = einsum('xje, ymine->yxminj', t1a, eris.ooov[km,:,kn])
            tmp_aaaaJ-= einsum('yie, xmjne->yxminj', t1a, eris.ooov[km,:,kn])
            tmp_bbbbJ = einsum('xje, ymine->yxminj', t1b, eris.OOOV[km,:,kn])
            tmp_bbbbJ-= einsum('yie, xmjne->yxminj', t1b, eris.OOOV[km,:,kn])
            tmp_aabbJ = einsum('xje, ymine->yxminj', t1b, eris.ooOV[km,:,kn])
            tmp_bbaaJ = einsum('xje, ymine->yxminj', t1a, eris.OOov[km,:,kn])
            tmp_abbaJ = -einsum('yie,xmjne->yxminj', t1b, eris.ooOV[km,:,kn])
            tmp_baabJ = -einsum('yie,xmjne->yxminj', t1a, eris.OOov[km,:,kn])

            for ki in range(nkpts):
                kj = kconserv[km,ki,kn]
                Woooo[km,ki,kn] += tmp_aaaaJ[ki,kj]
                Woooo[km,ki,kn] += 0.25*einsum('xijef,xmenf->minj', tau_aa[ki,kj],eris.ovov[km,:,kn])
                WOOOO[km,ki,kn] += tmp_bbbbJ[ki,kj]
                WOOOO[km,ki,kn] += 0.25*einsum('xijef,xmenf->minj', tau_bb[ki,kj],eris.OVOV[km,:,kn])
                WooOO[km,ki,kn] += tmp_aabbJ[ki,kj]
                WooOO[km,ki,kn] += 0.25*einsum('xijef,xmenf->minj', tau_ab[ki,kj],eris.ovOV[km,:,kn])
                WooOO[kn,ki,km] -= tmp_baabJ[ki,kj].transpose(2,1,0,3)
                WooOO[km,ki,kn] += 0.25*einsum('yijfe,ynfme->nimj', tau_ab[ki,kj], eris.ovOV[km,:,kn])

    Woooo = Woooo - Woooo.transpose(2,1,0,5,4,3,6)
    WOOOO = WOOOO - WOOOO.transpose(2,1,0,5,4,3,6)
    return Woooo, WooOO, WOOOO


def cc_Wvvvv(cc, t1, t2, eris):
    t1a, t1b = t1
    nkpts = cc.nkpts
    kconserv = cc.khelper.kconserv
    P = kconserv_mat(nkpts, kconserv)

    #:wvvvv = eris.vvvv.copy()
    #:Wvvvv += np.einsum('ymb,zyxemfa,zxyw->wzyaebf', t1a, eris.vovv.conj(), P)
    #:Wvvvv -= np.einsum('ymb,xyzfmea,xzyw->wzyaebf', t1a, eris.vovv.conj(), P)
    #:Wvvvv = Wvvvv - Wvvvv.transpose(2,1,0,5,4,3,6)
    Wvvvv = np.zeros_like(eris.vvvv)
    for ka, kb, ke in kpts_helper.loop_kkk(cc.nkpts):
        kf = kconserv[ka,ke,kb]
        aebf = eris.vvvv[ka,ke,kb].copy()
        aebf += np.einsum('mb,emfa->aebf', t1a[kb], eris.vovv[ke,kb,kf].conj())
        aebf -= np.einsum('mb,fmea->aebf', t1a[kb], eris.vovv[kf,kb,ke].conj())
        Wvvvv[ka,ke,kb] += aebf
        Wvvvv[kb,ke,ka] -= aebf.transpose(2,1,0,3)

    #:WvvVV = eris.vvVV.copy()
    #:WvvVV -= np.einsum('xma,zxwemFB,zwxy->xzyaeBF', t1a, eris.voVV.conj(), P)
    #:WvvVV -= np.einsum('yMB,wyzFMea,wzyx->xzyaeBF', t1b, eris.VOvv.conj(), P)
    WvvVV = np.empty_like(eris.vvVV)
    for ka, kb, ke in kpts_helper.loop_kkk(cc.nkpts):
        kf = kconserv[ka,ke,kb]
        aebf = eris.vvVV[ka,ke,kb].copy()
        aebf -= np.einsum('ma,emfb->aebf', t1a[ka], eris.voVV[ke,ka,kf].conj())
        aebf -= np.einsum('mb,fmea->aebf', t1b[kb], eris.VOvv[kf,kb,ke].conj())
        WvvVV[ka,ke,kb] = aebf

    #:WVVVV = eris.VVVV.copy()
    #:WVVVV += np.einsum('ymb,zyxemfa,zxyw->wzyaebf', t1b, eris.VOVV.conj(), P)
    #:WVVVV -= np.einsum('ymb,xyzfmea,xzyw->wzyaebf', t1b, eris.VOVV.conj(), P)
    #:WVVVV = WVVVV - WVVVV.transpose(2,1,0,5,4,3,6)
    WVVVV = np.zeros_like(eris.VVVV)
    for ka, kb, ke in kpts_helper.loop_kkk(cc.nkpts):
        kf = kconserv[ka,ke,kb]
        aebf = eris.VVVV[ka,ke,kb].copy()
        aebf += np.einsum('mb,emfa->aebf', t1b[kb], eris.VOVV[ke,kb,kf].conj())
        aebf -= np.einsum('mb,fmea->aebf', t1b[kb], eris.VOVV[kf,kb,ke].conj())
        WVVVV[ka,ke,kb] += aebf
        WVVVV[kb,ke,ka] -= aebf.transpose(2,1,0,3)
    return Wvvvv, WvvVV, WVVVV

def Wvvvv(cc, t1, t2, eris):
    t1a, t1b = t1
    nkpts = cc.nkpts
    kconserv = cc.khelper.kconserv
    P = kconserv_mat(nkpts, kconserv)

    tauaa, tauab, taubb = make_tau(cc, t2, t1, t1)
    Wvvvv, WvvVV, WVVVV = cc_Wvvvv(cc, t1, t2, eris)
    for ka, kb, ke in kpts_helper.loop_kkk(cc.nkpts):
        kf = kconserv[ka,ke,kb]
        for km in range(nkpts):
            kn = kconserv[ka,km,kb]
            Wvvvv[ka,ke,kb] += einsum('mnab,menf->aebf', tauaa[km,kn,ka], eris.ovov[km,ke,kn])
            WvvVV[ka,ke,kb] += einsum('mNaB,meNF->aeBF', tauab[km,kn,ka], eris.ovOV[km,ke,kn])
            WVVVV[ka,ke,kb] += einsum('mnab,menf->aebf', taubb[km,kn,ka], eris.OVOV[km,ke,kn])
    return Wvvvv, WvvVV, WVVVV

def cc_Wovvo(cc, t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape[1:]
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)

    P = kconserv_mat(cc.nkpts, kconserv)
    Wovvo = np.einsum('xyzaijb,xzyw->yxwiabj', eris.voov, P).conj()
    WovVO = np.einsum('xyzaijb,xzyw->yxwiabj', eris.voOV, P).conj()
    WOVvo = np.einsum('wzybjia,xzyw->yxwiabj', eris.voOV, P)
    WOVVO = np.einsum('xyzaijb,xzyw->yxwiabj', eris.VOOV, P).conj()

    Wovvo -= np.einsum('xyzijab,xzyw->xwzibaj', eris.oovv, P)
    WOVVO -= np.einsum('xyzijab,xzyw->xwzibaj', eris.OOVV, P)
    WoVVo = np.einsum('xyzijab,xzyw->xwzibaj', eris.ooVV, -P).copy()
    WOvvO = np.einsum('xyzijab,xzyw->xwzibaj', eris.OOvv, -P).copy()

    for km in range(nkpts):
        for kb in range(nkpts):
            for ke in range(nkpts):
                kj = kconserv[km,ke,kb]
                Wovvo[km,ke,kb] += einsum('jf, emfb->mebj', t1a[kj], eris.vovv[ke,km,kj].conj())
                WOVVO[km,ke,kb] += einsum('jf, emfb->mebj', t1b[kj], eris.VOVV[ke,km,kj].conj())
                WovVO[km,ke,kb] += einsum('jf, emfb->mebj', t1b[kj], eris.voVV[ke,km,kj].conj())
                WOVvo[km,ke,kb] += einsum('jf, emfb->mebj', t1a[kj], eris.VOvv[ke,km,kj].conj())
                ##### warnings for Ks
                Wovvo[km,ke,kb] -= einsum('jf, fmeb->mebj', t1a[kj], eris.vovv[kj,km,ke].conj())
                WOVVO[km,ke,kb] -= einsum('jf, fmeb->mebj', t1b[kj], eris.VOVV[kj,km,ke].conj())
                WOvvO[km,ke,kb] -= einsum('jf, fmeb->mebj', t1b[kj], eris.VOvv[kj,km,ke].conj())
                WoVVo[km,ke,kb] -= einsum('jf, fmeb->mebj', t1a[kj], eris.voVV[kj,km,ke].conj())

                Wovvo[km,ke,kb] -= einsum('nb, njme->mebj', t1a[kb], eris.ooov[kb,kj,km])
                WOVvo[km,ke,kb] -= einsum('nb, njme->mebj', t1a[kb], eris.ooOV[kb,kj,km])
                WOVVO[km,ke,kb] -= einsum('nb, njme->mebj', t1b[kb], eris.OOOV[kb,kj,km])
                WovVO[km,ke,kb] -= einsum('nb, njme->mebj', t1b[kb], eris.OOov[kb,kj,km])

                Wovvo[km,ke,kb] += einsum('nb, mjne->mebj', t1a[kb], eris.ooov[km,kj,kb])
                WOVVO[km,ke,kb] += einsum('nb, mjne->mebj', t1b[kb], eris.OOOV[km,kj,kb])
                WoVVo[km,ke,kb] += einsum('nb, mjne->mebj', t1b[kb], eris.ooOV[km,kj,kb])
                WOvvO[km,ke,kb] += einsum('nb, mjne->mebj', t1a[kb], eris.OOov[km,kj,kb])

                for kn in range(nkpts):
                    kf = kconserv[km,ke,kn]

                    Wovvo[km,ke,kb] -= 0.5*einsum('jnfb,menf->mebj', t2aa[kj,kn,kf], eris.ovov[km,ke,kn])
                    Wovvo[km,ke,kb] += 0.5*einsum('jnbf,menf->mebj', t2ab[kj,kn,kb], eris.ovOV[km,ke,kn])
                    WOVVO[km,ke,kb] -= 0.5*einsum('jnfb,menf->mebj', t2bb[kj,kn,kf], eris.OVOV[km,ke,kn])
                    WOVVO[km,ke,kb] += 0.5*einsum('njfb,nfme->mebj', t2ab[kn,kj,kf], eris.ovOV[kn,kf,km])
                    WovVO[km,ke,kb] += 0.5*einsum('njfb,menf->mebj', t2ab[kn,kj,kf], eris.ovov[km,ke,kn])
                    WovVO[km,ke,kb] -= 0.5*einsum('jnfb,menf->mebj', t2bb[kj,kn,kf], eris.ovOV[km,ke,kn])
                    WOVvo[km,ke,kb] -= 0.5*einsum('jnfb,nfme->mebj', t2aa[kj,kn,kf], eris.ovOV[kn,kf,km])
                    WOVvo[km,ke,kb] += 0.5*einsum('jnbf,menf->mebj', t2ab[kj,kn,kb], eris.OVOV[km,ke,kn])

                    Wovvo[km,ke,kb] += 0.5*einsum('jnfb,nemf->mebj', t2aa[kj,kn,kf], eris.ovov[kn,ke,km])
                    WOVVO[km,ke,kb] += 0.5*einsum('jnfb,nemf->mebj', t2bb[kj,kn,kf], eris.OVOV[kn,ke,km])
                    WovVO[km,ke,kb] -= 0.5*einsum('njfb,nemf->mebj', t2ab[kn,kj,kf], eris.ovov[kn,ke,km])
                    WOVvo[km,ke,kb] -= 0.5*einsum('jnbf,nemf->mebj', t2ab[kj,kn,kb], eris.OVOV[kn,ke,km])
                    WoVVo[km,ke,kb] += 0.5*einsum('jnfb,mfne->mebj', t2ab[kj,kn,kf], eris.ovOV[km,kf,kn])
                    WOvvO[km,ke,kb] += 0.5*einsum('njbf,nemf->mebj', t2ab[kn,kj,kb], eris.ovOV[kn,ke,km])

                    if kn == kb and kf == kj:
                        Wovvo[km,ke,kb] -= einsum('jf,nb,menf->mebj',t1a[kj],t1a[kn], eris.ovov[km,ke,kn])
                        WOVVO[km,ke,kb] -= einsum('jf,nb,menf->mebj',t1b[kj],t1b[kn], eris.OVOV[km,ke,kn])
                        WovVO[km,ke,kb] -= einsum('jf,nb,menf->mebj',t1b[kj],t1b[kn], eris.ovOV[km,ke,kn])
                        WOVvo[km,ke,kb] -= einsum('jf,nb,nfme->mebj',t1a[kj],t1a[kn], eris.ovOV[kn,kf,km])

                        Wovvo[km,ke,kb] += einsum('jf,nb,nemf->mebj',t1a[kj],t1a[kn], eris.ovov[kn,ke,km])
                        WOVVO[km,ke,kb] += einsum('jf,nb,nemf->mebj',t1b[kj],t1b[kn], eris.OVOV[kn,ke,km])
                        WoVVo[km,ke,kb] += einsum('jf,nb,mfne->mebj',t1a[kj],t1b[kn], eris.ovOV[km,kf,kn])
                        WOvvO[km,ke,kb] += einsum('jf,nb,nemf->mebj',t1b[kj],t1a[kn], eris.ovOV[kn,ke,km])

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
    kconserv = cc.khelper.kconserv
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    _, _, nkpts, nocca, noccb, nvira, nvirb = t2ab.shape

    Fova, Fovb = cc_Fov(cc,t1,t2,eris)
    Fooa, Foob = cc_Foo(cc,t1,t2,eris)
    for ki in range(nkpts):
        Fooa[ki] += 0.5*einsum('ie,me->mi',t1a[ki],Fova[ki])
        Foob[ki] += 0.5*einsum('ie,me->mi',t1b[ki],Fovb[ki])
    return Fooa, Foob

def Fvv(cc,t1,t2,eris):
    kconserv = cc.khelper.kconserv
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    _, _, nkpts, nocca, noccb, nvira, nvirb = t2ab.shape

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
    _, _, nkpts, nocca, noccb, nvira, nvirb = t2ab.shape

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
    _, _, nkpts, nocca, noccb, nvira, nvirb = t2ab.shape


    fova, fovb = cc_Fov(cc, t1, t2, eris)
    Wvvvv_imd, WvvVV, WVVVV = Wvvvv(cc,t1,t2,eris)
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

            tmp1aa -= np.einsum('nibf,menf->mebi',t2aa[kn,ki,kb], ovov)
            tmp1aa += np.einsum('iNbF,meNF->mebi',t2ab[ki,kn,kb], ovOV)

            tmp1ab += np.einsum('nIfB,menf->meBI',t2ab[kn,ki,kf], ovov)
            tmp1ab -= np.einsum('NIBF,meNF->meBI',t2bb[kn,ki,kb], ovOV)

            tmp1ba += np.einsum('iNbF,MENF->MEbi',t2ab[ki,kn,kb], OVOV)
            tmp1ba -= np.einsum('nibf,MEnf->MEbi',t2aa[kn,ki,kb], OVov)

            tmp1bb -= np.einsum('NIBF,MENF->MEBI',t2bb[kn,ki,kb], OVOV)
            tmp1bb += np.einsum('nIfB,MEnf->MEBI',t2ab[kn,ki,kf], OVov)

        aebi = np.einsum('ma,mebi->aebi',t1a[km],(ovvo+tmp1aa))
        Wvvvo[ka,ke,kb] -= aebi

        WVVvo[ka,ke,kb] -= np.einsum('MA,MEbi->AEbi',t1b[km],OVvo+tmp1ba)
        WvvVO[ka,ke,kb] -= np.einsum('ma,meBI->aeBI',t1a[km],ovVO+tmp1ab)

        AEBI = np.einsum('MA,MEBI->AEBI',t1b[km],(OVVO+tmp1bb))
        WVVVO[ka,ke,kb] -= AEBI


    for ka, ke, kb in itertools.product(range(nkpts),repeat=3):
        ki = kconserv[ka, ke, kb]
        # P(ab) <mb||ef> t2(miaf) (alpha alpha beta beta) and (beta beta alpha alpha)
        for km in range(nkpts):
            kf = kconserv[km,ke,ka]
            OVVV = eris.VOVV[ke,km,kf].transpose(1,0,3,2).conj() - eris.VOVV[kf,km,ke].transpose(1,2,3,0).conj()
            ovvv = eris.vovv[ke,km,kf].transpose(1,0,3,2).conj() - eris.vovv[kf,km,ke].transpose(1,2,3,0).conj()

            WVVvo[ka,ke,kb] -= lib.einsum('mfAE,mibf->AEbi', eris.voVV[kf,km,ke].transpose(1,0,3,2).conj(), t2aa[km,ki,kb])
            WVVvo[ka,ke,kb] -= lib.einsum('MEAF,iMbF->AEbi', OVVV, t2ab[ki,km,kb])

            WvvVO[ka,ke,kb] -= lib.einsum('MFae,MIBF->aeBI', eris.VOvv[kf,km,ke].transpose(1,0,3,2).conj(), t2bb[km,ki,kb])
            WvvVO[ka,ke,kb] -= lib.einsum('meaf,mIfB->aeBI', ovvv, t2ab[km,ki,kf])

        # P(ab) -t1(ma) (<mb||ei> - t2(nibf) <mn||ef>) for all spin configurations
        km = kb
        ovvo = eris.voov[ke,km,ki].transpose(1,0,3,2).conj() - eris.oovv[km,ki,ka].transpose(0,3,2,1)
        OVVO = eris.VOOV[ke,km,ki].transpose(1,0,3,2).conj() - eris.OOVV[km,ki,ka].transpose(0,3,2,1)

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

            tmp1aa -= np.einsum('niaf,menf->meai',t2aa[kn,ki,ka], ovov)
            tmp1aa += np.einsum('iNaF,meNF->meai',t2ab[ki,kn,ka], ovOV)

            tmp1ab += np.einsum('nIaF,MFne->MeaI',t2ab[kn,ki,ka], eris.OVov[km,kf,kn])

            tmp1ba += np.einsum('iNfA,mfNE->mEAi',t2ab[ki,kn,kf], eris.ovOV[km,kf,kn])

            tmp1bb -= np.einsum('NIAF,MENF->MEAI',t2bb[kn,ki,ka], OVOV)
            tmp1bb += np.einsum('nIfA,MEnf->MEAI',t2ab[kn,ki,kf], OVov)

        aebi = np.einsum('mb,meai->aebi',t1a[km],(ovvo+tmp1aa))
        Wvvvo[ka,ke,kb] += aebi

        WVVvo[ka,ke,kb] += np.einsum('mb,mEAi->AEbi',t1a[km], -eris.ooVV[km,ki,ka].transpose(0,3,2,1)+tmp1ba)
        WvvVO[ka,ke,kb] += np.einsum('MB,MeaI->aeBI',t1b[km], -eris.OOvv[km,ki,ka].transpose(0,3,2,1)+tmp1ab)

        AEBI = np.einsum('MB,MEAI->AEBI',t1b[km],(OVVO+tmp1bb))
        WVVVO[ka,ke,kb] += AEBI

    # Remaining terms
    for ka, ke, kb in itertools.product(range(nkpts),repeat=3):
        ki = kconserv[ka, ke, kb]
        Wvvvo[ka,ke,kb] += eris.vovv[kb,ki,ka].transpose(2,3,0,1) - eris.vovv[ka,ki,kb].transpose(0,3,2,1)
        WVVvo[ka,ke,kb] += eris.voVV[kb,ki,ka].transpose(2,3,0,1)
        WvvVO[ka,ke,kb] += eris.VOvv[kb,ki,ka].transpose(2,3,0,1)
        WVVVO[ka,ke,kb] += eris.VOVV[kb,ki,ka].transpose(2,3,0,1) - eris.VOVV[ka,ki,kb].transpose(0,3,2,1)

        Wvvvo[ka,ke,kb] -= lib.einsum('me,miab->aebi',fova[ke],t2aa[ke,ki,ka])
        WVVvo[ka,ke,kb] -= lib.einsum('ME,iMbA->AEbi',fovb[ke],t2ab[ki,ke,kb])
        WvvVO[ka,ke,kb] -= lib.einsum('me,mIaB->aeBI',fova[ke],t2ab[ke,ki,ka])
        WVVVO[ka,ke,kb] -= lib.einsum('ME,MIAB->AEBI',fovb[ke],t2bb[ke,ki,ka])

        Wvvvo[ka,ke,kb] += lib.einsum('if,aebf->aebi',t1a[ki],Wvvvv_imd[ka,ke,kb])
        WVVvo[ka,ke,kb] += lib.einsum('if,bfAE->AEbi',t1a[ki],WvvVV[kb,ki,ka])
        WvvVO[ka,ke,kb] += lib.einsum('IF,aeBF->aeBI',t1b[ki],WvvVV[ka,ke,kb])
        WVVVO[ka,ke,kb] += lib.einsum('IF,AEBF->AEBI',t1b[ki],WVVVV[ka,ke,kb])

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

    Woooo = eris.oooo.copy()
    WooOO = eris.ooOO.copy()
    WOOOO = eris.OOOO.copy()

    tau_aa, tau_ab, tau_bb = make_tau(cc, t2, t1, t1)
    for km in range(nkpts):
        for kn in range(nkpts):
            tmp_aaaaJ = einsum('xje, ymine->yxminj', t1a, eris.ooov[km,:,kn])
            tmp_aaaaJ-= einsum('yie, xmjne->yxminj', t1a, eris.ooov[km,:,kn])
            tmp_bbbbJ = einsum('xje, ymine->yxminj', t1b, eris.OOOV[km,:,kn])
            tmp_bbbbJ-= einsum('yie, xmjne->yxminj', t1b, eris.OOOV[km,:,kn])
            tmp_aabbJ = einsum('xje, ymine->yxminj', t1b, eris.ooOV[km,:,kn])
            tmp_bbaaJ = einsum('xje, ymine->yxminj', t1a, eris.OOov[km,:,kn])
            tmp_abbaJ = -einsum('yie,xmjne->yxminj', t1b, eris.ooOV[km,:,kn])
            tmp_baabJ = -einsum('yie,xmjne->yxminj', t1a, eris.OOov[km,:,kn])
            tmp_aabbJ = einsum('xje, ymine->yxminj', t1b, eris.ooOV[km,:,kn])

            for ki in range(nkpts):
                kj = kconserv[km,ki,kn]
                Woooo[km,ki,kn] += tmp_aaaaJ[ki,kj]
                WOOOO[km,ki,kn] += tmp_bbbbJ[ki,kj]
                WooOO[km,ki,kn] += tmp_aabbJ[ki,kj]
                WooOO[kn,ki,km] -= tmp_baabJ[ki,kj].transpose(2,1,0,3)

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
    _, _, nkpts, nocca, noccb, nvira, nvirb = t2ab.shape
    P = kconserv_mat(nkpts, kconserv)
    Woovo = np.einsum('xyzimjb, xzyw->yxwmibj', eris.ooov, P).conj() - np.einsum('zyxjmib, xzyw->yxwmibj', eris.ooov, P).conj()
    WooVO = np.einsum('xyzimJB, xzyw->yxwmiBJ', eris.ooOV, P).conj()
    WOOvo = np.einsum('xyzIMjb, xzyw->yxwMIbj', eris.OOov, P).conj()
    WOOVO = np.einsum('xyzIMJB, xzyw->yxwMIBJ', eris.OOOV, P).conj() - np.einsum('zyxJMIB, xzyw->yxwMIBJ', eris.OOOV, P).conj()

    for km, kb, ki in kpts_helper.loop_kkk(nkpts):
        kj = kconserv[km, ki, kb]
        for kn in range(nkpts):
            ke = kconserv[km,ki,kn]
            ooov = eris.ooov[km,ki,kn] - eris.ooov[kn,ki,km].transpose(2,1,0,3)
            OOOV = eris.OOOV[km,ki,kn] - eris.OOOV[kn,ki,km].transpose(2,1,0,3)

            Woovo[km,ki,kb] += einsum('mine,jnbe->mibj', ooov, t2aa[kj,kn,kb]) + einsum('miNE,jNbE->mibj', eris.ooOV[km,ki,kn], t2ab[kj,kn,kb])
            WooVO[km,ki,kb] += einsum('mine,nJeB->miBJ', ooov, t2ab[kn,kj,ke]) + einsum('miNE,JNBE->miBJ', eris.ooOV[km,ki,kn], t2bb[kj,kn,kb])
            WOOvo[km,ki,kb] += einsum('MINE,jNbE->MIbj', OOOV, t2ab[kj,kn,kb]) + einsum('MIne,jnbe->MIbj', eris.OOov[km,ki,kn], t2aa[kj,kn,kb])
            WOOVO[km,ki,kb] += einsum('MINE,JNBE->MIBJ', OOOV, t2bb[kj,kn,kb]) + einsum('MIne,nJeB->MIBJ', eris.OOov[km,ki,kn], t2ab[kn,kj,ke])
            # P(ij)
            ke = kconserv[km,kj,kn]
            ooov = eris.ooov[km,kj,kn] - eris.ooov[kn,kj,km].transpose(2,1,0,3)
            OOOV = eris.OOOV[km,kj,kn] - eris.OOOV[kn,kj,km].transpose(2,1,0,3)

            Woovo[km,ki,kb] -= einsum('mjne,inbe->mibj', ooov, t2aa[ki,kn,kb]) + einsum('mjNE,iNbE->mibj', eris.ooOV[km,kj,kn], t2ab[ki,kn,kb])
            WooVO[km,ki,kb] -= einsum('NJme,iNeB->miBJ', eris.OOov[kn,kj,km], t2ab[ki,kn,ke])
            WOOvo[km,ki,kb] -= einsum('njME,nIbE->MIbj', eris.ooOV[kn,kj,km], t2ab[kn,ki,kb])
            WOOVO[km,ki,kb] -= einsum('MJNE,INBE->MIBJ', OOOV, t2bb[ki,kn,kb]) + einsum('MJne,nIeB->MIBJ', eris.OOov[km,kj,kn], t2ab[kn,ki,ke])

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
            Woovo[km,ki,kb] -= einsum('ie,njbf,menf->mibj', t1a[ki], t2aa[kn,kj,kb], ovov) - einsum('ie,jNbF,meNF->mibj', t1a[ki], t2ab[kj,kn,kb], eris.ovOV[km,ki,kn])
            WooVO[km,ki,kb] -= -einsum('ie,nJfB,menf->miBJ', t1a[ki], t2ab[kn,kj,kf], ovov) + einsum('ie,NJBF,meNF->miBJ', t1a[ki], t2bb[kn,kj,kb], eris.ovOV[km,ki,kn])
            WOOvo[km,ki,kb] -= -einsum('IE,jNbF,MENF->MIbj', t1b[ki], t2ab[kj,kn,kb], OVOV) + einsum('IE,njbf,MEnf->MIbj', t1b[ki], t2aa[kn,kj,kb], eris.OVov[km,ki,kn])
            WOOVO[km,ki,kb] -= einsum('IE,NJBF,MENF->MIBJ', t1b[ki], t2bb[kn,kj,kb], OVOV) - einsum('IE,nJfB,MEnf->MIBJ', t1b[ki], t2ab[kn,kj,kf], eris.OVov[km,ki,kn])
            #P(ij)
            kn = kconserv[kb, ki, kf]
            ovov = eris.ovov[km,kj,kn] - eris.ovov[km,kf,kn].transpose(0,3,2,1)
            OVOV = eris.OVOV[km,kj,kn] - eris.OVOV[km,kf,kn].transpose(0,3,2,1)
            Woovo[km,ki,kb] += einsum('je,nibf,menf->mibj', t1a[kj], t2aa[kn,ki,kb], ovov) - einsum('je,iNbF,meNF->mibj', t1a[kj], t2ab[ki,kn,kb], eris.ovOV[km,kj,kn])
            WooVO[km,ki,kb] += -einsum('JE,iNfB,mfNE->miBJ', t1b[kj], t2ab[ki,kn,kf], eris.ovOV[km, kf, kn])
            WOOvo[km,ki,kb] += -einsum('je,nIbF,MFne->MIbj', t1a[kj], t2ab[kn,ki,kb], eris.OVov[km, kf, kn])
            WOOVO[km,ki,kb] += einsum('JE,NIBF,MENF->MIBJ', t1b[kj], t2bb[kn,ki,kb], OVOV) - einsum('JE,nIfB,MEnf->MIBJ', t1b[kj], t2ab[kn,ki,kf], eris.OVov[km,kj,kn])

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
	nkpts = t1a.shape[0]

	P = kconserv_mat(nkpts, kconserv)
	Wooov = eris.ooov - np.einsum('zyxnime,xzyw->xyzmine', eris.ooov, P)
	WooOV = eris.ooOV
	WOOov = eris.OOov
	WOOOV = eris.OOOV - np.einsum('zyxNIME,xzyw->xyzMINE', eris.OOOV, P)

	Wooov += np.einsum('yif,xyzmfne->xyzmine', t1a, eris.ovov) - np.einsum('yif,xwzmenf,xzyw->xyzmine', t1a, eris.ovov, P)
	WooOV += np.einsum('yif,xyzmfNE->xyzmiNE', t1a, eris.ovOV)
	WOOov += np.einsum('yIF,xyzMFne->xyzMIne', t1b, eris.OVov)
	WOOOV += np.einsum('yIF,xyzMFNE->xyzMINE', t1b, eris.OVOV) - np.einsum('yIF,xwzMENF,xzyw->xyzMINE', t1b, eris.OVOV, P)

	return Wooov, WooOV, WOOov, WOOOV

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
    nvir_a = len(idxva[0])
    nvir_b = len(idxvb[0])

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

    eri_aaaa = np.zeros((nkpts,nkpts,nkpts,len(idx1a[0]),len(idx2a[0]),len(idx3a[0]),len(idx4a[0])), dtype=np.complex128)
    eri_aabb = np.zeros((nkpts,nkpts,nkpts,len(idx1a[0]),len(idx2a[0]),len(idx3b[0]),len(idx4b[0])), dtype=np.complex128)
    eri_bbaa = np.zeros((nkpts,nkpts,nkpts,len(idx1b[0]),len(idx2b[0]),len(idx3a[0]),len(idx4a[0])), dtype=np.complex128)
    eri_bbbb = np.zeros((nkpts,nkpts,nkpts,len(idx1b[0]),len(idx2b[0]),len(idx3b[0]),len(idx4b[0])), dtype=np.complex128)
    if cross_ab:
        eri_abba = np.zeros((nkpts,nkpts,nkpts,len(idx1a[0]),len(idx2b[0]),len(idx3b[0]),len(idx4a[0])), dtype=np.complex128)
        eri_baab = np.zeros((nkpts,nkpts,nkpts,len(idx1b[0]),len(idx2a[0]),len(idx3a[0]),len(idx4b[0])), dtype=np.complex128)
    kconserv = kpts_helper.get_kconserv(eris.cell, eris.kpts)
    for ki, kj, kk in kpts_helper.loop_kkk(nkpts):
        kl = kconserv[ki, kj, kk]
        eri_aaaa[ki,kj,kk] = chemist_eri_spin[ki,kj,kk, idx1a[ki][:,None,None,None],idx2a[kj][:,None,None],idx3a[kk][:,None],idx4a[kl]]
        eri_aabb[ki,kj,kk] = chemist_eri_spin[ki,kj,kk, idx1a[ki][:,None,None,None],idx2a[kj][:,None,None],idx3b[kk][:,None],idx4b[kl]]
        eri_bbaa[ki,kj,kk] = chemist_eri_spin[ki,kj,kk, idx1b[ki][:,None,None,None],idx2b[kj][:,None,None],idx3a[kk][:,None],idx4a[kl]]
        eri_bbbb[ki,kj,kk] = chemist_eri_spin[ki,kj,kk, idx1b[ki][:,None,None,None],idx2b[kj][:,None,None],idx3b[kk][:,None],idx4b[kl]]
        if cross_ab:
            eri_abba[ki,kj,kk] = chemist_eri_spin[ki,kj,kk, idx1a[ki][:,None,None,None],idx2b[kj][:,None,None],idx3b[kk][:,None],idx4a[kl]]
            eri_baab[ki,kj,kk] = chemist_eri_spin[ki,kj,kk, idx1b[ki][:,None,None,None],idx2a[kj][:,None,None],idx3a[kk][:,None],idx4b[kl]]
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
    nvir_a = len(idxva[0])
    nvir_b = len(idxvb[0])

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
        eri[ki,kj,kk, idx1a[ki][:,None,None,None],idx2a[kj][:,None,None],idx3a[kk][:,None],idx4a[kl]] = eri_aaaa[ki,kj,kk]
        eri[ki,kj,kk, idx1a[ki][:,None,None,None],idx2a[kj][:,None,None],idx3b[kk][:,None],idx4b[kl]] = eri_aabb[ki,kj,kk]
        eri[ki,kj,kk, idx1b[ki][:,None,None,None],idx2b[kj][:,None,None],idx3a[kk][:,None],idx4a[kl]] = eri_bbaa[ki,kj,kk]
        eri[ki,kj,kk, idx1b[ki][:,None,None,None],idx2b[kj][:,None,None],idx3b[kk][:,None],idx4b[kl]] = eri_bbbb[ki,kj,kk]
        if cross_ab:
            eri[ki,kj,kk, idx1a[ki][:,None,None,None],idx2b[kj][:,None,None],idx3b[kk][:,None],idx4a[kl]] = eri_abba[ki,kj,kk]
            eri[ki,kj,kk, idx1b[ki][:,None,None,None],idx2a[kj][:,None,None],idx3a[kk][:,None],idx4b[kl]] = eri_baab[ki,kj,kk]
    return eri

def Wovvo(cc, t1, t2, eris):
    kconserv = cc.khelper.kconserv
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    _, _, nkpts, nocca, noccb, nvira, nvirb = t2ab.shape

    Wovvo, WovVO, WOVvo, WOVVO, WoVVo, WOvvO = cc_Wovvo(cc,t1,t2,eris)
    for km, kb, ke in kpts_helper.loop_kkk(nkpts):
        kj = kconserv[km, ke, kb]
        for kn in range(nkpts):
            kf = kconserv[km, ke, kn]
            Wovvo[km, kb, ke] -= einsum('jnbf,mnef->mbej',t2aa[km, kn, ke],
                                        eris.oovv[kn, kj, kb])
            WOVVO[km, kb, ke] -= einsum('NJBF,MNEF->MBEJ',t2bb[km, kn, ke],
                                        eris.OOVV[kn, kj, kb])
            WOVvo[km, kb, ke] -= einsum('NjBf,MNef->MBej',t2ab[km, kn, ke],
                                        eris.oovv[kn, kj, kb])
            WovVO[km, kb, ke] -= einsum('nJbF,mnEF->mbEJ',t2ab[km, kn, ke],
                                        eris.oovv[kn, kj, kb])
    return Wovvo, WovVO, WOVvo, WOVVO
