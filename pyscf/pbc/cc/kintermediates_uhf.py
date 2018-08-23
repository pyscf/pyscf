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
    ''' This function returns the Js and Ks intermediates for Wmnij
    intermediates in physicist's notation, eg,[km,kn,ki,m,n,i,j]. abba and
    baab for cross excitation in chemist's notation, eg,
    abba->(alpha:mj, beta:ni)
    '''

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
    '''
    This function returns the Js and Ks intermediates for Wmnij intermediates
    in physicist's notation, eg,[km,kn,ki,m,n,i,j]. abba and baab stands for
    cross excitation in chemist's notation, eg, abba->(alpha:mj, beta:ni)
    '''

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
    kconserv = cc.khelper.kconserv
    Fme = cc_Fov(cc,t1,t2,eris)
    return Fme

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

def Wovoo(cc,t1,t2,eris):
    kconserv = cc.khelper.kconserv
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    _, _, nkpts, nocca, noccb, nvira, nvirb = t2ab.shape

    Woovo = eris.oovo - eris.vooo.transpose(2,1,0,5,4,3,6)
    WooVO = eris.ooVO
    WOOvo = eris.OOvo
    WOOVO = eris.OOVO - eris.VOOO.transpose(2,1,0,5,4,3,6)

    ooov = eris.ooov - eris.ooov.transpose(2,1,0,5,4,3,6)
    OOOV = eris.OOOV - eris.OOOV.transpose(2,1,0,5,4,3,6)

    ovvo = eris.ovvo - eris.vvoo.transpose(2,1,0,5,4,3,6)
    OVVO = eris.OVVO - eris.VVOO.transpose(2,1,0,5,4,3,6)

    ovov = eris.ovov - eris.ovov.transpose(2,1,0,5,4,3,6)
    OVOV = eris.OVOV - eris.OVOV.transpose(2,1,0,5,4,3,6)

    for km, kb, ki in kpts_helper.loop_kkk(nkpts):
        kj = kconserv[km, ki, kb]
        for kn in range(nkpts):

            ke = kconserv[km,ki,kn]
            Woovo[km,ki,kb] += einsum('mine,jnbe->mibj', ooov[km,ki,kn], t2aa[kj,kn,kb]) + einsum('miNE,jNbE->mibj', eris.ooOV[km,ki,kn], t2ab[kj,kn,kb])
            WooVO[km,ki,kb] += einsum('mine,nJeB->miBJ', ooov[km,ki,kn], t2ab[kn,kj,ke]) + einsum('miNE,JNBE->miBJ', eris.ooOV[km,ki,kn], t2bb[kj,kn,kb])
            WOOvo[km,ki,kb] += einsum('MINE,jNbE->MIbj', OOOV[km,ki,kn], t2ab[kj,kn,kb]) + einsum('MIne,jnbe->MIbj', eris.OOov[km,ki,kn], t2aa[kj,kn,kb])
            WOOVO[km,ki,kb] += einsum('MINE,JNBE->MIBJ', OOOV[km,ki,kn], t2bb[kj,kn,kb]) + einsum('MIne,nJeB->MIBJ', eris.OOvo[km,ki,kn], t2ab[kn,kj,ke])

        Woovo[km,ki,kb] += einsum('ie,mebj->mibj', t1a[ki], ovvo[km,ki,kb])
        WooVO[km,ki,kb] += einsum('ie,meBJ->miBJ', t1a[ki], eris.ovVO[km,ki,kb])
        WOOvo[km,ki,kb] += einsum('IE,MEbj->MIbj', t1b[ki], eris.OVvo[km,ki,kb])
        WOOVO[km,ki,kb] += einsum('IE,MEBJ->MIBJ', t1b[ki], OVVO[km,ki,kb])


        for kf in range(nkpts):
            kn = kconserv[kb, kj, kf]

            Woovo[km, ki, kb] -= einsum('ie,njbf,menf->mbij', t1a[ki], t2aa[kn,kj,kb], ovov) - einsum('ie,jNbF,meNF->mbij', t1a[ki], t2ab[kj,kn,kb], eris.ovOV[km,ki,kn])
            WooVO[km, ki, kb] -= -einsum('ie,nJfB,menf->mBiJ', t1a[ki], t2ab[kn,kj,kf], ovov) + einsum('ie,NJBF,meNF->mBiJ', t1a[ki], t2bb[kn,kj,kb], eris.ovOV[km,ki,kn])
            WOOvo[km, ki, kb] -= -einsum('IE,jNbF,MENF->MbIj', t1b[ki], t2ab[kj,kn,kb], OVOV) + einsum('IE,njbf,MEnf->MbIj', t1b[ki], t2aa[kn,kj,kb], eris.OVov[km,ki,kn])
            WOOVO[km, ki, kb] -= einsum('IE,NJBF,MENF->MBIJ', t1b[ki], t2bb[kn,kj,kb], OVOV) - einsum('IE,nJfB,MEnf->MBIJ', t1b[ki], t2ab[kn,kj,kf], eris.OVov[km,ki,kn])
        # P(ij)
        for kn in range(nkpts):
            Wmbij[km, kb, ki] -= einsum('mnje,inbe->mbij', eris.ooov[km, kn, kj], t2[ki, kn, kb])

            ke = kconserv[km,kj,kn]
            Woovo[km,ki,kb] -= einsum('mjne,inbe->mibj', ooov[km,kj,kn], t2aa[ki,kn,kb]) + einsum('mjNE,iNbE->mibj', eris.ooOV[km,kj,kn], t2ab[ki,kn,kb])
            WooVO[km,ki,kb] -= einsum('meNJ,iNeB->miBJ', eris.ovOO[km,ke,kn], t2ab[ki,kn,ke])
            WOOvo[km,ki,kb] -= einsum('MEnj,nIbE->MIbj', eris.OVoo[km,ke,kn], t2ab[kn,ki,kb])
            WOOVO[km,ki,kb] -= einsum('MJNE,INBE->MIBJ', OOOV[km,kj,kn], t2bb[ki,kn,kb]) + einsum('MJne,nIeB->MIBJ', eris.OOvo[km,kj,kn], t2ab[kn,ki,ke])




            #### stop line

        Wmbij[km, kb, ki] -= einsum('je,mbei->mbij', t1[kj], -eris.ovov[km, kb, ki].transpose(0, 1, 3, 2))
        for kf in range(nkpts):
            kn = kconserv[kb, ki, kf]
            Wmbij[km, kb, ki] += einsum('je,nibf,mnef->mbij', t1[kj], t2[kn, ki, kb], eris.oovv[km, kn, kj])

    Fov, FOV = Fov(cc, t1, t2, eris)
    Woooo, WooOO, WOOoo, WOOOO = Woooo(cc,t1,t2,eris)
    tauaa, tauab, taubb = make_tau(cc, t2, t1, t1p, fac=1.)
    for km, kb, ki in kpts_helper.loop_kkk(nkpts):
        kj = kconserv[km, ki, kb]
        Wmbij[km, kb, ki] -= einsum('me,ijbe->mbij', FFov[km], t2[ki, kj, kb])
        Wmbij[km, kb, ki] -= einsum('nb,mnij->mbij', t1[kb], WWoooo[km, kb, ki])

    for km, kb, ki in kpts_helper.loop_kkk(nkpts):
        kj = kconserv[km, ki, kb]
        Wmbij[km, kb, ki] += 0.5 * einsum('xmbef,xijef->mbij', eris.ovvv[km, kb, :], tau[ki, kj, :])

    return Wmbij
