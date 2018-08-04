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

    fov_ = eris.fock[0][:,:nocc_a,nocc_a:]
    fOV_ = eris.fock[1][:,:nocc_b,nocc_b:]
    fvv_ = eris.fock[0][:,nocc_a:,nocc_a:]
    fVV_ = eris.fock[1][:,nocc_b:,nocc_b:]

    for ka in range(nkpts):
     fa[ka]+=fvv_[ka]
     fb[ka]+=fVV_[ka]
     fa[ka]-=0.5*einsum('me,ma->ae',fov_[ka],t1a[ka])
     fb[ka]-=0.5*einsum('me,ma->ae',fOV_[ka],t1b[ka])
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

    fov_ = eris.fock[0][:,:nocc_a,nocc_a:]
    fOV_ = eris.fock[1][:,:nocc_b,nocc_b:]
    foo_ = eris.fock[0][:,:nocc_a,:nocc_a]
    fOO_ = eris.fock[1][:,:nocc_b,:nocc_b]

    for ka in range(nkpts):
     fa[ka]+=foo_[ka]
     fb[ka]+=fOO_[ka]
     fa[ka]+=0.5*einsum('me,ne->mn',fov_[ka],t1a[ka])
     fb[ka]+=0.5*einsum('me,ne->mn',fOV_[ka],t1b[ka])
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

    fov_ = eris.fock[0][:,:nocc_a,nocc_a:]
    fOV_ = eris.fock[1][:,:nocc_b,nocc_b:]

    fa = np.zeros((nkpts,nocc_a,nvir_a), dtype=np.complex128)
    fb = np.zeros((nkpts,nocc_b,nvir_b), dtype=np.complex128)

    for km in range(nkpts):
     fa[km]+=fov_[km]
     fb[km]+=fOV_[km]
     for kn in range(nkpts):
      fa[km]+=einsum('nf,menf->me',t1a[kn],eris.ovov[km,km,kn])
      fa[km]+=einsum('nf,menf->me',t1b[kn],eris.ovOV[km,km,kn])
      fa[km]-=einsum('nf,mfne->me',t1a[kn],eris.ovov[km,kn,kn])
      fb[km]+=einsum('nf,menf->me',t1b[kn],eris.OVOV[km,km,kn])
      fb[km]+=einsum('nf,nfme->me',t1a[kn],eris.ovOV[kn,kn,km])
      fb[km]-=einsum('nf,mfne->me',t1b[kn],eris.OVOV[km,kn,kn])

    return fa,fb #Fov, FOV

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
    P = kconserv_mat(cc.nkpts, cc.khelper.kconserv)

    tauaa, tauab, taubb = make_tau(cc, t2, t1, t1)
    Wvvvv = eris.vvvv.copy()
    Wvvvv += np.einsum('ymb,zyxemfa,zxyw->wzyaebf', t1a, eris.vovv.conj(), P)
    Wvvvv -= np.einsum('ymb,xyzfmea,xzyw->wzyaebf', t1a, eris.vovv.conj(), P)
    Wvvvv = Wvvvv - Wvvvv.transpose(2,1,0,5,4,3,6)

    WvvVV = eris.vvVV.copy()
    WvvVV -= np.einsum('xma,zxwemFB,zwxy->xzyaeBF', t1a, eris.voVV.conj(), P)
    WvvVV -= np.einsum('yMB,wyzFMea,wzyx->xzyaeBF', t1b, eris.VOvv.conj(), P)

    WVVVV = eris.VVVV.copy()
    WVVVV += np.einsum('ymb,zyxemfa,zxyw->wzyaebf', t1b, eris.VOVV.conj(), P)
    WVVVV -= np.einsum('ymb,xyzfmea,xzyw->wzyaebf', t1b, eris.VOVV.conj(), P)
    WVVVV = WVVVV - WVVVV.transpose(2,1,0,5,4,3,6)
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
