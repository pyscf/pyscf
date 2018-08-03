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

def cc_Fvv(cc, t1, t2, uccsd_eris):
    from pyscf.pbc.cc import kccsd_uhf
    from pyscf.pbc.cc import kccsd
    from pyscf.pbc.cc import kintermediates
    from pyscf.pbc.lib import kpts_helper
    from numpy import einsum
    import numpy as np

    orbspin = uccsd_eris._kccsd_eris.orbspin
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2

    nkpts = len(orbspin)
    nocc_a, nocc_b = uccsd_eris.nocc
    nocc  = nocc_a+nocc_b
    idxva = [np.where(orbspin[k][nocc:] == 0)[0] for k in range(nkpts)]
    idxvb = [np.where(orbspin[k][nocc:] == 1)[0] for k in range(nkpts)]
    nvir_a = len(idxva[0])
    nvir_b = len(idxvb[0])

    fa = np.zeros((nkpts,nvir_a,nvir_a), dtype=np.complex128)
    fb = np.zeros((nkpts,nvir_b,nvir_b), dtype=np.complex128)

    tau_tildeaa,tau_tildeab,tau_tildebb=make_tau(cc,t2,t1,t1,fac=0.5)

    fov_ = uccsd_eris.fock[0][:,:nocc_a,nocc_a:]
    fOV_ = uccsd_eris.fock[1][:,:nocc_b,nocc_b:]
    fvv_ = uccsd_eris.fock[0][:,nocc_a:,nocc_a:]
    fVV_ = uccsd_eris.fock[1][:,nocc_b:,nocc_b:]
    
    for ka in range(nkpts):
     fa[ka]+=fvv_[ka]
     fb[ka]+=fVV_[ka]
     fa[ka]-=0.5*einsum('me,ma->ae',fov_[ka],t1a[ka])
     fb[ka]-=0.5*einsum('me,ma->ae',fOV_[ka],t1b[ka])
     for km in range(nkpts):
      fa[ka]+=einsum('mf,aemf->ae',t1a[km],uccsd_eris.vvov[ka,ka,km])
      fa[ka]-=einsum('mf,afme->ae',t1a[km],uccsd_eris.vvov[ka,km,km])
      fa[ka]+=einsum('mf,aemf->ae',t1b[km],uccsd_eris.vvOV[ka,ka,km])

      fb[ka]+=einsum('mf,aemf->ae',t1b[km],uccsd_eris.VVOV[ka,ka,km])
      fb[ka]-=einsum('mf,afme->ae',t1b[km],uccsd_eris.VVOV[ka,km,km])
      fb[ka]+=einsum('mf,aemf->ae',t1a[km],uccsd_eris.VVov[ka,ka,km])

      for kn in range(nkpts):
       fa[ka]-=0.5*einsum('mnaf,menf->ae',tau_tildeaa[km,kn,ka],uccsd_eris.ovov[km,ka,kn]) # v
       fa[ka]-=0.5*einsum('mNaF,meNF->ae',tau_tildeab[km,kn,ka],uccsd_eris.ovOV[km,ka,kn]) # v
       kf=kconserv[km,ka,kn]
       fa[ka]+=0.5*einsum('mnaf,mfne->ae',tau_tildeaa[km,kn,ka],uccsd_eris.ovov[km,kf,kn]) # v
       fa[ka]-=0.5*einsum('nMaF,MFne->ae',tau_tildeab[kn,km,ka],uccsd_eris.OVov[km,kf,kn]) # c

       fb[ka]-=0.5*einsum('mnaf,menf->ae',tau_tildebb[km,kn,ka],uccsd_eris.OVOV[km,ka,kn]) # v
       fb[ka]-=0.5*einsum('NmFa,meNF->ae',tau_tildeab[kn,km,kf],uccsd_eris.OVov[km,ka,kn]) # v
       kf=kconserv[km,ka,kn]
       fb[ka]+=0.5*einsum('mnaf,mfne->ae',tau_tildebb[km,kn,ka],uccsd_eris.OVOV[km,kf,kn]) # v
       fb[ka]-=0.5*einsum('MnFa,MFne->ae',tau_tildeab[km,kn,kf],uccsd_eris.ovOV[km,kf,kn]) # c

    '''
    from pyscf.pbc.cc import kccsd_uhf
    from pyscf.pbc.cc import kccsd
    from pyscf.pbc.cc import kintermediates
    from pyscf.pbc.lib import kpts_helper
    orbspin = uccsd_eris._kccsd_eris.orbspin
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    t1 = kccsd.spatial2spin((t1a, t1b), orbspin, kconserv)
    t2 = kccsd.spatial2spin((t2aa, t2ab, t2bb), orbspin, kconserv)

    # mimic the UCCSD contraction with the kccsd.cc_Fvv function as below.
    # It should be removed when finishing the project
    # uccsd_eris._kccsd_eris holds KCCSD spin-orbital tensor, anti-symmetrized, Physicist's notation
    gkccsd_Fvv = kintermediates.cc_Fvv(cc, t1, t2, uccsd_eris._kccsd_eris)

    # Use only uccsd_eris the spatial-orbital integral tensor in Chemist's notation

    Fvv, FVV = kccsd_uhf._eri_spin2spatial(gkccsd_Fvv, 'vv', uccsd_eris)

    t1 = kccsd.spatial2spin((t1a, t1b), orbspin, kconserv)
    t2 = kccsd.spatial2spin((t2aa, t2ab, t2bb), orbspin, kconserv)

    gkccsd_Fvv = kintermediates.cc_Fvv(cc, t1, t2, uccsd_eris._kccsd_eris)

    # Use only uccsd_eris the spatial-orbital integral tensor in Chemist's notation
    Fvv, FVV = kccsd_uhf._eri_spin2spatial(gkccsd_Fvv, 'vv', uccsd_eris)
    for ka in range(nkpts):
     print ka,np.abs(fa[ka]-Fvv[ka]).max(),np.abs(fb[ka]-FVV[ka]).max()
    '''

    return fa,fb #Fvv, FVV


def cc_Foo(cc, t1, t2, uccsd_eris):
    from pyscf.pbc.cc import kccsd_uhf
    from pyscf.pbc.cc import kccsd
    from pyscf.pbc.cc import kintermediates
    from pyscf.pbc.lib import kpts_helper
    from numpy import einsum
    orbspin = uccsd_eris._kccsd_eris.orbspin
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2

    nkpts = len(orbspin)
    nocc_a, nocc_b = uccsd_eris.nocc
    nocc  = nocc_a+nocc_b
    idxva = [np.where(orbspin[k][nocc:] == 0)[0] for k in range(nkpts)]
    idxvb = [np.where(orbspin[k][nocc:] == 1)[0] for k in range(nkpts)]
    nvir_a = len(idxva[0])
    nvir_b = len(idxvb[0])

    fa = np.zeros((nkpts,nocc_a,nocc_a), dtype=np.complex128)
    fb = np.zeros((nkpts,nocc_b,nocc_b), dtype=np.complex128)

    tau_tildeaa,tau_tildeab,tau_tildebb=make_tau(cc,t2,t1,t1,fac=0.5)

    fov_ = uccsd_eris.fock[0][:,:nocc_a,nocc_a:]
    fOV_ = uccsd_eris.fock[1][:,:nocc_b,nocc_b:]
    foo_ = uccsd_eris.fock[0][:,:nocc_a,:nocc_a]
    fOO_ = uccsd_eris.fock[1][:,:nocc_b,:nocc_b]

    for ka in range(nkpts):
     fa[ka]+=foo_[ka]
     fb[ka]+=fOO_[ka]
     fa[ka]+=0.5*einsum('me,ne->mn',fov_[ka],t1a[ka])
     fb[ka]+=0.5*einsum('me,ne->mn',fOV_[ka],t1b[ka])
     for km in range(nkpts):
      fa[ka]+=einsum('oa,mnoa->mn',t1a[km],uccsd_eris.ooov[ka,ka,km])
      fa[ka]+=einsum('oa,mnoa->mn',t1b[km],uccsd_eris.ooOV[ka,ka,km])
      fa[ka]-=einsum('oa,maon->mn',t1a[km],uccsd_eris.ovoo[ka,km,km])

      fb[ka]+=einsum('oa,mnoa->mn',t1b[km],uccsd_eris.OOOV[ka,ka,km])
      fb[ka]+=einsum('oa,mnoa->mn',t1a[km],uccsd_eris.OOov[ka,ka,km])
      fb[ka]-=einsum('oa,maon->mn',t1b[km],uccsd_eris.OVOO[ka,km,km])

    for km in range(nkpts):
     for kn in range(nkpts):
      for ke in range(nkpts):
       fa[km]+=0.5*einsum('inef,menf->mi',tau_tildeaa[km,kn,ke],uccsd_eris.ovov[km,ke,kn]) # v
       fa[km]+=0.5*einsum('iNeF,meNF->mi',tau_tildeab[km,kn,ke],uccsd_eris.ovOV[km,ke,kn]) # v
       kf=kconserv[km,ke,kn]
       fa[km]-=0.5*einsum('inef,mfne->mi',tau_tildeaa[km,kn,ke],uccsd_eris.ovov[km,kf,kn]) # v
       fb[km]+=0.5*einsum('NiEf,mfNE->mi',tau_tildeab[kn,km,ke],uccsd_eris.OVov[km,kf,kn]) # c

       fb[km]+=0.5*einsum('INEF,MENF->MI',tau_tildebb[km,kn,ke],uccsd_eris.OVOV[km,ke,kn]) # v
       fb[km]+=0.5*einsum('nIfE,MEnf->MI',tau_tildeab[kn,km,kf],uccsd_eris.OVov[km,ke,kn]) # v
       kf=kconserv[km,ke,kn]
       fb[km]-=0.5*einsum('INEF,MFNE->MI',tau_tildebb[km,kn,ke],uccsd_eris.OVOV[km,kf,kn]) # v
       fa[km]+=0.5*einsum('InFe,MFne->MI',tau_tildeab[km,kn,kf],uccsd_eris.ovOV[km,kf,kn]) # c

    '''
    from pyscf.pbc.cc import kccsd_uhf
    from pyscf.pbc.cc import kccsd
    from pyscf.pbc.cc import kintermediates
    from pyscf.pbc.lib import kpts_helper
    orbspin = uccsd_eris._kccsd_eris.orbspin
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    t1 = kccsd.spatial2spin((t1a, t1b), orbspin, kconserv)
    t2 = kccsd.spatial2spin((t2aa, t2ab, t2bb), orbspin, kconserv)

    gkccsd_Foo = kintermediates.cc_Foo(cc, t1, t2, uccsd_eris._kccsd_eris)

    Foo, FOO = kccsd_uhf._eri_spin2spatial(gkccsd_Foo, 'oo', uccsd_eris)

    t1 = kccsd.spatial2spin((t1a, t1b), orbspin, kconserv)
    t2 = kccsd.spatial2spin((t2aa, t2ab, t2bb), orbspin, kconserv)

    gkccsd_Foo = kintermediates.cc_Foo(cc, t1, t2, uccsd_eris._kccsd_eris)

    Foo, FOO = kccsd_uhf._eri_spin2spatial(gkccsd_Foo, 'oo', uccsd_eris)

    for ka in range(nkpts):
     print ka,np.abs(fa[ka]-Foo[ka]).max(),np.abs(fb[ka]-FOO[ka]).max()
    '''

    return fa,fb #Foo, FOO


def cc_Fov(cc, t1, t2, uccsd_eris):

    from pyscf.pbc.cc import kccsd_uhf
    from pyscf.pbc.cc import kccsd
    from pyscf.pbc.cc import kintermediates
    from pyscf.pbc.lib import kpts_helper

    import numpy as np
    from numpy import einsum

    orbspin = uccsd_eris._kccsd_eris.orbspin
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2

    nkpts = len(orbspin)
    nocc_a, nocc_b = uccsd_eris.nocc
    nocc  = nocc_a+nocc_b
    idxva = [np.where(orbspin[k][nocc:] == 0)[0] for k in range(nkpts)]
    idxvb = [np.where(orbspin[k][nocc:] == 1)[0] for k in range(nkpts)]
    nvir_a = len(idxva[0])
    nvir_b = len(idxvb[0])

    fov_ = uccsd_eris.fock[0][:,:nocc_a,nocc_a:]
    fOV_ = uccsd_eris.fock[1][:,:nocc_b,nocc_b:]

    fa = np.zeros((nkpts,nocc_a,nvir_a), dtype=np.complex128)
    fb = np.zeros((nkpts,nocc_b,nvir_b), dtype=np.complex128)

    for km in range(nkpts):
     fa[km]+=fov_[km]
     fb[km]+=fOV_[km]
     for kn in range(nkpts):
      fa[km]+=einsum('nf,menf->me',t1a[kn],uccsd_eris.ovov[km,km,kn])
      fa[km]+=einsum('nf,menf->me',t1b[kn],uccsd_eris.ovOV[km,km,kn])
      fa[km]-=einsum('nf,mfne->me',t1a[kn],uccsd_eris.ovov[km,kn,kn])
      fb[km]+=einsum('nf,menf->me',t1b[kn],uccsd_eris.OVOV[km,km,kn])
      fb[km]+=einsum('nf,menf->me',t1a[kn],uccsd_eris.OVov[km,km,kn])
      fb[km]-=einsum('nf,mfne->me',t1b[kn],uccsd_eris.OVOV[km,kn,kn])

    '''
    from pyscf.pbc.cc import kccsd_uhf
    from pyscf.pbc.cc import kccsd
    from pyscf.pbc.cc import kintermediates
    from pyscf.pbc.lib import kpts_helper
    orbspin = uccsd_eris._kccsd_eris.orbspin
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    t1 = kccsd.spatial2spin((t1a, t1b), orbspin, kconserv)
    t2 = kccsd.spatial2spin((t2aa, t2ab, t2bb), orbspin, kconserv)

    gkccsd_Fov = kintermediates.cc_Fov(cc, t1, t2, uccsd_eris._kccsd_eris)

    Fov, FOV = kccsd_uhf._eri_spin2spatial(gkccsd_Fov, 'ov', uccsd_eris)

    t1 = kccsd.spatial2spin((t1a, t1b), orbspin, kconserv)
    t2 = kccsd.spatial2spin((t2aa, t2ab, t2bb), orbspin, kconserv)
    gkccsd_Fov = kintermediates.cc_Fov(cc, t1, t2, uccsd_eris._kccsd_eris)
    Fov, FOV = kccsd_uhf._eri_spin2spatial(gkccsd_Fov, 'ov', uccsd_eris)
    for ka in range(nkpts):
     print ka,np.abs(fa[ka]-Fov[ka]).max(),np.abs(fb[ka]-FOV[ka]).max()
    '''

    return fa,fb #Fov, FOV

def cc_Woooo(cc, t1, t2, uccsd_eris):
    ''' This function returns the Js and Ks intermediates for Wmnij
    intermediates in physicist's notation, eg,[km,kn,ki,m,n,i,j]. abba and
    baab for cross excitation in chemist's notation, eg,
    abba->(alpha:mj, beta:ni)
    '''

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape[1:]

    Wmnij_aaaaJ = uccsd_eris.oooo.transpose(0,2,1,3,5,4,6).copy()
    Wmnij_aabbJ = uccsd_eris.ooOO.transpose(0,2,1,3,5,4,6).copy()
    Wmnij_bbaaJ = uccsd_eris.OOoo.transpose(0,2,1,3,5,4,6).copy()
    Wmnij_bbbbJ = uccsd_eris.OOOO.transpose(0,2,1,3,5,4,6).copy()

    Wmnij_abbaJ = np.zeros([nkpts,nkpts,nkpts,nocca,noccb,noccb,nocca], dtype=Wmnij_aabbJ.dtype)
    Wmnij_baabJ = np.zeros([nkpts,nkpts,nkpts,noccb,nocca,nocca,noccb], dtype=Wmnij_aabbJ.dtype)

    kconserv = cc.khelper.kconserv
    P = kconserv_mat(cc.nkpts, cc.khelper.kconserv)
    tau_aa, tau_ab, tau_bb = make_tau(cc, t2, t1, t1)
    tau_ba = np.einsum('badjilk,abcd->abcijkl', tau_ab, P)
    for km in range(nkpts):
        for kn in range(nkpts):
            tmp_aaaaJ = einsum('xje, ymnie->yxmnij', t1a, uccsd_eris.ooov.transpose(0,2,1,3,5,4,6)[km,kn])
            tmp_aaaaJ-= einsum('yie, xmnje->yxmnij', t1a, uccsd_eris.ooov.transpose(0,2,1,3,5,4,6)[km,kn])
            tmp_bbbbJ = einsum('xje, ymnie->yxmnij', t1b, uccsd_eris.OOOV.transpose(0,2,1,3,5,4,6)[km,kn])
            tmp_bbbbJ-= einsum('yie, xmnje->yxmnij', t1b, uccsd_eris.OOOV.transpose(0,2,1,3,5,4,6)[km,kn])
            tmp_aabbJ = einsum('xje, ymnie->yxmnij', t1b, uccsd_eris.ooOV.transpose(0,2,1,3,5,4,6)[km,kn])
            tmp_bbaaJ = einsum('xje, ymnie->yxmnij', t1a, uccsd_eris.OOov.transpose(0,2,1,3,5,4,6)[km,kn])
            tmp_abbaJ = -einsum('yie,xmjne->yxmnij', t1b, uccsd_eris.ooOV[km,:,kn])
            tmp_baabJ = -einsum('yie,xmjne->yxmnij', t1a, uccsd_eris.OOov[km,:,kn])

            for ki in range(nkpts):
                kj = kconserv[km,ki,kn]
                Wmnij_aaaaJ[km,kn,ki] += tmp_aaaaJ[ki,kj]
                Wmnij_aaaaJ[km,kn,ki] += 0.25*einsum('xijef,xmnef->mnij',
                        tau_aa[ki,kj],uccsd_eris.ovov.transpose(0,2,1,3,5,4,6)[km,kn])
                Wmnij_bbbbJ[km,kn,ki] += tmp_bbbbJ[ki,kj]
                Wmnij_bbbbJ[km,kn,ki] += 0.25*einsum('xijef,xmnef->mnij',
                        tau_bb[ki,kj],uccsd_eris.OVOV.transpose(0,2,1,3,5,4,6)[km,kn])
                Wmnij_aabbJ[km,kn,ki] +=tmp_aabbJ[ki,kj]
                Wmnij_aabbJ[km,kn,ki] += 0.25*einsum('xijef,xmnef->mnij',
                        tau_ab[ki,kj],uccsd_eris.ovOV.transpose(0,2,1,3,5,4,6)[km,kn])
                Wmnij_bbaaJ[km,kn,ki] +=tmp_bbaaJ[ki,kj]
                Wmnij_bbaaJ[km,kn,ki] += 0.25*einsum('xijef,xmnef->mnij',
                        tau_ba[ki,kj],uccsd_eris.OVov.transpose(0,2,1,3,5,4,6)[km,kn])
                Wmnij_abbaJ[km,kn,ki] += tmp_abbaJ[ki,kj]
                Wmnij_abbaJ[km,kn,ki] -=0.25*einsum('yijfe,ynfme->mnij', tau_ba[ki,kj], uccsd_eris.OVov[kn,:,km])
                Wmnij_baabJ[km,kn,ki] += tmp_baabJ[ki,kj]
                Wmnij_baabJ[km,kn,ki] -=0.25*einsum('yijfe,ynfme->mnij', tau_ab[ki,kj], uccsd_eris.ovOV[kn,:,km])

    ##### Symmetry relations in Wmnij
    Wmnij_aaaaK = Wmnij_aaaaJ.transpose(1,0,2,4,3,5,6)
    Wmnij_bbbbK = Wmnij_bbbbJ.transpose(1,0,2,4,3,5,6)
    Wmnij_aabbK = Wmnij_baabJ.transpose(1,0,2,4,3,5,6)

    Woooo = Wmnij_aaaaJ.transpose(0,2,1,3,5,4,6) - Wmnij_aaaaK.transpose(0,2,1,3,5,4,6)
    WooOO = Wmnij_aabbJ.transpose(0,2,1,3,5,4,6) - Wmnij_aabbK.transpose(0,2,1,3,5,4,6)
    WOOOO = Wmnij_bbbbJ.transpose(0,2,1,3,5,4,6) - Wmnij_bbbbK.transpose(0,2,1,3,5,4,6)
    return Woooo, WooOO, WOOOO


def cc_Wvvvv(cc, t1, t2, eris):
    t1a, t1b = t1
    P = kconserv_mat(cc.nkpts, cc.khelper.kconserv)

    tauaa, tauab, taubb = make_tau(cc, t2, t1, t1)
    Wvvvv = eris.vvvv.copy()
    Wvvvv += np.einsum('ymb,yzxmeaf->xzyaebf', t1a, eris.ovvv)
    Wvvvv -= np.einsum('ymb,ywxmfae,yxwz->xzyaebf', t1a, eris.ovvv, P)
    Wvvvv = Wvvvv - Wvvvv.transpose(2,1,0,5,4,3,6)

    WvvVV = eris.vvVV.copy()
    WvvVV -= np.einsum('xma,xzymeBF->xzyaeBF', t1a, eris.ovVV)
    WvvVV -= np.einsum('yMB,ywxMFae,yxwz->xzyaeBF', t1b, eris.OVvv, P)

    WVVVV = eris.VVVV.copy()
    WVVVV += np.einsum('ymb,yzxmeaf->xzyaebf', t1b, eris.OVVV)
    WVVVV -= np.einsum('ymb,ywxmfae,yxwz->xzyaebf', t1b, eris.OVVV, P)
    WVVVV = WVVVV - WVVVV.transpose(2,1,0,5,4,3,6)
    return Wvvvv, WvvVV, WVVVV

def cc_Wovvo(cc, t1, t2, uccsd_eris):
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

    P = kconserv_mat(cc.nkpts, cc.khelper.kconserv)
    Wmbej_aaaaJ = np.einsum('xyzaijb,xzyw->ywxibaj', uccsd_eris.voov.conj(), P)
    Wmbej_aabbJ = np.einsum('xyzaijb,xzyw->ywxibaj', uccsd_eris.voOV.conj(), P)
    Wmbej_bbaaJ = np.einsum('xyzaijb,xzyw->ywxibaj', uccsd_eris.VOov.conj(), P)
    Wmbej_bbbbJ = np.einsum('xyzaijb,xzyw->ywxibaj', uccsd_eris.VOOV.conj(), P)
    Wmbej_abbaJ = np.zeros([nkpts, nkpts, nkpts, nocca, nvirb, nvirb, nocca], dtype = Wmbej_aaaaJ.dtype)
    Wmbej_baabJ = np.zeros([nkpts, nkpts, nkpts, noccb, nvira, nvira, noccb], dtype = Wmbej_aaaaJ.dtype)

    Wmbej_aaaaK = uccsd_eris.vvoo.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6).copy()
    Wmbej_aabbK = np.zeros(Wmbej_aabbJ.shape, dtype = Wmbej_aabbJ.dtype)
    Wmbej_bbaaK = np.zeros(Wmbej_bbaaJ.shape, dtype = Wmbej_bbaaJ.dtype)
    Wmbej_bbbbK = uccsd_eris.VVOO.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6).copy()
    Wmbej_abbaK = uccsd_eris.VVoo.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6).copy()
    Wmbej_baabK = uccsd_eris.vvOO.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6).copy()

    for km in range(nkpts):
        for kb in range(nkpts):
            for ke in range(nkpts):
                kj = kconserv[km,ke,kb]
                Wmbej_aaaaJ[km,kb,ke] += einsum('jf, mbef->mbej', t1a[kj,:,:], uccsd_eris.ovvv.transpose(0,2,1,3,5,4,6)[km,kb,ke])
                Wmbej_bbbbJ[km,kb,ke] += einsum('jf, mbef->mbej', t1b[kj,:,:], uccsd_eris.OVVV.transpose(0,2,1,3,5,4,6)[km,kb,ke])
                Wmbej_aabbJ[km,kb,ke] += einsum('jf, mbef->mbej', t1b[kj,:,:], uccsd_eris.ovVV.transpose(0,2,1,3,5,4,6)[km,kb,ke])
                Wmbej_bbaaJ[km,kb,ke] += einsum('jf, mbef->mbej', t1a[kj,:,:], uccsd_eris.OVvv.transpose(0,2,1,3,5,4,6)[km,kb,ke])
                ##### warnings for Ks
                Wmbej_aaaaK[km,kb,ke] += einsum('jf, mbef->mbej', t1a[kj,:,:], uccsd_eris.vvov.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)[km,kb,ke])
                Wmbej_bbbbK[km,kb,ke] += einsum('jf, mbef->mbej', t1b[kj,:,:], uccsd_eris.VVOV.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)[km,kb,ke])
                Wmbej_abbaK[km,kb,ke] += einsum('jf, mbef->mbej', t1a[kj,:,:], uccsd_eris.VVov.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)[km,kb,ke])
                Wmbej_baabK[km,kb,ke] += einsum('jf, mbef->mbej', t1b[kj,:,:], uccsd_eris.vvOV.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)[km,kb,ke])

                Wmbej_aaaaJ[km,kb,ke] -= einsum('nb, mnej->mbej', t1a[kb,:,:], uccsd_eris.ovoo.transpose(0,2,1,3,5,4,6)[km,kb,ke])
                Wmbej_bbbbJ[km,kb,ke] -= einsum('nb, mnej->mbej', t1b[kb,:,:], uccsd_eris.OVOO.transpose(0,2,1,3,5,4,6)[km,kb,ke])
                Wmbej_aabbJ[km,kb,ke] -= einsum('nb, mnej->mbej', t1b[kb,:,:], uccsd_eris.ovOO.transpose(0,2,1,3,5,4,6)[km,kb,ke])
                Wmbej_bbaaJ[km,kb,ke] -= einsum('nb, mnej->mbej', t1a[kb,:,:], uccsd_eris.OVoo.transpose(0,2,1,3,5,4,6)[km,kb,ke])

                Wmbej_aaaaK[km,kb,ke] -= einsum('nb, mnej->mbej', t1a[kb,:,:], uccsd_eris.ovoo.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)[km,kb,ke])
                Wmbej_bbbbK[km,kb,ke] -= einsum('nb, mnej->mbej', t1b[kb,:,:], uccsd_eris.OVOO.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)[km,kb,ke])
                Wmbej_abbaK[km,kb,ke] -= einsum('nb, mnej->mbej', t1b[kb,:,:], uccsd_eris.OVoo.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)[km,kb,ke])
                Wmbej_baabK[km,kb,ke] -= einsum('nb, mnej->mbej', t1a[kb,:,:], uccsd_eris.ovOO.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)[km,kb,ke])

                for kn in range(nkpts):
                    kf = kconserv[km,ke,kn]

                    Wmbej_aaaaJ[km,kb,ke] += -0.5*einsum('jnfb,mnef->mbej', t2aa[kj,kn,kf], uccsd_eris.ovov.transpose(0,2,1,3,5,4,6)[km,kn,ke])+0.5*einsum('jnbf,mnef->mbej',t2ab[kj,kn,kb], uccsd_eris.ovOV.transpose(0,2,1,3,5,4,6)[km,kn,ke])
                    Wmbej_bbbbJ[km,kb,ke] += -0.5*einsum('jnfb,mnef->mbej', t2bb[kj,kn,kf], uccsd_eris.OVOV.transpose(0,2,1,3,5,4,6)[km,kn,ke])+0.5*einsum('njfb,mnef->mbej',t2ab[kn,kj,kf], uccsd_eris.OVov.transpose(0,2,1,3,5,4,6)[km,kn,ke])
                    Wmbej_aabbJ[km,kb,ke] += 0.5*einsum('njfb,mnef->mbej', t2ab[kn,kj,kf], uccsd_eris.ovov.transpose(0,2,1,3,5,4,6)[km,kn,ke]) - 0.5*einsum('jnfb,mnef->mbej',t2bb[kj,kn,kf],uccsd_eris.ovOV.transpose(0,2,1,3,5,4,6)[km,kn,ke])
                    Wmbej_bbaaJ[km,kb,ke] += -0.5*einsum('jnfb,mnef->mbej', t2aa[kj,kn,kf], uccsd_eris.OVov.transpose(0,2,1,3,5,4,6)[km,kn,ke]) + 0.5*einsum('jnbf,mnef->mbej',t2ab[kj,kn,kb], uccsd_eris.OVOV.transpose(0,2,1,3,5,4,6)[km,kn,ke])

                    Wmbej_aaaaK[km,kb,ke] += -0.5*einsum('jnfb,mnef->mbej',t2aa[kj,kn,kf], uccsd_eris.ovov.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)[km,kn,ke])
                    Wmbej_bbbbK[km,kb,ke] += -0.5*einsum('jnfb,mnef->mbej',t2bb[kj,kn,kf], uccsd_eris.OVOV.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)[km,kn,ke])
                    Wmbej_aabbK[km,kb,ke] += 0.5*einsum('njfb,mnef->mbej',t2ab[kn,kj,kf], uccsd_eris.ovov.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)[km,kn,ke])
                    Wmbej_bbaaK[km,kb,ke] += 0.5*einsum('jnbf,mnef->mbej',t2ab[kj,kn,kb], uccsd_eris.OVOV.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)[km,kn,ke])
                    Wmbej_abbaK[km,kb,ke] += -0.5*einsum('jnfb,mnef->mbej',t2ab[kj,kn,kf], uccsd_eris.OVov.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)[km,kn,ke])
                    Wmbej_baabK[km,kb,ke] += -0.5*einsum('njbf,mnef->mbej',t2ab[kn,kj,kb], uccsd_eris.ovOV.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)[km,kn,ke])

                    if kn == kb and kf == kj:
                        Wmbej_aaaaJ[km,kb,ke] += -einsum('jf,nb,mnef->mbej',t1a[kj],t1a[kn], uccsd_eris.ovov.transpose(0,2,1,3,5,4,6)[km,kn,ke])
                        Wmbej_bbbbJ[km,kb,ke] += -einsum('jf,nb,mnef->mbej',t1b[kj],t1b[kn], uccsd_eris.OVOV.transpose(0,2,1,3,5,4,6)[km,kn,ke])
                        Wmbej_aabbJ[km,kb,ke] += -einsum('jf,nb,mnef->mbej',t1b[kj],t1b[kn], uccsd_eris.ovOV.transpose(0,2,1,3,5,4,6)[km,kn,ke])
                        Wmbej_bbaaJ[km,kb,ke] += -einsum('jf,nb,mnef->mbej',t1a[kj],t1a[kn], uccsd_eris.OVov.transpose(0,2,1,3,5,4,6)[km,kn,ke])

                        Wmbej_aaaaK[km,kb,ke] += -einsum('jf,nb,mnef->mbej',t1a[kj],t1a[kn], uccsd_eris.ovov.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)[km,kn,ke])
                        Wmbej_bbbbK[km,kb,ke] += -einsum('jf,nb,mnef->mbej',t1b[kj],t1b[kn], uccsd_eris.OVOV.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)[km,kn,ke])
                        Wmbej_abbaK[km,kb,ke] += -einsum('jf,nb,mnef->mbej',t1a[kj],t1b[kn], uccsd_eris.OVov.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)[km,kn,ke])
                        Wmbej_baabK[km,kb,ke] += -einsum('jf,nb,mnef->mbej',t1b[kj],t1a[kn], uccsd_eris.ovOV.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)[km,kn,ke])

    Wovvo = Wmbej_aaaaJ.transpose(0,2,1,3,5,4,6) - Wmbej_aaaaK.transpose(0,2,1,3,5,4,6)
    WovVO = Wmbej_aabbJ.transpose(0,2,1,3,5,4,6) - Wmbej_aabbK.transpose(0,2,1,3,5,4,6)
    WOVvo = Wmbej_bbaaJ.transpose(0,2,1,3,5,4,6) - Wmbej_bbaaK.transpose(0,2,1,3,5,4,6)
    WOVVO = Wmbej_bbbbJ.transpose(0,2,1,3,5,4,6) - Wmbej_bbbbK.transpose(0,2,1,3,5,4,6)
    WoVVo = Wmbej_abbaJ.transpose(0,2,1,3,5,4,6) - Wmbej_abbaK.transpose(0,2,1,3,5,4,6)
    WOvvO = Wmbej_baabJ.transpose(0,2,1,3,5,4,6) - Wmbej_baabK.transpose(0,2,1,3,5,4,6)
    return Wovvo, WovVO, WOVvo, WOVVO, WoVVo, WOvvO

def kconserv_mat(nkpts, kconserv):
    P = np.zeros((nkpts,nkpts,nkpts,nkpts))
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                P[ki,kj,ka,kb] = 1
    return P
