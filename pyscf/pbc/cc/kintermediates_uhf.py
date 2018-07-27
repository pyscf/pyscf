import numpy as np
from functools import reduce

from pyscf import lib
from pyscf.pbc.lib import kpts_helper
einsum=lib.einsum

def make_tau(cc, t2, t1, fac=1.):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocc_a, nvir_a = t1a.shape
    nocc_b, nvir_b = t1b.shape[1:]
    tau_aa = t2aa.copy()
    tau_ab = t2ab.copy()
    tau_bb = t2bb.copy()

    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
    for ki in range(nkpts):
        for ka in range(nkpts):
            for kj in range(nkpts):
                    kb = kconserv[ki,ka,kj]
                    tmp_aa = np.zeros((nocc_a,nocc_a,nvir_a,nvir_a),dtype=t2aa.dtype)
                    tmp_ab = np.zeros((nocc_a,nocc_b,nvir_a,nvir_b),dtype=t2ab.dtype)
                    tmp_bb = np.zeros((nocc_b,nocc_b,nvir_b,nvir_b),dtype=t2bb.dtype)
                    if ki == ka and kj == kb:
                        tmp_aa += 2*einsum('ia,jb->ijab', t1a[ki],t1a[kj])
                        tmp_bb += 2*einsum('ia,jb->ijab', t1b[ki],t1b[kj])
                        tmp_ab += 2*einsum('ia,jb->ijab', t1a[ki],t1b[kj])
                    if ki == kb and kj == ka:
                        tmp_aa -= 2*einsum('ib,ja->ijab', t1a[ki],t1a[kj])
                        tmp_bb -= 2*einsum('ib,ja->ijab', t1b[ki],t1b[kj])
                    tau_aa[ki,kj,ka] += fac*0.5*tmp_aa
                    tau_ab[ki,kj,ka] += fac*0.5*tmp_ab
                    tau_bb[ki,kj,ka] += fac*0.5*tmp_bb
    return tau_aa, tau_ab, tau_bb

def cc_Fvv(cc, t1, t2, uccsd_eris):
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
    return Fvv, FVV


def cc_Foo(cc, t1, t2, uccsd_eris):
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
    return Foo, FOO


def cc_Fov(cc, t1, t2, uccsd_eris):
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
    return Fov, FOV

def cc_Woooo(cc, t1, t2, uccsd_eris):

    '''
    This function returns the Js and Ks intermediates for Wmnij intermediates in physicist's notation, eg,[km,kn,ki,m,n,i,j]. abba and baab for cross excitation in chemist's notation, eg, abba->(alpha:mj, beta:ni)
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

    P = np.zeros((nkpts,nkpts,nkpts,nkpts))
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                P[ki,kj,ka,kb] = 1
    t2ba = einsum('abcijkl,abcd->badjilk',t2ab, P)
    tau_aa, tau_ab, tau_bb = make_tau(cc, t2, t1)
    tau_ba = einsum('badjilk,abcd->abcijkl',tau_ab, P)
    for km in range(nkpts):
        for kn in range(nkpts):
            tmp_aaaaJ = einsum('xje, ymnie->yxmnij', t1a, uccsd_eris.ooov.transpose(0,2,1,3,5,4,6)[km,kn]) - einsum('yie, xmnje->yxmnij', t1a, uccsd_eris.ooov.transpose(0,2,1,3,5,4,6)[km,kn])
            tmp_bbbbJ = einsum('xje, ymnie->yxmnij', t1b, uccsd_eris.OOOV.transpose(0,2,1,3,5,4,6)[km,kn]) - einsum('yie, xmnje->yxmnij', t1b, uccsd_eris.OOOV.transpose(0,2,1,3,5,4,6)[km,kn])
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
    Wmnij_aaaaK = Wmnij_aaaaJ.transpose(1,0,2,4,3,5,6).copy()
    Wmnij_bbbbK = Wmnij_bbbbJ.transpose(1,0,2,4,3,5,6).copy()
    Wmnij_aabbK = Wmnij_baabJ.transpose(1,0,2,4,3,5,6).copy()
    Wmnij_bbaaK = Wmnij_abbaJ.transpose(1,0,2,4,3,5,6).copy()
    Wmnij_abbaK = Wmnij_bbaaJ.transpose(1,0,2,4,3,5,6).copy()
    Wmnij_baabK = Wmnij_aabbJ.transpose(1,0,2,4,3,5,6).copy()

    Wmnij_J = (Wmnij_aaaaJ, Wmnij_aabbJ, Wmnij_bbaaJ, Wmnij_bbbbJ, Wmnij_abbaJ, Wmnij_baabJ)
    Wmnij_K = (Wmnij_aaaaK, Wmnij_aabbK, Wmnij_bbaaK, Wmnij_bbbbK, Wmnij_abbaK, Wmnij_baabK)
    return (Wmnij_J, Wmnij_K)

def cc_Woooo_ref(cc, t1, t2, uccsd_eris):
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

    Woooo_J = kintermediates.cc_Woooo(cc, t1, t2, uccsd_eris._kccsd_eris_j).transpose(0,2,1,3,5,4,6)
    Woooo_K = kintermediates.cc_Woooo(cc, t1, t2, uccsd_eris._kccsd_eris_k).transpose(0,2,1,3,5,4,6)

    uccsd_Woooo_J = kccsd_uhf._eri_spin2spatial(Woooo_J, 'oooo', uccsd_eris, cross_ab=True)
    uccsd_Woooo_K = kccsd_uhf._eri_spin2spatial(Woooo_K, 'oooo', uccsd_eris, cross_ab=True)
    # Woooo_J_, WooOO_J_, WOOoo_J_, WOOOO_J_, WoOOo_J_, WOooO_J_ = uccsd_Woooo_J
    # Woooo_K_, WooOO_K_, WOOoo_K_, WOOOO_K_, WoOOo_K_, WOooO_K_ = uccsd_Woooo_K
    return uccsd_Woooo_J, uccsd_Woooo_K

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
    This function returns the Js and Ks intermediates for Wmnij intermediates in physicist's notation, eg,[km,kn,ki,m,n,i,j]. abba and baab stands for cross excitation in chemist's notation, eg, abba->(alpha:mj, beta:ni)

    '''

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nkpts, nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape[1:]
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)

    Wmbej_aaaaJ = uccsd_eris.ovvo.transpose(0,2,1,3,5,4,6).copy()
    Wmbej_aabbJ = uccsd_eris.ovVO.transpose(0,2,1,3,5,4,6).copy()
    Wmbej_bbaaJ = uccsd_eris.OVvo.transpose(0,2,1,3,5,4,6).copy()
    Wmbej_bbbbJ = uccsd_eris.OVVO.transpose(0,2,1,3,5,4,6).copy()
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


    Wmbej_J = (Wmbej_aaaaJ, Wmbej_aabbJ, Wmbej_bbaaJ, Wmbej_bbbbJ, Wmbej_abbaJ, Wmbej_baabJ)
    Wmbej_K = (Wmbej_aaaaK, Wmbej_aabbK, Wmbej_bbaaK, Wmbej_bbbbK, Wmbej_abbaK, Wmbej_baabK)
    return (Wmbej_J, Wmbej_K)




def cc_Wovvo_ref(cc, t1, t2, uccsd_eris):
    '''
    reference from general ccsd
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

    Wovvo_J = kintermediates.cc_Wovvo_jk(cc, t1, t2, uccsd_eris._kccsd_eris_j).transpose(0,2,1,3,5,4,6)
    Wovvo_K = kintermediates.cc_Wovvo_jk(cc, t1, t2, uccsd_eris._kccsd_eris_k).transpose(0,2,1,3,5,4,6)

    uccsd_Wovvo_J = kccsd_uhf._eri_spin2spatial(Wovvo_J, 'ovvo', uccsd_eris, cross_ab=True)
    uccsd_Wovvo_K = kccsd_uhf._eri_spin2spatial(Wovvo_K, 'ovvo', uccsd_eris, cross_ab=True)

    # Wovvo_J = _eri_spatial2spin(uccsd_Wovvo_J, 'ovvo', uccsd_eris, cross_ab=True).transpose(0,2,1,3,5,4,6)
    # Wovvo_K = _eri_spatial2spin(uccsd_Wovvo_K, 'ovvo', uccsd_eris, cross_ab=True).transpose(0,2,1,3,5,4,6)
    return uccsd_Wovvo_J, uccsd_Wovvo_K

def kconserv_mat(nkpts, kconserv):
    P = np.zeros((nkpts,nkpts,nkpts,nkpts))
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                P[ki,kj,ka,kb] = 1
    return P
