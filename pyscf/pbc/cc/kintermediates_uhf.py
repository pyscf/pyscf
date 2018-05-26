import numpy as np
from functools import reduce

from pyscf import lib
from pyscf.pbc.lib import kpts_helper

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
    nkpts = len(t1a)
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
    P = kconserv_mat(nkpts, kconserv)

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

    Wovvo_J = kintermediates.cc_Wovvo(cc, t1, t2, uccsd_eris._kccsd_eris_j).transpose(0,2,1,3,5,4,6)
    Wovvo_K = kintermediates.cc_Wovvo(cc, t1, t2, uccsd_eris._kccsd_eris_k).transpose(0,2,1,3,5,4,6)

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
