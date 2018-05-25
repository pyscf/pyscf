import numpy as np
from functools import reduce


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
    return uccsd_Woooo_J, uccsd_Woooo_K

def cc_Wvvvv(cc, t1, t2, uccsd_eris):
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


#    nkpts, nocc, nvir = t1.shape
#    eris_vovv = - eris.ovvv.transpose(1,0,2,4,3,5,6)
#    tau = make_tau(cc,t2,t1,t1)
#    Wabef = eris.vvvv.copy()
#    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
#    for ka in range(nkpts):
#        for kb in range(nkpts):
#            for ke in range(nkpts):
#                km = kb
#                tmp  = einsum('mb,amef->abef',t1[kb],eris_vovv[ka,km,ke])
#                km = ka
#                tmp -= einsum('ma,bmef->abef',t1[ka],eris_vovv[kb,km,ke])
#                Wabef[ka,kb,ke] += -tmp
#                # km + kn - ka = kb
#                # => kn = ka - km + kb
#                for km in range(nkpts):
#                    kn = kconserv[ka,km,kb]
#                    Wabef[ka,kb,ke] += 0.25*einsum('mnab,mnef->abef',tau[km,kn,ka],
#                                                   eris.oovv[km,kn,ke])
#    return Wabef

    Wvvvv_J = kintermediates.cc_Wvvvv(cc, t1, t2, uccsd_eris._kccsd_eris_j).transpose(0,2,1,3,5,4,6)
    Wvvvv_K = kintermediates.cc_Wvvvv(cc, t1, t2, uccsd_eris._kccsd_eris_k).transpose(0,2,1,3,5,4,6)

    uccsd_Wvvvv_J = kccsd_uhf._eri_spin2spatial(Wvvvv_J, 'vvvv', uccsd_eris, cross_ab=True)
    uccsd_Wvvvv_K = kccsd_uhf._eri_spin2spatial(Wvvvv_K, 'vvvv', uccsd_eris, cross_ab=True)
    return uccsd_Wvvvv_J, uccsd_Wvvvv_K

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
    return uccsd_Wovvo_J, uccsd_Wovvo_K
