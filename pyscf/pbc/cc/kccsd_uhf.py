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
#          Mario Motta
#          Yang Gao
#          Qiming Sun <osirpt.sun@gmail.com>
#          Jason Yu
#

import time
from functools import reduce
import numpy as np
import h5py

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import scf
from pyscf.cc import uccsd
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import member, gamma_point
from pyscf import __config__
from pyscf.pbc.cc import kintermediates_uhf

einsum = lib.einsum

def update_amps(cc, t1, t2, eris):
    from pyscf.pbc.cc import kccsd
    from pyscf.pbc.cc import kintermediates as imdk
    from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM
    time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)

    t1a, t1b = uccsd_t1 = t1
    t2aa, t2ab, t2bb = uccsd_t2 = t2
    Ht1a = np.zeros_like(t1a)
    Ht1b = np.zeros_like(t1b)
    Ht2aa = np.zeros_like(t2aa)
    Ht2ab = np.zeros_like(t2ab)
    Ht2bb = np.zeros_like(t2bb)

    orbspin = eris._kccsd_eris.orbspin
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
    t1 = kccsd.spatial2spin(t1, orbspin, kconserv)
    t2 = kccsd.spatial2spin(t2, orbspin, kconserv)

    nocca, nvira = t1a.shape[1:]
    noccb, nvirb = t1b.shape[1:]
    fvv_ = eris.fock[0][:,nocca:,nocca:]
    fVV_ = eris.fock[1][:,noccb:,noccb:]
    foo_ = eris.fock[0][:,:nocca,:nocca]
    fOO_ = eris.fock[1][:,:noccb,:noccb]
    fov_ = eris.fock[0][:,:nocca,nocca:]
    fOV_ = eris.fock[1][:,:noccb,noccb:]

    #Ht1, Ht2 = kccsd.update_amps(cc, t1, t2, eris._kccsd_eris)
    eris, uccsd_eris = eris._kccsd_eris, eris
    nkpts, nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:, :nocc, nocc:].copy()
    foo = fock[:, :nocc, :nocc].copy()
    fvv = fock[:, nocc:, nocc:].copy()

    tau = imdk.make_tau(cc, t2, t1, t1)

    Fvv = imdk.cc_Fvv(cc, t1, t2, eris)
    Foo = imdk.cc_Foo(cc, t1, t2, eris)
    Fov = imdk.cc_Fov(cc, t1, t2, eris)
    # Fvv_, FVV_ = UCCSD_imdk.cc_Fvv(cc, kccsd.spin2spatial(t1, orbspin, kconserv), t2, uccsd_eris)  # MM
    # Foo_, FOO_ = UCCSD_imdk.cc_Foo(cc, kccsd.spin2spatial(t1, orbspin, kconserv), t2, eris)  # MM
    # Fov_, FOV_ = UCCSD_imdk.cc_Fov(cc, t1, t2, eris)  # MM
    # Convert to UCCSD tensors and use them below
    # lower-case o/v stand for alpha, upper-case O/V stand for beta. Fvv is
    # the alpha part of Fock virtual-virtual block, FVV is beta part of Fock
    # virtual-virtual block.
    #Fvv_, FVV_ = _eri_spin2spatial(imdk.cc_Fvv(cc, t1, t2, eris), 'vv', uccsd_eris)
    #Foo_, FOO_ = _eri_spin2spatial(imdk.cc_Foo(cc, t1, t2, eris), 'oo', uccsd_eris)
    #Fov_, FOV_ = _eri_spin2spatial(imdk.cc_Fov(cc, t1, t2, eris), 'ov', uccsd_eris)
    Fvv_, FVV_ = kintermediates_uhf.cc_Fvv(cc, uccsd_t1, uccsd_t2, uccsd_eris)
    Foo_, FOO_ = kintermediates_uhf.cc_Foo(cc, uccsd_t1, uccsd_t2, uccsd_eris)
    Fov_, FOV_ = kintermediates_uhf.cc_Fov(cc, uccsd_t1, uccsd_t2, uccsd_eris)

    # The UCCSD tensor Woooo_J_ or Woooo_K_ are kind of chemist's convention
    uccsd_Woooo_J, uccsd_Woooo_K = kintermediates_uhf.cc_Woooo(cc, uccsd_t1, uccsd_t2, uccsd_eris)
    Woooo_J_, WooOO_J_, WOOoo_J_, WOOOO_J_, WoOOo_J_, WOooO_J_ = uccsd_Woooo_J
    Woooo_K_, WooOO_K_, WOOoo_K_, WOOOO_K_, WoOOo_K_, WOooO_K_ = uccsd_Woooo_K
    # * transpose(0,2,1,3,5,4,6) to change back to physicist's convention.
    #   Check whether the UCCSD tensors are the same to symmetry-allowed blocks
    #   of spin-integral tensor Woooo
    Woooo_J = _eri_spatial2spin(uccsd_Woooo_J, 'oooo', uccsd_eris, cross_ab=True).transpose(0,2,1,3,5,4,6)
    Woooo_K = _eri_spatial2spin(uccsd_Woooo_K, 'oooo', uccsd_eris, cross_ab=True).transpose(0,2,1,3,5,4,6)

    Woooo = imdk.cc_Woooo(cc, t1, t2, eris)  # For testing only
    assert(abs(Woooo_J - Woooo_K - Woooo).max() < 1e-12)

    # The UCCSD tensor Wvvvv_J_ or Wvvvv_K_ are kind of chemist's convention
    uccsd_Wovvo_J, uccsd_Wovvo_K = kintermediates_uhf.cc_Wovvo(cc, uccsd_t1, uccsd_t2, uccsd_eris)
    Wovvo_J_, WovVO_J_, WOVvo_J_, WOVVO_J_, WoVVo_J_, WOvvO_J_ = uccsd_Wovvo_J
    Wovvo_K_, WovVO_K_, WOVvo_K_, WOVVO_K_, WoVVo_K_, WOvvO_K_ = uccsd_Wovvo_K
    # * transpose(0,2,1,3,5,4,6) to change back to physicist's convention.
    #   Check whether the UCCSD tensors are the same to symmetry-allowed blocks
    #   of spin-integral tensor Woooo
    Wovvo_J = _eri_spatial2spin(uccsd_Wovvo_J, 'ovvo', uccsd_eris, cross_ab=True).transpose(0,2,1,3,5,4,6)
    Wovvo_K = _eri_spatial2spin(uccsd_Wovvo_K, 'ovvo', uccsd_eris, cross_ab=True).transpose(0,2,1,3,5,4,6)

    Wovvo = imdk.cc_Wovvo(cc, t1, t2, eris)  # For testing only
    assert(abs(Wovvo_J - Wovvo_K - Wovvo).max() < 1e-12)

#DELETEME    # * Use spin2spatial functiont to transform gccsd tensor to uccsd tensors
#DELETEME    #   Pass uccsd_eris to kuhf_cc_Wovvo, in which the anti-symmetrized
#DELETEME    #   gccsd.eris spin-orbital tensors can be accessed via uccsd_eris._kccsd_eris.
#DELETEME    # * cc_Wovvo function should return two types of ovvo intermediates: Wovvo_J and Wovvo_K
#DELETEME    #   The J part of the gccsd.eris spin-orbital tensors can be accessed via
#DELETEME    #   uccsd_eris._kccsd_eris_j. The K part of gccsd.eris spin-orbital tensor
#DELETEME    #   can be accessed vira uccsd_eris._kccsd_eris_k
#DELETEME    #Wovvo_J, Wovvo_K = kuhf_cc_Wovvo(cc, kccsd.spin2spatial(t1, orbspin, kconserv),
#DELETEME    #                       kccsd.spin2spatial(t2, orbspin, kconserv), uccsd_eris)
#DELETEME    # * transpose(0,2,1,3,5,4,6) to transform the tensor to chemist's
#DELETEME    #   convention,  function _eri_spin2spatial only supports spin-integral
#DELETEME    #   tensor in chemist's convention
#DELETEME    Wovvo_J = imdk.cc_Wovvo(cc, t1, t2, uccsd_eris._kccsd_eris_j).transpose(0,2,1,3,5,4,6)
#DELETEME    Wovvo_K = imdk.cc_Wovvo(cc, t1, t2, uccsd_eris._kccsd_eris_k).transpose(0,2,1,3,5,4,6)
#DELETEME    assert(abs(Wovvo_J - Wovvo_K - Wovvo.transpose(0,2,1,3,5,4,6)).max())
#DELETEME
#DELETEME    uccsd_Wovvo_J = _eri_spin2spatial(Wovvo_J, 'ovvo', uccsd_eris, cross_ab=True)
#DELETEME    uccsd_Wovvo_K = _eri_spin2spatial(Wovvo_K, 'ovvo', uccsd_eris, cross_ab=True)
#DELETEME    assert(abs(Wovvo_J - _eri_spatial2spin(uccsd_Wovvo_J, 'ovvo', uccsd_eris, cross_ab=True)).max() < 1e-12)
#DELETEME    assert(abs(Wovvo_K - _eri_spatial2spin(uccsd_Wovvo_K, 'ovvo', uccsd_eris, cross_ab=True)).max() < 1e-12)
#DELETEME
#DELETEME    # The 6 non-zero blocks of the spin-orbital Wovvo tensor are
#DELETEME    Wovvo_J_, WovVO_J_, WOVvo_J_, WOVVO_J_, WoVVo_J_, WOvvO_J_ = uccsd_Wovvo_J
#DELETEME    Wovvo_K_, WovVO_K_, WOVvo_K_, WOVVO_K_, WoVVo_K_, WOvvO_K_ = uccsd_Wovvo_K

#DELETEME    # Similar transformation can be applied on the spin-orbital Woooo and Wvvvv tensors.
#DELETEME    # Many of UCCSD Woooo J and K tensors are equivalent by using permutation symmetry.
#DELETEME    # The number of unique tensors should be 3.
#DELETEME    Woooo_J = imdk.cc_Woooo(cc, t1, t2, uccsd_eris._kccsd_eris_j).transpose(0,2,1,3,5,4,6)
#DELETEME    Woooo_K = imdk.cc_Woooo(cc, t1, t2, uccsd_eris._kccsd_eris_k).transpose(0,2,1,3,5,4,6)
#DELETEME    assert(abs(Woooo_J - Woooo_K - Woooo.transpose(0,2,1,3,5,4,6)).max())
#DELETEME    uccsd_Woooo_J = _eri_spin2spatial(Woooo_J, 'oooo', uccsd_eris, cross_ab=True)
#DELETEME    uccsd_Woooo_K = _eri_spin2spatial(Woooo_K, 'oooo', uccsd_eris, cross_ab=True)
#DELETEME    Woooo_J_, WooOO_J_, WOOoo_J_, WOOOO_J_, WoOOo_J_, WOooO_J_ = uccsd_Woooo_J
#DELETEME    Woooo_K_, WooOO_K_, WOOoo_K_, WOOOO_K_, WoOOo_K_, WOooO_K_ = uccsd_Woooo_K

    # Move energy terms to the other side
    for k in range(nkpts):
        Fvv[k] -= np.diag(np.diag(fvv[k]))
        Foo[k] -= np.diag(np.diag(foo[k]))

    for k in range(nkpts):
        Fvv_[k] -= np.diag(np.diag(fvv_[k]))
        FVV_[k] -= np.diag(np.diag(fVV_[k]))
        Foo_[k] -= np.diag(np.diag(foo_[k]))
        FOO_[k] -= np.diag(np.diag(fOO_[k]))

    # Get the momentum conservation array
    # Note: chemist's notation for momentum conserving t2(ki,kj,ka,kb), even though
    # integrals are in physics notation
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)

    # eris_ovvo can be accessed constructed with _kccsd_eris_j.ovvo  and  eris._kccsd_eris_k.
    # eris_ovvo = uccsd_eris._kccsd_eris_j.ovvo - uccsd.eris._kccsd_eris_k.ovvo
    #
    # Note _kccsd_eris_j.ovvo are in physicist's convention, transpose(0,2,1,3,5,4,6)
    # to change it to chemist's notation which was required by _eri_spin2spatial function.
    # Relation between uccsd_eris integrals and _kccsd_eris_j.ovvo are
    # uccsd_eris.ovvo, uccsd_eris.ovVO, uccsd_eris.OVvo, uccsd_eris.OVVO = _eri_spin2spatial(_kccsd_eris_j.ovvo.transpose(0,2,1,3,5,4,6), 'ovvo', eris)
    #
    # Relation between uccsd_eris integrals and _kccsd_eris_k.ovvo are
    # uccsd_eris.ovvo, uccsd_eris.ovVO, uccsd_eris.OVvo, uccsd_eris.OVVO = _eri_spin2spatial(_kccsd_eris_k.ovvo.transpose(1,0,2,4,3,5,6).transpose(0,2,1,3,5,4,6), 'ovvo', eris)
    #
    # Similar transformation can be found for eris_oovo, eris_vvvo below.

    eris_ovvo = np.zeros(shape=(nkpts, nkpts, nkpts, nocc, nvir, nvir, nocc), dtype=t2.dtype)
    eris_oovo = np.zeros(shape=(nkpts, nkpts, nkpts, nocc, nocc, nvir, nocc), dtype=t2.dtype)
    eris_vvvo = np.zeros(shape=(nkpts, nkpts, nkpts, nvir, nvir, nvir, nocc), dtype=t2.dtype)
    eris_oovv = eris.oovv
    for km, kb, ke in kpts_helper.loop_kkk(nkpts):
        kj = kconserv[km, ke, kb]
        # <mb||je> -> -<mb||ej>
        eris_ovvo[km, kb, ke] = -eris.ovov[km, kb, kj].transpose(0, 1, 3, 2)
        # <mn||je> -> -<mn||ej>
        # let kb = kn as a dummy variable
        eris_oovo[km, kb, ke] = -eris.ooov[km, kb, kj].transpose(0, 1, 3, 2)
        # <ma||be> -> - <be||am>*
        # let kj = ka as a dummy variable
        kj = kconserv[km, ke, kb]
        eris_vvvo[ke, kj, kb] = -eris.ovvv[km, kb, ke].transpose(2, 3, 1, 0).conj()

    # T1 equation
    t1new = np.zeros(shape=t1.shape, dtype=t1.dtype)
    for ka in range(nkpts):
        ki = ka
        t1new[ka] += np.array(fov[ka, :, :]).conj()
        #:t1new[ka] += einsum('ie,ae->ia', t1[ka], Fvv[ka])
        #:t1new[ka] += -einsum('ma,mi->ia', t1[ka], Foo[ka])
        for km in range(nkpts):
            t1new[ka] += einsum('imae,me->ia', t2[ka, km, ka], Fov[km])
            #t1new[ka] += -einsum('nf,naif->ia', t1[km], uccsd_eris._kccsd_eris_j.ovov[km, ka, ki])
            #t1new[ka] -= -einsum('nf,naif->ia', t1[km], uccsd_eris._kccsd_eris_k.ovov[km, ka, ki])
            for kn in range(nkpts):
                ke = kconserv[km, ki, kn]
                #t1new[ka] += -0.5 * einsum('imef,maef->ia', t2[ki, km, ke], eris.ovvv[km, ka, ke])
                t1new[ka] += -0.5 * einsum('mnae,nmei->ia', t2[km, kn, ka], eris_oovo[kn, km, ke])

    for ka in range(nkpts):
        Ht1a[ka] += einsum('ie,ae->ia', t1a[ka], Fvv_[ka])
        Ht1b[ka] += einsum('ie,ae->ia', t1b[ka], FVV_[ka])
        Ht1a[ka] -= einsum('ma,mi->ia', t1a[ka], Foo_[ka])
        Ht1b[ka] -= einsum('ma,mi->ia', t1b[ka], FOO_[ka])

        for km in range(nkpts):
            # ka == ki; km == kf == km
            # <ma||if> = [mi|af] - [mf|ai]
            #         => [mi|af] - [fm|ia]
            Ht1a[ka] += einsum('mf,aimf->ia', t1a[km], uccsd_eris.voov[ka, ka, km])
            Ht1a[ka] += - 1. * einsum('mf,afmi->ia', t1a[km], uccsd_eris.vvoo[ka, km, km])
            Ht1a[ka] += einsum('MF,aiMF->ia', t1b[km], uccsd_eris.voOV[ka, ka, km])

            # miaf - mfai => miaf - fmia
            Ht1b[ka] += einsum('MF,AIMF->IA', t1b[km], uccsd_eris.VOOV[ka, ka, km])
            Ht1b[ka] += - 1. * einsum('MF,AFMI->IA', t1b[km], uccsd_eris.VVOO[ka, km, km])
            Ht1b[ka] += einsum('mf,AImf->IA', t1a[km], uccsd_eris.VOov[ka, ka, km])

            for kf in range(nkpts):
                ki = ka
                ke = kconserv[ki, kf, km]
                Ht1a[ka] +=   0.5 * einsum('imef,aemf->ia', t2aa[ki, km, ke], uccsd_eris.vvov[ka, ke, km])
                Ht1a[ka] +=   1.0 * einsum('iMeF,aeMF->ia', t2ab[ki, km, ke], uccsd_eris.vvOV[ka, ke, km])
                Ht1a[ka] += - 0.5 * einsum('imef,afme->ia', t2aa[ki, km, ke], uccsd_eris.vvov[ka, kf, km])

                Ht1b[ka] +=   0.5 * einsum('IMEF,AEMF->IA', t2bb[ki, km, ke], uccsd_eris.VVOV[ka, ke, km])
                Ht1b[ka] +=   1.0 * einsum('mIfE,AEmf->IA', t2ab[km, ki, kf], uccsd_eris.VVov[ka, ke, km])
                Ht1b[ka] += - 0.5 * einsum('IMEF,AFME->IA', t2bb[ki, km, ke], uccsd_eris.VVOV[ka, kf, km])


    t1new += kccsd.spatial2spin((Ht1a, Ht1b), orbspin, kconserv)

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        # Chemist's notation for momentum conserving t2(ki,kj,ka,kb)
        kb = kconserv[ki, ka, kj]

        # Fvv equation
        Ftmpa_kb = Fvv_[kb] - 0.5 * einsum('mb,me->be', t1a[kb], Fov_[kb])
        Ftmpb_kb = FVV_[kb] - 0.5 * einsum('MB,ME->BE', t1b[kb], FOV_[kb])

        Ftmpa_ka = Fvv_[ka] - 0.5 * einsum('mb,me->be', t1a[ka], Fov_[ka])
        Ftmpb_ka = FVV_[ka] - 0.5 * einsum('MB,ME->BE', t1b[ka], FOV_[ka])

        tmp = einsum('ijae,be->ijab', t2aa[ki, kj, ka], Ftmpa_kb)
        Ht2aa[ki, kj, ka] += tmp

        tmp = einsum('IJAE,BE->IJAB', t2bb[ki, kj, ka], Ftmpb_kb)
        Ht2bb[ki, kj, ka] += tmp

        tmp = einsum('iJaE,BE->iJaB', t2ab[ki, kj, ka], Ftmpb_kb)
        Ht2ab[ki, kj, ka] += tmp

        tmp = einsum('iJeB,ae->iJaB', t2ab[ki, kj, ka], Ftmpa_ka)
        Ht2ab[ki, kj, ka] += tmp

        #P(ab)
        tmp = einsum('ijbe,ae->ijab', t2aa[ki, kj, kb], Ftmpa_ka)
        Ht2aa[ki, kj, ka] -= tmp

        tmp = einsum('IJBE,AE->IJAB', t2bb[ki, kj, kb], Ftmpb_ka)
        Ht2bb[ki, kj, ka] -= tmp

        # Foo equation
        Ftmpa_kj = Foo_[kj] + 0.5 * einsum('je,me->mj', t1a[kj], Fov_[kj])
        Ftmpb_kj = FOO_[kj] + 0.5 * einsum('JE,ME->MJ', t1b[kj], FOV_[kj])

        Ftmpa_ki = Foo_[ki] + 0.5 * einsum('je,me->mj', t1a[ki], Fov_[ki])
        Ftmpb_ki = FOO_[ki] + 0.5 * einsum('JE,ME->MJ', t1b[ki], FOV_[ki])

        tmp = einsum('imab,mj->ijab', t2aa[ki, kj, ka], Ftmpa_kj)
        Ht2aa[ki, kj, ka] -= tmp

        tmp = einsum('IMAB,MJ->IJAB', t2bb[ki, kj, ka], Ftmpb_kj)
        Ht2bb[ki, kj, ka] -= tmp

        tmp = einsum('iMaB,MJ->iJaB', t2ab[ki, kj, ka], Ftmpb_kj)
        Ht2ab[ki, kj, ka] -= tmp

        tmp = einsum('mJaB,mi->iJaB', t2ab[ki, kj, ka], Ftmpa_ki)
        Ht2ab[ki, kj, ka] -= tmp

        #P(ij)
        tmp = einsum('jmab,mi->ijab', t2aa[kj, ki, ka], Ftmpa_ki)
        Ht2aa[ki, kj, ka] += tmp

        tmp = einsum('JMAB,MI->IJAB', t2bb[kj, ki, ka], Ftmpb_ki)
        Ht2bb[ki, kj, ka] += tmp

    # T2 equation
    t2new = np.array(eris.oovv).conj()
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        # Chemist's notation for momentum conserving t2(ki,kj,ka,kb)
        kb = kconserv[ki, ka, kj]

        #Ftmp = Fvv[kb] - 0.5 * einsum('mb,me->be', t1[kb], Fov[kb])
        #tmp = einsum('ijae,be->ijab', t2[ki, kj, ka], Ftmp)
        #t2new[ki, kj, ka] += tmp

        ##t2new[ki,kj,kb] -= tmp.transpose(0,1,3,2)
        #Ftmp = Fvv[ka] - 0.5 * einsum('ma,me->ae', t1[ka], Fov[ka])
        #tmp = einsum('ijbe,ae->ijab', t2[ki, kj, kb], Ftmp)
        #t2new[ki, kj, ka] -= tmp

        #Ftmp = Foo[kj] + 0.5 * einsum('je,me->mj', t1[kj], Fov[kj])
        #tmp = einsum('imab,mj->ijab', t2[ki, kj, ka], Ftmp)
        #t2new[ki, kj, ka] -= tmp

        ##t2new[kj,ki,ka] += tmp.transpose(1,0,2,3)
        #Ftmp = Foo[ki] + 0.5 * einsum('ie,me->mi', t1[ki], Fov[ki])
        #tmp = einsum('jmab,mi->ijab', t2[kj, ki, ka], Ftmp)
        #t2new[ki, kj, ka] += tmp

        for km in range(nkpts):
            # Wminj
            #   - km - kn + ka + kb = 0
            # =>  kn = ka - km + kb
            kn = kconserv[ka, km, kb]
            t2new[ki, kj, ka] += 0.5 * einsum('mnab,mnij->ijab', tau[km, kn, ka], Woooo[km, kn, ki])
#            ke = km
#            t2new[ki, kj, ka] += 0.5 * einsum('ijef,abef->ijab', tau[ki, kj, ke], Wvvvv[ka, kb, ke])

            # Wmbej
            #     - km - kb + ke + kj = 0
            #  => ke = km - kj + kb
            ke = kconserv[km, kj, kb]
            tmp = einsum('imae,mbej->ijab', t2[ki, km, ka], Wovvo[km, kb, ke])
            #     - km - kb + ke + kj = 0
            # =>  ke = km - kj + kb
            #
            # t[i,e] => ki = ke
            # t[m,a] => km = ka
            if km == ka and ke == ki:
                tmp -= einsum('ie,ma,mbej->ijab', t1[ki], t1[km], eris_ovvo[km, kb, ke])
            t2new[ki, kj, ka] += tmp
            t2new[ki, kj, kb] -= tmp.transpose(0, 1, 3, 2)
            t2new[kj, ki, ka] -= tmp.transpose(1, 0, 2, 3)
            t2new[kj, ki, kb] += tmp.transpose(1, 0, 3, 2)

        ke = ki
        tmp = einsum('ie,abej->ijab', t1[ki], eris_vvvo[ka, kb, ke])
        t2new[ki, kj, ka] += tmp
        # P(ij) term
        ke = kj
        tmp = einsum('je,abei->ijab', t1[kj], eris_vvvo[ka, kb, ke])
        t2new[ki, kj, ka] -= tmp

        km = ka
        tmp = einsum('ma,mbij->ijab', t1[ka], eris.ovoo[km, kb, ki])
        t2new[ki, kj, ka] -= tmp
        # P(ab) term
        km = kb
        tmp = einsum('mb,maij->ijab', t1[kb], eris.ovoo[km, ka, ki])
        t2new[ki, kj, ka] += tmp

    tau = None
    add_vvvv_(cc, (Ht2aa, Ht2ab, Ht2bb), uccsd_t1, uccsd_t2, uccsd_eris)
    t2new += kccsd.spatial2spin((Ht2aa, Ht2ab, Ht2bb), orbspin, kconserv)

    eia = np.zeros(shape=(nocc, nvir), dtype=t1new.dtype)
    for ki in range(nkpts):
        eia = foo[ki].diagonal()[:, None] - fvv[ki].diagonal()[None, :]
        # When padding the occupied/virtual arrays, some fock elements will be zero
        idx = np.where(abs(eia) < LOOSE_ZERO_TOL)[0]
        eia[idx] = LARGE_DENOM

        t1new[ki] /= eia

    eijab = np.zeros(shape=(nocc, nocc, nvir, nvir), dtype=t2new.dtype)
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]
        eijab = (foo[ki].diagonal()[:, None, None, None] + foo[kj].diagonal()[None, :, None, None] -
                 fvv[ka].diagonal()[None, None, :, None] - fvv[kb].diagonal()[None, None, None, :])
        # Due to padding; see above discussion concerning t1new in update_amps()
        idx = np.where(abs(eijab) < LOOSE_ZERO_TOL)[0]
        eijab[idx] = LARGE_DENOM

        t2new[ki, kj, ka] /= eijab

    Ht1 = kccsd.spin2spatial(t1new, orbspin, kconserv)
    Ht2 = kccsd.spin2spatial(t2new, orbspin, kconserv)

    time0 = log.timer_debug1('update t1 t2', *time0)
    return Ht1, Ht2


def get_normt_diff(cc, t1, t2, t1new, t2new):
    '''Calculates norm(t1 - t1new) + norm(t2 - t2new).'''
    return (np.linalg.norm(t1new[0] - t1[0])**2 +
            np.linalg.norm(t1new[1] - t1[1])**2 +
            np.linalg.norm(t2new[0] - t2[0])**2 +
            np.linalg.norm(t2new[1] - t2[1])**2 +
            np.linalg.norm(t2new[2] - t2[2])**2) ** .5


def energy(cc, t1, t2, eris):
    from pyscf.pbc.cc import kccsd

    orbspin = eris.mo_coeff.orbspin
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
    t1 = kccsd.spatial2spin(t1, orbspin, kconserv)
    t2 = kccsd.spatial2spin(t2, orbspin, kconserv)

    nkpts, nocc, nvir = t1.shape
    fock = eris.fock
    eris_oovv = eris.oovv.copy()
    e = 0.0 + 0j
    for ki in range(nkpts):
        e += einsum('ia,ia', fock[ki, :nocc, nocc:], t1[ki, :, :])
    t1t1 = np.zeros(shape=t2.shape, dtype=t2.dtype)
    for ki in range(nkpts):
        ka = ki
        for kj in range(nkpts):
            #kb = kj
            t1t1[ki, kj, ka, :, :, :, :] = einsum('ia,jb->ijab', t1[ki, :, :], t1[kj, :, :])
    tau = t2 + 2 * t1t1
    e += 0.25 * np.dot(tau.flatten(), eris_oovv.flatten())
    e /= nkpts
    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in KCCSD energy %s', e)
    return e.real


def get_nocc(cc, per_kpoint=False):
    '''See also function get_nocc in pyscf/pbc/mp2/kmp2.py'''
    if cc._nocc is not None:
        return cc._nocc

    assert(cc.frozen == 0)

    if isinstance(cc.frozen, (int, np.integer)):
        nocca = [(np.count_nonzero(cc.mo_occ[0][k] > 0) - cc.frozen) for k in range(cc.nkpts)]
        noccb = [(np.count_nonzero(cc.mo_occ[1][k] > 0) - cc.frozen) for k in range(cc.nkpts)]

    else:
        raise NotImplementedError

    if not per_kpoint:
        nocca = np.amax(nocca)
        noccb = np.amax(noccb)
    return nocca, noccb

def get_nmo(cc, per_kpoint=False):
    '''See also function get_nmo in pyscf/pbc/mp2/kmp2.py'''
    if cc._nmo is not None:
        return cc._nmo

    assert(cc.frozen == 0)

    if isinstance(cc.frozen, (int, np.integer)):
        nmoa = [(cc.mo_occ[0][k].size - cc.frozen) for k in range(cc.nkpts)]
        nmob = [(cc.mo_occ[1][k].size - cc.frozen) for k in range(cc.nkpts)]

    else:
        raise NotImplementedError

    if not per_kpoint:
        nmoa = np.amax(nmoa)
        nmob = np.amax(nmob)
    return nmoa, nmob

def get_frozen_mask(cc):
    '''See also get_frozen_mask function in pyscf/pbc/mp2/kmp2.py'''

    moidxa = [np.ones(x.size, dtype=np.bool) for x in cc.mo_occ[0]]
    moidxb = [np.ones(x.size, dtype=np.bool) for x in cc.mo_occ[1]]
    assert(cc.frozen == 0)

    if isinstance(cc.frozen, (int, np.integer)):
        for idx in moidxa:
            idx[:cc.frozen] = False
        for idx in moidxb:
            idx[:cc.frozen] = False
    else:
        raise NotImplementedError

    return moidxa, moisxb

def amplitudes_to_vector(t1, t2):
    return np.hstack((t1[0].ravel(), t1[1].ravel(),
                      t2[0].ravel(), t2[1].ravel(), t2[2].ravel()))

def vector_to_amplitudes(vec, nmo, nocc, nkpts=1):
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb

    t1a = vec[:nkpts*nocca*nvira].reshape(nkpts,nocca,nvira)
    vec = vec[nkpts*nocca*nvira:]

    t1b = vec[:nkpts*noccb*nvirb].reshape(nkpts,noccb,nvirb)
    vec = vec[nkpts*noccb*nvirb:]

    t2aa = vec[:nkpts**3*nocca**2*nvira**2]
    t2aa = t2aa.reshape(nkpts,nkpts,nkpts,nocca,nocca,nvira,nvira)
    vec = vec[nkpts**3*nocca**2*nvira**2:]

    t2ab = vec[:nkpts**3*nocca*noccb*nvira*nvirb]
    t2ab = t2ab.reshape(nkpts,nkpts,nkpts,nocca,noccb,nvira,nvirb)
    vec = vec[nkpts**3*nocca*noccb*nvira*nvirb:]

    t2bb = vec.reshape(nkpts,nkpts,nkpts,noccb,noccb,nvirb,nvirb)
    return (t1a,t1b), (t2aa,t2ab,t2bb)

def add_vvvv_(cc, Ht2, t1, t2, eris):
    nocca, noccb = cc.nocc
    nmoa, nmob = cc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    nkpts = cc.nkpts
    kconserv = cc.khelper.kconserv

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    Ht2aa, Ht2ab, Ht2bb = Ht2

    if cc.direct and hasattr(eris, 'Lpv'):
        def get_Wvvvv(ka, kc, kb):
            kd = kconserv[ka,kc,kb]

            Lbd = (eris.Lpv[kb,kd,:,nocca:] -
                   lib.einsum('Lkd,kb->Lbd', eris.Lpv[kb,kd,:,:nocca], t1a[kb]))
            Wvvvv = lib.einsum('Lac,Lbd->acbd', eris.Lpv[ka,kc,:,nocca:], Lbd)
            kcbd = lib.einsum('Lkc,Lbd->kcbd', eris.Lpv[ka,kc,:,:nocca],
                              eris.Lpv[kb,kd,:,nocca:])
            Wvvvv -= lib.einsum('kcbd,ka->acbd', kcbd, t1a[ka])

            LBD = (eris.LPV[kb,kd,:,noccb:] -
                   lib.einsum('Lkd,kb->Lbd', eris.LPV[kb,kd,:,:noccb], t1b[kb]))

            WvvVV = lib.einsum('Lac,Lbd->acbd', eris.Lpv[ka,kc,:,nocca:], LBD)
            kcbd = lib.einsum('Lkc,Lbd->kcbd', eris.Lpv[ka,kc,:,:nocca],
                              eris.LPV[kb,kd,:,noccb:])
            WvvVV -= lib.einsum('kcbd,ka->acbd', kcbd, t1a[ka])

            WVVVV = lib.einsum('Lac,Lbd->acbd', eris.LPV[ka,kc,:,noccb:], LBD)
            kcbd = lib.einsum('Lkc,Lbd->kcbd', eris.LPV[ka,kc,:,:noccb],
                              eris.LPV[kb,kd,:,noccb:])
            WVVVV -= lib.einsum('kcbd,ka->acbd', kcbd, t1b[ka])

            Wvvvv *= (1./nkpts)
            WvvVV *= (1./nkpts)
            WVVVV *= (1./nkpts)
            return Wvvvv, WvvVV, WVVVV

        for ka, kb, kc in kpts_helper.loop_kkk(nkpts):
            kd = kconserv[ka,kc,kb]
            Wvvvv, WvvVV, WVVVV = get_Wvvvv(ka, kc, kb)
            for ki in range(nkpts):
                kj = kconserv[ka,ki,kb]
                tauaa = t2aa[ki,kj,kc].copy()
                tauab = t2ab[ki,kj,kc].copy()
                taubb = t2bb[ki,kj,kc].copy()
                if ki == kc and kj == kd:
                    tauaa += np.einsum('ic,jd->ijcd', t1a[ki], t1a[kj])
                    tauab += np.einsum('ic,jd->ijcd', t1a[ki], t1b[kj])
                    taubb += np.einsum('ic,jd->ijcd', t1b[ki], t1b[kj])
                if ki == kd and kj == kc:
                    tauaa -= np.einsum('id,jc->ijcd', t1a[ki], t1a[kj])
                    taubb -= np.einsum('id,jc->ijcd', t1b[ki], t1b[kj])

                tmp = lib.einsum('acbd,ijcd->ijab', Wvvvv, tauaa) * .5
                Ht2aa[ki,kj,ka] += tmp
                Ht2aa[ki,kj,kb] -= tmp.transpose(0,1,3,2)

                tmp = lib.einsum('acbd,ijcd->ijab', WVVVV, taubb) * .5
                Ht2bb[ki,kj,ka] += tmp
                Ht2bb[ki,kj,kb] -= tmp.transpose(0,1,3,2)

                Ht2ab[ki,kj,ka] += lib.einsum('acbd,ijcd->ijab', WvvVV, tauab)
            Wvvvv = WvvVV = WVVVV = None
    else:
        _Wvvvv, _WvvVV, _WVVVV = kintermediates_uhf.cc_Wvvvv(cc, t1, t2, eris)
        def get_Wvvvv(ka, kc, kb):
            return _Wvvvv[ka,kc,kb], _WvvVV[ka,kc,kb], _WVVVV[ka,kc,kb]

        #:Ht2aa += np.einsum('xyuijef,zuwaebf,xyuv,zwuv->xyzijab', tauaa, _Wvvvv, P, P) * .5
        #:Ht2bb += np.einsum('xyuijef,zuwaebf,xyuv,zwuv->xyzijab', taubb, _WVVVV, P, P) * .5
        #:Ht2ab += np.einsum('xyuiJeF,zuwaeBF,xyuv,zwuv->xyziJaB', tauab, _WvvVV, P, P)
        for ka, kb, kc in kpts_helper.loop_kkk(nkpts):
            kd = kconserv[ka,kc,kb]
            Wvvvv, WvvVV, WVVVV = get_Wvvvv(ka, kc, kb)
            for ki in range(nkpts):
                kj = kconserv[ka,ki,kb]
                tauaa = t2aa[ki,kj,kc].copy()
                tauab = t2ab[ki,kj,kc].copy()
                taubb = t2bb[ki,kj,kc].copy()
                if ki == kc and kj == kd:
                    tauaa += np.einsum('ic,jd->ijcd', t1a[ki], t1a[kj])
                    tauab += np.einsum('ic,jd->ijcd', t1a[ki], t1b[kj])
                    taubb += np.einsum('ic,jd->ijcd', t1b[ki], t1b[kj])
                if ki == kd and kj == kc:
                    tauaa -= np.einsum('id,jc->ijcd', t1a[ki], t1a[kj])
                    taubb -= np.einsum('id,jc->ijcd', t1b[ki], t1b[kj])

                Ht2aa[ki,kj,ka] += lib.einsum('acbd,ijcd->ijab', Wvvvv, tauaa) * .5
                Ht2bb[ki,kj,ka] += lib.einsum('acbd,ijcd->ijab', WVVVV, taubb) * .5
                Ht2ab[ki,kj,ka] += lib.einsum('acbd,ijcd->ijab', WvvVV, tauab)
        _Wvvvv = _WvvVV = _WVVVV = None

    tauaa, tauab, taubb = kintermediates_uhf.make_tau(cc, t2, t1, t1)
    P = kintermediates_uhf.kconserv_mat(cc.nkpts, cc.khelper.kconserv)
    eris_ovov = eris.ovov - eris.ovov.transpose(2,1,0,5,4,3,6)
    eris_ovOV = eris.ovOV
    eris_OVOV = eris.OVOV - eris.OVOV.transpose(2,1,0,5,4,3,6)
    minj = np.einsum('xwymenf,uvwijef,xywz,uvwz->xuyminj', eris_ovov, tauaa, P, P)
    MINJ = np.einsum('xwymenf,uvwijef,xywz,uvwz->xuyminj', eris_OVOV, taubb, P, P)
    miNJ = np.einsum('xwymeNF,uvwiJeF,xywz,uvwz->xuymiNJ', eris_ovOV, tauab, P, P)
    Ht2aa += np.einsum('xuyminj,xywmnab,xyuv->uvwijab', minj, tauaa, P) * .125
    Ht2bb += np.einsum('xuyminj,xywmnab,xyuv->uvwijab', MINJ, taubb, P) * .125
    Ht2ab += np.einsum('xuymiNJ,xywmNaB,xyuv->uvwiJaB', miNJ, tauab, P) * .5
    return (Ht2aa, Ht2ab, Ht2bb)


class KUCCSD(uccsd.UCCSD):

    max_space = getattr(__config__, 'pbc_cc_kccsd_uhf_KUCCSD_max_space', 20)

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        assert(isinstance(mf, scf.khf.KSCF))
        uccsd.UCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.kpts = mf.kpts
        self.mo_energy = mf.mo_energy
        self.khelper = kpts_helper.KptsHelper(mf.cell, self.kpts)
        self.direct = True  # If possible, use GDF to compute Wvvvv on-the-fly

        keys = set(['kpts', 'mo_energy', 'khelper', 'max_space', 'direct'])
        self._keys = self._keys.union(keys)

    @property
    def nkpts(self):
        return len(self.kpts)

    get_normt_diff = get_normt_diff
    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    update_amps = update_amps
    energy = energy

    def dump_flags(self):
        return uccsd.UCCSD.dump_flags(self)

    def ao2mo(self, mo_coeff=None):
        nkpts = self.nkpts
        nmoa, nmob = self.nmo
        mem_incore = nkpts**3 * (nmoa**4 + nmob**4) * 8 / 1e6
        mem_now = lib.current_memory()[0]

        if (mem_incore + mem_now < self.max_memory) or self.mol.incore_anyway:
            return _make_eris_incore(self, mo_coeff)
        else:
            raise NotImplementedError

    def init_amps(self, eris):
        from pyscf.pbc.cc import kccsd
        time0 = time.clock(), time.time()

        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb = nmoa - nocca, nmob - noccb

        eris, uccsd_eris = eris._kccsd_eris, eris
        orbspin = eris.orbspin
        nocc = nocca + noccb
        nvir = nvira + nvirb

        nkpts = self.nkpts
        t1 = np.zeros((nkpts, nocc, nvir), dtype=np.complex128)
        t2 = np.zeros((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=np.complex128)
        self.emp2 = 0
        foo = eris.fock[:, :nocc, :nocc].copy()
        fvv = eris.fock[:, nocc:, nocc:].copy()
        fov = eris.fock[:, :nocc, nocc:].copy()
        eris_oovv = eris.oovv.copy()
        eia = np.zeros((nocc, nvir))
        eijab = np.zeros((nocc, nocc, nvir, nvir))

        kconserv = kpts_helper.get_kconserv(self._scf.cell, self.kpts)
        for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
            kb = kconserv[ki, ka, kj]
            eijab = (foo[ki].diagonal()[:, None, None, None] + foo[kj].diagonal()[None, :, None, None] -
                     fvv[ka].diagonal()[None, None, :, None] - fvv[kb].diagonal()[None, None, None, :])
            # Due to padding; see above discussion concerning t1new in update_amps()
            idx = np.where(abs(eijab) < LOOSE_ZERO_TOL)[0]
            eijab[idx] = LARGE_DENOM

            t2[ki, kj, ka] = eris_oovv[ki, kj, ka] / eijab

        t2 = np.conj(t2)
        self.emp2 = 0.25 * np.einsum('pqrijab,pqrijab', t2, eris_oovv).real
        self.emp2 /= nkpts

        t1 = kccsd.spin2spatial(t1, orbspin, kconserv)
        t2 = kccsd.spin2spatial(t2, orbspin, kconserv)

        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2.real)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2


    def amplitudes_to_vector(self, t1, t2):
        return amplitudes_to_vector(t1, t2)

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None, nkpts=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        if nkpts is None: nkpts = self.nkpts
        return vector_to_amplitudes(vec, nmo, nocc, nkpts)

UCCSD = KUCCSD


def _make_eris_incore(cc, mo_coeff=None):
    import copy
    from pyscf.pbc import scf
    from pyscf.pbc.cc import kccsd
    cput0 = (time.clock(), time.time())
    log = logger.new_logger(cc)
    eris = uccsd._ChemistsERIs()
    if mo_coeff is None:
        mo_coeff = cc.mo_coeff
    eris.mo_coeff = mo_coeff  # TODO: handle frozen orbitals
    eris.nocc = cc.nocc
    eris.cell = cc._scf.cell  # TODO: delete later
    eris.kpts = cc.kpts  # TODO: delete later

    kgcc = kccsd.GCCSD(scf.addons.convert_to_ghf(cc._scf))
    _kccsd_eris = eris._kccsd_eris = kccsd._make_eris_incore(kgcc, kgcc._scf.mo_coeff)
    orbspin = eris._kccsd_eris.orbspin
    nkpts = cc.nkpts
    nocca, noccb = eris.nocc
    nmoa, nmob = cc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    nocc = nocca + noccb
    nvir = nvira + nvirb
    idxoa = [np.where(orbspin[k][:nocc] == 0)[0] for k in range(nkpts)]
    idxob = [np.where(orbspin[k][:nocc] == 1)[0] for k in range(nkpts)]
    idxva = [np.where(orbspin[k][nocc:] == 0)[0] for k in range(nkpts)]
    idxvb = [np.where(orbspin[k][nocc:] == 1)[0] for k in range(nkpts)]

    # Re-make our fock MO matrix elements from density and fock AO
    focka = [_kccsd_eris.fock[k][orbspin[k]==0][:,orbspin[k]==0] for k in range(nkpts)]
    fockb = [_kccsd_eris.fock[k][orbspin[k]==1][:,orbspin[k]==1] for k in range(nkpts)]
    eris.fock = (np.asarray(focka), np.asarray(fockb))

    kpts = cc.kpts
    nao = _kccsd_eris.mo_coeff[0].shape[0] // 2
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
    so_coeff = [mo[:nao] + mo[nao:] for mo in _kccsd_eris.mo_coeff]

    nocc = nocca + noccb
    nvir = nvira + nvirb
    nmo = nocc + nvir
    eri = np.empty((nkpts, nkpts, nkpts, nmo, nmo, nmo, nmo), dtype=np.complex128)
    fao2mo = cc._scf.with_df.ao2mo
    for kp, kq, kr in kpts_helper.loop_kkk(nkpts):
        ks = kconserv[kp, kq, kr]
        eri_kpt = fao2mo(
            (so_coeff[kp], so_coeff[kq], so_coeff[kr], so_coeff[ks]), (kpts[kp], kpts[kq], kpts[kr], kpts[ks]),
            compact=False)
        eri_kpt[(orbspin[kp][:, None] != orbspin[kq]).ravel()] = 0
        eri_kpt[:, (orbspin[kr][:, None] != orbspin[ks]).ravel()] = 0
        eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo)
        eri[kp, kq, kr] = eri_kpt
    # In chemist's notation
    oooo = eri[:, :, :, :nocc, :nocc, :nocc, :nocc] / nkpts
    ooov = eri[:, :, :, :nocc, :nocc, :nocc, nocc:] / nkpts
    ovoo = eri[:, :, :, :nocc, nocc:, :nocc, :nocc] / nkpts
    oovv = eri[:, :, :, :nocc, :nocc, nocc:, nocc:] / nkpts
    vvoo = eri[:, :, :, nocc:, nocc:, :nocc, :nocc] / nkpts
    ovov = eri[:, :, :, :nocc, nocc:, :nocc, nocc:] / nkpts
    ovvv = eri[:, :, :, :nocc, nocc:, nocc:, nocc:] / nkpts
    voov = eri[:, :, :, nocc:, :nocc, :nocc, nocc:] / nkpts
    vovv = eri[:, :, :, nocc:, :nocc, nocc:, nocc:] / nkpts
    vvov = eri[:, :, :, nocc:, nocc:, :nocc, nocc:] / nkpts
    vvvv = eri[:, :, :, nocc:, nocc:, nocc:, nocc:] / nkpts

    eris.oooo, eris.ooOO, eris.OOoo, eris.OOOO = _eri_spin2spatial(oooo, 'oooo', eris)
    eris.ooov, eris.ooOV, eris.OOov, eris.OOOV = _eri_spin2spatial(ooov, 'ooov', eris)
    eris.ovoo, eris.ovOO, eris.OVoo, eris.OVOO = _eri_spin2spatial(ovoo, 'ovoo', eris)
    eris.oovv, eris.ooVV, eris.OOvv, eris.OOVV = _eri_spin2spatial(oovv, 'oovv', eris)
    eris.vvoo, eris.vvOO, eris.VVoo, eris.VVOO = _eri_spin2spatial(vvoo, 'vvoo', eris)
    eris.ovov, eris.ovOV, eris.OVov, eris.OVOV = _eri_spin2spatial(ovov, 'ovov', eris)
    eris.voov, eris.voOV, eris.VOov, eris.VOOV = _eri_spin2spatial(voov, 'voov', eris)
    eris.vovv, eris.voVV, eris.VOvv, eris.VOVV = _eri_spin2spatial(vovv, 'vovv', eris)
    eris.vvov, eris.vvOV, eris.VVov, eris.VVOV = _eri_spin2spatial(vvov, 'vvov', eris)
    eris.ovvv, eris.ovVV, eris.OVvv, eris.OVVV = _eri_spin2spatial(ovvv, 'ovvv', eris)
    eris.vvvv, eris.vvVV, eris.VVvv, eris.VVVV = _eri_spin2spatial(vvvv, 'vvvv', eris)

    # For testing only
    eris._kccsd_eris_j = copy.copy(_kccsd_eris)
    eris._kccsd_eris_j.oooo = oooo.transpose(0,2,1,3,5,4,6)
    eris._kccsd_eris_j.ooov = ooov.transpose(0,2,1,3,5,4,6)
    eris._kccsd_eris_j.oovo = ovoo.transpose(0,2,1,3,5,4,6)
    eris._kccsd_eris_j.ovov = oovv.transpose(0,2,1,3,5,4,6)
    eris._kccsd_eris_j.oovv = ovov.transpose(0,2,1,3,5,4,6)
    eris._kccsd_eris_j.ovvv = ovvv.transpose(0,2,1,3,5,4,6)
    eris._kccsd_eris_j.voov = voov.transpose(0,2,1,3,5,4,6)
    eris._kccsd_eris_j.vvvv = vvvv.transpose(0,2,1,3,5,4,6)
    eris._kccsd_eris_k = copy.copy(_kccsd_eris)
    eris._kccsd_eris_k.oooo = oooo.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)
    eris._kccsd_eris_k.ooov = ooov.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)
    eris._kccsd_eris_k.oovo = ovoo.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)
    eris._kccsd_eris_k.ovov = voov.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)
    eris._kccsd_eris_k.oovv = ovov.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)
    eris._kccsd_eris_k.ovvv = vvov.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)
    eris._kccsd_eris_k.voov = oovv.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)
    eris._kccsd_eris_k.vvvv = vvvv.transpose(2,1,0,5,4,3,6).transpose(0,2,1,3,5,4,6)

    log.timer('CCSD integral transformation', *cput0)
    return eris

def _make_df_eris(cc, mo_coeff=None):
    from pyscf.ao2mo import _ao2mo
    if cc._scf.with_df._cderi is None:
        cc._scf.with_df.build()

    eris = _make_eris_incore(cc, mo_coeff)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    nao = cc._scf.cell.nao_nr()
    mo_kpts_a, mo_kpts_b = eris.mo_coeff

    kpts = cc.kpts
    nkpts = len(kpts)
    naux = cc._scf.with_df.get_naoaux()
    if gamma_point(kpts):
        dtype = np.double
    else:
        dtype = np.complex128
    dtype = np.result_type(dtype, *mo_kpts_a)
    eris.Lpv = np.empty((nkpts,nkpts,naux,nmoa,nvira), dtype=dtype)
    eris.LPV = np.empty((nkpts,nkpts,naux,nmob,nvirb), dtype=dtype)

    with h5py.File(cc._scf.with_df._cderi, 'r') as f:
        kptij_lst = f['j3c-kptij'].value
        tao = []
        ao_loc = None
        for ki, kpti in enumerate(kpts):
            for kj, kptj in enumerate(kpts):
                kpti_kptj = np.array((kpti,kptj))
                k_id = member(kpti_kptj, kptij_lst)
                if len(k_id) > 0:
                    Lpq = np.asarray(f['j3c/' + str(k_id[0])])
                else:
                    kptji = kpti_kptj[[1,0]]
                    k_id = member(kptji, kptij_lst)
                    Lpq = np.asarray(f['j3c/' + str(k_id[0])])
                    Lpq = lib.transpose(Lpq.reshape(naux,nao,nao), axes=(0,2,1))
                    Lpq = Lpq.conj()

                mo_a = np.hstack((mo_kpts_a[ki], mo_kpts_a[kj][:,nocca:]))
                mo_b = np.hstack((mo_kpts_b[ki], mo_kpts_b[kj][:,noccb:]))
                mo_a = np.asarray(mo_a, dtype=dtype, order='F')
                mo_b = np.asarray(mo_b, dtype=dtype, order='F')
                if dtype == np.double:
                    _ao2mo.nr_e2(Lpq, mo_a, (0, nmoa, nmoa, nmoa+nvira), aosym='s2',
                                 out=eris.Lpv[ki,kj])
                    _ao2mo.nr_e2(Lpq, mo_b, (0, nmob, nmob, nmob+nvirb), aosym='s2',
                                 out=eris.LPV[ki,kj])
                else:
                    if Lpq.size != naux*nao**2: # aosym = 's2'
                        Lpq = lib.unpack_tril(Lpq).astype(np.complex128)
                    _ao2mo.r_e2(Lpq, mo_a, (0, nmoa, nmoa, nmoa+nvira), tao, ao_loc,
                                out=eris.Lpv[ki,kj])
                    _ao2mo.r_e2(Lpq, mo_b, (0, nmob, nmob, nmob+nvirb), tao, ao_loc,
                                out=eris.LPV[ki,kj])
    return eris

def _eri_spin2spatial(chemist_eri_spin, vvvv, eris, cross_ab=False):
    orbspin = eris._kccsd_eris.orbspin
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

def _eri_spatial2spin(eri_aa_ab_ba_bb, vvvv, eris, cross_ab=False):
    orbspin = eris._kccsd_eris.orbspin
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


if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc
    from pyscf import lo

    cell = gto.Cell()
    cell.atom='''
    He 0.000000000000   0.000000000000   0.000000000000
    He 1.685068664391   1.685068664391   1.685068664391
    '''
    #cell.basis = [[0, (1., 1.)], [1, (.5, 1.)]]
    cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.build()

    np.random.seed(2)
    # Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
    kmf = scf.KUHF(cell, kpts=cell.make_kpts([1,1,3]), exxdiv=None)
    nmo = cell.nao_nr()
    kmf.mo_occ = np.zeros((2,3,nmo))
    kmf.mo_occ[0,:,:3] = 1
    kmf.mo_occ[1,:,:1] = 1
    kmf.mo_energy = np.arange(nmo) + np.random.random((2,3,nmo)) * .3
    kmf.mo_energy[kmf.mo_occ == 0] += 2

    mo = (np.random.random((2,3,nmo,nmo)) +
          np.random.random((2,3,nmo,nmo))*1j - .5-.5j)
    s = kmf.get_ovlp()
    kmf.mo_coeff = np.empty_like(mo)
    nkpts = len(kmf.kpts)
    for k in range(nkpts):
        kmf.mo_coeff[0,k] = lo.orth.vec_lowdin(mo[0,k], s[k])
        kmf.mo_coeff[1,k] = lo.orth.vec_lowdin(mo[1,k], s[k])

    def rand_t1_t2(mycc):
        nkpts = mycc.nkpts
        nocca, noccb = mycc.nocc
        nmoa, nmob = mycc.nmo
        nvira, nvirb = nmoa - nocca, nmob - noccb
        np.random.seed(1)
        t1a = (np.random.random((nkpts,nocca,nvira)) +
               np.random.random((nkpts,nocca,nvira))*1j - .5-.5j)
        t1b = (np.random.random((nkpts,noccb,nvirb)) +
               np.random.random((nkpts,noccb,nvirb))*1j - .5-.5j)
        t2aa = (np.random.random((nkpts,nkpts,nkpts,nocca,nocca,nvira,nvira)) +
                np.random.random((nkpts,nkpts,nkpts,nocca,nocca,nvira,nvira))*1j - .5-.5j)
        kconserv = kpts_helper.get_kconserv(kmf.cell, kmf.kpts)
        t2aa = t2aa - t2aa.transpose(1,0,2,4,3,5,6)
        tmp = t2aa.copy()
        for ki, kj, kk in kpts_helper.loop_kkk(nkpts):
            kl = kconserv[ki, kk, kj]
            t2aa[ki,kj,kk] = t2aa[ki,kj,kk] - tmp[ki,kj,kl].transpose(0,1,3,2)
        t2ab = (np.random.random((nkpts,nkpts,nkpts,nocca,noccb,nvira,nvirb)) +
                np.random.random((nkpts,nkpts,nkpts,nocca,noccb,nvira,nvirb))*1j - .5-.5j)
        t2bb = (np.random.random((nkpts,nkpts,nkpts,noccb,noccb,nvirb,nvirb)) +
                np.random.random((nkpts,nkpts,nkpts,noccb,noccb,nvirb,nvirb))*1j - .5-.5j)
        t2bb = t2bb - t2bb.transpose(1,0,2,4,3,5,6)
        tmp = t2bb.copy()
        for ki, kj, kk in kpts_helper.loop_kkk(nkpts):
            kl = kconserv[ki, kk, kj]
            t2bb[ki,kj,kk] = t2bb[ki,kj,kk] - tmp[ki,kj,kl].transpose(0,1,3,2)

        t1 = (t1a, t1b)
        t2 = (t2aa, t2ab, t2bb)
        return t1, t2

    mycc = KUCCSD(kmf)
    eris = mycc.ao2mo()
    t1, t2 = rand_t1_t2(mycc)
    Ht1, Ht2 = mycc.update_amps(t1, t2, eris)
    print(lib.finger(Ht1[0]) - (-1.2692088297292825-12.893074780897923j))
    print(lib.finger(Ht1[1]) - (-11.831413366451148+19.95758532598137j ))
    print(lib.finger(Ht2[0])*1e-2 - (0.97436765562779959 +0.16548728742427826j ))
    print(lib.finger(Ht2[1])*1e-2 - (-1.7752605990115735 +4.2106261874056212j  ))
    print(lib.finger(Ht2[2])*1e-3 - (-0.52223406190978494-0.91888685193234421j))

    kmf.mo_occ[:] = 0
    kmf.mo_occ[:,:,:2] = 1
    mycc = KUCCSD(kmf)
    eris = mycc.ao2mo()
    t1, t2 = rand_t1_t2(mycc)
    Ht1, Ht2 = mycc.update_amps(t1, t2, eris)
    print(lib.finger(Ht1[0]) - (3.7571382837650931+3.6719235677672519j))
    print(lib.finger(Ht1[1])*1e-2 - (-0.42270622344333642+0.65025799860663025j))
    print(lib.finger(Ht2[0])*1e-2 - (2.5124103335695689  -1.3180553113575906j ))
    print(lib.finger(Ht2[1])*1e-2 - (-2.4427382960124304 +0.15329780363467621j))
    print(lib.finger(Ht2[2])*1e-2 - (3.0683780903085842  +2.580910132273615j  ))

    from pyscf.pbc.cc import kccsd
    kgcc = kccsd.GCCSD(scf.addons.convert_to_ghf(kmf))
    kccsd_eris = kccsd._make_eris_incore(kgcc, kgcc._scf.mo_coeff)
    r1 = kgcc.spatial2spin(t1)
    r2 = kgcc.spatial2spin(t2)
    r1, r2 = kgcc.update_amps(r1, r2, kccsd_eris)
    print(abs(r1 - kgcc.spatial2spin(Ht1)).max())
    print(abs(r2 - kgcc.spatial2spin(Ht2)).max())
    exit()

    kmf = kmf.density_fit(auxbasis=[[0, (1., 1.)], [0, (.5, 1.)]])
    mycc = KUCCSD(kmf)
    eris = _make_df_eris(mycc, mycc.mo_coeff)
    t1, t2 = rand_t1_t2(mycc)
    Ht1, Ht2 = mycc.update_amps(t1, t2, eris)
    print(lib.finger(Ht1[0]) - (3.6569734813260473 +3.8092774902489754j))
    print(lib.finger(Ht1[1]) - (-105.8651917884019 +219.86020519421155j))
    print(lib.finger(Ht2[0]) - (-265.25767382882208+215.41888861285341j))
    print(lib.finger(Ht2[1]) - (-115.13953446128346-49.303887916188629j))
    print(lib.finger(Ht2[2]) - (122.51835547779413 +33.85757422327751j ))
