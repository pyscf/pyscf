#!/usr/bin/env python
# Copyright 2022-2023 The PySCF Developers. All Rights Reserved.
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
# Authors: Xing Zhang <zhangxing.nju@gmail.com>
#

from functools import reduce
import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.lib import ktensor
from pyscf.pbc.lib.kpts import MORotationMatrix, KQuartets

from pyscf.pbc.df import GDF, RSGDF
from pyscf.pbc.mp.kmp2 import (
    get_nocc,
    padded_mo_coeff,
    padding_k_idx,
)
from pyscf.pbc.cc import kintermediates_rhf_ksymm as imdk
from pyscf.pbc.cc.kccsd_rhf import (
    RCCSD,
    _get_epq,
    _init_df_eris,
)

einsum = lib.einsum

def update_amps(cc, t1, t2, eris):
    kpts = cc.kpts
    kqrts = cc.kqrts
    rmat = cc.rmat
    kconserv = cc.khelper.kconserv

    nkpts, nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = [e[:nocc] for e in eris.mo_energy]
    mo_e_v = [e[nocc:] for e in eris.mo_energy]

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(cc, kind="split")

    ki_ibz_bz = kpts.ibz2bz[np.arange(kpts.nkpts_ibz)]
    fov = fock[:, :nocc, nocc:]
    kconserv = cc.khelper.kconserv

    Foo = imdk.cc_Foo(kpts, kqrts, t1, t2, eris, rmat)
    Fvv = imdk.cc_Fvv(kpts, kqrts, t1, t2, eris, rmat)
    Fov = imdk.cc_Fov(kpts, kqrts, t1, t2, eris, rmat)
    Loo = imdk.Loo(kpts, kqrts, t1, t2, eris, rmat)
    Lvv = imdk.Lvv(kpts, kqrts, t1, t2, eris, rmat)

    Fov = Fov.todense()

    # Move energy terms to the other side
    for ki_ibz in range(kpts.nkpts_ibz):
        ki = kpts.ibz2bz[ki_ibz]
        Foo[ki][np.diag_indices(nocc)] -= mo_e_o[ki]
        Fvv[ki][np.diag_indices(nvir)] -= mo_e_v[ki]
        Loo[ki][np.diag_indices(nocc)] -= mo_e_o[ki]
        Lvv[ki][np.diag_indices(nvir)] -= mo_e_v[ki]

    t1new = ktensor.empty_like(t1)
    t1 = t1.todense()
    t2new = ktensor.empty_like(t2)
    t2 = t2.todense()

    # T1 equation
    for ka_ibz in range(kpts.nkpts_ibz):
        ki = ka = kpts.ibz2bz[ka_ibz]
        t1new[ka]  = fov[ka].conj()
        t1new[ka] += -2. * einsum('kc,ka,ic->ia', fov[ki], t1[ka], t1[ki])
        t1new[ka] += einsum('ac,ic->ia', Fvv[ka], t1[ki])
        t1new[ka] += -einsum('ki,ka->ia', Foo[ki], t1[ka])

    for i, kq in enumerate(kqrts.kqrts_ibz):
        ki, kk, kc, kd = kq
        ka = ki

        Svovv = 2 * eris.vovv[ka, kk, kc] - eris.vovv[ka, kk, kd].transpose(0, 1, 3, 2)
        tau_term_1 = t2[ki, kk, kc].copy()
        if ki == kc and kk == kd:
            tau_term_1 += einsum('ic,kd->ikcd', t1[ki], t1[kk])
        fock = einsum('akcd,ikcd->ia', Svovv, tau_term_1)
        for _, iop in kqrts.loop_stabilizer(i):
            rmat_oo = rmat.oo[ka][iop]
            rmat_vv = rmat.vv[ka][iop]
            t1new[ka] += einsum('ia,im,ae->me', fock, rmat_oo, rmat_vv.conj())

    for i, kq in enumerate(kqrts.kqrts_ibz):
        kk, kl, ki, kd = kq
        ka = ki
        Sooov = 2 * eris.ooov[kk, kl, ki] - eris.ooov[kl, kk, ki].transpose(1, 0, 2, 3)
        tau_term_1 = t2[kk, kl, ka].copy()
        if kk == ka and kl == kc:
            tau_term_1 += einsum('ka,lc->klac', t1[ka], t1[kc])
        fock = -einsum('klic,klac->ia', Sooov, tau_term_1)

        op_group = kqrts.stars_ops[i]
        ka_prim = kpts.k2opk[ka, op_group]
        mask = np.isin(ka_prim, ki_ibz_bz)
        for iop, ka_p in zip(op_group[mask], ka_prim[mask]):
            rmat_oo = rmat.oo[ka][iop]
            rmat_vv = rmat.vv[ka][iop]
            t1new[ka_p] += einsum('ia,im,ae->me', fock, rmat_oo, rmat_vv.conj())

    for i, kq in enumerate(kqrts.kqrts_ibz):
        ki, kk, ka, kc = kq
        if ka == ki and kk == kc:
            tau_term = 2 * t2[kk, ki, kk] - t2[ki, kk, kk].transpose(1, 0, 2, 3)
            if ki == kk:
                tau_term += einsum('ic,ka->kica', t1[ki], t1[ka])

            fock = einsum('kc,kica->ia', Fov[kc], tau_term)
            fock += einsum('akic,kc->ia', 2 * eris.voov[ka, kk, ki], t1[kc])
            fock += einsum('kaic,kc->ia', -eris.ovov[kk, ka, ki], t1[kc])

            for _, iop in kqrts.loop_stabilizer(i):
                rmat_oo = rmat.oo[ka][iop]
                rmat_vv = rmat.vv[ka][iop]
                t1new[ka] += einsum('ia,im,ae->me', fock, rmat_oo, rmat_vv.conj())

    for ki_ibz in range(kpts.nkpts_ibz):
        ka = ki = kpts.ibz2bz[ki_ibz]
        # Remove zero/padded elements from denominator
        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                       [0,nvir,ka,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        t1new[ki] /= eia


    # T2 equation
    Loo = Loo.todense()
    Lvv = Lvv.todense()

    for i, kq in enumerate(kqrts.kqrts_ibz):
        ki, kj, ka, kb = kq
        t2new[ki, kj, ka] = eris.oovv[ki, kj, ka].conj()

    mem_now = lib.current_memory()[0]
    if (cc.incore_complete or
        _memory_4d(cc, [nocc,]*4) + mem_now < cc.max_memory * .9):
        Woooo = imdk.cc_Woooo(kpts, kqrts, t1, t2, eris, rmat)
    else:
        metadata = {'kpts': kpts, 'kqrts': kqrts, 'rmat': rmat,
                    'label': 'oooo', 'trans': 'ccnn',
                    'incore': False}
        Woooo = ktensor.empty([nocc,]*4, dtype=t1.dtype, metadata=metadata)
        Woooo = imdk.cc_Woooo(kpts, kqrts, t1, t2, eris, rmat, Woooo)

    mem_now = lib.current_memory()[0]
    if (not cc.ktensor_direct and
        _memory_4d(cc, [nocc,]*4, False) + mem_now < cc.max_memory * .9):
        Woooo = Woooo.todense()

    def _t2_oooo(ki,kj,ka,kb):
        t2new_tmp = 0
        for kl in range(nkpts):
            kk = kconserv[kj, kl, ki]
            tau_term = t2[kk, kl, ka].copy()
            if kl == kb and kk == ka:
                tau_term += einsum('ic,jd->ijcd', t1[ka], t1[kb])
            t2new_tmp += 0.5 * einsum('klij,klab->ijab', Woooo[kk, kl, ki], tau_term)
        return t2new_tmp

    for i, kq in enumerate(kqrts.kqrts_ibz):
        ki, kj, ka, kb = kq
        t2new_tmp = _t2_oooo(ki,kj,ka,kb)
        t2new[ki, kj, ka] += t2new_tmp
        if lib.isin_1d((kj,ki,kb,ka), kqrts.kqrts_ibz):
            t2new[kj, ki, kb] += t2new_tmp.transpose(1, 0, 3, 2)
        else:
            t2new_tmp = _t2_oooo(kj,ki,kb,ka)
            t2new[ki, kj, ka] += t2new_tmp.transpose(1, 0, 3, 2)

    Woooo = None

    add_vvvv_(cc, t2new, t1, t2, eris)

    def _t2_voov1(ki,kj,ka,kb):
        t2new_tmp = einsum('ac,ijcb->ijab', Lvv[ka], t2[ki, kj, ka])
        t2new_tmp += einsum('ki,kjab->ijab', -Loo[ki], t2[ki, kj, ka])

        kc = kj
        tmp2 = np.asarray(eris.vovv[kc, ki, kb]).transpose(3, 2, 1, 0).conj() \
               - einsum('kbic,ka->abic', eris.ovov[ka, kb, ki], t1[ka])
        t2new_tmp += einsum('abic,jc->ijab', tmp2, t1[kj])

        kk = kb
        tmp2 = np.asarray(eris.ooov[kj, ki, kk]).transpose(3, 2, 1, 0).conj() \
               + einsum('akic,jc->akij', eris.voov[ka, kk, ki], t1[kj])
        t2new_tmp -= einsum('akij,kb->ijab', tmp2, t1[kb])
        return t2new_tmp

    for i, kq in enumerate(kqrts.kqrts_ibz):
        ki, kj, ka, kb = kq
        t2new_tmp = _t2_voov1(ki,kj,ka,kb)
        t2new[ki, kj, ka] += t2new_tmp
        if lib.isin_1d((kj,ki,kb,ka), kqrts.kqrts_ibz):
            t2new[kj, ki, kb] += t2new_tmp.transpose(1, 0, 3, 2)
        else:
            t2new_tmp = _t2_voov1(kj,ki,kb,ka)
            t2new[ki, kj, ka] += t2new_tmp.transpose(1, 0, 3, 2)

    mem_now = lib.current_memory()[0]
    if (cc.incore_complete or
        _memory_4d(cc, [nocc,nocc,nvir,nvir])*2 + mem_now < cc.max_memory*.9):
        Wvoov = imdk.cc_Wvoov(kpts, kqrts, t1, t2, eris, rmat)
        Wvovo = imdk.cc_Wvovo(kpts, kqrts, t1, t2, eris, rmat)
    else:
        metadata = {'kpts': kpts, 'kqrts': kqrts, 'rmat': rmat,
                    'trans': 'ccnn', 'incore': False}
        Wvoov = ktensor.empty([nvir,nocc,nocc,nvir], dtype=t1.dtype,
                              metadata={**metadata, 'label':'voov'})
        Wvovo = ktensor.empty([nvir,nocc,nvir,nocc], dtype=t1.dtype,
                              metadata={**metadata, 'label':'vovo'})
        Wvoov = imdk.cc_Wvoov(kpts, kqrts, t1, t2, eris, rmat, Wvoov)
        Wvovo = imdk.cc_Wvovo(kpts, kqrts, t1, t2, eris, rmat, Wvovo)

    mem_now = lib.current_memory()[0]
    if (not cc.ktensor_direct and
        _memory_4d(cc, [nocc,nocc,nvir,nvir], False)*2 + mem_now < cc.max_memory*.9):
        Wvoov = Wvoov.todense()
        Wvovo = Wvovo.todense()

    def _t2_voov2(ki,kj,ka,kb):
        t2new_tmp = 0
        for kk in range(nkpts):
            kc = kconserv[ka, ki, kk]
            tmp_voov = 2. * Wvoov[ka, kk, ki] - Wvovo[ka, kk, kc].transpose(0, 1, 3, 2)
            t2new_tmp += einsum('akic,kjcb->ijab', tmp_voov, t2[kk, kj, kc])

            kc = kconserv[ka, ki, kk]
            t2new_tmp -= einsum('akic,kjbc->ijab', Wvoov[ka, kk, ki], t2[kk, kj, kb])

            kc = kconserv[kk, ka, kj]
            t2new_tmp -= einsum('bkci,kjac->ijab', Wvovo[kb, kk, kc], t2[kk, kj, ka])
        return t2new_tmp

    for i, kq in enumerate(kqrts.kqrts_ibz):
        ki, kj, ka, kb = kq
        t2new_tmp = _t2_voov2(ki,kj,ka,kb)
        t2new[ki, kj, ka] += t2new_tmp
        if lib.isin_1d((kj,ki,kb,ka), kqrts.kqrts_ibz):
            t2new[kj, ki, kb] += t2new_tmp.transpose(1, 0, 3, 2)
        else:
            t2new_tmp = _t2_voov2(kj,ki,kb,ka)
            t2new[ki, kj, ka] += t2new_tmp.transpose(1, 0, 3, 2)

    Wvoov = Wvovo = None

    for i, kq in enumerate(kqrts.kqrts_ibz):
        ki, kj, ka, kb = kq
        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                       [0,nvir,ka,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                       [0,nvir,kb,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        eijab = eia[:, None, :, None] + ejb[:, None, :]
        t2new[ki, kj, ka] /= eijab

    return t1new, t2new

def add_vvvv_(cc, Ht2, t1, t2, eris):
    kpts = cc.kpts
    kqrts = cc.kqrts
    rmat = cc.rmat

    nocc = cc.nocc
    nmo = cc.nmo
    nvir = nmo - nocc
    nkpts = kpts.nkpts
    kconserv = cc.khelper.kconserv

    mem_now = lib.current_memory()[0]
    if (not cc.incore_complete and
        cc.direct and getattr(eris, 'Lpv', None) is not None):
        def get_Wvvvv(ka, kb, kc):
            Lpv = eris.Lpv
            kd = kconserv[ka, kc, kb]
            Lbd = (Lpv[kb,kd][:,nocc:] -
                   lib.einsum('Lkd,kb->Lbd', Lpv[kb,kd][:,:nocc], t1[kb]))
            Wvvvv = lib.einsum('Lac,Lbd->abcd', Lpv[ka,kc][:,nocc:], Lbd)
            Lbd = None
            kcbd = lib.einsum('Lkc,Lbd->kcbd', Lpv[ka,kc][:,:nocc],
                              Lpv[kb,kd][:,nocc:])
            Wvvvv -= lib.einsum('kcbd,ka->abcd', kcbd, t1[ka])
            Wvvvv *= (1. / nkpts)
            return Wvvvv
    elif (cc.incore_complete or
          _memory_4d(cc, [nvir,]*4) + mem_now < cc.max_memory * .9):
        _Wvvvv = imdk.cc_Wvvvv(kpts, kqrts, t1, t2, eris, rmat)

        mem_now = lib.current_memory()[0]
        if (not cc.ktensor_direct and
            _memory_4d(cc, [nvir,]*4, False) + mem_now < cc.max_memory * .9):
            _Wvvvv = _Wvvvv.todense()

        get_Wvvvv = lambda ka, kb, kc: _Wvvvv[ka, kb, kc]
    else:
        metadata = {'kpts': kpts, 'kqrts': kqrts, 'rmat': rmat,
                    'label': 'vvvv', 'trans': 'ccnn',
                    'incore': False}
        _Wvvvv = ktensor.empty([nvir,]*4, dtype=t1.dtype, metadata=metadata)
        _Wvvvv = imdk.cc_Wvvvv(kpts, kqrts, t1, t2, eris, rmat, _Wvvvv)

        mem_now = lib.current_memory()[0]
        if (not cc.ktensor_direct and
            _memory_4d(cc, [nvir,]*4, False) + mem_now < cc.max_memory * .9):
            _Wvvvv = _Wvvvv.todense()

        get_Wvvvv = lambda ka, kb, kc: _Wvvvv[ka, kb, kc]

    kakb, igroup = np.unique(kqrts.kqrts_ibz[:,2:], axis=0, return_inverse=True)
    for i in range(np.amax(igroup) + 1):
        ka, kb = kakb[i]
        idx = np.where(igroup==i)[0]

        for kc in range(nkpts):
            kd = kconserv[ka, kc, kb]
            Wvvvv = get_Wvvvv(ka, kb, kc)
            for m in idx:
                ki,kj,kaa,kbb = kqrts.kqrts_ibz[m]
                assert kaa==ka and kbb==kb
                tau = t2[ki, kj, kc].copy()
                if ki == kc and kj == kd:
                    tau += np.einsum('ic,jd->ijcd', t1[ki], t1[kj])
                Ht2[ki, kj, ka] += einsum('abcd,ijcd->ijab', Wvvvv, tau)

    _Wvvvv = None
    return Ht2

def energy(cc, t1, t2, eris):
    kpts = cc.kpts
    kqrts = cc.kqrts

    nkpts, nocc, nvir = t1.shape
    fock = eris.fock
    e = 0

    for ki_ibz in range(kpts.nkpts_ibz):
        ki = kpts.ibz2bz[ki_ibz]
        weight = kpts.weights_ibz[ki_ibz]
        e += 2 * einsum('ia,ia', fock[ki,:nocc,nocc:], t1[ki]) * weight

    tau = ktensor.empty_like(t2)
    for kq in kqrts.kqrts_ibz:
        ki, kj, ka, kb = kq
        tau[ki, kj, ka] = t2[ki, kj, ka]
        if ki == ka and kj == kb:
            tau[ki, kj, ka] += einsum('ia,jb->ijab', t1[ki], t1[kj])

    kq_weights = kqrts.weights_ibz
    for k, kq in enumerate(kqrts.kqrts_ibz):
        ki, kj, ka, kb = kq
        weight = kq_weights[k] * nkpts**3
        e += 2 * einsum('ijab,ijab', tau[ki, kj, ka], eris.oovv[ki, kj, ka]) * weight
        e += -einsum('ijab,ijba', tau[ki, kj, ka], eris.oovv[ki, kj, kb]) * weight

    e /= nkpts
    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in KRCCSD energy %s', e)
    return e.real


class KsymAdaptedRCCSD(RCCSD):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        '''
        Attributes:
            ktensor_direct : bool
                If set to True, the tensors will be stored as block-sparse,
                and the symmetry related blocks are computed on-the-fly when needed.
                Otherwise, the tensors will be converted to dense tensors whenever
                there is enough memory. Default is False.
            eris_outcore : bool
                If set to True, the integrals will be always stored on the disk.
                Otherwise, whether the integrals are stored on the disk or in memory
                depends on the available memory size. Default is False.
        '''
        # NOTE self._scf is a non-symmetry object, see RCCSD.__init__
        RCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.kqrts = KQuartets(self.kpts).build()
        self.rmat = None
        self.ktensor_direct = False
        self.eris_outcore = False

        keys = set(['kqrts', 'rmat', 'ktensor_direct', 'eris_outcore'])
        self._keys = self._keys.union(keys)

    def ao2mo(self, mo_coeff=None):
        eris = _PhysicistsERIs()
        eris._common_init_(self, mo_coeff)

        # use padded mo_coeff to construct the rotation matrix
        self.rmat = MORotationMatrix(self.kpts, eris.mo_coeff,
                                     self._scf.get_ovlp(), eris.nocc, eris.nmo)
        self.rmat.build()

        mem_now = lib.current_memory()[0]
        mem_incore = _mem_usage(self)
        if not self.eris_outcore and (
                self.incore_complete or mem_incore + mem_now < self.max_memory):
            eris = _make_eris_incore(self, eris, self._scf.with_df.ao2mo)
        else:
            eris = _make_eris_outcore(self, eris, self._scf.with_df.ao2mo)
        return eris

    def init_amps(self, eris):
        time0 = logger.process_clock(), logger.perf_counter()
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts

        kpts = self.kpts
        kqrts = self.kqrts
        rmat = self.rmat
        assert rmat is not None

        metadata = {'kpts': kpts, 'rmat': rmat,
                    'label': 'ov', 'trans': 'nc', 'incore': True}
        t1 = ktensor.zeros((nocc, nvir), dtype=eris.fock.dtype,
                           metadata=metadata)

        metadata = {'kpts': kpts, 'kqrts': kqrts, 'rmat': rmat,
                    'label': 'oovv', 'trans': 'nncc', 'incore': True}
        t2 = ktensor.empty((nocc,nocc,nvir,nvir), dtype=eris.fock.dtype,
                           metadata=metadata)
        mo_e_o = [eris.mo_energy[k][:nocc] for k in range(nkpts)]
        mo_e_v = [eris.mo_energy[k][nocc:] for k in range(nkpts)]

        # Get location of padded elements in occupied and virtual space
        nonzero_opadding, nonzero_vpadding = padding_k_idx(self, kind="split")

        emp2 = 0
        for i, kq in enumerate(kqrts.kqrts_ibz):
            ki, kj, ka, kb = kq
            weight = kqrts.weights_ibz[i] * nkpts**3

            eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                           [0,nvir,ka,mo_e_v,nonzero_vpadding],
                           fac=[1.0,-1.0])

            ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                           [0,nvir,kb,mo_e_v,nonzero_vpadding],
                           fac=[1.0,-1.0])
            eijab = eia[:, None, :, None] + ejb[:, None, :]

            eris_ijab = eris.oovv[ki, kj, ka]
            eris_ijba = eris.oovv[ki, kj, kb]
            t2[ki, kj, ka] = eris_ijab.conj() / eijab
            woovv = 2 * eris_ijab - eris_ijba.transpose(0, 1, 3, 2)
            emp2 += np.einsum('ijab,ijab', t2[ki, kj, ka], woovv) * weight

        self.emp2 = emp2.real / nkpts
        logger.info(self, 'Init t2, MP2 energy (with fock eigenvalue shift) = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2

    def amplitudes_to_vector(self, t1, t2):
        t1_raw = np.asarray(getattr(t1, 'data', t1))
        t2_raw = np.asarray(getattr(t2, 'data', t2))
        return np.concatenate((t1_raw, t2_raw), axis=None)

    def vector_to_amplitudes(self, vec):
        kpts = self.kpts
        kqrts = self.kqrts
        rmat = self.rmat
        nocc = self.nocc
        nvir = self.nmo - nocc
        t1_size = kpts.nkpts_ibz * nocc * nvir
        t1_flat = vec[:t1_size]
        t2_flat = vec[t1_size:]

        metadata = {'kpts': kpts, 'rmat': rmat,
                    'label': 'ov', 'trans': 'nc', 'incore': True}
        t1 = ktensor.fromraw(t1_flat, (nocc,nvir), dtype=vec.dtype,
                             metadata=metadata)

        metadata = {'kpts': kpts, 'kqrts': kqrts, 'rmat': rmat,
                    'label': 'oovv', 'trans': 'nncc', 'incore': True}
        t2 = ktensor.fromraw(t2_flat, (nocc,nocc,nvir,nvir), dtype=vec.dtype,
                             metadata=metadata)
        return t1, t2

    energy = energy
    update_amps = update_amps


def _make_eris_incore(cc, eris, fao2mo):
    log = logger.Logger(cc.stdout, cc.verbose)
    log.info('using incore ERI storage')
    cput0 = (logger.process_clock(), logger.perf_counter())

    mo_coeff = eris.mo_coeff
    nocc = eris.nocc
    nvir = eris.nvir
    nmo = eris.nmo
    dtype = eris.dtype

    common_metadata = {'kpts'  : cc.kpts,
                       'kqrts' : cc.kqrts,
                       'rmat'  : cc.rmat,
                       'trans' : 'ccnn',
                       'incore': True}

    eris.oooo = ktensor.empty([nocc,nocc,nocc,nocc], dtype=dtype,
                              metadata={**common_metadata, 'label': 'oooo'})
    eris.ooov = ktensor.empty([nocc,nocc,nocc,nvir], dtype=dtype,
                              metadata={**common_metadata, 'label': 'ooov'})
    eris.oovv = ktensor.empty([nocc,nocc,nvir,nvir], dtype=dtype,
                              metadata={**common_metadata, 'label': 'oovv'})
    eris.ovov = ktensor.empty([nocc,nvir,nocc,nvir], dtype=dtype,
                              metadata={**common_metadata, 'label': 'ovov'})
    eris.voov = ktensor.empty([nvir,nocc,nocc,nvir], dtype=dtype,
                              metadata={**common_metadata, 'label': 'voov'})
    eris.vovv = ktensor.empty([nvir,nocc,nvir,nvir], dtype=dtype,
                              metadata={**common_metadata, 'label': 'vovv'})
    eris.vvvv = ktensor.empty([nvir,nvir,nvir,nvir], dtype=dtype,
                              metadata={**common_metadata, 'label': 'vvvv'})

    kpts = cc.kpts.kpts
    nkpts = len(kpts)
    kqrts = cc.kqrts
    khelper = cc.khelper
    kconserv = khelper.kconserv
    kptlist = kqrts.kqrts_ibz[:,:3][:,[0,2,1]] #chemists' notation
    khelper.build_symm_map(kptlist=kptlist)

    for (iki, ika, ikj) in khelper.symm_map.keys():
        ikb = kconserv[iki, ika, ikj]
        eri_kpt = fao2mo((mo_coeff[iki],mo_coeff[ika],mo_coeff[ikj],mo_coeff[ikb]),
                         (kpts[iki],kpts[ika],kpts[ikj],kpts[ikb]), compact=False)

        if np.issubdtype(dtype, np.floating):
            eri_kpt = eri_kpt.real
        eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo)

        for (ki, ka, kj) in khelper.symm_map[(iki, ika, ikj)]:
            eri_kpt_symm = khelper.transform_symm(eri_kpt, ki, ka, kj).transpose(0, 2, 1, 3)
            eris.oooo[ki, kj, ka] = eri_kpt_symm[:nocc, :nocc, :nocc, :nocc] / nkpts
            eris.ooov[ki, kj, ka] = eri_kpt_symm[:nocc, :nocc, :nocc, nocc:] / nkpts
            eris.oovv[ki, kj, ka] = eri_kpt_symm[:nocc, :nocc, nocc:, nocc:] / nkpts
            eris.ovov[ki, kj, ka] = eri_kpt_symm[:nocc, nocc:, :nocc, nocc:] / nkpts
            eris.voov[ki, kj, ka] = eri_kpt_symm[nocc:, :nocc, :nocc, nocc:] / nkpts
            eris.vovv[ki, kj, ka] = eri_kpt_symm[nocc:, :nocc, nocc:, nocc:] / nkpts
            eris.vvvv[ki, kj, ka] = eri_kpt_symm[nocc:, nocc:, nocc:, nocc:] / nkpts

    if not cc.ktensor_direct:
        eris.oooo = eris.oooo.todense()
        eris.ooov = eris.ooov.todense()
        eris.oovv = eris.oovv.todense()
        eris.ovov = eris.ovov.todense()
        eris.voov = eris.voov.todense()
        eris.vovv = eris.vovv.todense()
        eris.vvvv = eris.vvvv.todense()

    log.timer('CCSD integral transformation', *cput0)
    return eris

def _make_eris_outcore(cc, eris, fao2mo):
    log = logger.Logger(cc.stdout, cc.verbose)
    log.info('using outcore ERI storage')
    cput0 = (logger.process_clock(), logger.perf_counter())

    mo_coeff = eris.mo_coeff
    nocc = eris.nocc
    nvir = eris.nvir
    nmo = eris.nmo
    dtype = eris.dtype

    common_metadata = {'kpts'  : cc.kpts,
                       'kqrts' : cc.kqrts,
                       'rmat'  : cc.rmat,
                       'trans' : 'ccnn',
                       'incore': False}

    eris.oooo = ktensor.empty([nocc,nocc,nocc,nocc], dtype=dtype,
                              metadata={**common_metadata, 'label': 'oooo'})
    eris.ooov = ktensor.empty([nocc,nocc,nocc,nvir], dtype=dtype,
                              metadata={**common_metadata, 'label': 'ooov'})
    eris.oovv = ktensor.empty([nocc,nocc,nvir,nvir], dtype=dtype,
                              metadata={**common_metadata, 'label': 'oovv'})
    eris.ovov = ktensor.empty([nocc,nvir,nocc,nvir], dtype=dtype,
                              metadata={**common_metadata, 'label': 'ovov'})
    eris.voov = ktensor.empty([nvir,nocc,nocc,nvir], dtype=dtype,
                              metadata={**common_metadata, 'label': 'voov'})
    eris.vovv = ktensor.empty([nvir,nocc,nvir,nvir], dtype=dtype,
                              metadata={**common_metadata, 'label': 'vovv'})

    vvvv_required = (not cc.direct
                     or not isinstance(cc._scf.with_df, (GDF, RSGDF))
                     or cc._scf.cell.dimension == 2)
    if vvvv_required:
        eris.vvvv = ktensor.empty([nvir,nvir,nvir,nvir], dtype=dtype,
                                  metadata={**common_metadata, 'label': 'vvvv'})

    kpts = cc.kpts.kpts
    nkpts = len(kpts)
    kqrts = cc.kqrts

    # <ij|pq>  = (ip|jq)
    cput1 = logger.process_clock(), logger.perf_counter()
    for kprqs in kqrts.kqrts_ibz:
        kp, kr, kq, ks = kprqs
        orbo_p = mo_coeff[kp][:, :nocc]
        orbo_r = mo_coeff[kr][:, :nocc]
        buf_kpt = fao2mo((orbo_p, mo_coeff[kq], orbo_r, mo_coeff[ks]),
                         (kpts[kp], kpts[kq], kpts[kr], kpts[ks]), compact=False)
        if np.issubdtype(dtype, np.floating):
            buf_kpt = buf_kpt.real
        buf_kpt = buf_kpt.reshape(nocc, nmo, nocc, nmo).transpose(0, 2, 1, 3)
        eris.oooo[kp, kr, kq] = buf_kpt[:, :, :nocc, :nocc] / nkpts
        eris.ooov[kp, kr, kq] = buf_kpt[:, :, :nocc, nocc:] / nkpts
        eris.oovv[kp, kr, kq] = buf_kpt[:, :, nocc:, nocc:] / nkpts
    cput1 = log.timer_debug1('transforming oopq', *cput1)

    # <ia|jb> = (ij|ab)
    for kiajb in kqrts.kqrts_ibz:
        ki, ka, kj, kb = kiajb
        orb_i = mo_coeff[ki][:, :nocc]
        orb_a = mo_coeff[ka][:, nocc:]
        orb_j = mo_coeff[kj][:, :nocc]
        orb_b = mo_coeff[kb][:, nocc:]
        buf_kpt = fao2mo((orb_i, orb_j, orb_a, orb_b),
                         (kpts[ki], kpts[kj], kpts[ka], kpts[kb]), compact=False)
        if np.issubdtype(dtype, np.floating):
            buf_kpt = buf_kpt.real
        buf_kpt = buf_kpt.reshape(nocc, nocc, nvir, nvir).transpose(0, 2, 1, 3)
        eris.ovov[ki, ka, kj] = buf_kpt / nkpts
    cput1 = log.timer_debug1('transforming ovov', *cput1)

    # <ai|pq> = (ap|iq)
    for kaipq in kqrts.kqrts_ibz:
        ka, ki, kp, kq = kaipq
        orb_a = mo_coeff[ka][:, nocc:]
        orb_i = mo_coeff[ki][:, :nocc]
        buf_kpt = fao2mo((orb_a, mo_coeff[kp], orb_i, mo_coeff[kq]),
                         (kpts[ka], kpts[kp], kpts[ki], kpts[kq]), compact=False)
        if np.issubdtype(dtype, np.floating):
            buf_kpt = buf_kpt.real
        buf_kpt = buf_kpt.reshape(nvir, nmo, nocc, nmo).transpose(0, 2, 1, 3)
        # TODO: compute vovv on the fly
        eris.vovv[ka, ki, kp] = buf_kpt[:, :, nocc:, nocc:] / nkpts
        eris.voov[ka, ki, kp] = buf_kpt[:, :, :nocc, nocc:] / nkpts
    cput1 = log.timer_debug1('transforming vopq', *cput1)

    mem_now = lib.current_memory()[0]
    if not vvvv_required:
        _init_df_eris(cc, eris)
    elif nvir ** 4 * 16 / 1e6 + mem_now < cc.max_memory:
        khelper = cc.khelper
        kconserv = khelper.kconserv
        kptlist = kqrts.kqrts_ibz[:,:3][:,[0,2,1]] #chemists' notation
        khelper.build_symm_map(kptlist=kptlist)
        for (ikp, ikq, ikr) in khelper.symm_map.keys():
            iks = kconserv[ikp, ikq, ikr]
            orbv_p = mo_coeff[ikp][:, nocc:]
            orbv_q = mo_coeff[ikq][:, nocc:]
            orbv_r = mo_coeff[ikr][:, nocc:]
            orbv_s = mo_coeff[iks][:, nocc:]
            buf_kpt = fao2mo((orbv_p,orbv_q,orbv_r,orbv_s),
                             kpts[[ikp,ikq,ikr,iks]], compact=False)
            if np.issubdtype(dtype, np.floating):
                buf_kpt = buf_kpt.real
            buf_kpt = buf_kpt.reshape([nvir,]*4)
            for (kp, kq, kr) in khelper.symm_map[(ikp, ikq, ikr)]:
                buf_kpt_symm = khelper.transform_symm(buf_kpt, kp, kq, kr).transpose(0, 2, 1, 3)
                eris.vvvv[kp, kr, kq] = buf_kpt_symm / nkpts
    else:
        raise MemoryError(f'Minimal memory requirements '
                          f'{mem_now + nvir ** 4 / 1e6 * 16 * 2} MB')
    cput1 = log.timer_debug1('transforming vvvv', *cput1)

    log.timer('CCSD integral transformation', *cput0)
    return eris


class _PhysicistsERIs():
    def __init__(self, cell=None):
        self.cell = cell
        self.kpts = None
        self.mo_coeff = None
        self.nocc = None
        self.nmo = None
        self.nvir = None
        self.fock = None
        self.dtype = None

        self.oooo = None
        self.ooov = None
        self.oovv = None
        self.ovov = None
        self.voov = None
        self.vovv = None
        self.vvvv = None
        self.Lpv = None

    def _common_init_(self, cc, mo_coeff=None):
        from pyscf.pbc import tools
        from pyscf.pbc.cc.ccsd import _adjust_occ
        mf = cc._scf
        self.cell = cell = mf.cell
        self.kpts = kpts = cc.kpts
        self.nocc = nocc = cc.nocc
        self.nmo = nmo = cc.nmo
        self.nvir = nmo - nocc

        if mo_coeff is None:
            mo_coeff = cc.mo_coeff
        self.dtype = mo_coeff[-1].dtype

        # Re-make our fock MO matrix elements from density and fock AO
        # FIXME what if mo_coeff is not consistent with cc.mo_occ?
        dm = mf.make_rdm1(mo_coeff, cc.mo_occ)
        exxdiv = mf.exxdiv if cc.keep_exxdiv else None
        with lib.temporary_env(mf, exxdiv=exxdiv):
            # _scf.exxdiv affects eris.fock. HF exchange correction should be
            # excluded from the Fock matrix.
            vhf = mf.get_veff(cell, dm)
        fockao = mf.get_hcore() + vhf

        self.mo_coeff = mo_coeff = padded_mo_coeff(cc, mo_coeff)

        fock = np.asarray([reduce(np.dot, (mo.T.conj(), fockao[k], mo))
                          for k, mo in enumerate(mo_coeff)])
        self.fock = fock

        mo_energy = [fock[k].diagonal().real for k in range(len(fock))]
        if not cc.keep_exxdiv:
            # Add HFX correction in the self.mo_energy to improve convergence in
            # CCSD iteration. It is useful for the 2D systems since their occupied and
            # the virtual orbital energies may overlap which may lead to numerical
            # issue in the CCSD iterations.
            # FIXME: Whether to add this correction for other exxdiv treatments?
            # Without the correction, MP2 energy may be largely off the correct value.
            madelung = tools.madelung(cell, kpts.kpts)
            mo_energy = [_adjust_occ(mo_e, nocc, -madelung)
                         for k, mo_e in enumerate(mo_energy)]

        # Get location of padded elements in occupied and virtual space.
        nocc_per_kpt = get_nocc(cc, per_kpoint=True)
        nonzero_padding = padding_k_idx(cc, kind="joint")

        # Check direct and indirect gaps for possible issues with CCSD convergence.
        mo_e = [mo_energy[kp][nonzero_padding[kp]] for kp in range(len(mo_energy))]
        mo_e = np.sort([y for x in mo_e for y in x])  # Sort de-nested array
        gap = mo_e[np.sum(nocc_per_kpt)] - mo_e[np.sum(nocc_per_kpt)-1]
        if gap < 1e-5:
            logger.warn(cc, 'HOMO-LUMO gap %s too small for KCCSD. '
                            'May cause issues in convergence.', gap)
        self.mo_energy = mo_energy


def _mem_usage(cc):
    nocc = cc.nocc
    nmo = cc.nmo
    nvir = nmo - nocc
    nkpts = cc.kpts.nkpts
    #Wvvvv, eris.vvvv
    incore  = _memory_4d(cc, [nvir,]*4) * 2
    #eris.vovv
    incore += _memory_4d(cc, [nvir,nvir,nvir,nocc])
    #Wvoov, Wvovo, eris.oovv, eris.ovov, eris.voov, t2new, t2
    incore += _memory_4d(cc, [nvir,nvir,nocc,nocc]) * 7
    #eris.ooov
    incore += _memory_4d(cc, [nvir,nocc,nocc,nocc])
    #Woooo, eris.oooo
    incore += _memory_4d(cc, [nocc,]*4) * 2

    if not cc.ktensor_direct:
        incore += _memory_4d(cc, [nvir,]*4, False) * 2
        incore += _memory_4d(cc, [nvir,nvir,nvir,nocc], False)
        incore += _memory_4d(cc, [nvir,nvir,nocc,nocc], False) * 5
        incore += _memory_4d(cc, [nvir,nocc,nocc,nocc], False)
        incore += _memory_4d(cc, [nocc,]*4, False) * 2

    #temp
    incore += nkpts * nmo**4 * 16 / 1e6
    #1e, fock, t1
    incore += nkpts * nmo**2 * 16 / 1e6 * 7
    #t2_bz
    incore += _memory_4d(cc, [nvir,nvir,nocc,nocc], False)

    logger.info(cc, f"Incore memory estimation: {incore} MB")
    return incore

def _memory_4d(cc, shape, ibz=True):
    if ibz:
        nk = len(cc.kqrts.kqrts_ibz)
    else:
        nk = cc.kpts.nkpts**3
    return nk * np.prod(shape) * 16 / 1e6
