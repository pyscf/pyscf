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

import numpy as np
from pyscf import lib
from pyscf.pbc.lib import ktensor

einsum = lib.einsum

def cc_Foo(kpts, kqrts, t1, t2, eris, rmat):
    nkpts, nocc, nvir = t1.shape
    #Fki = np.empty((nkpts,nocc,nocc), dtype=t2.dtype)
    metadata = {'kpts': kpts, 'rmat': rmat,
                'label': 'oo', 'trans': 'cn',
                'incore': True}
    Fki = ktensor.empty([nocc,nocc], dtype=t2.dtype, metadata=metadata)

    for i in range(kpts.nkpts_ibz):
        ki = kpts.ibz2bz[i]
        Fki[ki] = eris.fock[ki,:nocc,:nocc].copy()

    for i, kq in enumerate(kqrts.kqrts_ibz):
        ki, kl, kc, kd = kq
        kk = ki
        Soovv = 2 * eris.oovv[kk,kl,kc] - eris.oovv[kk,kl,kd].transpose(0,1,3,2)
        fock = einsum('klcd,ilcd->ki', Soovv, t2[ki,kl,kc])
        if ki == kc:
            fock += einsum('klcd,ic,ld->ki', Soovv, t1[ki], t1[kl])
        for _, iop in kqrts.loop_stabilizer(i):
            rmat_oo = rmat.oo[ki][iop]
            Fki[ki] += einsum('ki,km,in->mn', fock, rmat_oo.conj(), rmat_oo)
    return Fki

def cc_Fov(kpts, kqrts, t1, t2, eris, rmat):
    nkpts, nocc, nvir = t1.shape
    #Fkc = np.empty((nkpts,nocc,nvir), dtype=t2.dtype)
    metadata = {'kpts': kpts, 'rmat': rmat,
                'label': 'ov', 'trans': 'cn',
                'incore': True}
    Fkc = ktensor.empty([nocc,nvir], dtype=t2.dtype, metadata=metadata)

    for i in range(kpts.nkpts_ibz):
        ki = kpts.ibz2bz[i]
        Fkc[ki] = eris.fock[ki,:nocc,nocc:].copy()

    for i, kq in enumerate(kqrts.kqrts_ibz):
        kk, kl, kc, kd = kq
        if kc == kk and kl == kd:
            Soovv = 2 * eris.oovv[kk,kl,kk] - eris.oovv[kk,kl,kl].transpose(0,1,3,2)
            fock = einsum('klcd,ld->kc', Soovv, t1[kl])
            for _, iop in kqrts.loop_stabilizer(i):
                rmat_oo = rmat.oo[kk][iop]
                rmat_vv = rmat.vv[kk][iop]
                Fkc[kk] += einsum('kc,km,cb->mb', fock, rmat_oo.conj(), rmat_vv)
    return Fkc

def cc_Fvv(kpts, kqrts, t1, t2, eris, rmat):
    nkpts, nocc, nvir = t1.shape
    #Fac = np.empty((nkpts,nvir,nvir), dtype=t2.dtype)
    metadata = {'kpts': kpts, 'rmat': rmat,
                'label': 'vv', 'trans': 'cn',
                'incore': True}
    Fac = ktensor.empty([nvir,nvir], dtype=t2.dtype, metadata=metadata)

    for i in range(kpts.nkpts_ibz):
        ki = kpts.ibz2bz[i]
        Fac[ki] = eris.fock[ki,nocc:,nocc:].copy()

    ka_ibz_bz = kpts.ibz2bz[np.arange(kpts.nkpts_ibz)]
    for i, kq in enumerate(kqrts.kqrts_ibz):
        kk, kl, ka, kd = kq
        kc = ka
        Soovv = 2*eris.oovv[kk,kl,kc] - eris.oovv[kk,kl,kd].transpose(0,1,3,2)
        fock = -einsum('klcd,klad->ac', Soovv, t2[kk,kl,ka])
        if kk == ka:
            fock += -einsum('klcd,ka,ld->ac', Soovv, t1[ka], t1[kl])

        op_group = kqrts.stars_ops[i]
        ka_prim = kpts.k2opk[ka, op_group]
        mask = np.isin(ka_prim, ka_ibz_bz)
        for iop, ka_p in zip(op_group[mask], ka_prim[mask]):
            rmat_vv = rmat.vv[ka][iop]
            Fac[ka_p] += einsum('ac,ae,cf->ef', fock, rmat_vv.conj(), rmat_vv)
    return Fac

def Loo(kpts, kqrts, t1, t2, eris, rmat):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:]

    Lki = cc_Foo(kpts, kqrts, t1, t2, eris, rmat)
    for ki_ibz in range(kpts.nkpts_ibz):
        ki = kpts.ibz2bz[ki_ibz]
        Lki[ki] += einsum('kc,ic->ki', fov[ki], t1[ki])

    for i, kq in enumerate(kqrts.kqrts_ibz):
        ki, kl, ka, kb = kq
        if ki == ka:
            fock = (2*einsum('klic,lc->ki', eris.ooov[ki,kl,ki], t1[kl])
                     -einsum('lkic,lc->ki', eris.ooov[kl,ki,ki], t1[kl]))
            for _, iop in kqrts.loop_stabilizer(i):
                rmat_oo = rmat.oo[ki][iop]
                Lki[ki] += einsum('ki,km,in->mn', fock, rmat_oo.conj(), rmat_oo)
    return Lki

def Lvv(kpts, kqrts, t1, t2, eris, rmat):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:]

    Lac = cc_Fvv(kpts, kqrts, t1, t2, eris, rmat)
    for ka_ibz in range(kpts.nkpts_ibz):
        ka = kpts.ibz2bz[ka_ibz]
        Lac[ka] += -einsum('kc,ka->ac', fov[ka], t1[ka])

    for i, kq in enumerate(kqrts.kqrts_ibz):
        ka, kk, kc, kl = kq
        if ka == kc:
            Svovv = 2 * eris.vovv[ka,kk,ka] - eris.vovv[ka,kk,kk].transpose(0,1,3,2)
            fock = einsum('akcd,kd->ac', Svovv, t1[kk])
            for _, iop in kqrts.loop_stabilizer(i):
                rmat_vv = rmat.vv[ka][iop]
                Lac[ka] += einsum('ac,ae,cf->ef', fock, rmat_vv.conj(), rmat_vv)
    return Lac


def _s2_index(kqrts_ibz):
    kqrts_ibz_t = kqrts_ibz[:, np.array([1,0,3,2])]
    diff = kqrts_ibz_t[:,None,:] - kqrts_ibz[None,:,:]
    idx = np.sum(abs(diff), axis=-1)
    idx = np.array(np.where(idx == 0)).T
    idx1 = np.where(idx[:,0] < idx[:,1])[0]
    return idx[idx1][:,1]

def cc_Woooo(kpts, kqrts, t1, t2, eris, rmat, out=None):
    nkpts, nocc, nvir = t1.shape
    if out is None: #default incore
        #Wklij = ktensor.empty_like(eris.oooo)
        metadata = {'kpts': kpts, 'kqrts': kqrts, 'rmat': rmat,
                    'label': 'oooo', 'trans': 'ccnn',
                    'incore': True}
        Wklij = ktensor.empty([nocc,nocc,nocc,nocc], dtype=t1.dtype,
                              metadata=metadata)
    else:
        Wklij = out

    s2_idx = _s2_index(kqrts.kqrts_ibz)

    for i, kq in enumerate(kqrts.kqrts_ibz):
        if i in s2_idx:
            continue
        kk, kl, ki, kj = kq
        oooo  = einsum('klic,jc->klij',eris.ooov[kk,kl,ki],t1[kj])
        oooo += einsum('lkjc,ic->klij',eris.ooov[kl,kk,kj],t1[ki])
        oooo += eris.oooo[kk,kl,ki]

        vvoo = eris.oovv[kk,kl].transpose(0,3,4,1,2).reshape(nkpts*nvir,nvir,nocc,nocc)
        t2t  = t2[ki,kj].copy().transpose(0,3,4,1,2)
        t2t[ki] += einsum('ic,jd->cdij',t1[ki],t1[kj])
        t2t = t2t.reshape(nkpts*nvir,nvir,nocc,nocc)
        oooo += einsum('cdkl,cdij->klij',vvoo,t2t)
        Wklij[kk,kl,ki] = oooo

    for i, kq in enumerate(kqrts.kqrts_ibz[s2_idx]):
        kl, kk, kj, ki = kq
        Wklij[kl,kk,kj] = Wklij[kk,kl,ki].transpose(1,0,3,2)
    return Wklij

def cc_Wvvvv(kpts, kqrts, t1, t2, eris, rmat, out=None):
    nkpts, nocc, nvir = t1.shape
    if out is None: #default incore
        #Wabcd = ktensor.empty_like(eris.vvvv)
        metadata = {'kpts': kpts, 'kqrts': kqrts, 'rmat': rmat,
                    'label': 'vvvv', 'trans': 'ccnn',
                    'incore': True}
        Wabcd = ktensor.empty([nvir,nvir,nvir,nvir], dtype=t1.dtype,
                              metadata=metadata)
    else:
        Wabcd = out

    s2_idx = _s2_index(kqrts.kqrts_ibz)

    for i, kq in enumerate(kqrts.kqrts_ibz):
        if i in s2_idx:
            continue
        ka, kb, kc, kd = kq
        vvvv  = einsum('akcd,kb->abcd', eris.vovv[ka,kb,kc], -t1[kb])
        vvvv += einsum('bkdc,ka->abcd', eris.vovv[kb,ka,kd], -t1[ka])
        vvvv += eris.vvvv[ka,kb,kc]
        Wabcd[ka,kb,kc] = vvvv

    for i, kq in enumerate(kqrts.kqrts_ibz[s2_idx]):
        kb, ka, kd, kc = kq
        Wabcd[kb,ka,kd] = Wabcd[ka,kb,kc].transpose(1,0,3,2)
    return Wabcd

def cc_Wvoov(kpts, kqrts, t1, t2, eris, rmat, out=None):
    nkpts, nocc, nvir = t1.shape
    if out is None: #default incore
        #Wakic = ktensor.empty_like(eris.voov)
        metadata = {'kpts': kpts, 'kqrts': kqrts, 'rmat': rmat,
                    'label': 'voov', 'trans': 'ccnn',
                    'incore': True}
        Wakic = ktensor.empty([nvir,nocc,nocc,nvir], dtype=t1.dtype,
                              metadata=metadata)
    else:
        Wakic = out

    for i, kq in enumerate(kqrts.kqrts_ibz):
        ka, kk, ki, kc = kq
        voov  = einsum('akdc,id->akic', eris.vovv[ka,kk,ki], t1[ki])
        voov -= einsum('lkic,la->akic', eris.ooov[ka,kk,ki], t1[ka])
        voov += eris.voov[ka,kk,ki]

        kd = ki
        tau = t2[:,ki,ka].copy()
        tau[ka] += 2*einsum('id,la->liad', t1[kd], t1[ka])
        oovv_tmp = np.array(eris.oovv[kk,:,kc])
        voov -= 0.5*einsum('xklcd,xliad->akic', oovv_tmp, tau)
        Soovv_tmp = 2*oovv_tmp - eris.oovv[:,kk,kc].transpose(0,2,1,3,4)
        voov += 0.5*einsum('xklcd,xilad->akic', Soovv_tmp, t2[ki,:,ka])

        Wakic[ka,kk,ki] = voov
    return Wakic

def cc_Wvovo(kpts, kqrts, t1, t2, eris, rmat, out=None):
    nkpts, nocc, nvir = t1.shape
    if out is None: #default incore
        #Wakci = np.empty((nkpts,nkpts,nkpts,nvir,nocc,nvir,nocc), t1.dtype)
        metadata = {'kpts': kpts, 'kqrts': kqrts, 'rmat': rmat,
                    'label': 'vovo', 'trans': 'ccnn',
                    'incore': True}
        Wakci = ktensor.empty([nvir,nocc,nvir,nocc], dtype=t1.dtype,
                              metadata=metadata)
    else:
        Wakci = out

    for i, kq in enumerate(kqrts.kqrts_ibz):
        ka, kk, kc, ki = kq
        vovo  = einsum('akcd,id->akci',eris.vovv[ka,kk,kc],t1[ki])
        vovo -= einsum('klic,la->akci',eris.ooov[kk,ka,ki],t1[ka])
        vovo += np.asarray(eris.ovov[kk,ka,ki]).transpose(1,0,3,2)

        oovvf = eris.oovv[:,kk,kc].reshape(nkpts*nocc,nocc,nvir,nvir)
        t2f   = t2[:,ki,ka].copy()
        kd = ki
        t2f[ka] += 2*einsum('id,la->liad',t1[kd],t1[ka])
        t2f = t2f.reshape(nkpts*nocc,nocc,nvir,nvir)
        vovo -= 0.5*einsum('lkcd,liad->akci',oovvf,t2f)

        Wakci[ka,kk,kc] = vovo
    return Wakci
