#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#

#import numpy as np
from itertools import product
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.lib import kpts_helper
import numpy
from pyscf.lib.parameters import LARGE_DENOM  # noqa
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)  # noqa
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.pbc.cc.kccsd_t_rhf import _get_epqr

#einsum = numpy.einsum
einsum = lib.einsum

#################################################
# FOLLOWING:                                    #
# J. Gauss and J. F. Stanton,                   #
# J. Chem. Phys. 103, 3561 (1995) Table III     #
#################################################

### Section (a)

def make_tau(cc, t2, t1, t1p, kconserv, fac=1., out=None):
    nkpts, nocc, nvir = t1.shape
    tau1 = numpy.ndarray(t2.shape, dtype=t2.dtype, buffer=out)
    tau1[:] = t2
    for ki in range(nkpts):
        for ka in range(nkpts):
            for kj in range(nkpts):
                    kb = kconserv[ki,ka,kj]
                    tmp = numpy.zeros((nocc,nocc,nvir,nvir),dtype=t2.dtype)
                    if ki == ka and kj == kb:
                        tmp += einsum('ia,jb->ijab',t1[ki],t1p[kj])
                    if ki == kb and kj == ka:
                        tmp -= einsum('ib,ja->ijab',t1[ki],t1p[kj])
                    if kj == ka and ki == kb:
                        tmp -= einsum('ja,ib->ijab',t1[kj],t1p[ki])
                    if kj == kb and ki == ka:
                        tmp += einsum('jb,ia->ijab',t1[kj],t1p[ki])
                    tau1[ki,kj,ka] += fac*0.5*tmp
    return tau1

def cc_Fvv(cc,t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:].copy()
    fvv = eris.fock[:,nocc:,nocc:].copy()
    # <o(k1)v(k2)||v(k3)v(k4)> = <v(k2)o(k1)||v(k4)v(k3)> = -<v(k2)o(k1)||v(k3)v(k4)>
    eris_vovv = -eris.ovvv.transpose(1,0,2,4,3,5,6)
    tau_tilde = make_tau(cc,t2,t1,t1,kconserv,fac=0.5)
    Fae = numpy.zeros(fvv.shape, t1.dtype)
    #kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
    for ka in range(nkpts):
        Fae[ka] += fvv[ka]
        Fae[ka] += -0.5*einsum('me,ma->ae',fov[ka],t1[ka])
        for km in range(nkpts):
            Fae[ka] += einsum('mf,amef->ae',t1[km],eris_vovv[ka,km,ka])
            for kn in range(nkpts):
                #kb = kconserv[km,ka,kn]
                Fae[ka] += -0.5*einsum('mnaf,mnef->ae',tau_tilde[km,kn,ka],
                                       eris.oovv[km,kn,ka])
    return Fae

def cc_Foo(cc,t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:].copy()
    foo = eris.fock[:,:nocc,:nocc].copy()
    tau_tilde = make_tau(cc,t2,t1,t1,kconserv,fac=0.5)
    Fmi = numpy.zeros(foo.shape, t1.dtype)
    for km in range(nkpts):
        Fmi[km] += foo[km]
        Fmi[km] += 0.5*einsum('me,ie->mi',fov[km],t1[km])
        for kn in range(nkpts):
            Fmi[km] += einsum('ne,mnie->mi',t1[kn],eris.ooov[km,kn,km])
            for ke in range(nkpts):
                Fmi[km] += 0.5*einsum('inef,mnef->mi',tau_tilde[km,kn,ke],
                                      eris.oovv[km,kn,ke])
    return Fmi

def cc_Fov(cc,t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:].copy()
    Fme = numpy.zeros(fov.shape, t1.dtype)
    for km in range(nkpts):
        Fme[km] += fov[km]
        for kf in range(nkpts):
            kn = kf
            Fme[km] -= einsum('nf,mnfe->me',t1[kf],eris.oovv[km,kn,kf])
    return Fme

def cc_Woooo(cc,t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    tau = make_tau(cc,t2,t1,t1,kconserv)
    Wmnij = eris.oooo.copy()
    for km in range(nkpts):
        for kn in range(nkpts):
            # Since it's not enough just to switch i and j and need to create the k_i and k_j
            # so that P(ij) switches both i,j and k_i,k_j
            #   t1[ k_j, j, e ] * v[ k_m, k_n, k_i, m, n, i, e ] -> tmp[ k_i, k_j, m, n, i, j ]
            # Here, x = k_j and y = k_i
            tmp = einsum('xje,ymnie->yxmnij',t1,eris.ooov[km,kn])
            tmp = tmp - tmp.transpose(1,0,2,3,5,4)

            ki = numpy.arange(nkpts)
            kj = kconserv[km,ki,kn]
            kij = (ki,kj)
            Wmnij[km,kn,:] += 0.25*einsum('yxijef,xmnef->ymnij',tau[kij],eris.oovv[km,kn])

            for ki in range(nkpts):
                kj = kconserv[km,ki,kn]
                Wmnij[km,kn,ki] += tmp[ki,kj]
    return Wmnij

def cc_Wvvvv(cc,t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    eris_vovv = - eris.ovvv.transpose(1,0,2,4,3,5,6)
    tau = make_tau(cc,t2,t1,t1,kconserv)
    Wabef = eris.vvvv.copy()
    for ka in range(nkpts):
        for kb in range(nkpts):

            km = numpy.arange(nkpts).tolist()
            kn = kconserv[ka,km,kb].tolist()
            kmn = tuple([km,kn])
            Wabef[ka,kb] += 0.25*einsum('xmnab,xymnef->yabef',tau.transpose(2,0,1,3,4,5,6)[ka][kmn],eris.oovv[kmn])

            for ke in range(nkpts):
                km = kb
                tmp  = einsum('mb,amef->abef',t1[kb],eris_vovv[ka,km,ke])
                km = ka
                tmp -= einsum('ma,bmef->abef',t1[ka],eris_vovv[kb,km,ke])
                Wabef[ka,kb,ke] += -tmp
                # km + kn - ka = kb
                # => kn = ka - km + kb

    return Wabef

def cc_Wovvo(cc,t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    eris_ovvo = numpy.zeros(shape=(nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc),dtype=t2.dtype)
    eris_oovo = numpy.zeros(shape=(nkpts,nkpts,nkpts,nocc,nocc,nvir,nocc),dtype=t2.dtype)
    for km in range(nkpts):
        for kb in range(nkpts):
            for ke in range(nkpts):
                kj = kconserv[km,ke,kb]
                # <mb||je> -> -<mb||ej>
                eris_ovvo[km,kb,ke] = -eris.ovov[km,kb,kj].transpose(0,1,3,2)
                # <mn||je> -> -<mn||ej>
                # let kb = kn as a dummy variable
                eris_oovo[km,kb,ke] = -eris.ooov[km,kb,kj].transpose(0,1,3,2)
    Wmbej = eris_ovvo.copy()
    for km in range(nkpts):
        for kb in range(nkpts):
            for ke in range(nkpts):
                kj = kconserv[km,ke,kb]
                Wmbej[km,kb,ke] += einsum('jf,mbef->mbej',t1[kj,:,:],eris.ovvv[km,kb,ke])
                Wmbej[km,kb,ke] += -einsum('nb,mnej->mbej',t1[kb,:,:],eris_oovo[km,kb,ke])
                temp = numpy.zeros([nkpts, nocc, nocc, nvir, nvir], dtype=t2.dtype)
                for kn in range(nkpts):
                    kf = kconserv[km,ke,kn]
                    temp[kn] = -0.5*t2[kj,kn,kf].copy()
                    if kn == kb and kf == kj:
                        temp[kn] -= einsum('jf,nb->jnfb', t1[kj], t1[kn])
                Wmbej[km,kb,ke] += einsum('xjnfb, xmnef->mbej', temp, eris.oovv[km,:,ke])
    return Wmbej

def cc_Wovvo_jk(cc, t1, t2, eris, kconserv):
    nkpts, nocc, nvir = t1.shape
    Wmbej = eris.ovvo.copy()
    for km in range(nkpts):
        for kb in range(nkpts):
            for ke in range(nkpts):
                kj = kconserv[km,ke,kb]
                Wmbej[km,kb,ke] += einsum('jf,mbef->mbej',t1[kj,:,:],eris.ovvv[km,kb,ke])
                Wmbej[km,kb,ke] += -einsum('nb,mnej->mbej',t1[kb,:,:],eris.oovo[km,kb,ke])
                for kn in range(nkpts):
                    kf = kconserv[km,ke,kn]
                    Wmbej[km,kb,ke] += -0.5*einsum('jnfb,mnef->mbej',t2[kj,kn,kf],
                                                   eris.oovv[km,kn,ke])
                    if kn == kb and kf == kj:
                        Wmbej[km,kb,ke] += -einsum('jf,nb,mnef->mbej',t1[kj],t1[kn],
                                                   eris.oovv[km,kn,ke])
    return Wmbej

### Section (b)

def Fvv(cc,t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    ccFov = cc_Fov(cc,t1,t2,eris,kconserv)
    Fae = cc_Fvv(cc,t1,t2,eris,kconserv)
    for km in range(nkpts):
        Fae[km] -= 0.5*einsum('ma,me->ae', t1[km], ccFov[km])
    return Fae

def Foo(cc,t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    ccFov = cc_Fov(cc,t1,t2,eris,kconserv)
    Fmi = cc_Foo(cc,t1,t2,eris,kconserv)
    for km in range(nkpts):
        Fmi[km] += 0.5*einsum('ie,me->mi',t1[km],ccFov[km])
    return Fmi

def Fov(cc,t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    Fme = cc_Fov(cc,t1,t2,eris,kconserv)
    return Fme

def Woooo(cc,t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    tau = make_tau(cc,t2,t1,t1,kconserv)
    Wmnij = cc_Woooo(cc,t1,t2,eris,kconserv)
    for km in range(nkpts):
        for kn in range(nkpts):
            for ki in range(nkpts):
                kj = kconserv[km ,ki, kn]
                Wmnij[km, kn, ki] += 0.25*einsum('xijef,xmnef->mnij',tau[ki, kj, :],
                                                 eris.oovv[km, kn, :])
    return Wmnij

def Wvvvv(cc,t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    tau = make_tau(cc,t2,t1,t1,kconserv)
    Wabef = cc_Wvvvv(cc,t1,t2,eris,kconserv)
    for ka, kb, ke in kpts_helper.loop_kkk(nkpts):
        kf = kconserv[ka, ke, kb]
        for km in range(nkpts):
            kn = kconserv[ka, km, kb]
            Wabef[ka, kb, ke] += 0.25*einsum('mnab,mnef->abef',tau[km, kn, ka],
                                             eris.oovv[km, kn, ke])
    return Wabef

def Wovvo(cc,t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    Wmbej = cc_Wovvo(cc,t1,t2,eris,kconserv)
    for km, kb, ke in kpts_helper.loop_kkk(nkpts):
        kj = kconserv[km, ke, kb]
        for kn in range(nkpts):
            kf = kconserv[km, ke, kn]
            Wmbej[km, kb, ke] -= 0.5*einsum('jnfb,mnef->mbej',t2[kj, kn, kf],
                                            eris.oovv[km, kn, ke])
    return Wmbej

# Indices in the following can be safely permuted.

def Wooov(cc,t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    Wmnie = eris.ooov.copy()
    for km, kn, ki in kpts_helper.loop_kkk(nkpts):
        kf = ki
        Wmnie[km, kn, ki] += einsum('if,mnfe->mnie',t1[ki], eris.oovv[km, kn, kf])
    return Wmnie

def Wvovv(cc,t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape
    Wamef = numpy.empty((nkpts, nkpts, nkpts, nvir, nocc, nvir, nvir), dtype=eris.ovvv.dtype)
    for ka, km, ke in kpts_helper.loop_kkk(nkpts):
        kn = ka
        Wamef[ka, km, ke] = -eris.ovvv[km, ka, ke].transpose(1, 0, 2, 3)
        Wamef[ka, km, ke] -= einsum('na,nmef->amef',t1[kn],eris.oovv[kn, km, ke])
    return Wamef

def Wovoo(cc,t1,t2,eris,kconserv):
    nkpts, nocc, nvir = t1.shape

    Wmbij = eris.ovoo.copy()
    for km, kb, ki in kpts_helper.loop_kkk(nkpts):
        kj = kconserv[km, ki, kb]
        for kn in range(nkpts):
            Wmbij[km, kb, ki] += einsum('mnie,jnbe->mbij', eris.ooov[km, kn, ki], t2[kj, kn, kb])

        Wmbij[km, kb, ki] += einsum('ie,mbej->mbij', t1[ki], -eris.ovov[km, kb, kj].transpose(0, 1, 3, 2))
        for kf in range(nkpts):
            kn = kconserv[kb, kj, kf]
            Wmbij[km, kb, ki] -= einsum('ie,njbf,mnef->mbij', t1[ki], t2[kn, kj, kb], eris.oovv[km, kn, ki])
        # P(ij)
        for kn in range(nkpts):
            Wmbij[km, kb, ki] -= einsum('mnje,inbe->mbij', eris.ooov[km, kn, kj], t2[ki, kn, kb])

        Wmbij[km, kb, ki] -= einsum('je,mbei->mbij', t1[kj], -eris.ovov[km, kb, ki].transpose(0, 1, 3, 2))
        for kf in range(nkpts):
            kn = kconserv[kb, ki, kf]
            Wmbij[km, kb, ki] += einsum('je,nibf,mnef->mbij', t1[kj], t2[kn, ki, kb], eris.oovv[km, kn, kj])

    FFov = Fov(cc,t1,t2,eris,kconserv)
    WWoooo = Woooo(cc,t1,t2,eris,kconserv)
    tau = make_tau(cc,t2,t1,t1,kconserv)
    for km, kb, ki in kpts_helper.loop_kkk(nkpts):
        kj = kconserv[km, ki, kb]
        Wmbij[km, kb, ki] -= einsum('me,ijbe->mbij', FFov[km], t2[ki, kj, kb])
        Wmbij[km, kb, ki] -= einsum('nb,mnij->mbij', t1[kb], WWoooo[km, kb, ki])

    for km, kb, ki in kpts_helper.loop_kkk(nkpts):
        kj = kconserv[km, ki, kb]
        Wmbij[km, kb, ki] += 0.5 * einsum('xmbef,xijef->mbij', eris.ovvv[km, kb, :], tau[ki, kj, :])

    return Wmbij

def Wvvvo(cc,t1,t2,eris,kconserv,WWvvvv=None):
    nkpts, nocc, nvir = t1.shape
    FFov = Fov(cc,t1,t2,eris,kconserv)
    if WWvvvv is None:
        WWvvvv = Wvvvv(cc,t1,t2,eris,kconserv)

    eris_ovvo = numpy.zeros(shape=(nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc),dtype=t2.dtype)
    for km in range(nkpts):
        for kb in range(nkpts):
            for ke in range(nkpts):
                kj = kconserv[km,ke,kb]
                eris_ovvo[km,kb,ke] = -eris.ovov[km,kb,kj].transpose(0,1,3,2)

    tmp1 = numpy.zeros((nkpts, nkpts, nkpts, nvir, nvir, nvir, nocc),dtype=t2.dtype)
    tmp2 = numpy.zeros((nkpts, nkpts, nkpts, nvir, nvir, nvir, nocc),dtype=t2.dtype)
    for ka, kb, ke in kpts_helper.loop_kkk(nkpts):
        ki = kconserv[ka,ke,kb]
        tmp2[ka,kb,ke] += einsum('ma,mbei->abei',t1[ka],eris_ovvo[ka,kb,ke])
        for kn in range(nkpts):
            tmp2[ka,kb,ke] -= einsum('ma,nibf,mnef->abei',t1[ka],t2[kn,ki,kb],eris.oovv[ka,kn,ke])
        for km in range(nkpts):
            tmp1[ka,kb,ke] += einsum('mbef,miaf->abei',eris.ovvv[km,kb,ke],t2[km,ki,ka])
    tau = make_tau(cc,t2,t1,t1,kconserv)

    Wabei = -tmp1 + tmp1.transpose(1,0,2,4,3,5,6)
    Wabei -= tmp2 - tmp2.transpose(1,0,2,4,3,5,6)
    for ka, kb, ke in kpts_helper.loop_kkk(nkpts):
        ki = kconserv[ka, ke, kb]
        Wabei[ka, kb, ke] += eris.ovvv[ki, ke, kb].conj().transpose(3, 2, 1, 0)
        Wabei[ka, kb, ke] += einsum('if,abef->abei',t1[ki],WWvvvv[ka, kb, ke])
        Wabei[ka, kb, ke] -= einsum('me,miab->abei',FFov[ke],t2[ke, ki, ka])
        for km in range(nkpts):
            kn = kconserv[ka, km, kb]
            Wabei[ka, kb, ke] += 0.5 * einsum('nmie,mnab->abei',
                                              eris.ooov[kn, km, ki],
                                              tau[km, kn, ka])
    return Wabei

def get_full_t3p2(mycc, t1, t2, eris):
    '''Build the entire T3[2] array in memory.
    '''
    nkpts = mycc.nkpts
    nocc = mycc.nocc
    nmo = mycc.nmo
    nvir = nmo - nocc
    kconserv = mycc.khelper.kconserv

    def get_wijkabc(ki, kj, kk, ka, kb, kc):
        '''Build T3[2] for `ijkabc` at a given set of k-points'''
        km = kconserv[kc, kk, kb]
        kf = kconserv[kk, kc, kj]
        ret = einsum('kjcf,ifab->ijkabc', t2[kk,kj,kc], eris.ovvv[ki,kf,ka].conj())
        ret = ret - einsum('jima,mkbc->ijkabc', eris.ooov[kj,ki,km].conj(), t2[km,kk,kb])
        return ret

    fock = eris.fock
    #fov = fock[:, :nocc, nocc:]
    #foo = numpy.array([fock[ikpt, :nocc, :nocc].diagonal() for ikpt in range(nkpts)])
    #fvv = numpy.array([fock[ikpt, nocc:, nocc:].diagonal() for ikpt in range(nkpts)])
    mo_energy_occ = numpy.array([eris.mo_energy[ki][:nocc] for ki in range(nkpts)])
    mo_energy_vir = numpy.array([eris.mo_energy[ki][nocc:] for ki in range(nkpts)])

    mo_e_o = mo_energy_occ
    mo_e_v = mo_energy_vir

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mycc, kind="split")

    t3 = numpy.empty((nkpts,nkpts,nkpts,nkpts,nkpts,nocc,nocc,nocc,nvir,nvir,nvir),
                      dtype=t2.dtype)
    for ki, kj, kk, ka, kb in product(range(nkpts), repeat=5):
        kc = kpts_helper.get_kconserv3(mycc._scf.cell, mycc.kpts,
                                       [ki, kj, kk, ka, kb])
        # Perform P(abc)
        t3[ki,kj,kk,ka,kb] =  get_wijkabc(ki,kj,kk,ka,kb,kc)
        t3[ki,kj,kk,ka,kb] += get_wijkabc(ki,kj,kk,kb,kc,ka).transpose(0,1,2,5,3,4)
        t3[ki,kj,kk,ka,kb] += get_wijkabc(ki,kj,kk,kc,ka,kb).transpose(0,1,2,4,5,3)

    # Perform P(ijk)
    t3 = (t3.transpose(0,1,2,3,4,5,6,7,8,9,10) +
          t3.transpose(1,2,0,3,4,6,7,5,8,9,10) +
          t3.transpose(2,0,1,3,4,7,5,6,8,9,10))

    for ki, kj, kk in product(range(nkpts), repeat=3):
        eijk = _get_epqr([0,nocc,ki,mo_e_o,nonzero_opadding],
                         [0,nocc,kj,mo_e_o,nonzero_opadding],
                         [0,nocc,kk,mo_e_o,nonzero_opadding])
        for ka, kb in product(range(nkpts), repeat=2):
            kc = kpts_helper.get_kconserv3(mycc._scf.cell, mycc.kpts,
                                           [ki, kj, kk, ka, kb])
            eabc = _get_epqr([0,nvir,ka,mo_e_v,nonzero_vpadding],
                             [0,nvir,kb,mo_e_v,nonzero_vpadding],
                             [0,nvir,kc,mo_e_v,nonzero_vpadding],
                             fac=[-1.,-1.,-1.])
            eijkabc = eijk[:, :, :, None, None, None] + eabc[None, None, None, :, :, :]
            t3[ki,kj,kk,ka,kb] /= eijkabc

    return t3

def get_t3p2_imds_slow(cc, t1, t2, eris=None, t3p2_ip_out=None, t3p2_ea_out=None):
    """Calculates T1, T2 amplitudes corrected by second-order T3 contribution
    and intermediates used in IP/EA-CCSD(T)a

    Args:
        cc (:obj:`KGCCSD`):
            Object containing coupled-cluster results.
        t1 (:obj:`ndarray`):
            T1 amplitudes.
        t2 (:obj:`ndarray`):
            T2 amplitudes from which the T3[2] amplitudes are formed.
        eris (:obj:`_PhysicistsERIs`):
            Antisymmetrized electron-repulsion integrals in physicist's notation.
        t3p2_ip_out (:obj:`ndarray`):
            Store results of the intermediate used in IP-EOM-CCSD(T)a.
        t3p2_ea_out (:obj:`ndarray`):
            Store results of the intermediate used in EA-EOM-CCSD(T)a.

    Returns:
        delta_ccsd (float):
            Difference of perturbed and unperturbed CCSD ground-state energy,
                energy(T1 + T1[2], T2 + T2[2]) - energy(T1, T2)
        pt1 (:obj:`ndarray`):
            Perturbatively corrected T1 amplitudes.
        pt2 (:obj:`ndarray`):
            Perturbatively corrected T2 amplitudes.

    Reference:
        D. A. Matthews, J. F. Stanton "A new approach to approximate..."
            JCP 145, 124102 (2016); DOI:10.1063/1.4962910, Equation 14
        Shavitt and Bartlett "Many-body Methods in Physics and Chemistry"
            2009, Equation 10.33
    """
    if eris is None:
        eris = cc.ao2mo()
    fock = eris.fock
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.khelper.kconserv

    fov = [fock[ikpt, :nocc, nocc:] for ikpt in range(nkpts)]
    #foo = [fock[ikpt, :nocc, :nocc].diagonal() for ikpt in range(nkpts)]
    #fvv = [fock[ikpt, nocc:, nocc:].diagonal() for ikpt in range(nkpts)]
    mo_energy_occ = numpy.array([eris.mo_energy[ki][:nocc] for ki in range(nkpts)])
    mo_energy_vir = numpy.array([eris.mo_energy[ki][nocc:] for ki in range(nkpts)])

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(cc, kind="split")

    mo_e_o = mo_energy_occ
    mo_e_v = mo_energy_vir

    ccsd_energy = cc.energy(t1, t2, eris)
    dtype = numpy.result_type(t1, t2)

    if t3p2_ip_out is None:
        t3p2_ip_out = numpy.zeros((nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc), dtype=dtype)
    Wmcik = t3p2_ip_out

    if t3p2_ea_out is None:
        t3p2_ea_out = numpy.zeros((nkpts,nkpts,nkpts,nvir,nvir,nvir,nocc), dtype=dtype)
    Wacek = t3p2_ea_out

    t3 = get_full_t3p2(cc, t1, t2, eris)

    pt1 = numpy.zeros((nkpts, nocc, nvir), dtype=dtype)
    for ki in range(nkpts):
        ka = ki
        for km, kn, ke in product(range(nkpts), repeat=3):
            pt1[ki] += 0.25 * lib.einsum('mnef,imnaef->ia', eris.oovv[km,kn,ke], t3[ki,km,kn,ka,ke])
        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                       [0,nvir,ka,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        pt1[ki] /= eia

    pt2 = numpy.zeros((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=dtype)
    for ki, kj, ka in product(range(nkpts), repeat=3):
        kb = kconserv[ki,ka,kj]
        for km in range(nkpts):
            pt2[ki,kj,ka] += lib.einsum('ijmabe,me->ijab', t3[ki,kj,km,ka,kb], fov[km])
            for ke in range(nkpts):
                kf = kconserv[km,ke,kb]
                pt2[ki,kj,ka] += 0.5 * lib.einsum('ijmaef,mbfe->ijab', t3[ki,kj,km,ka,ke], eris.ovvv[km,kb,kf])
                kf = kconserv[km,ke,ka]
                pt2[ki,kj,ka] -= 0.5 * lib.einsum('ijmbef,mafe->ijab', t3[ki,kj,km,kb,ke], eris.ovvv[km,ka,kf])

            for kn in range(nkpts):
                pt2[ki,kj,ka] -= 0.5 * lib.einsum('inmabe,nmje->ijab', t3[ki,kn,km,ka,kb], eris.ooov[kn,km,kj])
                pt2[ki,kj,ka] += 0.5 * lib.einsum('jnmabe,nmie->ijab', t3[kj,kn,km,ka,kb], eris.ooov[kn,km,ki])

        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                       [0,nvir,ka,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                       [0,nvir,kb,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        eijab = eia[:, None, :, None] + ejb[:, None, :]
        pt2[ki,kj,ka] /= eijab

    pt1 += t1
    pt2 += t2

    for ki, kj, kk, ka, kb in product(range(nkpts), repeat=5):
        kc = kpts_helper.get_kconserv3(cc._scf.cell, cc.kpts,
                                       [ki, kj, kk, ka, kb])
        tmp = t3[ki,kj,kk,ka,kb]
        km = kconserv[ki,kc,kk]
        ke = kconserv[ka,kk,kc]

        Wmcik[km,kc,ki] += 0.5*lib.einsum('ijkabc,mjab->mcik', tmp, eris.oovv[km,kj,ka])
        Wacek[ka,kc,ke] += -0.5*lib.einsum('ijkabc,ijeb->acek', tmp, eris.oovv[ki,kj,ke])

    delta_ccsd_energy = cc.energy(pt1, pt2, eris) - ccsd_energy
    logger.info(cc, 'CCSD energy T3[2] correction : %14.8e', delta_ccsd_energy)
    return delta_ccsd_energy, pt1, pt2, Wmcik, Wacek
