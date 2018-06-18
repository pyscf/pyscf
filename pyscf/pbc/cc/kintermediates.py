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
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#

#import numpy as np
from pyscf import lib
from pyscf.pbc.lib import kpts_helper
import numpy

#einsum = np.einsum
einsum = lib.einsum

#################################################
# FOLLOWING:                                    #
# J. Gauss and J. F. Stanton,                   #
# J. Chem. Phys. 103, 3561 (1995) Table III     #
#################################################

### Section (a)

def make_tau(cc, t2, t1a, t1b, fac=1., out=None):
    nkpts, nocc, nvir = t1a.shape
    tau1 = t2.copy()
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
    for ki in range(nkpts):
        for ka in range(nkpts):
            for kj in range(nkpts):
                    kb = kconserv[ki,ka,kj]
                    tmp = numpy.zeros((nocc,nocc,nvir,nvir),dtype=t2.dtype)
                    if ki == ka and kj == kb:
                        tmp += einsum('ia,jb->ijab',t1a[ki],t1b[kj])
                    if ki == kb and kj == ka:
                        tmp -= einsum('ib,ja->ijab',t1a[ki],t1b[kj])
                    if kj == ka and ki == kb:
                        tmp -= einsum('ja,ib->ijab',t1a[kj],t1b[ki])
                    if kj == kb and ki == ka:
                        tmp += einsum('jb,ia->ijab',t1a[kj],t1b[ki])
                    tau1[ki,kj,ka] += fac*0.5*tmp
    return tau1

def cc_Fvv(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:].copy()
    fvv = eris.fock[:,nocc:,nocc:].copy()
    # <o(k1)v(k2)||v(k3)v(k4)> = <v(k2)o(k1)||v(k4)v(k3)> = -<v(k2)o(k1)||v(k3)v(k4)>
    eris_vovv = -eris.ovvv.transpose(1,0,2,4,3,5,6)
    tau_tilde = make_tau(cc,t2,t1,t1,fac=0.5)
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

def cc_Foo(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:].copy()
    foo = eris.fock[:,:nocc,:nocc].copy()
    tau_tilde = make_tau(cc,t2,t1,t1,fac=0.5)
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

def cc_Fov(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    fov = eris.fock[:,:nocc,nocc:].copy()
    Fme = numpy.zeros(fov.shape, t1.dtype)
    for km in range(nkpts):
        Fme[km] += fov[km]
        for kf in range(nkpts):
            kn = kf
            Fme[km] -= einsum('nf,mnfe->me',t1[kf],eris.oovv[km,kn,kf])
    return Fme

def cc_Woooo(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    tau = make_tau(cc,t2,t1,t1)
    Wmnij = eris.oooo.copy()
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
    for km in range(nkpts):
        for kn in range(nkpts):
            # Since it's not enough just to switch i and j and need to create the k_i and k_j
            # so that P(ij) switches both i,j and k_i,k_j
            #   t1[ k_j, j, e ] * v[ k_m, k_n, k_i, m, n, i, e ] -> tmp[ k_i, k_j, m, n, i, j ]
            # Here, x = k_j and y = k_i
            tmp = einsum('xje,ymnie->yxmnij',t1,eris.ooov[km,kn])
            tmp = tmp - tmp.transpose(1,0,2,3,5,4)
            for ki in range(nkpts):
                kj = kconserv[km,ki,kn]
                Wmnij[km,kn,ki] += tmp[ki,kj]
                # Here, x = k_e
                Wmnij[km,kn,ki] += 0.25*einsum('xijef,xmnef->mnij',
                        tau[ki,kj],eris.oovv[km,kn])
    return Wmnij

def cc_Wvvvv(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    eris_vovv = - eris.ovvv.transpose(1,0,2,4,3,5,6)
    tau = make_tau(cc,t2,t1,t1)
    Wabef = eris.vvvv.copy()
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
    for ka in range(nkpts):
        for kb in range(nkpts):
            for ke in range(nkpts):
                km = kb
                tmp  = einsum('mb,amef->abef',t1[kb],eris_vovv[ka,km,ke])
                km = ka
                tmp -= einsum('ma,bmef->abef',t1[ka],eris_vovv[kb,km,ke])
                Wabef[ka,kb,ke] += -tmp
                # km + kn - ka = kb
                # => kn = ka - km + kb
                for km in range(nkpts):
                    kn = kconserv[ka,km,kb]
                    Wabef[ka,kb,ke] += 0.25*einsum('mnab,mnef->abef',tau[km,kn,ka],
                                                   eris.oovv[km,kn,ke])
    return Wabef

def cc_Wovvo(cc,t1,t2,eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
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
                for kn in range(nkpts):
                    kf = kconserv[km,ke,kn]
                    Wmbej[km,kb,ke] += -0.5*einsum('jnfb,mnef->mbej',t2[kj,kn,kf],
                                                   eris.oovv[km,kn,ke])
                    if kn == kb and kf == kj:
                        Wmbej[km,kb,ke] += -einsum('jf,nb,mnef->mbej',t1[kj],t1[kn],
                                                   eris.oovv[km,kn,ke])
    return Wmbej

### Section (b)

def Fvv(t1,t2,eris):
    ccFov = cc_Fov(t1,t2,eris)
    Fae = cc_Fvv(t1,t2,eris) - 0.5*einsum('ma,me->ae',t1,ccFov)
    return Fae

def Foo(t1,t2,eris):
    ccFov = cc_Fov(t1,t2,eris)
    Fmi = cc_Foo(t1,t2,eris) + 0.5*einsum('ie,me->mi',t1,ccFov)
    return Fmi

def Fov(t1,t2,eris):
    Fme = cc_Fov(t1,t2,eris)
    return Fme

def Woooo(t1,t2,eris):
    tau = make_tau(cc,t2,t1,t1)
    Wmnij = cc_Woooo(t1,t2,eris) + 0.25*einsum('ijef,mnef->mnij',tau,eris.oovv)
    return Wmnij

def Wvvvv(t1,t2,eris):
    tau = make_tau(cc,nt2,t1,t1)
    Wabef = cc_Wvvvv(t1,t2,eris) + 0.25*einsum('mnab,mnef->abef',tau,eris.oovv)
    return Wabef

def Wovvo(t1,t2,eris):
    Wmbej = cc_Wovvo(t1,t2,eris) - 0.5*einsum('jnfb,mnef->mbej',t2,eris.oovv)
    return Wmbej

# Indices in the following can be safely permuted.

def Wooov(t1,t2,eris):
    Wmnie = eris.ooov + einsum('if,mnfe->mnie',t1,eris.oovv)
    return Wmnie

def Wvovv(t1,t2,eris):
    eris_vovv = -eris.ovvv.transpose(1,0,2,3)
    Wamef = eris_vovv - einsum('na,nmef->amef',t1,eris.oovv)
    return Wamef

def Wovoo(t1,t2,eris):
    eris_ovvo = -eris.ovov.transpose(0,1,3,2)
    tmp1 = einsum('mnie,jnbe->mbij',eris.ooov,t2)
    tmp2 = ( einsum('ie,mbej->mbij',t1,eris_ovvo)
            - einsum('ie,njbf,mnef->mbij',t1,t2,eris.oovv) )
    FFov = Fov(t1,t2,eris)
    WWoooo = Woooo(t1,t2,eris)
    tau = make_tau(cc,t2,t1,t1)
    Wmbij = ( eris.ovoo - einsum('me,ijbe->mbij',FFov,t2)
              - einsum('nb,mnij->mbij',t1,WWoooo)
              + 0.5 * einsum('mbef,ijef->mbij',eris.ovvv,tau)
              + tmp1 - tmp1.transpose(0,1,3,2)
              + tmp2 - tmp2.transpose(0,1,3,2) )
    return Wmbij

def Wvvvo(t1,t2,eris):
    eris_ovvo = -eris.ovov.transpose(0,1,3,2)
    eris_vvvo = eris.ovvv.transpose(2,3,1,0).conj()
    eris_oovo = -eris.ooov.transpose(0,1,3,2)
    tmp1 = einsum('mbef,miaf->abei',eris.ovvv,t2)
    tmp2 = ( einsum('ma,mbei->abei',t1,eris_ovvo)
            - einsum('ma,nibf,mnef->abei',t1,t2,eris.oovv) )
    FFov = Fov(t1,t2,eris)
    WWvvvv = Wvvvv(t1,t2,eris)
    tau = make_tau(cc,t2,t1,t1)
    Wabei = ( eris_vvvo - einsum('me,miab->abei',FFov,t2)
                    + einsum('if,abef->abei',t1,WWvvvv)
                    + 0.5 * einsum('mnei,mnab->abei',eris_oovo,tau)
                    - tmp1 + tmp1.transpose(1,0,2,3)
                    - tmp2 + tmp2.transpose(1,0,2,3) )
    return Wabei

