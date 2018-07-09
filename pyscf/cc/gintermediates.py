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

import numpy as np
from pyscf import lib

#einsum = np.einsum
einsum = lib.einsum

# Ref: Gauss and Stanton, J. Chem. Phys. 103, 3561 (1995) Table III

# Section (a)

def make_tau(t2, t1a, t1b, fac=1, out=None):
    t1t1 = einsum('ia,jb->ijab', fac*0.5*t1a, t1b)
    t1t1 = t1t1 - t1t1.transpose(1,0,2,3)
    tau1 = t1t1 - t1t1.transpose(0,1,3,2)
    tau1 += t2
    return tau1

def cc_Fvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    fvv = eris.fock[nocc:,nocc:]
    eris_vovv = np.asarray(eris.ovvv).transpose(1,0,3,2)
    tau_tilde = make_tau(t2, t1, t1,fac=0.5)
    Fae = fvv - 0.5*einsum('me,ma->ae',fov, t1)
    Fae += einsum('mf,amef->ae', t1, eris_vovv)
    Fae -= 0.5*einsum('mnaf,mnef->ae', tau_tilde, eris.oovv)
    return Fae

def cc_Foo(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    foo = eris.fock[:nocc,:nocc]
    tau_tilde = make_tau(t2, t1, t1,fac=0.5)
    Fmi = ( foo + 0.5*einsum('me,ie->mi',fov, t1)
            + einsum('ne,mnie->mi', t1, eris.ooov)
            + 0.5*einsum('inef,mnef->mi', tau_tilde, eris.oovv) )
    return Fmi

def cc_Fov(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Fme = fov + einsum('nf,mnef->me', t1, eris.oovv)
    return Fme

def cc_Woooo(t1, t2, eris):
    tau = make_tau(t2, t1, t1)
    tmp = einsum('je,mnie->mnij', t1, eris.ooov)
    Wmnij = eris.oooo + tmp - tmp.transpose(0,1,3,2)
    Wmnij += 0.25*einsum('ijef,mnef->mnij', tau, eris.oovv)
    return Wmnij

def cc_Wvvvv(t1, t2, eris):
    tau = make_tau(t2, t1, t1)
    eris_ovvv = np.asarray(eris.ovvv)
    tmp = einsum('mb,mafe->bafe', t1, eris_ovvv)
    Wabef = np.asarray(eris.vvvv) - tmp + tmp.transpose(1,0,2,3)
    Wabef += einsum('mnab,mnef->abef', tau, 0.25*np.asarray(eris.oovv))
    return Wabef

def cc_Wovvo(t1, t2, eris):
    eris_ovvo = -np.asarray(eris.ovov).transpose(0,1,3,2)
    eris_oovo = -np.asarray(eris.ooov).transpose(0,1,3,2)
    Wmbej  = einsum('jf,mbef->mbej', t1, eris.ovvv)
    Wmbej -= einsum('nb,mnej->mbej', t1, eris_oovo)
    Wmbej -= 0.5*einsum('jnfb,mnef->mbej', t2, eris.oovv)
    Wmbej -= einsum('jf,nb,mnef->mbej', t1, t1, eris.oovv)
    Wmbej += eris_ovvo
    return Wmbej

### Section (b)

def Fvv(t1, t2, eris):
    ccFov = cc_Fov(t1, t2, eris)
    Fae = cc_Fvv(t1, t2, eris) - 0.5*einsum('ma,me->ae', t1,ccFov)
    return Fae

def Foo(t1, t2, eris):
    ccFov = cc_Fov(t1, t2, eris)
    Fmi = cc_Foo(t1, t2, eris) + 0.5*einsum('ie,me->mi', t1,ccFov)
    return Fmi

def Fov(t1, t2, eris):
    Fme = cc_Fov(t1, t2, eris)
    return Fme

def Woooo(t1, t2, eris):
    tau = make_tau(t2, t1, t1)
    Wmnij = 0.25*einsum('ijef,mnef->mnij', tau, eris.oovv)
    Wmnij += cc_Woooo(t1, t2, eris)
    return Wmnij

def Wvvvv(t1, t2, eris):
    tau = make_tau(t2, t1, t1)
    Wabef = cc_Wvvvv(t1, t2, eris)
    Wabef += einsum('mnab,mnef->abef', tau, .25*np.asarray(eris.oovv))
    return Wabef

def Wovvo(t1, t2, eris):
    Wmbej = -0.5*einsum('jnfb,mnef->mbej', t2, eris.oovv)
    Wmbej += cc_Wovvo(t1, t2, eris)
    return Wmbej

def Wooov(t1, t2, eris):
    Wmnie = einsum('if,mnfe->mnie', t1, eris.oovv)
    Wmnie += eris.ooov
    return Wmnie

def Wvovv(t1, t2, eris):
    Wamef = einsum('na,nmef->amef', -t1, eris.oovv)
    Wamef -= np.asarray(eris.ovvv).transpose(1,0,2,3)
    return Wamef

def Wovoo(t1, t2, eris):
    eris_ovvo = -np.asarray(eris.ovov).transpose(0,1,3,2)
    tmp1 = einsum('mnie,jnbe->mbij', eris.ooov, t2)
    tmp2 = einsum('ie,mbej->mbij', t1, eris_ovvo)
    tmp2 -= einsum('ie,njbf,mnef->mbij', t1, t2, eris.oovv)
    FFov = Fov(t1, t2, eris)
    WWoooo = Woooo(t1, t2, eris)
    tau = make_tau(t2, t1, t1)
    Wmbij = einsum('me,ijbe->mbij', -FFov, t2)
    Wmbij -= einsum('nb,mnij->mbij', t1, WWoooo)
    Wmbij += 0.5 * einsum('mbef,ijef->mbij', eris.ovvv, tau)
    Wmbij += tmp1 - tmp1.transpose(0,1,3,2)
    Wmbij += tmp2 - tmp2.transpose(0,1,3,2)
    Wmbij += np.asarray(eris.ooov).conj().transpose(2,3,0,1)
    return Wmbij

def Wvvvo(t1, t2, eris, _Wvvvv=None):
    eris_ovvo = -np.asarray(eris.ovov).transpose(0,1,3,2)
    eris_vvvo = -np.asarray(eris.ovvv).transpose(2,3,1,0).conj()
    eris_oovo = -np.asarray(eris.ooov).transpose(0,1,3,2)
    tmp1 = einsum('mbef,miaf->abei', eris.ovvv, t2)
    tmp2 = einsum('ma,mbei->abei', t1, eris_ovvo)
    tmp2 -= einsum('ma,nibf,mnef->abei', t1, t2, eris.oovv)
    FFov = Fov(t1, t2, eris)
    tau = make_tau(t2, t1, t1)
    Wabei  = 0.5 * einsum('mnei,mnab->abei', eris_oovo, tau)
    Wabei -= einsum('me,miab->abei', FFov, t2)
    Wabei += eris_vvvo
    Wabei -= tmp1 - tmp1.transpose(1,0,2,3)
    Wabei -= tmp2 - tmp2.transpose(1,0,2,3)
    nocc,nvir = t1.shape
    if _Wvvvv is None:
        _Wvvvv = Wvvvv(t1, t2, eris)
    Wabei += einsum('abef,if->abei', _Wvvvv, t1)
    return Wabei

