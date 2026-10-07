#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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

'''
Intermediates for restricted CCSD.  Complex integrals are supported.
'''

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo

# This is restricted (R)CCSD
# Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)

### Eqs. (37)-(39) "kappa"

def cc_Foo(t1, t2, eris):
    nocc, nvir = t1.shape
    foo = eris.fock[:nocc,:nocc]
    ovov = np.asarray(eris.ovov)
    Fki  = 2*lib.einsum('kcld,ilcd->ki', ovov, t2)
    Fki -=   lib.einsum('kdlc,ilcd->ki', ovov, t2)
    Fki += 2*lib.einsum('kcld,ic,ld->ki', ovov, t1, t1)
    Fki -=   lib.einsum('kdlc,ic,ld->ki', ovov, t1, t1)
    Fki += foo
    ovov = None
    return Fki

def cc_Fvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fvv = eris.fock[nocc:,nocc:]
    ovov = np.asarray(eris.ovov)
    Fac  =-2*lib.einsum('kcld,klad->ac', ovov, t2)
    Fac +=   lib.einsum('kdlc,klad->ac', ovov, t2)
    Fac -= 2*lib.einsum('kcld,ka,ld->ac', ovov, t1, t1)
    Fac +=   lib.einsum('kdlc,ka,ld->ac', ovov, t1, t1)
    Fac += fvv
    ovov = None
    return Fac

def cc_Fov(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    ovov = np.asarray(eris.ovov)
    Fkc  = 2*lib.einsum('kcld,ld->kc', ovov, t1)
    Fkc -=   lib.einsum('kdlc,ld->kc', ovov, t1)
    Fkc += fov
    ovov = None
    return Fkc

# intermediate to avoid ovvv
def cc_Lvo(t1, eris):
    nocc, nvir = t1.shape
    naux = eris.Lvv.shape[0]
    Lvv = np.asarray(eris.Lvv).reshape((naux, nvir, nvir))
    Lov = lib.einsum('Lab,ib->Lai', Lvv, t1)
    return Lvo

def cc_Loo(t1, eris):
    print("build Loo", flush=True)
    nocc, nvir = t1.shape
    naux = eris.Lov.shape[0]
    Lov = np.asarray(eris.Lov).reshape((naux, nocc, nvir))
    Loo = lib.einsum('Lia,ja->Lij', Lov, t1)
    return Loo

### Eqs. (40)-(41) "lambda"

def Loo(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Lki = cc_Foo(t1, t2, eris) + lib.einsum('kc,ic->ki',fov, t1)
    Lki += 2*lib.einsum('lcki,lc->ki', eris.ovoo, t1)
    Lki -=   lib.einsum('kcli,lc->ki', eris.ovoo, t1)
    return Lki

def Lvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Lac = cc_Fvv(t1, t2, eris) - lib.einsum('kc,ka->ac',fov, t1)
    dsize = eris.mo_coeff.itemsize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    mem_avail *= 1e6 / dsize
    nvir3 = nvir**3
    blksize = min(nocc, int(0.7*((mem_avail-nvir**2) / (2*nvir3+nvir) )))
    assert blksize > 0, "enlarge mem"
    for i0, i1 in lib.prange(0, nocc, blksize):
        eris_ovvv = np.asarray(eris.get_ovvv(slice(i0,i1)))
        Lac += 2*np.einsum('kdac,kd->ac', eris_ovvv, t1[i0:i1])
        Lac -=   np.einsum('kcad,kd->ac', eris_ovvv, t1[i0:i1])
        eris_ovvv = None
    return Lac


def cc_Woooo(t1, t2, eris):
    Wklij  = lib.einsum('lcki,jc->klij', eris.ovoo, t1)
    Wklij += lib.einsum('kclj,ic->klij', eris.ovoo, t1)
    ovov = np.asarray(eris.ovov)
    Wklij += lib.einsum('kcld,ijcd->klij', ovov, t2)
    Wklij += lib.einsum('kcld,jd,ic->klij', ovov, t1, t1)
    ovov = None
    Wklij += np.asarray(eris.oooo).transpose(0,2,1,3)
    return Wklij

def cc_Wvvvv(t1, t2, eris):
    print("build Wvvvv", flush=True)
    # Incore
    eris_ovvv = np.asarray(eris.get_ovvv())
    Wabcd  = lib.einsum('kdac,kb->abcd', eris_ovvv,-t1)
    Wabcd -= lib.einsum('kcbd,ka->abcd', eris_ovvv, t1)
    Wabcd += np.asarray(_get_vvvv(eris)).transpose(0,2,1,3)
    return Wabcd
###

def cc_Wvoov(t1, t2, eris):
    nocc, nvir = t1.shape
    dsize = eris.mo_coeff.itemsize
    nvir3 = nvir**3
    Wakic = -lib.einsum('kcli,la->akic', eris.ovoo, t1)
    tmp = np.asarray(eris.ovvo).transpose(2,0,3,1)
    Wakic += tmp
    tmp = None
    ovov = np.asarray(eris.ovov)
    Wakic -= 0.5*lib.einsum('ldkc,ilda->akic', ovov, t2)
    Wakic -= 0.5*lib.einsum('lckd,ilad->akic', ovov, t2)
    Wakic -= lib.einsum('ldkc,id,la->akic', ovov, t1, t1)
    Wakic += lib.einsum('ldkc,ilad->akic', ovov, t2)
    ovov = None
    mem_avail = eris.max_memory - lib.current_memory()[0]
    mem_avail *= 1e6 / dsize
    blksize = min(nocc, int(0.7*(mem_avail - nocc * nvir) / (2*nvir3 + nocc*nvir**2)))
    assert blksize > 0, "enlarge mem"
    for i0, i1 in lib.prange(0, nocc, blksize):
        eris_ovvv = eris.get_ovvv(slice(i0,i1))
        Wakic[:,i0:i1] += lib.einsum('kcad,id->akic', eris_ovvv, t1)
        eris_ovvv = None
    return Wakic

def cc_Wvovo(t1, t2, eris):
    nocc, nvir = t1.shape
    dsize = eris.mo_coeff.itemsize
    nvir3 = nvir**3
    Wakci = -lib.einsum('lcki,la->akci', eris.ovoo, t1)
    tmp = np.asarray(eris.oovv).transpose(2,0,3,1)
    Wakci += tmp
    tmp = None
    ovov = np.asarray(eris.ovov)
    Wakci -= 0.5*lib.einsum('lckd,ilda->akci', ovov, t2)
    Wakci -= lib.einsum('lckd,id,la->akci', ovov, t1, t1)
    ovov = None
    mem_avail = eris.max_memory - lib.current_memory()[0]
    mem_avail *= 1e6 / dsize
    blksize = min(nocc, int(0.7*(mem_avail - nocc * nvir) / (2*nvir3 + nocc*nvir**2)))
    assert blksize > 0, "enlarge mem"
    for i0, i1 in lib.prange(0, nocc, blksize):
        eris_ovvv = eris.get_ovvv(slice(i0,i1))
        Wakci[:,i0:i1] += lib.einsum('kdac,id->akci', eris_ovvv, t1)
        eris_ovvv = None
    return Wakci

