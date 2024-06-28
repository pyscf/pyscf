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
# Author: Ning Zhang <ningzhang1024@gmail.com>
#

'''
Restricted CCSD

Ref: Stanton et al., J. Chem. Phys. 94, 4334 (1990)
Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)
'''

from functools import reduce

import numpy as np

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.pbc.df.isdf.thc_cc_helper import thc_rintermediates as imd
from pyscf.lib import linalg_helper
import pyscf.pbc.df.isdf.thc_cc_helper._einsum_holder as einsum_holder

# note MO integrals are treated in chemist's notation

einsum = einsum_holder.thc_einsum_sybolic

def update_amps(cc, t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris, thc_scheduler:einsum_holder.THC_scheduler):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    #assert (isinstance(eris, ccsd._ChemistsERIs))
    #nocc, nvir = t1.shape
    nocc = eris.nocc
    nvir = eris.nvir
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc].copy()
    mo_e_v = eris.mo_energy[nocc:].copy() + cc.level_shift

    fov = fock[:nocc,nocc:].copy()
    foo = fock[:nocc,:nocc].copy()
    fvv = fock[nocc:,nocc:].copy()
    thc_scheduler.add_input("fov", fov)
    thc_scheduler.add_input("foo", foo)
    thc_scheduler.add_input("fvv", fvv)
    
    foo = einsum_holder._expr_foo()
    fov = einsum_holder._expr_fov()
    fvv = einsum_holder._expr_fvv()

    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)

    # Move energy terms to the other side
    # Foo[np.diag_indices(nocc)] -= mo_e_o
    # Fvv[np.diag_indices(nvir)] -= mo_e_v

    thc_scheduler.add_input("mo_e_o", mo_e_o)
    thc_scheduler.add_input("mo_e_v", mo_e_v)
    
    mo_e_o = einsum_holder._einsum_term("mo_e_o", "ij", 1.0, args=["mo_e_o"])
    mo_e_v = einsum_holder._einsum_term("mo_e_v", "ab", 1.0, args=["mo_e_v"])
    
    Foo -= mo_e_o
    Fvv -= mo_e_v
    
    thc_scheduler.register_intermediates("FOO", Foo)
    thc_scheduler.register_intermediates("FVV", Fvv)
    thc_scheduler.register_intermediates("FOV", Fov)

    # T1 equation
    t1new  =-2*einsum('kc,ka,ic->ia', fov, t1, t1)
    t1new +=   einsum('ac,ic->ia', Fvv, t1)
    t1new +=  -einsum('ki,ka->ia', Foo, t1)
    t1new += 2*einsum('kc,kica->ia', Fov, t2)
    t1new +=  -einsum('kc,ikca->ia', Fov, t2)
    t1new +=   einsum('kc,ic,ka->ia', Fov, t1, t1)
    t1new += fov.conj()
    eris_ovvo = einsum_holder._thc_eri_ovvo()
    eris_oovv = einsum_holder._thc_eri_oovv()
    t1new += 2*einsum('kcai,kc->ia', eris_ovvo, t1)
    t1new +=  -einsum('kiac,kc->ia', eris_oovv, t1)
    eris_ovvv = einsum_holder._thc_eri_ovvv()
    t1new += 2*einsum('kdac,ikcd->ia', eris_ovvv, t2)
    t1new +=  -einsum('kcad,ikcd->ia', eris_ovvv, t2)
    t1new += 2*einsum('kdac,kd,ic->ia', eris_ovvv, t1, t1)
    t1new +=  -einsum('kcad,kd,ic->ia', eris_ovvv, t1, t1)
    eris_ovoo = einsum_holder._thc_eri_ovoo()
    t1new +=-2*einsum('lcki,klac->ia', eris_ovoo, t2)
    t1new +=   einsum('kcli,klac->ia', eris_ovoo, t2)
    t1new +=-2*einsum('lcki,lc,ka->ia', eris_ovoo, t1, t1)
    t1new +=   einsum('kcli,lc,ka->ia', eris_ovoo, t1, t1)

    # T2 equation
    
    tmp2  = einsum('kibc,ka->abic', eris_oovv, -t1)
    tmp2 += eris_ovvv.conj().transpose((1,3,0,2))
    tmp   = einsum('abic,jc->ijab', tmp2, t1)
    t2new = tmp + tmp.transpose((1,0,3,2))
    tmp2  = einsum('kcai,jc->akij', eris_ovvo, t1)
    tmp2 += eris_ovoo.transpose((1,3,0,2)).conj()
    tmp   = einsum('akij,kb->ijab', tmp2, t1)
    t2new -= tmp + tmp.transpose((1,0,3,2))
    eris_ovov = einsum_holder._thc_eri_ovov()
    t2new += eris_ovov.conj().transpose((0,2,1,3))
    
    if cc.cc2:
        eris_oooo = einsum_holder._thc_eri_oooo()
        Woooo2  = eris_oooo.transpose((0,2,1,3))
        Woooo2 += einsum('lcki,jc->klij', eris_ovoo, t1)
        Woooo2 += einsum('kclj,ic->klij', eris_ovoo, t1)
        Woooo2 += einsum('kcld,ic,jd->klij', eris_ovov, t1, t1)
        t2new  += einsum('klij,ka,lb->ijab', Woooo2, t1, t1)
        Wvvvv  = einsum('kcbd,ka->abcd', eris_ovvv, -t1)
        Wvvvv  = Wvvvv + Wvvvv.transpose(1,0,3,2)
        eris_vvvv = einsum_holder._thc_eri_vvvv()
        Wvvvv += eris_vvvv.transpose((0,2,1,3))
        t2new += einsum('abcd,ic,jd->ijab', Wvvvv, t1, t1)
        Lvv2   = fvv - np.einsum('kc,ka->ac', fov, t1)
        fvv_diag = np.diag(np.diag(fvv))
        thc_scheduler.add_input("fvv_diag", fvv_diag)
        fvv_diag = einsum_holder._einsum_term("fvv_diag", "ac", 1.0, args=["fvv_diag"])
        #Lvv2  -= np.diag(np.diag(fvv))
        Lvv2  -= fvv_diag
        thc_scheduler.register_intermediates("Lvv2", Lvv2)
        tmp = einsum('ac,ijcb->ijab', Lvv2, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        Loo2 = foo + einsum('kc,ic->ki', fov, t1)
        #Loo2 -= np.diag(np.diag(foo))
        foo_diag = np.diag(np.diag(foo))
        thc_scheduler.add_input("foo_diag", foo_diag)
        foo_diag = einsum_holder._einsum_term("foo_diag", "ki", 1.0, args=["foo_diag"])
        Loo2 -= foo_diag
        thc_scheduler.register_intermediates("Loo2", Loo2)
        tmp = einsum('ki,kjab->ijab', Loo2, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
    else:
        Loo = imd.Loo(t1, t2, eris)
        Lvv = imd.Lvv(t1, t2, eris)
        #Loo[np.diag_indices(nocc)] -= mo_e_o
        #Lvv[np.diag_indices(nvir)] -= mo_e_v
        Loo -= mo_e_o
        Lvv -= mo_e_v
        
        thc_scheduler.register_intermediates("LOO", Loo)
        thc_scheduler.register_intermediates("LVV", Lvv)

        Woooo = imd.cc_Woooo(t1, t2, eris)
        Wvoov = imd.cc_Wvoov(t1, t2, eris)
        Wvovo = imd.cc_Wvovo(t1, t2, eris)
        Wvvvv = imd.cc_Wvvvv(t1, t2, eris)

        tau = t2 + einsum('ia,jb->ijab', t1, t1)
        t2new += einsum('klij,klab->ijab', Woooo, tau)
        t2new += einsum('abcd,ijcd->ijab', Wvvvv, tau)
        tmp = einsum('ac,ijcb->ijab', Lvv, t2)
        t2new += (tmp + tmp.transpose((1,0,3,2)))
        tmp = einsum('ki,kjab->ijab', Loo, t2)
        t2new -= (tmp + tmp.transpose((1,0,3,2)))
        tmp  = 2*einsum('akic,kjcb->ijab', Wvoov, t2)
        tmp -=   einsum('akci,kjcb->ijab', Wvovo, t2)
        t2new += (tmp + tmp.transpose((1,0,3,2)))
        tmp = einsum('akic,kjbc->ijab', Wvoov, t2)
        t2new -= (tmp + tmp.transpose((1,0,3,2)))
        tmp = einsum('bkci,kjac->ijab', Wvovo, t2)
        t2new -= (tmp + tmp.transpose((1,0,3,2)))

    mo_e_o = eris.mo_energy[:nocc].copy()
    mo_e_v = eris.mo_energy[nocc:].copy() + cc.level_shift
    eia = mo_e_o[:,None] - mo_e_v
    eia = 1.0 / eia
    thc_scheduler.add_input("eia", eia)
    eia = einsum_holder._einsum_term("eia", "ia", 1.0, args=["eia"])
    t1new = einsum('ia,ia->ia', t1new, eia, cached=True)
    
    #eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
    #t2new /= eijab
    t2_occ_vir_str = t2new.occvir_str
    if t2_occ_vir_str == "ovov":
        t2new = t2new.transpose((0,2,1,3))
    ene_denominator = einsum_holder._energy_denominator()
    t2new = einsum('ijab,ijab->ijab', t2new, ene_denominator)

    #return t1new, t2new

    thc_scheduler.register_expr(einsum_holder.THC_scheduler.t1_new_name, t1new)
    thc_scheduler.register_expr(einsum_holder.THC_scheduler.t2_new_name, t2new)


def energy(cc, t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris=None, thc_scheduler:einsum_holder.THC_scheduler=None):
    '''RCCSD correlation energy'''
    #if t1 is None: t1 = cc.t1
    #if t2 is None: t2 = cc.t2
    #if eris is None: eris = cc.ao2mo()
    assert t1 is not None
    assert t2 is not None
    assert eris is not None
    assert thc_scheduler is not None

    #nocc, nvir = t1.shape
    
    fock = eris.fock
    fov = fock[:nocc,nocc:].copy()
    thc_scheduler.add_input("fov", fov)  ## "fov" must be this name here!
    fov_expr = einsum_holder._expr_fov()
    
    e = 2*einsum('ia,ia->', fov_expr, t1, cached=True)
    tau = einsum('ia,jb->ijab', t1, t1)
    t2_str = t2.occvir_str
    if t2_str == "ovov":
        t2 = t2.transpose((0,2,1,3))
    tau += t2
    eris_ovov = einsum_holder._thc_eri_ovov()
    e += 2*einsum('ijab,iajb->', tau, eris_ovov, cached=True)
    e +=  -einsum('ijab,ibja->', tau, eris_ovov, cached=True)
    
    thc_scheduler.register_expr(einsum_holder.THC_scheduler.ccsd_energy_name, e) 

################## used only for testing ##################

class _fake_eris:
    def __init__(self, nocc, nvir):
        self.nocc = nocc
        self.nvir = nvir
        self.fock = np.random.rand(nocc+nvir, nocc+nvir)
        self.fock += self.fock.T
        self.mo_energy = np.random.rand(nocc+nvir)
        self.mo_energy = np.sort(self.mo_energy)
        #self.fock
        np.fill_diagonal(self.fock, self.mo_energy)

class _fake_cc:
    def __init__(self, nocc, nvir, cc2=False):
        self.nocc = nocc
        self.nvir = nvir
        self.level_shift = 0.1
        self.cc2 = cc2

###########################################################

if __name__ == "__main__":
    
    ### generate random input ### 
    
    nocc = 8
    nvir = 12
    nthc = 80
    nlaplace = 9
    
    Xo    = np.random.rand(nocc, nthc)
    Xv    = np.random.rand(nvir, nthc)
    Tau_o = np.random.rand(nocc, nlaplace)
    Tau_v = np.random.rand(nvir, nlaplace)
    THC_INT = np.random.rand(nthc, nthc)
    THC_INT+= THC_INT.T
    Xo_T2 = np.random.rand(nocc, nthc)
    Xv_T2 = np.random.rand(nvir, nthc)
    PROJ  = np.random.rand(nthc, nthc)

        
    THC_T2  = np.random.rand(nthc, nthc)
    THC_T2 += THC_T2.T
    T1      = np.random.rand(nocc, nvir)
    
    scheduler = einsum_holder.THC_scheduler(
        X_O=Xo,
        X_V=Xv,
        TAU_O=Tau_o,
        TAU_V=Tau_v,
        THC_INT=THC_INT,
        T1=T1,
        THC_T2=THC_T2,
        XO_T2=Xo_T2,
        XV_T2=Xv_T2,
        PROJECTOR=PROJ
    )

    eris = _fake_eris(nocc, nvir)
    cc   = _fake_cc(nocc, nvir)
    
    t1 = einsum_holder._expr_t1()
    t2 = einsum_holder._expr_t2()
    t2 = t2.transpose((0,2,1,3))
    
    update_amps(cc, t1, t2, eris, scheduler)
    energy(cc, t1, t2, eris, scheduler)
    
    scheduler._build_expression()
    print(scheduler)
    scheduler._build_contraction(backend="opt_einsum", optimize=True)
    #scheduler._build_contraction(backend="cotengra")
    
    res = scheduler.evaluate_t1_t2(T1, THC_T2)
    print('ene = ', res[0])
    print('t1  = ', res[1])
    print('t2  = ', res[2])