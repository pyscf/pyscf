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

def update_amps(cc, t1:einsum_holder._expr_holder, t2:einsum_holder._expr_holder, eris, thc_scheduler:einsum_holder.THC_scheduler=None, 
                t2_with_denominator=True):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    #assert (isinstance(eris, ccsd._ChemistsERIs))
    #nocc, nvir = t1.shape
    nocc = eris.nocc
    nvir = eris.nvir
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc].copy()
    mo_e_o = np.diag(mo_e_o)
    mo_e_v = eris.mo_energy[nocc:].copy() + cc.level_shift
    mo_e_v = np.diag(mo_e_v)

    fov = fock[:nocc,nocc:].copy()
    foo = fock[:nocc,:nocc].copy()
    fvv = fock[nocc:,nocc:].copy()
    
    fvv_val = fvv.copy()
    foo_val = foo.copy()

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
    
    if eris is None:
        eris_ovvo = einsum_holder._thc_eri_ovvo()
        eris_oovv = einsum_holder._thc_eri_oovv()
        eris_ovvv = einsum_holder._thc_eri_ovvv()
        eris_ovoo = einsum_holder._thc_eri_ovoo()
        eris_ovov = einsum_holder._thc_eri_ovov()
        eris_oooo = einsum_holder._thc_eri_oooo()
        eris_vvvv = einsum_holder._thc_eri_vvvv()
    else:
        eris_ovvo = eris.ovvo
        eris_oovv = eris.oovv
        eris_ovvv = eris.ovvv
        eris_ovoo = eris.ovoo
        eris_ovov = eris.ovov
        eris_oooo = eris.oooo
        eris_vvvv = eris.vvvv
           

    # T1 equation
    t1new = einsum_holder.to_expr_holder(fov.conj(), cached=True)
    t1new  +=-2*einsum('kc,ka,ic->ia', fov, t1, t1)
    t1new +=   einsum('ac,ic->ia', Fvv, t1)
    t1new +=  -einsum('ki,ka->ia', Foo, t1)
    t1new += 2*einsum('kc,kica->ia', Fov, t2)
    t1new +=  -einsum('kc,ikca->ia', Fov, t2)
    t1new +=   einsum('kc,ic,ka->ia', Fov, t1, t1)
    t1new += 2*einsum('kcai,kc->ia', eris_ovvo, t1)
    t1new +=  -einsum('kiac,kc->ia', eris_oovv, t1)
    t1new += 2*einsum('kdac,ikcd->ia', eris_ovvv, t2)
    t1new +=  -einsum('kcad,ikcd->ia', eris_ovvv, t2)
    t1new += 2*einsum('kdac,kd,ic->ia', eris_ovvv, t1, t1)
    t1new +=  -einsum('kcad,kd,ic->ia', eris_ovvv, t1, t1)
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
    t2new += eris_ovov.conj().transpose((0,2,1,3))
    
    if cc.cc2:
        Woooo2  = eris_oooo.transpose((0,2,1,3))
        Woooo2 += einsum('lcki,jc->klij', eris_ovoo, t1)
        Woooo2 += einsum('kclj,ic->klij', eris_ovoo, t1)
        Woooo2 += einsum('kcld,ic,jd->klij', eris_ovov, t1, t1)
        t2new  += einsum('klij,ka,lb->ijab', Woooo2, t1, t1)
        Wvvvv  = einsum('kcbd,ka->abcd', eris_ovvv, -t1)
        Wvvvv  = Wvvvv + Wvvvv.transpose((1,0,3,2))
        Wvvvv += eris_vvvv.transpose((0,2,1,3))
        t2new += einsum('abcd,ic,jd->ijab', Wvvvv, t1, t1)
        Lvv2   = einsum_holder.to_expr_holder(fvv) - einsum('kc,ka->ac', fov, t1)
        fvv_diag = np.diag(np.diag(fvv_val))
        thc_scheduler.add_input("fvv_diag", fvv_diag)
        fvv_diag = einsum_holder._einsum_term("fvv_diag", "ac", 1.0, args=["fvv_diag"])
        #Lvv2  -= np.diag(np.diag(fvv))
        Lvv2  -= fvv_diag
        Lvv2.cached = True
        thc_scheduler.register_intermediates("Lvv2", Lvv2)
        tmp = einsum('ac,ijcb->ijab', Lvv2, t2)
        t2new += (tmp + tmp.transpose((1,0,3,2)))
        Loo2 = foo + einsum('kc,ic->ki', fov, t1)
        #Loo2 -= np.diag(np.diag(foo))
        foo_diag = np.diag(np.diag(foo_val))
        thc_scheduler.add_input("foo_diag", foo_diag)
        foo_diag = einsum_holder._einsum_term("foo_diag", "ki", 1.0, args=["foo_diag"])
        Loo2 -= foo_diag
        Loo2.cached = True
        thc_scheduler.register_intermediates("Loo2", Loo2)
        tmp = einsum('ki,kjab->ijab', Loo2, t2)
        t2new -= (tmp + tmp.transpose((1,0,3,2)))
    else:
        Loo = imd.Loo(t1, t2, eris)
        Lvv = imd.Lvv(t1, t2, eris)
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
    #print("eia = ", eia)
    thc_scheduler.add_input("eia", eia)
    eia = einsum_holder._einsum_term("eia", "ia", 1.0, args=["eia"])
    t1new = einsum('ia,ia->ia', t1new, eia, cached=True)
    
    #eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
    #t2new /= eijab
    t2_occ_vir_str = t2new.occvir_str
    if t2_occ_vir_str == "ovov":
        t2new = t2new.transpose((0,2,1,3))
    if t2_with_denominator:
        ene_denominator = einsum_holder._energy_denominator()
        t2new = einsum('ijab,ijab->ijab', t2new, ene_denominator)
    t2new = -t2new # the sign is due to the fact that ene_denominator is positive defniete
                   # but in pyscf the var eijab = (ei-ea+ej-eb) which is negative definite

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
    nocc = eris.nocc    
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
        #eri exprs
        self.ovvo = einsum_holder._thc_eri_ovvo()
        self.oovv = einsum_holder._thc_eri_oovv()
        self.ovov = einsum_holder._thc_eri_ovov()
        self.ovoo = einsum_holder._thc_eri_ovoo()
        self.ovvv = einsum_holder._thc_eri_ovvv()
        self.oooo = einsum_holder._thc_eri_oooo()
        self.vvvv = einsum_holder._thc_eri_vvvv()
        

class _fake_eris_full:
    def __init__(self, fock, Xo, Xv, THC_INT):
        
        self.fock = fock
        self.mo_energy = np.diag(fock)
        self.ovvo = np.einsum("iP,aP,PQ,jQ,bQ->iabj", Xo, Xv, THC_INT, Xo, Xv, optimize=True)
        self.oovv = np.einsum("iP,jP,PQ,aQ,bQ->ijab", Xo, Xo, THC_INT, Xv, Xv, optimize=True)
        self.ovov = np.einsum("iP,aP,PQ,jQ,bQ->iajb", Xo, Xv, THC_INT, Xo, Xv, optimize=True)
        self.ovoo = np.einsum("iP,aP,PQ,jQ,kQ->iajk", Xo, Xv, THC_INT, Xo, Xo, optimize=True)
        self.ovvv = np.einsum("iP,aP,PQ,bQ,cQ->iabc", Xo, Xv, THC_INT, Xv, Xv, optimize=True)
        self.ooov = np.einsum("iP,jP,PQ,kQ,aQ->ijka", Xo, Xo, THC_INT, Xo, Xv, optimize=True)
        self.oooo = np.einsum("iP,jP,PQ,kQ,lQ->ijkl", Xo, Xo, THC_INT, Xo, Xo, optimize=True)
        self.vvvv = np.einsum("aP,bP,PQ,cQ,dQ->abcd", Xv, Xv, THC_INT, Xv, Xv, optimize=True)
        self.nocc = Xo.shape[0]
        self.nvir = Xv.shape[0]
        self.nthc = THC_INT.shape[0]
    
    def get_ovvv(self):
        return self.ovvv

class _fake_cc:
    def __init__(self, nocc, nvir, cc2=False):
        self.nocc = nocc
        self.nvir = nvir
        self.level_shift = 0.1
        self.cc2 = cc2

###########################################################

def _tensor_to_cpu(arg):
    try:
        import torch    
        if isinstance(arg, torch.Tensor):
            return arg.cpu().numpy()
        else:
            return arg
    except Exception as e:
        return arg

if __name__ == "__main__":
    
    ### generate random input ### 
    
    nocc = 4
    nvir = 5
    nthc = 17
    nlaplace = 7
    
    Xo    = np.random.rand(nocc, nthc) * 0.1
    Xv    = np.random.rand(nvir, nthc) * 0.1
    Tau_o = np.random.rand(nocc, nlaplace) * 0.1
    Tau_v = np.random.rand(nvir, nlaplace) * 0.1
    THC_INT = np.random.rand(nthc, nthc) * 0.1
    THC_INT+= THC_INT.T
    Xo_T2 = np.random.rand(nocc, nthc) * 0.1
    Xv_T2 = np.random.rand(nvir, nthc) * 0.1
    PROJ  = np.random.rand(nthc, nthc) * 0.1
    PROJ += PROJ.T
        
    THC_T2  = np.random.rand(nthc, nthc) * 0.1
    THC_T2 += THC_T2.T
    T1      = np.random.rand(nocc, nvir) * 0.1
    
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
        PROJECTOR=PROJ,
        use_torch=True,
        with_gpu=True
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
    #print('ene = ', res[0])
    #print('t1  = ', res[1])
    #print('t2  = ', res[2])
    
    t1 = T1
    t2_full = np.einsum("iP,aP,PQ,jQ,bQ->ijab", Xo_T2, Xv_T2, THC_T2, Xo_T2, Xv_T2, optimize=True)
    
    eris_full = _fake_eris_full(eris.fock, Xo, Xv, THC_INT)
    
    #### bench mark #### 
    
    import pyscf.cc.rccsd_slow as rccsd_slow
    import pyscf.cc.rccsd as rccsd
    import pyscf.cc.rintermediates as r_imd
    
    ene_benchmark = rccsd_slow.energy(cc, t1, t2_full, eris_full)
    print("ene_benchmark = ", ene_benchmark)

    scheduler.update_t1(T1)
    scheduler.update_t2(THC_T2)
    
    ene = scheduler._evaluate(einsum_holder.THC_scheduler.ccsd_energy_name)
    print("ene           = ", ene)
    
    try:
        assert np.allclose(ene, ene_benchmark)
    except Exception as e:
        ene = ene.cpu()
        assert np.allclose(ene, ene_benchmark)
    
    ########## bench mark Foo ##########
    
    Foo_bench = r_imd.cc_Foo(t1, t2_full, eris_full)
    mo_e_o = eris.mo_energy[:nocc].copy()
    Foo_bench -= np.diag(mo_e_o)
    
    Foo = scheduler._evaluate("FOO")
    Foo = _tensor_to_cpu(Foo)
    
    diff = np.linalg.norm(Foo - Foo_bench)
    print("diff = ", diff)
    
    assert np.allclose(Foo, Foo_bench)
    
    ########## bench mark Fov ########## 
    
    Fov_bench = r_imd.cc_Fov(t1, t2_full, eris_full)
    Fov = scheduler._evaluate("FOV")
    Fov = _tensor_to_cpu(Fov)
    
    diff = np.linalg.norm(Fov - Fov_bench)
    print("diff = ", diff)
    
    assert np.allclose(Fov, Fov_bench)
    
    ########## bench mark Fvv ##########
    
    Fvv_bench = r_imd.cc_Fvv(t1, t2_full, eris_full)
    mo_e_v = eris.mo_energy[nocc:].copy() + cc.level_shift
    #mo_e_v = eris.mo_energy[nocc:].copy()
    Fvv_bench -= np.diag(mo_e_v)
    
    Fvv = scheduler._evaluate("FVV")
    Fvv = _tensor_to_cpu(Fvv)
    
    diff = np.linalg.norm(Fvv - Fvv_bench)
    print("diff = ", diff)
    
    assert np.allclose(Fvv, Fvv_bench)
    
    ########## bench t1 t2 ########## 
    
    t1_new, t2_new = rccsd.update_amps(cc, t1, t2_full, eris_full)
    t2_new = -t2_new
    
    ene, t1_new_2, thc_t2_new = scheduler.evaluate_t1_t2(T1, THC_T2)
    print("ene = ", ene)
    t1_new_2 = _tensor_to_cpu(t1_new_2)
    thc_t2_new = _tensor_to_cpu(thc_t2_new)
    diff_t1 = np.linalg.norm(t1_new - t1_new_2)
    
    print("t1_new   = ", t1_new)
    print("t1_new_2 = ", t1_new_2)
    
    print("diff_t1 = ", diff_t1)
    assert np.allclose(t1_new, t1_new_2)
    
    eia = mo_e_o[:,None] - mo_e_v
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
    t2_new *= eijab
    eijab_new = np.einsum("iP,jP,aP,bP->ijab", Tau_o, Tau_o, Tau_v, Tau_v, optimize=True)
    t2_new *= eijab_new
    
    t2_new_projected = np.einsum("AP,iP,aP,ijab,jQ,bQ,QB->AB", PROJ, Xo_T2, Xv_T2, t2_new, Xo_T2, Xv_T2, PROJ, optimize=True) 
    diff = np.linalg.norm(thc_t2_new - t2_new_projected)
    print(thc_t2_new)
    print(t2_new_projected)
    print(thc_t2_new-t2_new_projected)
    print("diff_t2 = ", diff)