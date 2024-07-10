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

class _energy_denominator_ijkabc(einsum_holder._einsum_term):
    def __init__(self):
        super().__init__("ene_deno", "iT,jT,kT,aT,bT,cT->ijkabc", args=["TAUO","TAUO","TAUO","TAUV","TAUV","TAUV"]) 
        
def _apply_PL(W:einsum_holder._expr_holder):
    
    '''
    the input tensor is of the order iajbkc 
    '''
    
    return W + W.transpose((0,1,4,5,2,3)) + W.transpose((2,3,0,1,4,5)) + W.transpose((2,3,4,5,0,1)) + W.transpose((4,5,0,1,2,3)) + W.transpose((4,5,2,3,0,1))

def _apply_PS(V:einsum_holder._expr_holder):
    
    '''
    the input tensor is of the order iajbkc 
    '''
    
    return V + V.transpose((2,3,0,1,4,5)) + V.transpose((4,5,0,1,2,3))


def _rccsd_t_kernel(mycc=None, eris=None, t1:einsum_holder._expr_holder=None, t2:einsum_holder._expr_holder=None, verbose=logger.NOTE):
    #if isinstance(verbose, logger.Logger):
    #    log = verbose
    #else:
    #    if mycc is not None:
    #        log = logger.Logger(mycc.stdout, verbose)
    #    else:
    #        log = logger.Logger()

    if eris is not None:
        eris_ovvv = einsum_holder._thc_eri_ovvv()
        eris_ovoo = einsum_holder._thc_eri_ovoo()
        eris_ovov = einsum_holder._thc_eri_ovov()
    else:
        eris_ovvv = eris.ovvv
        eris_ovoo = eris.ovoo
        eris_ovov = eris.ovov
    
    W = einsum("iabd,kcjd->iajbkc", eris_ovvv, t2)
    W-= einsum("iajl,kclb->iajbkc", eris_ovoo, t2) 
    W = _apply_PL(W)
    
    V = einsum("ia,jbkc->iajbkc", t1, eris_ovov)
    V = _apply_PS(V)
    V+= W
    
    W1 = 4 * W + W.transpose((0,3,2,5,4,1)) + W.transpose((0,5,2,1,4,3))
    V1 = V - V.transpose((0,5,2,3,4,1))
    
    e_ijkabc = _energy_denominator_ijkabc()
    
    return (1.0/3.0) * einsum("iajbkc,iajbkc,ijkabc->", W1, V1, e_ijkabc)

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
        PROJECTOR=PROJ
    )
    
    t1 = einsum_holder._expr_t1()
    t2 = einsum_holder._expr_t2()
    
    rccsd_t_eq = _rccsd_t_kernel(t1=t1,t2=t2)
    
    scheduler.register_expr("rccsd_t", rccsd_t_eq) 
    
    scheduler._build_expression()
    print(scheduler)