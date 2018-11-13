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
# Restricted Coupled Cluster Greens Functions 
#
# Authors: James D. McClain
#          Jason Yu <jasonmyu1@gmail.com>
#

import collections
import numpy as np
import scipy.sparse.linalg as spla
from pyscf.cc import eom_rccsd
from pyscf.cc.eom_rccsd import EOMIP, EOMEA
from pyscf.pbc.lib import kpts_helper
import time
import sys

###################
# EA Greens       #
###################

def greens_b_vector_ea_rhf(cc, p, kp=None):
    nkpts, nocc, nvir = cc.t1.shape
    ds_type = cc.t1.dtype
    vector1 = np.zeros((nvir),dtype=ds_type)
    vector2 = np.zeros((nkpts,nkpts,nocc,nvir,nvir),dtype=ds_type)
    if p < nocc:
        vector1 += -cc.t1[kp,p,:]
        for ki in range(nkpts):
            for kj in range(nkpts):
                vector2[ki,kj] += -cc.t2[kp,ki,kj,p,:,:,:]
    else:
        vector1[ p-nocc ] = 1.0
    return eom_rccsd.amplitudes_to_vector_ea(vector1,vector2)

def greens_e_vector_ea_rhf(cc, p, kp=None):
    nkpts, nocc, nvir = cc.t1.shape
    ds_type = cc.t1.dtype
    vector1 = np.zeros((nvir),dtype=ds_type)
    vector2 = np.zeros((nkpts,nkpts,nocc,nvir,nvir),dtype=ds_type)

    if hasattr(cc, 'l1') and cc.l1 is not None:
        l1 = cc.l1
        l2 = cc.l2
    else:
        l1 = np.conj(cc.t1)
        l2 = np.conj(cc.t2)

    if p < nocc:
        vector1 += l1[kp,p,:]
        for ki in range(nkpts):
            for kj in range(nkpts):
                kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
                kb = kconserv[ki,kj,kp]
                ka = kconserv[ki,kb,kj]
                vector2[ki,kj] += 2*l2[kp,ki,kj,p,:,:,:]
                vector2[ki,kj] -= l2[ki,kp,kj,:,p,:,:]

    else:
        vector1[ p-nocc ] = -1.0
        vector1 += np.einsum('ia,i->a', l1[kp], cc.t1[kp,:,p-nocc])
        for kk in range(nkpts):
            for kl in range(nkpts):
                kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
                kc = kconserv[kl,kp,kk]
        
                vector1 += 2 * np.einsum('klca,klc->a', l2[kk,kl,kc], \
                           cc.t2[kk,kl,kc,:,:,:,p-nocc])
                vector1 -= np.einsum('klca,lkc->a', l2[kk,kl,kc], \
                           cc.t2[kl,kk,kc,:,:,:,p-nocc])

        for kb in range(nkpts):
            vector2[kb,kp,:,p-nocc,:] += -2.*l1[kb]
    
        for ka in range(nkpts):
            # kj == ka
            # kb == kc == kp
            vector2[ka,ka,:,:,p-nocc] += l1[ka]

        for kj in range(nkpts):
            for ka in range(nkpts):
                kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
                kb = kconserv[kp,kj,ka]
                
                vector2[kj,ka] += 2*np.einsum('k,jkba->jab', \
                                  cc.t1[kp,:,p-nocc], l2[kj,kp,kb,:,:,:,:])
                vector2[kj,ka] -= np.einsum('k,jkab->jab', \
                                  cc.t1[kp,:,p-nocc], l2[kj,kp,ka,:,:,:,:])

    return eom_rccsd.amplitudes_to_vector_ea(vector1,vector2)

###################
# IP Greens       #
###################

def greens_b_vector_ip_rhf(cc,p,kp=None):
    nkpts, nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nkpts,nkpts,nocc,nocc,nvir),dtype=complex)

    if p < nocc:
        vector1[p] = 1.0
    else:
        vector1 += cc.t1[kp,:,p-nocc]
        for ki in range(nkpts):
            for kj in range(nkpts):
                kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
                ka = kconserv[ki,kj,kp]                              
                vector2[ki,kj] += cc.t2[ki,kj,ka,:,:,:,p-nocc] 
    return eom_rccsd.amplitudes_to_vector_ip(vector1,vector2)

def greens_e_vector_ip_rhf(cc,p,kp=None):
    nkpts, nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nkpts,nkpts,nocc,nocc,nvir),dtype=complex)

    if hasattr(cc, 'l1') and cc.l1 is not None:
        l1 = cc.l1
        l2 = cc.l2
    else:
        l1 = np.conj(cc.t1)
        l2 = np.conj(cc.t2)

    if p < nocc:
        vector1[p] = -1.0
        vector1 += np.einsum('ia,a->i', l1[kp], cc.t1[kp,p,:])
        for kl in range(nkpts):
            for kc in range(nkpts):
                 kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
                 kd = kconserv[kp,kl,kc]
                 vector1 += 2 * np.einsum('ilcd,lcd->i', \
                       l2[kp,kl,kc], cc.t2[kp,kl,kc,p,:,:,:])
                 vector1 -= np.einsum('ilcd,ldc->i',   \
                       l2[kp,kl,kc], cc.t2[kp,kl,kd,p,:,:,:])

        for kj in range(nkpts):
            vector2[kp,kj,p,:,:] += -2*l1[kj]

        for ki in range(nkpts):
            # kj == kk == kp, ki == kb
            vector2[ki,kp,:,p,:] +=  l1[ki]

            for kj in range(nkpts):
                # kc == kk == kp
                vector2[ki,kj] += 2*np.einsum('c,ijcb->ijb', \
                       cc.t1[kp,p,:], l2[ki,kj,kp,:,:,:,:])
        
                vector2[ki,kj] -= np.einsum('c,jicb->ijb', \
                       cc.t1[kp,p,:], l2[kj,ki,kp,:,:,:,:]) 

    else:
        vector1 += -l1[kp,:,p-nocc]
        for ki in range(nkpts):
            for kj in range(nkpts):
                kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
                kb = kconserv[ki,kj,kp]
                vector2[ki, kj] += -2*l2[ki,kj,kp,:,:,p-nocc,:] + \
                                   l2[ki,kj,kb,:,:,:,p-nocc]

    return eom_rccsd.amplitudes_to_vector_ip(vector1,vector2)

def greens_func_multiply(ham,vector,linear_part,kp):
    return np.array(ham(vector,kp) + (linear_part)*vector)

def initial_ip_guess(cc):
    nkpts, nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nkpts,nkpts,nocc,nocc,nvir),dtype=complex)
    return eom_rccsd.amplitudes_to_vector_ip(vector1,vector2)

def initial_ea_guess(cc):
    nkpts, nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nvir),dtype=complex)
    vector2 = np.zeros((nkpts,nkpts,nocc,nvir,nvir),dtype=complex)
    return eom_rccsd.amplitudes_to_vector_ea(vector1,vector2)


class OneParticleGF(object):
    def __init__(self, cc, eta=0.01):
        self.cc = cc
        self.eomip = EOMIP(cc)
        self.eomea = EOMEA(cc)
        self.eta = eta

    def solve_ip(self, kptlist, ps, qs, omegas):
        if not isinstance(ps, collections.Iterable): ps = [ps]
        if not isinstance(qs, collections.Iterable): qs = [qs]
        cc = self.cc
        print("solving ip portion")
        S0 = initial_ip_guess(cc)
        gfvals = np.zeros((len(kptlist), len(ps),len(qs),len(omegas)),dtype=complex)
        for kp, ikpt in enumerate(kptlist): 
            e_vector=list()
            for q in qs:
                e_vector.append(greens_e_vector_ip_rhf(cc,q,kp))
            for ip, p in enumerate(ps):
                b_vector = greens_b_vector_ip_rhf(cc,p,kp)
                cc.kshift = kp
                diag = cc.ipccsd_diag(kp)
                for iw, omega in enumerate(omegas):
                    invprecond_multiply = lambda x: x/(omega + diag + 1j*self.eta)
                    def matr_multiply(vector,args=None):
                        return greens_func_multiply(cc.ipccsd_matvec, vector, omega + 1j*self.eta, kp)
                    size = len(b_vector)
                    Ax = spla.LinearOperator((size,size), matr_multiply)
                    mx = spla.LinearOperator((size,size), invprecond_multiply)

                    start = time.time()
                    Sw, info = spla.gcrotmk(Ax, b_vector, x0=S0, atol=0, tol=1e-2)
                    end = time.time()
                    print 'past gcrotmk with info and time',info,(end-start)
                    sys.stdout.flush()

                    if info != 0:
                        raise RuntimeError
                    for iq,q in enumerate(qs):
                        gfvals[kp,ip,iq,iw]  = -np.dot(e_vector[iq],Sw)
        if len(ps) == 1 and len(qs) == 1:
            return gfvals[:,0,0,:]
        else:
            return gfvals

    def solve_ea(self, kptlist, ps, qs, omegas):
        if not isinstance(ps, collections.Iterable): ps = [ps]
        if not isinstance(qs, collections.Iterable): qs = [qs]
        cc = self.cc
        print("solving ea portion")
        S0 = initial_ea_guess(cc)
        gfvals = np.zeros((len(kptlist),len(ps),len(qs),len(omegas)),dtype=complex)
        for kp, ikpt in enumerate(kptlist):
            e_vector=list()
            for p in ps:
                e_vector.append(greens_e_vector_ea_rhf(cc,p,kp))
            for iq, q in enumerate(qs):
                b_vector = greens_b_vector_ea_rhf(cc,q,kp)
                cc.kshift = kp
                diag = cc.eaccsd_diag(kp)
                for iw, omega in enumerate(omegas):
                    invprecond_multiply = lambda x: x/(-omega + diag + 1j*self.eta)
                    def matr_multiply(vector,args=None):
                        return greens_func_multiply(cc.eaccsd_matvec, vector, -omega + 1j*self.eta, kp)
                    size = len(b_vector)
                    Ax = spla.LinearOperator((size,size), matr_multiply)
                    mx = spla.LinearOperator((size,size), invprecond_multiply)
                    
                    start = time.time()
                    Sw, info = spla.gcrotmk(Ax, b_vector, x0=S0, atol=0, tol=1e-2)
                    end = time.time()
                    print 'past gcrotmk with info and time',info,(end-start)
                    sys.stdout.flush()
                    
                    if info != 0:
                        raise RuntimeError
                    for ip,p in enumerate(ps):
                        gfvals[kp,ip,iq,iw] = np.dot(e_vector[ip],Sw)
        if len(ps) == 1 and len(qs) == 1:
            return gfvals[:,0,0,:]
        else:
            return gfvals

    def kernel(self, k, p, q, omegas):
        return self.solve_ip(k, p, q, omegas), self.solve_ea(k, p, q, omegas)
