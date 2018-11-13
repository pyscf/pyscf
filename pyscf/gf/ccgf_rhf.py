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
# Author:
# James D. Mcclain
#

import collections
import numpy as np
import scipy.sparse.linalg as spla
from pyscf.cc import eom_rccsd
from pyscf.cc.eom_rccsd import EOMIP, EOMEA

###################
# EA Greens       #
###################

def greens_b_vector_ea_rhf(cc,p):
    nocc, nvir = cc.t1.shape
    ds_type = cc.t1.dtype
    vector1 = np.zeros((nvir),dtype=ds_type)
    vector2 = np.zeros((nocc,nvir,nvir),dtype=ds_type)
    if p < nocc:
        # Changed both to minus
        vector1 += -cc.t1[p,:]
        vector2 += -cc.t2[p,:,:,:]
    else:
        vector1[ p-nocc ] = 1.0
    return eom_rccsd.amplitudes_to_vector_ea(vector1,vector2)

def greens_e_vector_ea_rhf(cc,p):
    nocc, nvir = cc.t1.shape
    ds_type = cc.t1.dtype
    vector1 = np.zeros((nvir),dtype=ds_type)
    vector2 = np.zeros((nocc,nvir,nvir),dtype=ds_type)

    if hasattr(cc, 'l1') and cc.l1 is not None:
        l1 = cc.l1
        l2 = cc.l2
    else:
        l1 = cc.t1
        l2 = cc.t2

    if p < nocc:
        # Changed both to plus
        vector1 += l1[p,:]
        vector2 += (2*l2[p,:,:,:] - l2[:,p,:,:])
    else:
        vector1[ p-nocc ] = -1.0
        vector1 += np.einsum('ia,i->a', l1, cc.t1[:,p-nocc])

        vector1 += 2*np.einsum('klca,klc->a', l2, cc.t2[:,:,:,p-nocc])
        vector1 -=   np.einsum('klca,lkc->a', l2, cc.t2[:,:,:,p-nocc])

        vector2[:,p-nocc,:] += -2.*l1
        vector2[:,:,p-nocc] += l1

        vector2 += 2*np.einsum('k,jkba->jab', cc.t1[:,p-nocc], l2)
        vector2 -= np.einsum('k,jkab->jab', cc.t1[:,p-nocc], l2)

    return eom_rccsd.amplitudes_to_vector_ea(vector1,vector2)

###################
# IP Greens       #
###################

def greens_b_vector_ip_rhf(cc,p):
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=complex)
    if p < nocc:
        vector1[ p ] = 1.0
    else:
        vector1 += cc.t1[:,p-nocc]
        vector2 += cc.t2[:,:,:,p-nocc]
    return eom_rccsd.amplitudes_to_vector_ip(vector1,vector2)

def greens_e_vector_ip_rhf(cc,p):
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=complex)

    if hasattr(cc, 'l1') and cc.l1 is not None:
        l1 = cc.l1
        l2 = cc.l2
    else:
        l1 = cc.t1
        l2 = cc.t2

    if p < nocc:
        vector1[ p ] = -1.0
        vector1 += np.einsum('ia,a->i', l1, cc.t1[p,:])
        vector1 += 2*np.einsum('ilcd,lcd->i', l2, cc.t2[p,:,:,:])
        vector1 -=   np.einsum('ilcd,ldc->i', l2, cc.t2[p,:,:,:])

        vector2[p,:,:] += -2.*l1
        vector2[:,p,:] += l1
        vector2 += 2*np.einsum('c,ijcb->ijb', cc.t1[p,:], l2)
        vector2 -=   np.einsum('c,jicb->ijb', cc.t1[p,:], l2)
    else:
        vector1 += -l1[:,p-nocc]
        vector2 += -2*l2[:,:,p-nocc,:] + l2[:,:,:,p-nocc]
    return eom_rccsd.amplitudes_to_vector_ip(vector1,vector2)

def greens_func_multiply(ham,vector,linear_part,*args):
    return np.array(ham(vector,*args) + (linear_part)*vector)

def initial_ip_guess(cc):
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=complex)
    return eom_rccsd.amplitudes_to_vector_ip(vector1,vector2)

def initial_ea_guess(cc):
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nvir),dtype=complex)
    vector2 = np.zeros((nocc,nvir,nvir),dtype=complex)
    return eom_rccsd.amplitudes_to_vector_ea(vector1,vector2)


class OneParticleGF(object):
    def __init__(self, cc, eta=0.01):
        self.cc = cc
        self.eomip = EOMIP(cc)
        self.eomea = EOMEA(cc)
        self.eta = eta

    def solve_ip(self, ps, qs, omegas):
        if not isinstance(ps, collections.Iterable): ps = [ps]
        if not isinstance(qs, collections.Iterable): qs = [qs]
        cc = self.cc
        print("solving ip portion")
        Sw = initial_ip_guess(cc)
        Sw += np.random.rand(Sw.shape[0])
        diag = self.eomip.get_diag()
        imds = self.eomip.make_imds()
        e_vector = list()
        for q in qs:
            e_vector.append(greens_e_vector_ip_rhf(cc,q))
        gfvals = np.zeros((len(ps),len(qs),len(omegas)),dtype=complex)
        for ip, p in enumerate(ps):
            print 'gf idx', ip
            b_vector = greens_b_vector_ip_rhf(cc,p)
            for iw, omega in enumerate(omegas):
                invprecond_multiply = lambda x: x/(omega + diag + 1j*self.eta)
                def matr_multiply(vector,args=None):
                    return greens_func_multiply(self.eomip.matvec, vector, omega + 1j*self.eta, imds)
                size = len(b_vector)
                Ax = spla.LinearOperator((size,size), matr_multiply)
                mx = spla.LinearOperator((size,size), invprecond_multiply)
                Sw, info = spla.gmres(Ax, b_vector, x0=Sw, tol=1e-14, M=mx)
                if info != 0:
                    raise RuntimeError
                for iq,q in enumerate(qs):
                    gfvals[ip,iq,iw]  = -np.dot(e_vector[iq],Sw)
        if len(ps) == 1 and len(qs) == 1:
            return gfvals[0,0,:]
        else:
            return gfvals

    def solve_ea(self, ps, qs, omegas):
        if not isinstance(ps, collections.Iterable): ps = [ps]
        if not isinstance(qs, collections.Iterable): qs = [qs]
        cc = self.cc
        print("solving ea portion")
        Sw = initial_ea_guess(cc)
        diag = self.eomea.get_diag()
        e_vector = list()
        for p in ps:
            e_vector.append(greens_e_vector_ea_rhf(cc,p))
        gfvals = np.zeros((len(ps),len(qs),len(omegas)),dtype=complex)
        for iq, q in enumerate(qs):
            b_vector = greens_b_vector_ea_rhf(cc,q)
            for iw, omega in enumerate(omegas):
                invprecond_multiply = lambda x: x/(-omega + diag + 1j*self.eta)
                def matr_multiply(vector,args=None):
                    return greens_func_multiply(self.eomea.matvec, vector, -omega + 1j*self.eta)
                size = len(b_vector)
                Ax = spla.LinearOperator((size,size), matr_multiply)
                mx = spla.LinearOperator((size,size), invprecond_multiply)
                Sw, info = spla.gmres(Ax, b_vector, x0=Sw, tol=1e-15, M=mx)
                for ip,p in enumerate(ps):
                    gfvals[ip,iq,iw] = np.dot(e_vector[ip],Sw)
        if len(ps) == 1 and len(qs) == 1:
            return gfvals[0,0,:]
        else:
            return gfvals

    def kernel(self, p, q, omegas):
        return self.solve_ip(p, q, omegas), self.solve_ea(p, q, omegas)
