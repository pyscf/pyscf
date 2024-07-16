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

import numpy
import pyscf
from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo.incore import iden_coeffs
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts_helper
from pyscf.lib import logger
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique
from pyscf import __config__
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv, _contract_compact
import pyscf.pbc.gto as pbcgto

import numpy as np
import ctypes
libpbc = lib.load_library('libpbc')

from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time
import pyscf.pbc.df.isdf.isdf_linear_scaling as ISDF
from pyscf.pbc.df.isdf.isdf_tools_cell import build_supercell, build_supercell_with_partition
from pyscf.pbc.df.isdf.isdf_ao2mo import LS_THC, LS_THC_eri, laplace_holder

from pyscf.pbc.df.isdf.isdf_posthf import _restricted_THC_posthf_holder 

from pyscf.pbc.df.isdf.thc_einsum     import thc_einsum, energy_denomimator, thc_holder, t2_thc_robust_proj_holder, ccsd_t2_holder
from pyscf.mp.mp2                     import MP2


from pyscf.cc import ccsd
from pyscf import __config__
import pyscf.pbc.df.isdf.thc_cc_helper._thc_rccsd as _thc_rccsd_ind
import pyscf.pbc.df.isdf.thc_cc_helper._thc_eom_rccsd as _thc_eom_rccsd_ind
import pyscf.pbc.df.isdf.thc_cc_helper._einsum_holder as einsum_holder
from pyscf.pbc.df.isdf.thc_rccsd import THC_RCCSD, _THC_ERIs, _make_eris_incore

from pyscf.pbc.df.isdf.thc_backend import *

import pyscf.cc.eom_rccsd as eom_rccsd

########################################
# THC-EOM-IP-CCSD
########################################

class THC_EOM_IP_RCCSD(eom_rccsd.EOM):
    
    def __init__(self, 
                 cc, 
                 **kwargs):
        
        #### initialization ####
        
        eom_rccsd.EOM.__init__(self, cc)
        
        my_mf = cc._scf
        my_isdf = my_mf.with_df
                
        self._backend = self._cc._backend
        self._memory  = self._cc._memory
        self._use_torch = self._cc._use_torch
        self._with_gpu  = self._cc._with_gpu
        self.nthc = self._cc.X_o.shape[1]
        self.eris = self._cc._thc_eris
        self._t2_with_denominator = self._cc._t2_with_denominator
        self._imds = None
        self._imds = self.make_imds()
        
        self.nocc = self._cc.nocc
        self.nmo = self._cc.nmo
        self.nvir = self.nmo - self.nocc
        
        ## NOTE: all the THC information is contained in self._cc
        
        #### init t1 and t2 #### 
        
        self.t1, self.t2 = self._cc.t1, self._cc.t2
        self._t1_expr, self._t2_expr = self._cc._t1_expr, self._cc._t2_expr
    
        #### init scheduler #### 

        self._thc_scheduler = einsum_holder.THC_scheduler(
            X_O=self._cc.X_o, X_V=self._cc.X_v, THC_INT=self._cc.Z,
            T1=self._cc.t1, 
            XO_T2=self._cc.X_o, XV_T2=self._cc.X_v, THC_T2=self._cc.t2,
            TAU_O=self._cc.tau_o, TAU_V=self._cc.tau_v,
            grid_partition=self._cc.grid_partition, ## possible None
            proj_type=self._cc.proj_type,
            use_torch=self._cc._use_torch,
            with_gpu=self._cc._with_gpu,
            **kwargs
        )
    
        ### build exprs ###
        
        self._build_r_matvec = False
        self._build_l_matvec = False
    
    def make_imds(self, eris=None):
        if hasattr(self, '_imds'):
            if self._imds is not None:
                return self._imds
        if eris is None:
            eris = self.eris
        #print("make_imds is called")
        imds = _thc_eom_rccsd_ind._IMDS_symbolic(self._cc, eris=eris, MRPT2=self._cc.mbpt2)
        imds.make_ip(self.partition)
        self._imds = imds
        return imds
    
    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nocc + nocc*nocc*nvir
    
    @property
    def eip(self):
        return self.e
    
    ######### identical to the original EOM-IP #########

    spatial2spin = staticmethod(eom_rccsd.spatial2spin_ip)
    
    @staticmethod
    def amplitudes_to_vector(r1, r2):
        r1 = to_numpy_cpu(r1)
        r2 = to_numpy_cpu(r2)
        vector = np.hstack((r1, r2.ravel()))
        return vector
    
    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None:
            nmo = self.nmo
        if nocc is None:
            nocc = self.nocc
        nvir = nmo - nocc
        r1 = vector[:nocc].copy()
        r2 = vector[nocc:].copy().reshape(nocc,nocc,nvir)
        #if self._use_torch:
        #    r1 = to_torch(r1, self._with_gpu)
        #    r2 = to_torch(r2, self._with_gpu)
        return r1, r2
    
    ######### slightly different to the original EOM #########
    
    def kernel(self, nroots=1, left=False, koopmans=False, guess=None,
           partition=None, eris=None, imds=None):
        self._nroots = nroots  ## used to build exprs!
        return eom_rccsd.ipccsd(self, nroots, left, koopmans, guess, partition, eris, imds)
    
    ipccsd = kernel
    
    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        res = eom_rccsd.EOMIP.get_init_guess(self, nroots, koopmans, diag)
        #self._nroots = nroots  ### used in generate expression
        if hasattr(self, '_nroots'):
            assert self._nroots == nroots
        else:
            self._nroots = nroots
        return res
    
    ######### not identical to the original EOM #########
    
    def get_diag(self, imds=None):
        
        if hasattr(self, '_diag'):
            if self._diag is not None:
                return self._diag
        
        if imds is None: imds = self.make_imds()
        t1, t2 = self.t1, self.t2
        t1 = to_numpy_cpu(t1)
        t2 = to_numpy_cpu(t2)
        dtype = np.result_type(t1, t2)
        nocc, nvir = t1.shape
        fock = imds.eris.fock
        foo = fock[:nocc,:nocc].copy()
        fvv = fock[nocc:,nocc:].copy()
        
        Hr1 = np.zeros((nocc), dtype)
        Hr2 = np.zeros((nocc,nocc,nvir), dtype)
        
        if self._use_torch:
            Hr1 = to_torch(Hr1, self._with_gpu)
            Hr2 = to_torch(Hr2, self._with_gpu)
        
        ############# build local scheduler and local exprs #############
        
        local_scheduler = einsum_holder.THC_scheduler(
            X_O=self._cc.X_o, X_V=self._cc.X_v, THC_INT=self._cc.Z,
            T1=self._cc.t1, 
            XO_T2=self._cc.X_o, XV_T2=self._cc.X_v, THC_T2=self._cc.t2,
            TAU_O=self._cc.tau_o, TAU_V=self._cc.tau_v,
            grid_partition=self._cc.grid_partition, ## possible None
            proj_type=self._cc.proj_type,
            use_torch=self._cc._use_torch,
            with_gpu=self._cc._with_gpu
        )

        ###### register foo, fvv ######
        fock = imds.eris.fock
        foo = fock[:nocc,:nocc].copy()
        fvv = fock[nocc:,nocc:].copy()
        fov = fock[:nocc,nocc:].copy()
        local_scheduler.add_input("foo", foo)
        local_scheduler.add_input("fvv", fvv)
        local_scheduler.add_input("fov", fov)
        foo = einsum_holder.to_expr_holder(einsum_holder._expr_foo())
        fvv = einsum_holder.to_expr_holder(einsum_holder._expr_fvv())
        fov = einsum_holder.to_expr_holder(einsum_holder._expr_fov())
        ###############################

        local_scheduler.register_expr("LOO", imds.Loo)
        if self.partition != 'mp':
            local_scheduler.register_expr("LVV", imds.Lvv)
            
            Woooo_ij = einsum_holder.thc_einsum_sybolic("ijij->ij", imds.Woooo)
            Wovvo_jb = einsum_holder.thc_einsum_sybolic("jbbj->jb", imds.Wovvo)
            Wovov_jb = einsum_holder.thc_einsum_sybolic("jbjb->jb", imds.Wovov)
            Hr2_0    = einsum_holder.thc_einsum_sybolic("jiba,ijab->ijb", imds.Woovv, self._t2_expr)
            Hr2_1    = einsum_holder.thc_einsum_sybolic("ijba,ijab->ijb", imds.Woovv, self._t2_expr)

            local_scheduler.register_expr("Woooo_ij", Woooo_ij)
            local_scheduler.register_expr("Wovvo_jb", Wovvo_jb)
            local_scheduler.register_expr("Wovov_jb", Wovov_jb)
            local_scheduler.register_expr("Hr2_0", Hr2_0)
            local_scheduler.register_expr("Hr2_1", Hr2_1)
            
        local_scheduler._build_expression()
        local_scheduler._build_contraction(backend=self._backend)
        
        #################################################################
        
        ############# perform calculation #############
        
        local_scheduler.evaluate_all()
        
        Loo = local_scheduler.get_tensor("LOO")
        
        ######## Hr1 ########
        
        if self._use_torch:  ### USE: autoray's DO function
            Loo = Loo.cpu().numpy()
            Hr1 = Hr1.cpu().numpy()
            Hr1 -= np.diag(Loo)
            Hr1 = to_torch(Hr1, self._with_gpu)
        else:
            Hr1 -= np.diag(Loo)
        
        ######## Hr2 ######## 
        
        if self.partition == 'mp':
            for i in range(nocc):
                for j in range(nocc):
                    for a in range(nvir):
                        Hr2[i,j,a] = fvv[a,a] - foo[i,i] - foo[j,j]
        else:
            
            Lvv      = local_scheduler.get_tensor("LVV")
            Woooo_ij = local_scheduler.get_tensor("Woooo_ij")
            Wovvo_jb = local_scheduler.get_tensor("Wovvo_jb")
            Wovov_jb = local_scheduler.get_tensor("Wovov_jb")
            Hr2_0    = local_scheduler.get_tensor("Hr2_0")
            Hr2_1    = local_scheduler.get_tensor("Hr2_1")

            #print("Wovvo_jb.shape = ", Wovvo_jb.shape)

            Hr2 += (-2*Hr2_0+Hr2_1)
            for b in range(nvir):
                Hr2[:,:,b] += Lvv[b,b]
            for i in range(nocc):
                Hr2[i,:,:] -= Loo[i,i]
                Hr2[:,i,:] -= Loo[i,i]
                #for b in range(nvir):
                #    Hr2[i,i,b] -= Wovvo_jb[i,b]
                Hr2[i,i,:] -= Wovvo_jb[i,:]
            for b in range(nvir):
                Hr2[:,:,b] += Woooo_ij
            for i in range(nocc):
                Hr2[i,:,:] += 2*Wovvo_jb
                Hr2[:,i,:] -= Wovov_jb
                Hr2[i,:,:] -= Wovov_jb
        
        local_scheduler = None
        
        ###############################################
        
        vector = self.amplitudes_to_vector(Hr1, Hr2)
        
        self._diag = vector
        
        return vector

    def _l_matvec(self, vectors):
        r1 = []
        r2 = []
        #for i in range(self._nroots):
        for i in range(len(vectors)):
            r1_, r2_ = self.vector_to_amplitudes(vectors[i], self.nmo, self.nocc)
            r1.append(r1_)
            r2.append(r2_)
        if self._use_torch:
            r1 = [to_torch(r1[i], self._with_gpu) for i in range(len(vectors))]
            r2 = [to_torch(r2[i], self._with_gpu) for i in range(len(vectors))]
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
        else:
            r1 = np.array(r1)
            r2 = np.array(r2)
        #if self._nroots == 1:
        #    r1 = r1[0]
        #    r2 = r2[0]
        res = []
        for i in range(len(vectors)):
            self._thc_scheduler.update_r1(r1[i])
            self._thc_scheduler.update_r2(r2[i])
            hr1 = self._thc_scheduler._evaluate(einsum_holder.THC_scheduler.ip_hr1_l_name)
            hr2 = self._thc_scheduler._evaluate(einsum_holder.THC_scheduler.ip_hr2_l_name)
        #if self._nroots == 1:
        #    return [self.amplitudes_to_vector(hr1, hr2)]
        #else:
            #for i in range(self._nroots):
            res.append(self.amplitudes_to_vector(hr1, hr2))
        return res
    
    def _r_matvec(self, vectors, imds=None, diag=None):
        r1 = []
        r2 = []
        #print("vectors.shape = ", len(vectors))
        #for i in range(self._nroots):
        for i in range(len(vectors)):
            r1_, r2_ = self.vector_to_amplitudes(vectors[i], self.nmo, self.nocc)
            r1.append(r1_)
            r2.append(r2_)
        if self._use_torch:
            r1 = [to_torch(r1[i], self._with_gpu) for i in range(len(vectors))]
            r2 = [to_torch(r2[i], self._with_gpu) for i in range(len(vectors))]
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
        else:
            r1 = np.array(r1)
            r2 = np.array(r2)
        #if self._nroots == 1:
        #    r1 = r1[0]
        #    r2 = r2[0]
        res = []
        for i in range(len(vectors)):
            self._thc_scheduler.update_r1(r1[i])
            self._thc_scheduler.update_r2(r2[i])
            hr1 = self._thc_scheduler._evaluate(einsum_holder.THC_scheduler.ip_hr1_r_name)
            hr2 = self._thc_scheduler._evaluate(einsum_holder.THC_scheduler.ip_hr2_r_name)
        #if self._nroots == 1:
        #    return [self.amplitudes_to_vector(hr1, hr2)]
        #else:
            #res = []
            #for i in range(self._nroots):
            res.append(self.amplitudes_to_vector(hr1, hr2))
        return res

    def gen_matvec(self, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(imds)
        
        #if self._nroots == 1:
        r1_zeros = np.zeros(self.nocc)
        r2_zeros = np.zeros((self.nocc, self.nocc, self.nvir))
        #else:
        #    r1_zeros = np.zeros((self._nroots, self.nocc))
        #    r2_zeros = np.zeros((self._nroots, self.nocc, self.nocc, self.nvir))
        if self._use_torch:
            r1_zeros = to_torch(r1_zeros, self._with_gpu)
            r2_zeros = to_torch(r2_zeros, self._with_gpu)
            
        if left:
            if not self._build_l_matvec:
                self._build_l_matvec = True
                _thc_eom_rccsd_ind.lipccsd_matvec(self, imds, False, diag, self._thc_scheduler)
                self._thc_scheduler._build_expression()
                self._thc_scheduler.update_r1(r1_zeros)
                self._thc_scheduler.update_r2(r2_zeros)
                self._thc_scheduler._build_contraction(self._backend)
                self._thc_scheduler.evaluate_all_intermediates()
            #matvec = lambda xs: [self.l_matvec(x, imds, diag) for x in xs]
            matvec = self._l_matvec
        else:
            if not self._build_r_matvec:
                self._build_r_matvec = True
                _thc_eom_rccsd_ind.ipccsd_matvec(self, imds, False, diag, self._thc_scheduler)
                self._thc_scheduler._build_expression()
                self._thc_scheduler.update_r1(r1_zeros)
                self._thc_scheduler.update_r2(r2_zeros)
                self._thc_scheduler._build_contraction(self._backend)
                self._thc_scheduler.evaluate_all_intermediates()
            #matvec = lambda xs: [self.matvec(x, imds, diag) for x in xs]
            matvec = self._r_matvec
        
        del r1_zeros
        del r2_zeros
        r1_zeros = None
        r2_zeros = None
        
        return matvec, diag

########################################
# THC-EOM-EA-CCSD
########################################

class THC_EOM_EA_RCCSD(eom_rccsd.EOM):
    
    def __init__(self, 
                 cc, 
                 **kwargs):
        
        #### initialization ####
        
        eom_rccsd.EOM.__init__(self, cc)
        
        my_mf = cc._scf
        my_isdf = my_mf.with_df
                
        self._backend   = self._cc._backend
        self._memory    = self._cc._memory
        self._use_torch = self._cc._use_torch
        self._with_gpu  = self._cc._with_gpu
        self.nthc = self._cc.X_o.shape[1]
        self.eris = self._cc._thc_eris
        self._t2_with_denominator = self._cc._t2_with_denominator
        self._imds = None
        self._imds = self.make_imds()
        
        self.nocc = self._cc.nocc
        self.nmo = self._cc.nmo
        self.nvir = self.nmo - self.nocc
        
        ## NOTE: all the THC information is contained in self._cc
        
        #### init t1 and t2 #### 
        
        self.t1, self.t2 = self._cc.t1, self._cc.t2
        self._t1_expr, self._t2_expr = self._cc._t1_expr, self._cc._t2_expr
    
        #### init scheduler #### 

        self._thc_scheduler = einsum_holder.THC_scheduler(
            X_O=self._cc.X_o, X_V=self._cc.X_v, THC_INT=self._cc.Z,
            T1=self._cc.t1, 
            XO_T2=self._cc.X_o, XV_T2=self._cc.X_v, THC_T2=self._cc.t2,
            TAU_O=self._cc.tau_o, TAU_V=self._cc.tau_v,
            grid_partition=self._cc.grid_partition, ## possible None
            proj_type=self._cc.proj_type,
            use_torch=self._cc._use_torch,
            with_gpu=self._cc._with_gpu,
            **kwargs
        )
    
        ### build exprs ###
        
        self._build_r_matvec = False
        self._build_l_matvec = False
    
    def make_imds(self, eris=None):
        if hasattr(self, '_imds'):
            if self._imds is not None:
                return self._imds
        if eris is None:
            eris = self.eris
        imds = _thc_eom_rccsd_ind._IMDS_symbolic(self._cc, eris=eris, MRPT2=self._cc.mbpt2)
        imds.make_ea(self.partition)
        self._imds = imds
        return imds
    
    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nvir + nocc*nvir*nvir
    
    @property
    def eea(self):
        return self.e
    
    ######### identical to the original EOM-EA #########

    spatial2spin = staticmethod(eom_rccsd.spatial2spin_ea)
    
    @staticmethod
    def amplitudes_to_vector(r1, r2):
        r1 = to_numpy_cpu(r1)
        r2 = to_numpy_cpu(r2)
        vector = np.hstack((r1, r2.ravel()))
        return vector
    
    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None:
            nmo = self.nmo
        if nocc is None:
            nocc = self.nocc
        nvir = nmo - nocc
        r1 = vector[:nvir].copy()
        r2 = vector[nvir:].copy().reshape(nocc,nvir,nvir)
        #if self._use_torch:
        #    r1 = to_torch(r1, self._with_gpu)
        #    r2 = to_torch(r2, self._with_gpu)
        return r1, r2
    
    ######### slightly different to the original EOM #########
    
    def kernel(self, nroots=1, left=False, koopmans=False, guess=None,
           partition=None, eris=None, imds=None):
        self._nroots = nroots  ## used to build exprs!
        return eom_rccsd.eaccsd(self, nroots, left, koopmans, guess, partition, eris, imds)
    
    eaccsd = kernel
    
    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        res = eom_rccsd.EOMEA.get_init_guess(self, nroots, koopmans, diag)
        #self._nroots = nroots  ### used in generate expression
        if hasattr(self, '_nroots'):
            assert self._nroots == nroots
        else:
            self._nroots = nroots
        return res
    
    ######### not identical to the original EOM #########
    
    def get_diag(self, imds=None):
        
        if hasattr(self, '_diag'):
            if self._diag is not None:
                return self._diag
        
        if imds is None: imds = self.make_imds()
        t1, t2 = self.t1, self.t2
        t1 = to_numpy_cpu(t1)
        t2 = to_numpy_cpu(t2)
        dtype = np.result_type(t1, t2)
        nocc, nvir = t1.shape
        fock = imds.eris.fock
        foo = fock[:nocc,:nocc].copy()
        fvv = fock[nocc:,nocc:].copy()
        
        Hr1 = np.zeros((nvir), dtype)
        Hr2 = np.zeros((nocc,nvir,nvir), dtype)
        
        if self._use_torch:
            Hr1 = to_torch(Hr1, self._with_gpu)
            Hr2 = to_torch(Hr2, self._with_gpu)
        
        ############# build local scheduler and local exprs #############
        
        local_scheduler = einsum_holder.THC_scheduler(
            X_O=self._cc.X_o, X_V=self._cc.X_v, THC_INT=self._cc.Z,
            T1=self._cc.t1, 
            XO_T2=self._cc.X_o, XV_T2=self._cc.X_v, THC_T2=self._cc.t2,
            TAU_O=self._cc.tau_o, TAU_V=self._cc.tau_v,
            grid_partition=self._cc.grid_partition, ## possible None
            proj_type=self._cc.proj_type,
            use_torch=self._cc._use_torch,
            with_gpu=self._cc._with_gpu
        )

        ###### register foo, fvv ######
        fock = imds.eris.fock
        foo = fock[:nocc,:nocc].copy()
        fvv = fock[nocc:,nocc:].copy()
        fov = fock[:nocc,nocc:].copy()
        local_scheduler.add_input("foo", foo)
        local_scheduler.add_input("fvv", fvv)
        local_scheduler.add_input("fov", fov)
        foo = einsum_holder.to_expr_holder(einsum_holder._expr_foo())
        fvv = einsum_holder.to_expr_holder(einsum_holder._expr_fvv())
        fov = einsum_holder.to_expr_holder(einsum_holder._expr_fov())
        ###############################

        local_scheduler.register_expr("LVV", imds.Lvv)
        if self.partition != 'mp':
            local_scheduler.register_expr("LOO", imds.Loo)            
            
            Wvvvv_ab = einsum_holder.thc_einsum_sybolic("abab->ab", imds.Wvvvv)
            Wovvo_jb = einsum_holder.thc_einsum_sybolic("jbbj->jb", imds.Wovvo)
            Wovov_jb = einsum_holder.thc_einsum_sybolic("jbjb->jb", imds.Wovov)
            Hr2_0    = einsum_holder.thc_einsum_sybolic("ijab,ijab->jab", imds.Woovv, self._t2_expr)
            Hr2_1    = einsum_holder.thc_einsum_sybolic("ijba,ijab->jab", imds.Woovv, self._t2_expr)

            local_scheduler.register_expr("Wvvvv_ab", Wvvvv_ab)
            local_scheduler.register_expr("Wovvo_jb", Wovvo_jb)
            local_scheduler.register_expr("Wovov_jb", Wovov_jb)
            local_scheduler.register_expr("Hr2_0", Hr2_0)
            local_scheduler.register_expr("Hr2_1", Hr2_1)
            
        local_scheduler._build_expression()
        local_scheduler._build_contraction(backend=self._backend)
        
        #################################################################
        
        ############# perform calculation #############
        
        local_scheduler.evaluate_all()
        
        Lvv = local_scheduler.get_tensor("LVV")
        
        ######## Hr1 ########
        
        if self._use_torch:  ### USE: autoray's DO function
            Lvv = Lvv.cpu().numpy()
            Hr1 = Hr1.cpu().numpy()
            Hr1 += np.diag(Lvv)
            Hr1 = to_torch(Hr1, self._with_gpu)
        else:
            Hr1 += np.diag(Lvv)
        
        ######## Hr2 ######## 
        
        if self.partition == 'mp':
            for a in range(nvir):
                for b in range(nvir):
                    for j in range(nocc):
                        Hr2[j,a,b] = fvv[a,a] + fvv[b,b] - foo[j,j]
        else:
            
            Loo      = local_scheduler.get_tensor("LOO")
            Wvvvv_ab = local_scheduler.get_tensor("Wvvvv_ab")
            Wovvo_jb = local_scheduler.get_tensor("Wovvo_jb")
            Wovov_jb = local_scheduler.get_tensor("Wovov_jb")
            Hr2_0    = local_scheduler.get_tensor("Hr2_0")
            Hr2_1    = local_scheduler.get_tensor("Hr2_1")

            Hr2 += (-2*Hr2_0+Hr2_1)
            for b in range(nvir):
                Hr2[:,:,b] += Lvv[b,b]
                Hr2[:,b,:] += Lvv[b,b]
            for i in range(nocc):
                Hr2[i,:,:] -= Loo[i,i]
                Hr2[i,:,:] += Wvvvv_ab
            for a in range(nvir):
                Hr2[:,a,:] += 2*Wovvo_jb
                Hr2[:,a,:] -= Wovov_jb
                Hr2[:,:,a] -= Wovov_jb
                Hr2[:,a,a] -= Wovvo_jb[:,a]
        
        local_scheduler = None
        
        ###############################################
        
        vector = self.amplitudes_to_vector(Hr1, Hr2)
        
        self._diag = vector
        
        return vector

    def _l_matvec(self, vectors):
        r1 = []
        r2 = []
        #for i in range(self._nroots):
        for i in range(len(vectors)):
            r1_, r2_ = self.vector_to_amplitudes(vectors[i], self.nmo, self.nocc)
            r1.append(r1_)
            r2.append(r2_)
        if self._use_torch:
            r1 = [to_torch(r1[i], self._with_gpu) for i in range(len(vectors))]
            r2 = [to_torch(r2[i], self._with_gpu) for i in range(len(vectors))]
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
        else:
            r1 = np.array(r1)
            r2 = np.array(r2)
        #if self._nroots == 1:
        #    r1 = r1[0]
        #    r2 = r2[0]
        res = []
        for i in range(len(vectors)):
            self._thc_scheduler.update_r1(r1[i])
            self._thc_scheduler.update_r2(r2[i])
            hr1 = self._thc_scheduler._evaluate(einsum_holder.THC_scheduler.ea_hr1_l_name)
            hr2 = self._thc_scheduler._evaluate(einsum_holder.THC_scheduler.ea_hr2_l_name)
        #if self._nroots == 1:
        #    return [self.amplitudes_to_vector(hr1, hr2)]
        #else:
            #res = []
            #for i in range(self._nroots):
            res.append(self.amplitudes_to_vector(hr1, hr2))
        return res
    
    def _r_matvec(self, vectors, imds=None, diag=None):
        r1 = []
        r2 = []
        #for i in range(self._nroots):
        for i in range(len(vectors)):
            r1_, r2_ = self.vector_to_amplitudes(vectors[i], self.nmo, self.nocc)
            r1.append(r1_)
            r2.append(r2_)
        if self._use_torch:
            r1 = [to_torch(r1[i], self._with_gpu) for i in range(len(vectors))]
            r2 = [to_torch(r2[i], self._with_gpu) for i in range(len(vectors))]
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
        else:
            r1 = np.array(r1)
            r2 = np.array(r2)
        #if self._nroots == 1:
        #    r1 = r1[0]
        #    r2 = r2[0]
        res = []
        for i in range(len(vectors)):
            self._thc_scheduler.update_r1(r1[i])
            self._thc_scheduler.update_r2(r2[i])
            hr1 = self._thc_scheduler._evaluate(einsum_holder.THC_scheduler.ea_hr1_r_name)
            hr2 = self._thc_scheduler._evaluate(einsum_holder.THC_scheduler.ea_hr2_r_name)
        #if self._nroots == 1:
        #    return [self.amplitudes_to_vector(hr1, hr2)]
        #else:
            #res = []
            #for i in range(self._nroots):
            res.append(self.amplitudes_to_vector(hr1, hr2))
        return res

    def gen_matvec(self, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(imds)
        
        #if self._nroots == 1:
        r1_zeros = np.zeros(self.nvir)
        r2_zeros = np.zeros((self.nocc, self.nvir, self.nvir))
        #else:
        #    r1_zeros = np.zeros((self._nroots, self.nocc))
        #    r2_zeros = np.zeros((self._nroots, self.nocc, self.nocc, self.nvir))
        if self._use_torch:
            r1_zeros = to_torch(r1_zeros, self._with_gpu)
            r2_zeros = to_torch(r2_zeros, self._with_gpu)
            
        if left:
            if not self._build_l_matvec:
                self._build_l_matvec = True
                _thc_eom_rccsd_ind.leaccsd_matvec(self, imds, False, diag, self._thc_scheduler)
                self._thc_scheduler._build_expression()
                self._thc_scheduler.update_r1(r1_zeros)
                self._thc_scheduler.update_r2(r2_zeros)
                self._thc_scheduler._build_contraction(self._backend)
                self._thc_scheduler.evaluate_all_intermediates()
            #matvec = lambda xs: [self.l_matvec(x, imds, diag) for x in xs]
            matvec = self._l_matvec
        else:
            if not self._build_r_matvec:
                self._build_r_matvec = True
                _thc_eom_rccsd_ind.eaccsd_matvec(self, imds, False, diag, self._thc_scheduler)
                self._thc_scheduler._build_expression()
                self._thc_scheduler.update_r1(r1_zeros)
                self._thc_scheduler.update_r2(r2_zeros)
                self._thc_scheduler._build_contraction(self._backend)
                self._thc_scheduler.evaluate_all_intermediates()
            #matvec = lambda xs: [self.matvec(x, imds, diag) for x in xs]
            matvec = self._r_matvec
        
        del r1_zeros
        del r2_zeros
        r1_zeros = None
        r2_zeros = None
        
        return matvec, diag


if __name__ == "__main__":
    
    c = 15
    N = 1
    
    cell   = pbcgto.Cell()
    boxlen = 3.5668
    cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]]) 
    
    cell.atom = [
                ['C', (0.     , 0.     , 0.    )],
                ['C', (0.8917 , 0.8917 , 0.8917)],
                ['C', (1.7834 , 1.7834 , 0.    )],
                ['C', (2.6751 , 2.6751 , 0.8917)],
                ['C', (1.7834 , 0.     , 1.7834)],
                ['C', (2.6751 , 0.8917 , 2.6751)],
                ['C', (0.     , 1.7834 , 1.7834)],
                ['C', (0.8917 , 2.6751 , 2.6751)],
            ] 

    cell.basis   = 'gth-szv'
    #cell.basis   = 'gth-dzvp'
    cell.pseudo  = 'gth-pade'
    cell.verbose = 10
    cell.ke_cutoff = 70
    cell.max_memory = 800  # 800 Mb
    cell.precision  = 1e-8  # integral precision
    cell.use_particle_mesh_ewald = True 
    
    verbose = 10
    
    prim_cell = build_supercell(cell.atom, cell.a, Ls = [1,1,1], ke_cutoff=cell.ke_cutoff, basis=cell.basis, pseudo=cell.pseudo, verbose=verbose)   
    prim_partition = [[0,1,2,3], [4,5,6,7]]
    prim_mesh = prim_cell.mesh
    
    Ls = [1, 1, N]
    Ls = np.array(Ls, dtype=np.int32)
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    cell, group_partition = build_supercell_with_partition(
                                    cell.atom, cell.a, mesh=mesh, 
                                    Ls=Ls,
                                    basis=cell.basis, 
                                    pseudo=cell.pseudo,
                                    partition=prim_partition, ke_cutoff=cell.ke_cutoff, verbose=verbose) 
    cell.charge = 2
    cell.build()
        
    import numpy
    from pyscf.pbc import gto, scf, mp    
    
    myisdf = ISDF.PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, aoR_cutoff=1e-8, direct=False, use_occ_RI_K=False)
    myisdf.build_IP_local(c=c, m=5, group=group_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
    myisdf.build_auxiliary_Coulomb(debug=True)
            
    mf_isdf           = scf.RHF(cell)
    myisdf.direct_scf = mf_isdf.direct_scf
    mf_isdf.with_df   = myisdf
    mf_isdf.max_cycle = 8
    mf_isdf.conv_tol  = 1e-8
    mf_isdf.kernel()

    ########## TEST EOM-EA/IP diag ##########

    NROOTS = 1
    MBPT2  = False
    CC2    = True

    from pyscf.pbc.df.isdf.isdf_posthf import RCCSD_isdf
    
    mycc_isdf, eris_ccsd = RCCSD_isdf(mf_isdf, run=False, cc2=CC2)
    mycc_isdf.kernel(eris=eris_ccsd, mbpt2=MBPT2)
    
    from pyscf.cc import eom_rccsd
    
    eom_IP = eom_rccsd.EOMIP(mycc_isdf)
    imds = eom_IP.make_imds(eris_ccsd)
    IP_diag = eom_IP.get_diag(imds)
    
    eom_EA = eom_rccsd.EOMEA(mycc_isdf)
    imds = eom_EA.make_imds(eris_ccsd)
    EA_diag = eom_EA.get_diag(imds)
    
    ########## THC-EOM-EA/IP diag ##########
    
    from pyscf.pbc.df.isdf.thc_rccsd import THC_RCCSD
    
    X, partition = myisdf.aoRg_full()
    thc_ccsd = THC_RCCSD(my_mf=mf_isdf, X=X, partition=partition, memory=2**31, 
                         backend="opt_einsum", 
                         use_torch=True, with_gpu=True, 
                         projector_t="thc", cc2=CC2, t2_with_denominator=False)
    thc_ccsd.ccsd(mbpt2=MBPT2)

    thc_eom_ip = THC_EOM_IP_RCCSD(thc_ccsd)
    IP_diag_test = thc_eom_ip.get_diag()

    thc_eom_ea = THC_EOM_EA_RCCSD(thc_ccsd)
    EA_diag_test = thc_eom_ea.get_diag()
    
    print(IP_diag_test)
    print(IP_diag)
    print(EA_diag_test)
    print(EA_diag)
    
    IP_diag_test = to_numpy_cpu(IP_diag_test)
    IP_diag = to_numpy_cpu(IP_diag)
    EA_diag_test = to_numpy_cpu(EA_diag_test)
    EA_diag = to_numpy_cpu(EA_diag)
    
    diff_IP = np.max(np.abs(IP_diag_test - IP_diag))
    diff_EA = np.max(np.abs(EA_diag_test - EA_diag))
    diff_mean_IP = np.mean(np.abs(IP_diag_test - IP_diag))
    diff_mean_EA = np.mean(np.abs(EA_diag_test - EA_diag))
    diff_var_IP = np.var(np.abs(IP_diag_test - IP_diag))
    diff_var_EA = np.var(np.abs(EA_diag_test - EA_diag))
    print(diff_IP, diff_mean_IP, diff_var_IP)
    print(diff_EA, diff_mean_EA, diff_var_EA)
    
    thc_eom_ip.kernel(nroots=NROOTS)
    eom_IP.kernel(nroots=NROOTS, eris=eris_ccsd)
    
    thc_eom_ea.kernel(nroots=NROOTS)
    eom_EA.kernel(nroots=NROOTS, eris=eris_ccsd)