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

class THC_EOM_RCCSD(eom_rccsd.EOM):
    
    def __init__(self, 
                 cc, 
                 **kwargs):
        
        #### initialization ####
        
        eom_rccsd.EOM.__init__(self, cc)
        
        my_mf = cc._scf
        my_isdf = my_mf.with_df
                
        self._backend = backend
        self._memory  = memory
        self.nthc = self.X_o.shape[1]
        self.eris = self.cc._thc_eris
        self._t2_with_denominator = self.cc._t2_with_denominator
        self._imds = None
        self._imds = self.make_imds()
        
        ## NOTE: all the THC information is contained in self.cc
        
        #### init t1 and t2 #### 
        
        self.t1, self.t2 = self.cc.t1, self.cc.t2
        self._t1_expr, self._t2_expr = self.cc._t1_expr, self.cc._t2_expr
    
        #### init scheduler #### 

        self._thc_scheduler = einsum_holder.THC_scheduler(
            X_O=self.cc.X_o, X_V=self.cc.X_v, THC_INT=self.cc.Z,
            T1=self.cc.t1, 
            XO_T2=self.cc.X_o, XV_T2=self.cc.X_v, THC_T2=self.cc.t2,
            TAU_O=self.cc.tau_o, TAU_V=self.cc.tau_v,
            grid_partition=self.cc.grid_partition, ## possible None
            proj_type=self.cc.proj_type,
            use_torch=self.cc._use_torch,
            with_gpu=self.cc._with_gpu,
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
        imds.make_ip(self.partition)
        return imds
    
    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nocc + nocc*nocc*nvir
    
    @property
    def eip(self):
        return self.e
    
    ######### identical to the original EOM #########

    spatial2spin = staticmethod(eom_rccsd.spatial2spin_ip)
    
    @staticmethod
    def amplitudes_to_vector(r1, r2):
        r1 = to_numpy_cpu(r1)
        r2 = to_numpy_cpu(r2)
        vector = np.hstack((r1, r2.ravel()))
        return vector
    
    def vector_to_amplitudes(self, vector, nmo, nocc):
        if nmo is None:
            nmo = self.nmo
        if nocc is None:
            nocc = self.nocc
        nvir = nmo - nocc
        r1 = vector[:nocc].copy()
        r2 = vector[nocc:].copy().reshape(nocc,nocc,nvir)
        if self._use_torch:
            r1 = to_torch(r1, self._with_gpu)
            r2 = to_torch(r2, self._with_gpu)
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
    
    ######### not identical to the original EOM #########
    
    def get_diag(self, imds=None):
        
        if hasattr(self, '_diag'):
            if self._diag is not None:
                return self._diag
        
        if imds is None: imds = self.make_imds()
        t1, t2 = self.t1, self.t2
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
            X_O=self.cc.X_o, X_V=self.cc.X_v, THC_INT=self.cc.Z,
            T1=self.cc.t1, 
            XO_T2=self.cc.X_o, XV_T2=self.cc.X_v, THC_T2=self.cc.t2,
            TAU_O=self.cc.tau_o, TAU_V=self.cc.tau_v,
            grid_partition=self.cc.grid_partition, ## possible None
            proj_type=self.cc.proj_type,
            use_torch=self.cc._use_torch,
            with_gpu=self.cc._with_gpu
        )

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
        local_scheduler._build_contraction()
        
        #################################################################
        
        ############# perform calculation #############
        
        local_scheduler.evaluate_all()
        
        Loo = local_scheduler.get_tensor("LOO")
        
        ######## Hr1 ########
        
        if use_torch:  ### USE: autoray's DO function
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

            Hr2 += (-2*Hr2_0+Hr2_1)
            for b in range(nvir):
                Hr2[:,:,b] += Lvv[b,b]
            for i in range(nocc):
                Hr2[i,:,:] -= Loo[i,i]
                Hr2[:,i,:] -= Loo[i,i]
                Hr2[i,i,:] -= Wovvo_jb[i,:]
            for i in range(nocc):
                for j in range(nocc):
                    Hr2[i,j,:] += Woooo_ij[i,j]
            for j in range(nocc):
                for b in range(nvir):
                    Hr2[:,j,b] += 2*Wovvo_jb[j,b]
                    Hr2[j,:,b] -= Wovov_jb[j,b]
                    Hr2[:,j,b] -= Wovov_jb[j,b]
        
        local_scheduler = None
        
        ###############################################
        
        vector = self.amplitudes_to_vector(Hr1, Hr2)
        
        self._diag = vector
        
        return vector

    def _l_matvec(self, vectors):
        r1 = []
        r2 = []
        for i in range(self._nroots):
            r1_, r2_ = self.vector_to_amplitudes(vector[i])
            r1.append(r1_)
            r2.append(r2_)
        if self._use_torch:
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
        else:
            r1 = np.array(r1)
            r2 = np.array(r2)
        if self._nroots == 1:
            r1 = r1[0]
            r2 = r2[0]
        self._thc_scheduler.update_r1(r1)
        self._thc_scheduler.update_r2(r2)
        hr1 = self._thc_scheduler._evaluate(einsum_holder.THC_scheduler.ip_hr1_l_name)
        hr2 = self._thc_scheduler._evaluate(einsum_holder.THC_scheduler.ip_hr2_l_name)
        if self._nroots == 1:
            return [self.amplitudes_to_vector(hr1, hr2)]
        else:
            res = []
            for i in range(self._nroots):
                res.append(self.amplitudes_to_vector(hr1[i], hr2[i]))
            return res
    
    def _r_matvec(self, vectors, imds=None, diag=None):
        r1 = []
        r2 = []
        for i in range(self._nroots):
            r1_, r2_ = self.vector_to_amplitudes(vector[i])
            r1.append(r1_)
            r2.append(r2_)
        if self._use_torch:
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
        else:
            r1 = np.array(r1)
            r2 = np.array(r2)
        if self._nroots == 1:
            r1 = r1[0]
            r2 = r2[0]
        self._thc_scheduler.update_r1(r1)
        self._thc_scheduler.update_r2(r2)
        hr1 = self._thc_scheduler._evaluate(einsum_holder.THC_scheduler.ip_hr1_r_name)
        hr2 = self._thc_scheduler._evaluate(einsum_holder.THC_scheduler.ip_hr2_r_name)
        if self._nroots == 1:
            return [self.amplitudes_to_vector(hr1, hr2)]
        else:
            res = []
            for i in range(self._nroots):
                res.append(self.amplitudes_to_vector(hr1[i], hr2[i]))
            return res

    def gen_matvec(self, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(imds)
        
        if self._nroots == 1:
            r1_zeros = np.zeros(self.nocc)
            r2_zeros = np.zeros((self.nocc, self.nocc, self.nvir))
        else:
            r1_zeros = np.zeros((self._nroots, self.nocc))
            r2_zeros = np.zeros((self._nroots, self.nocc, self.nocc, self.nvir))
        if self._use_torch:
            r1_zeros = to_torch(r1_zeros, self._with_gpu)
            r2_zeros = to_torch(r2_zeros, self._with_gpu)
            
        if left:
            if not self._build_l_matvec:
                self._build_l_matvec = True
                _thc_eom_rccsd_ind.lipccsd_matvec(self, imds, diag, self._thc_scheduler)
                self._thc_scheduler._build_expression()
                self._thc_scheduler.update_r1(r1_zeros)
                self._thc_scheduler.update_r2(r2_zeros)
                self._thc_scheduler._build_contraction()
                self.evaluate_all_intermediates()
            #matvec = lambda xs: [self.l_matvec(x, imds, diag) for x in xs]
            matvec = self._l_matvec
        else:
            if not self._build_r_matvec:
                self._build_r_matvec = True
                _thc_eom_rccsd_ind.ipccsd_matvec(self, imds, diag, self._thc_scheduler)
                self._thc_scheduler._build_expression()
                self._thc_scheduler.update_r1(r1_zeros)
                self._thc_scheduler.update_r2(r2_zeros)
                self._thc_scheduler._build_contraction()
                self.evaluate_all_intermediates()
            #matvec = lambda xs: [self.matvec(x, imds, diag) for x in xs]
            matvec = self._r_matvec
        
        del r1_zeros
        del r2_zeros
        r1_zeros = None
        r2_zeros = None
        
        return matvec, diag