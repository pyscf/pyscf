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

from pyscf.pbc.df.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast
from pyscf.pbc.df.isdf.thc_einsum     import thc_einsum, energy_denomimator, thc_holder
# from pyscf.mp.mp2                     import MP2
from pyscf.pbc.df.isdf.thc_rmp2 import THC_RMP2

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)

from pyscf.pbc.df.isdf.thc_backend import *

class THC_RMP3(THC_RMP2):
    
    def __init__(self, 
                 #my_isdf, 
                 my_mf, 
                 frozen=None, mo_coeff=None, mo_occ=None,
                 X=None, ## used in XXZXX
                 laplace_rela_err = 1e-7,
                 laplace_order    = 2,
                 no_LS_THC        = False,
                 use_torch=False,
                 with_gpu =False):
        
        super(THC_RMP3, self).__init__(
            #my_isdf, 
                                       my_mf, 
                                       frozen=frozen, 
                                       mo_coeff=mo_coeff, 
                                       mo_occ=mo_occ, 
                                       X=X, 
                                       laplace_rela_err=laplace_rela_err, 
                                       laplace_order=laplace_order, 
                                       no_LS_THC=no_LS_THC,
                                       use_torch=use_torch,
                                       with_gpu=with_gpu)
    
    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, backend="opt_einsum", memory=2**28, return_path_only=False):
        '''
        Args:
            with_t2 : bool
                Whether to generate and hold t2 amplitudes in memory.
        '''
        
        ### build the THC ERI tensor and laplace tensor used in thc_einsum ###
        
        nocc    = np.sum(self.mo_occ > 0)
        print("nocc", nocc)
        THC_ERI = thc_holder(self.X_o, self.X_v, self.Z)
        LAPLACE = energy_denomimator(self.tau_o, self.tau_v)
        
        #if use_cotengra:
        #    backend = 'cotengra'
        #else:
        #    backend = None
        
        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        mp3_CC = thc_einsum("iajb,jbkc,iakc,ijab,ikac->", THC_ERI, THC_ERI, THC_ERI, LAPLACE, LAPLACE, backend=backend, memory=memory, return_path_only=return_path_only)
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, 'THC_RMP3: mp3-CC ', self._scf)
        
        mp3_CX1 = thc_einsum("iajb,jkbc,iakc,ijab,ikac->", THC_ERI, THC_ERI, THC_ERI, LAPLACE, LAPLACE, backend=backend, memory=memory, return_path_only=return_path_only)
        mp3_CX2 = thc_einsum("ibja,jbkc,iakc,ijab,ikac->", THC_ERI, THC_ERI, THC_ERI, LAPLACE, LAPLACE, backend=backend, memory=memory, return_path_only=return_path_only)
        mp3_CX3 = thc_einsum("iajb,jbkc,kaic,ijab,ikac->", THC_ERI, THC_ERI, THC_ERI, LAPLACE, LAPLACE, backend=backend, memory=memory, return_path_only=return_path_only)
        t3 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t2, t3, 'THC_RMP3: mp3-CX ', self._scf)
        
        mp3_XX1 = thc_einsum("ibja,jkbc,iakc,ijab,ikac->", THC_ERI, THC_ERI, THC_ERI, LAPLACE, LAPLACE, backend=backend, memory=memory, return_path_only=return_path_only)
        mp3_XX2 = thc_einsum("ibja,jbkc,kaic,ijab,ikac->", THC_ERI, THC_ERI, THC_ERI, LAPLACE, LAPLACE, backend=backend, memory=memory, return_path_only=return_path_only)
        mp3_XX3 = thc_einsum("iajb,acbd,icjd,ijab,ijcd->", THC_ERI, THC_ERI, THC_ERI, LAPLACE, LAPLACE, backend=backend, memory=memory, return_path_only=return_path_only)
        mp3_XX4 = thc_einsum("ibja,acbd,icjd,ijab,ijcd->", THC_ERI, THC_ERI, THC_ERI, LAPLACE, LAPLACE, backend=backend, memory=memory, return_path_only=return_path_only)
        mp3_XX5 = thc_einsum("iajb,ikjl,kalb,ijab,klab->", THC_ERI, THC_ERI, THC_ERI, LAPLACE, LAPLACE, backend=backend, memory=memory, return_path_only=return_path_only)
        mp3_XX6 = thc_einsum("ibja,ikjl,kalb,ijab,klab->", THC_ERI, THC_ERI, THC_ERI, LAPLACE, LAPLACE, backend=backend, memory=memory, return_path_only=return_path_only)
        mp3_XX7 = thc_einsum("iajb,jkac,kbic,ijab,ikbc->", THC_ERI, THC_ERI, THC_ERI, LAPLACE, LAPLACE, backend=backend, memory=memory, return_path_only=return_path_only)
        mp3_XX8 = thc_einsum("ibja,jkac,kbic,ijab,ikbc->", THC_ERI, THC_ERI, THC_ERI, LAPLACE, LAPLACE, backend=backend, memory=memory, return_path_only=return_path_only)
        t4 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t3, t4, 'THC_RMP3: mp3-XX ', self._scf)
        
        if return_path_only:
            
            self.e_corr = None
            
            if backend == "opt_einsum":
                
                print("RMP3_CC,  path = ")
                print(mp3_CC[1])
                
                print("RMP3_CX1, path = ")
                print(mp3_CX1[1])
                print("RMP3_CX2, path = ")
                print(mp3_CX2[1])
                print("RMP3_CX3, path = ")
                print(mp3_CX3[1])
                
                print("RMP3_XX1, path = ")
                print(mp3_XX1[1])
                print("RMP3_XX2, path = ")
                print(mp3_XX2[1])
                print("RMP3_XX3, path = ")
                print(mp3_XX3[1])
                print("RMP3_XX4, path = ")
                print(mp3_XX4[1])
                
                print("RMP3_XX5, path = ")
                print(mp3_XX5[1])
                print("RMP3_XX6, path = ")
                print(mp3_XX6[1])
                print("RMP3_XX7, path = ")
                print(mp3_XX7[1])
                print("RMP3_XX8, path = ")
                print(mp3_XX8[1])
            
            return [mp3_CC], [mp3_CX1, mp3_CX2, mp3_CX3], [mp3_XX1, mp3_XX2, mp3_XX3, mp3_XX4, mp3_XX5, mp3_XX6, mp3_XX7, mp3_XX8]
        else:
            self.e_corr  = 8 * mp3_CC - 4 * (mp3_CX1 + mp3_CX2 + mp3_CX3)
            self.e_corr += 2 * (mp3_XX1 + mp3_XX2 + mp3_XX3) - mp3_XX4 + 2 * mp3_XX5 - mp3_XX6 - 4 * mp3_XX7 + 2 * mp3_XX8  
        
        #if self._use_torch:
        #    self.e_corr = self.e_corr.cpu().detach().item()
        self.e_corr = to_scalar(self.e_corr)
            
        return self.e_corr, None
            


if __name__ == "__main__":
    
    c = 15 
    N = 2
    
    #if rank == 0:
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
    cell.pseudo  = 'gth-pade'
    cell.verbose = 10
    cell.ke_cutoff = 128
    cell.max_memory = 800  # 800 Mb
    cell.precision  = 1e-8  # integral precision
    cell.use_particle_mesh_ewald = True 
    
    verbose = 10
    
    prim_cell = build_supercell(cell.atom, cell.a, Ls = [1,1,1], ke_cutoff=cell.ke_cutoff, basis=cell.basis, pseudo=cell.pseudo, verbose=verbose)   
    prim_partition = [[0,1,2,3], [4,5,6,7]]
    #prim_partition=  [[0,1]]
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
    
    ####### isdf MP2 can perform directly! ####### 
    
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
     
    isdf_pt = mp.RMP2(mf_isdf)
    isdf_pt.kernel()
    
    ####### thc rmp3 #######
    
    X,_        = myisdf.aoRg_full()
    thc_rmp3 = THC_RMP3(mf_isdf, X=X, use_torch=True, with_gpu=True)
    e_mp3, _ = thc_rmp3.kernel(backend="cotengra")
    print("ISDF MP3 energy", e_mp3)
    e_mp3, _ = thc_rmp3.kernel(backend="opt_einsum")
    print("ISDF MP3 energy", e_mp3)
    
    