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
from pyscf.pbc.df.isdf._thc._RMP2 import *
from pyscf.pbc.df.isdf._thc._RMP2_forloop import * 

SCHEDULE_TYPE_NAIVE   = 0
SCHEDULE_TYPE_C       = 1
SCHEDULE_TYPE_OPT_MEM = 2
SCHEDULE_TYPE_FORLOOP = 3


class THC_RMP2(_restricted_THC_posthf_holder):
    
    def __init__(self, my_isdf, my_mf, X,
                 laplace_rela_err = 1e-7,
                 laplace_order    = 2,
                 memory           = 128 * 1000 * 1000):
        
        super().__init__(my_isdf, my_mf, X,
                            laplace_rela_err = laplace_rela_err,
                            laplace_order    = laplace_order)
        
        self.memory = memory
        self.buffer = None
    
    #### kernels for THC-RMP2 #### 
    
    def _kernel_naive(self):
        
        t1 = (logger.process_clock(), logger.perf_counter())
        mp2_J = RMP2_J_naive(self.Z,
                             self.X_o,
                             self.X_v,
                             self.tau_o,
                             self.tau_v)
        t2 = (logger.process_clock(), logger.perf_counter())
        _benchmark_time(t1, t2, "RMP2_J_naive")
        mp2_K = RMP2_K_naive(self.Z,
                             self.X_o,
                             self.X_v,
                             self.tau_o,
                             self.tau_v)
        t3 = (logger.process_clock(), logger.perf_counter())
        _benchmark_time(t2, t3, "RMP2_K_naive")
        print("E_corr(RMP2) = " + str(-2*mp2_J + mp2_K))
        return -2*mp2_J + mp2_K

    def _kernel_C(self):
        
        t1 = (logger.process_clock(), logger.perf_counter())
        mp2_J = RMP2_J(self.Z,
                       self.X_o,
                       self.X_v,
                       self.tau_o,
                       self.tau_v)
        t2 = (logger.process_clock(), logger.perf_counter())
        _benchmark_time(t1, t2, "RMP2_J")
        mp2_K = RMP2_K(self.Z,
                       self.X_o,
                       self.X_v,
                       self.tau_o,
                       self.tau_v)
        t3 = (logger.process_clock(), logger.perf_counter())
        _benchmark_time(t2, t3, "RMP2_K")
        print("E_corr(RMP2) = " + str(-2*mp2_J + mp2_K))
        return -2*mp2_J + mp2_K

    def _kernel_opt_mem(self):
        
        if self.buffer is None:
            self.buffer = np.zeros((self.memory//8))
        
        t1 = (logger.process_clock(), logger.perf_counter())
        mp2_J = RMP2_J_opt_mem(self.Z,
                               self.X_o,
                               self.X_v,
                               self.tau_o,
                               self.tau_v,
                               self.buffer)
        t2 = (logger.process_clock(), logger.perf_counter())
        _benchmark_time(t1, t2, "RMP2_J_opt_mem")
        mp2_K = RMP2_K_opt_mem(self.Z,
                               self.X_o,
                               self.X_v,
                               self.tau_o,
                               self.tau_v,
                               self.buffer)
        t3 = (logger.process_clock(), logger.perf_counter())
        _benchmark_time(t2, t3, "RMP2_K_opt_mem")
        print("E_corr(RMP2) = " + str(-2*mp2_J + mp2_K))
        return -2*mp2_J + mp2_K

    #### driver #### 
    
    def kernel(self, schedule=SCHEDULE_TYPE_FORLOOP):
        if schedule == SCHEDULE_TYPE_NAIVE:
            return self._kernel_naive()
        elif schedule == SCHEDULE_TYPE_C:
            return self._kernel_C()
        elif schedule == SCHEDULE_TYPE_OPT_MEM:
            return self._kernel_opt_mem()
        else:
            raise ValueError("schedule type not supported")



if __name__ == "__main__":
    
    c = 25 
    N = 2
    
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
    cell.verbose = 4
    cell.ke_cutoff = 128
    cell.max_memory = 800  # 800 Mb
    cell.precision  = 1e-8  # integral precision
    cell.use_particle_mesh_ewald = True 
    
    verbose = 4
    
    prim_cell = build_supercell(cell.atom, cell.a, Ls = [1,1,1], ke_cutoff=cell.ke_cutoff, basis=cell.basis, pseudo=cell.pseudo)   
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
    
    ####### isdf MP2 can perform directly! ####### 
    
    import numpy
    from pyscf.pbc import gto, scf, mp    
            
    myisdf = ISDF.PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, aoR_cutoff=1e-8, direct=False, use_occ_RI_K=False)
    myisdf.build_IP_local(c=c, m=5, group=group_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
    myisdf.build_auxiliary_Coulomb(debug=True)
            
    mf_isdf = scf.RHF(cell)
    myisdf.direct_scf = mf_isdf.direct_scf
    mf_isdf.with_df = myisdf
    mf_isdf.max_cycle = 8
    mf_isdf.conv_tol = 1e-8
    mf_isdf.kernel()
     
    isdf_pt = mp.RMP2(mf_isdf)
    isdf_pt.kernel()
    
    ####### thc rmp2 #######
    
    _myisdf = ISDF.PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, aoR_cutoff=1e-8, direct=False, use_occ_RI_K=False)
    _myisdf.build_IP_local(c=12, m=5, group=group_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
    X          = _myisdf.aoRg_full() 
    thc_rmp2 = THC_RMP2(myisdf, mf_isdf, X)
    thc_rmp2.kernel(SCHEDULE_TYPE_NAIVE)
    thc_rmp2.kernel(SCHEDULE_TYPE_C)
    thc_rmp2.kernel(SCHEDULE_TYPE_OPT_MEM)