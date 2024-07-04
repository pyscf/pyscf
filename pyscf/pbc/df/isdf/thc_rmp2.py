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
from pyscf.mp.mp2                     import MP2

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)

from pyscf.pbc.df.isdf.thc_backend import *

class THC_RMP2(_restricted_THC_posthf_holder, MP2):
    
    def __init__(self, 
                 my_mf=None, 
                 frozen=None, mo_coeff=None, mo_occ=None,
                 my_isdf=None, 
                 X=None, ## used in XXZXX
                 laplace_rela_err = 1e-7,
                 laplace_order    = 2,
                 #memory           = 128 * 1000 * 1000,
                 #with_mpi         = False,
                 no_LS_THC        = False,
                 use_torch=False,
                 with_gpu =False):
        
        if my_isdf is None:
            assert my_mf is not None
            my_isdf = my_mf.with_df

        _restricted_THC_posthf_holder.__init__(self, my_isdf, my_mf, X,
                                                laplace_rela_err = laplace_rela_err,
                                                laplace_order    = laplace_order,
                                                no_LS_THC        = no_LS_THC,
                                                use_torch        = use_torch,
                                                with_gpu         = with_gpu)

        MP2.__init__(self, my_mf, frozen, mo_coeff, mo_occ)
    
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
        mp2_J  = thc_einsum("iajb,iajb,ijab->", THC_ERI, THC_ERI, LAPLACE, backend=backend, memory=memory, return_path_only=return_path_only)
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        mp2_K  = thc_einsum("iajb,ibja,ijab->", THC_ERI, THC_ERI, LAPLACE, backend=backend, memory=memory, return_path_only=return_path_only)
        t3 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        _benchmark_time(t1, t2, 'THC_RMP2: mp2-J ', self._scf)
        _benchmark_time(t2, t3, 'THC_RMP2: mp2-K ', self._scf)
        
        if return_path_only:
            self.e_corr = None
            if backend == "opt_einsum":
                print("RMP2_J, path = ")
                print(mp2_J[1])
                print("RMP2_K, path = ")
                print(mp2_K[1])
            elif backend == "cotengra":
                print("RMP2_J, path = ")
                mp2_J.print_contractions()
                print("RMP2_K, path = ")
                mp2_K.print_contractions()
            return mp2_J, mp2_K
        else:
            self.e_corr = -2*mp2_J+mp2_K
        
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
    
    ####### thc rmp2 #######
    
    X,_ = myisdf.aoRg_full()
    
    thc_rmp2 = THC_RMP2(my_mf=mf_isdf, X=X, use_torch=True, with_gpu=True)
    
    e_mp2, _ = thc_rmp2.kernel(backend='cotengra')
    
    print("ISDF MP2 energy", e_mp2)
    
    e_mp2, _ = thc_rmp2.kernel(backend="opt_einsum")
    
    print("ISDF MP2 energy", e_mp2)
    
    