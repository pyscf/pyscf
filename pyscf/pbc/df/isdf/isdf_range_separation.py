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

import copy
from functools import reduce
import numpy as np
import ctypes
from multiprocessing import Pool
from memory_profiler import profile


from pyscf import lib
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.gto import Cell
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member
from pyscf.gto.mole import *
libpbc = lib.load_library('libpbc')

from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto 
import pyscf.pbc.df.isdf.isdf_linear_scaling as ISDF_LinearScaling
from pyscf.pbc.df.isdf.isdf_linear_scaling import PBC_ISDF_Info_Quad
import pyscf.pbc.df.isdf.isdf_tools_local as ISDF_Local_Utils
from pyscf.pbc.df.isdf.isdf_linear_scaling_k_jk import get_jk_dm_k_quadratic
from pyscf.pbc.df.isdf.isdf_tools_mpi import rank, comm, comm_size, allgather, bcast, reduce, gather, alltoall, _comm_bunch, allgather_pickle
from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time

# ------------ ISDF Range Separation ------------ 
# special consideration is needed 
# basically CC part cannot be expressed in a separable form
# but CC is localized anyway 

class PBC_ISDF_Info_RS(PBC_ISDF_Info_Quad):
    
    def __init__(self, mol:Cell, 
                with_robust_fitting=True,
                Ls=None,
                verbose = 1,
                rela_cutoff_QRCP = None,
                aoR_cutoff = 1e-8,
                direct=False,
                omega=None
                ):
        
        super().__init__(mol, with_robust_fitting, Ls, verbose, rela_cutoff_QRCP, aoR_cutoff, direct)
        
        #### deal with range separation #### 
        
        ########## supporting Range_separation ########## 

        self.use_aft_ao = False
        self.ke_cutoff_pp = self.cell.ke_cutoff
        self.ke_cutoff_ft_ao = self.cell.ke_cutoff
        self.ft_ao_mesh = self.mesh.copy()
        
        self.omega = omega 
        if omega is not None:
            self.omega = abs(omega)       ## LR ##
            # self.cell.omega = -abs(omega) ## SR ##
            self.cell_rsjk = self.cell.copy()
            self.cell_rsjk.omega = -abs(omega)

            nk = [1,1,1]
            kpts = self.cell_rsjk.make_kpts(nk)

            t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
            self.rsjk = RangeSeparatedJKBuilder(self.cell_rsjk, kpts)
            self.rsjk.exclude_dd_block = False
            self.rsjk.allow_drv_nodddd = False
            self.rsjk.build(omega=abs(omega))
            t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
            
            assert self.rsjk.has_long_range() == False
            assert self.rsjk.exclude_dd_block == False
            # assert self.rsjk._sr_without_dddd == False
            
            # self.cell.ke_cutoff = max(2*self.rsjk.ke_cutoff, self.cell.ke_cutoff)
            self.cell_rsjk.ke_cutoff = self.rsjk.ke_cutoff
            self.cell_rsjk.mesh = None
            self.cell_rsjk.build()
            mesh_tmp = self.cell_rsjk.mesh
            # if mesh_tmp[0] % 2 != 0:
            #     mesh_tmp[0] += 1
            # if mesh_tmp[1] % 2 != 0:
            #     mesh_tmp[1] += 1
            # if mesh_tmp[2] % 2 != 0:
            #     mesh_tmp[2] += 1
            self.cell_rsjk.build(mesh=mesh_tmp)
            self.mesh = self.cell_rsjk.mesh
            self.cell.build(mesh=mesh_tmp)
            
            self.ft_ao_mesh[0] = ((self.ft_ao_mesh[0] + self.mesh[0]-1) // self.mesh[0]) * self.mesh[0]
            self.ft_ao_mesh[1] = ((self.ft_ao_mesh[1] + self.mesh[1]-1) // self.mesh[1]) * self.mesh[1]
            self.ft_ao_mesh[2] = ((self.ft_ao_mesh[2] + self.mesh[2]-1) // self.mesh[2]) * self.mesh[2]
            
            ######## rebuild self.coords ######## 
            
            from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2

            df_tmp = MultiGridFFTDF2(self.cell_rsjk)
            self.coords = np.asarray(df_tmp.grids.coords).reshape(-1,3)
            self.ngrids = self.coords.shape[0]
            
            ke_cutoff_rsjk = self.rsjk.ke_cutoff
            
            if self.cell_rsjk.ke_cutoff < ke_cutoff_rsjk:
                print(" WARNING : ke_cutoff = %12.6e is smaller than ke_cutoff_rsjk = %12.6e" % (self.cell_rsjk.ke_cutoff, ke_cutoff_rsjk))
            
            print("ke_cutoff = %12.6e, ke_cutoff_rsjk = %12.6e" % (self.cell_rsjk.ke_cutoff, ke_cutoff_rsjk))
            
            if self.verbose:
                _benchmark_time(t1, t2, "build_RSJK") 
                
            t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
            cell_gdf = mol.copy()
            from pyscf.pbc.df import GDF
            self.gdf = GDF(cell_gdf, kpts)
            t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
            
            if self.verbose:
                _benchmark_time(t1, t2, "build_GDF")
            
            self.use_aft_ao = True
            
        else:
            self.rsjk = None
            self.cell_rsjk = None
        
        self.get_coulG()