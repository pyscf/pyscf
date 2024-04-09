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
from pyscf import lib
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.gto import Cell
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member
from pyscf.gto.mole import *
from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time
import pyscf.pbc.df.isdf.isdf_ao2mo as isdf_ao2mo
import pyscf.pbc.df.isdf.isdf_jk as isdf_jk

from pyscf.pbc.df.isdf.isdf_fast import PBC_ISDF_Info

import pyscf.pbc.df.isdf.isdf_fast as ISDF
import pyscf.pbc.df.isdf.isdf_k as ISDF_K

from pyscf.pbc.df.isdf.isdf_tools_mpi import rank, comm, comm_size, allgather, bcast, reduce, gather, alltoall, _comm_bunch, allgather_pickle

from pyscf.pbc.df.isdf.isdf_fast_mpi import get_jk_dm_mpi

import ctypes, sys

from multiprocessing import Pool

from memory_profiler import profile

libpbc = lib.load_library('libpbc')

from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto
from pyscf.pbc.df.isdf.isdf_k import build_supercell
import pyscf.pbc.df.isdf.isdf_linear_scaling_base as ISDF_LinearScalingBase
import pyscf.pbc.df.isdf.isdf_linear_scaling as ISDF_LinearScaling
import pyscf.pbc.df.isdf.isdf_linear_scaling_jk as ISDF_LinearScalingJK

class PBC_ISDF_Info_Quad_MPI(ISDF_LinearScaling.PBC_ISDF_Info_Quad):
    
    # Quad stands for quadratic scaling
    
    def __init__(self, mol:Cell, 
                 # aoR: np.ndarray = None,
                 # with_robust_fitting=True,
                 Ls=None,
                 # get_partition=True,
                 verbose = 1,
                 rela_cutoff_QRCP = None,
                 aoR_cutoff = 1e-8,
                 # direct=False
                 ):
        
        super().__init__(mol, True, Ls, verbose, rela_cutoff_QRCP, aoR_cutoff, True)
        self.use_mpi = True

C = 15

from pyscf.lib.parameters import BOHR
from pyscf.pbc.df.isdf.isdf_split_grid import build_supercell_with_partition

if __name__ == '__main__':
    
    verbose = 4
    if rank != 0:
        verbose = 0
        
    # cell   = pbcgto.Cell()
    # boxlen = 3.5668
    # cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    # prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    # atm = [
    #     ['C', (0.     , 0.     , 0.    )],
    #     ['C', (0.8917 , 0.8917 , 0.8917)],
    #     ['C', (1.7834 , 1.7834 , 0.    )],
    #     ['C', (2.6751 , 2.6751 , 0.8917)],
    #     ['C', (1.7834 , 0.     , 1.7834)],
    #     ['C', (2.6751 , 0.8917 , 2.6751)],
    #     ['C', (0.     , 1.7834 , 1.7834)],
    #     ['C', (0.8917 , 2.6751 , 2.6751)],
    # ] 
    
    prim_a = np.array(
                    [[14.572056092/2, 0.000000000, 0.000000000],
                     [0.000000000, 14.572056092/2, 0.000000000],
                     [0.000000000, 0.000000000,  6.010273939],]) * BOHR
    atm = [
['Cu1',	(1.927800,	1.927800,	1.590250)],
['O1',	(1.927800,	0.000000,	1.590250)],
['O1',	(0.000000,	1.927800,	1.590250)],
['Ca',	(0.000000,	0.000000,	0.000000)],
    ]
    basis = {
        'Cu1':'ecpccpvdz', 'Cu2':'ecpccpvdz', 'O1': 'ecpccpvdz', 'Ca':'ecpccpvdz'
    }
    pseudo = {'Cu1': 'gth-pbe-q19', 'Cu2': 'gth-pbe-q19', 'O1': 'gth-pbe', 'Ca': 'gth-pbe'}
    ke_cutoff = 128 
    prim_cell = ISDF_K.build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=ke_cutoff, basis=basis, pseudo=pseudo)
    prim_mesh = prim_cell.mesh
    
    # KE_CUTOFF = 70
    KE_CUTOFF = 128
        
    # prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=KE_CUTOFF)
    prim_mesh = prim_cell.mesh
    prim_partition = [[0], [1], [2], [3], [4], [5], [6], [7]]
    # prim_partition = [[0, 1, 2, 3, 4, 5, 6, 7]]
    # prim_partition = [[0,1],[2,3],[4,5],[6,7]]
    
    prim_partition = [[0], [1], [2], [3]]
    # prim_partition = [[0, 1, 2, 3]]
    
    Ls = [2, 2, 1]
    Ls = np.array(Ls, dtype=np.int32)
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    cell, group_partition = build_supercell_with_partition(atm, prim_a, mesh=mesh, 
                                                     Ls=Ls,
                                                     basis=basis, pseudo=pseudo,
                                                     partition=prim_partition, ke_cutoff=KE_CUTOFF, verbose=verbose)
    print("group_partition = ", group_partition)
    
    pbc_isdf_info = PBC_ISDF_Info_Quad_MPI(cell, aoR_cutoff=1e-8, verbose=verbose)
    # pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
    pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition, Ls=[Ls[0]*3, Ls[1]*3, Ls[2]*3])
    pbc_isdf_info.Ls = Ls
    pbc_isdf_info.build_auxiliary_Coulomb(debug=True)
    
    # exit(1)
    
    from pyscf.pbc.df.isdf.isdf_tools_densitymatrix import init_guess_by_atom
    
    atm_config = {
        'Cu': {'charge': 2, 'occ_config': [6,12,9,0]},
        'O': {'charge': -2, 'occ_config': [4,6,0,0]},
        'Ca': {'charge': 2, 'occ_config': [6,12,0,0]},
    }
    
    dm = init_guess_by_atom(cell, atm_config) # a better init guess than the default one ! 
    
    if comm_size > 1:
        dm = bcast(dm, root=0)
    
    from pyscf.pbc import scf

    if comm_size > 1:
        comm.Barrier()

    mf = scf.RHF(cell)
    mf = scf.addons.smearing_(mf, sigma=0.2, method='fermi')
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 16
    mf.conv_tol = 0.0
    
    mf.kernel(dm)
    
    comm.Barrier()
    
    from pyscf.pbc.df.isdf.isdf_tools_densitymatrix import analysis_dm, analysis_dm_on_grid
    
    dm = mf.make_rdm1()