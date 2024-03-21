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

# from mpi4pyscf.tools import mpi
# from mpi4pyscf.tools.mpi import allgather, bcast,  reduce

from pyscf.pbc.df.isdf.isdf_fast import rank, comm, comm_size, allgather, bcast, matrix_all2all_Col2Row, matrix_all2all_Row2Col, reduce

# comm = mpi.comm
# rank = mpi.rank

import ctypes
from multiprocessing import Pool
from memory_profiler import profile

libpbc = lib.load_library('libpbc')
from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto

import pyscf.pbc.df.isdf.isdf_fast_mpi as isdf_fast_mpi

C = 7

if __name__ == '__main__':
    
    # from pyscf.pbc.df.isdf import isdf_fast_mpi

    cell   = pbcgto.Cell()
    
    boxlen = 4.2
    cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    cell.atom = '''
Li 0.0   0.0   0.0
Li 2.1   2.1   0.0
Li 0.0   2.1   2.1
Li 2.1   0.0   2.1
H  0.0   0.0   2.1
H  0.0   2.1   0.0
H  2.1   0.0   0.0
H  2.1   2.1   2.1
'''

    cell.basis   = 'gth-dzvp'
    # cell.basis   = 'gth-tzvp'
    cell.pseudo  = 'gth-pade'
    
    if rank == 0:
        cell.verbose = 4
    else:
        cell.verbose = 0

    # cell.ke_cutoff  = 32   # kinetic energy cutoff in a.u.
    cell.ke_cutoff = 128
    cell.max_memory = 800  # 800 Mb
    cell.precision  = 1e-8  # integral precision
    cell.use_particle_mesh_ewald = True

    cell.build()

    cell = tools.super_cell(cell, [1, 1, 1])
    
    pbc_isdf_info = isdf_fast_mpi.PBC_ISDF_Info_MPI(cell)
    # build_partition(pbc_isdf_info)        
    pbc_isdf_info.build_IP_Sandeep(C, 5, global_IP_selection=True, debug=True)
    pbc_isdf_info.build_auxiliary_Coulomb(debug=True)
    
    sys.stdout.flush()
    
    ### run scf ###
    
    comm.Barrier()
    
    pbc_isdf_info_benchmark = None
    
    aux_bas = comm.gather(pbc_isdf_info.aux_basis, root=0)
    aoR = comm.gather(pbc_isdf_info.aoR, root=0)
    V_R = comm.gather(pbc_isdf_info.V_R, root=0)
    
    aux_bas_fft = comm.gather(pbc_isdf_info.basis_fft, root=0)
    
    # if rank == 0:
    #     print("aux_bas_fft = ", aux_bas_fft)
    # if rank == 0:
    #     import pyscf.pbc.df.isdf.isdf_fast as isdf
    #     from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2, _eval_rhoG
    #     df_tmp = MultiGridFFTDF2(cell)
    #     grids  = df_tmp.grids
    #     mesh   = grids.mesh
    #     ngrids = np.prod(mesh)
    #     coords = np.asarray(grids.coords).reshape(-1,3)
    #     assert ngrids == coords.shape[0]
    #     aoR_bench   = df_tmp._numint.eval_ao(cell, coords)[0].T  # the T is important
    #     aoR_bench  *= np.sqrt(cell.vol / ngrids)
    #     df_tmp = None
    #     pbc_isdf_info_benchmark = isdf.PBC_ISDF_Info(cell, aoR=aoR_bench)
    #     pbc_isdf_info_benchmark.build_IP_Sandeep(C, 5, global_IP_selection=True, debug=True, IP_ID=pbc_isdf_info.IP_ID)
    #     pbc_isdf_info_benchmark.build_auxiliary_Coulomb(debug=True)
    #     aux_bas = np.hstack(aux_bas) 
    #     aoR  = np.hstack(aoR)
    #     V_R  = np.hstack(V_R)
    #     aux_bas_fft = np.hstack(aux_bas_fft)
    #     W1 = pbc_isdf_info.W
    #     W2 = pbc_isdf_info_benchmark.W
    #     if np.allclose(W1, W2):
    #         print("W PASS")
    #     else:
    #         print("W FAIL with diff = ", np.linalg.norm(W1 - W2))
    #     aux_bas_bench = pbc_isdf_info_benchmark.aux_basis
    #     if np.allclose(aux_bas, aux_bas_bench):
    #         print("aux_bas PASS")
    #     else:
    #         print("aux_bas FAIL with diff = ", np.linalg.norm(aux_bas - aux_bas_bench))
    #     V_R_bench = pbc_isdf_info_benchmark.V_R
    #     if np.allclose(V_R, V_R_bench):
    #         print("V_R PASS")
    #     else:
    #         print("V_R FAIL with diff = ", np.linalg.norm(V_R - V_R_bench))
    #     aoR_bench = pbc_isdf_info_benchmark.aoR
    #     if np.allclose(aoR, aoR_bench):
    #         print("aoR PASS")
    #     else:
    #         print("aoR FAIL with diff = ", np.linalg.norm(aoR - aoR_bench))
    #     ncomplex = cell.mesh[0] * cell.mesh[1] * (cell.mesh[2] // 2 + 1) * 2
    #     aux_bas_bench = aux_bas_bench.reshape(-1, *cell.mesh)
    #     # print("aux_bas_bench = ", aux_bas_bench.shape)
    #     aux_bas_rfft_bench = np.fft.rfftn(aux_bas_bench, axes=(1,2,3))
    #     aux_bas_rfft_bench = aux_bas_rfft_bench.reshape(aux_bas_bench.shape[0], -1)
    #     aux_bas_rfft_bench_real = np.ndarray((aux_bas_bench.shape[0], ncomplex), dtype=np.float64, buffer=aux_bas_rfft_bench) 
    #     # print("aux_bas_rfft_bench_real = ", aux_bas_rfft_bench_real.shape)
    #     # print("aux_bas_fft = ", aux_bas_fft.shape)
    #     print("aux_bas_rfft_bench_real = ", aux_bas_rfft_bench_real[0, 0:5])
    #     print("aux_bas_fft = ", aux_bas_fft[0, 0:5])
    #     if np.allclose(aux_bas_fft, aux_bas_rfft_bench_real):
    #         print("aux_bas_fft PASS")
    #     else:
    #         print("aux_bas_fft FAIL with diff = ", np.linalg.norm(aux_bas_fft - aux_bas_rfft_bench_real))
    #     from pyscf.pbc import scf
    #     mf = scf.RHF(cell)
    #     # pbc_isdf_info.direct_scf = mf.direct_scf
    #     mf.with_df = pbc_isdf_info_benchmark
    #     mf.max_cycle = 100
    #     mf.conv_tol = 1e-7
    #     # print("mf.direct_scf = ", mf.direct_scf)
    #     mf.kernel()
        
        #### set all W matrix into zero #### 
        
        # pbc_isdf_info_benchmark.W = np.zeros_like(pbc_isdf_info_benchmark.W)
        
    # pbc_isdf_info.W = np.zeros_like(pbc_isdf_info.W)
    
    # pbc_isdf_info.W = comm.bcast(pbc_isdf_info.W, root=0)
        
    for _ in range(4):

        dm = None 
        if rank == 0:
            dm = np.random.random((1, cell.nao, cell.nao))
        dm = bcast(dm, root=0)
        res = pbc_isdf_info.get_jk(dm, with_j=True, with_k=True)
        # pbc_isdf_info_benchmark = None
        
        # if rank == 0:
        #     res_bench = pbc_isdf_info_benchmark.get_jk(dm, with_j=True, with_k=True)
        #     print(res[0][0,-10:])
        #     print(res_bench[0][0,-10:])
        #     print(res[1][0,-10:])
        #     print(res_bench[1][0,-10:])
        #     # print(res[0] - res_bench[0])
        #     # print(res[1] - res_bench[1])
        #     # assert np.allclose(res[0], res_bench[0])
        #     # assert np.allclose(res[1], res_bench[1])
        #     if np.allclose(res[0], res_bench[0]) and np.allclose(res[1], res_bench[1]):
        #         print("PASS")
        #     else:
        #         print("FAIL")
    
    # exit(1)
    
    from pyscf.pbc import scf
    
    for _ in range(2):

        print("SCF CYCLE ", )
        mf = scf.RHF(cell)
        # pbc_isdf_info.direct_scf = mf.direct_scf
        mf.with_df = pbc_isdf_info
        mf.max_cycle = 100
        mf.conv_tol = 1e-7
        # print("mf.direct_scf = ", mf.direct_scf)
        # mf.kernel()
        # mf = None
        
        dm1 = mf.get_init_guess(cell, 'atom')
        
        dm1 = bcast(dm1, root=0)

        res = pbc_isdf_info.get_jk(dm, with_j=True, with_k=True)
        print("finish atm dm1 get jk")
    
    # if rank == 0:
    #     dm = mf.make_rdm1()
    #     jk1 = pbc_isdf_info.get_jk(dm, with_j=True, with_k=True)
    #     jk2 = pbc_isdf_info_benchmark.get_jk(dm, with_j=True, with_k=True)
    #     np.allclose(jk1, jk2)
    
    
    pbc_isdf_info = None