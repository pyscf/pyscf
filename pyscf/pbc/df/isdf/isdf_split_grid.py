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

from pyscf.pbc.df.isdf.isdf_mpi_tools import rank, comm, comm_size, allgather, bcast, reduce, gather, alltoall, _comm_bunch

from pyscf.pbc.df.isdf.isdf_fast_mpi import get_jk_dm_mpi

import ctypes, sys

from multiprocessing import Pool

from memory_profiler import profile

libpbc = lib.load_library('libpbc')

from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto
 
def build_auxiliary_Coulomb_local_bas_wo_robust_fitting(mydf, debug=True, use_mpi=False):
    
    if use_mpi:
        raise NotImplementedError("use_mpi = True is not supported")
        #### NOTE: one should bcast aux_basis first! ####

    
    t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    cell = mydf.cell
    mesh = cell.mesh
    mesh_int32         = np.array(mesh, dtype=np.int32)
    # mydf._allocate_jk_buffer(mydf.aoRg.dtype, mydf.ngrids_local)
    # mydf._allocate_jk_buffer(mydf.aoRg.dtype, mydf.ngrids_local)
    
    naux = mydf.naux
    
    ncomplex = mesh[0] * mesh[1] * (mesh[2] // 2 + 1) * 2 
    
    group_begin = mydf.group_begin
    group_end = mydf.group_end
    ngroup = len(mydf.group)
    
    grid_ordering = mydf.grid_ID_ordered 
    
    coulG = tools.get_coulG(cell, mesh=mesh)
    mydf.coulG = coulG.copy()
    coulG_real         = coulG.reshape(*mesh)[:, :, :mesh[2]//2+1].reshape(-1).copy()
    
    def construct_V(aux_basis:np.ndarray, buf, V, grid_ID, grid_ordering):
        fn = getattr(libpbc, "_construct_V_local_bas", None)
        assert(fn is not None)
        
        nThread = buf.shape[0]
        bufsize_per_thread = buf.shape[1]
        nrow = aux_basis.shape[0]
        ncol = aux_basis.shape[1]
        shift_row = 0
        
        fn(mesh_int32.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nrow),
                ctypes.c_int(ncol),
                grid_ID.ctypes.data_as(ctypes.c_void_p),
                aux_basis.ctypes.data_as(ctypes.c_void_p),
                coulG_real.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(shift_row),
                V.ctypes.data_as(ctypes.c_void_p),
                grid_ordering.ctypes.data_as(ctypes.c_void_p),
                buf.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(bufsize_per_thread))
        
    ####### allocate buf for V ########
    
    nThread = lib.num_threads()
    bufsize_per_thread = (coulG_real.shape[0] * 2 + mesh[0] * mesh[1] * mesh[2])
    buf = np.zeros((nThread, bufsize_per_thread), dtype=np.double)
    
    assert len(mydf.aux_basis) == ngroup
    
    naux_local = 0
    max_naux_bunch = 0
    for i in range(group_begin, group_end):
        naux_local += mydf.aux_basis[i].shape[0]    
        max_naux_bunch = max(max_naux_bunch, mydf.aux_basis[i].shape[0])
    
    if hasattr(mydf, "grid_pnt_near_atm"):
        max_naux_bunch = max(max_naux_bunch, len(mydf.grid_pnt_near_atm))
        if use_mpi == False or (use_mpi and rank == comm_size - 1):
            naux_local += len(mydf.grid_pnt_near_atm)
    
    V = np.zeros((max_naux_bunch, np.prod(mesh_int32)), dtype=np.double)
    
    naux = mydf.naux
    
    W = np.zeros((naux_local, naux), dtype=np.double)
    
    aux_row_loc = 0
    
    if hasattr(mydf, "grid_pnt_near_atm"):
        grid_ID_near_atm = mydf.grid_pnt_near_atm
    else:
        grid_ID_near_atm = []
        grid_ID_near_atm = np.array(grid_ID_near_atm, dtype=np.int32)
    for i in range(group_begin, group_end):
        
        aux_basis_now = mydf.aux_basis[i]
        naux_bra = aux_basis_now.shape[0]
        grid_ID = mydf.partition_group_to_gridID[i]
        
        construct_V(aux_basis_now, buf, V, grid_ID, grid_ordering)
        
        grid_shift = 0
        aux_col_loc = 0
        for j in range(0, ngroup):
            grid_ID_now = mydf.partition_group_to_gridID[j]
            aux_bas_ket = mydf.aux_basis[j]
            naux_ket = aux_bas_ket.shape[0]
            ngrid_now = grid_ID_now.size
            W[aux_row_loc:aux_row_loc+naux_bra, aux_col_loc:aux_col_loc+naux_ket] = lib.ddot(V[:naux_bra, grid_shift:grid_shift+ngrid_now], aux_bas_ket.T)
            grid_shift += ngrid_now
            aux_col_loc += naux_ket
        print("aux_row_loc = %d, aux_col_loc = %d" % (aux_row_loc, aux_col_loc))
        print("V.shape = ", V[:naux_bra,:].shape)
        W[aux_row_loc:aux_row_loc+naux_bra, aux_col_loc:] = V[:naux_bra, grid_shift:]
        aux_row_loc += aux_basis_now.shape[0]
    
    if (use_mpi == False or (use_mpi and rank == comm_size - 1)) and len(grid_ID_near_atm) != 0:
        ### construct the final row ### 
        grid_ID = grid_ID_near_atm
        aux_basis_now = np.identity(len(grid_ID), dtype=np.double)
        construct_V(aux_basis_now, buf, V, grid_ID, grid_ordering)
        grid_shift = 0
        aux_col_loc = 0
        naux_bra = len(grid_ID)
        for j in range(0, ngroup):
            grid_ID_now = mydf.partition_group_to_gridID[j]
            aux_bas_ket = mydf.aux_basis[j]
            naux_ket = aux_bas_ket.shape[0]
            ngrid_now = grid_ID_now.size
            W[aux_row_loc:aux_row_loc+naux_bra, aux_col_loc:aux_col_loc+naux_ket] = lib.ddot(V[:naux_bra, grid_shift:grid_shift+ngrid_now], aux_bas_ket.T)
            grid_shift += ngrid_now
            aux_col_loc += naux_ket
        assert aux_row_loc == aux_col_loc
        W[aux_row_loc:, aux_col_loc:] = V[:naux_bra, grid_shift:]
    
    del buf
    buf = None
    del V
    V = None
    
    mydf.W = W
    
    if use_mpi:
        comm.Barrier()
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    if mydf.verbose > 0:
        _benchmark_time(t0, t1, 'build_auxiliary_Coulomb')

def build_auxiliary_Coulomb_local_bas(mydf, debug=True, use_mpi=False):
    
    if hasattr(mydf, "grid_pnt_near_atm") and len(mydf.grid_pnt_near_atm) != 0 :
        raise NotImplementedError("grid_pnt_near_atm is not supported")
    
    t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    cell = mydf.cell
    mesh = cell.mesh
    
    # mydf._allocate_jk_buffer(mydf.aoR.dtype, mydf.ngrids_local)
    
    naux = mydf.naux
    
    ncomplex = mesh[0] * mesh[1] * (mesh[2] // 2 + 1) * 2 
    
    group_begin = mydf.group_begin
    group_end = mydf.group_end
    
    grid_ordering = mydf.grid_ID_ordered
    
    def construct_V_CCode(aux_basis:list[np.ndarray], mesh, coul_G):
        
        coulG_real         = coul_G.reshape(*mesh)[:, :, :mesh[2]//2+1].reshape(-1).copy()
        nThread            = lib.num_threads()
        bufsize_per_thread = int((coulG_real.shape[0] * 2 + mesh[0] * mesh[1] * mesh[2]) * 1.1)
        bufsize_per_thread = (bufsize_per_thread + 15) // 16 * 16
        
        buf = np.zeros((nThread, bufsize_per_thread), dtype=np.double)
        
        # nAux               = aux_basis.shape[0]
        
        nAux = 0
        for x in aux_basis:
            nAux += x.shape[0]
        
        ngrids             = mesh[0] * mesh[1] * mesh[2]
        mesh_int32         = np.array(mesh, dtype=np.int32)

        V                  = np.zeros((nAux, ngrids), dtype=np.double)
        
        fn = getattr(libpbc, "_construct_V_local_bas", None)
        assert(fn is not None)

        # print("V.shape = ", V.shape)
        # # print("aux_basis.shape = ", aux_basis.shape)
        # print("self.jk_buffer.size    = ", mydf.jk_buffer.size)
        # print("self.jk_buffer.shape   = ", mydf.jk_buffer.shape)
        # sys.stdout.flush()
        # print("len(aux_bas) = ", len(aux_basis))
        
        shift_row = 0
        ngrid_now = 0
        for i in range(len(aux_basis)):
            
            aux_basis_now = aux_basis[i]
            grid_ID = mydf.partition_group_to_gridID[group_begin+i]
            assert aux_basis_now.shape[1] == grid_ID.size 
            ngrid_now += grid_ID.size
            # print("i = ", i)
            # print("shift_row = ", shift_row) 
            # print("aux_bas_now = ", aux_basis_now.shape)
            # print("ngrid_now = ", ngrid_now)
            # print("buf = ", buf.shape)
            # print("ngrid_ordering = ", grid_ordering.size)
            # sys.stdout.flush()
        
            fn(mesh_int32.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(aux_basis_now.shape[0]),
                ctypes.c_int(aux_basis_now.shape[1]),
                grid_ID.ctypes.data_as(ctypes.c_void_p),
                aux_basis_now.ctypes.data_as(ctypes.c_void_p),
                coulG_real.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(shift_row),
                V.ctypes.data_as(ctypes.c_void_p),
                grid_ordering.ctypes.data_as(ctypes.c_void_p),
                buf.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(bufsize_per_thread))
        
            shift_row += aux_basis_now.shape[0]

        del buf
        buf = None

        return V
    
    ########### construct V ###########

    coulG = tools.get_coulG(cell, mesh=mesh)
    mydf.coulG = coulG.copy()
    V = construct_V_CCode(mydf.aux_basis, mesh, coulG)

    if use_mpi:

        ############# the only communication #############
    
        grid_segment = mydf.grid_segment 
        assert len(grid_segment) == comm_size + 1
    
        t0_comm = (lib.logger.process_clock(), lib.logger.perf_counter())
    
        sendbuf = []
        for i in range(comm_size):
            p0 = grid_segment[i]
            p1 = grid_segment[i+1]
            sendbuf.append(V[:, p0:p1])
        del V
        V = None
        V_fullrow = np.vstack(alltoall(sendbuf, split_recvbuf=True))
        del sendbuf
        sendbuf = None
    
        mydf.V_R = V_fullrow
    
        t1_comm = (lib.logger.process_clock(), lib.logger.perf_counter()) 
    
        t_comm = t1_comm[1] - t0_comm[1]
    
        if mydf.verbose > 0:
            print("rank = %d, t_comm = %12.6e" % (rank, t_comm))
    else:
        t_comm = 0.0
        mydf.V_R = V

    ########### construct W ###########
    
    aux_group_shift = [0]
    naux_now = 0
    for i in range(len(mydf.IP_group)):
        IP_group_now = mydf.IP_group[i]
        naux_now += len(IP_group_now)
        aux_group_shift.append(naux_now)
    
    mydf.W = np.zeros((mydf.naux, mydf.naux), dtype=np.float64) 
    
    grid_shift = 0
    for i in range(group_begin, group_end):
        aux_begin = aux_group_shift[i]
        aux_end   = aux_group_shift[i+1]
        ngrid_now = mydf.partition_group_to_gridID[i].size
        # print("aux_begin = %d, aux_end = %d, ngrid_now = %d" % (aux_begin, aux_end, ngrid_now))
        # print("grid_shift = %d" % grid_shift)
        # print("shape 1 = ", mydf.V_R[:,grid_shift:grid_shift+ngrid_now].shape)
        # print("shape 2 = ", mydf.aux_basis[i-group_begin].T.shape)
        sys.stdout.flush()
        mydf.W[:, aux_begin:aux_end] = lib.ddot(mydf.V_R[:, grid_shift:grid_shift+ngrid_now], mydf.aux_basis[i-group_begin].T)
        grid_shift += ngrid_now
    
    if use_mpi:
        comm.Barrier()
    
    # print("W = ", mydf.W[:5,:5])
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    if mydf.verbose > 0:
        _benchmark_time(t0, t1, 'build_auxiliary_Coulomb')
    
    sys.stdout.flush()

from pyscf.pbc.df.isdf.isdf_k import build_supercell

def build_supercell_with_partition(prim_atm, 
                                   prim_a, 
                                   mesh=None, 
                                   Ls = [1,1,1],
                                   partition = None, 
                                   basis='gth-dzvp', 
                                   pseudo='gth-pade', 
                                   ke_cutoff=70, 
                                   max_memory=2000, 
                                   precision=1e-8,
                                   use_particle_mesh_ewald=True,
                                   verbose=4):

    cell = build_supercell(prim_atm, prim_a, mesh=mesh, Ls=Ls, basis=basis, pseudo=pseudo, ke_cutoff=ke_cutoff, max_memory=max_memory, precision=precision, use_particle_mesh_ewald=use_particle_mesh_ewald, verbose=verbose)

    natm_prim = len(prim_atm)
    
    if partition is None:
        partition = []
        for i in range(natm_prim):
            partition.append([i])

    partition_supercell = []

    for ix in range(Ls[0]):
        for iy in range(Ls[1]):
            for iz in range(Ls[2]):
                cell_id = ix * Ls[1] * Ls[2] + iy * Ls[2] + iz
                for sub_partition in partition:
                    partition_supercell.append([x + cell_id * natm_prim for x in sub_partition])

    return cell, partition_supercell
