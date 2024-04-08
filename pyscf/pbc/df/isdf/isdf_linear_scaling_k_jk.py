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

import ctypes
from multiprocessing import Pool
from memory_profiler import profile
libpbc = lib.load_library('libpbc')
from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto 

import pyscf.pbc.df.isdf.isdf_k as ISDF_K
import pyscf.pbc.df.isdf.isdf_linear_scaling as ISDF_LinearScaling
import pyscf.pbc.df.isdf.isdf_linear_scaling_base as ISDF_LinearScalingBase
from pyscf.pbc.df.isdf.isdf_fast import rank, comm, comm_size, allgather, bcast, reduce, gather, alltoall, _comm_bunch, allgather_pickle

def _contract_j_dm_k_ls(mydf, dm, use_mpi=False):
    
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        # raise NotImplementedError("MPI is not supported yet.")
        assert mydf.direct == True
    
    t1 = (logger.process_clock(), logger.perf_counter())

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]
        
    nao  = dm.shape[0]
    cell = mydf.cell
    assert cell.nao == nao
    vol = cell.vol
    mesh = np.array(cell.mesh, dtype=np.int32)
    ngrid = np.prod(mesh)
    ngrid_prim = ngrid // np.prod(mydf.kmesh)

    aoR  = mydf.aoR
    assert isinstance(aoR, list)
    naux = mydf.naux
    aoR1 = mydf.aoR1
    assert isinstance(aoR1, list)
    
    #### step 0. allocate buffer 
    
    max_nao_involved = np.max([aoR_holder.aoR.shape[0] for aoR_holder in aoR if aoR_holder is not None])
    max_nao_involved1 = np.max([aoR_holder.aoR.shape[0] for aoR_holder in aoR1 if aoR_holder is not None])
    max_nao_involved = max(max_nao_involved, max_nao_involved1)
    max_ngrid_involved = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None])
    max_ngrid_involved1 = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoR1 if aoR_holder is not None])
    max_ngrid_involved = max(max_ngrid_involved, max_ngrid_involved1)
    # ngrids_local = np.sum([aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None])
    
    # density_R = np.zeros((ngrids_local,), dtype=np.float64)
    # density_R = np.zeros((ngrid,), dtype=np.float64)
    density_R_prim = np.zeros((ngrid_prim,), dtype=np.float64)
    
    dm_buf = np.zeros((max_nao_involved, max_nao_involved), dtype=np.float64)
    # ddot_buf = np.zeros((max_nao_involved, max_ngrid_involved), dtype=np.float64)
    max_dim_buf = max(max_ngrid_involved, max_nao_involved)
    ddot_buf = np.zeros((max_dim_buf, max_dim_buf), dtype=np.float64)
    aoR_buf1 = np.zeros((max_nao_involved, max_ngrid_involved), dtype=np.float64)
    
    ##### get the involved C function ##### 
    
    fn_extract_dm = getattr(libpbc, "_extract_dm_involved_ao", None) 
    assert fn_extract_dm is not None
    
    fn_packadd_dm = getattr(libpbc, "_packadd_local_dm", None)
    assert fn_packadd_dm is not None
    
    #### step 1. get density value on real space grid and IPs
    
    # lib.ddot(dm, aoR, c=buffer1) 
    # tmp1 = buffer1
    # density_R = np.asarray(lib.multiply_sum_isdf(aoR, tmp1, out=buffer2), order='C')
    
    density_R_tmp = None
    
    for atm_id, aoR_holder in enumerate(aoR):
        
        if aoR_holder is None:
            continue
        
        if use_mpi:
            if atm_id % comm_size != rank:
                continue
            
        ngrids_now = aoR_holder.aoR.shape[1]
        nao_involved = aoR_holder.aoR.shape[0]
        
        if nao_involved < nao:
            fn_extract_dm(
                dm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao),
                dm_buf.ctypes.data_as(ctypes.c_void_p),
                aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_involved),
            )
        else:
            dm_buf.ravel()[:] = dm.ravel()
        
        dm_now = np.ndarray((nao_involved, nao_involved), buffer=dm_buf)
    
        ddot_res = np.ndarray((nao_involved, ngrids_now), buffer=ddot_buf)
        
        lib.ddot(dm_now, aoR_holder.aoR, c=ddot_res)
        density_R_tmp = lib.multiply_sum_isdf(aoR_holder.aoR, ddot_res)
        
        global_gridID_begin = aoR_holder.global_gridID_begin
        
        density_R_prim[global_gridID_begin:global_gridID_begin+ngrids_now] = density_R_tmp
        # local_grid_loc += ngrids_now
    
    # assert local_grid_loc == ngrids_local
    
    # print("density_R = ", density_R_prim[:64])
    # sys.stdout.flush()
    
    # if use_mpi == False:
    #     assert ngrids_local == np.prod(mesh)
    
    if use_mpi:
        # density_R = np.hstack(gather(density_R, comm, rank, root=0, split_recvbuf=True))
        density_R_prim = reduce(density_R_prim, root=0)
        
    # if hasattr(mydf, "grid_ID_ordered"):
    
    grid_ID_ordered = mydf.grid_ID_ordered_prim
    
    # print("grid_ID_ordered = ", grid_ID_ordered[:64])
    # print("grid_ID_ordered.shape = ", grid_ID_ordered.shape)
    # sys.stdout.flush()
    
    if (use_mpi and rank == 0) or (use_mpi == False):
        
        density_R_original = np.zeros_like(density_R_prim)
            
        fn_order = getattr(libpbc, "_Reorder_Grid_to_Original_Grid", None)
        assert fn_order is not None
            
        fn_order(
            ctypes.c_int(density_R_prim.size),
            mydf.grid_ID_ordered_prim.ctypes.data_as(ctypes.c_void_p),
            density_R_prim.ctypes.data_as(ctypes.c_void_p),
            density_R_original.ctypes.data_as(ctypes.c_void_p),
        )

        density_R_prim = density_R_original.copy()
    
    # print("density_R = ", density_R_prim[:64])
    # sys.stdout.flush()
    
    J = None
    
    if (use_mpi and rank == 0) or (use_mpi == False):
    
        fn_J = getattr(libpbc, "_construct_J", None)
        assert(fn_J is not None)

        if hasattr(mydf, "coulG_prim") == False:
            mydf.coulG_prim = tools.get_coulG(mydf.primCell, mesh=mydf.primCell.mesh)

        J = np.zeros_like(density_R_prim)

        mesh_prim = np.array(mydf.primCell.mesh, dtype=np.int32)

        fn_J(
            mesh_prim.ctypes.data_as(ctypes.c_void_p),
            density_R_prim.ctypes.data_as(ctypes.c_void_p),
            mydf.coulG_prim.ctypes.data_as(ctypes.c_void_p),
            J.ctypes.data_as(ctypes.c_void_p),
        )
        
        # if hasattr(mydf, "grid_ID_ordered"):
            
        J_ordered = np.zeros_like(J)

        fn_order = getattr(libpbc, "_Original_Grid_to_Reorder_Grid", None)
        assert fn_order is not None 
            
        fn_order(
            ctypes.c_int(J.size),
            grid_ID_ordered.ctypes.data_as(ctypes.c_void_p),
            J.ctypes.data_as(ctypes.c_void_p),
            J_ordered.ctypes.data_as(ctypes.c_void_p),
        )
            
        J = J_ordered.copy()
            
    # print("J = ", J[:64])
    # sys.stdout.flush()
            
    if use_mpi:
        # grid_segment = mydf.grid_segment
        # sendbuf = None
        # if rank == 0:
        #     sendbuf = []
        #     for i in range(size):
        #         p0 = grid_segment[i]
        #         p1 = grid_segment[i+1]
        #         sendbuf.append(J[p0:p1])
        # J = scatter(sendbuf, comm, rank, root=0)
        # del sendbuf
        # sendbuf = None 
        
        J = bcast(J, root=0)
    
    #### step 3. get J , using translation symmetry
    
    # J = np.asarray(lib.d_ij_j_ij(aoR, J, out=buffer1), order='C') 
    # J = lib.ddot_withbuffer(aoR, J.T, buf=mydf.ddot_buf)

    # local_grid_loc = 0

    nao_prim = mydf.nao_prim
    J_Res = np.zeros((nao_prim, nao), dtype=np.float64)

    partition_activated_ID = mydf.partition_activated_id
    
    kmesh = mydf.kmesh
    natm_prim = mydf.natmPrim
    
    # grid_segment = [0]
    # grid_tot = 0
    # for i in range(natm_prim):
    #     grid_now = mydf.aoR1[i].aoR.shape[1]
    #     grid_segment.append(grid_segment[-1] + grid_now)
    #     grid_tot += grid_now
    
    grid_segment = mydf.grid_segment
    # print("grid_segment = ", grid_segment)  
    
    fn_packadd_J = getattr(libpbc, "_buildJ_k_packaddrow", None)
    assert fn_packadd_J is not None
    
    for task_id, box_id in enumerate(partition_activated_ID):
        
        if use_mpi:
            if task_id % comm_size != rank:
                continue
        
        box_loc1 = box_id // natm_prim
        box_loc2 = box_id % natm_prim
        
        box_x = box_loc1 // (kmesh[1] * kmesh[2])
        box_y = box_loc1 % (kmesh[1] * kmesh[2]) // kmesh[2]
        box_z = box_loc1 % kmesh[2]
        
        aoR_holder_bra = aoR1[box_id]
    
        permutation = mydf._get_permutation_column_aoR(box_x, box_y, box_z, box_loc2)
        
        aoR_holder_ket = aoR[box_loc2]
        
        J_tmp = J[grid_segment[box_loc2]:grid_segment[box_loc2+1]]
        
        # print("box_id   = ", box_id)
        # print("box_loc1 = ", box_loc1)
        # print("box_loc2 = ", box_loc2)
        # print("J_tmp.size = ", J_tmp.size)
        # print("aoR_holder_ket.aoR.shape = ", aoR_holder_ket.aoR.shape)
        # print("grid_segment = ", grid_segment)
        
        assert aoR_holder_ket.aoR.shape[1] == J_tmp.size
        
        
        
        aoR_J_res = np.ndarray(aoR_holder_bra.aoR.shape, buffer=aoR_buf1)
        lib.d_ij_j_ij(aoR_holder_bra.aoR, J_tmp, out=aoR_J_res)
        
        nao_bra = aoR_holder_bra.aoR.shape[0]
        nao_ket = aoR_holder_ket.aoR.shape[0]

        ddot_res = np.ndarray((nao_bra, nao_ket), buffer=ddot_buf)
        lib.ddot(aoR_J_res, aoR_holder_ket.aoR.T, c=ddot_res)
        
        #### pack and add the result to J_Res
        
        fn_packadd_J(
            J_Res.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_prim),
            ctypes.c_int(nao),
            ddot_res.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_bra),
            ctypes.c_int(nao_ket),
            aoR_holder_bra.ao_involved.ctypes.data_as(ctypes.c_void_p),
            permutation.ctypes.data_as(ctypes.c_void_p),
        )
    
    # assert local_grid_loc == ngrids_local

    J = J_Res

    if use_mpi:
        # J = mpi_reduce(J, comm, rank, op=MPI.SUM, root=0)
        J = reduce(J, root=0)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    if mydf.verbose:
        _benchmark_time(t1, t2, "_contract_j_dm_k_ls")
    
    ######### delete the buffer #########
    
    del dm_buf, ddot_buf, density_R_prim
    del density_R_tmp
    del aoR_buf1
    
    J = ISDF_K._pack_JK(J, mydf.kmesh, nao_prim)
    
    # diff = np.max(np.abs(J - J.T))
    # print("diff = ", diff)    
    # assert np.allclose(J, J.T)
    
    return J * ngrid / vol



def get_jk_dm_k_quadratic(mydf, dm, hermi=1, kpt=np.zeros(3),
                          kpts_band=None, with_j=True, with_k=True, omega=None, 
                          **kwargs):
    
    '''JK for given k-point'''
    
    direct = mydf.direct
    use_mpi = mydf.use_mpi
    
    if use_mpi :
        raise NotImplementedError("ISDF does not support use_mpi")
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    if hasattr(mydf, 'Ls'):
        from pyscf.pbc.df.isdf.isdf_k import _symmetrize_dm
        dm = _symmetrize_dm(dm, mydf.Ls)

    if use_mpi:
        dm = bcast(dm, root=0)

    #### perform the calculation ####

    # if mydf.jk_buffer is None:  # allocate the buffer for get jk, NOTE: do not need anymore
    #     mydf._allocate_jk_buffer(dm.dtype)

    if "exxdiv" in kwargs:
        exxdiv = kwargs["exxdiv"]
    else:
        exxdiv = None

    vj = vk = None

    if kpts_band is not None and abs(kpt-kpts_band).sum() > 1e-9:
        raise NotImplementedError("ISDF does not support kpts_band != kpt")

    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

    j_real = gamma_point(kpt)
    k_real = gamma_point(kpt) and not np.iscomplexobj(dm)

    assert j_real
    assert k_real

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, (mydf.max_memory - mem_now))

    log.debug1('max_memory = %d MB (%d in use)', max_memory, mem_now)

    if with_j:
        vj = _contract_j_dm_k_ls(mydf, dm, use_mpi)  
        # if rank == 0:
        # print("vj = ", vj[0, :16])
        # print("vj = ", vj[0, -16:])
        sys.stdout.flush()
    if with_k:
        if mydf.direct:
            # vk = _contract_k_dm_k_quadratic_direct(mydf, dm, use_mpi=use_mpi)
            raise NotImplementedError
        else:
            # vk = _contract_k_dm_quadratic(mydf, dm, mydf.with_robust_fitting, use_mpi=use_mpi)
            # raise NotImplementedError
            if mydf.with_robust_fitting:
                vk = ISDF_K._get_k_kSym_robust_fitting(mydf, dm)
            else:
                vk = ISDF_K._get_k_kSym(mydf, dm)
        # if rank == 0:
        # print("vk = ", vk[0, :16])
        # print("vk = ", vk[0, -16:])
        if exxdiv == 'ewald':
            print("WARNING: ISDF does not support ewald")

    t1 = log.timer('sr jk', *t1)

    return vj, vk