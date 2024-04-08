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

def _get_k_kSym_robust_fitting_fast(mydf, dm):
    
    '''
    this is a slow version, abandon ! 
    '''
 
    #### preprocess ####  
    
    mydf._allocate_jk_buffer(dm.dtype)
    t1 = (logger.process_clock(), logger.perf_counter())
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]
    
    nao  = dm.shape[0]
    cell = mydf.cell    
    assert cell.nao == nao
    ngrid = np.prod(cell.mesh)
    vol = cell.vol
    
    W         = mydf.W
    # aoRg      = mydf.aoRg
    # aoRg_Prim = mydf.aoRg_Prim
    # naux      = aoRg.shape[1]
    naux = mydf.naux
    
    Ls = np.array(mydf.Ls, dtype=np.int32)
    mesh = mydf.mesh
    meshPrim = np.array(mesh) // np.array(Ls)
    nGridPrim = mydf.nGridPrim
    ncell = np.prod(Ls)
    ncell_complex = Ls[0] * Ls[1] * (Ls[2]//2+1)
    nIP_prim = mydf.nIP_Prim
    nao_prim = nao // ncell
    
    #### allocate buffer ####
     
    
    offset = 0
    
    DM_complex = np.ndarray((nao_prim,nao_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    # DM_complex = np.ndarray((nao_prim,nao_prim*ncell_complex), dtype=np.complex128)
    DM_real = np.ndarray((nao_prim,nao), dtype=np.float64, buffer=DM_complex)
    DM_real.ravel()[:] = dm[:nao_prim, :].ravel()[:]
    offset += DM_complex.size * DM_complex.itemsize
    
    offset_after_dm = offset
    
    DM_RgRg_complex = np.ndarray((nIP_prim,nIP_prim*ncell_complex), dtype=np.complex128,  buffer=mydf.jk_buffer, offset=offset)
    DM_RgRg_real = np.ndarray((nIP_prim,nIP_prim*ncell), dtype=np.float64, buffer=DM_RgRg_complex)
    offset += DM_RgRg_complex.size * DM_RgRg_complex.itemsize

    offset_after_DM = offset
    
    #### get D ####
    
    #_get_DM_RgRg_real(mydf, DM_real, DM_complex, DM_RgRg_real, DM_RgRg_complex, offset)
    
    fn1 = getattr(libpbc, "_FFT_Matrix_Col_InPlace", None)
    assert fn1 is not None
    
    fn_packcol2 = getattr(libpbc, "_buildK_packcol2", None)
    assert fn_packcol2 is not None
    fn_packcol3 = getattr(libpbc, "_buildK_packcol3", None)
    assert fn_packcol3 is not None
    
    fn_copy = getattr(libpbc, "_buildK_copy", None)
    assert fn_copy is not None
    
    buf_fft = np.ndarray((nao_prim, nao_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    t3 = (logger.process_clock(), logger.perf_counter())
    
    fn1(
        DM_real.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao_prim),
        ctypes.c_int(nao_prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "_fft1")
    
    buf_A = np.ndarray((nao_prim, nao_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    offset2 = offset + (nao_prim * nao_prim) * buf_A.itemsize
    buf_B = np.ndarray((nao_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset2)
    
    offset3 = offset2 + (nao_prim * nIP_prim) * buf_B.itemsize
    buf_C = np.ndarray((nao_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset3)
    
    offset4 = offset3 + (nao_prim * nIP_prim) * buf_C.itemsize
    buf_D = np.ndarray((nIP_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset4)
    
    aoRg_FFT = mydf.aoRg_FFT
    
    t3 = (logger.process_clock(), logger.perf_counter())
    
    if isinstance(aoRg_FFT, list):
        
        for i in range(ncell_complex):
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            # buf_A[:] = DM_complex[:, k_begin:k_end]
            fn_packcol2(
                buf_A.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_prim),
                ctypes.c_int(2*nao_prim),
                DM_complex.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(DM_complex.shape[0]),
                ctypes.c_int(2*DM_complex.shape[1]),
                ctypes.c_int(2*k_begin),
                ctypes.c_int(2*k_end)   # 2 due to complex number
            )
            
            # buf_B[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim]
            # buf_B.ravel()[:] = aoRg_FFT[i].ravel()[:]
            fn_copy(
                buf_B.ctypes.data_as(ctypes.c_void_p),
                aoRg_FFT[i].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_size_t(2*buf_B.size) # 2 due to complex number
            )
        
            lib.dot(buf_A, buf_B, c=buf_C)
            lib.dot(buf_B.T.conj(), buf_C, c=buf_D)
        
            k_begin = i * nIP_prim
            k_end   = (i + 1) * nIP_prim
        
            # DM_RgRg_complex[:, k_begin:k_end] = buf_D 
            fn_packcol3(
                DM_RgRg_complex.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(DM_RgRg_complex.shape[0]),
                ctypes.c_int(2*DM_RgRg_complex.shape[1]),
                ctypes.c_int(2*k_begin),
                ctypes.c_int(2*k_end),
                buf_D.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(buf_D.shape[0]),
                ctypes.c_int(2*buf_D.shape[1]),
            )
            
    else:
    
        raise NotImplementedError("not implemented yet.")
    
        for i in range(ncell_complex):
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            buf_A[:] = DM_complex[:, k_begin:k_end]
            buf_B[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim]
        
            lib.dot(buf_A, buf_B, c=buf_C)
            lib.dot(buf_B.T.conj(), buf_C, c=buf_D)
        
            k_begin = i * nIP_prim
            k_end   = (i + 1) * nIP_prim
        
            DM_RgRg_complex[:, k_begin:k_end] = buf_D
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "DM_RgRg_complex")
    
    t3 = t4
    
    buf_fft = np.ndarray((nIP_prim, nIP_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    fn2 = getattr(libpbc, "_iFFT_Matrix_Col_InPlace", None)
    assert fn2 is not None
        
    fn2(
        DM_RgRg_complex.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_prim),
        ctypes.c_int(nIP_prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "DM_RgRg_complex 2")
    t3 = t4
    
    # inplace multiplication
    
    lib.cwise_mul(mydf.W, DM_RgRg_real, out=DM_RgRg_real)
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "lib.cwise_mul 2")
    t3 = t4
    
    offset = offset_after_DM
    
    buf_fft = np.ndarray((nIP_prim, nIP_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    fn1(
        DM_RgRg_real.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_prim),
        ctypes.c_int(nIP_prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "DM_RgRg_real")
    t3 = t4
    
    K_complex_buf = np.ndarray((nao_prim, nao_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    K_real_buf    = np.ndarray((nao_prim, nao_prim*ncell), dtype=np.float64, buffer=mydf.jk_buffer, offset=offset)
    offset += (nao_prim * nao_prim * ncell_complex) * K_complex_buf.itemsize
    offset_now = offset    
    
    buf_A = np.ndarray((nIP_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_now)
    offset_now += (nIP_prim * nIP_prim) * buf_A.itemsize
    buf_B = np.ndarray((nao_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_now)
    offset_now += (nao_prim * nIP_prim) * buf_B.itemsize
    buf_C = np.ndarray((nIP_prim, nao_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_now)
    offset_now += (nIP_prim * nao_prim) * buf_C.itemsize
    buf_D = np.ndarray((nao_prim, nao_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_now)
    
    if isinstance(aoRg_FFT, list):
        for i in range(ncell_complex):
        
            k_begin = i * nIP_prim
            k_end   = (i + 1) * nIP_prim
        
            # buf_A.ravel()[:] = DM_RgRg_complex[:, k_begin:k_end].ravel()[:]
            fn_packcol2(
                buf_A.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nIP_prim),
                ctypes.c_int(2*nIP_prim),
                DM_RgRg_complex.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(DM_RgRg_complex.shape[0]),
                ctypes.c_int(2*DM_RgRg_complex.shape[1]),
                ctypes.c_int(2*k_begin),
                ctypes.c_int(2*k_end)
            )
            
            # buf_B.ravel()[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim].ravel()[:]
            # buf_B.ravel()[:] = aoRg_FFT[i].ravel()[:]
            fn_copy(
                buf_B.ctypes.data_as(ctypes.c_void_p),
                aoRg_FFT[i].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_size_t(2*buf_B.size) # 2 due to complex number
            )
            
        
            lib.dot(buf_A, buf_B.T.conj(), c=buf_C)
            lib.dot(buf_B, buf_C, c=buf_D)
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            # K_complex_buf[:, k_begin:k_end] = buf_D
            
            fn_packcol3(
                K_complex_buf.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(K_complex_buf.shape[0]),
                ctypes.c_int(2*K_complex_buf.shape[1]),
                ctypes.c_int(2*k_begin),
                ctypes.c_int(2*k_end),
                buf_D.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(buf_D.shape[0]),
                ctypes.c_int(2*buf_D.shape[1]),
            )
            
    else:
        
        raise NotImplementedError("not implemented yet.")
        
        for i in range(ncell_complex):
        
            k_begin = i * nIP_prim
            k_end   = (i + 1) * nIP_prim
        
            buf_A.ravel()[:] = DM_RgRg_complex[:, k_begin:k_end].ravel()[:]
            buf_B.ravel()[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim].ravel()[:]
        
            lib.dot(buf_A, buf_B.T.conj(), c=buf_C)
            lib.dot(buf_B, buf_C, c=buf_D)
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            K_complex_buf[:, k_begin:k_end] = buf_D
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "K_complex_buf")
    t3 = t4
    
    buf_fft = np.ndarray((nao_prim, nao_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    fn2(
        K_complex_buf.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao_prim),
        ctypes.c_int(nao_prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "K_real_buf")
    t3 = t4
    
    K_real_buf *= (ngrid / vol)
    
    K = -ISDF_K._pack_JK(K_real_buf, Ls, nao_prim, output=None) # "-" due to robust fitting
    
    # return -K
    
    ########### do the same thing on V ###########
    
    DM_RgR_complex = np.ndarray((nIP_prim,nGridPrim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_after_dm)
    DM_RgR_real = np.ndarray((nIP_prim,nGridPrim*ncell), dtype=np.float64, buffer=DM_RgR_complex)
    
    offset_now = offset_after_dm + DM_RgR_complex.size * DM_RgR_complex.itemsize
    
    aoR_FFT = mydf.aoR_FFT
    
    offset_A = offset_now
    buf_A = np.ndarray((nao_prim, nao_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_A)
    offset_B = offset_A + buf_A.size * buf_A.itemsize
    buf_B = np.ndarray((nao_prim, nGridPrim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_B)
    offset_B2 = offset_B + buf_B.size * buf_B.itemsize
    buf_B2 = np.ndarray((nao_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_B2)
    offset_C = offset_B2 + buf_B2.size * buf_B2.itemsize
    buf_C = np.ndarray((nao_prim, nGridPrim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_C)
    offset_D = offset_C + buf_C.size * buf_C.itemsize
    buf_D = np.ndarray((nIP_prim, nGridPrim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_D)
    
    if isinstance(aoRg_FFT, list):
        assert isinstance(aoR_FFT, list)
        
        for i in range(ncell_complex):
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            # buf_A[:] = DM_complex[:, k_begin:k_end]
            fn_packcol2(
                buf_A.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_prim),
                ctypes.c_int(2*nao_prim),
                DM_complex.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(DM_complex.shape[0]),
                ctypes.c_int(2*DM_complex.shape[1]),
                ctypes.c_int(2*k_begin),
                ctypes.c_int(2*k_end)
            )
            
            # buf_B[:] = aoR_FFT[:, i*nGridPrim:(i+1)*nGridPrim]
            # buf_B2[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim]
            # buf_B.ravel()[:] = aoR_FFT[i].ravel()[:]
            # buf_B2.ravel()[:] = aoRg_FFT[i].ravel()[:]
            fn_copy(
                buf_B.ctypes.data_as(ctypes.c_void_p),
                aoR_FFT[i].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_size_t(2*buf_B.size) # 2 due to complex number
            )
            fn_copy(
                buf_B2.ctypes.data_as(ctypes.c_void_p),
                aoRg_FFT[i].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_size_t(2*buf_B2.size) # 2 due to complex number
            )

        
            lib.dot(buf_A, buf_B, c=buf_C)
            lib.dot(buf_B2.T.conj(), buf_C, c=buf_D)
        
            k_begin = i * nGridPrim
            k_end   = (i + 1) * nGridPrim
        
            # DM_RgR_complex[:, k_begin:k_end] = buf_D
            fn_packcol3(
                DM_RgR_complex.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(DM_RgR_complex.shape[0]),
                ctypes.c_int(2*DM_RgR_complex.shape[1]),
                ctypes.c_int(2*k_begin),
                ctypes.c_int(2*k_end),
                buf_D.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(buf_D.shape[0]),
                ctypes.c_int(2*buf_D.shape[1]),
            )
    
    else:
        
        raise NotImplementedError("not implemented yet.")
        
        for i in range(ncell_complex):
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            buf_A[:] = DM_complex[:, k_begin:k_end]
            buf_B[:] = aoR_FFT[:, i*nGridPrim:(i+1)*nGridPrim]
            buf_B2[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim]
        
            lib.dot(buf_A, buf_B, c=buf_C)
            lib.dot(buf_B2.T.conj(), buf_C, c=buf_D)
        
            k_begin = i * nGridPrim
            k_end   = (i + 1) * nGridPrim
        
            DM_RgR_complex[:, k_begin:k_end] = buf_D
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "DM_RgR_complex")
    t3 = t4
    
    # DM_RgRg1 = DM_RgR_complex[:, mydf.IP_ID]
    # DM_RgRg2 = DM_RgRg_complex2[:, :nIP_prim]
    # diff = np.linalg.norm(DM_RgRg1 - DM_RgRg2)
    # print("diff = ", diff)
    # for i in range(10):
    #     for j in range(10):
    #         print(DM_RgRg1[i,j], DM_RgRg2[i,j])
    # assert np.allclose(DM_RgRg1, DM_RgRg2)
    
    buf_A = None
    buf_B = None
    buf_B2 = None
    buf_C = None
    buf_D = None
    
    offset_now_fft = offset_now
    
    buf_fft = np.ndarray((nIP_prim, nGridPrim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_now_fft)
    
    fn2(
        DM_RgR_complex.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_prim),
        ctypes.c_int(nGridPrim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
        
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "DM_RgR_real")
    t3 = t4
        
    # inplace multiplication
    
    # print("DM_RgR_complex = ", DM_RgR_complex[:5,:5])
    # print("mydf.V_R       = ", mydf.V_R[:5,:5])
    
    lib.cwise_mul(mydf.V_R, DM_RgR_real, out=DM_RgR_real)
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "cwise_mul")
    t3 = t4
    
    # buf_fft = np.ndarray((nIP_prim, nGridPrim*ncell_complex), dtype=np.complex128)
    
    fn1(
        DM_RgR_real.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_prim),
        ctypes.c_int(nGridPrim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "DM_RgR_complex 2")
    t3 = t4
    
    # print("DM_RgR_complex = ", DM_RgR_complex[:5,:5])
    
    buf_fft = None
    
    offset_K = offset_now
    
    K_complex_buf = np.ndarray((nao_prim, nao_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_K)
    K_real_buf    = np.ndarray((nao_prim, nao_prim*ncell), dtype=np.float64, buffer=K_complex_buf)
    
    offset_after_K = offset_K + K_complex_buf.size * K_complex_buf.itemsize
    
    offset_A = offset_K + K_complex_buf.size * K_complex_buf.itemsize
    buf_A = np.ndarray((nIP_prim, nGridPrim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_A)
    offset_B = offset_A + buf_A.size * buf_A.itemsize
    buf_B = np.ndarray((nao_prim, nGridPrim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_B)
    offset_B2 = offset_B + buf_B.size * buf_B.itemsize
    buf_B2 = np.ndarray((nao_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_B2)
    offset_C = offset_B2 + buf_B2.size * buf_B2.itemsize
    buf_C = np.ndarray((nIP_prim, nao_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_C)
    offset_D = offset_C + buf_C.size * buf_C.itemsize
    buf_D = np.ndarray((nao_prim, nao_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_D)
    
    if isinstance(aoRg_FFT, list):
        
        for i in range(ncell_complex):
        
            k_begin = i * nGridPrim
            k_end   = (i + 1) * nGridPrim
        
            # buf_A.ravel()[:] = DM_RgR_complex[:, k_begin:k_end].ravel()[:]
            fn_packcol2(
                buf_A.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nIP_prim),
                ctypes.c_int(2*nGridPrim),
                DM_RgR_complex.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(DM_RgR_complex.shape[0]),
                ctypes.c_int(2*DM_RgR_complex.shape[1]),
                ctypes.c_int(2*k_begin),
                ctypes.c_int(2*k_end)
            )
            
            # print("buf_A = ", buf_A[:5,:5])
            # buf_B.ravel()[:] = aoR_FFT[:, i*nGridPrim:(i+1)*nGridPrim].ravel()[:]
            # print("buf_B = ", buf_B[:5,:5])
            # buf_B2.ravel()[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim].ravel()[:]  
            # print("buf_B2 = ", buf_B2[:5,:5]) 
            
            # buf_B.ravel()[:] = aoR_FFT[i].ravel()[:]
            # buf_B2.ravel()[:] = aoRg_FFT[i].ravel()[:]
            fn_copy(
                buf_B.ctypes.data_as(ctypes.c_void_p),
                aoR_FFT[i].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_size_t(2*buf_B.size) # 2 due to complex number
            )
            fn_copy(
                buf_B2.ctypes.data_as(ctypes.c_void_p),
                aoRg_FFT[i].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_size_t(2*buf_B2.size) # 2 due to complex number
            )
            
        
            lib.dot(buf_A, buf_B.T.conj(), c=buf_C)
            lib.dot(buf_B2, buf_C, c=buf_D)
        
            # print("buf_D = ", buf_D[:5,:5])
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            # K_complex_buf[:, k_begin:k_end] = buf_D
            fn_packcol3(
                K_complex_buf.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(K_complex_buf.shape[0]),
                ctypes.c_int(2*K_complex_buf.shape[1]),
                ctypes.c_int(2*k_begin),
                ctypes.c_int(2*k_end),
                buf_D.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(buf_D.shape[0]),
                ctypes.c_int(2*buf_D.shape[1]),
            )
        
    else:
        
        raise NotImplementedError("not implemented yet.")
        
        for i in range(ncell_complex):
        
            k_begin = i * nGridPrim
            k_end   = (i + 1) * nGridPrim
        
            buf_A.ravel()[:] = DM_RgR_complex[:, k_begin:k_end].ravel()[:]
            # print("buf_A = ", buf_A[:5,:5])
            buf_B.ravel()[:] = aoR_FFT[:, i*nGridPrim:(i+1)*nGridPrim].ravel()[:]
            # print("buf_B = ", buf_B[:5,:5])
            buf_B2.ravel()[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim].ravel()[:]  
            # print("buf_B2 = ", buf_B2[:5,:5]) 
        
            lib.dot(buf_A, buf_B.T.conj(), c=buf_C)
            lib.dot(buf_B2, buf_C, c=buf_D)
        
            # print("buf_D = ", buf_D[:5,:5])
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            K_complex_buf[:, k_begin:k_end] = buf_D
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "K_complex_buf 1")
    t3 = t4
    
    buf_A = None
    buf_B = None
    buf_B2 = None
    buf_C = None
    buf_D = None
    
    offset_now = offset_after_K
    
    buf_fft = np.ndarray((nao_prim, nao_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_now)
    
   #  print("K_complex_buf = ", K_complex_buf[:5,:5])
    
    fn2(
        K_complex_buf.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao_prim),
        ctypes.c_int(nao_prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "K_complex_buf 2")
    t3 = t4
    
    buf_fft = None
    
    K_real_buf *= (ngrid / vol)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    _benchmark_time(t1, t2, "_contract_k_dm")
    
    t1 = t2
    
    K2 = ISDF_K._pack_JK(K_real_buf, Ls, nao_prim, output=None)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    _benchmark_time(t1, t2, "_pack_JK")
    
    # print("K2 = ", K2[:5,:5])
    # print("K = ", -K[:5,:5])
    
    K += K2 + K2.T
    
    # print("K = ", K[:5,:5])
    
    DM_RgR_complex = None
    DM_RgR_real = None
    
    return K
    
    # return DM_RgRg_real # temporary return for debug


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
                vk = _get_k_kSym_robust_fitting_fast(mydf, dm)
            else:
                vk = ISDF_K._get_k_kSym(mydf, dm)
        # if rank == 0:
        # print("vk = ", vk[0, :16])
        # print("vk = ", vk[0, -16:])
        if exxdiv == 'ewald':
            print("WARNING: ISDF does not support ewald")

    t1 = log.timer('sr jk', *t1)

    return vj, vk