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
from pyscf.lib import logger, zdotNN, zdotCN, zdotNC
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member

#### MPI SUPPORT ####

from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time
from pyscf.pbc.df.isdf.isdf_fast import rank, comm, comm_size, allgather, bcast, reduce, gather, alltoall, _comm_bunch, scatter
import pyscf.pbc.df.isdf.isdf_linear_scaling_base as ISDF_LinearScalingBase

from memory_profiler import profile
import ctypes

libpbc = lib.load_library('libpbc')

##################################################
#
# only Gamma Point
#
##################################################

### ls = linear scaling

def _contract_j_dm_ls(mydf, dm, use_mpi=False):
    
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        raise NotImplementedError("MPI is not supported yet.")
    
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

    aoR  = mydf.aoR
    assert isinstance(aoR, list)
    naux = mydf.naux
    
    #### step 0. allocate buffer 
    
    max_nao_involved = np.max([aoR_holder.aoR.shape[0] for aoR_holder in aoR if aoR_holder is not None])
    max_ngrid_involved = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None])
    ngrids_local = np.sum([aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None])
    
    density_R = np.zeros((ngrids_local,), dtype=np.float64)
    
    dm_buf = np.zeros((max_nao_involved, max_nao_involved), dtype=np.float64)
    ddot_buf = np.zeros((max_nao_involved, max_ngrid_involved), dtype=np.float64)
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
    
    local_grid_loc = 0
    density_R_tmp = None
    
    for aoR_holder in aoR:
        if aoR_holder is None:
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
        
        density_R[local_grid_loc:local_grid_loc+ngrids_now] = density_R_tmp
        local_grid_loc += ngrids_now
    
    assert local_grid_loc == ngrids_local
    
    if use_mpi == False:
        assert ngrids_local == np.prod(mesh)
    
    if use_mpi:
        density_R = np.hstack(gather(density_R, comm, rank, root=0, split_recvbuf=True))
        
    # if hasattr(mydf, "grid_ID_ordered"):
    
    grid_ID_ordered = mydf.grid_ID_ordered
    
    # print("grid_ID_ordered = ", grid_ID_ordered[:64])
    
    if (use_mpi and rank == 0) or (use_mpi == False):
        density_R_original = np.zeros_like(density_R)
            
        fn_order = getattr(libpbc, "_Reorder_Grid_to_Original_Grid", None)
        assert fn_order is not None
            
        fn_order(
            ctypes.c_int(density_R.size),
            mydf.grid_ID_ordered.ctypes.data_as(ctypes.c_void_p),
            density_R.ctypes.data_as(ctypes.c_void_p),
            density_R_original.ctypes.data_as(ctypes.c_void_p),
        )

        density_R = density_R_original.copy()
    
    # print("density_R = ", density_R[:64])
    
    J = None
    
    if (use_mpi and rank == 0) or (use_mpi == False):
    
        fn_J = getattr(libpbc, "_construct_J", None)
        assert(fn_J is not None)

        J = np.zeros_like(density_R)

        fn_J(
            mesh.ctypes.data_as(ctypes.c_void_p),
            density_R.ctypes.data_as(ctypes.c_void_p),
            mydf.coulG.ctypes.data_as(ctypes.c_void_p),
            J.ctypes.data_as(ctypes.c_void_p),
        )
        
        # if hasattr(mydf, "grid_ID_ordered"):
            
        J_ordered = np.zeros_like(J)

        fn_order = getattr(libpbc, "_Original_Grid_to_Reorder_Grid", None)
        assert fn_order is not None 
            
        fn_order(
            ctypes.c_int(J.size),
            mydf.grid_ID_ordered.ctypes.data_as(ctypes.c_void_p),
            J.ctypes.data_as(ctypes.c_void_p),
            J_ordered.ctypes.data_as(ctypes.c_void_p),
        )
            
        J = J_ordered.copy()
            
    if use_mpi:
        grid_segment = mydf.grid_segment
        sendbuf = None
        if rank == 0:
            sendbuf = []
            for i in range(size):
                p0 = grid_segment[i]
                p1 = grid_segment[i+1]
                sendbuf.append(J[p0:p1])
        J = scatter(sendbuf, comm, rank, root=0)
        del sendbuf
        sendbuf = None
    
    #### step 3. get J 
    
    # J = np.asarray(lib.d_ij_j_ij(aoR, J, out=buffer1), order='C') 
    # J = lib.ddot_withbuffer(aoR, J.T, buf=mydf.ddot_buf)

    local_grid_loc = 0

    J_Res = np.zeros((nao, nao), dtype=np.float64)

    for aoR_holder in aoR:
        if aoR_holder is None:
            continue
        
        ngrids_now = aoR_holder.aoR.shape[1]
        nao_involved = aoR_holder.aoR.shape[0]
        
        J_tmp = J[local_grid_loc:local_grid_loc+ngrids_now] 
        
        aoR_J_res = np.ndarray(aoR_holder.aoR.shape, buffer=aoR_buf1)
        lib.d_ij_j_ij(aoR_holder.aoR, J_tmp, out=aoR_J_res)
        ddot_res = np.ndarray((nao_involved, nao_involved), buffer=ddot_buf)
        lib.ddot(aoR_holder.aoR, aoR_J_res.T, c=ddot_res)
        
        if nao_involved == nao:
            J_Res += ddot_res
        else:
            fn_packadd_dm(
                ddot_res.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_involved),
                aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p),
                J_Res.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao)
            )        

        local_grid_loc += ngrids_now
    
    assert local_grid_loc == ngrids_local

    J = J_Res

    if use_mpi:
        J = mpi_reduce(J, comm, rank, op=MPI.SUM, root=0)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    if mydf.verbose:
        _benchmark_time(t1, t2, "_contract_j_dm_fast")
    
    ######### delete the buffer #########
    
    del dm_buf, ddot_buf, density_R
    del density_R_tmp
    del aoR_buf1
    
    return J * ngrid / vol


############# quadratic scaling (not cubic!) #############

def __get_DensityMatrixonGrid_qradratic(dm, bra_aoR_holder, ket_aoR_holder, verbose = 1, use_mpi=False):
    
    t1 = (logger.process_clock(), logger.perf_counter())
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]
    
    assert dm.shape[0] == dm.shape[1]
    nao = dm.shape[0]
    
    ngrid_bra = np.sum([aoR_holder.aoR.shape[1] for aoR_holder in bra_aoR_holder if aoR_holder is not None])
    ngrid_ket = np.sum([aoR_holder.aoR.shape[1] for aoR_holder in ket_aoR_holder if aoR_holder is not None])
    
    max_ngrid_ket = np.max([aoR_holder.aoR.shape[1] for aoR_holder in ket_aoR_holder if aoR_holder is not None])
    
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        raise NotImplementedError("MPI is not supported yet.")

    res = np.zeros((ngrid_bra, ngrid_ket), dtype=np.float64)
    
    ### allocate buf ###
    
    tmp1 = np.zeros((ngrid_bra, dm.shape[0]), dtype=np.float64) 
    ddot_buf = np.zeros((ngrid_bra, max(max_ngrid_ket, dm.shape[0])), dtype=np.float64)
    
    ### perform aoR_bra.T * dm
    
    ngrid_loc = 0
    for aoR_holder in bra_aoR_holder:
        
        if aoR_holder is None:
            continue
        
        ngrid_now = aoR_holder.aoR.shape[1]
        nao_involved = aoR_holder.aoR.shape[0]
        
        dm_packed = dm[aoR_holder.ao_involved, :].copy()
        
        ddot_res = np.ndarray((ngrid_now, nao), buffer=ddot_buf)
        lib.ddot(aoR_holder.aoR.T, dm_packed, c=ddot_res)
        tmp1[ngrid_loc:ngrid_loc+ngrid_now, :] = ddot_res
        
        ngrid_loc += ngrid_now
        
    del dm_packed
    assert ngrid_loc == ngrid_bra
    
    ### perform tmp1 * aoR_ket
    
    ngrid_loc = 0
    for aoR_holder in ket_aoR_holder:
        
        if aoR_holder is None:
            continue
        
        ngrid_now = aoR_holder.aoR.shape[1]
        nao_involved = aoR_holder.aoR.shape[0]
        
        tmp1_packed = tmp1[:, aoR_holder.ao_involved].copy()
        
        ddot_res = np.ndarray((ngrid_bra, ngrid_now), buffer=ddot_buf)
        lib.ddot(tmp1_packed, aoR_holder.aoR, c=ddot_res)
        res[:, ngrid_loc:ngrid_loc+ngrid_now] = ddot_res
        
        ngrid_loc += ngrid_now
    del tmp1_packed
    assert ngrid_loc == ngrid_ket
    
    #### free buffer 
    
    del tmp1
    del ddot_buf
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    if verbose:
        _benchmark_time(t1, t2, "__get_DensityMatrixonGrid_qradratic")
        
    
    return res

def _contract_k_dm_quadratic(mydf, dm, with_robust_fitting=True, use_mpi=False):
    
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    
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
    
    aoRg = mydf.aoRg
    assert isinstance(aoRg, list)
    aoR = mydf.aoR
    assert isinstance(aoR, list)
    
    naux = mydf.naux
    nao = cell.nao
    
    #### step 0. allocate buffer
    
    max_nao_involved = np.max([aoR_holder.aoR.shape[0] for aoR_holder in aoR if aoR_holder is not None])
    max_ngrid_involved = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None])
    ddot_res_buf = np.zeros((naux, max_nao_involved), dtype=np.float64)
    
    #### step 1. get density matrix value on real space grid and IPs
    
    Density_RgRg = __get_DensityMatrixonGrid_qradratic(dm, aoRg, aoRg, use_mpi)
    if with_robust_fitting:
        Density_RgR  = __get_DensityMatrixonGrid_qradratic(dm, aoRg, aoR, use_mpi)
    else:
        Density_RgR = None
    
    # print("density_RgRg.shape = ", Density_RgRg.shape)
    # print("Density_RgRg = ", Density_RgRg[0, :16])
    # if with_robust_fitting:
    #     print("Density_RgR.shape = ", Density_RgR.shape)
    #     print("Density_RgR = ", Density_RgR[0, :16])
    
    #### step 2. get K, those part which W is involved 
    
    W = mydf.W
    assert W is not None
    assert isinstance(W, np.ndarray)
    
    lib.cwise_mul(W, Density_RgRg, out=Density_RgRg)
    
    K1 = np.zeros((naux, nao), dtype=np.float64)
    
    ### TODO: consider MPI 
    
    nIP_loc = 0
    for aoRg_holder in aoRg:
        
        if aoRg_holder is None:
            continue
    
        nIP_now = aoRg_holder.aoR.shape[1]
        nao_invovled = aoRg_holder.aoR.shape[0]
        
        W_tmp = Density_RgRg[:, nIP_loc:nIP_loc+nIP_now].copy()
        
        ddot_res = np.ndarray((naux, nao_invovled), buffer=ddot_res_buf)
        lib.ddot(W_tmp, aoRg_holder.aoR.T, c=ddot_res)
        
        K1[: , aoRg_holder.ao_involved] += ddot_res

        nIP_loc += nIP_now
    del W_tmp
    assert nIP_loc == naux
    
    K = np.zeros((nao, nao), dtype=np.float64) 
    
    nIP_loc = 0
    for aoRg_holder in aoRg:
        
        if aoRg_holder is None:
            continue
    
        nIP_now = aoRg_holder.aoR.shape[1]
        nao_invovled = aoRg_holder.aoR.shape[0]
        
        K_tmp = K1[nIP_loc:nIP_loc+nIP_now, :].copy()
        
        ddot_res = np.ndarray((nao_invovled, nao), buffer=ddot_res_buf)
        # lib.ddot(K_tmp, aoRg_holder.ao.T, c=ddot_res)
        lib.ddot(aoRg_holder.aoR, K_tmp, c=ddot_res)
        
        K[aoRg_holder.ao_involved, :] += ddot_res
        
        nIP_loc += nIP_now
    del K_tmp
    assert nIP_loc == naux
    
    
    #### step 3. get K, those part which W is not involved, with robust fitting
    
    if with_robust_fitting:
        K = -K
        
        ### calcualte those parts where V is involved 
        
        V_R = mydf.V_R
        assert V_R is not None
        assert isinstance(V_R, np.ndarray)
        
        lib.cwise_mul(V_R, Density_RgR, out=Density_RgR)
        
        K2 = K1
        K2.ravel()[:] = 0.0    
    
        ngrid_loc = 0
        for aoR_holder in aoR:
            
            if aoR_holder is None:
                continue
            
            ngrid_now = aoR_holder.aoR.shape[1]
            nao_invovled = aoR_holder.aoR.shape[0]
            
            V_tmp = Density_RgR[:, ngrid_loc:ngrid_loc+ngrid_now].copy()
            
            ddot_res = np.ndarray((naux, nao_invovled), buffer=ddot_res_buf)
            lib.ddot(V_tmp, aoR_holder.aoR.T, c=ddot_res)
            
            K2[: , aoR_holder.ao_involved] += ddot_res
            
            ngrid_loc += ngrid_now
        del V_tmp

        assert ngrid_loc == ngrid
        
        K_add = np.zeros((nao, nao), dtype=np.float64)
        
        nIP_loc = 0
        for aoRg_holder in aoRg:
            
            if aoRg_holder is None:
                continue
        
            nIP_now = aoRg_holder.aoR.shape[1]
            nao_invovled = aoRg_holder.aoR.shape[0]
            
            K_tmp = K2[nIP_loc:nIP_loc+nIP_now, :].copy()
            
            ddot_res = np.ndarray((nao_invovled, nao), buffer=ddot_res_buf)
            lib.ddot(aoRg_holder.aoR, K_tmp, c=ddot_res)
            
            K_add[aoRg_holder.ao_involved, :] += ddot_res
            
            nIP_loc += nIP_now
        del K_tmp
        assert nIP_loc == naux
        
        K_add += K_add.T
        
        K += K_add
    
    ######### finally delete the buffer #########
    
    del Density_RgRg
    del Density_RgR
    del ddot_res_buf
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    if mydf.verbose:
        _benchmark_time(t1, t2, "_contract_k_dm_quadratic")
    
    return K * ngrid / vol
    

def get_jk_dm_quadratic(mydf, dm, hermi=1, kpt=np.zeros(3),
                        kpts_band=None, with_j=True, with_k=True, omega=None, 
                        use_mpi = False, **kwargs):
    
    '''JK for given k-point'''
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    if hasattr(mydf, 'Ls'):
        from pyscf.pbc.df.isdf.isdf_k import _symmetrize_dm
        dm = _symmetrize_dm(dm, mydf.Ls)

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
        vj = _contract_j_dm_ls(mydf, dm, use_mpi)  
        # print("vj = ", vj[0, :16])
    if with_k:
        # if mydf.with_robust_fitting:
        #     vk = _contract_k_dm(mydf, dm, mydf.with_robust_fitting, use_mpi)
        # else:
        #     vk = _contract_k_dm_wo_robust_fitting(mydf, dm, mydf.with_robust_fitting, use_mpi)
        vk = _contract_k_dm_quadratic(mydf, dm, mydf.with_robust_fitting, use_mpi=use_mpi)
        # print("vk = ", vk[0, :16])
        if exxdiv == 'ewald':
            print("WARNING: ISDF does not support ewald")

    t1 = log.timer('sr jk', *t1)

    return vj, vk


############# linear scaling implementation ############# 