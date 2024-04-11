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
from pyscf.pbc.df.isdf.isdf_tools_mpi import rank, comm, comm_size, allgather, bcast, reduce, gather, alltoall, _comm_bunch, scatter

# from memory_profiler import profile
import ctypes
from profilehooks import profile

libpbc = lib.load_library('libpbc')

##################################################
#
# only Gamma Point
#
##################################################

### ls = linear scaling

def _half_J(mydf, dm, use_mpi=False):
    
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        #raise NotImplementedError("MPI is not supported yet.")
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

    aoR  = mydf.aoR
    assert isinstance(aoR, list)
    naux = mydf.naux
    
    #### step 0. allocate buffer 
    
    max_nao_involved = np.max([aoR_holder.aoR.shape[0] for aoR_holder in aoR if aoR_holder is not None])
    max_ngrid_involved = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None])
    ngrids_local = np.sum([aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None])
    
    # density_R = np.zeros((ngrids_local,), dtype=np.float64)
    density_R = np.zeros((ngrid,), dtype=np.float64)
    
    dm_buf = np.zeros((max_nao_involved, max_nao_involved), dtype=np.float64)
    max_dim_buf = max(max_ngrid_involved, max_nao_involved)
    ddot_buf = np.zeros((max_dim_buf, max_dim_buf), dtype=np.float64)
    # aoR_buf1 = np.zeros((max_nao_involved, max_ngrid_involved), dtype=np.float64)
    
    ##### get the involved C function ##### 
    
    fn_extract_dm = getattr(libpbc, "_extract_dm_involved_ao", None) 
    assert fn_extract_dm is not None
    
    fn_packadd_dm = getattr(libpbc, "_packadd_local_dm", None)
    assert fn_packadd_dm is not None
    
    #### step 1. get density value on real space grid and IPs
    
    # lib.ddot(dm, aoR, c=buffer1) 
    # tmp1 = buffer1
    # density_R = np.asarray(lib.multiply_sum_isdf(aoR, tmp1, out=buffer2), order='C')
    
    group = mydf.group
    ngroup = len(group)
    
    # local_grid_loc = 0
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
        
        density_R[global_gridID_begin:global_gridID_begin+ngrids_now] = density_R_tmp
        # local_grid_loc += ngrids_now
    
    # assert local_grid_loc == ngrids_local
    
    if use_mpi == False:
        assert ngrids_local == np.prod(mesh)
    
    if use_mpi:
        # density_R = np.hstack(gather(density_R, comm, rank, root=0, split_recvbuf=True))
        density_R = reduce(density_R, root=0)
        
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

        if hasattr(mydf, "coulG") == False:
            mydf.coulG = tools.get_coulG(cell, mesh=mesh)

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
        J = bcast(J, root=0)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    del dm_buf, ddot_buf, density_R
    del density_R_tmp
    
    if mydf.verbose:
        _benchmark_time(t1, t2, "half_J")
    
    return J

# @profile
def _contract_j_dm_ls(mydf, dm, use_mpi=False):
    
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        #raise NotImplementedError("MPI is not supported yet.")
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

    aoR  = mydf.aoR
    assert isinstance(aoR, list)
    naux = mydf.naux
    
    #### step 0. allocate buffer 
    
    max_nao_involved = np.max([aoR_holder.aoR.shape[0] for aoR_holder in aoR if aoR_holder is not None])
    max_ngrid_involved = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None])
    ngrids_local = np.sum([aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None])
    
    density_R = np.zeros((ngrid,), dtype=np.float64)
    
    max_dim_buf = max(max_ngrid_involved, max_nao_involved)
    ddot_buf = np.zeros((max_dim_buf, max_dim_buf), dtype=np.float64)
    aoR_buf1 = np.zeros((max_nao_involved, max_ngrid_involved), dtype=np.float64)
    
    ##### get the involved C function ##### 
    
    fn_extract_dm = getattr(libpbc, "_extract_dm_involved_ao", None) 
    assert fn_extract_dm is not None
    
    fn_packadd_dm = getattr(libpbc, "_packadd_local_dm", None)
    assert fn_packadd_dm is not None
    
    #### step 1 2. get density value on real space grid and IPs
    
    group = mydf.group
    ngroup = len(group)

    J = _half_J(mydf, dm, use_mpi)

    #### step 3. get J 
    
    # J = np.asarray(lib.d_ij_j_ij(aoR, J, out=buffer1), order='C') 
    # J = lib.ddot_withbuffer(aoR, J.T, buf=mydf.ddot_buf)

    # local_grid_loc = 0

    J_Res = np.zeros((nao, nao), dtype=np.float64)

    for atm_id, aoR_holder in enumerate(aoR):
        
        if aoR_holder is None:
            continue
        
        if use_mpi:
            if atm_id % comm_size != rank:
                continue
        
        ngrids_now = aoR_holder.aoR.shape[1]
        nao_involved = aoR_holder.aoR.shape[0]
        
        global_gridID_begin = aoR_holder.global_gridID_begin
        
        J_tmp = J[global_gridID_begin:global_gridID_begin+ngrids_now] 
        
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

        # local_grid_loc += ngrids_now
    
    # assert local_grid_loc == ngrids_local

    J = J_Res

    if use_mpi:
        # J = mpi_reduce(J, comm, rank, op=MPI.SUM, root=0)
        J = reduce(J, root=0)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    if mydf.verbose:
        _benchmark_time(t1, t2, "_contract_j_dm_fast")
    
    ######### delete the buffer #########
    
    # del dm_buf, 
    del ddot_buf 
    # density_R
    # del density_R_tmp
    del aoR_buf1
    
    return J * ngrid / vol

def _contract_j_dm_wo_robust_fitting(mydf, dm, use_mpi=False):
    
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        #raise NotImplementedError("MPI is not supported yet.")
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

    aoRg  = mydf.aoRg
    assert isinstance(aoRg, list)
    naux = mydf.naux
    W = mydf.W
    
    #### step 0. allocate buffer 
    
    max_nao_involved = np.max([aoR_holder.aoR.shape[0] for aoR_holder in aoRg if aoR_holder is not None])
    max_ngrid_involved = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoRg if aoR_holder is not None])
    ngrids_local = np.sum([aoR_holder.aoR.shape[1] for aoR_holder in aoRg if aoR_holder is not None])
    
    # density_R = np.zeros((ngrids_local,), dtype=np.float64)
    density_Rg = np.zeros((naux,), dtype=np.float64)
    
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
    
    group = mydf.group
    ngroup = len(group)
    
    # local_grid_loc = 0
    density_R_tmp = None
    
    for atm_id, aoR_holder in enumerate(aoRg):
        
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
        
        density_Rg[global_gridID_begin:global_gridID_begin+ngrids_now] = density_R_tmp
        # local_grid_loc += ngrids_now
    
    # assert local_grid_loc == ngrids_local
    
    if use_mpi == False:
        assert ngrids_local == naux
    
    if use_mpi:
        # density_R = np.hstack(gather(density_R, comm, rank, root=0, split_recvbuf=True))
        density_Rg = reduce(density_Rg, root=0)
        
            
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
    
    #### step 3. get J 
    
    J = np.asarray(lib.dot(W, density_Rg.reshape(-1,1)), order='C').reshape(-1)
    
    # J = np.asarray(lib.d_ij_j_ij(aoR, J, out=buffer1), order='C') 
    # J = lib.ddot_withbuffer(aoR, J.T, buf=mydf.ddot_buf)

    # local_grid_loc = 0

    J_Res = np.zeros((nao, nao), dtype=np.float64)

    for aoR_holder in aoRg:
        
        if aoR_holder is None:
            continue
        
        if use_mpi:
            if atm_id % comm_size != rank:
                continue
        
        ngrids_now = aoR_holder.aoR.shape[1]
        nao_involved = aoR_holder.aoR.shape[0]
        
        global_gridID_begin = aoR_holder.global_gridID_begin
        
        J_tmp = J[global_gridID_begin:global_gridID_begin+ngrids_now] 
        
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

        # local_grid_loc += ngrids_now
    
    # assert local_grid_loc == ngrids_local

    J = J_Res

    if use_mpi:
        # J = mpi_reduce(J, comm, rank, op=MPI.SUM, root=0)
        J = reduce(J, root=0)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    if mydf.verbose:
        _benchmark_time(t1, t2, "_contract_j_dm_fast")
    
    ######### delete the buffer #########
    
    del dm_buf, ddot_buf, density_Rg
    del density_R_tmp
    del aoR_buf1
    
    return J * ngrid / vol


############# quadratic scaling (not cubic!) #############

# @profile
def __get_DensityMatrixonGrid_qradratic(mydf, dm, bra_aoR_holder, ket_aoR_holder, res:np.ndarray=None, verbose = 1, use_mpi=False):
    
    t1 = (logger.process_clock(), logger.perf_counter())
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]
    
    assert dm.shape[0] == dm.shape[1]
    nao = dm.shape[0]
    
    ngrid_bra = np.sum([aoR_holder.aoR.shape[1] for aoR_holder in bra_aoR_holder if aoR_holder is not None])
    ngrid_ket = np.sum([aoR_holder.aoR.shape[1] for aoR_holder in ket_aoR_holder if aoR_holder is not None])
    
    max_ngrid_ket = np.max([aoR_holder.aoR.shape[1] for aoR_holder in ket_aoR_holder if aoR_holder is not None])
    max_ao_involved = np.max([aoR_holder.aoR.shape[0] for aoR_holder in ket_aoR_holder if aoR_holder is not None])
    
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        raise NotImplementedError("MPI is not supported yet.")

    if res is None:
        res = np.zeros((ngrid_bra, ngrid_ket), dtype=np.float64)
    else:
        assert res.ndim == 2
        assert res.shape[0] == ngrid_bra
        assert res.shape[1] == ngrid_ket
    
    ### allocate buf ###
    
    # tmp1 = np.zeros((ngrid_bra, dm.shape[0]), dtype=np.float64) 
    tmp1 = np.ndarray((ngrid_bra, dm.shape[0]), buffer=mydf.build_k_buf)
    offset = tmp1.size * tmp1.dtype.itemsize
    # ddot_buf = np.zeros((ngrid_bra, max(max_ngrid_ket, dm.shape[0])), dtype=np.float64)
    ddot_buf = np.ndarray((ngrid_bra, max(max_ngrid_ket, dm.shape[0])), buffer=mydf.build_k_buf, offset=offset)
    offset += ddot_buf.size * ddot_buf.dtype.itemsize
    dm_pack_buf = np.ndarray((dm.shape[0], dm.shape[1]), buffer=mydf.build_k_buf, offset=offset)
    offset += dm_pack_buf.size * dm_pack_buf.dtype.itemsize
    tmp1_pack_buf = np.ndarray((ngrid_bra, max_ao_involved), buffer=mydf.build_k_buf, offset=offset)
    
    nao = dm.shape[0]
    
    ### get pack fn ### 
    
    fn_packrow = getattr(libpbc, "_buildK_packrow", None)
    assert fn_packrow is not None
    fn_packcol = getattr(libpbc, "_buildK_packcol", None)
    assert fn_packcol is not None
    
    ### perform aoR_bra.T * dm
    
    ngrid_loc = 0
    for aoR_holder in bra_aoR_holder:
        
        if aoR_holder is None:
            continue
        
        ngrid_now = aoR_holder.aoR.shape[1]
        nao_involved = aoR_holder.aoR.shape[0]
        
        if nao_involved == nao:
            dm_packed = dm
        else:
            # dm_packed = dm[aoR_holder.ao_involved, :]
            dm_packed = np.ndarray((nao_involved, nao), buffer=dm_pack_buf)
            fn_packrow(
                dm_packed.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_involved),
                ctypes.c_int(nao),
                dm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao),
                ctypes.c_int(nao),
                aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
            )
        
        ddot_res = np.ndarray((ngrid_now, nao), buffer=ddot_buf)
        lib.ddot(aoR_holder.aoR.T, dm_packed, c=ddot_res)
        tmp1[ngrid_loc:ngrid_loc+ngrid_now, :] = ddot_res
        
        ngrid_loc += ngrid_now
        
    # del dm_packed
    assert ngrid_loc == ngrid_bra
    
    ### perform tmp1 * aoR_ket
    
    ngrid_loc = 0
    for aoR_holder in ket_aoR_holder:
        
        if aoR_holder is None:
            continue
        
        ngrid_now = aoR_holder.aoR.shape[1]
        nao_involved = aoR_holder.aoR.shape[0]
        
        if nao_involved == nao:
            tmp1_packed = tmp1
        else:
            # tmp1_packed = tmp1[:, aoR_holder.ao_involved]
            tmp1_packed = np.ndarray((ngrid_bra, nao_involved), buffer=tmp1_pack_buf)
            fn_packcol(
                tmp1_packed.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(ngrid_bra),
                ctypes.c_int(nao_involved),
                tmp1.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(ngrid_bra),
                ctypes.c_int(nao),
                aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
            )
        
        ddot_res = np.ndarray((ngrid_bra, ngrid_now), buffer=ddot_buf)
        lib.ddot(tmp1_packed, aoR_holder.aoR, c=ddot_res)
        res[:, ngrid_loc:ngrid_loc+ngrid_now] = ddot_res
        
        ngrid_loc += ngrid_now
    # del tmp1_packed
    assert ngrid_loc == ngrid_ket
    
    t2 = (logger.process_clock(), logger.perf_counter())
    if verbose>0:
        _benchmark_time(t1, t2, "__get_DensityMatrixonGrid_qradratic")
    return res

def __get_DensityMatrixonRgAO_qradratic(mydf, dm, bra_aoR_holder, res:np.ndarray=None, verbose = 1):
    
    t1 = (logger.process_clock(), logger.perf_counter())
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]
    
    assert dm.shape[0] == dm.shape[1]
    nao = dm.shape[0]
    
    ngrid_bra = np.sum([aoR_holder.aoR.shape[1] for aoR_holder in bra_aoR_holder if aoR_holder is not None])
    
    max_ngrid_bra = np.max([aoR_holder.aoR.shape[1] for aoR_holder in bra_aoR_holder if aoR_holder is not None])
    max_ao_involved = np.max([aoR_holder.aoR.shape[0] for aoR_holder in bra_aoR_holder if aoR_holder is not None])
    
    # if use_mpi:
    #     from mpi4py import MPI
    #     comm = MPI.COMM_WORLD
    #     rank = comm.Get_rank()
    #     size = comm.Get_size()
    #     raise NotImplementedError("MPI is not supported yet.")

    if res is None:
        res = np.zeros((ngrid_bra, nao), dtype=np.float64)
    else:
        assert res.ndim == 2
        assert res.shape[0] == ngrid_bra
        assert res.shape[1] == nao
    
    ### allocate buf ###
    
    offset = 0
    ddot_buf = np.ndarray((max_ngrid_bra, nao), buffer=mydf.build_k_buf, offset=offset)
    offset  += ddot_buf.size * ddot_buf.dtype.itemsize
    dm_pack_buf = np.ndarray((dm.shape[0], dm.shape[1]), buffer=mydf.build_k_buf, offset=offset)
        
    ### get pack fn ### 
    
    fn_packrow = getattr(libpbc, "_buildK_packrow", None)
    assert fn_packrow is not None
    fn_packcol = getattr(libpbc, "_buildK_packcol", None)
    assert fn_packcol is not None
    
    ### perform aoR_bra.T * dm
    
    ngrid_loc = 0
    for aoR_holder in bra_aoR_holder:
        
        if aoR_holder is None:
            continue
        
        ngrid_now = aoR_holder.aoR.shape[1]
        nao_involved = aoR_holder.aoR.shape[0]
        
        if nao_involved == nao:
            dm_packed = dm
        else:
            dm_packed = np.ndarray((nao_involved, nao), buffer=dm_pack_buf)
            fn_packrow(
                dm_packed.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_involved),
                ctypes.c_int(nao),
                dm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao),
                ctypes.c_int(nao),
                aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
            )
        
        ddot_res = np.ndarray((ngrid_now, nao), buffer=ddot_buf)
        lib.ddot(aoR_holder.aoR.T, dm_packed, c=ddot_res)
        res[ngrid_loc:ngrid_loc+ngrid_now, :] = ddot_res
        
        ngrid_loc += ngrid_now
        
    assert ngrid_loc == ngrid_bra
        
    t2 = (logger.process_clock(), logger.perf_counter())
    # if verbose>0:
    #     _benchmark_time(t1, t2, "__get_DensityMatrixonRgAO_qradratic")
    return res

# @profile
def _contract_k_dm_quadratic(mydf, dm, with_robust_fitting=True, use_mpi=False):
    
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
    
    aoRg = mydf.aoRg
    assert isinstance(aoRg, list)
    aoR = mydf.aoR
    assert isinstance(aoR, list)
    
    naux = mydf.naux
    nao = cell.nao
    
    #### step 0. allocate buffer
    
    max_nao_involved = np.max([aoR_holder.aoR.shape[0] for aoR_holder in aoR if aoR_holder is not None])
    max_ngrid_involved = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None])
    max_nIP_involved = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoRg if aoR_holder is not None])
    
    mydf.allocate_k_buffer()
    
    # ddot_res_buf = np.zeros((naux, max_nao_involved), dtype=np.float64)
    ddot_res_buf = mydf.build_k_buf
    
    ##### get the involved C function ##### 
    
    fn_packadd_row = getattr(libpbc, "_buildK_packaddrow", None)
    assert fn_packadd_row is not None
    fn_packadd_col = getattr(libpbc, "_buildK_packaddcol", None)
    assert fn_packadd_col is not None
    
    fn_packcol1 = getattr(libpbc, "_buildK_packcol", None)
    fn_packcol2 = getattr(libpbc, "_buildK_packcol2", None)
    assert fn_packcol1 is not None
    assert fn_packcol2 is not None
    
    #### step 1. get density matrix value on real space grid and IPs
    
    Density_RgAO = __get_DensityMatrixonRgAO_qradratic(mydf, dm, aoRg, mydf.Density_RgAO_buf, use_mpi)
    
    # print("Density_RgAO = ", Density_RgAO[0, :16])
    
    # if with_robust_fitting:
    #     Density_RgR  = __get_DensityMatrixonGrid_qradratic(mydf, dm, aoRg, aoR, mydf.Density_RgR_buf, use_mpi)
    # else:
    #     Density_RgR = None
    
    #### step 2. get K, those part which W is involved 
    
    W = mydf.W
    assert W is not None
    assert isinstance(W, np.ndarray)
        
    K1 = np.zeros((naux, nao), dtype=np.float64)
    
    ####### buf for the first loop #######
    
    offset = 0
    ddot_buf1 = np.ndarray((naux, max_nIP_involved), buffer=ddot_res_buf, offset=offset, dtype=np.float64)
    offset = ddot_buf1.size * ddot_res_buf.dtype.itemsize
    pack_buf = np.ndarray((naux, max_nao_involved), buffer=ddot_res_buf, offset=offset, dtype=np.float64)
    offset+= pack_buf.size * pack_buf.dtype.itemsize
    ddot_buf2 = np.ndarray((naux, max(max_nIP_involved, max_nao_involved)), buffer=ddot_res_buf, offset=offset, dtype=np.float64)
    
    ### TODO: consider MPI 
    
    nIP_loc = 0
    for aoRg_holder in aoRg:
        
        if aoRg_holder is None:
            continue
    
        nIP_now = aoRg_holder.aoR.shape[1]
        nao_invovled = aoRg_holder.aoR.shape[0]
        
        #### pack the density matrix ####
        
        if nao_invovled == nao:
            Density_RgAO_packed = Density_RgAO
        else:
            # Density_RgAO_packed = Density_RgAO[:, aoRg_holder.ao_involved]
            Density_RgAO_packed = np.ndarray((naux, nao_invovled), buffer=pack_buf)
            fn_packcol1(
                Density_RgAO_packed.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(naux),
                ctypes.c_int(nao_invovled),
                Density_RgAO.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(naux),
                ctypes.c_int(nao),
                aoRg_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
            )
        
        # W_tmp = Density_RgRg[:, nIP_loc:nIP_loc+nIP_now] * W[:, nIP_loc:nIP_loc+nIP_now]
        
        ddot_res1 = np.ndarray((naux, nIP_now), buffer=ddot_buf1)
        lib.ddot(Density_RgAO_packed, aoRg_holder.aoR, c=ddot_res1)
        Density_RgRg = ddot_res1
        W_packed = np.ndarray((naux, nIP_now), buffer=ddot_buf2)
        fn_packcol2(
            W_packed.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(naux),
            ctypes.c_int(nIP_now),
            W.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(naux),
            ctypes.c_int(naux),
            ctypes.c_int(nIP_loc),
            ctypes.c_int(nIP_loc+nIP_now)
        )
        lib.cwise_mul(W_packed, Density_RgRg, out=Density_RgRg)
        W_tmp = Density_RgRg

        # ddot
        
        ddot_res = np.ndarray((naux, nao_invovled), buffer=ddot_buf2)
        lib.ddot(W_tmp, aoRg_holder.aoR.T, c=ddot_res)
        
        if nao_invovled == nao:
            K1 += ddot_res
        else:
            # K1[: , aoRg_holder.ao_involved] += ddot_res
            fn_packadd_col(
                K1.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(K1.shape[0]),
                ctypes.c_int(K1.shape[1]),
                ddot_res.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(ddot_res.shape[0]),
                ctypes.c_int(ddot_res.shape[1]),
                aoRg_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
            )

        nIP_loc += nIP_now
    # del W_tmp
    assert nIP_loc == naux
        
    K = np.zeros((nao, nao), dtype=np.float64) 
    
    nIP_loc = 0
    for aoRg_holder in aoRg:
        
        if aoRg_holder is None:
            continue
    
        nIP_now = aoRg_holder.aoR.shape[1]
        nao_invovled = aoRg_holder.aoR.shape[0]
        
        K_tmp = K1[nIP_loc:nIP_loc+nIP_now, :]
        
        ddot_res = np.ndarray((nao_invovled, nao), buffer=ddot_res_buf)
        lib.ddot(aoRg_holder.aoR, K_tmp, c=ddot_res)
        
        if nao_invovled == nao:
            K += ddot_res
        else:
            # K[aoRg_holder.ao_involved, :] += ddot_res 
            fn_packadd_row(
                K.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(K.shape[0]),
                ctypes.c_int(K.shape[1]),
                ddot_res.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(ddot_res.shape[0]),
                ctypes.c_int(ddot_res.shape[1]),
                aoRg_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
            )
        
        nIP_loc += nIP_now
    # del K_tmp
    assert nIP_loc == naux
    
    #### step 3. get K, those part which W is not involved, with robust fitting
    
    if with_robust_fitting:
        
        K = -K
        
        ### calcualte those parts where V is involved 
        
        V_R = mydf.V_R
        assert V_R is not None
        assert isinstance(V_R, np.ndarray)
        
        # lib.cwise_mul(V_R, Density_RgR, out=Density_RgR)
        
        K2 = K1
        K2.ravel()[:] = 0.0    
    
        # fn_packcol = getattr(libpbc, "_buildK_packcol2", None)
        # assert fn_packcol is not None

        ddot_buf1 = np.ndarray((naux, max_nao_involved), buffer=ddot_res_buf)
        offset = naux * max_nao_involved * ddot_res_buf.dtype.itemsize
        V_tmp_buf = np.ndarray((naux, max_ngrid_involved), buffer=ddot_res_buf, offset=offset)
        offset += V_tmp_buf.size * V_tmp_buf.dtype.itemsize
        pack_buf = np.ndarray((naux, max_nao_involved), buffer=ddot_res_buf, offset=offset)
        offset += pack_buf.size * pack_buf.dtype.itemsize
        ddot_buf2 = np.ndarray((naux, max_ngrid_involved), buffer=ddot_res_buf, offset=offset)
    
        ngrid_loc = 0
        for aoR_holder in aoR:
            
            if aoR_holder is None:
                continue
            
            ngrid_now = aoR_holder.aoR.shape[1]
            nao_invovled = aoR_holder.aoR.shape[0]
            
            #### pack the density matrix ####
            
            if nao_invovled == nao:
                Density_RgAO_packed = Density_RgAO_packed
            else:
                # Density_RgAO_packed = Density_RgAO[:, aoR_holder.ao_involved]
                Density_RgAO_packed = np.ndarray((naux, nao_invovled), buffer=pack_buf)
                fn_packcol1(
                    Density_RgAO_packed.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(naux),
                    ctypes.c_int(nao_invovled),
                    Density_RgAO.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(naux),
                    ctypes.c_int(nao),
                    aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
                )
            
            # V_tmp = Density_RgR[:, ngrid_loc:ngrid_loc+ngrid_now] * V_R[:, ngrid_loc:ngrid_loc+ngrid_now]
            
            ddot_res2 = np.ndarray((naux, ngrid_now), buffer=ddot_buf2)
            lib.ddot(Density_RgAO_packed, aoR_holder.aoR, c=ddot_res2)
            Density_RgR = ddot_res2
            V_packed = np.ndarray((naux, ngrid_now), buffer=V_tmp_buf)
            fn_packcol2(
                V_packed.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(naux),
                ctypes.c_int(ngrid_now),
                V_R.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(naux),
                ctypes.c_int(ngrid),
                ctypes.c_int(ngrid_loc),
                ctypes.c_int(ngrid_loc+ngrid_now)
            )
            lib.cwise_mul(V_packed, Density_RgR, out=Density_RgR)
            V_tmp = Density_RgR
            
            # V_tmp = Density_RgR[:, ngrid_loc:ngrid_loc+ngrid_now]
            # V_tmp = np.ndarray((naux, ngrid_now), buffer=V_tmp_buf)
            # fn_packcol2(
            #     V_tmp.ctypes.data_as(ctypes.c_void_p),
            #     ctypes.c_int(naux),
            #     ctypes.c_int(ngrid_now),
            #     Density_RgR.ctypes.data_as(ctypes.c_void_p),
            #     ctypes.c_int(naux),
            #     ctypes.c_int(ngrid),
            #     ctypes.c_int(ngrid_loc),
            #     ctypes.c_int(ngrid_loc+ngrid_now)
            # )
            
            ddot_res = np.ndarray((naux, nao_invovled), buffer=ddot_buf1)
            lib.ddot(V_tmp, aoR_holder.aoR.T, c=ddot_res)
            
            if nao_invovled == nao:
                K2 += ddot_res
            else:
                # K2[: , aoR_holder.ao_involved] += ddot_res 
                fn_packadd_col(
                    K2.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(K2.shape[0]),
                    ctypes.c_int(K2.shape[1]),
                    ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(ddot_res.shape[0]),
                    ctypes.c_int(ddot_res.shape[1]),
                    aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
                )
            
            ngrid_loc += ngrid_now
        # del V_tmp

        assert ngrid_loc == ngrid
        
        K_add = np.zeros((nao, nao), dtype=np.float64)
        
        nIP_loc = 0
        for aoRg_holder in aoRg:
            
            if aoRg_holder is None:
                continue
        
            nIP_now = aoRg_holder.aoR.shape[1]
            nao_invovled = aoRg_holder.aoR.shape[0]
            
            K_tmp = K2[nIP_loc:nIP_loc+nIP_now, :] # no need to pack, continguous anyway
            
            ddot_res = np.ndarray((nao_invovled, nao), buffer=ddot_res_buf)
            lib.ddot(aoRg_holder.aoR, K_tmp, c=ddot_res)
            
            if nao == nao_invovled:
                K_add += ddot_res
            else:
                # K_add[aoRg_holder.ao_involved, :] += ddot_res 
                fn_packadd_row(
                    K_add.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(K_add.shape[0]),
                    ctypes.c_int(K_add.shape[1]),
                    ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(ddot_res.shape[0]),
                    ctypes.c_int(ddot_res.shape[1]),
                    aoRg_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
                )
            
            nIP_loc += nIP_now
        # del K_tmp
        assert nIP_loc == naux
        
        K_add += K_add.T
        
        K += K_add
    
    ######### finally delete the buffer #########
    
    # del Density_RgRg
    # del Density_RgR
    # del ddot_res_buf
    
    del K1
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    if mydf.verbose:
        _benchmark_time(t1, t2, "_contract_k_dm_quadratic")
    
    return K * ngrid / vol


def _contract_k_dm_quadratic_direct(mydf, dm, use_mpi=False):
    
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    
    t1 = (logger.process_clock(), logger.perf_counter())
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]
        
    aoR = mydf.aoR
    aoRg = mydf.aoRg    
    
    max_nao_involved = np.max([aoR_holder.aoR.shape[0] for aoR_holder in aoR if aoR_holder is not None])
    max_ngrid_involved = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None])
    max_nIP_involved = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoRg if aoR_holder is not None])
    
    maxsize_group_naux = mydf._get_maxsize_group_naux()
        
    ####### preparing the data #######
        
    nao  = dm.shape[0]
    cell = mydf.cell
    assert cell.nao == nao
    vol = cell.vol
    mesh = np.array(cell.mesh, dtype=np.int32)
    mesh_int32 = mesh
    ngrid = np.prod(mesh)
    
    aoRg = mydf.aoRg
    assert isinstance(aoRg, list)
    aoR = mydf.aoR
    assert isinstance(aoR, list)
    
    naux = mydf.naux
    nao = cell.nao
    aux_basis = mydf.aux_basis
    
    grid_ordering = mydf.grid_ID_ordered 
    if hasattr(mydf, "coulG") == False:
        mydf.coulG = tools.get_coulG(cell, mesh=mesh)
    coulG = mydf.coulG
    coulG_real = coulG.reshape(*mesh)[:, :, :mesh[2]//2+1].reshape(-1).copy()
    
    mydf.allocate_k_buffer()
    build_k_buf = mydf.build_k_buf
    build_VW_buf = mydf.build_VW_in_k_buf
    
    group = mydf.group
    assert len(group) == len(aux_basis)
    
    if hasattr(mydf, "atm_ordering"):
        atm_ordering = mydf.atm_ordering
    else:
        atm_ordering = []
        for group_idx, atm_idx in enumerate(group):
            atm_idx.sort()
            atm_ordering.extend(atm_idx)
        atm_ordering = np.array(atm_ordering, dtype=np.int32)
        mydf.atm_ordering = atm_ordering
    
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
    
    ######### allocate buffer ######### 
        
    Density_RgAO_buf = mydf.Density_RgAO_buf
    # print(Density_RgAO_buf.shape)
    
    nThread = lib.num_threads()
    bufsize_per_thread = (coulG_real.shape[0] * 2 + np.prod(mesh))
    buf_build_V = np.ndarray((nThread, bufsize_per_thread), dtype=np.float64, buffer=build_VW_buf) 
    
    offset_now = buf_build_V.size * buf_build_V.dtype.itemsize
    
    offset_build_now = 0
    offset_Density_RgR_buf = 0
    Density_RgR_buf = np.ndarray((maxsize_group_naux, ngrid), buffer=build_k_buf, offset=offset_build_now)
    
    offset_build_now += Density_RgR_buf.size * Density_RgR_buf.dtype.itemsize
    offset_ddot_res_RgR_buf = offset_build_now
    ddot_res_RgR_buf = np.ndarray((maxsize_group_naux, max_ngrid_involved), buffer=build_k_buf, offset=offset_ddot_res_RgR_buf)
    
    offset_build_now += ddot_res_RgR_buf.size * ddot_res_RgR_buf.dtype.itemsize
    offset_K1_tmp1_buf = offset_build_now
    K1_tmp1_buf = np.ndarray((maxsize_group_naux, nao), buffer=build_k_buf, offset=offset_K1_tmp1_buf)
    
    offset_build_now += K1_tmp1_buf.size * K1_tmp1_buf.dtype.itemsize
    offset_K1_tmp1_ddot_res_buf = offset_build_now
    K1_tmp1_ddot_res_buf = np.ndarray((maxsize_group_naux, max_nao_involved), buffer=build_k_buf, offset=offset_K1_tmp1_ddot_res_buf)
    
    offset_build_now += K1_tmp1_ddot_res_buf.size * K1_tmp1_ddot_res_buf.dtype.itemsize
    offset_V_pack_buf = offset_build_now
    V_pack_buf = np.ndarray((maxsize_group_naux, max_ngrid_involved), buffer=build_k_buf, offset=offset_V_pack_buf)
    
    offset_build_now += V_pack_buf.size * V_pack_buf.dtype.itemsize
    offset_K1_final_ddot_buf = offset_build_now
    K1_final_ddot_buf = np.ndarray((max_nao_involved, nao), buffer=build_k_buf, offset=offset_K1_final_ddot_buf)
    
    ########### get involved C function ###########
    
    fn_packcol1 = getattr(libpbc, "_buildK_packcol", None)
    assert fn_packcol1 is not None
    fn_packcol2 = getattr(libpbc, "_buildK_packcol2", None)
    assert fn_packcol2 is not None
    fn_packadd_col = getattr(libpbc, "_buildK_packaddcol", None)
    assert fn_packadd_col is not None
    fn_packadd_row = getattr(libpbc, "_buildK_packaddrow", None)
    assert fn_packadd_row is not None

    ######### begin work #########
    
    K1 = np.zeros((nao, nao), dtype=np.float64) # contribution from V matrix
    K2 = np.zeros((nao, nao), dtype=np.float64) # contribution from W matrix
    
    for group_id, atm_ids in enumerate(group):
        
        if use_mpi:
            if group_id % comm_size != rank:
                continue
        
        naux_tmp = 0
        aoRg_holders = []
        for atm_id in atm_ids:
            naux_tmp += aoRg[atm_id].aoR.shape[1]
            aoRg_holders.append(aoRg[atm_id])
        assert naux_tmp == aux_basis[group_id].shape[0]
        
        aux_basis_tmp = aux_basis[group_id]
        
        #### 1. build the involved DM_RgR #### 
        
        Density_RgAO_tmp = np.ndarray((naux_tmp, nao), buffer=Density_RgAO_buf)
        # print("Density_RgAO_tmp.shape = ", Density_RgAO_tmp.shape)
        offset_density_RgAO_buf = Density_RgAO_tmp.size * Density_RgAO_buf.dtype.itemsize
        
        Density_RgAO_tmp.ravel()[:] = 0.0
        Density_RgAO_tmp = __get_DensityMatrixonRgAO_qradratic(mydf, dm, aoRg_holders, Density_RgAO_tmp, verbose=mydf.verbose)
        
        #### 2. build the V matrix #### 
        
        V_tmp = np.ndarray((naux_tmp, ngrid), buffer=build_VW_buf, offset=offset_now, dtype=np.float64)
        offset_after_V_tmp = offset_now + V_tmp.size * V_tmp.dtype.itemsize
        
        aux_basis_grip_ID = mydf.partition_group_to_gridID[group_id]

        construct_V(aux_basis_tmp, buf_build_V, V_tmp, aux_basis_grip_ID, grid_ordering)
        
        #### 3. build the K1 matrix ####
        
        ###### 3.1 build density RgR
        
        Density_RgR_tmp = np.ndarray((naux_tmp, ngrid), buffer=Density_RgR_buf)
        
        # ngrid_loc = 0
        # for aoR_holder in aoR:
        for atm_id in atm_ordering:
            
            aoR_holder = aoR[atm_id]
                
            if aoR_holder is None:
                raise ValueError("aoR_holder is None")
                
            ngrid_now = aoR_holder.aoR.shape[1]
            nao_invovled = aoR_holder.aoR.shape[0]
                
            if nao_invovled == nao:
                Density_RgAO_packed = Density_RgAO_tmp
            else:
                # Density_RgAO_packed = Density_RgAO[:, aoR_holder.ao_involved]
                Density_RgAO_packed = np.ndarray((naux_tmp, nao_invovled), buffer=Density_RgAO_buf, offset=offset_density_RgAO_buf)
                fn_packcol1(
                    Density_RgAO_packed.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(naux_tmp),
                    ctypes.c_int(nao_invovled),
                    Density_RgAO_tmp.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(naux_tmp),
                    ctypes.c_int(nao),
                    aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
                )
            
            grid_begin = aoR_holder.global_gridID_begin
            ddot_res_RgR = np.ndarray((naux_tmp, ngrid_now), buffer=ddot_res_RgR_buf)
            lib.ddot(Density_RgAO_packed, aoR_holder.aoR, c=ddot_res_RgR)
            Density_RgR_tmp[:, grid_begin:grid_begin+ngrid_now] = ddot_res_RgR
                
            # ngrid_loc += ngrid_now
        
        # assert ngrid_loc == ngrid
        
        Density_RgR = Density_RgR_tmp
        
        # print("RgRg = ", RgRg[0, :16])
        
        #### 3.2 V_tmp = Density_RgR * V
        
        lib.cwise_mul(V_tmp, Density_RgR, out=Density_RgR)
        V2_tmp = Density_RgR
        
        #### 3.3 K1_tmp1 = V2_tmp * aoR.T
        
        K1_tmp1 = np.ndarray((naux_tmp, nao), buffer=K1_tmp1_buf)
        K1_tmp1.ravel()[:] = 0.0
        
        #ngrid_loc = 0
        for atm_id in atm_ordering:
            aoR_holder = aoR[atm_id]
                
            if aoR_holder is None:
                raise ValueError("aoR_holder is None")
                
            ngrid_now = aoR_holder.aoR.shape[1]
            nao_invovled = aoR_holder.aoR.shape[0]
                
            ddot_res = np.ndarray((naux_tmp, nao_invovled), buffer=K1_tmp1_ddot_res_buf)
            
            grid_loc_begin = aoR_holder.global_gridID_begin
            
            V_packed = np.ndarray((naux_tmp, ngrid_now), buffer=V_pack_buf)
            fn_packcol2(
                V_packed.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(naux_tmp),
                ctypes.c_int(ngrid_now),
                V2_tmp.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(naux_tmp),
                ctypes.c_int(ngrid),
                ctypes.c_int(grid_loc_begin),
                ctypes.c_int(grid_loc_begin+ngrid_now)
            )
            
            lib.ddot(V_packed, aoR_holder.aoR.T, c=ddot_res)
            # K1_tmp1[:, ngrid_loc:ngrid_loc+ngrid_now] = ddot_res
                
            if nao_invovled == nao:
                K1_tmp1 += ddot_res
            else:
                # K1_tmp1[: , aoR_holder.ao_involved] += ddot_res
                fn_packadd_col(
                    K1_tmp1.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(K1_tmp1.shape[0]),
                    ctypes.c_int(K1_tmp1.shape[1]),
                    ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(ddot_res.shape[0]),
                    ctypes.c_int(ddot_res.shape[1]),
                    aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
                )
                
            # ngrid_loc += ngrid_now
        # assert ngrid_loc == ngrid
        
        #### 3.4 K1 += aoRg * K1_tmp1
        
        ngrid_loc = 0
        
        # print("K1_tmp1.shape = ", K1_tmp1.shape)
        
        for atm_id in atm_ids:
            aoRg_holder = aoRg[atm_id]
            
            if aoRg_holder is None:
                raise ValueError("aoRg_holder is None")
            
            nIP_now = aoRg_holder.aoR.shape[1]
            nao_invovled = aoRg_holder.aoR.shape[0]
            
            # grid_loc_begin = aoRg_holder.global_gridID_begin
            # print("grid_loc_begin = ", grid_loc_begin)
            # print("nIP_now = ", nIP_now)
            
            K_tmp = K1_tmp1[ngrid_loc:ngrid_loc+nIP_now, :]
            
            # print("K_tmp.shape = ", K_tmp.shape)
            # print("aoRg_holder.aoR.shape = ", aoRg_holder.aoR.shape)
            
            ddot_res = np.ndarray((nao_invovled, nao), buffer=K1_final_ddot_buf)
            
            lib.ddot(aoRg_holder.aoR, K_tmp, c=ddot_res)
            
            if nao_invovled == nao:
                K1 += ddot_res
            else:
                # K1[aoRg_holder.ao_involved, :] += ddot_res
                fn_packadd_row(
                    K1.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(K1.shape[0]),
                    ctypes.c_int(K1.shape[1]),
                    ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(ddot_res.shape[0]),
                    ctypes.c_int(ddot_res.shape[1]),
                    aoRg_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
                )
            
            ngrid_loc += nIP_now
        assert ngrid_loc == naux_tmp
        
        #### 4. build the W matrix ####
        
        W_tmp = np.ndarray((naux_tmp, naux), dtype=np.float64, buffer=build_VW_buf, offset=offset_after_V_tmp)
        
        grid_shift = 0
        aux_col_loc = 0
        for j in range(len(group)):
            grid_ID_now = mydf.partition_group_to_gridID[j]
            aux_bas_ket = aux_basis[j]
            naux_ket = aux_bas_ket.shape[0]
            ngrid_now = grid_ID_now.size
            W_tmp[:, aux_col_loc:aux_col_loc+naux_ket] = lib.ddot(V_tmp[:, grid_shift:grid_shift+ngrid_now], aux_bas_ket.T)
            grid_shift += ngrid_now
            aux_col_loc += naux_ket
        
        assert grid_shift == ngrid
        
        #### 5. build the K2 matrix ####
        
        ###### 5.1 build density RgRg
        
        Density_RgRg_tmp = np.ndarray((naux_tmp, naux), buffer=Density_RgR_buf)
        
        nIP_loc = 0
        for atm_id in atm_ordering:
            aoRg_holder = aoRg[atm_id]
                
            if aoRg_holder is None:
                raise ValueError("aoRg_holder is None")
                
            nIP_now = aoRg_holder.aoR.shape[1]
            nao_invovled = aoRg_holder.aoR.shape[0]
                
            if nao_invovled == nao:
                Density_RgAO_packed = Density_RgAO_tmp
            else:
                # Density_RgAO_packed = Density_RgAO[:, aoRg_holder.ao_involved]
                Density_RgAO_packed = np.ndarray((naux_tmp, nao_invovled), buffer=Density_RgAO_buf, offset=offset_density_RgAO_buf)
                fn_packcol1(
                    Density_RgAO_packed.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(naux_tmp),
                    ctypes.c_int(nao_invovled),
                    Density_RgAO_tmp.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(naux_tmp),
                    ctypes.c_int(nao),
                    aoRg_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
                )
            
            assert nIP_loc == aoRg_holder.global_gridID_begin
            
            ddot_res_RgRg = np.ndarray((naux_tmp, nIP_now), buffer=ddot_res_RgR_buf)
            lib.ddot(Density_RgAO_packed, aoRg_holder.aoR, c=ddot_res_RgRg)
            Density_RgRg_tmp[:, nIP_loc:nIP_loc+nIP_now] = ddot_res_RgRg
                
            nIP_loc += nIP_now
        
        assert nIP_loc == naux 
        
        Density_RgRg = Density_RgRg_tmp
        
        #### 5.2 W_tmp = Density_RgRg * W
        
        lib.cwise_mul(W_tmp, Density_RgRg, out=Density_RgRg)
        W2_tmp = Density_RgRg
        
        #### 5.3 K2_tmp1 = W2_tmp * aoRg.T
        
        K2_tmp1 = np.ndarray((naux_tmp, nao), buffer=K1_tmp1_buf)
        K2_tmp1.ravel()[:] = 0.0
        
        nIP_loc = 0
        for atm_id in atm_ordering:
            aoRg_holder = aoRg[atm_id]
                
            if aoRg_holder is None:
                raise ValueError("aoRg_holder is None")
                
            nIP_now = aoRg_holder.aoR.shape[1]
            nao_invovled = aoRg_holder.aoR.shape[0]
                
            ddot_res = np.ndarray((naux_tmp, nao_invovled), buffer=K1_tmp1_ddot_res_buf)
            
            W_packed = np.ndarray((naux_tmp, nIP_now), buffer=V_pack_buf)
            fn_packcol2(
                W_packed.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(naux_tmp),
                ctypes.c_int(nIP_now),
                W2_tmp.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(naux_tmp),
                ctypes.c_int(naux),
                ctypes.c_int(nIP_loc),
                ctypes.c_int(nIP_loc+nIP_now)
            )
            
            lib.ddot(W_packed, aoRg_holder.aoR.T, c=ddot_res)
            # K2_tmp1[:, nIP_loc:nIP_loc+nIP_now] = ddot_res
                
            if nao_invovled == nao:
                K2_tmp1 += ddot_res
            else:
                # K2_tmp1[: , aoRg_holder.ao_involved] += ddot_res
                fn_packadd_col(
                    K2_tmp1.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(K2_tmp1.shape[0]),
                    ctypes.c_int(K2_tmp1.shape[1]),
                    ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(ddot_res.shape[0]),
                    ctypes.c_int(ddot_res.shape[1]),
                    aoRg_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
                )
                
            nIP_loc += nIP_now
        
        #### 5.4 K2 += aoRg * K2_tmp1
        
        nIP_loc = 0
        for atm_id in atm_ids:
            aoRg_holder = aoRg[atm_id]
            
            if aoRg_holder is None:
                raise ValueError("aoRg_holder is None")
            
            nIP_now = aoRg_holder.aoR.shape[1]
            nao_invovled = aoRg_holder.aoR.shape[0]
            
            K_tmp = K2_tmp1[nIP_loc:nIP_loc+nIP_now, :]
            
            ddot_res = np.ndarray((nao_invovled, nao), buffer=K1_final_ddot_buf)
            
            lib.ddot(aoRg_holder.aoR, K_tmp, c=ddot_res)
            
            if nao_invovled == nao:
                K2 += ddot_res
            else:
                # K2[aoRg_holder.ao_involved, :] += ddot_res
                fn_packadd_row(
                    K2.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(K2.shape[0]),
                    ctypes.c_int(K2.shape[1]),
                    ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(ddot_res.shape[0]),
                    ctypes.c_int(ddot_res.shape[1]),
                    aoRg_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
                )
            
            nIP_loc += nIP_now
        assert nIP_loc == naux_tmp
        
    ######### finally delete the buffer #########
    
    if use_mpi:
        comm.Barrier()
    
    # K1.ravel()[:] = 0.0
    # K2.ravel()[:] = 0.0
    # K2 = -K2
    
    # print("K1 = ", K1[0, :16])
    # print("K2 = ", K2[0, :16])
    
    if use_mpi:
        K1 = reduce(K1, root = 0)
        K2 = reduce(K2, root = 0)
        if rank == 0:
            K = K1 + K1.T - K2
        else:
            K = None
        K = bcast(K, root = 0)
    else:
        K = K1 + K1.T - K2 
    
    del K1
    del K2
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    if mydf.verbose:
        _benchmark_time(t1, t2, "_contract_k_dm_quadratic_direct")
    
    return K * ngrid / vol

def get_jk_dm_quadratic(mydf, dm, hermi=1, kpt=np.zeros(3),
                        kpts_band=None, with_j=True, with_k=True, omega=None, 
                        **kwargs):
    
    '''JK for given k-point'''
    
    direct = mydf.direct
    use_mpi = mydf.use_mpi
    
    if use_mpi and direct == False:
        raise NotImplementedError("ISDF does not support use_mpi and direct=False")
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    if hasattr(mydf, 'Ls') and mydf.Ls is not None:
        from pyscf.pbc.df.isdf.isdf_tools_densitymatrix import symmetrize_dm
        dm = symmetrize_dm(dm, mydf.Ls)
    else:
        if hasattr(mydf, 'kmesh') and mydf.kmesh is not None:
            from pyscf.pbc.df.isdf.isdf_tools_densitymatrix import symmetrize_dm
            dm = symmetrize_dm(dm, mydf.kmesh)

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
        from pyscf.pbc.df.isdf.isdf_jk import _contract_j_dm
        vj = _contract_j_dm_ls(mydf, dm, use_mpi)  
        # vj = _contract_j_dm_wo_robust_fitting(mydf, dm, use_mpi)
        # vj = _contract_j_dm(mydf, dm, use_mpi)
        # if rank == 0:
        # print("vj = ", vj[0, :16])
        # print("vj = ", vj[0, -16:])
    if with_k:
        if mydf.direct:
            vk = _contract_k_dm_quadratic_direct(mydf, dm, use_mpi=use_mpi)
        else:
            vk = _contract_k_dm_quadratic(mydf, dm, mydf.with_robust_fitting, use_mpi=use_mpi)
        # if rank == 0:
        # print("vk = ", vk[0, :16])
        # print("vk = ", vk[0, -16:])
        if exxdiv == 'ewald':
            print("WARNING: ISDF does not support ewald")

    t1 = log.timer('sr jk', *t1)

    return vj, vk

############# occ RI #############

def get_jk_occRI(mydf, mo_coeff, nocc, dm, use_mpi=False):

    if mydf.with_robust_fitting or mydf.direct:
        raise NotImplementedError("get_jk_occRI does not support robust fitting or direct=True")

    if use_mpi:
        raise NotImplementedError("get_jk_occRI does not support use_mpi=True")

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    occ_mo = mo_coeff[:, :nocc].copy()

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
    
    ddot_buf = np.zeros((max_dim_buf, max_dim_buf), dtype=np.float64)
    aoR_buf1 = np.zeros((max_nao_involved, max_ngrid_involved), dtype=np.float64)
    
    if hasattr(mydf, "moRg") is False:
        mydf.moRg = []
        for aoR_holder in aoR:
            if aoR_holder is None:
                mydf.moRg.append(None)
            else:
                moRg_buf = np.zeros((nocc, aoR_holder.aoR.shape[1]), dtype=np.float64)
                mydf.moRg.append(moRg_buf)
    
    moR_buf     = np.zeros((nocc, max_ngrid_involved), dtype=np.float64) # which can generated on the fly
    mo_coeff_pack_buf = np.zeros((nao, max_nao_involved), dtype=np.float64)

    ####### involved functions #######
    
    fn_packrow = getattr(libpbc, "_buildK_packrow", None)
    assert fn_packrow is not None

    fn_packadd_col = getattr(libpbc, "_buildK_packaddcol", None)
    assert fn_packadd_col is not None

    #### step 0 get_half_J ####

    J = _half_J(mydf, dm, use_mpi=use_mpi)

    J_Res_occRI = np.zeros((nocc, nao), dtype=np.float64)

    #### step 1 get_J ####

    for aoR_holder in aoR:
        
        if aoR_holder is None:
            continue
        
        if use_mpi:
            if atm_id % comm_size != rank:
                continue
        
        ngrids_now = aoR_holder.aoR.shape[1]
        nao_involved = aoR_holder.aoR.shape[0]
        
        mo_coeff_packed = np.ndarray((nocc, nao_involved), buffer=mo_coeff_pack_buf)
        fn_packrow(
            mo_coeff_packed.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nocc),
            ctypes.c_int(nao_involved),
            occ_mo.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao),
            ctypes.c_int(nocc),
            aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
        )
        moR_now = np.ndarray((nocc, ngrids_now), buffer=moR_buf)
        
        lib.ddot(mo_coeff_packed.T, aoR_holder.aoR, c=moR_now)
        
        global_gridID_begin = aoR_holder.global_gridID_begin
        
        J_tmp = J[global_gridID_begin:global_gridID_begin+ngrids_now] 
        
        aoR_J_res = np.ndarray((nocc, ngrids_now), buffer=aoR_buf1)
        lib.d_ij_j_ij(moR_now, J_tmp, out=aoR_J_res)
        ddot_res = np.ndarray((nocc, nao_involved), buffer=ddot_buf)
        lib.ddot(aoR_J_res, aoR_holder.aoR.T, c=ddot_res)
        
        if nao_involved == nao:
            J_Res += ddot_res
        else:
            # fn_packadd_dm(
            #     ddot_res.ctypes.data_as(ctypes.c_void_p),
            #     ctypes.c_int(nao_involved),
            #     aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p),
            #     J_Res.ctypes.data_as(ctypes.c_void_p),
            #     ctypes.c_int(nao)
            # )      
            
            fn_packadd_col(
                J_Res_occRI.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(J_Res_occRI.shape[0]),
                ctypes.c_int(J_Res_occRI.shape[1]),
                ddot_res.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(ddot_res.shape[0]),
                ctypes.c_int(ddot_res.shape[1]),
                aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
            )  

        # local_grid_loc += ngrids_now
    
    # assert local_grid_loc == ngrids_local

    J = J_Res

    K = np.zeros((nocc, nao), dtype=np.float64)

    #### step 2 get moRg ####
    
    
    del J_Res_occRI
    
    ### delete buf ###
    
    del ddot_buf, aoR_buf1, moR_buf, mo_coeff_pack_buf
    
    
    return J * ngrid / vol, K * ngrid / vol

############# linear scaling implementation ############# 