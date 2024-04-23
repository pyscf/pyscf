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

import copy, sys
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

def _half_J(mydf, dm, use_mpi=False,
            first_pass = None,
            short_range = False):
    
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        #raise NotImplementedError("MPI is not supported yet.")
        assert mydf.direct == True
        
    ######### prepare the parameter #########
    
    assert first_pass in [None, "only_dd", "only_cc", "exclude_cc", "all"]
    
    if first_pass is None:
        first_pass = "all"
    
    first_pass_all = first_pass == "all"
    first_pass_has_dd = first_pass in ["all", "only_dd", "exclude_cc"]
    first_pass_has_cc = first_pass in ["all", "only_cc"]
    first_pass_has_cd = first_pass in ["all", "exclude_cc"]
    
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
    
    dm_buf = np.zeros((max_nao_involved, max_nao_involved), dtype=np.float64)
    max_dim_buf = max(max_ngrid_involved, max_nao_involved)
    ddot_buf = np.zeros((max_dim_buf, max_dim_buf), dtype=np.float64)
    
    ##### get the involved C function ##### 
    
    fn_extract_dm = getattr(libpbc, "_extract_dm_involved_ao", None) 
    assert fn_extract_dm is not None
    
    fn_extract_dm2 = getattr(libpbc, "_extract_dm_involved_ao_RS", None)
    assert fn_extract_dm is not None
    
    fn_packadd_dm = getattr(libpbc, "_packadd_local_dm", None)
    assert fn_packadd_dm is not None 
    
    # fn_packadd_dm2 = getattr(libpbc, "_packadd_local_RS", None)
    # assert fn_packadd_dm2 is not None
    
    #### step 1. get density value on real space grid and IPs
    
    group = mydf.group
    ngroup = len(group)
    
    density_R_tmp = None
    
    def _get_rhoR(
        bra_aoR, 
        bra_ao_involved,
        ket_aoR, 
        ket_ao_involved,
        bra_type,
        ket_type
    ):
        
        nbra_ao = bra_aoR.shape[0]
        nket_ao = ket_aoR.shape[0]
        if bra_type == ket_type:
            dm_now = np.ndarray((nbra_ao, nbra_ao), buffer=dm_buf)
            if nbra_ao < nao:
                fn_extract_dm(
                    dm.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao),
                    dm_now.ctypes.data_as(ctypes.c_void_p),
                    bra_ao_involved.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nbra_ao),
                )
            else:
                dm_now.ravel()[:] = dm.ravel()
            ddot_res = np.ndarray((nbra_ao, ket_aoR.shape[1]), buffer=ddot_buf)
            lib.ddot(dm_now, ket_aoR, c=ddot_res)
            density_R_tmp = lib.multiply_sum_isdf(bra_aoR, ddot_res)
            return density_R_tmp
        else:
            dm_now = np.ndarray((nbra_ao, nket_ao), buffer=dm_buf)
            fn_extract_dm2(
                dm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao),
                dm_now.ctypes.data_as(ctypes.c_void_p),
                bra_ao_involved.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(bra_ao_involved.shape[0]),
                ket_ao_involved.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(ket_ao_involved.shape[0]),
            )
            ddot_res = np.ndarray((nbra_ao, ket_aoR.shape[1]), buffer=ddot_buf)
            lib.ddot(dm_now, ket_aoR, c=ddot_res)
            density_R_tmp = lib.multiply_sum_isdf(bra_aoR, ddot_res)
            return density_R_tmp * 2.0
    
    for atm_id, aoR_holder in enumerate(aoR):
        
        if aoR_holder is None:
            continue
        
        if use_mpi:
            if atm_id % comm_size != rank:
                continue
            
        ngrids_now = aoR_holder.aoR.shape[1]
        nao_involved = aoR_holder.aoR.shape[0]
        global_gridID_begin = aoR_holder.global_gridID_begin
        
        if first_pass_all:        
            density_R_tmp = _get_rhoR(
                aoR_holder.aoR, 
                aoR_holder.ao_involved, 
                aoR_holder.aoR, 
                aoR_holder.ao_involved,
                "all",
                "all"
            )
        
            density_R[global_gridID_begin:global_gridID_begin+ngrids_now] = density_R_tmp
        else: 
            
            if first_pass_has_cc:
                density_R_tmp = _get_rhoR(
                    aoR_holder.aoR[:nao_involved,:], 
                    aoR_holder.ao_involved[:nao_involved], 
                    aoR_holder.aoR[:nao_involved,:], 
                    aoR_holder.ao_involved[:nao_involved],
                    "compact",
                    "compact"
                )
                
                density_R[global_gridID_begin:global_gridID_begin+ngrids_now] = density_R_tmp
            
            if first_pass_has_dd:
                density_R_tmp = _get_rhoR(
                    aoR_holder.aoR[nao_involved:,:], 
                    aoR_holder.ao_involved[nao_involved:], 
                    aoR_holder.aoR[nao_involved:,:], 
                    aoR_holder.ao_involved[nao_involved:],
                    "diffuse",
                    "diffuse"
                )
                
                density_R[global_gridID_begin:global_gridID_begin+ngrids_now] = density_R_tmp
            
            if first_pass_has_cd:
                density_R_tmp = _get_rhoR(
                    aoR_holder.aoR[:nao_involved,:], 
                    aoR_holder.ao_involved[:nao_involved], 
                    aoR_holder.aoR[nao_involved:,:], 
                    aoR_holder.ao_involved[nao_involved:],
                    "compact",
                    "diffuse"
                )
                
                density_R[global_gridID_begin:global_gridID_begin+ngrids_now] = density_R_tmp
    
    # assert local_grid_loc == ngrids_local
    
    if use_mpi:
        density_R = reduce(density_R, root=0)
    else:
        assert ngrids_local == np.prod(mesh)
        
    # if hasattr(mydf, "grid_ID_ordered"):
    
    grid_ID_ordered = mydf.grid_ID_ordered
    
    # print("grid_ID_ordered = ", grid_ID_ordered[:64])
    
    # print("density_R = ", density_R[:16])
    
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
    
    # print("density_R = ", density_R[:16])
    
    J = None
    
    if (use_mpi and rank == 0) or (use_mpi == False):
    
        fn_J = getattr(libpbc, "_construct_J", None)
        assert(fn_J is not None)

        J = np.zeros_like(density_R)

        if short_range:
            coulG = mydf.coulG_SR
        else:
            coulG = mydf.coulG

        fn_J(
            mesh.ctypes.data_as(ctypes.c_void_p),
            density_R.ctypes.data_as(ctypes.c_void_p),
            coulG.ctypes.data_as(ctypes.c_void_p),
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
def _contract_j_dm_ls(mydf, dm, use_mpi=False, 
                      first_pass = None, 
                      second_pass = None,
                      short_range = False):
    
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        #raise NotImplementedError("MPI is not supported yet.")
        assert mydf.direct == True
    
    ###### Prepocess parameter for RS ######
    
    assert first_pass  in [None, "only_dd", "only_cc", "exclude_cc", "all"]
    assert second_pass in [None, "only_dd", "only_cc", "exclude_cc", "all"]
    
    if short_range:
        assert first_pass is "only_dd"
        assert second_pass is "only_dd"
    
    if first_pass is None:
        first_pass = "all"
    if second_pass is None:
        second_pass = "all"
    
    second_pass_all    = second_pass == "all"
    second_pass_has_dd = second_pass in ["all", "only_dd", "exclude_cc"]
    second_pass_has_cc = second_pass in ["all", "only_cc"]
    second_pass_has_cd = second_pass in ["all", "exclude_cc"]
    
    ####### Start the calculation ########
    
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
    
    fn_packadd_dm = getattr(libpbc, "_packadd_local_dm", None)
    assert fn_packadd_dm is not None
    
    fn_packadd_dm2 = getattr(libpbc, "_packadd_local_RS", None)
    assert fn_packadd_dm2 is not None
    
    #### step 1 2. get density value on real space grid and IPs
    
    group = mydf.group
    ngroup = len(group)

    J = _half_J(mydf, dm, use_mpi, first_pass, short_range)

    #### step 3. get J 
    
    # J = np.asarray(lib.d_ij_j_ij(aoR, J, out=buffer1), order='C') 
    # J = lib.ddot_withbuffer(aoR, J.T, buf=mydf.ddot_buf)

    # local_grid_loc = 0

    J_Res = np.zeros((nao, nao), dtype=np.float64)

    def _get_j_pass2_ls(aoR_bra, 
                        ao_involved_bra, 
                        aoR_ket,
                        ao_involved_ket,
                        bra_type,
                        ket_type,
                        potential,
                        Res):
        
        nao_bra = aoR_bra.shape[0]
        nao_ket = aoR_ket.shape[0]
        
        if bra_type == ket_type:
            
            aoR_J_res = np.ndarray(aoR_ket.shape, buffer=aoR_buf1)
            lib.d_ij_j_ij(aoR_ket, potential, out=aoR_J_res)
            ddot_res = np.ndarray((nao_ket, nao_ket), buffer=ddot_buf)
            lib.ddot(aoR_ket, aoR_J_res.T, c=ddot_res)
            
            if nao_ket == nao:
                Res += ddot_res
            else:
                fn_packadd_dm(
                    ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao_ket),
                    ao_involved_ket.ctypes.data_as(ctypes.c_void_p),
                    Res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(Res.shape[0])
                )
        else:
            ### J_Res = ddot_res + ddot_res.T
            
            aoR_J_res = np.ndarray(aoR_ket.shape, buffer=aoR_buf1)
            lib.d_ij_j_ij(aoR_ket, potential, out=aoR_J_res)
            ddot_res = np.ndarray((nao_bra, nao_ket), buffer=ddot_buf)
            lib.ddot(aoR_bra, aoR_J_res.T, c=ddot_res)
            
            fn_packadd_dm2(
                ddot_res.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_bra),
                ao_involved_bra.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_ket),
                ao_involved_ket.ctypes.data_as(ctypes.c_void_p),
                Res.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(Res.shape[0])
            )


    for atm_id, aoR_holder in enumerate(aoR):
        
        if aoR_holder is None:
            continue
        
        if use_mpi:
            if atm_id % comm_size != rank:
                continue
        
        ngrids_now = aoR_holder.aoR.shape[1]
        nao_involved = aoR_holder.nao_invovled
        nao_compact  = aoR_holder.nCompact
        nao_diffuse  = nao_involved - nao_compact
        
        global_gridID_begin = aoR_holder.global_gridID_begin
        
        J_tmp = J[global_gridID_begin:global_gridID_begin+ngrids_now] 
        
        if second_pass_all:  ### with RS case ###
              
            _get_j_pass2_ls(
                aoR_holder.aoR, 
                aoR_holder.ao_involved, 
                aoR_holder.aoR,
                aoR_holder.ao_involved,
                "all",
                "all",
                J_tmp,
                J_Res
            )   
        
        else:
            
            if second_pass_has_cc:
                
                _get_j_pass2_ls(
                    aoR_holder.aoR[:nao_compact,:], 
                    aoR_holder.ao_involved[:nao_compact], 
                    aoR_holder.aoR[:nao_compact,:],
                    aoR_holder.ao_involved[:nao_compact],
                    "compact",
                    "compact",
                    J_tmp,
                    J_Res
                )
                
            if second_pass_has_dd:
                
                _get_j_pass2_ls(
                    aoR_holder.aoR[nao_compact:,:], 
                    aoR_holder.ao_involved[nao_compact:], 
                    aoR_holder.aoR[nao_compact:,:],
                    aoR_holder.ao_involved[nao_compact:],
                    "diffuse",
                    "diffuse",
                    J_tmp,
                    J_Res
                )
                
            if second_pass_has_cd:
                
                _get_j_pass2_ls(
                    aoR_holder.aoR[:nao_compact,:], 
                    aoR_holder.ao_involved[:nao_compact], 
                    aoR_holder.aoR[nao_compact:,:],
                    aoR_holder.ao_involved[nao_compact:],
                    "compact",
                    "diffuse",
                    J_tmp,
                    J_Res
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

############# Xing's multigrid get_j function #############

from pyscf.pbc.df.df_jk import (
    _format_dms,
    _format_kpts_band,
    _format_jks,
)
from pyscf.pbc.dft.multigrid.multigrid import _eval_rhoG, _get_j_pass2

def _contract_j_multigrid(mydf, 
           # xc_code, 
           dm_kpts, hermi=1, kpts=None,
           kpts_band=None, 
           with_j=True, 
           return_j=True, 
           verbose=None):
    '''Compute the XC energy and RKS XC matrix at sampled k-points.
    multigrid version of function pbc.dft.numint.nr_rks.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        exc : XC energy
        nelec : number of electrons obtained from the numerical integration
        veff : (nkpts, nao, nao) ndarray
            or list of veff if the input dm_kpts is a list of DMs
        vj : (nkpts, nao, nao) ndarray
            or list of vj if the input dm_kpts is a list of DMs
    '''
    if kpts is None: kpts = mydf.kpts
    log = logger.new_logger(mydf, verbose)
    cell = mydf.cell
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band

    ni = mydf._numint
    deriv = 0
    # xctype = ni._xc_type(xc_code)
    # if xctype == 'LDA':
    #     deriv = 0
    # elif xctype == 'GGA':
    #     deriv = 1
    # elif xctype == 'MGGA':
    #     deriv = 2 if MGGA_DENSITY_LAPL else 1
    #     raise NotImplementedError
    rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv)

    mesh = mydf.mesh
    ngrids = numpy.prod(mesh)
    coulG = tools.get_coulG(cell, mesh=mesh)
    vG = numpy.einsum('ng,g->ng', rhoG[:,0], coulG)

    if getattr(mydf, "vpplocG_part1", None) is not None and not mydf.pp_with_erf:
        # for i in range(nset):
        #     vG[i] += mydf.vpplocG_part1 * 2
        raise NotImplementedError("vpplocG_part1 is not supported yet.")

    ecoul = .5 * numpy.einsum('ng,ng->n', rhoG[:,0].real, vG.real)
    ecoul+= .5 * numpy.einsum('ng,ng->n', rhoG[:,0].imag, vG.imag)
    ecoul /= cell.vol
    log.debug('Multigrid Coulomb energy %s', ecoul)

    if getattr(mydf, "vpplocG_part1", None) is not None and not mydf.pp_with_erf:
        #for i in range(nset):
        #    vG[i] -= mydf.vpplocG_part1
        raise NotImplementedError("vpplocG_part1 is not supported yet.")

    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    rhoR = rhoR.reshape(nset,-1,ngrids)
    nelec = rhoR[:,0].sum(axis=1) * weight

    # wv_freq = []
    # excsum = numpy.zeros(nset)
    # for i in range(nset):
    #     if xctype == 'LDA':
    #         exc, vxc = ni.eval_xc_eff(xc_code, rhoR[i,0], deriv=1, xctype=xctype)[:2]
    #     else:
    #         exc, vxc = ni.eval_xc_eff(xc_code, rhoR[i], deriv=1, xctype=xctype)[:2]
    #     excsum[i] += (rhoR[i,0]*exc).sum() * weight
    #     wv = weight * vxc
    #     wv_freq.append(tools.fft(wv, mesh))
    # wv_freq = numpy.asarray(wv_freq).reshape(nset,-1,*mesh)
    rhoR = rhoG = None

    if nset == 1:
        ecoul = ecoul[0]
        nelec = nelec[0]
        # excsum = excsum[0]
    # log.debug('Multigrid exc %s  nelec %s', excsum, nelec)
    log.debug('Multigrid nelec %s', nelec)

    # kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    # if xctype == 'LDA':
    #     if with_j:
    #         wv_freq[:,0] += vG.reshape(nset,*mesh)
    #     veff = _get_j_pass2(mydf, wv_freq, hermi, kpts_band, verbose=log)
    # elif xctype == 'GGA':
    #     if with_j:
    #         wv_freq[:,0] += vG.reshape(nset,*mesh)
    #     # *.5 because v+v.T is always called in _get_gga_pass2
    #     wv_freq[:,0] *= .5
    #     veff = _get_gga_pass2(mydf, wv_freq, hermi, kpts_band, verbose=log)
    # veff = _format_jks(veff, dm_kpts, input_band, kpts)

    if return_j:
        vj = _get_j_pass2(mydf, vG, hermi, kpts_band, verbose=log)
        vj = _format_jks(vj, dm_kpts, input_band, kpts)
    else:
        vj = None

    # shape = list(dm_kpts.shape)
    # if len(shape) == 3 and shape[0] != kpts_band.shape[0]:
    #     shape[0] = kpts_band.shape[0]
    # veff = veff.reshape(shape)
    # veff = lib.tag_array(veff, ecoul=ecoul, exc=excsum, vj=vj, vk=None)
    
    # return nelec, excsum, veff 
    return vj

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

def __get_DensityMatrixonRgAO_qradratic(mydf, dm, 
                                        bra_aoR_holder, 
                                        bra_type       = None,
                                        res:np.ndarray = None, 
                                        verbose        = 1):
    
    assert bra_type in [None, "all", "compact", "diffuse"]
    
    # print("bra_type = ", bra_type)
    
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
        nao_compact  = aoR_holder.nCompact
        
        ao_begin_indx = 0
        ao_end_indx   = nao_involved
        if bra_type == "compact":
            ao_end_indx = nao_compact
        elif bra_type == "diffuse":
            ao_begin_indx = nao_compact 
        
        nao_at_work = ao_end_indx - ao_begin_indx
        
        if (nao_at_work) == nao:
            dm_packed = dm
        else:
            dm_packed = np.ndarray((nao_at_work, nao), buffer=dm_pack_buf)
            fn_packrow(
                dm_packed.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_at_work),
                ctypes.c_int(nao),
                dm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao),
                ctypes.c_int(nao),
                aoR_holder.ao_involved[ao_begin_indx:ao_end_indx].ctypes.data_as(ctypes.c_void_p)
            )
        
        ddot_res = np.ndarray((ngrid_now, nao), buffer=ddot_buf)
        lib.ddot(aoR_holder.aoR[ao_begin_indx:ao_end_indx,:].T, dm_packed, c=ddot_res)
        grid_loc_begin = aoR_holder.global_gridID_begin
        res[grid_loc_begin:grid_loc_begin+ngrid_now, :] = ddot_res
        
        # ngrid_loc += ngrid_now   
    #assert ngrid_loc == ngrid_bra
        
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
    
    Density_RgAO = __get_DensityMatrixonRgAO_qradratic(mydf, dm, aoRg, "all", mydf.Density_RgAO_buf, use_mpi)
    
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
        if mydf.omega is not None:
            assert mydf.omega >= 0.0
        # mydf.coulG = tools.get_coulG(cell, mesh=mesh, omega=mydf.omega)
        raise NotImplementedError("coulG is not implemented yet.")
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
        Density_RgAO_tmp = __get_DensityMatrixonRgAO_qradratic(mydf, dm, aoRg_holders, "all", Density_RgAO_tmp, verbose=mydf.verbose)
        
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

############# occ RI #############

def get_jk_occRI(mydf, dm, use_mpi=False, with_j=True, with_k=True):

    assert mydf.omega is None or mydf.omega == 0.0
    # assert with_j_occRI is False

    t1 = (logger.process_clock(), logger.perf_counter())
    t0 = t1

    if mydf.direct:
        raise NotImplementedError("get_jk_occRI does not support robust fitting or direct=True")

    if use_mpi:
        raise NotImplementedError("get_jk_occRI does not support use_mpi=True")

    # print("dm.shape = ", dm.shape)

    if getattr(dm, 'mo_coeff', None) is not None:
        mo_coeff = dm.mo_coeff
        mo_occ   = dm.mo_occ
    else:
        raise NotImplementedError("mo_coeff is not provided yet")

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    ##### fetch the basic info #####

    nao  = dm.shape[0]
    cell = mydf.cell
    assert cell.nao == nao
    vol   = cell.vol
    mesh  = np.array(cell.mesh, dtype=np.int32)
    ngrid = np.prod(mesh)

    aoR  = mydf.aoR
    aoRg = mydf.aoRg
    assert isinstance(aoR, list)
    naux = mydf.naux

    weight = np.sqrt(cell.vol/ngrid)

    ######### weighted mo_coeff #########
    
    occ_tol    = mydf.occ_tol
    nocc       = np.count_nonzero(mo_occ > occ_tol)
    occ_weight = np.sqrt(mo_occ[mo_occ > occ_tol])
    # print("occ_weight = ", occ_weight)
    mo_coeff_full     = mo_coeff.copy()
    mo_coeff_original = mo_coeff[:,mo_occ > occ_tol].copy()
    mo_coeff = mo_coeff[:,mo_occ > occ_tol] * occ_weight ## NOTE: it is a weighted mo_coeff
    mo_coeff = mo_coeff.copy()                           ## NOTE: nonsense thing in python
    assert mo_coeff.shape[1] == nocc
    assert mo_coeff.shape[0] == nao
    
    # dm2 = np.dot(mo_coeff, mo_coeff.T)
    # assert np.allclose(dm, dm2)

    # print("mo_coeff_original = ", mo_coeff_original[:,0])
    # print("mo_coeff          = ", mo_coeff[:,0])

    ####### determine whether to construct moR #######
    
    construct_moR    = with_j or (with_k and mydf.with_robust_fitting is True)
    construct_dmRgRg = with_k
    construct_dmRgR  = with_k and mydf.with_robust_fitting is True

    #### step -2. allocate buffer 

    max_nao_involved   = np.max([aoR_holder.aoR.shape[0] for aoR_holder in aoR if aoR_holder is not None])
    max_ngrid_involved = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None])
    ngrids_local       = np.sum([aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None])
    max_dim_buf        = max(max_ngrid_involved, max_nao_involved)
    max_nIP_involved   = np.max([aoRg_holder.aoR.shape[1] for aoRg_holder in aoRg if aoRg_holder is not None])
        
    mydf.deallocate_k_buffer()
    
    if hasattr(mydf, "moRg") is False:
        mydf.moRg = np.zeros((nocc, naux), dtype=np.float64)
    else:
        if nocc != mydf.moRg.shape[0]:
            mydf.moRg = np.zeros((nocc, naux), dtype=np.float64)
            
    if hasattr(mydf, "K1_packbuf") is False:
        mydf.K1_packbuf = np.zeros((nocc, max_ngrid_involved), dtype=np.float64)
    else:
        if nocc != mydf.K1_packbuf.shape[0]:
            mydf.K1_packbuf = np.zeros((nocc, max_ngrid_involved), dtype=np.float64)
    
    if construct_moR:
        if hasattr(mydf, "moR") is False:
            mydf.moR = np.zeros((nocc, ngrid), dtype=np.float64)
        else:
            if nocc != mydf.moR.shape[0]:
                mydf.moR = np.zeros((nocc, ngrid), dtype=np.float64)
            
    if construct_dmRgR:
        if hasattr(mydf, "dmRgR") is False:
            mydf.dmRgR = np.zeros((naux, ngrid), dtype=np.float64)
    if construct_dmRgRg:
        if hasattr(mydf, "dmRgRg") is False:
            mydf.dmRgRg = np.zeros((naux, naux), dtype=np.float64)
    
    ddot_buf          = np.zeros((max_dim_buf, max_dim_buf), dtype=np.float64)
    aoR_buf1          = np.zeros((max_nao_involved, max_ngrid_involved), dtype=np.float64)
    moR_buf           = np.zeros((nocc, max_ngrid_involved), dtype=np.float64) # which can generated on the fly
    mo_coeff_pack_buf = np.zeros((nao, max_nao_involved), dtype=np.float64)

    ####### involved functions #######
    
    fn_packrow = getattr(libpbc, "_buildK_packrow", None)
    assert fn_packrow is not None
    
    fn_packadd_row = getattr(libpbc, "_buildK_packaddrow", None)
    assert fn_packadd_row is not None
    
    fn_packcol = getattr(libpbc, "_buildK_packcol", None)
    assert fn_packcol is not None
    
    fn_packcol2 = getattr(libpbc, "_buildK_packcol2", None)
    assert fn_packcol2 is not None
    
    fn_packcol3 = getattr(libpbc, "_buildK_packcol3", None)
    assert fn_packcol3 is not None

    fn_packadd_col = getattr(libpbc, "_buildK_packaddcol", None)
    assert fn_packadd_col is not None
    
    fn_packadd_dm = getattr(libpbc, "_packadd_local_dm", None)
    assert fn_packadd_dm is not None

    #### step -1. construct moR, moRg, dmRgRg, dmRg ####

    IP_loc_in_ordered_grids = mydf.IP_loc_in_ordered_grids

    def _get_mo_values_on_grids(_aoR_holders, out_):
        
        for aoR_holder in _aoR_holders:
            
            ngrids_now   = aoR_holder.aoR.shape[1]
            nao_involved = aoR_holder.aoR.shape[0]
            
            mo_coeff_packed = np.ndarray((nao_involved, nocc), buffer=mo_coeff_pack_buf)
            # assert mo_coeff_packed.shape[0] == aoR_holder.ao_involved.shape[0]
            # assert mo_coeff_packed.shape[1] == mo_coeff.shape[1]
            
            fn_packrow(
                mo_coeff_packed.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(mo_coeff_packed.shape[0]),
                ctypes.c_int(mo_coeff_packed.shape[1]),
                mo_coeff.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(mo_coeff.shape[0]),
                ctypes.c_int(mo_coeff.shape[1]),
                aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
            )
            
            moR_now = np.ndarray((nocc, ngrids_now), buffer=moR_buf)
            lib.ddot(mo_coeff_packed.T, aoR_holder.aoR, c=moR_now)
            global_gridID_begin = aoR_holder.global_gridID_begin
            fn_packcol3(
                out_.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(out_.shape[0]),
                ctypes.c_int(out_.shape[1]),
                ctypes.c_int(global_gridID_begin),
                ctypes.c_int(global_gridID_begin+ngrids_now),
                moR_now.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(moR_now.shape[0]),
                ctypes.c_int(moR_now.shape[1])
            )
                

    t3 = (logger.process_clock(), logger.perf_counter())
    
    if hasattr(mydf, "moR"):
        moR = mydf.moR
    else:
        moR = None  
    moRg = mydf.moRg

    if construct_moR:
        _get_mo_values_on_grids(aoR, moR)
        fn_packcol(
            moRg.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(moRg.shape[0]),
            ctypes.c_int(moRg.shape[1]),
            moR.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(moR.shape[0]),
            ctypes.c_int(moR.shape[1]),
            IP_loc_in_ordered_grids.ctypes.data_as(ctypes.c_void_p)
        )
        
    else:
        moR = None
        _get_mo_values_on_grids(aoRg, moRg)

    t4 = (logger.process_clock(), logger.perf_counter())
    
    if mydf.verbose:
        _benchmark_time(t3, t4, "get_mo over grids")
        sys.stdout.flush()

    t3 = (logger.process_clock(), logger.perf_counter())
    
    if construct_dmRgR:
        dmRgR = mydf.dmRgR
        lib.ddot(moRg.T, moR, c=dmRgR)
        dmRgRg = mydf.dmRgRg
        fn_packcol(
            dmRgRg.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(naux),
            ctypes.c_int(naux),
            dmRgR.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(naux),
            ctypes.c_int(ngrid),
            IP_loc_in_ordered_grids.ctypes.data_as(ctypes.c_void_p)
        )
    else:
        dmRgR = None
        dmRgRg = mydf.dmRgRg
        lib.ddot(moRg.T, moRg, c=dmRgRg)
    
    t4 = (logger.process_clock(), logger.perf_counter())
    
    if mydf.verbose:
        _benchmark_time(t3, t4, "get_dm over grids")
        
    #### step 0 get_half_J ####

    if with_j:
        
        # weighted moR to densityR
        
        rhoR = np.zeros((ngrid), dtype=np.float64)
        
        fn_rhoR = getattr(libpbc, "moR_to_Density", None)
        assert fn_rhoR is not None
        
        fn_rhoR(
            ctypes.c_int(ngrid), 
            ctypes.c_int(nocc),
            moR.ctypes.data_as(ctypes.c_void_p),
            rhoR.ctypes.data_as(ctypes.c_void_p)
        )
                        
        # from rhoG to the potential # 
        
        rhoR_original = np.zeros_like(rhoR)
        
        fn_order = getattr(libpbc, "_Reorder_Grid_to_Original_Grid", None)
        assert fn_order is not None
        
        fn_order(
            ctypes.c_int(ngrid),
            mydf.grid_ID_ordered.ctypes.data_as(ctypes.c_void_p),
            rhoR.ctypes.data_as(ctypes.c_void_p),
            rhoR_original.ctypes.data_as(ctypes.c_void_p)
        )        
        
        rhoR = rhoR_original
                
        fn_J = getattr(libpbc, "_construct_J", None)
        assert fn_J is not None
        
        if hasattr(mydf, "coulG") == False:
            if mydf.omega is not None:
                assert mydf.omega >= 0.0
            print("mydf.omega = ", mydf.omega)
            # mydf.coulG = tools.get_coulG(cell, mesh=mesh, omega=mydf.omega)
            raise ValueError("mydf.coulG is not found.")
        
        J = np.zeros_like(rhoR)
        
        fn_J(
            mesh.ctypes.data_as(ctypes.c_void_p),
            rhoR.ctypes.data_as(ctypes.c_void_p),
            mydf.coulG.ctypes.data_as(ctypes.c_void_p),
            J.ctypes.data_as(ctypes.c_void_p)
        )
        
        J_ordered = np.zeros_like(J)
        
        fn_order = getattr(libpbc, "_Original_Grid_to_Reorder_Grid", None)
        assert fn_order is not None
        
        fn_order(
            ctypes.c_int(ngrid),
            mydf.grid_ID_ordered.ctypes.data_as(ctypes.c_void_p),
            J.ctypes.data_as(ctypes.c_void_p),
            J_ordered.ctypes.data_as(ctypes.c_void_p)
        )
        
        rhoR = J_ordered.copy() 
        
    else:
        rhoR = None

    J_Res = np.zeros((nao, nao), dtype=np.float64)

    #### step 1 get_J ####

    t1 = (logger.process_clock(), logger.perf_counter())

    for aoR_holder in aoR:
        
        if with_j is False:
            continue
        
        if aoR_holder is None:
            continue
        
        if use_mpi:
            if atm_id % comm_size != rank:
                continue
        
        ngrids_now = aoR_holder.aoR.shape[1]
        nao_involved = aoR_holder.aoR.shape[0]
        
        global_gridID_begin = aoR_holder.global_gridID_begin
        rhoR_tmp = rhoR[global_gridID_begin:global_gridID_begin+ngrids_now] 
        
        aoR_rhoR_res = np.ndarray((nao_involved, ngrids_now), buffer=aoR_buf1)
        lib.d_ij_j_ij(aoR_holder.aoR, rhoR_tmp, out=aoR_rhoR_res)
        ddot_res = np.ndarray((nao_involved, nao_involved), buffer=ddot_buf)
        lib.ddot(aoR_rhoR_res, aoR_holder.aoR.T, c=ddot_res)
        
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

    J = J_Res

    if with_j is False:
        J = None

    t2 = (logger.process_clock(), logger.perf_counter())
    
    if mydf.verbose and with_j:
        _benchmark_time(t1, t2, "get_j")

    t1 = (logger.process_clock(), logger.perf_counter())

    if with_k is False:
        K = None
        return J * ngrid / vol, K

    K = np.zeros((nocc, nao), dtype=np.float64)

    #### in the following steps, mo should not be weighted ####

    occ_weight_inv = (1.0 / occ_weight).copy()
    if moR is not None:
        lib.d_i_ij_ij(occ_weight_inv, moR,  out=moR)
    if moRg is not None:
        lib.d_i_ij_ij(occ_weight_inv, moRg, out=moRg)

    #### step 2 get moRg and dmRgRg ####
        
    ### step 3. get_K ###
    
    lib.cwise_mul(mydf.W, dmRgRg, out=dmRgRg)
    W2 = dmRgRg
    if construct_dmRgR:
        lib.cwise_mul(mydf.V_R, dmRgR, out=dmRgR)
        V2 = dmRgR
    else:
        V2 = None
        
    K1 = lib.ddot(moRg, W2)       ### moRg * W2 * aoRg.T
    K1_res = np.zeros((nocc, nao), dtype=np.float64)
    if mydf.with_robust_fitting:
        K2 = lib.ddot(moRg, V2)   ### moRg * V2 * aoR.T
        K3 = lib.ddot(V2, moR.T)  ### aoRg * V2 * moR.T
        K2_res = np.zeros((nocc, nao), dtype=np.float64)
        K3_res = np.zeros((nao, nocc), dtype=np.float64)
    else:
        K2 = None
        K3 = None
    
    K = np.zeros((nocc, nao), dtype=np.float64)
    K1_packbuf = mydf.K1_packbuf
    
    ##### construct with aoRg #####
    
    for aoR_holder in mydf.aoRg:
        
        ngrids_now = aoR_holder.aoR.shape[1]
        nao_involved = aoR_holder.aoR.shape[0]
        
        ########## for (moRg * W2) * aoRg.T ##########
        
        K1_pack = np.ndarray((nocc, ngrids_now), buffer=K1_packbuf)
        
        grid_loc_now = aoR_holder.global_gridID_begin
        
        fn_packcol2(
            K1_pack.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nocc),
            ctypes.c_int(ngrids_now),
            K1.ctypes.data_as(ctypes.c_void_p),    
            ctypes.c_int(nocc),
            ctypes.c_int(naux),
            ctypes.c_int(grid_loc_now),
            ctypes.c_int(grid_loc_now+ngrids_now)
        )
        
        ddot_res = np.ndarray((nocc, nao_involved), buffer=ddot_buf)

        lib.ddot(K1_pack, aoR_holder.aoR.T, c=ddot_res)
        
        if nao_involved == nao:
            K1_res += ddot_res
        else:
            fn_packadd_col(
                K1_res.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(K1_res.shape[0]),
                ctypes.c_int(K1_res.shape[1]),
                ddot_res.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(ddot_res.shape[0]),
                ctypes.c_int(ddot_res.shape[1]),
                aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
            )
        
        ########## aoRg * (V2 * moR.T) ##########
        
        if mydf.with_robust_fitting:
            K3_pack = K3[grid_loc_now:grid_loc_now+ngrids_now, :]
            ddot_res = np.ndarray((nao_involved, nocc), buffer=ddot_buf)
            lib.ddot(aoR_holder.aoR, K3_pack, c=ddot_res)
            if nao_involved == nao:
                K3_res += ddot_res
            else:
                fn_packadd_row(
                    K3_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(K3_res.shape[0]),
                    ctypes.c_int(K3_res.shape[1]),
                    ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(ddot_res.shape[0]),
                    ctypes.c_int(ddot_res.shape[1]),
                    aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
                )
        
        grid_loc_now += ngrids_now
    
    
    if mydf.with_robust_fitting:
        
        for aoR_holder in mydf.aoR:
            
            ngrids_now = aoR_holder.aoR.shape[1]
            nao_involved = aoR_holder.aoR.shape[0]
            
            ########## (moRg * V2) * aoR.T ##########
    
            K2_pack = np.ndarray((nocc, ngrids_now), buffer=K1_packbuf)
            
            grid_loc_now = aoR_holder.global_gridID_begin
            
            fn_packcol2(
                K2_pack.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nocc),
                ctypes.c_int(ngrids_now),
                K2.ctypes.data_as(ctypes.c_void_p),    
                ctypes.c_int(nocc),
                ctypes.c_int(ngrid),
                ctypes.c_int(grid_loc_now),
                ctypes.c_int(grid_loc_now+ngrids_now)
            )
    
            ddot_res = np.ndarray((nocc, nao_involved), buffer=ddot_buf)
            
            lib.ddot(K2_pack, aoR_holder.aoR.T, c=ddot_res)
            
            if nao_involved == nao:
                K2_res += ddot_res
            else:
                fn_packadd_col(
                    K2_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(K2_res.shape[0]),
                    ctypes.c_int(K2_res.shape[1]),
                    ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(ddot_res.shape[0]),
                    ctypes.c_int(ddot_res.shape[1]),
                    aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
                )
    
    if mydf.with_robust_fitting:
        K1 = K1_res
        K2 = K2_res
        K3 = K3_res
        K = -K1 + K2 + K3.T
    else:
        K1 = K1_res
        K = K1
    
    ### delete buf ###
    
    del ddot_buf, aoR_buf1, moR_buf, mo_coeff_pack_buf
    
    t2 = (logger.process_clock(), logger.perf_counter())
    if mydf.verbose:
        _benchmark_time(t1, t2, "get_k_occRI")
    
    # Kiv = K.copy() # for debug
    
    ##### final step from Kiv -> kuv ####
    
    t1 = (logger.process_clock(), logger.perf_counter())
    
    ovlp = mydf.ovlp
    K1 = lib.ddot(mo_coeff_original, K)
    K1 = lib.ddot(ovlp, K1)
    # print("K.shape = ", K.shape)
    # print("mo_coeff_original.shape = ", mo_coeff_original.shape)
    Kij = lib.ddot(K, mo_coeff_original)
    assert np.allclose(Kij, Kij.T)
    K2 = lib.ddot(mo_coeff_original, Kij)
    K2 = lib.ddot(ovlp, K2)
    K2 = lib.ddot(K2, mo_coeff_original.T)
    K2 = lib.ddot(K2, ovlp)
    K = K1 + K1.T - K2
    
    # Kip = lib.ddot(K, mo_coeff_full)
    # Kpq = np.zeros((nao, nao), dtype=np.float64)
    # Kpq[:nocc, :] = Kip
    # Kpq[nocc:, :nocc] = Kip[:,nocc:].T
    # K = lib.ddot(mo_coeff_full, Kpq)
    # K = lib.ddot(K, mo_coeff_full.T)
        
    t2 = (logger.process_clock(), logger.perf_counter())
    t00 = t2
    if mydf.verbose:
        _benchmark_time(t1, t2, "get_k_iv_2_uv")
        _benchmark_time(t0, t00, "get_jk_occ-RI-K")
    
    del K1, K2, K3
    
    return J * ngrid / vol, K * ngrid / vol


def get_jk_dm_quadratic(mydf, dm, hermi=1, kpt=np.zeros(3),
                        kpts_band=None, with_j=True, with_k=True, omega=None, 
                        **kwargs):
    
    '''JK for given k-point'''
    
    ############ deal with occ-RI-K ############
    
    use_occ_RI_K = False
    
    if getattr(mydf, "occ_RI_K", None) is not None:
        use_occ_RI_K = mydf.occ_RI_K
    
    if getattr(dm, '__dict__', None) is not None:
        # print(dm.__dict__.keys())
        mo_coeff = dm.__dict__['mo_coeff']
        mo_occ   = dm.__dict__['mo_occ']
        if mo_coeff is not None:
            assert mo_occ is not None
            assert mo_coeff.shape[1] == mo_occ.shape[0]
            assert mo_coeff.ndim == 2
            assert mo_occ.ndim == 1
        if use_occ_RI_K and mo_coeff is None:
            # use_occ_RI_K = False 
            dm = np.asarray(dm)
            if len(dm.shape) == 3:
                assert dm.shape[0] == 1
                dm = dm[0]
            mo_occ, mo_coeff = mydf.diag_dm(dm)
            dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
            print("Dm without mo_coeff and mo_occ is provided, but use_occ_RI_K is True, so mo_coeff and mo_occ are generated from dm")
            print("mo_occ = ", mo_occ[mo_occ>1e-10])
    else:
        # mo_coeff = None
        # mo_occ = None
        # use_occ_RI_K = False
        dm = np.asarray(dm)
        if len(dm.shape) == 3:
            assert dm.shape[0] == 1
            dm = dm[0]
        if use_occ_RI_K:
            # ovlp = mydf.ovlp
            mo_occ, mo_coeff = mydf.diag_dm(dm)
            dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
            print("Dm without mo_coeff and mo_occ is provided, but use_occ_RI_K is True, so mo_coeff and mo_occ are generated from dm")
            print("mo_occ = ", mo_occ[mo_occ>1e-10])
        else:
            mo_occ = None
            mo_coeff = None
    
    if use_occ_RI_K:
        if mydf.direct == True:
            raise ValueError("ISDF does not support direct=True for occ-RI-K")
    
    ############ end deal with occ-RI-K ############
    
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

    dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)

    ############ end deal with dm with tags ############

    #### perform the calculation ####

    if "exxdiv" in kwargs:
        exxdiv = kwargs["exxdiv"]
        kwargs.pop("exxdiv")
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

    if use_occ_RI_K:
        vj, vk = get_jk_occRI(mydf, dm, use_mpi, with_j, with_k)
    else:
        if with_j:
            from pyscf.pbc.df.isdf.isdf_jk import _contract_j_dm
            vj = _contract_j_dm_ls(mydf, dm, use_mpi)  
            # vj = _contract_j_dm_wo_robust_fitting(mydf, dm, use_mpi)
            # vj = _contract_j_dm(mydf, dm, use_mpi)
            # if rank == 0:
            # print("In isdf get_jk vj = ", vj[0, :16])
            # print("In isdf get_jk vj = ", vj[-1, -16:])
        if with_k:
            if mydf.direct:
                vk = _contract_k_dm_quadratic_direct(mydf, dm, use_mpi=use_mpi)
            else:
                vk = _contract_k_dm_quadratic(mydf, dm, mydf.with_robust_fitting, use_mpi=use_mpi)
            # if rank == 0:
            # print("In isdf get_jk vk = ", vk[0, :16])
            # print("In isdf get_jk vk = ", vk[-1, -16:])
            if exxdiv == 'ewald':
                print("WARNING: ISDF does not support ewald")

    if mydf.rsjk is not None:
        vj_sr, vk_sr = mydf.rsjk.get_jk(
            dm, 
            hermi, 
            kpt, 
            kpts_band, 
            with_j, 
            with_k, 
            omega, 
            exxdiv, **kwargs)
        # print("In isdf get_jk  vj_sr = ", vj_sr[0,:16])
        # print("In isdf get_jk  vj_sr = ", vj_sr[-1,-16:])
        # print("In isdf get_jk  vk_sr = ", vk_sr[0,:16])
        # print("In isdf get_jk  vk_sr = ", vk_sr[-1,-16:])
        if with_j:
            vj += vj_sr
        if with_k:
            vk += vk_sr

    t1 = log.timer('sr jk', *t1)

    return vj, vk

############# linear scaling implementation ############# 