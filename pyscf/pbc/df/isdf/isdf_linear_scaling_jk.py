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

############ sys module ############

import copy, sys
import ctypes
import numpy as np

############ pyscf module ############

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
libpbc = lib.load_library('libpbc')

############ isdf utils ############

from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time
from pyscf.pbc.df.isdf._isdf_local_K_direct import _isdf_get_K_direct_kernel_1

############ GLOBAL PARAMETER ############

J_MAX_GRID_BUNCHSIZE = 8192

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
        assert mydf.direct == True
        from pyscf.pbc.df.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast, reduce
        size = comm.Get_size()
        
    ######### prepare the parameter #########
    
    assert first_pass in [None, "only_dd", "only_cc", "exclude_cc", "all"]
    
    if first_pass is None:
        first_pass = "all"
    
    first_pass_all    = first_pass == "all"
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
    
    max_nao_involved   = np.max([aoR_holder.aoR.shape[0] for aoR_holder in aoR if aoR_holder is not None])
    max_ngrid_involved = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None])
    ngrids_local       = np.sum([aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None])
    
    density_R = np.zeros((ngrid,), dtype=np.float64)
    
    dm_buf      = np.zeros((max_nao_involved, max_nao_involved), dtype=np.float64)
    max_col_buf = min(max_ngrid_involved, J_MAX_GRID_BUNCHSIZE)
    ddot_buf    = np.zeros((max_nao_involved, max_col_buf), dtype=np.float64)
    
    fn_multiplysum = getattr(libpbc, "_fn_J_dmultiplysum", None)
    assert fn_multiplysum is not None
    
    ##### get the involved C function ##### 
    
    fn_extract_dm = getattr(libpbc, "_extract_dm_involved_ao", None) 
    assert fn_extract_dm is not None
    
    fn_extract_dm2 = getattr(libpbc, "_extract_dm_involved_ao_RS", None)
    assert fn_extract_dm is not None
    
    fn_packadd_dm = getattr(libpbc, "_packadd_local_dm", None)
    assert fn_packadd_dm is not None 
    
    #### step 1. get density value on real space grid and IPs
    
    group = mydf.group
    ngroup = len(group)
    
    density_R_tmp = None
    
    density_R_tmp_buf = np.zeros((max_ngrid_involved,), dtype=np.float64)
    
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
            fn_extract_dm(
                dm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao),
                dm_now.ctypes.data_as(ctypes.c_void_p),
                bra_ao_involved.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbra_ao),
            )
            
            # _density_R_tmp = np.zeros((ket_aoR.shape[1],), dtype=np.float64)
            _density_R_tmp = np.ndarray((ket_aoR.shape[1],), buffer=density_R_tmp_buf)
           
            for p0, p1 in lib.prange(0, ket_aoR.shape[1], J_MAX_GRID_BUNCHSIZE):
                ddot_res = np.ndarray((nbra_ao, p1-p0), buffer=ddot_buf)
                lib.ddot(dm_now, ket_aoR[:,p0:p1], c=ddot_res)
                _res_tmp = np.ndarray((p1-p0,), 
                                      dtype =_density_R_tmp.dtype, 
                                      buffer=_density_R_tmp, 
                                      offset=p0*_density_R_tmp.dtype.itemsize)
                fn_multiplysum(
                    _res_tmp.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nbra_ao),
                    ctypes.c_int(p1-p0),
                    bra_aoR.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(bra_aoR.shape[0]),
                    ctypes.c_int(bra_aoR.shape[1]),
                    ctypes.c_int(0),
                    ctypes.c_int(p0),
                    ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nbra_ao),
                    ctypes.c_int(p1-p0),
                    ctypes.c_int(0),
                    ctypes.c_int(0))
            return _density_R_tmp
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
            # _density_R_tmp = np.zeros((ket_aoR.shape[1],), dtype=np.float64)
            _density_R_tmp = np.ndarray((ket_aoR.shape[1],), buffer=density_R_tmp_buf)
            
            for p0, p1 in lib.prange(0, ket_aoR.shape[1], J_MAX_GRID_BUNCHSIZE):
                ddot_res = np.ndarray((nbra_ao, p1-p0), buffer=ddot_buf)
                lib.ddot(dm_now, ket_aoR[:,p0:p1], c=ddot_res)
                _res_tmp = np.ndarray((p1-p0,), 
                                      dtype =_density_R_tmp.dtype, 
                                      buffer=_density_R_tmp, 
                                      offset=p0*_density_R_tmp.dtype.itemsize)
                fn_multiplysum(
                    _res_tmp.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nbra_ao),
                    ctypes.c_int(p1-p0),
                    bra_aoR.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(bra_aoR.shape[0]),
                    ctypes.c_int(bra_aoR.shape[1]),
                    ctypes.c_int(0),
                    ctypes.c_int(p0),
                    ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nbra_ao),
                    ctypes.c_int(p1-p0),
                    ctypes.c_int(0),
                    ctypes.c_int(0))
            
            return _density_R_tmp * 2.0
    
    for atm_id, aoR_holder in enumerate(aoR):
        
        if aoR_holder is None:
            continue
        
        if use_mpi:
            if atm_id % comm_size != rank:
                continue
            
        ngrids_now          = aoR_holder.aoR.shape[1]
        nao_involved        = aoR_holder.aoR.shape[0]
        global_gridID_begin = aoR_holder.global_gridID_begin
        nCompact            = aoR_holder.nCompact
        
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
                    aoR_holder.aoR[:nCompact,:], 
                    aoR_holder.ao_involved[:nCompact], 
                    aoR_holder.aoR[:nCompact,:], 
                    aoR_holder.ao_involved[:nCompact],
                    "compact",
                    "compact"
                )
                
                density_R[global_gridID_begin:global_gridID_begin+ngrids_now] += density_R_tmp
            
            if first_pass_has_dd:
                density_R_tmp = _get_rhoR(
                    aoR_holder.aoR[nCompact:,:], 
                    aoR_holder.ao_involved[nCompact:], 
                    aoR_holder.aoR[nCompact:,:], 
                    aoR_holder.ao_involved[nCompact:],
                    "diffuse",
                    "diffuse"
                )
                
                density_R[global_gridID_begin:global_gridID_begin+ngrids_now] += density_R_tmp
            
            if first_pass_has_cd:
                density_R_tmp = _get_rhoR(
                    aoR_holder.aoR[:nCompact,:], 
                    aoR_holder.ao_involved[:nCompact], 
                    aoR_holder.aoR[nCompact:,:], 
                    aoR_holder.ao_involved[nCompact:],
                    "compact",
                    "diffuse"
                )                
                density_R[global_gridID_begin:global_gridID_begin+ngrids_now] += density_R_tmp
    
    # assert local_grid_loc == ngrids_local
    
    if use_mpi:
        density_R = reduce(density_R, root=0)
    else:
        assert ngrids_local == np.prod(mesh)
            
    grid_ID_ordered = mydf.grid_ID_ordered
    
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
    
    _benchmark_time(t1, t2, "half_J", mydf)
    
    return J

def _contract_j_dm_ls(mydf, dm, 
                      use_mpi     = False, 
                      first_pass  = None, 
                      second_pass = None,
                      short_range = False):
    
    if use_mpi:
        assert mydf.direct == True
        from pyscf.pbc.df.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast, reduce
        size = comm.Get_size()
    
    ###### Prepocess parameter for RS ######
    
    assert first_pass  in [None, "only_dd", "only_cc", "exclude_cc", "all"]
    assert second_pass in [None, "only_dd", "only_cc", "exclude_cc", "all"]
    
    if short_range:
        assert first_pass == "only_dd"
        assert second_pass == "only_dd"
    
    if first_pass is None:
        first_pass = "all"
    if second_pass is None:
        second_pass = "all"
    
    second_pass_all    = second_pass == "all"
    second_pass_has_dd = second_pass in ["all", "only_dd", "exclude_cc"]
    second_pass_has_cc = second_pass in ["all", "only_cc"]
    second_pass_has_cd = second_pass in ["all", "exclude_cc"]
    
    ####### judge whether to call the original one #######
    
    if isinstance(mydf.aoRg, np.ndarray):
        has_aoR = False
        if hasattr(mydf, "aoR") and mydf.aoR is not None:
            assert isinstance(mydf.aoR, np.ndarray)
            has_aoR = True
        ### call the original get_j ###
        from pyscf.pbc.df.isdf.isdf_jk import _contract_j_dm_fast, _contract_j_dm_wo_robust_fitting
        if has_aoR:
            return _contract_j_dm_fast(mydf, dm, use_mpi=use_mpi)
        else:
            return _contract_j_dm_wo_robust_fitting(mydf, dm, use_mpi=use_mpi)
    
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
    
    max_nao_involved   = np.max([aoR_holder.aoR.shape[0] for aoR_holder in aoR if aoR_holder is not None])
    max_ngrid_involved = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None])
    ngrids_local       = np.sum([aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None])
    
    density_R = np.zeros((ngrid,), dtype=np.float64)
    
    # max_dim_buf = max(max_ngrid_involved, max_nao_involved)
    max_dim_buf = max_nao_involved
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

    J_Res = np.zeros((nao, nao), dtype=np.float64)

    ordered_ao_ind = np.arange(nao)

    def _get_j_pass2_ls(_aoR_bra, 
                        _ao_involved_bra, 
                        _aoR_ket,
                        _ao_involved_ket,
                        _bra_type,
                        _ket_type,
                        _potential,
                        _Res):
        
        nao_bra = _aoR_bra.shape[0]
        nao_ket = _aoR_ket.shape[0]
                
        if _bra_type == _ket_type:
            
            aoR_J_res = np.ndarray(_aoR_ket.shape, buffer=aoR_buf1)
            lib.d_ij_j_ij(_aoR_ket, _potential, out=aoR_J_res)
            ddot_res = np.ndarray((nao_ket, nao_ket), buffer=ddot_buf)
            lib.ddot(_aoR_ket, aoR_J_res.T, c=ddot_res)
            
            if nao_ket == nao and np.allclose(_ao_involved_ket, ordered_ao_ind):
                _Res += ddot_res
            else:
                fn_packadd_dm(
                    ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao_ket),
                    _ao_involved_ket.ctypes.data_as(ctypes.c_void_p),
                    _Res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(_Res.shape[0])
                )
        else:
            
            ### J_Res = ddot_res + ddot_res.T
            
            aoR_J_res = np.ndarray(_aoR_ket.shape, buffer=aoR_buf1)
            lib.d_ij_j_ij(_aoR_ket, _potential, out=aoR_J_res)
            ddot_res = np.ndarray((nao_bra, nao_ket), buffer=ddot_buf)
            lib.ddot(_aoR_bra, aoR_J_res.T, c=ddot_res)
            
            fn_packadd_dm2(
                ddot_res.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_bra),
                _ao_involved_bra.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_ket),
                _ao_involved_ket.ctypes.data_as(ctypes.c_void_p),
                _Res.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(_Res.shape[0])
            )


    for atm_id, aoR_holder in enumerate(aoR):
        
        if aoR_holder is None:
            continue
        
        if use_mpi:
            if atm_id % comm_size != rank:
                continue
        
        ngrids_now   = aoR_holder.aoR.shape[1]
        nao_involved = aoR_holder.nao_involved
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

    J = J_Res

    if use_mpi:
        J = reduce(J, root=0)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    _benchmark_time(t1, t2, "_contract_j_dm_fast", mydf)
    
    ######### delete the buffer #########

    del ddot_buf 
    del aoR_buf1
    
    return J * ngrid / vol

def _contract_j_dm_wo_robust_fitting(mydf, dm, use_mpi=False):
    
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        assert mydf.direct == True
    
    ####### judge whether to call the original one #######
    
    if isinstance(mydf.aoRg, np.ndarray):
        from pyscf.pbc.df.isdf.isdf_jk import _contract_j_dm_wo_robust_fitting
        _contract_j_dm_wo_robust_fitting(mydf, dm, use_mpi=use_mpi)
    
    ######## start the calculation ########
    
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
    
    density_Rg = np.zeros((naux,), dtype=np.float64)
    
    dm_buf = np.zeros((max_nao_involved, max_nao_involved), dtype=np.float64)
    max_dim_buf = max(max_ngrid_involved, max_nao_involved)
    ddot_buf = np.zeros((max_dim_buf, max_dim_buf), dtype=np.float64)
    aoR_buf1 = np.zeros((max_nao_involved, max_ngrid_involved), dtype=np.float64)
    
    ##### get the involved C function ##### 
    
    fn_extract_dm = getattr(libpbc, "_extract_dm_involved_ao", None) 
    assert fn_extract_dm is not None
    
    fn_packadd_dm = getattr(libpbc, "_packadd_local_dm", None)
    assert fn_packadd_dm is not None
    
    #### step 1. get density value on real space grid and IPs

    group = mydf.group
    ngroup = len(group)
    
    density_R_tmp = None
    ordered_ao_ind = np.arange(nao)
    
    for atm_id, aoR_holder in enumerate(aoRg):
        
        if aoR_holder is None:
            continue
        
        if use_mpi:
            if atm_id % comm_size != rank:
                continue
            
        ngrids_now = aoR_holder.aoR.shape[1]
        nao_involved = aoR_holder.aoR.shape[0]
        
        if nao_involved < nao or (nao_involved == nao and not np.allclose(aoR_holder.ao_involved, ordered_ao_ind)):
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
        
    if use_mpi == False:
        assert ngrids_local == naux
    
    if use_mpi:
        density_Rg = reduce(density_Rg, root=0)
        J = bcast(J, root=0)
    
    #### step 3. get J 
    
    J = np.asarray(lib.dot(W, density_Rg.reshape(-1,1)), order='C').reshape(-1)
    
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
        
        if nao_involved == nao and np.allclose(aoR_holder.ao_involved, ordered_ao_ind):
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

    if use_mpi:
        J = reduce(J, root=0)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    _benchmark_time(t1, t2, "_contract_j_dm_fast", mydf)
    
    ######### delete the buffer #########
    
    del dm_buf, ddot_buf, density_Rg
    del density_R_tmp
    del aoR_buf1
    
    return J * ngrid / vol

############# quadratic scaling (not cubic!) #############

def __get_DensityMatrixonRgAO_qradratic(mydf, dm, 
                                        bra_aoR_holder, 
                                        bra_type       = None,
                                        res:np.ndarray = None, 
                                        verbose        = 1):
    
    assert bra_type in [None, "all", "compact", "diffuse"]
        
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

    #print("nao       = ", nao)
    #print("ngrid_bra = ", ngrid_bra)

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
    
    ordered_ao_ind = np.arange(nao)
    grid_shift = None
    ngrid_loc = 0
    
    for aoR_holder in bra_aoR_holder:
        
        if aoR_holder is None:
            continue
        
        ngrid_now = aoR_holder.aoR.shape[1]
        nao_involved = aoR_holder.aoR.shape[0]
        nao_compact  = aoR_holder.nCompact
        
        #print("ngrid_now    = ", ngrid_now)
        #print("nao_involved = ", nao_involved)
        
        ao_begin_indx = 0
        ao_end_indx   = nao_involved
        if bra_type == "compact":
            ao_end_indx = nao_compact
        elif bra_type == "diffuse":
            ao_begin_indx = nao_compact 
        
        nao_at_work = ao_end_indx - ao_begin_indx
        
        if (nao_at_work) == nao and np.allclose(aoR_holder.ao_involved, ordered_ao_ind):
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
        #print("grid_loc_begin = ", grid_loc_begin)
    
        if grid_shift is None:
            grid_shift = grid_loc_begin
        else:
            assert grid_loc_begin>=grid_shift
        
        res[grid_loc_begin-grid_shift:grid_loc_begin-grid_shift+ngrid_now, :] = ddot_res
        
        # ngrid_loc += ngrid_now   
    #assert ngrid_loc == ngrid_bra
        
    t2 = (logger.process_clock(), logger.perf_counter())
    # if verbose>0:
    #     _benchmark_time(t1, t2, "__get_DensityMatrixonRgAO_qradratic", mydf)
    return res

def _contract_k_dm_quadratic(mydf, dm, with_robust_fitting=True, use_mpi=False):
    
    if use_mpi:
        raise NotImplementedError("MPI is not supported yet.")
    
    ####### judge whether to call the original one #######
    
    if isinstance(mydf.aoRg, np.ndarray):
        from pyscf.pbc.df.isdf.isdf_jk import _contract_k_dm, _contract_k_dm_wo_robust_fitting
        if mydf.aoR is None:
            return _contract_k_dm_wo_robust_fitting(mydf, dm, False, use_mpi=use_mpi)
        else:
            return _contract_k_dm(mydf, dm, with_robust_fitting, use_mpi=use_mpi)
    
    ######## start the calculation ########
    
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
    
    ordered_ao_ind = np.arange(nao)
    
    ### TODO: consider MPI 
    
    nIP_loc = 0
    for aoRg_holder in aoRg:
        
        if aoRg_holder is None:
            continue
    
        nIP_now = aoRg_holder.aoR.shape[1]
        nao_involved = aoRg_holder.aoR.shape[0]
        
        #### pack the density matrix ####
        
        if nao_involved == nao and np.allclose(aoRg_holder.ao_involved, ordered_ao_ind):
            Density_RgAO_packed = Density_RgAO
        else:
            # Density_RgAO_packed = Density_RgAO[:, aoRg_holder.ao_involved]
            Density_RgAO_packed = np.ndarray((naux, nao_involved), buffer=pack_buf)
            fn_packcol1(
                Density_RgAO_packed.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(naux),
                ctypes.c_int(nao_involved),
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
        
        ddot_res = np.ndarray((naux, nao_involved), buffer=ddot_buf2)
        lib.ddot(W_tmp, aoRg_holder.aoR.T, c=ddot_res)
        
        if nao_involved == nao and np.allclose(aoRg_holder.ao_involved, ordered_ao_ind):
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
        nao_involved = aoRg_holder.aoR.shape[0]
        
        K_tmp = K1[nIP_loc:nIP_loc+nIP_now, :]
        
        ddot_res = np.ndarray((nao_involved, nao), buffer=ddot_res_buf)
        lib.ddot(aoRg_holder.aoR, K_tmp, c=ddot_res)
        
        if nao_involved == nao and np.allclose(aoRg_holder.ao_involved, ordered_ao_ind):
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
        offset    = naux * max_nao_involved * ddot_res_buf.dtype.itemsize
        V_tmp_buf = np.ndarray((naux, max_ngrid_involved), buffer=ddot_res_buf, offset=offset)
        offset   += V_tmp_buf.size * V_tmp_buf.dtype.itemsize
        pack_buf  = np.ndarray((naux, max_nao_involved), buffer=ddot_res_buf, offset=offset)
        offset   += pack_buf.size * pack_buf.dtype.itemsize
        ddot_buf2 = np.ndarray((naux, max_ngrid_involved), buffer=ddot_res_buf, offset=offset)
    
        ngrid_loc = 0
        
        for aoR_holder in aoR:
            
            if aoR_holder is None:
                continue
            
            ngrid_now = aoR_holder.aoR.shape[1]
            nao_involved = aoR_holder.aoR.shape[0]
            
            #### pack the density matrix ####
            
            if nao_involved == nao and np.allclose(aoR_holder.ao_involved, ordered_ao_ind):
                Density_RgAO_packed = Density_RgAO_packed
            else:
                # Density_RgAO_packed = Density_RgAO[:, aoR_holder.ao_involved]
                Density_RgAO_packed = np.ndarray((naux, nao_involved), buffer=pack_buf)
                fn_packcol1(
                    Density_RgAO_packed.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(naux),
                    ctypes.c_int(nao_involved),
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
                        
            ddot_res = np.ndarray((naux, nao_involved), buffer=ddot_buf1)
            lib.ddot(V_tmp, aoR_holder.aoR.T, c=ddot_res)
            
            if nao_involved == nao and np.allclose(aoR_holder.ao_involved, ordered_ao_ind):
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
            nao_involved = aoRg_holder.aoR.shape[0]
            
            K_tmp = K2[nIP_loc:nIP_loc+nIP_now, :] # no need to pack, continguous anyway
            
            ddot_res = np.ndarray((nao_involved, nao), buffer=ddot_res_buf)
            lib.ddot(aoRg_holder.aoR, K_tmp, c=ddot_res)
            
            if nao == nao_involved and np.allclose(aoRg_holder.ao_involved, ordered_ao_ind):
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
    
    # if mydf.verbose:
    _benchmark_time(t1, t2, "_contract_k_dm_quadratic", mydf)
    
    return K * ngrid / vol

def _contract_k_dm_quadratic_direct(mydf, dm, use_mpi=False):
    
    if use_mpi:
        assert mydf.direct == True
        from pyscf.pbc.df.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast, reduce
        size = comm.Get_size()
    
    t1 = (logger.process_clock(), logger.perf_counter())
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]
        
    aoR = mydf.aoR
    aoRg = mydf.aoRg    
    
    #max_nao_involved   = np.max([aoR_holder.aoR.shape[0] for aoR_holder in aoR if aoR_holder is not None])
    #max_ngrid_involved = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None])
    #max_nIP_involved   = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoRg if aoR_holder is not None])
    
    max_nao_involved   = mydf.max_nao_involved
    max_ngrid_involved = mydf.max_ngrid_involved
    max_nIP_involved   = mydf.max_nIP_involved
    
    maxsize_group_naux = mydf.maxsize_group_naux
        
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
    
    # if hasattr(mydf, "atm_ordering"):
    #     atm_ordering = mydf.atm_ordering
    # else:
    #     atm_ordering = []
    #     for group_idx, atm_idx in enumerate(group):
    #         atm_idx.sort()
    #         atm_ordering.extend(atm_idx)
    #     atm_ordering = np.array(atm_ordering, dtype=np.int32)
    #     mydf.atm_ordering = atm_ordering
    
    # def construct_V(aux_basis:np.ndarray, buf, V, grid_ID, grid_ordering):
    #     fn = getattr(libpbc, "_construct_V_local_bas", None)
    #     assert(fn is not None)
    #     nThread = buf.shape[0]
    #     bufsize_per_thread = buf.shape[1]
    #     nrow = aux_basis.shape[0]
    #     ncol = aux_basis.shape[1]
    #     shift_row = 0
    #     fn(mesh_int32.ctypes.data_as(ctypes.c_void_p),
    #             ctypes.c_int(nrow),
    #             ctypes.c_int(ncol),
    #             grid_ID.ctypes.data_as(ctypes.c_void_p),
    #             aux_basis.ctypes.data_as(ctypes.c_void_p),
    #             coulG_real.ctypes.data_as(ctypes.c_void_p),
    #             ctypes.c_int(shift_row),
    #             V.ctypes.data_as(ctypes.c_void_p),
    #             grid_ordering.ctypes.data_as(ctypes.c_void_p),
    #             buf.ctypes.data_as(ctypes.c_void_p),
    #             ctypes.c_int(bufsize_per_thread))
    
    ######### allocate buffer ######### 
        
    Density_RgAO_buf = mydf.Density_RgAO_buf
    # print(Density_RgAO_buf.shape)
    
    nThread            = lib.num_threads()
    bufsize_per_thread = (coulG_real.shape[0] * 2 + np.prod(mesh))
    buf_build_V        = np.ndarray((nThread, bufsize_per_thread), dtype=np.float64, buffer=build_VW_buf) 
    
    offset_now = buf_build_V.size * buf_build_V.dtype.itemsize
    
    build_K_bunchsize = min(maxsize_group_naux, mydf._build_K_bunchsize)
    
    offset_build_now       = 0
    offset_Density_RgR_buf = 0
    Density_RgR_buf        = np.ndarray((build_K_bunchsize, ngrid), buffer=build_k_buf, offset=offset_build_now)
    
    offset_build_now        += Density_RgR_buf.size * Density_RgR_buf.dtype.itemsize
    offset_ddot_res_RgR_buf  = offset_build_now
    ddot_res_RgR_buf         = np.ndarray((build_K_bunchsize, max_ngrid_involved), buffer=build_k_buf, offset=offset_ddot_res_RgR_buf)
    
    offset_build_now   += ddot_res_RgR_buf.size * ddot_res_RgR_buf.dtype.itemsize
    offset_K1_tmp1_buf  = offset_build_now
    K1_tmp1_buf         = np.ndarray((maxsize_group_naux, nao), buffer=build_k_buf, offset=offset_K1_tmp1_buf)
    
    offset_build_now            += K1_tmp1_buf.size * K1_tmp1_buf.dtype.itemsize
    offset_K1_tmp1_ddot_res_buf  = offset_build_now
    K1_tmp1_ddot_res_buf         = np.ndarray((maxsize_group_naux, nao), buffer=build_k_buf, offset=offset_K1_tmp1_ddot_res_buf)
    
    offset_build_now += K1_tmp1_ddot_res_buf.size * K1_tmp1_ddot_res_buf.dtype.itemsize

    offset_K1_final_ddot_buf = offset_build_now
    K1_final_ddot_buf        = np.ndarray((nao, nao), buffer=build_k_buf, offset=offset_K1_final_ddot_buf)
    
    ########### get involved C function ###########
    
    fn_packcol1 = getattr(libpbc, "_buildK_packcol", None)
    assert fn_packcol1 is not None
    fn_packcol2 = getattr(libpbc, "_buildK_packcol2", None)
    assert fn_packcol2 is not None
    fn_packadd_col = getattr(libpbc, "_buildK_packaddcol", None)
    assert fn_packadd_col is not None
    fn_packadd_row = getattr(libpbc, "_buildK_packaddrow", None)
    assert fn_packadd_row is not None

    ordered_ao_ind = np.arange(nao)

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
        
        Density_RgAO_tmp        = np.ndarray((naux_tmp, nao), buffer=Density_RgAO_buf)
        offset_density_RgAO_buf = Density_RgAO_tmp.size * Density_RgAO_buf.dtype.itemsize
        Density_RgAO_tmp.ravel()[:] = 0.0
        Density_RgAO_tmp            = __get_DensityMatrixonRgAO_qradratic(mydf, dm, aoRg_holders, "all", Density_RgAO_tmp, verbose=mydf.verbose)
        
        #### 2. build the V matrix #### 
        
        W_tmp = _isdf_get_K_direct_kernel_1(
            mydf, coulG_real,
            group_id, Density_RgAO_tmp,
            None, True,
            ##### buffer #####
            buf_build_V,
            build_VW_buf,
            offset_now,
            Density_RgR_buf,
            Density_RgAO_buf,
            offset_density_RgAO_buf,
            ddot_res_RgR_buf,
            K1_tmp1_buf,
            K1_tmp1_ddot_res_buf,
            K1_final_ddot_buf,
            ##### bunchsize #####
            #maxsize_group_naux,
            build_K_bunchsize,
            ##### other info #####
            use_mpi=use_mpi,
            ##### out #####
            K1_or_2=K1)
        
        # V_tmp              = np.ndarray((naux_tmp, ngrid), buffer=build_VW_buf, offset=offset_now, dtype=np.float64)
        # offset_after_V_tmp = offset_now + V_tmp.size * V_tmp.dtype.itemsize
        # aux_basis_grip_ID  = mydf.partition_group_to_gridID[group_id]
        # construct_V(aux_basis_tmp, buf_build_V, V_tmp, aux_basis_grip_ID, grid_ordering)
        
        #### 3. build the K1 matrix ####
        
        ###### 3.1 build density RgR
        
        # Density_RgR_tmp = np.ndarray((naux_tmp, ngrid), buffer=Density_RgR_buf)   # can be extremely large! 
        # for atm_id in atm_ordering:
        #     aoR_holder = aoR[atm_id]
        #     if aoR_holder is None:
        #         raise ValueError("aoR_holder is None")
        #     ngrid_now = aoR_holder.aoR.shape[1]
        #     nao_involved = aoR_holder.aoR.shape[0]
        #     if nao_involved == nao:
        #         Density_RgAO_packed = Density_RgAO_tmp
        #     else:
        #         # Density_RgAO_packed = Density_RgAO[:, aoR_holder.ao_involved]
        #         Density_RgAO_packed = np.ndarray((naux_tmp, nao_involved), buffer=Density_RgAO_buf, offset=offset_density_RgAO_buf)
        #         fn_packcol1(
        #             Density_RgAO_packed.ctypes.data_as(ctypes.c_void_p),
        #             ctypes.c_int(naux_tmp),
        #             ctypes.c_int(nao_involved),
        #             Density_RgAO_tmp.ctypes.data_as(ctypes.c_void_p),
        #             ctypes.c_int(naux_tmp),
        #             ctypes.c_int(nao),
        #             aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
        #         )
        #     grid_begin = aoR_holder.global_gridID_begin
        #     ddot_res_RgR = np.ndarray((naux_tmp, ngrid_now), buffer=ddot_res_RgR_buf)
        #     lib.ddot(Density_RgAO_packed, aoR_holder.aoR, c=ddot_res_RgR)
        #     Density_RgR_tmp[:, grid_begin:grid_begin+ngrid_now] = ddot_res_RgR        
        # Density_RgR = Density_RgR_tmp
                
        #### 3.2 V_tmp = Density_RgR * V
        
        #lib.cwise_mul(V_tmp, Density_RgR, out=Density_RgR)
        #V2_tmp = Density_RgR
        
        #### 3.3 K1_tmp1 = V2_tmp * aoR.T
        
        # K1_tmp1 = np.ndarray((naux_tmp, nao), buffer=K1_tmp1_buf)
        # K1_tmp1.ravel()[:] = 0.0
        # # ngrid_loc = 0
        # for atm_id in atm_ordering:
        #     aoR_holder = aoR[atm_id]   
        #     if aoR_holder is None:
        #         raise ValueError("aoR_holder is None")
        #     ngrid_now = aoR_holder.aoR.shape[1]
        #     nao_involved = aoR_holder.aoR.shape[0] 
        #     ddot_res = np.ndarray((naux_tmp, nao_involved), buffer=K1_tmp1_ddot_res_buf)
        #     grid_loc_begin = aoR_holder.global_gridID_begin
        #     V_packed = np.ndarray((naux_tmp, ngrid_now), buffer=V_pack_buf)
        #     fn_packcol2(
        #         V_packed.ctypes.data_as(ctypes.c_void_p),
        #         ctypes.c_int(naux_tmp),
        #         ctypes.c_int(ngrid_now),
        #         V2_tmp.ctypes.data_as(ctypes.c_void_p),
        #         ctypes.c_int(naux_tmp),
        #         ctypes.c_int(ngrid),
        #         ctypes.c_int(grid_loc_begin),
        #         ctypes.c_int(grid_loc_begin+ngrid_now)
        #     )
        #     lib.ddot(V_packed, aoR_holder.aoR.T, c=ddot_res)
        #     if nao_involved == nao and np.allclose(aoR_holder.ao_involved, ordered_ao_ind):
        #         K1_tmp1 += ddot_res
        #     else:
        #         # K1_tmp1[: , aoR_holder.ao_involved] += ddot_res
        #         fn_packadd_col(
        #             K1_tmp1.ctypes.data_as(ctypes.c_void_p),
        #             ctypes.c_int(K1_tmp1.shape[0]),
        #             ctypes.c_int(K1_tmp1.shape[1]),
        #             ddot_res.ctypes.data_as(ctypes.c_void_p),
        #             ctypes.c_int(ddot_res.shape[0]),
        #             ctypes.c_int(ddot_res.shape[1]),
        #             aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
        #         )
        #     # ngrid_loc += ngrid_now
        # # assert ngrid_loc == ngrid
        
        #### 3.4 K1 += aoRg * K1_tmp1
        
        # ngrid_loc = 0        
        # for atm_id in atm_ids:
        #     aoRg_holder = aoRg[atm_id]
        #     if aoRg_holder is None:
        #         raise ValueError("aoRg_holder is None")
        #     nIP_now = aoRg_holder.aoR.shape[1]
        #     nao_involved = aoRg_holder.aoR.shape[0]
        #     # grid_loc_begin = aoRg_holder.global_gridID_begin
        #     # print("grid_loc_begin = ", grid_loc_begin)
        #     # print("nIP_now = ", nIP_now)
        #     K_tmp = K1_tmp1[ngrid_loc:ngrid_loc+nIP_now, :]
        #     # print("K_tmp.shape = ", K_tmp.shape)
        #     # print("aoRg_holder.aoR.shape = ", aoRg_holder.aoR.shape)
        #     ddot_res = np.ndarray((nao_involved, nao), buffer=K1_final_ddot_buf)
        #     lib.ddot(aoRg_holder.aoR, K_tmp, c=ddot_res)
        #     if nao_involved == nao and np.allclose(aoRg_holder.ao_involved, ordered_ao_ind):
        #         K1 += ddot_res
        #     else:
        #         # K1[aoRg_holder.ao_involved, :] += ddot_res
        #         fn_packadd_row(
        #             K1.ctypes.data_as(ctypes.c_void_p),
        #             ctypes.c_int(K1.shape[0]),
        #             ctypes.c_int(K1.shape[1]),
        #             ddot_res.ctypes.data_as(ctypes.c_void_p),
        #             ctypes.c_int(ddot_res.shape[0]),
        #             ctypes.c_int(ddot_res.shape[1]),
        #             aoRg_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
        #         )
        #     ngrid_loc += nIP_now
        # assert ngrid_loc == naux_tmp
        
        #### 4. build the W matrix ####
        
        # W_tmp = np.ndarray((naux_tmp, naux), dtype=np.float64, buffer=build_VW_buf, offset=offset_after_V_tmp)
        # grid_shift = 0
        # aux_col_loc = 0
        # for j in range(len(group)):
        #     grid_ID_now = mydf.partition_group_to_gridID[j]
        #     aux_bas_ket = aux_basis[j]
        #     naux_ket = aux_bas_ket.shape[0]
        #     ngrid_now = grid_ID_now.size
        #     W_tmp[:, aux_col_loc:aux_col_loc+naux_ket] = lib.ddot(V_tmp[:, grid_shift:grid_shift+ngrid_now], aux_bas_ket.T)
        #     grid_shift += ngrid_now
        #     aux_col_loc += naux_ket
        # assert grid_shift == ngrid
        
        _isdf_get_K_direct_kernel_1(
            mydf, coulG_real,
            group_id, Density_RgAO_tmp,
            W_tmp, False,
            ##### buffer #####
            buf_build_V,
            build_VW_buf,
            offset_now,
            Density_RgR_buf,
            Density_RgAO_buf,
            offset_density_RgAO_buf,
            ddot_res_RgR_buf,
            K1_tmp1_buf,
            K1_tmp1_ddot_res_buf,
            K1_final_ddot_buf,
            ##### bunchsize #####
            #maxsize_group_naux,
            build_K_bunchsize,
            ##### other info #####
            use_mpi=use_mpi,
            ##### out #####
            K1_or_2=K2)
        
        #### 5. build the K2 matrix ####
        
        ###### 5.1 build density RgRg
        
        # Density_RgRg_tmp = np.ndarray((naux_tmp, naux), buffer=Density_RgR_buf)
        # nIP_loc = 0
        # for atm_id in atm_ordering:
        #     aoRg_holder = aoRg[atm_id]
        #     if aoRg_holder is None:
        #         raise ValueError("aoRg_holder is None")
        #     nIP_now = aoRg_holder.aoR.shape[1]
        #     nao_involved = aoRg_holder.aoR.shape[0]
        #     if nao_involved == nao and np.allclose(aoRg_holder.ao_involved, ordered_ao_ind):
        #         Density_RgAO_packed = Density_RgAO_tmp
        #     else:
        #         # Density_RgAO_packed = Density_RgAO[:, aoRg_holder.ao_involved]
        #         Density_RgAO_packed = np.ndarray((naux_tmp, nao_involved), buffer=Density_RgAO_buf, offset=offset_density_RgAO_buf)
        #         fn_packcol1(
        #             Density_RgAO_packed.ctypes.data_as(ctypes.c_void_p),
        #             ctypes.c_int(naux_tmp),
        #             ctypes.c_int(nao_involved),
        #             Density_RgAO_tmp.ctypes.data_as(ctypes.c_void_p),
        #             ctypes.c_int(naux_tmp),
        #             ctypes.c_int(nao),
        #             aoRg_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
        #         )
        #     assert nIP_loc == aoRg_holder.global_gridID_begin
        #     ddot_res_RgRg = np.ndarray((naux_tmp, nIP_now), buffer=ddot_res_RgR_buf)
        #     lib.ddot(Density_RgAO_packed, aoRg_holder.aoR, c=ddot_res_RgRg)
        #     Density_RgRg_tmp[:, nIP_loc:nIP_loc+nIP_now] = ddot_res_RgRg
        #     nIP_loc += nIP_now
        # assert nIP_loc == naux 
        # Density_RgRg = Density_RgRg_tmp
        
        #### 5.2 W_tmp = Density_RgRg * W
        
        # lib.cwise_mul(W_tmp, Density_RgRg, out=Density_RgRg)
        # W2_tmp = Density_RgRg
        
        #### 5.3 K2_tmp1 = W2_tmp * aoRg.T
        
        # K2_tmp1 = np.ndarray((naux_tmp, nao), buffer=K1_tmp1_buf)
        # K2_tmp1.ravel()[:] = 0.0
        # nIP_loc = 0
        # for atm_id in atm_ordering:
        #     aoRg_holder = aoRg[atm_id]
        #     if aoRg_holder is None:
        #         raise ValueError("aoRg_holder is None")
        #     nIP_now = aoRg_holder.aoR.shape[1]
        #     nao_involved = aoRg_holder.aoR.shape[0]
        #     ddot_res = np.ndarray((naux_tmp, nao_involved), buffer=K1_tmp1_ddot_res_buf)
        #     # W_packed = np.ndarray((naux_tmp, nIP_now), buffer=V_pack_buf)
        #     # fn_packcol2(
        #     #     W_packed.ctypes.data_as(ctypes.c_void_p),
        #     #     ctypes.c_int(naux_tmp),
        #     #     ctypes.c_int(nIP_now),
        #     #     W2_tmp.ctypes.data_as(ctypes.c_void_p),
        #     #     ctypes.c_int(naux_tmp),
        #     #     ctypes.c_int(naux),
        #     #     ctypes.c_int(nIP_loc),
        #     #     ctypes.c_int(nIP_loc+nIP_now)
        #     # )
        #     # lib.ddot(W_packed, aoRg_holder.aoR.T, c=ddot_res)
        #     lib.ddot(W2_tmp[:, nIP_loc:nIP_loc+nIP_now], aoRg_holder.aoR.T, c=ddot_res)
        #     if nao_involved == nao and np.allclose(aoRg_holder.ao_involved, ordered_ao_ind):
        #         K2_tmp1 += ddot_res
        #     else:
        #         fn_packadd_col(
        #             K2_tmp1.ctypes.data_as(ctypes.c_void_p),
        #             ctypes.c_int(K2_tmp1.shape[0]),
        #             ctypes.c_int(K2_tmp1.shape[1]),
        #             ddot_res.ctypes.data_as(ctypes.c_void_p),
        #             ctypes.c_int(ddot_res.shape[0]),
        #             ctypes.c_int(ddot_res.shape[1]),
        #             aoRg_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
        #         )
        #     nIP_loc += nIP_now
        
        #### 5.4 K2 += aoRg * K2_tmp1
        
        # nIP_loc = 0
        # for atm_id in atm_ids:
        #     aoRg_holder = aoRg[atm_id]
        #     if aoRg_holder is None:
        #         raise ValueError("aoRg_holder is None")
        #     nIP_now = aoRg_holder.aoR.shape[1]
        #     nao_involved = aoRg_holder.aoR.shape[0]
        #     K_tmp = K2_tmp1[nIP_loc:nIP_loc+nIP_now, :]
        #     ddot_res = np.ndarray((nao_involved, nao), buffer=K1_final_ddot_buf)
        #     lib.ddot(aoRg_holder.aoR, K_tmp, c=ddot_res)
        #     if nao_involved == nao and np.allclose(aoRg_holder.ao_involved, ordered_ao_ind):
        #         K2 += ddot_res
        #     else:
        #         # K2[aoRg_holder.ao_involved, :] += ddot_res
        #         fn_packadd_row(
        #             K2.ctypes.data_as(ctypes.c_void_p),
        #             ctypes.c_int(K2.shape[0]),
        #             ctypes.c_int(K2.shape[1]),
        #             ddot_res.ctypes.data_as(ctypes.c_void_p),
        #             ctypes.c_int(ddot_res.shape[0]),
        #             ctypes.c_int(ddot_res.shape[1]),
        #             aoRg_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
        #         )
        #     nIP_loc += nIP_now
        # assert nIP_loc == naux_tmp
        
    ######### finally delete the buffer #########
    
    if use_mpi:
        comm.Barrier()
    
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
    
    #if mydf.verbose:
    _benchmark_time(t1, t2, "_contract_k_dm_quadratic_direct", mydf)
        
    # print("K = ", K[0])
        
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
    
    #if mydf.verbose:
    _benchmark_time(t3, t4, "get_mo over grids", mydf)
        #sys.stdout.flush()

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
    
    #if mydf.verbose:
    _benchmark_time(t3, t4, "get_dm over grids", mydf)
        
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

    ordered_ao_ind = np.arange(nao, dtype=np.int32)

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
        
        if nao_involved == nao and np.allclose(aoR_holder.ao_involved, ordered_ao_ind):
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
    
    if with_j:
        _benchmark_time(t1, t2, "get_j", mydf)

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
        
        if nao_involved == nao and np.allclose(aoR_holder.ao_involved, ordered_ao_ind):
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
            if nao_involved == nao and np.allclose(aoR_holder.ao_involved, ordered_ao_ind):
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
            
            if nao_involved == nao and np.allclose(aoR_holder.ao_involved, ordered_ao_ind):
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

    _benchmark_time(t1, t2, "get_k_occRI", mydf)
    
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
    
    _benchmark_time(t1, t2, "get_k_iv_2_uv", mydf)
    _benchmark_time(t0, t00, "get_jk_occ-RI-K", mydf)
    
    del K1, K2, K3
    
    return J * ngrid / vol, K * ngrid / vol


def get_jk_dm_quadratic(mydf, dm, hermi=1, kpt=np.zeros(3),
                        kpts_band=None, with_j=True, with_k=True, omega=None, 
                        **kwargs):
    
    '''JK'''
    
    ############ deal with occ-RI-K ############
    
    use_occ_RI_K = False
    
    if getattr(mydf, "occ_RI_K", None) is not None:
        use_occ_RI_K = mydf.occ_RI_K
    
    if getattr(dm, '__dict__', None) is not None:
        mo_coeff = dm.__dict__['mo_coeff']
        mo_occ   = dm.__dict__['mo_occ']
        if mo_coeff is not None:
            assert mo_occ is not None
            if mo_coeff.ndim == 3:
                assert mo_coeff.shape[2] == mo_occ.shape[1]
                assert mo_occ.ndim == 2
            else:
                assert mo_coeff.shape[1] == mo_occ.shape[0]
                assert mo_coeff.ndim == 2
                assert mo_occ.ndim == 1
        if use_occ_RI_K and mo_coeff is None:
            dm = np.asarray(dm)
            if len(dm.shape) == 3:
                assert dm.shape[0] == 1
                dm = dm[0]
            mo_occ, mo_coeff = mydf.diag_dm(dm)
            dm = dm.reshape(1, dm.shape[0], dm.shape[1])
            dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    else:
        dm = np.asarray(dm)
        if len(dm.shape) == 3:
            assert dm.shape[0] <= 2
        if use_occ_RI_K:
            assert dm.shape[0] == 1
            dm = dm[0]
            mo_occ, mo_coeff = mydf.diag_dm(dm)
            dm = dm.reshape(1, dm.shape[0], dm.shape[1])
            dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        else:
            mo_occ = None
            mo_coeff = None
    
    if use_occ_RI_K:
        if mydf.direct == True:
            raise ValueError("ISDF does not support direct=True for occ-RI-K")
    
    assert dm.ndim == 3
    
    ############ end deal with occ-RI-K ############
    
    direct  = mydf.direct
    use_mpi = mydf.use_mpi
    
    if use_mpi and direct == False:
        raise NotImplementedError("ISDF does not support use_mpi and direct=False")
    
    if len(dm.shape) == 3:
        assert dm.shape[0] <= 2
        #dm = dm[0]

    if hasattr(mydf, 'Ls') and mydf.Ls is not None:
        from pyscf.pbc.df.isdf.isdf_tools_densitymatrix import symmetrize_dm
        dm = symmetrize_dm(dm, mydf.Ls)
    else:
        if hasattr(mydf, 'kmesh') and mydf.kmesh is not None:
            from pyscf.pbc.df.isdf.isdf_tools_densitymatrix import symmetrize_dm
            dm = symmetrize_dm(dm, mydf.kmesh)

    if use_mpi:
        from pyscf.pbc.df.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast, reduce
        dm = bcast(dm, root=0)

    dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)

    nset, nao = dm.shape[:2]

    ############ end deal with dm with tags ############

    #### perform the calculation ####

    if "exxdiv" in kwargs:
        exxdiv = kwargs["exxdiv"]
        kwargs.pop("exxdiv")
    else:
        exxdiv = None

    assert exxdiv in ["ewald", None]

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
        vj = np.zeros_like(dm)
        vk = np.zeros_like(dm)
        for iset in range(nset):
            if with_j:
                from pyscf.pbc.df.isdf.isdf_jk import _contract_j_dm
                vj[iset] = _contract_j_dm_ls(mydf, dm[iset], use_mpi)  
            if with_k:
                if mydf.direct:
                    vk[iset] = _contract_k_dm_quadratic_direct(mydf, dm[iset], use_mpi=use_mpi)
                    if iset >= 1:
                        logger.warn(mydf, "Current implementation with nset >= 2 is not efficient.")
                else:
                    vk[iset] = _contract_k_dm_quadratic(mydf, dm[iset], mydf.with_robust_fitting, use_mpi=use_mpi)

    if mydf.rsjk is not None:
        assert use_mpi == False
        assert dm.shape[0] == 1
        dm = dm[0]
        vj_sr, vk_sr = mydf.rsjk.get_jk(
            dm, 
            hermi, 
            kpt, 
            kpts_band, 
            with_j, 
            with_k, 
            omega, 
            exxdiv, **kwargs)
        if with_j:
            vj += vj_sr
        if with_k:
            vk += vk_sr
        dm = dm.reshape(1, dm.shape[0], dm.shape[1])

    ##### the following code is added to deal with _ewald_exxdiv_for_G0 #####
    
    if not use_mpi or (use_mpi and rank==0):
    
        kpts = kpt.reshape(1,3)
        kpts = np.asarray(kpts)
        dm_kpts = dm.reshape(-1, dm.shape[0], dm.shape[1]).copy()
        dm_kpts = lib.asarray(dm_kpts, order='C')
        dms = _format_dms(dm_kpts, kpts)
        nset, nkpts, nao = dms.shape[:3]
        assert nset <= 2
        assert nkpts == 1
    
        kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
        nband = len(kpts_band)
        assert nband == 1

        if is_zero(kpts_band) and is_zero(kpts):
            vk = vk.reshape(nset,nband,nao,nao)
        else:
            raise NotImplementedError("ISDF does not support kpts_band != 0")

        if exxdiv == 'ewald':
            _ewald_exxdiv_for_G0(mydf.cell, kpts, dms, vk, kpts_band=kpts_band)
    
        vk = vk[:,0,:,:]
    
    if use_mpi:
        vk = bcast(vk, root=0)

    ##### end of dealing with _ewald_exxdiv_for_G0 #####

    t1 = log.timer('sr jk', *t1)

    return vj, vk

############# linear scaling implementation ############# 