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
from pyscf.pbc.df.isdf.isdf_tools_local import _pack_aoR_holder, _get_aoR_holders_memory

############ GLOBAL PARAMETER ############

K_DIRECT_NAUX_BUNCHSIZE = 256

def construct_V(aux_basis:np.ndarray, 
                buf, 
                V, 
                ### some helper info ###
                grid_ID, grid_ordering,
                mesh, coulG_real):
    fn = getattr(libpbc, "_construct_V_local_bas", None)
    assert(fn is not None)
        
    nThread = buf.shape[0]
    bufsize_per_thread = buf.shape[1]
    nrow = aux_basis.shape[0]
    ncol = aux_basis.shape[1]
    shift_row = 0
        
    fn(mesh.ctypes.data_as(ctypes.c_void_p),
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
        
def _isdf_get_K_direct_kernel_1(
    mydf, 
    coulG_real,
    ##### input ####
    group_id,  ## the contribution of K from which group 
    dm_RgAO,
    ##### buffer #####, 
    buf_build_V_thread,
    build_VW_buf,
    offset_V_tmp,
    Density_RgR_buf,
    dm_RgAO_buf,
    dm_RgAO_packed_offset,
    ddot_res_RgR_buf,
    K1_tmp1_buf,
    K1_tmp1_ddot_res_buf,
    K1_final_ddot_buf,
    ##### bunchsize #####
    naux_bunchsize = K_DIRECT_NAUX_BUNCHSIZE,
    ##### other info #####
    use_mpi=False,
    ##### out #####
    K1 = None
):
    
    log = logger.Logger(mydf.stdout, mydf.verbose)
    
    ######### info #########
    
    assert K1 is not None
    
    if use_mpi:
        size = comm.Get_size()
        if group_id % comm_size != rank:
            raise ValueError
    
    nao   = mydf.nao
    mesh  = np.array(mydf.cell.mesh, dtype=np.int32)
    ngrid = np.prod(mesh)
    naux  = mydf.naux
    
    aoRg = mydf.aoRg
    assert isinstance(aoRg, list)
    aoR = mydf.aoR
    assert isinstance(aoR, list)
        
    # fetch the atm_ordering #
    
    if hasattr(mydf, "atm_ordering"):
        atm_ordering = mydf.atm_ordering
    else:
        atm_ordering = []
        for group_idx, atm_idx in enumerate(group):
            atm_idx.sort()
            atm_ordering.extend(atm_idx)
        atm_ordering = np.array(atm_ordering, dtype=np.int32)
        mydf.atm_ordering = atm_ordering
        
    group     = mydf.group
    aux_basis = mydf.aux_basis
    assert len(group) == len(aux_basis)
    
    ### the number of aux basis involved ###
    
    naux_tmp = 0
    aoRg_holders = []
    for atm_id in group[group_id]:
        naux_tmp += aoRg[atm_id].aoR.shape[1]
        aoRg_holders.append(aoRg[atm_id])
    assert naux_tmp == aux_basis[group_id].shape[0]
    
    # grid ID involved for the given group
    
    aux_basis_grip_ID  = mydf.partition_group_to_gridID[group_id]
    
    # pack aoRg for loop over Rg #
    
    aoRg_packed = _pack_aoR_holder(aoRg_holders, nao)
    memory = _get_aoR_holders_memory(aoRg_holders)
    
    # log.debug4('In _isdf_get_K_direct_kernel1 group_id = %d, naux = %d' % (group_id, naux_tmp))
    # log.debug4('In _isdf_get_K_direct_kernel1 aoRg_holders Memory = %d Bytes' % (memory))
    # log.debug4('In _isdf_get_K_direct_kernel1 naux_bunchsize      = %d' % (naux_bunchsize))
    
    assert aoRg_packed.ngrid_tot == naux_tmp
    
    ######### get involved C function #########
    
    fn_packcol1 = getattr(libpbc, "_buildK_packcol", None)
    assert fn_packcol1 is not None
    fn_packcol2 = getattr(libpbc, "_buildK_packcol2", None)
    assert fn_packcol2 is not None
    fn_packadd_col = getattr(libpbc, "_buildK_packaddcol", None)
    assert fn_packadd_col is not None
    fn_packadd_row = getattr(libpbc, "_buildK_packaddrow", None)
    assert fn_packadd_row is not None
    
    # determine bunchsize #
    
    bunchsize = min(naux_bunchsize, naux_tmp)
    
    ### allocate buf ###
    
    V_tmp = np.ndarray((bunchsize, ngrid), 
                       buffer=build_VW_buf, 
                       offset=offset_V_tmp, 
                       dtype =np.float64)
    offset_after_V_tmp = offset_V_tmp + V_tmp.size * V_tmp.dtype.itemsize
    
    # buffer for W_tmp # 
    
    W_tmp = np.ndarray((naux_tmp, naux), 
                       buffer=build_VW_buf, 
                       offset=offset_after_V_tmp, 
                       dtype =np.float64)

    #### loop over Rg ####
    
    for p0, p1 in lib.prange(0, naux_tmp, bunchsize):
        
        #### 2. build the V matrix ####
        
        V_tmp = np.ndarray((p1 - p0, ngrid), 
                            buffer=build_VW_buf, 
                            offset=offset_V_tmp, 
                            dtype =np.float64)
        
        construct_V(aux_basis[group_id][p0:p1, :], 
                    buf_build_V_thread,
                    V_tmp,
                    aux_basis_grip_ID,
                    mydf.grid_ID_ordered,
                    mesh,
                    coulG_real)
        
        #### 3. build the K1 matrix ####
        
        ###### 3.1 build density RgR
        
        Density_RgR_tmp = np.ndarray((p1 - p0, ngrid), 
                                     buffer=Density_RgR_buf, 
                                     offset=0, 
                                     dtype =np.float64)
    
        for atm_id in atm_ordering:
            
            aoR_holder = aoR[atm_id]
            
            if aoR_holder is None:
                raise ValueError("aoR_holder is None")

            ngrid_now    = aoR_holder.aoR.shape[1]
            nao_involved = aoR_holder.aoR.shape[0]
            
            ##### packed involved DgAO #####
            
            if nao_involved == nao:
                Density_RgAO_packed = dm_RgAO[p0:p1, :]
            else:
                Density_RgAO_packed = np.ndarray((p1-p0, nao_involved), 
                                                 buffer=dm_RgAO_buf, 
                                                 offset=dm_RgAO_packed_offset, 
                                                 dtype =np.float64)
                
                fn_packcol1(
                    Density_RgAO_packed.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(p1-p0),
                    ctypes.c_int(nao_involved),
                    dm_RgAO[p0:p1, :].ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(p1-p0),
                    ctypes.c_int(nao),
                    aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
                )
            
            grid_begin   = aoR_holder.global_gridID_begin
            ddot_res_RgR = np.ndarray((p1-p0, ngrid_now), buffer=ddot_res_RgR_buf)
            lib.ddot(Density_RgAO_packed, aoR_holder.aoR, c=ddot_res_RgR)
            Density_RgR_tmp[:, grid_begin:grid_begin+ngrid_now] = ddot_res_RgR
        
        Density_RgR = Density_RgR_tmp
        
        #### 3.2 V_tmp = Density_RgR * V
        
        lib.cwise_mul(V_tmp, Density_RgR, out=Density_RgR)
        V2_tmp = Density_RgR
        
        #### 3.3 K1_tmp1 = V2_tmp * aoR.T
        
        K1_tmp1 = np.ndarray((p1-p0, nao), buffer=K1_tmp1_buf)
        K1_tmp1.ravel()[:] = 0.0
        
        for atm_id in atm_ordering:
            aoR_holder = aoR[atm_id]
            
            ngrid_now      = aoR_holder.aoR.shape[1]
            nao_invovled   = aoR_holder.aoR.shape[0]
            ddot_res       = np.ndarray((p1-p0, nao_invovled), buffer=K1_final_ddot_buf)
            grid_loc_begin = aoR_holder.global_gridID_begin
            
            lib.ddot(V2_tmp[:, grid_loc_begin:grid_loc_begin+ngrid_now],
                     aoR_holder.aoR.T, 
                     c=ddot_res)
            
            if nao_involved == nao:
                K1_tmp1 += ddot_res
            else:
                fn_packadd_col(
                    K1_tmp1.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(K1_tmp1.shape[0]),
                    ctypes.c_int(K1_tmp1.shape[1]),
                    ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(ddot_res.shape[0]),
                    ctypes.c_int(ddot_res.shape[1]),
                    aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
                )
        
        #### 3.4 K1 += aoRg * K1_tmp1
        
        nao_invovled = aoRg_packed.nao_invovled
        ddot_res = np.ndarray((nao_invovled, nao), buffer=K1_tmp1_ddot_res_buf)
        lib.ddot(aoRg_packed.aoR[:,p0:p1], K1_tmp1, c=ddot_res)
        fn_packadd_row(
            K1.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(K1.shape[0]),
            ctypes.c_int(K1.shape[1]),
            ddot_res.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(ddot_res.shape[0]),
            ctypes.c_int(ddot_res.shape[1]),
            aoRg_packed.ao_involved.ctypes.data_as(ctypes.c_void_p)
        )
        
        #### 4. build the W matrix ####
        
        grid_shift  = 0
        aux_col_loc = 0
        for j in range(len(group)):
            grid_ID_now = mydf.partition_group_to_gridID[j]
            aux_bas_ket = aux_basis[j]
            naux_ket    = aux_bas_ket.shape[0]
            ngrid_now   = grid_ID_now.size
            W_tmp[p0:p1, aux_col_loc:aux_col_loc+naux_ket] = lib.ddot(V_tmp[:, grid_shift:grid_shift+ngrid_now], aux_bas_ket.T)
            grid_shift += ngrid_now
            aux_col_loc+= naux_ket
        
        assert grid_shift == ngrid
        
    return W_tmp
    