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

import copy
import ctypes
import numpy as np

############ pyscf module ############

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
libpbc = lib.load_library('libpbc')

############ isdf utils ############

from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time
from pyscf.pbc.df.isdf.isdf_linear_scaling_jk import __get_DensityMatrixonRgAO_qradratic
import pyscf.pbc.df.isdf.isdf_tools_local as ISDF_LOCAL_TOOL
from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto

############### sub terms for get K , design for range separation ################

def _contract_k_dm_quadratic_subterm(mydf, dm, 
                                     use_W = True,
                                     dm_bra_type = None, 
                                     dm_ket_type = None,
                                     K_bra_type  = None,
                                     K_ket_type  = None,
                                     use_mpi=False):
    
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        raise NotImplementedError("MPI is not supported yet.")
    
    if dm.ndim == 3:
        assert dm.shape[0] == 1
        dm = dm[0]
    
    ####### preprocess ######## 
    
    assert dm_bra_type in ["compact", "diffuse", "all"]
    assert dm_ket_type in ["compact", "diffuse", "all"]
    assert K_bra_type in  ["compact", "diffuse", "all"]
    assert K_ket_type in  ["compact", "diffuse", "all"]
    
    nCompactAO = len(mydf.CompactAOList)
    nao = dm.shape[0]
    nDiffuseAO = nao - nCompactAO
    
    if K_bra_type == "compact":
        nbraAO = nCompactAO
    elif K_bra_type == "diffuse":
        nbraAO = nDiffuseAO
    else:
        nbraAO = nao
    
    if K_ket_type == "compact":
        nketAO = nCompactAO
    elif K_ket_type == "diffuse":
        nketAO = nDiffuseAO
    else:
        nketAO = nao
    
    ####### start to contract ########
    
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
    
    max_nao_involved   = np.max([aoR_holder.aoR.shape[0] for aoR_holder in aoR  if aoR_holder is not None])
    max_ngrid_involved = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoR  if aoR_holder is not None])
    max_nIP_involved   = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoRg if aoR_holder is not None])
    
    mydf.allocate_k_buffer()    
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
    
    Density_RgAO = __get_DensityMatrixonRgAO_qradratic(mydf, dm, aoRg, dm_bra_type, mydf.Density_RgAO_buf, use_mpi)
        
    #### step 2. get K, those part which W is involved 
    
    # W = mydf.W
    CoulombMat = mydf.W
    if use_W is False:
        CoulombMat = mydf.V_R
    assert CoulombMat is not None
    assert isinstance(CoulombMat, np.ndarray)
        
    K1 = np.zeros((naux, nao), dtype=np.float64)
    
    if use_W is False:
        max_ncol = max_ngrid_involved
    else:
        max_ncol = max_nIP_involved
    
    ####### buf for the first loop #######
    
    offset    = 0
    ddot_buf1 = np.ndarray((naux, max_ncol),         buffer=ddot_res_buf, offset=offset, dtype=np.float64)
    offset    = ddot_buf1.size * ddot_res_buf.dtype.itemsize
    pack_buf  = np.ndarray((naux, max_nao_involved), buffer=ddot_res_buf, offset=offset, dtype=np.float64)
    offset   += pack_buf.size * pack_buf.dtype.itemsize
    ddot_buf2 = np.ndarray((naux, max(max_ncol, max_nao_involved)), buffer=ddot_res_buf, offset=offset, dtype=np.float64)
    
    ### contract ket first ### 

    if use_W is False:
        aoR_Ket = aoR
    else:
        aoR_Ket = aoRg

    ordered_ao_ind = np.arange(nao)

    # nIP_loc = 0 # refers to ngrid_loc if use_W is False
    for aoR_holder in aoR_Ket:
        
        if aoR_holder is None:
            continue
    
        ngrid_now    = aoR_holder.aoR.shape[1]
        grid_loc_now = aoR_holder.global_gridID_begin
        nao_involved = aoR_holder.aoR.shape[0]
        nao_compact  = aoR_holder.nCompact
        
        ao_begin_indx = 0 
        ao_end_indx = nao_involved
        if dm_ket_type == "compact":
            ao_end_indx   = nao_compact
        else:
            ao_begin_indx = nao_compact
        
        nao_involved = ao_end_indx - ao_begin_indx
        
        #### pack the density matrix ####
        
        if nao_involved == nao and np.allclose(ordered_ao_ind, aoR_holder.ao_involved):
            Density_RgAO_packed = Density_RgAO
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
                aoR_holder.ao_involved[ao_begin_indx:ao_end_indx].ctypes.data_as(ctypes.c_void_p)
            )
        
        # W_tmp = Density_RgRg[:, nIP_loc:nIP_loc+ngrid_now] * W[:, nIP_loc:nIP_loc+ngrid_now]
        
        ddot_res1 = np.ndarray((naux, ngrid_now), buffer=ddot_buf1)
        lib.ddot(Density_RgAO_packed, aoR_holder.aoR[ao_begin_indx:ao_end_indx, :], c=ddot_res1)
        Density_RgR = ddot_res1
        CoulombMat_packed = np.ndarray((naux, ngrid_now), buffer=ddot_buf2)
        fn_packcol2(
            CoulombMat_packed.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(naux),
            ctypes.c_int(ngrid_now),
            CoulombMat.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(CoulombMat.shape[0]),
            ctypes.c_int(CoulombMat.shape[1]),
            ctypes.c_int(grid_loc_now),
            ctypes.c_int(grid_loc_now+ngrid_now)
        )
        lib.cwise_mul(CoulombMat_packed, Density_RgR, out=Density_RgR)
        CoulombMat_tmp = Density_RgR

        # ddot
        
        nao_involved  = aoR_holder.aoR.shape[0]
        ao_begin_indx = 0
        ao_end_indx   = nao_involved
        if K_ket_type == "compact":
            ao_end_indx = nao_compact
        else:
            ao_begin_indx = nao_compact
        nao_involved = ao_end_indx - ao_begin_indx 
        
        ddot_res = np.ndarray((naux, nao_involved), buffer=ddot_buf2)
        lib.ddot(CoulombMat_tmp, aoR_holder.aoR[ao_begin_indx:ao_end_indx, :].T, c=ddot_res)
        
        if nao_involved == nao and np.allclose(ordered_ao_ind, aoR_holder.ao_involved):
            K1 += ddot_res
        else:
            # K1[: , aoR_holder.ao_involved] += ddot_res
            fn_packadd_col(
                K1.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(K1.shape[0]),
                ctypes.c_int(K1.shape[1]),
                ddot_res.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(ddot_res.shape[0]),
                ctypes.c_int(ddot_res.shape[1]),
                aoR_holder.ao_involved[ao_begin_indx:ao_end_indx].ctypes.data_as(ctypes.c_void_p)
            )
        
    K = np.zeros((nao, nao), dtype=np.float64) 
    
    # nIP_loc = 0
    for aoR_holder in aoRg:
        
        if aoR_holder is None:
            continue
    
        grid_loc_now = aoR_holder.global_gridID_begin
        ngrid_now = aoR_holder.aoR.shape[1]
        nao_involved = aoR_holder.aoR.shape[0]
        
        ao_begin_indx = 0
        ao_end_indx = nao_involved
        if K_bra_type == "compact":
            ao_end_indx = nao_compact
        else:
            ao_begin_indx = nao_compact
        nao_involved = ao_end_indx - ao_begin_indx
        
        K_tmp = K1[grid_loc_now:grid_loc_now+ngrid_now, :]
        
        ddot_res = np.ndarray((nao_involved, nao), buffer=ddot_res_buf)
        lib.ddot(aoR_holder.aoR[ao_begin_indx:ao_end_indx, :], K_tmp, c=ddot_res)
        
        if nao_involved == nao and np.allclose(ordered_ao_ind, aoR_holder.ao_involved):
            K += ddot_res
        else:
            # K[aoR_holder.ao_involved, :] += ddot_res 
            fn_packadd_row(
                K.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(K.shape[0]),
                ctypes.c_int(K.shape[1]),
                ddot_res.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(ddot_res.shape[0]),
                ctypes.c_int(ddot_res.shape[1]),
                aoR_holder.ao_involved[ao_begin_indx:ao_end_indx].ctypes.data_as(ctypes.c_void_p)
            )
    
    ######### finally delete the buffer #########
    
    del K1
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    if mydf.verbose:
        _benchmark_time(t1, t2, "_contract_k_dm_quadratic", mydf)
    
    return K * ngrid / vol

def _contract_k_dm_quadratic_subterm_merged():
    pass

##### the following func is just a test funcs and should not be called outside, used to test the kernel of get_K #####

def get_aoPairR_bruteforce(cell, coords, compactAO, ao2atmID, cutoff=1e-10):

    compactAO.sort()
    ngrids = coords.shape[0]
    aoR = ISDF_eval_gto(cell, coords=coords)
    weight = np.sqrt(cell.vol / len(coords))
    aoR *= weight 
    aoR_compact = aoR[compactAO,:].copy()
    
    natm = cell.natm
    
    Res = []
    
    for i in range(natm):
        
        AO_i_id = np.where(ao2atmID == i)[0] 
        AO_i_id = [x for x in AO_i_id if x in compactAO]
        if len(AO_i_id) == 0:
            Res.append(None)
            continue
        print("atm %d ao involved " % (i), AO_i_id)
        ao_i_R = aoR[AO_i_id, :] 
        aoPairR = np.einsum("pi, qi -> pqi", ao_i_R, aoR_compact).copy()
        print("atm %d aoPairR shape " % (i), aoPairR.shape)
        ## determine the involved grids 
        aoPairR = aoPairR.reshape(-1, ngrids)
        grid_max = np.max(np.abs(aoPairR), axis=0)
        assert grid_max.size == ngrids 
        grid_involved = np.where(grid_max > cutoff)[0]
        # grid_involved = np.arange(ngrids)
        ## shuffle grid_involved ## 
        grid_involved = np.array(grid_involved, dtype=np.int32)
        grid_involved = np.random.permutation(grid_involved)
        print("atm %d grid involved %d" % (i, grid_involved.size))
        aoPairR = aoPairR[:, grid_involved].copy()
        aoPairR = aoPairR.reshape(ao_i_R.shape[0], aoR_compact.shape[0], grid_involved.size).copy()
        Res.append(ISDF_LOCAL_TOOL.aoPairR_Holder(aoPairR, compactAO.copy(), grid_involved))
        
    return Res

############### sub terms for get K , design for RS, LR, CC ################

def _get_AOMOPairR_holder(mydf, 
                          compactAO: np.ndarray,
                          mo_coeff: np.ndarray,
                          AOPairR_holder: list[ISDF_LOCAL_TOOL.aoPairR_Holder],
                          debug=False
                          ):

    '''
    may not be that useful ! 
    '''

    #### basic info 

    nao  = mo_coeff.shape[0]
    nocc = mo_coeff.shape[1]
    compactAO.sort()
    
    mo_involved = np.arange(nocc)
    
    #### pack buf ####
    
    if hasattr(mydf, "pack_buf"):
        if mydf.pack_buf is None or mydf.pack_buf.size < compactAO.size * mo_coeff.shape[1]:
            mydf.pack_buf = np.zeros(compactAO.size * mo_coeff.shape[0])
    else:
        mydf.pack_buf = np.zeros(compactAO.size * mo_coeff.shape[0])        
    
    #### pack mo_coeff fn ####
        
    fn_packrow = getattr(libpbc, "_buildK_packrow", None)    
    assert fn_packrow is not None

    #### get involved AOMOPairR_holder ####
    
    Res = [] 
    
    ###### get the size of res ###### 
    
    size = 0
    size_max = 0
    for aoPairR_holder in AOPairR_holder:
        if aoPairR_holder is None:
            continue
        nao_atm = aoPairR_holder.aoPairR.shape[0]
        nao_involved = aoPairR_holder.aoPairR.shape[1]
        ngrid   = aoPairR_holder.aoPairR.shape[2]
        size += nao_atm * ngrid * nocc
        size_max = max(size_max, nao_atm * ngrid * nao_involved, nao_atm * ngrid * nocc)
    
    ###### allocate buffer ######
    
    if hasattr(mydf, "aomopairR"):
        if mydf.aomopairR is None or mydf.aomopairR.size < size:
            mydf.aomopairR = np.zeros(size)
    else:
        mydf.aomopairR = np.zeros(size)
        
    if hasattr(mydf, "transpose_buffer"):
        if mydf.transpose_buffer is None or mydf.transpose_buffer.size < size_max:
            mydf.transpose_buffer = np.zeros(size_max)
    else:
        mydf.transpose_buffer = np.zeros(size_max)        
    
    ##### loop over all involved AOMOPairR_holder ##### 
    
    aomopairR_shift = 0
    
    fn_012_to_021 = getattr(libpbc, "transpose_012_to_021", None)
    assert fn_012_to_021 is not None
    
    fn_012_to_021_inplace = getattr(libpbc, "transpose_012_to_021_InPlace", None)
    assert fn_012_to_021_inplace is not None
    
    for aoPairR_holder in AOPairR_holder:
        
        if aoPairR_holder is None:
            Res.append(None)
            continue
        
        nao_atm = aoPairR_holder.aoPairR.shape[0]
        nao_involved = aoPairR_holder.aoPairR.shape[1]
        ngrid   = aoPairR_holder.aoPairR.shape[2]
        
        mo_coeff_packed = np.ndarray((nao_involved, nocc), buffer=mydf.pack_buf)
        
        fn_packrow(
            mo_coeff_packed.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_involved),
            ctypes.c_int(nocc),
            mo_coeff.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(mo_coeff.shape[0]),
            ctypes.c_int(mo_coeff.shape[1]),
            aoPairR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p)
        )
        
        # Res.append(aoPairR_holder)
                
        ##### transpose 012 -> 021 ##### 
        
        transpose_buf = np.ndarray((nao_atm, ngrid, nao_involved), buffer=mydf.transpose_buffer)

        fn_012_to_021(
            transpose_buf.ctypes.data_as(ctypes.c_void_p),
            aoPairR_holder.aoPairR.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_atm),
            ctypes.c_int(nao_involved),
            ctypes.c_int(ngrid)
        )
                        
        transpose_buf = transpose_buf.reshape(nao_atm * ngrid, nao_involved)
        ddot_res      = np.ndarray((nao_atm * ngrid, nocc), buffer=mydf.aomopairR, offset=aomopairR_shift)
        lib.ddot(transpose_buf, mo_coeff_packed, c=ddot_res)
        aomoPairR     = np.ndarray((nao_atm, nocc, ngrid), buffer=mydf.aomopairR, offset=aomopairR_shift) 
        
        aomopairR_shift += nao_atm * nocc * ngrid * ddot_res.dtype.itemsize
        
        fn_012_to_021_inplace(
            aomoPairR.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_atm),
            ctypes.c_int(ngrid),
            ctypes.c_int(nocc),
            transpose_buf.ctypes.data_as(ctypes.c_void_p)
        )
        
        Res.append(ISDF_LOCAL_TOOL.aoPairR_Holder(aomoPairR, mo_involved, aoPairR_holder.grid_involved))
    
    if debug:
        
        for atm_id, aoPairR_holder in enumerate(AOPairR_holder):
            
            if aoPairR_holder is None:
                continue
            
            nao_atm = aoPairR_holder.aoPairR.shape[0]
            nao_involved = aoPairR_holder.aoPairR.shape[1]
            ngrid   = aoPairR_holder.aoPairR.shape[2]
            mo_coeff_packed = mo_coeff[aoPairR_holder.ao_involved, :]
            aomoPairR = np.einsum("ijR, jk -> ikR", aoPairR_holder.aoPairR, mo_coeff_packed)
            print("aomoPairR              shape = ", aomoPairR.shape)
            print("Res[atm_id].aoPair     shape = ", Res[atm_id].aoPairR.shape)
            # print("Res[atm_id] shape = ", Res[atm_id].aoPairR)
            # diff = np.linalg.norm(aomoPairR - Res[atm_id].aoPairR)
            # print("diff = ", diff/np.sqrt(np.prod(aomoPairR.shape)))
            assert np.allclose(aomoPairR, Res[atm_id].aoPairR)
    
    return Res

def _contract_k_dm_quadratic_subterm_CC( 
    # evaluate exactly ! exploring the locality ! #
    mydf, 
    dm,
    atm_2_compactAO: np.ndarray,
    aoPairR_holders: list[ISDF_LOCAL_TOOL.aoPairR_Holder],
    dm_ket_type = None,
    K_ket_type = None,
    ### calculate interaction between different mesh ###
    mesh_bra = None,
    mesh_ket = None,
    map_bra_2_ket = None,
    coulG = None
):
    ######## preprocess ########
    
    assert mesh_bra is not None
    assert mesh_ket is not None
    
    if dm.ndim == 3:
        assert dm.shape[0] == 1
        dm = dm[0]
    
    if np.allclose(mesh_bra, mesh_ket):
        ngrid_bra = np.prod(mesh_bra)
        map_bra_2_ket = np.arange(ngrid_bra, dtype=np.int32)
    else:
        assert map_bra_2_ket is not None
        assert map_bra_2_ket.size == np.prod(mesh_bra)
    
    assert dm_ket_type in ["compact", "diffuse", "all"]
    assert K_ket_type in ["compact", "diffuse", "all"]
    if dm_ket_type is None:
        dm_ket_type = "all"
    if K_ket_type is None:
        K_ket_type = "all"
        
    nCompactAO = len(mydf.CompactAOList)
    nao = mydf.cell.nao 
    nDiffuseAO = nao - nCompactAO
    
    nao_atm_max = 0
    nao_involved_max = 0
    ngrid_max = 0
    for aoPairR_holder in aoPairR_holders:
        if aoPairR_holder is None:
            continue
        nao_atm_max = max(nao_atm_max, aoPairR_holder.aoPairR.shape[0])
        ngrid_max = max(ngrid_max, aoPairR_holder.aoPairR.shape[2])
        nao_involved_max = max(nao_involved_max, aoPairR_holder.aoPairR.shape[1])
    
    nthread = lib.num_threads() 
    natm = len(aoPairR_holders)
    K = np.zeros((nao, nao), dtype=np.float64)
    ngrid_bra = np.prod(mesh_bra)
    ngrid_ket = np.prod(mesh_ket)
    ngrid_bra_complex = mesh_bra[0] * mesh_bra[1] * (mesh_bra[2]//2+1)*2
    ngrid_ket_complex = mesh_ket[0] * mesh_ket[1] * (mesh_ket[2]//2+1)*2
    vol = mydf.cell.vol
    
    coulG_real     = coulG.reshape(*mesh)[:, :, :mesh[2]//2+1].reshape(-1).copy()
    
    aoR = mydf.aoR
    partition = mydf.partition
    
    nao_involved_aoR_max = 0
    ngrid_involved_aoR_max = np.max([len(x) for x in partition])
    for aoR_holder in aoR:
        if aoR_holder is None:
            continue
        nao_involved_aoR_max = max(nao_involved_aoR_max, aoR_holder.ao_involved.size)
    
    ### allocate buffer ### 
    
    # NOTE: we do not optimize the storage need for the buffer, just allocate the maximum space needed
    
    size_pack1 = nao_atm_max * nao_involved_max * max(ngrid_bra,ngrid_ket)
    size_pack2 = nao_atm_max * nao_involved_aoR_max * ngrid_involved_aoR_max
    size_pack1 = max(size_pack1, size_pack2)
    if hasattr(mydf, "CC_pack_buf1"):
        if mydf.CC_pack_buf1 is None or mydf.CC_pack_buf1.size < size_pack1:
            mydf.CC_pack_buf1 = np.zeros(size_pack1)
    else:
        mydf.CC_pack_buf1 = np.zeros(size_pack1)
    
    if hasattr(mydf, "CC_pack_buf2"):
        if mydf.CC_pack_buf2 is None or mydf.CC_pack_buf2.size < size_pack1:
            mydf.CC_pack_buf2 = np.zeros(size_pack1)
    else:
        mydf.CC_pack_buf2 = np.zeros(size_pack1)
    
    size_fft_buf = nao_atm_max * nao_involved_max * ngrid_ket
    if hasattr(mydf, "CC_fft_buf"):
        if mydf.CC_fft_buf is None or mydf.CC_fft_buf.size < size_fft_buf:
            mydf.CC_fft_buf = np.zeros(size_fft_buf)
    else:
        mydf.CC_fft_buf = np.zeros(size_fft_buf)
    
    size_thread_buf = nao_atm_max * nthread * (ngrid_bra_complex+ngrid_ket_complex+32)
    if hasattr(mydf, "thread_buf"):
        if mydf.thread_buf is None or mydf.thread_buf.size < size_thread_buf:
            mydf.thread_buf = np.zeros(size_thread_buf)
    else:
        mydf.thread_buf = np.zeros(size_thread_buf)

    #### involved C function ####

    fn_fft = getattr(libpbc, "_construct_V_kernel", None) 
    assert fn_fft is not None
    
    fn_packcol1 = getattr(libpbc, "_buildK_packcol", None)
    assert fn_packcol1 is not None
    
    fn_unpackcol1 = getattr(libpbc, "_buildK_unpackcol", None)
    assert fn_unpackcol1 is not None
    
    fn_012_to_021 = getattr(libpbc, "transpose_012_to_021", None)
    assert fn_012_to_021 is not None
    
    fn_012_to_021_inplace = getattr(libpbc, "transpose_012_to_021_InPlace", None)
    assert fn_012_to_021_inplace is not None
    
    fn_contract_ipk_pk_to_ik = getattr(libpbc, "contract_ipk_pk_to_ik", None)
    assert fn_contract_ipk_pk_to_ik is not None
    
    fn_pack_add_K2 = getattr(libpbc, "_packadd_local_dm2", None)
    assert fn_pack_add_K2 is not None
    
    #### loop over all involved atoms ####
    
    intermediate1_buf  = np.zeros((nao_atm_max, ngrid_ket), dtype=np.float64)
    final_ddot_res_buf = np.zeros((nao_atm_max, nao), dtype=np.float64)
    
    for i in range(natm):
        
        aoPairR_holder = aoPairR_holders[i]
        
        if aoPairR_holder is None:
            continue    
    
        nao_atm = aoPairR_holder.aoPairR.shape[0]
        nao_involved = aoPairR_holder.aoPairR.shape[1]
        ngrid_invovled = aoPairR_holder.aoPairR.shape[2]
        
        # print("nao_atm %d" % nao_atm)
        # print("nao_involved %d" % nao_involved)
        # print("ngrid_invovled %d %d " % (ngrid_invovled, aoPairR_holder.grid_involved.size))
        # print("ngrid_bra %d" % ngrid_bra)
        
        #### perform fft ####
        
        V_tmp = np.ndarray((nao_atm * nao_involved, ngrid_ket), buffer=mydf.CC_fft_buf)

        bunchsize = min(nao_atm, nao_atm * nao_involved // nthread)
        bunchsize = max(bunchsize, 1)
        # bunchsize = 1
        thread_buf = np.ndarray((bunchsize, nthread, ngrid_bra_complex+ngrid_ket_complex+32), buffer=mydf.thread_buf)
        buffersize = bunchsize * (ngrid_bra_complex+ngrid_ket_complex+32)
        pack_buf1 = np.ndarray((nao_atm * nao_involved, ngrid_bra), buffer=mydf.CC_pack_buf1)
        
        # print("pack_buf1 = ", pack_buf1.shape)
        # print("aoPairR_holder.aoPairR = ", aoPairR_holder.aoPairR.shape)
        # print("atm %d" % i)
        
        # continue
        
        fn_unpackcol1(
            pack_buf1.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_atm * nao_involved),
            ctypes.c_int(ngrid_bra),
            aoPairR_holder.aoPairR.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_atm * nao_involved),
            ctypes.c_int(ngrid_invovled),
            aoPairR_holder.grid_involved.ctypes.data_as(ctypes.c_void_p)
        )
        
        # pack_buf1_benchmark = np.zeros_like(pack_buf1).reshape(nao_atm, nao_involved, ngrid_bra)
        # pack_buf1_benchmark[:, :, aoPairR_holder.grid_involved] = aoPairR_holder.aoPairR[:, :, :]
        # diff = np.linalg.norm(pack_buf1.ravel() - pack_buf1_benchmark.ravel())
        # print("diff = ", diff/np.sqrt(np.prod(pack_buf1.shape)))
        # assert np.allclose(pack_buf1.ravel(), pack_buf1_benchmark.ravel())
        
        # print("atm %d" % i)
        # continue
        
        fn_fft(
            mesh_bra.ctypes.data_as(ctypes.c_void_p),
            mesh_ket.ctypes.data_as(ctypes.c_void_p),
            map_bra_2_ket.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_atm*nao_involved),
            pack_buf1.ctypes.data_as(ctypes.c_void_p),
            coulG_real.ctypes.data_as(ctypes.c_void_p),
            V_tmp.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(bunchsize),
            thread_buf.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(buffersize)
        )
        
        # print("map_bra_2_ket = ", map_bra_2_ket)
        
        # V_benchmark = pack_buf1.reshape(nao_atm, nao_involved, *mesh_bra).copy()
        # V_benchmark = np.fft.fftn(V_benchmark, axes=(2, 3, 4)).reshape(nao_atm * nao_involved, ngrid_bra)
        # V_benchmark = V_benchmark * coulG
        # V_benchmark = np.fft.ifftn(V_benchmark.reshape(nao_atm, nao_involved, *mesh_bra), axes=(2, 3, 4)).reshape(nao_atm * nao_involved, ngrid_bra)
        # V_real = V_benchmark.real.copy()
        # V_imag = V_benchmark.imag.copy()
        # diff = np.linalg.norm(V_real - V_tmp)
        # print("diff = ", diff/np.sqrt(np.prod(V_real.shape)))
        # assert np.allclose(V_imag, 0)
        # assert np.allclose(V_real, V_tmp)

        # continue
                
        #### pack dm #### 
        
        dm_packed = dm[aoPairR_holder.ao_involved, :].copy()  
        
        # print("dm_packed = ", dm_packed)
        
        for j in range(natm):
            
            # continue
            # print("atm %d %d" % (i, j)) 
            # continue
            
            aoR_holder = aoR[j] 
            ket_grid_involved = partition[j]
            nket_grid_involved = ket_grid_involved.size
            
            #### deal with type #### 
            
            ket_ao_involved = aoR_holder.ao_involved
            ket_ao_begin = 0
            ket_ao_end = ket_ao_involved.size
            if dm_ket_type == "compact":
                ket_ao_end = aoR_holder.nCompact
            else:
                if dm_ket_type == "diffuse":
                    ket_ao_begin = aoR_holder.nCompact
            ket_nao_involved = ket_ao_end - ket_ao_begin
            
            # print("ket_ao_involved = ", ket_ao_involved)
            # print("ket_ao_begin    = ", ket_ao_begin)
            # print("ket_ao_end      = ", ket_ao_end)
            
            #### pack involved aoPairR and dm ####
            
            dm_involved = dm_packed[:, ket_ao_involved[ket_ao_begin:ket_ao_end]].copy() 
            
            # print("dm_involved = ", dm_involved)
            
            pack_buf1 = np.ndarray((nao_atm * nao_involved, nket_grid_involved), buffer=mydf.CC_pack_buf1)
            
            pack_buf2 = np.ndarray((nao_atm, nket_grid_involved, nao_involved), buffer=mydf.CC_pack_buf2)
            
            # pack_buf1[:, :, :] = V_tmp[:, :, ket_grid_involved]
            fn_packcol1(
                pack_buf1.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_atm * nao_involved),
                ctypes.c_int(nket_grid_involved),
                V_tmp.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_atm * nao_involved),
                ctypes.c_int(ngrid_ket), 
                ket_grid_involved.ctypes.data_as(ctypes.c_void_p)
            )
            # pack_buf1_bench = np.zeros_like(pack_buf1).reshape(nao_atm, nao_involved, nket_grid_involved)
            # pack_buf1_bench[:, :, :] = V_tmp.reshape(nao_atm,nao_involved,-1)[:, :, ket_grid_involved]
            # assert np.allclose(pack_buf1.ravel(), pack_buf1_bench.ravel())
            #### pack_buf2 = np.transpose(pack_buf1, (0, 2, 1))
            fn_012_to_021(
                pack_buf2.ctypes.data_as(ctypes.c_void_p),
                pack_buf1.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_atm),
                ctypes.c_int(nao_involved),
                ctypes.c_int(nket_grid_involved)
            )
            pack_buf2 = pack_buf2.reshape(nao_atm * nket_grid_involved, nao_involved)
            # pack_buf2_bench = np.transpose(pack_buf1_bench, (0, 2, 1)).copy()
            # assert np.allclose(pack_buf2.ravel(), pack_buf2_bench.ravel())
            #### pack_buf1 = pack_buf2 * dm 
            pack_buf1 = np.ndarray((nao_atm * nket_grid_involved,   ket_nao_involved), buffer=mydf.CC_pack_buf1)
            lib.ddot(pack_buf2, dm_involved, c=pack_buf1)
            pack_buf1 = pack_buf1.reshape(nao_atm, nket_grid_involved, ket_nao_involved)
            #### pack_buf2 = np.transpose(pack_buf1, (0, 2, 1))
            pack_buf2 = np.ndarray((nao_atm, ket_nao_involved, nket_grid_involved), buffer=mydf.CC_pack_buf2)
            fn_012_to_021(
                pack_buf2.ctypes.data_as(ctypes.c_void_p),
                pack_buf1.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_atm),
                ctypes.c_int(nket_grid_involved),
                ctypes.c_int(ket_nao_involved)
            )
            # pack_buf2_bench = np.transpose(pack_buf1.reshape(nao_atm, nket_grid_involved, -1), (0, 2, 1)).copy()
            # assert np.allclose(pack_buf2.ravel(), pack_buf2_bench.ravel())
            #### ipk,pk->ik
            intermediate1 = np.ndarray((nao_atm, nket_grid_involved), buffer=intermediate1_buf)
            fn_contract_ipk_pk_to_ik(
                pack_buf2.ctypes.data_as(ctypes.c_void_p),
                aoR_holder.aoR[ket_ao_begin:ket_ao_end,:].ctypes.data_as(ctypes.c_void_p),
                intermediate1.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_atm),
                ctypes.c_int(ket_nao_involved),
                ctypes.c_int(nket_grid_involved)
            )
            # print("pack_buf2.shape = ", pack_buf2.shape)
            # print("aoR_holder.aoR.shape = ", aoR_holder.aoR.shape)
            # benchmark = np.einsum("ipk,pk->ik", pack_buf2, aoR_holder.aoR[ket_ao_begin:ket_ao_end,:])
            # assert np.allclose(intermediate1, benchmark)
            # finall ddot
            #### deal with type #### 
            ket_ao_begin = 0
            ket_ao_end = ket_ao_involved.size
            if K_ket_type == "compact":
                ket_ao_end = aoR_holder.nCompact
            else:
                if K_ket_type == "diffuse":
                    ket_ao_begin = aoR_holder.nCompact
            ket_nao_involved = ket_ao_end - ket_ao_begin
            ddot_res = np.ndarray((nao_atm, ket_nao_involved), buffer=final_ddot_res_buf)
            lib.ddot(intermediate1, aoR_holder.aoR[ket_ao_begin:ket_ao_end, :].T, c=ddot_res)
            #### pack the result #### 
            # print("ddot_res = ", ddot_res)
            # print(atm_2_compactAO[i])
            # print(ket_ao_involved[ket_ao_begin:ket_ao_end])
            # print("nao_atm          = ", nao_atm)
            # print("ket_nao_involved = ", ket_nao_involved)
            # print("nao              = ", nao)
            fn_pack_add_K2(
                ddot_res.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_atm),
                atm_2_compactAO[i].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(ket_nao_involved),
                ket_ao_involved[ket_ao_begin:ket_ao_end].ctypes.data_as(ctypes.c_void_p),
                K.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao)
            )
    # print("K = ", K, np.allclose(K, 0.0))
    return K * np.sqrt(ngrid_bra*ngrid_ket / vol**2)

# def _contract_k_mo_quadratic_subterm_CCCC_bruteforce(
#     mydf, 
#     atm_2_compactAO: np.ndarray,
#     AOMOPairR_holder: list[ISDF_LOCAL_TOOL.aoPairR_Holder],
#     mesh: np.ndarray,
#     coulG: np.ndarray
# ):
#     aomoPairR = []
#     for aoPairR_holder in AOMOPairR_holder:
#         if aoPairR_holder is None:
#             continue
#         aomoPairR_ordered = np.zeros_like(aoPairR_holder.aoPairR)
#         aomoPairR_ordered[:, :, aoPairR_holder.grid_involved] = aoPairR_holder.aoPairR
#         aomoPairR.append(aoPairR_holder.aoPairR)
#     aomoPairR = np.concatenate(aomoPairR, axis=0).copy()
#     print("aomoPairR.shape = ", aomoPairR.shape)
#     naocompact = aomoPairR.shape[0]
#     nocc = aomoPairR.shape[1]
#     ngrid = aomoPairR.shape[2]
#     assert ngrid == np.prod(mesh)
#     V_benchmark = aomoPairR.reshape(naocompact, nocc, *mesh).copy()
#     V_benchmark = np.fft.ifftn(V_benchmark, axes=(2, 3, 4)).reshape(naocompact, nocc, ngrid)
#     V_benchmark = V_benchmark * coulG
#     V_benchmark = V_benchmark.reshape(naocompact, nocc, *mesh)
#     V_benchmark = np.fft.fftn(V_benchmark, axes=(2, 3, 4))
#     V_benchmark = V_benchmark.reshape(naocompact, nocc, ngrid)
#     V_real = V_benchmark.real.copy()
#     K = np.einsum("pij,qij->pq", V_real, aomoPairR)
#     assert np.allclose(K, K.T)
#     vol = mydf.cell.vol
#     return K * ngrid / vol

def _contract_k_mo_quadratic_subterm_CCCC(
    mydf, 
    atm_2_compactAO: np.ndarray,
    AOMOPairR_holder: list[ISDF_LOCAL_TOOL.aoPairR_Holder],
    mesh: np.ndarray,
    coulG: np.ndarray,
):
    
    ### first loop get the basic info ### 
    
    nao_atm_max = 0
    ngrid_max = 0
    nocc = 0
    for aoPairR_holder in AOMOPairR_holder:
        if aoPairR_holder is None:
            continue
        nao_atm_max = max(nao_atm_max, aoPairR_holder.aoPairR.shape[0])
        if nocc == 0:
            nocc = aoPairR_holder.aoPairR.shape[1]
        else:
            assert nocc == aoPairR_holder.aoPairR.shape[1]
        ngrid_max = max(ngrid_max, aoPairR_holder.aoPairR.shape[2])

    print("nao_atm_max = ", nao_atm_max)
    print("nocc        = ", nocc)
    print("ngrid_max   = ", ngrid_max)

    nthread = lib.num_threads()
    natm    = len(AOMOPairR_holder)
    nao     = mydf.cell.nao
    K       = np.zeros((nao, nao), dtype=np.float64)
    ngrid   = np.prod(mesh)
    ngrid_complex = mesh[0] * mesh[1] * (mesh[2]//2+1)*2
    vol     = mydf.cell.vol
    
    final_ddot_res = np.zeros((nao_atm_max, nao_atm_max), dtype=np.float64)
    coulG_real     = coulG.reshape(*mesh)[:, :, :mesh[2]//2+1].reshape(-1).copy()
    
    ### allocate buffer ### 
    
    if hasattr(mydf, "CC_pack_buf1"):
        if mydf.CC_pack_buf1 is None or mydf.CC_pack_buf1.size < nao_atm_max * nthread * ngrid:
            mydf.CC_pack_buf1 = np.zeros(nao_atm_max * nthread * ngrid)
    else:
        mydf.CC_pack_buf1 = np.zeros(nao_atm_max * nthread * ngrid)
    
    if hasattr(mydf, "CC_pack_buf2"):
        if mydf.CC_pack_buf2 is None or mydf.CC_pack_buf2.size < nao_atm_max * nthread * ngrid_max:
            mydf.CC_pack_buf2 = np.zeros(nao_atm_max * nthread * ngrid_max)
    else:
        mydf.CC_pack_buf2 = np.zeros(nao_atm_max * nthread * ngrid_max)
    
    if hasattr(mydf, "CC_fft_buf"):
        if mydf.CC_fft_buf is None or mydf.CC_fft_buf.size < nao_atm_max * nthread * ngrid:
            mydf.CC_fft_buf = np.zeros(nao_atm_max * nthread * ngrid)
    else:
        mydf.CC_fft_buf = np.zeros(nao_atm_max * nthread * ngrid)
    
    if hasattr(mydf, "thread_buf"):
        if mydf.thread_buf is None or mydf.thread_buf.size < nao_atm_max * nthread * ngrid_complex:
            mydf.thread_buf = np.zeros(nao_atm_max * nthread * ngrid_complex)
    else:
        mydf.thread_buf = np.zeros(nao_atm_max * nthread * ngrid_complex)
    
    fn_fft = getattr(libpbc, "_construct_V", None)
    assert fn_fft is not None
    
    fn_unpack_aoPair = getattr(libpbc, "_unpack_aoPairR", None)
    assert fn_unpack_aoPair is not None
    
    fn_pack_aoPair_ind1 = getattr(libpbc, "_pack_aoPairR_index1", None)
    assert fn_pack_aoPair_ind1 is not None
    
    fn_pack_add_K = getattr(libpbc, "_packadd_local_dm", None)
    assert fn_pack_add_K is not None
    
    fn_pack_add_K2 = getattr(libpbc, "_packadd_local_dm2_add_transpose", None)
    assert fn_pack_add_K2 is not None
    
    fn_packcol1 = getattr(libpbc, "_buildK_packcol", None)
    assert fn_packcol1 is not None
    
    for i in range(natm):
        
        aomoPairR_holder = AOMOPairR_holder[i]
        
        if aomoPairR_holder is None:
            continue
        
        nao_atm          = aomoPairR_holder.aoPairR.shape[0]
        ngrid_invovled   = aomoPairR_holder.aoPairR.shape[2]
        
        for p0, p1 in lib.prange(0, nocc, nthread):
            
            pack_buf1  = np.ndarray((nao_atm, p1-p0, ngrid), buffer=mydf.CC_pack_buf1)
            V_tmp      = np.ndarray((nao_atm, p1-p0, ngrid), buffer=mydf.CC_fft_buf)
            thread_buf = np.ndarray((nao_atm, p1-p0, ngrid_complex), buffer=mydf.thread_buf)
            
            fn_unpack_aoPair(
                pack_buf1.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_atm),
                ctypes.c_int(p1-p0),
                ctypes.c_int(ngrid),
                aomoPairR_holder.aoPairR.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_atm),
                ctypes.c_int(nocc),
                ctypes.c_int(ngrid_invovled),
                ctypes.c_int(p0),
                ctypes.c_int(p1),
                aomoPairR_holder.grid_involved.ctypes.data_as(ctypes.c_void_p)
            )
            
            fn_fft(
                mesh.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_atm*(p1-p0)),
                pack_buf1.ctypes.data_as(ctypes.c_void_p),
                coulG_real.ctypes.data_as(ctypes.c_void_p),
                V_tmp.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_atm), 
                thread_buf.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_atm*ngrid_complex)
            ) ### cubic scaling !!!, in the final version, this function should be called only once !!

            ### loop over all the ket ### 
            
            for j in range(i, natm): 
            # for j in range(natm):
                
                aomoPairR_holder_ket = AOMOPairR_holder[j] 
                
                if aomoPairR_holder_ket is None:
                    continue
                
                nao_atm_ket = aomoPairR_holder_ket.aoPairR.shape[0]
                ngrid_involved_ket = aomoPairR_holder_ket.aoPairR.shape[2] 
                
                pack_buf1 = np.ndarray((nao_atm, p1-p0, ngrid_involved_ket), buffer=mydf.CC_pack_buf1)
                
                fn_packcol1(
                    pack_buf1.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao_atm*(p1-p0)),
                    ctypes.c_int(ngrid_involved_ket),
                    V_tmp.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao_atm*(p1-p0)),
                    ctypes.c_int(ngrid),
                    aomoPairR_holder_ket.grid_involved.ctypes.data_as(ctypes.c_void_p),
                )
                
                pack_buf1 = pack_buf1.reshape(nao_atm, (p1-p0)*ngrid_involved_ket)
    
                pack_buf2 = np.ndarray((nao_atm_ket, p1-p0, ngrid_involved_ket), buffer=mydf.CC_pack_buf2)
                
                fn_pack_aoPair_ind1(
                    pack_buf2.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao_atm_ket),
                    ctypes.c_int(p1-p0),
                    ctypes.c_int(ngrid_involved_ket),
                    aomoPairR_holder_ket.aoPairR.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao_atm_ket),
                    ctypes.c_int(nocc),
                    ctypes.c_int(ngrid_involved_ket),
                    ctypes.c_int(p0),
                    ctypes.c_int(p1)
                )
    
                pack_buf2 = pack_buf2.reshape(nao_atm_ket, (p1-p0)*ngrid_involved_ket) 
                
                ddot_res = np.ndarray((nao_atm, nao_atm_ket), buffer=final_ddot_res)
                lib.ddot(pack_buf1, pack_buf2.T, c=ddot_res) 
                                
                if i == j:
                    # assert atm_2_compactAO[i].size == nao_atm
                    fn_pack_add_K(
                        ddot_res.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(nao_atm_ket),
                        atm_2_compactAO[i].ctypes.data_as(ctypes.c_void_p),
                        K.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(nao)
                    )
                else:
                    # assert atm_2_compactAO[i].size == nao_atm
                    # assert atm_2_compactAO[j].size == nao_atm_ket
                    fn_pack_add_K2(
                        ddot_res.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(nao_atm),
                        atm_2_compactAO[i].ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(nao_atm_ket),
                        atm_2_compactAO[j].ctypes.data_as(ctypes.c_void_p),
                        K.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(nao)
                    )
    
    ### clean buffer ###
    
    del final_ddot_res

    assert np.allclose(K, K.T)

    return K * ngrid / vol

def _contract_k_dm_quadratic_subterm_CCCC(
    mydf, 
    dm,
    atm_2_compactAO: np.ndarray,
    aoPairR_holders: list[ISDF_LOCAL_TOOL.aoPairR_Holder],
    mesh: np.ndarray,
    coulG: np.ndarray,
):
    
    if dm.ndim == 3:
        assert dm.shape[0] == 1
        dm = dm[0]
    
    nCompactAO = len(mydf.CompactAOList)
    nao        = mydf.cell.nao
    nDiffuseAO = nao - nCompactAO
    
    ### first loop get the basic info ### 
    
    nao_atm_max = 0
    ngrid_max = 0
    nao_involved_max = 0
    for aoPairR_holder in aoPairR_holders:
        if aoPairR_holder is None:
            continue
        nao_atm_max = max(nao_atm_max, aoPairR_holder.aoPairR.shape[0])
        ngrid_max = max(ngrid_max, aoPairR_holder.aoPairR.shape[2])
        nao_involved_max = max(nao_involved_max, aoPairR_holder.aoPairR.shape[1])

    nthread = lib.num_threads()
    natm    = len(aoPairR_holders)
    nao     = mydf.cell.nao
    K       = np.zeros((nao, nao), dtype=np.float64)
    ngrid   = np.prod(mesh)
    ngrid_complex = mesh[0] * mesh[1] * (mesh[2]//2+1)*2
    vol     = mydf.cell.vol
    
    final_ddot_res_buf = np.zeros((nao_atm_max, nao_atm_max), dtype=np.float64)
    coulG_real     = coulG.reshape(*mesh)[:, :, :mesh[2]//2+1].reshape(-1).copy()
    
    ### allocate buffer ### 
    
    size1 = nao_atm_max * nao_involved_max * ngrid
    if hasattr(mydf, "CC_pack_buf1"):
        if mydf.CC_pack_buf1 is None or mydf.CC_pack_buf1.size < size1:
            mydf.CC_pack_buf1 = np.zeros(size1)
    else:
        mydf.CC_pack_buf1 = np.zeros(size1)
    
    size2 = nao_atm_max * nao_involved_max * ngrid_max
    if hasattr(mydf, "CC_pack_buf2"):
        if mydf.CC_pack_buf2 is None or mydf.CC_pack_buf2.size < size2:
            mydf.CC_pack_buf2 = np.zeros(size2)
    else:
        mydf.CC_pack_buf2 = np.zeros(size2)
    
    size_fft = nao_atm_max * nao_involved_max * ngrid
    if hasattr(mydf, "CC_fft_buf"):
        if mydf.CC_fft_buf is None or mydf.CC_fft_buf.size < size_fft:
            mydf.CC_fft_buf = np.zeros(size_fft)
    else:
        mydf.CC_fft_buf = np.zeros(size_fft)
    
    if hasattr(mydf, "thread_buf"):
        if mydf.thread_buf is None or mydf.thread_buf.size < nao_atm_max * nthread * (ngrid_complex+32):
            mydf.thread_buf = np.zeros(nao_atm_max * nthread * (ngrid_complex+32))
    else:
        mydf.thread_buf = np.zeros(nao_atm_max * nthread * (ngrid_complex+32))
    
    fn_fft = getattr(libpbc, "_construct_V", None)
    assert fn_fft is not None
    
    fn_012_to_021 = getattr(libpbc, "transpose_012_to_021", None)
    assert fn_012_to_021 is not None
    
    # fn_unpack_aoPair = getattr(libpbc, "_unpack_aoPairR", None)
    # assert fn_unpack_aoPair is not None
    
    # fn_pack_aoPair_ind1 = getattr(libpbc, "_pack_aoPairR_index1", None)
    # assert fn_pack_aoPair_ind1 is not None
    
    fn_unpackcol1 = getattr(libpbc, "_buildK_unpackcol", None)
    assert fn_unpackcol1 is not None
    
    fn_pack_add_K = getattr(libpbc, "_packadd_local_dm", None)
    assert fn_pack_add_K is not None
    
    fn_pack_add_K2 = getattr(libpbc, "_packadd_local_dm2_add_transpose", None)
    assert fn_pack_add_K2 is not None
    
    fn_packcol1 = getattr(libpbc, "_buildK_packcol", None)
    assert fn_packcol1 is not None
    
    #### loop over all involved atoms ####
    
    for i in range(natm):
        
        aoPairR_holder = aoPairR_holders[i]
        
        if aoPairR_holder is None:
            continue    
    
        nao_atm        = aoPairR_holder.aoPairR.shape[0]
        nao_involved   = aoPairR_holder.aoPairR.shape[1]
        ngrid_invovled = aoPairR_holder.aoPairR.shape[2]
    
        #### perform fft #### 
        
        V_tmp = np.ndarray((nao_atm * nao_involved, ngrid), buffer=mydf.CC_fft_buf)
    
        bunchsize = min(nao_atm, nao_atm * nao_involved // nthread)
        bunchsize = max(bunchsize, 1)
        thread_buf = np.ndarray((bunchsize, nthread, ngrid_complex+32), buffer=mydf.thread_buf)
        buffersize = bunchsize * (ngrid_complex+32)
        pack_buf1 = np.ndarray((nao_atm * nao_involved, ngrid), buffer=mydf.CC_pack_buf1)

        # pack 
        
        fn_unpackcol1(
            pack_buf1.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_atm * nao_involved),
            ctypes.c_int(ngrid),
            aoPairR_holder.aoPairR.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_atm * nao_involved),
            ctypes.c_int(ngrid_invovled),
            aoPairR_holder.grid_involved.ctypes.data_as(ctypes.c_void_p)
        )
        
        # fft 
        
        fn_fft(
            mesh.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_atm * nao_involved),
            pack_buf1.ctypes.data_as(ctypes.c_void_p),
            coulG_real.ctypes.data_as(ctypes.c_void_p),
            V_tmp.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(bunchsize), 
            thread_buf.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(buffersize)
        )
        
        #### pack dm #### 
        
        dm_packed = dm[aoPairR_holder.ao_involved, :].copy()
        
        for j in range(i, natm):
            
            aoPairR_holder_ket = aoPairR_holders[j]
            ket_ao_involved = aoPairR_holder_ket.ao_involved
            ket_nao_involved = ket_ao_involved.size
            ket_grid_involved = aoPairR_holder_ket.grid_involved
            nket_grid_involved = ket_grid_involved.size
            nao_atm_ket = aoPairR_holder_ket.aoPairR.shape[0]

            #### pack involved aoPairR and dm ####
            
            dm_involved = dm_packed[:, ket_ao_involved].copy()
            
            pack_buf1 = np.ndarray((nao_atm * nao_involved, nket_grid_involved), buffer=mydf.CC_pack_buf1) 
            pack_buf2 = np.ndarray((nao_atm, nket_grid_involved, nao_involved), buffer=mydf.CC_pack_buf2)
            # pack_buf1[:, :, :] = V_tmp[:, :, ket_grid_involved]
            fn_packcol1(
                pack_buf1.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_atm * nao_involved),
                ctypes.c_int(nket_grid_involved),
                V_tmp.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_atm * nao_involved),
                ctypes.c_int(ngrid), 
                ket_grid_involved.ctypes.data_as(ctypes.c_void_p)
            )
            #### pack_buf2 = np.transpose(pack_buf1, (0, 2, 1))
            fn_012_to_021(
                pack_buf2.ctypes.data_as(ctypes.c_void_p),
                pack_buf1.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_atm),
                ctypes.c_int(nao_involved),
                ctypes.c_int(nket_grid_involved)
            )
            pack_buf2 = pack_buf2.reshape(nao_atm * nket_grid_involved, nao_involved)
            #### pack_buf1 = pack_buf2 * dm
            pack_buf1 = np.ndarray((nao_atm * nket_grid_involved, ket_nao_involved), buffer=mydf.CC_pack_buf1)
            lib.ddot(pack_buf2, dm_involved, c=pack_buf1)
            pack_buf1 = pack_buf1.reshape(nao_atm, nket_grid_involved, ket_nao_involved)
            #### pack_buf2 = np.transpose(pack_buf1, (0, 2, 1))
            pack_buf2 = np.ndarray((nao_atm, ket_nao_involved, nket_grid_involved), buffer=mydf.CC_pack_buf2)
            fn_012_to_021(
                pack_buf2.ctypes.data_as(ctypes.c_void_p),
                pack_buf1.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_atm),
                ctypes.c_int(nket_grid_involved),
                ctypes.c_int(ket_nao_involved)
            )
            pack_buf2 = pack_buf2.reshape(nao_atm, ket_nao_involved*nket_grid_involved)
            aoPaiR_ket = aoPairR_holder_ket.aoPairR.reshape(nao_atm_ket, ket_nao_involved*nket_grid_involved)
            ## ddot
            final_ddot_res = np.ndarray((nao_atm, nao_atm_ket), buffer=final_ddot_res_buf)
            lib.ddot(pack_buf2, aoPaiR_ket.T, c=final_ddot_res)
            if i == j:
                fn_pack_add_K(
                    final_ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao_atm),
                    atm_2_compactAO[i].ctypes.data_as(ctypes.c_void_p),
                    K.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao)
                )
            else:
                fn_pack_add_K2(
                    final_ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao_atm),
                    atm_2_compactAO[i].ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao_atm_ket),
                    atm_2_compactAO[j].ctypes.data_as(ctypes.c_void_p),
                    K.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao)
                )
    
    ### clean buffer ###
    
    del final_ddot_res
    
    return K * ngrid / vol

def _contract_k_mo_quadratic_subterm_CC_merged():
    pass


if __name__ == "__main__":
    
    import pyscf.pbc.df.isdf.isdf_linear_scaling_jk as ISDF_JK
    import pyscf.pbc.df.isdf.isdf_linear_scaling    as ISDF 
    from pyscf.pbc.df.isdf.isdf_tools_cell import build_supercell, build_supercell_with_partition
    import pyscf.pbc.gto as pbcgto
    
    ###### construct test system ######
    
    verbose = 4
    cell   = pbcgto.Cell()
    boxlen = 3.5668
    cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    atm = [
        ['C', (0.     , 0.     , 0.    )],
        ['C', (0.8917 , 0.8917 , 0.8917)],
        ['C', (1.7834 , 1.7834 , 0.    )],
        ['C', (2.6751 , 2.6751 , 0.8917)],
        ['C', (1.7834 , 0.     , 1.7834)],
        ['C', (2.6751 , 0.8917 , 2.6751)],
        ['C', (0.     , 1.7834 , 1.7834)],
        ['C', (0.8917 , 2.6751 , 2.6751)],
    ] 
    C = 25
    KE_CUTOFF = 70
    basis = 'gth-dzvp'
    pseudo = "gth-pade"   
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=KE_CUTOFF, basis=basis, pseudo=pseudo)    
    prim_partition = [[0,1],[2,3],[4,5],[6,7]]
    # prim_partition = [[0,1], [2,3]]
    prim_mesh = prim_cell.mesh
    
    Ls = [1, 1, 1]
    # Ls = [2, 2, 2]
    Ls = np.array(Ls, dtype=np.int32)
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    ###### construct ISDF object ###### 
    
    cell, group_partition = build_supercell_with_partition(atm, prim_a, mesh=mesh, 
                                                     Ls=Ls,
                                                     basis=basis, 
                                                     pseudo=pseudo,
                                                     partition=prim_partition, ke_cutoff=KE_CUTOFF, verbose=verbose)
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    pbc_isdf_info = ISDF.PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, aoR_cutoff=1e-8, direct=False, use_occ_RI_K=False)
    pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
    # pbc_isdf_info.Ls = Ls
    pbc_isdf_info.build_auxiliary_Coulomb(debug=True)
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    _benchmark_time(t1, t2, "build isdf", pbc_isdf_info)
    
    coords = pbc_isdf_info.coords
    ao2atmID = pbc_isdf_info.ao2atomID
    
    ###### run scf ######
    
    from pyscf.pbc import scf

    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 6
    mf.conv_tol = 1e-7
    
    mf.kernel()
    
    dm = mf.make_rdm1() 
    
    nao = cell.nao
    mo_coeff = dm.mo_coeff
    mo_occ = dm.mo_occ
    occ_tol = 1e-10
    nocc       = np.count_nonzero(mo_occ > occ_tol)
    occ_weight = np.sqrt(mo_occ[mo_occ > occ_tol])
    # print("occ_weight = ", occ_weight)
    mo_coeff_full     = mo_coeff.copy()
    mo_coeff_original = mo_coeff[:,mo_occ > occ_tol].copy()
    mo_coeff = mo_coeff[:,mo_occ > occ_tol] * occ_weight ## NOTE: it is a weighted mo_coeff
    mo_coeff = mo_coeff.copy()                           ## NOTE: nonsense thing in python
    assert mo_coeff.shape[1] == nocc
    assert mo_coeff.shape[0] == nao
    
    mo_coeff = np.random.rand(nao, nocc)
    mo_coeff = mo_coeff * occ_weight
    mo_coeff = mo_coeff.copy()
    dm = np.einsum("pi, qi -> pq", mo_coeff, mo_coeff)
    
    dm_backup = dm.copy()
    
    ###### benchmark J ######
    
    ### generate a random sequence with range nao ### 
    
    CompactAO = np.random.randint(0, cell.nao, cell.nao//3)
    CompactAO = list(set(CompactAO))
    CompactAO.sort()
    CompactAO = np.array(CompactAO, dtype=np.int32)
    
    pbc_isdf_info.aoR_RangeSeparation(CompactAO)
    
    DiffuseAO = np.array([i for i in range(cell.nao) if i not in CompactAO], dtype=np.int32)
    
    pbc_isdf_info.with_robust_fitting = False
    vj_benchmark, vk_benchmark = pbc_isdf_info.get_jk(dm)
    vj_benchmark2, _ = pbc_isdf_info.get_jk(dm)
    diff = np.linalg.norm(vj_benchmark - vj_benchmark2)
    print("diff = ", diff/np.sqrt(vj_benchmark.size))
    assert np.allclose(vj_benchmark, vj_benchmark2)
    pbc_isdf_info.with_robust_fitting = True
    vj_benchmark_robust, vk_benchmark_robust = pbc_isdf_info.get_jk(dm)
    
    # from pyscf.pbc.df.isdf.isdf_tools_densitymatrix import symmetrize_dm
    # dm = symmetrize_dm(dm, Ls)
    
    J1 = ISDF_JK._contract_j_dm_ls(pbc_isdf_info, dm, use_mpi=False, first_pass="exclude_cc", second_pass="exclude_cc")  # used in RS LR 
    J2 = ISDF_JK._contract_j_dm_ls(pbc_isdf_info, dm, use_mpi=False, first_pass="only_cc",    second_pass="exclude_cc")
    J3 = ISDF_JK._contract_j_dm_ls(pbc_isdf_info, dm, use_mpi=False, first_pass="only_cc",    second_pass="only_cc")
    J4 = ISDF_JK._contract_j_dm_ls(pbc_isdf_info, dm, use_mpi=False, first_pass="exclude_cc", second_pass="only_cc")     
    
    # J5 = ISDF_JK._contract_j_dm_ls(pbc_isdf_info, dm, use_mpi=False, first_pass="only_dd", second_pass="only_dd")        # used in RS SR (DD|DD)^{SR}     
    
    # J1 = ISDF_JK._contract_j_dm_ls(pbc_isdf_info, dm, use_mpi=False, first_pass="all", second_pass="exclude_cc")
    # J2 = ISDF_JK._contract_j_dm_ls(pbc_isdf_info, dm, use_mpi=False, first_pass="all", second_pass="only_cc")
    
    # J1 = ISDF_JK._contract_j_dm_ls(pbc_isdf_info, dm, use_mpi=False, first_pass="all", second_pass="all")
    # J1 = ISDF_JK._contract_j_dm_ls(pbc_isdf_info, dm, use_mpi=False)
    # J2 = np.zeros_like(J1)
    
    assert np.allclose(J1, J1.T)
    assert np.allclose(J2, J2.T)
    assert np.allclose(J3, J3.T)
    assert np.allclose(J4, J4.T)
    J1_CC = J1[CompactAO,:]
    J1_CC = J1_CC[:,CompactAO]
    J2_CC = J2[CompactAO,:]
    J2_CC = J2_CC[:,CompactAO]
    J3_DD = J3[DiffuseAO,:]
    J3_DD = J3_DD[:,DiffuseAO]
    J3_CD = J3[CompactAO,:]
    J3_CD = J3_CD[:,DiffuseAO]
    J4_DD = J4[DiffuseAO,:]
    J4_DD = J4_DD[:,DiffuseAO]
    J4_CD = J4[CompactAO,:]
    J4_CD = J4_CD[:,DiffuseAO]
    assert np.allclose(J1_CC, 0.0)
    assert np.allclose(J2_CC, 0.0)
    assert np.allclose(J3_DD, 0.0)
    assert np.allclose(J4_DD, 0.0)
    assert np.allclose(J3_CD, 0.0)
    assert np.allclose(J4_CD, 0.0)
    
    # J_CC = vj_benchmark[CompactAO,:]
    # J_CC = J_CC[:,CompactAO]
    # print("J_CC  = ", J_CC[0,:10])
    # print("J1_CC = ", J1_CC[0,:10])
    # assert np.allclose(J1_CC, J_CC)
    
    J = J1 + J2 + J3 + J4
    # J = J1 + J2
    diff = np.linalg.norm(J - vj_benchmark)
    print("diff = ", diff/np.sqrt(J.size))
    diff = J-vj_benchmark
    # print("diff = ", diff[0,:])
    print("J    = ", J[0,:10])
    print("vj   = ", vj_benchmark[0,:10])
    assert np.allclose(J, vj_benchmark)
    
    # exit(1)
    
    ###### benchmark K ###### 
    
    print("CompactAO = ", CompactAO)
    
    ## find the symmetry for all the involved case ##  
    
    K_W_DD_CD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="diffuse", dm_ket_type="diffuse", K_bra_type="compact", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    K_W_DD_DC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="diffuse", dm_ket_type="diffuse", K_bra_type="diffuse", K_ket_type="compact", use_mpi=False) # IN RS LR
    assert np.allclose(K_W_DD_CD, K_W_DD_DC.T)
    K_W_DD_DD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="diffuse", dm_ket_type="diffuse", K_bra_type="diffuse", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    assert np.allclose(K_W_DD_DD, K_W_DD_DD.T)
    K_W_DD_CC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="diffuse", dm_ket_type="diffuse", K_bra_type="compact", K_ket_type="compact", use_mpi=False) # IN RS LR
    assert np.allclose(K_W_DD_CC, K_W_DD_CC.T)
    K_W_DD_DD_CC_part = K_W_DD_DD[CompactAO,:]
    K_W_DD_DD_CC_part = K_W_DD_DD_CC_part[:,CompactAO]
    assert np.allclose(K_W_DD_DD_CC_part, 0.0)
    
    K_W_CC_CD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="compact", dm_ket_type="compact", K_bra_type="compact", K_ket_type="diffuse", use_mpi=False)
    K_W_CC_DC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="compact", dm_ket_type="compact", K_bra_type="diffuse", K_ket_type="compact", use_mpi=False)
    assert np.allclose(K_W_CC_CD, K_W_CC_DC.T)
    K_W_CC_DD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="compact", dm_ket_type="compact", K_bra_type="diffuse", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    print("diff = ", diff/np.sqrt(K_W_CC_DD.size))
    assert np.allclose(K_W_CC_DD, K_W_CC_DD.T)
    K_W_CC_CC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="compact", dm_ket_type="compact", K_bra_type="compact", K_ket_type="compact", use_mpi=False)
    assert np.allclose(K_W_CC_CC, K_W_CC_CC.T)
    # K_W_CC_CC_part = K_W_CC_CC[CompactAO,:]
    # K_W_CC_CC_part = K_W_CC_CC_part[:,CompactAO]
    # assert np.allclose(K_W_CC_CC[CompactAO,:], 0.0)
    
    K_W_CD_CC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="compact", dm_ket_type="diffuse", K_bra_type="compact", K_ket_type="compact", use_mpi=False)
    K_W_DC_CC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="diffuse", dm_ket_type="compact", K_bra_type="compact", K_ket_type="compact", use_mpi=False)
    assert np.allclose(K_W_CD_CC, K_W_DC_CC.T)
    K_W_CD_CD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="compact", dm_ket_type="diffuse", K_bra_type="compact", K_ket_type="diffuse", use_mpi=False)
    K_W_DC_DC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="diffuse", dm_ket_type="compact", K_bra_type="diffuse", K_ket_type="compact", use_mpi=False)
    assert np.allclose(K_W_CD_CD, K_W_DC_DC.T) 
    K_W_CD_DC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="compact", dm_ket_type="diffuse", K_bra_type="diffuse", K_ket_type="compact", use_mpi=False) # IN RS LR
    K_W_DC_CD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="diffuse", dm_ket_type="compact", K_bra_type="compact", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    assert np.allclose(K_W_CD_CC, K_W_DC_CC.T)
    K_W_CD_DD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="compact", dm_ket_type="diffuse", K_bra_type="diffuse", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    K_W_DC_DD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="diffuse", dm_ket_type="compact", K_bra_type="diffuse", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    assert np.allclose(K_W_CD_CC, K_W_DC_CC.T)
    
    K_W = (K_W_DD_CD + K_W_DD_DC + K_W_DD_DD + K_W_DD_CC) + (K_W_CC_CD + K_W_CC_DC + K_W_CC_DD + K_W_CC_CC) + K_W_CD_CC + K_W_DC_CC + K_W_CD_CD + K_W_DC_DC + K_W_CD_DC + K_W_DC_CD + K_W_CD_DD + K_W_DC_DD
    diff = np.linalg.norm(K_W - vk_benchmark)
    print("diff = ", diff/np.sqrt(K_W.size))
    assert np.allclose(K_W, vk_benchmark)
    
    K_V_DD_CD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="diffuse", dm_ket_type="diffuse", K_bra_type="compact", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    K_V_DD_DC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="diffuse", dm_ket_type="diffuse", K_bra_type="diffuse", K_ket_type="compact", use_mpi=False) # IN RS LR
    # diff = np.linalg.norm(K_V_DD_CD - K_V_DD_DC.T)
    # print("diff = ", diff/np.sqrt(K_V_DD_CD.size))
    # assert np.allclose(K_V_DD_CD, K_V_DD_DC.T)
    K_V_DD_DD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="diffuse", dm_ket_type="diffuse", K_bra_type="diffuse", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    K_V_DD_CC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="diffuse", dm_ket_type="diffuse", K_bra_type="compact", K_ket_type="compact", use_mpi=False) # IN RS LR
    
    K_V_CC_CD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="compact", dm_ket_type="compact", K_bra_type="compact", K_ket_type="diffuse", use_mpi=False)
    K_V_CC_DC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="compact", dm_ket_type="compact", K_bra_type="diffuse", K_ket_type="compact", use_mpi=False)
    # diff = np.linalg.norm(K_V_CC_CD - K_V_CC_DC.T)
    # print("diff = ", diff/np.sqrt(K_V_CC_CD.size))
    # assert np.allclose(K_V_CC_CD, K_V_CC_DC.T)
    K_V_CC_DD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="compact", dm_ket_type="compact", K_bra_type="diffuse", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    K_V_CC_CC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="compact", dm_ket_type="compact", K_bra_type="compact", K_ket_type="compact", use_mpi=False) 
    # assert np.allclose(K_V_CC_CC[CompactAO,:], 0.0)
    
    K_V   = (K_V_DD_CD+K_V_DD_CD.T+K_V_DD_DC+K_V_DD_DC.T) + K_V_DD_CC + K_V_DD_CC.T + K_V_DD_DD + K_V_DD_DD.T
    K_V  += (K_V_CC_CD+K_V_CC_CD.T+K_V_CC_DC+K_V_CC_DC.T) + K_V_CC_DD + K_V_CC_DD.T + K_V_CC_CC + K_V_CC_CC.T
    
    K_V_CD_DD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="compact", dm_ket_type="diffuse", K_bra_type="diffuse", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    K_V_DC_DD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="diffuse", dm_ket_type="compact", K_bra_type="diffuse", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    # diff = np.linalg.norm(K_V_CD_DD - K_V_DC_DD.T)
    # print("diff = ", diff/np.sqrt(K_V_CD_DD.size))
    # assert np.allclose(K_V_CD_DD, K_V_DC_DD.T)
    K_V_CD_CC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="compact", dm_ket_type="diffuse", K_bra_type="compact", K_ket_type="compact", use_mpi=False)
    K_V_DC_CC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="diffuse", dm_ket_type="compact", K_bra_type="compact", K_ket_type="compact", use_mpi=False)
    # diff = np.linalg.norm(K_V_CD_CC - K_V_DC_CC.T)
    # print("diff = ", diff/np.sqrt(K_V_CD_CC.size))
    # assert np.allclose(K_V_CD_CC, K_V_DC_CC.T)
    K_V_CD_CD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="compact", dm_ket_type="diffuse", K_bra_type="compact", K_ket_type="diffuse", use_mpi=False)
    K_V_DC_DC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="diffuse", dm_ket_type="compact", K_bra_type="diffuse", K_ket_type="compact", use_mpi=False)
    # diff = np.linalg.norm(K_V_CD_CD - K_V_DC_DC.T)
    # print("diff = ", diff/np.sqrt(K_V_CD_CD.size))
    # assert np.allclose(K_V_CD_CD, K_V_DC_DC.T)
    K_V_CD_DC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="compact", dm_ket_type="diffuse", K_bra_type="diffuse", K_ket_type="compact", use_mpi=False) # IN RS LR
    K_V_DC_CD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="diffuse", dm_ket_type="compact", K_bra_type="compact", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    # assert np.allclose(K_V_CD_DC, K_V_DC_CD.T)
    # diff = np.linalg.norm(K_V_CD_DC - K_V_DC_CD.T)
    # print("diff = ", diff/np.sqrt(K_V_CD_DC.size))
    
    # exit(1)
    
    K_V2 = (K_V_CD_DD + K_V_DC_DD + K_V_CD_CC + K_V_DC_CC + K_V_CD_CD + K_V_DC_DC + K_V_CD_DC + K_V_DC_CD) 
    K_V2 += K_V2.T
    K_V = K_V + K_V2
    
    K = K_V - K_W
    
    diff = np.linalg.norm(K - vk_benchmark_robust)
    print("diff = ", diff/np.sqrt(K.size))
    
    assert np.allclose(K, vk_benchmark_robust)
    # exit(1)
    
    dm[:,DiffuseAO] = 0.0
    dm[DiffuseAO,:] = 0.0
    
    _, vk_benchmark = pbc_isdf_info.get_jk(dm)
    
    vk_benchmark[:, DiffuseAO] = 0.0
    vk_benchmark[DiffuseAO, :] = 0.0
    
    ############ test the get_K CCCC ############ 
    
    aoPairR_holders = get_aoPairR_bruteforce(cell, coords, CompactAO, ao2atmID, cutoff=1e-10) 
    
    aomoPairR_holders = _get_AOMOPairR_holder(pbc_isdf_info, CompactAO, mo_coeff, aoPairR_holders, debug=True)
    
    atm_2_compactAO = []
    
    for i in range(cell.natm):
        atm_2_compactAO.append(np.array([j for j in CompactAO if ao2atmID[j]==i], dtype=np.int32))
    
    K_CCCC = _contract_k_mo_quadratic_subterm_CCCC(pbc_isdf_info, atm_2_compactAO, aomoPairR_holders, mesh, pbc_isdf_info.coulG) # cannot be correct and is very slow, explore locality!
    
    K_benchmark = K_V_CC_CC + K_V_CC_CC.T - K_W_CC_CC
    
    diff1 = np.linalg.norm(vk_benchmark - K_benchmark)
    print("diff1 = ", diff1/np.sqrt(K_benchmark.size))
    assert np.allclose(vk_benchmark, K_benchmark)
    
    K_CCCC_2 = _contract_k_dm_quadratic_subterm_CCCC(pbc_isdf_info, dm, atm_2_compactAO, aoPairR_holders, mesh, pbc_isdf_info.coulG)
    
    diff2 = np.linalg.norm(K_benchmark - K_CCCC_2)
    print("diff2 = ", diff2/np.sqrt(K_benchmark.size))
    # assert np.allclose(K_CCCC_2, K_benchmark)
    print("max_diff = ", np.max(np.abs(K_benchmark - K_CCCC_2)))
    print("max_diff = ", np.max(np.abs(K_CCCC - K_CCCC_2)))
    
    K1 = K_CCCC[CompactAO,:]
    K1 = K1[:,CompactAO]
    K2 = K_benchmark[CompactAO,:]
    K2 = K2[:,CompactAO]
    
    print("K_CCCC      = ", K1[0,:])
    print("K_benchmark = ", K2[0,:])
    diff = np.linalg.norm(K1 - K2)
    print("diff = ", diff/np.sqrt(K2.size))
    max_diff = np.max(np.abs(K1 - K2))
    print("max_diff = ", max_diff)
    
    # K_CCCC_bruteforce = _contract_k_mo_quadratic_subterm_CCCC_bruteforce(pbc_isdf_info, atm_2_compactAO, aomoPairR_holders, mesh, pbc_isdf_info.coulG)
    # K3 = K_CCCC_bruteforce[CompactAO,:]
    # K3 = K3[:,CompactAO]
    # K3 = K_CCCC_bruteforce
    # print("K_CCCC_bruteforce = ", K3[0,:])
    # diff2 = np.linalg.norm(K2 - K3)
    # print("diff2 = ", diff2/np.sqrt(K3.size))
    # max_diff = np.max(np.abs(K2 - K3))
    # print("max_diff = ", max_diff)
    # assert np.allclose(K3, K2)
    
    # assert np.allclose(K1, K2) this assertion is not always true, because the K1 is exact while K2 is not 
    
    
    ############ test the get_K CCCD CCDD CCDC ############
    
    dm = dm_backup.copy()
    
    mesh = np.array(mesh, dtype=np.int32)
    
    K_CC_CD = _contract_k_dm_quadratic_subterm_CC(
        pbc_isdf_info, 
        dm,
        atm_2_compactAO, 
        aoPairR_holders, 
        dm_ket_type="compact", K_ket_type="diffuse",
        mesh_bra=mesh,
        mesh_ket=mesh,
        coulG=pbc_isdf_info.coulG
    )
    
    K_benchmark = K_V_CC_CD + K_V_CC_DC.T - K_W_CC_CD 
    # K_benchmark = K_W_CC_CD
    
    # diff = K_benchmark - K_benchmark.T
    # print("diff = ", np.linalg.norm(diff)/np.sqrt(K_benchmark.size))
    
    # print("K_CC_CD = ", K_CC_CD[CompactAO[0],:])
    # print("K_benchmark = ", K_benchmark[CompactAO[0],:])
    # print("ratio = ", K_CC_CD[CompactAO[0],:]/K_benchmark[CompactAO[0],:])
    diff = np.linalg.norm(K_CC_CD - K_benchmark)
    print("diff = ", diff/np.sqrt(K_benchmark.size))
    
    # assert np.allclose(K_CC_CD, K_benchmark)  ### cannot be exacly the same as K_CC_CD is exact ! 
    
    K_CC_CC = _contract_k_dm_quadratic_subterm_CC(
        pbc_isdf_info, 
        dm,
        atm_2_compactAO, 
        aoPairR_holders, 
        dm_ket_type="compact", K_ket_type="compact",
        mesh_bra=mesh,
        mesh_ket=mesh,
        coulG=pbc_isdf_info.coulG
    )
    
    K_benchmark = K_V_CC_CC + K_V_CC_CC.T - K_W_CC_CC
    diff = np.linalg.norm(K_CC_CC - K_benchmark)
    print("diff = ", diff/np.sqrt(K_benchmark.size))
    
    K_CD_CC = _contract_k_dm_quadratic_subterm_CC(
        pbc_isdf_info, 
        dm,
        atm_2_compactAO, 
        aoPairR_holders, 
        dm_ket_type="diffuse", K_ket_type="compact",
        mesh_bra=mesh,
        mesh_ket=mesh,
        coulG=pbc_isdf_info.coulG
    )
    
    K_benchmark = K_V_CD_CC + K_V_DC_CC.T - K_W_CD_CC
    
    assert np.allclose(K_CD_CC[DiffuseAO,:], 0.0)
    assert np.allclose(K_CD_CC[:,DiffuseAO], 0.0)
    assert np.allclose(K_benchmark[DiffuseAO,:], 0.0)
    assert np.allclose(K_benchmark[:,DiffuseAO], 0.0)
    
    # print("K_CD_CC = ", K_CD_CC[CompactAO[0],CompactAO])
    # print("K_benchmark = ", K_benchmark[CompactAO[0],CompactAO])
    
    diff = np.linalg.norm(K_CD_CC - K_benchmark)
    print("diff = ", diff/np.sqrt(K_benchmark.size)) 
    
    K_CD_CD = _contract_k_dm_quadratic_subterm_CC(
        pbc_isdf_info, 
        dm,
        atm_2_compactAO, 
        aoPairR_holders, 
        dm_ket_type="diffuse", K_ket_type="diffuse",
        mesh_bra=mesh,
        mesh_ket=mesh,
        coulG=pbc_isdf_info.coulG
    )
    
    K_benchmark = K_V_CD_CD + K_V_DC_DC.T - K_W_CD_CD
    
    diff = np.linalg.norm(K_CD_CD - K_benchmark)
    print("diff = ", diff/np.sqrt(K_benchmark.size))