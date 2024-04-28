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
from pyscf.pbc.df.isdf.isdf_linear_scaling_jk import __get_DensityMatrixonRgAO_qradratic

# from memory_profiler import profile
import ctypes
from profilehooks import profile

libpbc = lib.load_library('libpbc')

############### sub terms for get K , design for RS ################

def _contract_k_dm_quadratic_subterm(mydf, dm, 
                                     # with_robust_fitting=True, 
                                     use_W = True,
                                     dm_bra_type = None, 
                                     dm_ket_type = None,
                                     K_bra_type = None,
                                     K_ket_type = None,
                                     use_mpi=False):
    
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        raise NotImplementedError("MPI is not supported yet.")
    
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
    ddot_buf1 = np.ndarray((naux, max_ncol), buffer=ddot_res_buf, offset=offset, dtype=np.float64)
    offset    = ddot_buf1.size * ddot_res_buf.dtype.itemsize
    pack_buf  = np.ndarray((naux, max_nao_involved), buffer=ddot_res_buf, offset=offset, dtype=np.float64)
    offset   += pack_buf.size * pack_buf.dtype.itemsize
    ddot_buf2 = np.ndarray((naux, max(max_ncol, max_nao_involved)), buffer=ddot_res_buf, offset=offset, dtype=np.float64)
    
    ### contract ket first ### 

    if use_W is False:
        aoR_Ket = aoR
    else:
        aoR_Ket = aoRg

    # nIP_loc = 0 # refers to ngrid_loc if use_W is False
    for aoR_holder in aoR_Ket:
        
        if aoR_holder is None:
            continue
    
        ngrid_now = aoR_holder.aoR.shape[1]
        grid_loc_now = aoR_holder.global_gridID_begin
        nao_invovled = aoR_holder.aoR.shape[0]
        nao_compact = aoR_holder.nCompact
        
        ao_begin_indx = 0 
        ao_end_indx = nao_invovled
        if dm_ket_type == "compact":
            ao_end_indx   = nao_compact
        else:
            ao_begin_indx = nao_compact
        
        nao_invovled = ao_end_indx - ao_begin_indx
        
        #### pack the density matrix ####
        
        if nao_invovled == nao:
            Density_RgAO_packed = Density_RgAO
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
        
        nao_invovled = aoR_holder.aoR.shape[0]
        ao_begin_indx = 0
        ao_end_indx   = nao_invovled
        if K_ket_type == "compact":
            ao_end_indx = nao_compact
        else:
            ao_begin_indx = nao_compact
        nao_invovled = ao_end_indx - ao_begin_indx
        
        ddot_res = np.ndarray((naux, nao_invovled), buffer=ddot_buf2)
        lib.ddot(CoulombMat_tmp, aoR_holder.aoR[ao_begin_indx:ao_end_indx, :].T, c=ddot_res)
        
        if nao_invovled == nao:
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

        # nIP_loc += ngrid_now
    # del W_tmp
    # assert nIP_loc == naux
        
    K = np.zeros((nao, nao), dtype=np.float64) 
    
    # nIP_loc = 0
    for aoR_holder in aoRg:
        
        if aoR_holder is None:
            continue
    
        grid_loc_now = aoR_holder.global_gridID_begin
        ngrid_now = aoR_holder.aoR.shape[1]
        nao_invovled = aoR_holder.aoR.shape[0]
        
        ao_begin_indx = 0
        ao_end_indx = nao_invovled
        if K_bra_type == "compact":
            ao_end_indx = nao_compact
        else:
            ao_begin_indx = nao_compact
        nao_invovled = ao_end_indx - ao_begin_indx
        
        K_tmp = K1[grid_loc_now:grid_loc_now+ngrid_now, :]
        
        ddot_res = np.ndarray((nao_invovled, nao), buffer=ddot_res_buf)
        lib.ddot(aoR_holder.aoR[ao_begin_indx:ao_end_indx, :], K_tmp, c=ddot_res)
        
        if nao_invovled == nao:
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
        
        # nIP_loc += ngrid_now
    # del K_tmp
    # assert nIP_loc == naux
    
    ######### finally delete the buffer #########
    
    del K1
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    if mydf.verbose:
        _benchmark_time(t1, t2, "_contract_k_dm_quadratic")
    
    return K * ngrid / vol



def _contract_k_dm_quadratic_subterm_merged():
    pass




if __name__ == "__main__":
    
    import pyscf.pbc.df.isdf.isdf_linear_scaling_jk as ISDF_JK
    import pyscf.pbc.df.isdf.isdf_linear_scaling    as ISDF 
    from pyscf.pbc.df.isdf.isdf_tools_cell import build_supercell, build_supercell_with_partition
    import pyscf.pbc.gto as pbcgto
    
    ###### construct test system ######
    
    verbose = 1
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
    C = 10
    KE_CUTOFF = 70
    basis = 'gth-dzvp'
    pseudo = "gth-pade"   
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=KE_CUTOFF, basis=basis, pseudo=pseudo)    
    prim_partition = [[0,1],[2,3],[4,5],[6,7]]
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
    pbc_isdf_info.Ls = Ls
    pbc_isdf_info.build_auxiliary_Coulomb(debug=True)
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    _benchmark_time(t1, t2, "build isdf")
    
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
    pbc_isdf_info.with_robust_fitting = True
    vj_benchmark_robust, vk_benchmark_robust = pbc_isdf_info.get_jk(dm)
    
    J1 = ISDF_JK._contract_j_dm_ls(pbc_isdf_info, dm, use_mpi=False, first_pass="exclude_cc", second_pass="exclude_cc")  # used in RS LR 
    J2 = ISDF_JK._contract_j_dm_ls(pbc_isdf_info, dm, use_mpi=False, first_pass="only_cc",    second_pass="exclude_cc")
    J3 = ISDF_JK._contract_j_dm_ls(pbc_isdf_info, dm, use_mpi=False, first_pass="only_cc",    second_pass="only_cc")
    J4 = ISDF_JK._contract_j_dm_ls(pbc_isdf_info, dm, use_mpi=False, first_pass="exclude_cc", second_pass="only_cc")     
    # J5 = ISDF_JK._contract_j_dm_ls(pbc_isdf_info, dm, use_mpi=False, first_pass="only_dd", second_pass="only_dd")        # used in RS SR (DD|DD)^{SR}     
    
    assert np.allclose(J1, J1.T)
    assert np.allclose(J2, J2.T)
    assert np.allclose(J3, J3.T)
    assert np.allclose(J4, J4.T)
    
    J = J1 + J2 + J3 + J4
    diff = np.linalg.norm(J - vj_benchmark)
    print("diff = ", diff/np.sqrt(J.size))
    assert np.allclose(J, vj_benchmark)
    
    ###### benchmark K ###### 
    
    ## find the symmetry for all the involved case ##  
    
    K_W_DD_CD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="diffuse", dm_ket_type="diffuse", K_bra_type="compact", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    K_W_DD_DC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="diffuse", dm_ket_type="diffuse", K_bra_type="diffuse", K_ket_type="compact", use_mpi=False) # IN RS LR
    assert np.allclose(K_W_DD_CD, K_W_DD_DC.T)
    K_W_DD_DD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="diffuse", dm_ket_type="diffuse", K_bra_type="diffuse", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    assert np.allclose(K_W_DD_DD, K_W_DD_DD.T)
    K_W_DD_CC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="diffuse", dm_ket_type="diffuse", K_bra_type="compact", K_ket_type="compact", use_mpi=False) # IN RS LR
    assert np.allclose(K_W_DD_CC, K_W_DD_CC.T)
    
    K_W_CC_CD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="compact", dm_ket_type="compact", K_bra_type="compact", K_ket_type="diffuse", use_mpi=False)
    K_W_CC_DC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="compact", dm_ket_type="compact", K_bra_type="diffuse", K_ket_type="compact", use_mpi=False)
    assert np.allclose(K_W_CC_CD, K_W_CC_DC.T)
    K_W_CC_DD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="compact", dm_ket_type="compact", K_bra_type="diffuse", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    print("diff = ", diff/np.sqrt(K_W_CC_DD.size))
    assert np.allclose(K_W_CC_DD, K_W_CC_DD.T)
    K_W_CC_CC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="compact", dm_ket_type="compact", K_bra_type="compact", K_ket_type="compact", use_mpi=False)
    assert np.allclose(K_W_CC_CC, K_W_CC_CC.T)
    
    K_W_CD_CC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="compact", dm_ket_type="diffuse", K_bra_type="compact", K_ket_type="compact", use_mpi=False)
    K_W_DC_CC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="diffuse", dm_ket_type="compact", K_bra_type="compact", K_ket_type="compact", use_mpi=False)
    assert np.allclose(K_W_CD_CC, K_W_DC_CC.T)
    K_W_CD_CD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="compact", dm_ket_type="diffuse", K_bra_type="compact", K_ket_type="diffuse", use_mpi=False)
    K_W_DC_DC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="diffuse", dm_ket_type="compact", K_bra_type="diffuse", K_ket_type="compact", use_mpi=False)
    assert np.allclose(K_W_CD_CD, K_W_DC_DC.T) 
    K_W_CD_DC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="compact", dm_ket_type="diffuse", K_bra_type="compact", K_ket_type="compact", use_mpi=False) # IN RS LR
    K_W_DC_CD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="diffuse", dm_ket_type="compact", K_bra_type="compact", K_ket_type="compact", use_mpi=False) # IN RS LR
    assert np.allclose(K_W_CD_CC, K_W_DC_CC.T)
    K_W_CD_DD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="compact", dm_ket_type="diffuse", K_bra_type="compact", K_ket_type="compact", use_mpi=False) # IN RS LR
    K_W_DC_DD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=True, dm_bra_type="diffuse", dm_ket_type="compact", K_bra_type="compact", K_ket_type="compact", use_mpi=False) # IN RS LR
    assert np.allclose(K_W_CD_CC, K_W_DC_CC.T)
    
    K_W = K_W_DD_CD + K_W_DD_DC + K_W_DD_DD + K_W_DD_CC + K_W_CC_CD + K_W_CC_DC + K_W_CC_DD + K_W_CC_CC + K_W_CD_CC + K_W_DC_CC + K_W_CD_CD + K_W_DC_DC + K_W_CD_DC + K_W_DC_CD + K_W_CD_DD + K_W_DC_DD
    assert np.allclose(K_W, vk_benchmark)
    
    K_V_DD_CD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="diffuse", dm_ket_type="diffuse", K_bra_type="compact", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    K_V_DD_DC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="diffuse", dm_ket_type="diffuse", K_bra_type="diffuse", K_ket_type="compact", use_mpi=False) # IN RS LR
    assert np.allclose(K_V_DD_CD, K_V_DD_DC.T)
    K_V_DD_DD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="diffuse", dm_ket_type="diffuse", K_bra_type="diffuse", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    K_V_DD_CC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="diffuse", dm_ket_type="diffuse", K_bra_type="compact", K_ket_type="compact", use_mpi=False) # IN RS LR
    
    K_V_CC_CD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="compact", dm_ket_type="compact", K_bra_type="compact", K_ket_type="diffuse", use_mpi=False)
    K_V_CC_DC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="compact", dm_ket_type="compact", K_bra_type="diffuse", K_ket_type="compact", use_mpi=False)
    assert np.allclose(K_V_CC_CD, K_V_CC_DC.T)
    K_V_CC_DD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="compact", dm_ket_type="compact", K_bra_type="diffuse", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    K_V_CC_CC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="compact", dm_ket_type="compact", K_bra_type="compact", K_ket_type="compact", use_mpi=False) 
    
    K_V   = 2*(K_V_DD_CD+K_V_DD_DC) + K_V_DD_CC + K_V_DD_CC.T + K_V_DD_CC + K_V_DD_CC.T
    K_V  += 2*(K_V_CC_CD+K_V_CC_DC) + K_V_CC_DD + K_V_CC_DD.T + K_V_CC_CC + K_V_CC_CC.T
    
    K_V_CD_DD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="compact", dm_ket_type="diffuse", K_bra_type="diffuse", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    K_V_DC_DD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="diffuse", dm_ket_type="compact", K_bra_type="diffuse", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    assert np.allclose(K_V_CD_DD, K_V_DC_DD.T)
    K_V_CD_CC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="compact", dm_ket_type="diffuse", K_bra_type="compact", K_ket_type="compact", use_mpi=False)
    K_V_DC_CC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="diffuse", dm_ket_type="compact", K_bra_type="compact", K_ket_type="compact", use_mpi=False)
    assert np.allclose(K_V_CD_CC, K_V_DC_CC.T)
    K_V_CD_CD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="compact", dm_ket_type="diffuse", K_bra_type="compact", K_ket_type="diffuse", use_mpi=False)
    K_V_DC_DC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="diffuse", dm_ket_type="compact", K_bra_type="diffuse", K_ket_type="compact", use_mpi=False)
    assert np.allclose(K_V_CD_CD, K_V_DC_DC.T)
    K_V_CD_DC = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="compact", dm_ket_type="diffuse", K_bra_type="diffuse", K_ket_type="compact", use_mpi=False) # IN RS LR
    K_V_DC_CD = _contract_k_dm_quadratic_subterm(pbc_isdf_info, dm, use_W=False, dm_bra_type="diffuse", dm_ket_type="compact", K_bra_type="compact", K_ket_type="diffuse", use_mpi=False) # IN RS LR
    assert np.allclose(K_V_CD_DC, K_V_DC_CD.T)
    
    K_V += (K_V_CD_DD + K_V_DC_DD + K_V_CD_CC + K_V_DC_CC + K_V_CD_CD + K_V_DC_DC + K_V_CD_DC + K_V_DC_CD) * 2
    
    K = K_V - K_W
    
    diff = np.linalg.norm(K - vk_benchmark_robust)
    print("diff = ", diff/np.sqrt(K.size))
    
    assert np.allclose(K, vk_benchmark_robust)