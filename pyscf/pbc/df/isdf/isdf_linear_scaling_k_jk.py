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
import numpy as np
import ctypes

############ pyscf module ############

from pyscf import lib
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from pyscf.gto.mole import *
from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0, _format_dms, _format_kpts_band, _format_jks
libpbc = lib.load_library('libpbc')

from pyscf.pbc.df.isdf.isdf_tools_densitymatrix import pack_JK, pack_JK_in_FFT_space
from pyscf.pbc.df.isdf.isdf_linear_scaling_jk   import J_MAX_GRID_BUNCHSIZE, __get_DensityMatrixonRgAO_qradratic
from pyscf.pbc.df.isdf.isdf_tools_kSampling     import _RowCol_FFT_bench
from pyscf.pbc.df.isdf._isdf_local_K_direct     import _isdf_get_K_direct_kernel_1

############ subroutines ############

def _preprocess_dm(mydf, dm):

    in_real_space = True

    kmesh = np.asarray(mydf.kmesh, dtype=np.int32)
    ncell_complex = kmesh[0] * kmesh[1] * (kmesh[2]//2+1)
    
    if len(dm.shape) == 3:
        if dm.shape[0] == 1:
            if dm.dtype == np.float64:
                dm = dm[0].real
            else:
                in_real_space = False
                dm = dm[0].real
        else:  
            
            #print("dm.shape = ", dm.shape)
            #print("dm = ", dm)
            #print("dtype = ", dm.dtype)
            
            in_real_space = False 
                        
            if dm.dtype == np.float64:
                #assert kmesh[0] in [1, 2]
                #assert kmesh[1] in [1, 2]
                #assert kmesh[2] in [1, 2]
                dm = np.asarray(dm, dtype=np.complex128)
            
            assert dm.dtype    == np.complex128
            assert dm.shape[1] == dm.shape[2]
            assert dm.shape[0] == np.prod(kmesh)
            
            nao_prim   = dm.shape[1]
            nkpts      = dm.shape[0]
            
            #dm_complex = np.transpose(dm, axes=(1, 0, 2)).copy()
            #dm_complex = dm_complex.reshape(nao_prim, -1)
            
            ### check the symmetry ###
            
            for ix in range(kmesh[0]):
                for iy in range(kmesh[1]):
                    for iz in range(kmesh[2]):
                        loc1 = ix * kmesh[1] * kmesh[2] + iy * kmesh[2] + iz
                        loc2 = (kmesh[0] - ix) % kmesh[0] * kmesh[1] * kmesh[2] + (kmesh[1] - iy) % kmesh[1] * kmesh[2] + (kmesh[2] - iz) % kmesh[2]
                        #print("loc1     = ", loc1, "loc2 = ", loc2)
                        #print("dm[loc1] = ", dm[loc1])
                        #print("dm[loc2] = ", dm[loc2])
                        diff = np.linalg.norm(dm[loc1] - dm[loc2].conj()) / np.sqrt(dm.size)
                        #print("diff = ", diff) ## NOTE: should be very small
                        #assert diff < 1e-7
                        if diff > 1e-7:
                            print("warning, the input density matrix is not symmetric.")
                            print("k1    = (%d, %d, %d) " % (ix, iy, iz))
                            print("k2    = (%d, %d, %d) " % ((kmesh[0] - ix) % kmesh[0], (kmesh[1] - iy) % kmesh[1], (kmesh[2] - iz) % kmesh[2]))
                            print("kmesh = ", kmesh)
                            print("diff  = ", diff)
            dm_complex = np.zeros((ncell_complex, nao_prim, nao_prim), dtype=np.complex128)
            loc = 0
            for ix in range(kmesh[0]):
                for iy in range(kmesh[1]):
                    for iz in range(kmesh[2]//2+1):
                        loc1 = ix * kmesh[1] * kmesh[2] + iy * kmesh[2] + iz
                        loc2 = (kmesh[0] - ix) % kmesh[0] * kmesh[1] * kmesh[2] + (kmesh[1] - iy) % kmesh[1] * kmesh[2] + (kmesh[2] - iz) % kmesh[2]
                        #dm_complex[loc].ravel()[:] = dm[loc1].ravel()[:]
                        dm_input = ((dm[loc1] + dm[loc2].conj()) / 2.0).copy()
                        dm_complex[loc].ravel()[:] = dm_input.ravel()[:]
                        loc += 1
            
            dm_complex = np.transpose(dm_complex, axes=(1, 0, 2)).copy()
            dm_complex = dm_complex.conj().copy()
            
            #print("dm_complex.shape = ", dm_complex.shape)
            #print("dm_complex = ", dm_complex[:, 0, :])
            #print("dm_complex = ", dm_complex[:, 1, :])
            
            ### do the FFT ### 
            
            dm_real = np.ndarray((nao_prim, nkpts * nao_prim), dtype=np.float64, buffer=dm_complex)
            buf_fft = np.zeros((nao_prim, ncell_complex, nao_prim), dtype=np.complex128)
            
            fn2 = getattr(libpbc, "_iFFT_Matrix_Col_InPlace", None)
            assert fn2 is not None

            fn2(
                dm_complex.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_prim),
                ctypes.c_int(nao_prim),
                kmesh.ctypes.data_as(ctypes.c_void_p),
                buf_fft.ctypes.data_as(ctypes.c_void_p)
            )
            
            #print("dm_real    = ", dm_real)
            #print("dm_complex = ", dm_complex)
            
            dm = pack_JK(dm_real, kmesh, nao_prim)
            
            #print("dm.shape = ", dm.shape)

    return dm, in_real_space
    
def _contract_j_dm_k_ls(mydf, _dm, use_mpi=False):
    
    dm, in_real_space = _preprocess_dm(mydf, _dm)
    
    if use_mpi:
        assert mydf.direct == True
        from pyscf.pbc.df.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast, reduce
        size = comm_size
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
    ngrid_prim = ngrid // np.prod(mydf.kmesh)

    aoR  = mydf.aoR
    assert isinstance(aoR, list)
    naux = mydf.naux
    aoR1 = mydf.aoR1
    assert isinstance(aoR1, list)
    
    kmesh = np.array(mydf.kmesh, dtype=np.int32)
    ncell = np.prod(kmesh)
    ncell_complex = kmesh[0] * kmesh[1] * (kmesh[2]//2+1)
    
    #### step 0. allocate buffer 
    
    max_nao_involved = np.max([aoR_holder.aoR.shape[0] for aoR_holder in aoR if aoR_holder is not None])
    max_nao_involved1 = np.max([aoR_holder.aoR.shape[0] for aoR_holder in aoR1 if aoR_holder is not None])
    max_nao_involved = max(max_nao_involved, max_nao_involved1)
    max_ngrid_involved = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None])
    max_ngrid_involved1 = np.max([aoR_holder.aoR.shape[1] for aoR_holder in aoR1 if aoR_holder is not None])
    max_ngrid_involved = max(max_ngrid_involved, max_ngrid_involved1)

    density_R_prim = np.zeros((ngrid_prim,), dtype=np.float64)
    
    dm_buf      = np.zeros((max_nao_involved, max_nao_involved), dtype=np.float64)
    # max_dim_buf = max(max_ngrid_involved, max_nao_involved)
    max_dim_buf = max_nao_involved
    max_col_buf = min(max_ngrid_involved, J_MAX_GRID_BUNCHSIZE)
    aoR_buf1    = np.zeros((max_nao_involved, max_ngrid_involved), dtype=np.float64)
    
    ##### get the involved C function ##### 
    
    fn_extract_dm = getattr(libpbc, "_extract_dm_involved_ao", None) 
    assert fn_extract_dm is not None
    
    fn_packadd_dm = getattr(libpbc, "_packadd_local_dm", None)
    assert fn_packadd_dm is not None
    
    fn_multiplysum = getattr(libpbc, "_fn_J_dmultiplysum", None)
    assert fn_multiplysum is not None
    
    #### step 1. get density value on real space grid and IPs
    
    density_R_tmp = None
    
    ddot_buf    = np.zeros((max_nao_involved, max_col_buf), dtype=np.float64)
    
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
        global_gridID_begin = aoR_holder.global_gridID_begin
        
        for p0, p1 in lib.prange(0, ngrids_now, J_MAX_GRID_BUNCHSIZE):
            ddot_res = np.ndarray((nao_involved, p1-p0), buffer=ddot_buf)
            lib.ddot(dm_now, aoR_holder.aoR[:,p0:p1], c=ddot_res)
            #density_R_tmp = lib.multiply_sum_isdf(aoR_holder.aoR[:,p0:p1], ddot_res)
            _res_tmp = np.ndarray((p1-p0,),
                                dtype =density_R_prim.dtype, 
                                buffer=density_R_prim, 
                                offset=(global_gridID_begin+p0)*density_R_prim.dtype.itemsize)
            # density_R_prim[global_gridID_begin+p0:global_gridID_begin+p1] = density_R_tmp
            fn_multiplysum(
                    _res_tmp.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao_involved),
                    ctypes.c_int(p1-p0),
                    aoR_holder.aoR.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(aoR_holder.aoR.shape[0]),
                    ctypes.c_int(aoR_holder.aoR.shape[1]),
                    ctypes.c_int(0),
                    ctypes.c_int(p0),
                    ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao_involved),
                    ctypes.c_int(p1-p0),
                    ctypes.c_int(0),
                    ctypes.c_int(0))
        #ddot_res = np.ndarray((nao_involved, ngrids_now), buffer=ddot_buf)
        #lib.ddot(dm_now, aoR_holder.aoR, c=ddot_res)
        #density_R_tmp = lib.multiply_sum_isdf(aoR_holder.aoR, ddot_res)
        #density_R_prim[global_gridID_begin:global_gridID_begin+ngrids_now] = density_R_tmp
    
    if use_mpi:
        density_R_prim = reduce(density_R_prim, root=0)
            
    grid_ID_ordered = mydf.grid_ID_ordered_prim
    
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
    
    J = None
    
    ddot_buf    = np.zeros((max_nao_involved, max_nao_involved), dtype=np.float64)
    
    if (use_mpi and rank == 0) or (use_mpi == False):
    
        fn_J = getattr(libpbc, "_construct_J", None)
        assert(fn_J is not None)

        if hasattr(mydf, "coulG_prim") == False:
            assert mydf.omega is None or mydf.omega == 0.0
            mydf.coulG_prim = tools.get_coulG(mydf.primCell, mesh=mydf.primCell.mesh)

        J = np.zeros_like(density_R_prim)

        mesh_prim = np.array(mydf.primCell.mesh, dtype=np.int32)

        fn_J(
            mesh_prim.ctypes.data_as(ctypes.c_void_p),
            density_R_prim.ctypes.data_as(ctypes.c_void_p),
            mydf.coulG_prim.ctypes.data_as(ctypes.c_void_p),
            J.ctypes.data_as(ctypes.c_void_p),
        )
                    
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
            
    if use_mpi:
        J = bcast(J, root=0)
    
    #### step 3. get J , using translation symmetry ###

    nao_prim = mydf.nao_prim
    J_Res = np.zeros((nao_prim, nao), dtype=np.float64)

    partition_activated_ID = mydf.partition_activated_id
        
    kmesh     = np.asarray(mydf.kmesh, dtype=np.int32)
    natm_prim = mydf.natmPrim
    
    grid_segment = mydf.grid_segment
    
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
    
    J = J_Res
    if use_mpi:
        J = reduce(J, root=0)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    _benchmark_time(t1, t2, "_contract_j_dm_k_ls", mydf)
    
    ######### delete the buffer #########
    
    del dm_buf, ddot_buf, density_R_prim
    del density_R_tmp
    del aoR_buf1
    
    J *= ngrid / vol
    
    if in_real_space:
        J = pack_JK(J, mydf.kmesh, nao_prim)
    else:
        ## transform J back to FFT space ##
        fn1 = getattr(libpbc, "_FFT_Matrix_Col_InPlace", None)
        assert fn1 is not None
        J_complex = np.ndarray((nao_prim,nao_prim*ncell_complex), dtype=np.complex128)
        fft_buf   = np.ndarray((nao_prim,nao_prim*ncell_complex), dtype=np.complex128)
        J_real    = np.ndarray((nao_prim,nao_prim*ncell),         dtype=np.float64,    buffer=J_complex)
        J_real.ravel()[:]    = J.ravel()[:]
        fn1(
            J_real.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_prim),
            ctypes.c_int(nao_prim),
            kmesh.ctypes.data_as(ctypes.c_void_p),
            fft_buf.ctypes.data_as(ctypes.c_void_p)
        )
        del fft_buf
        ## pack J in FFT space ##
        J_complex = J_complex.conj().copy()
        J = pack_JK_in_FFT_space(J_complex, mydf.kmesh, nao_prim)
    
    return J

def _get_k_kSym_robust_fitting_fast(mydf, _dm):
    
    '''
    this is a slow version, abandon ! 
    '''
 
    #### preprocess ####  
    
    dm, in_real_space = _preprocess_dm(mydf, _dm)
    
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
    
    W    = mydf.W
    naux = mydf.naux
    
    kmesh = np.array(mydf.kmesh, dtype=np.int32)
    mesh = mydf.mesh
    meshPrim = np.array(mesh) // np.array(kmesh)
    nGridPrim = mydf.nGridPrim
    ncell = np.prod(kmesh)
    ncell_complex = kmesh[0] * kmesh[1] * (kmesh[2]//2+1)
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
        kmesh.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "_fft1", mydf)
    
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
    _benchmark_time(t3, t4, "DM_RgRg_complex", mydf)
    
    t3 = t4
    
    buf_fft = np.ndarray((nIP_prim, nIP_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    fn2 = getattr(libpbc, "_iFFT_Matrix_Col_InPlace", None)
    assert fn2 is not None
        
    fn2(
        DM_RgRg_complex.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_prim),
        ctypes.c_int(nIP_prim),
        kmesh.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "DM_RgRg_complex 2", mydf)
    t3 = t4
    
    # inplace multiplication
    
    lib.cwise_mul(mydf.W, DM_RgRg_real, out=DM_RgRg_real)
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "lib.cwise_mul 2", mydf)
    t3 = t4
    
    offset = offset_after_DM
    
    buf_fft = np.ndarray((nIP_prim, nIP_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    fn1(
        DM_RgRg_real.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_prim),
        ctypes.c_int(nIP_prim),
        kmesh.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "DM_RgRg_real", mydf)
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
    _benchmark_time(t3, t4, "K_complex_buf", mydf)
    t3 = t4
    
    #if in_real_space:
    
    buf_fft = np.ndarray((nao_prim, nao_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    fn2(
        K_complex_buf.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao_prim),
        ctypes.c_int(nao_prim),
        kmesh.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "K_real_buf", mydf)
    t3 = t4
    
    K_real_buf *= (ngrid / vol)
    
    K = -pack_JK(K_real_buf, kmesh, nao_prim, output=None) # "-" due to robust fitting
    
    #else:
    #    K = -pack_JK_in_FFT_space(K_complex_buf, kmesh, nao_prim) / np.prod(kmesh)
    
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
    _benchmark_time(t3, t4, "DM_RgR_complex", mydf)
    t3 = t4
    
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
        kmesh.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
        
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "DM_RgR_real", mydf)
    t3 = t4
        
    # inplace multiplication
    
    lib.cwise_mul(mydf.V_R, DM_RgR_real, out=DM_RgR_real)
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "cwise_mul", mydf)
    t3 = t4
        
    fn1(
        DM_RgR_real.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_prim),
        ctypes.c_int(nGridPrim),
        kmesh.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "DM_RgR_complex 2", mydf)
    t3 = t4
        
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
    _benchmark_time(t3, t4, "K_complex_buf 1", mydf)
    t3 = t4
    
    buf_A = None
    buf_B = None
    buf_B2 = None
    buf_C = None
    buf_D = None
    
    offset_now = offset_after_K
    
    buf_fft = np.ndarray((nao_prim, nao_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_now)
        
    fn2(
        K_complex_buf.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao_prim),
        ctypes.c_int(nao_prim),
        kmesh.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "K_complex_buf 2", mydf)
    t3 = t4
    
    buf_fft = None
    
    K_real_buf *= (ngrid / vol)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    _benchmark_time(t1, t2, "_contract_k_dm", mydf)
    
    t1 = t2
    
    K2 = pack_JK(K_real_buf, kmesh, nao_prim, output=None)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    _benchmark_time(t1, t2, "_pack_JK", mydf)
    
    K += K2 + K2.T
    
    if in_real_space == False:
    #    K += K2 + K2.T
    #else:
    #    K2 = K2 + K2.T
    #    K2 = K2[:nao_prim,:]
        
        K = K[:nao_prim,:].copy()

        K_complex = np.ndarray((nao_prim, nao_prim*ncell_complex), dtype=np.complex128)
        K_real    = np.ndarray((nao_prim, nao_prim*ncell), dtype=np.float64, buffer=K_complex)
        K_real.ravel()[:]    = K.ravel()[:]
        buf_fft    = np.zeros_like(K_complex)
        
        fn1(
            K_real.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_prim),
            ctypes.c_int(nao_prim),
            kmesh.ctypes.data_as(ctypes.c_void_p),
            buf_fft.ctypes.data_as(ctypes.c_void_p)
        )
        
        K_complex = K_complex.conj().copy()
        K_complex = pack_JK_in_FFT_space(K_complex, kmesh, nao_prim)
        K = K_complex 
        
        
    DM_RgR_complex = None
    DM_RgR_real = None
    
    return K
    
    # return DM_RgRg_real # temporary return for debug

def _get_k_kSym(mydf, _dm):
 
    #### preprocess ####  
    
    dm, in_real_space = _preprocess_dm(mydf, _dm)
    
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
    
    kmesh = np.array(mydf.kmesh, dtype=np.int32)
    mesh = mydf.mesh
    meshPrim = np.array(mesh) // np.array(kmesh)
    nGridPrim = mydf.nGridPrim
    ncell = np.prod(kmesh)
    ncell_complex = kmesh[0] * kmesh[1] * (kmesh[2]//2+1)
    nIP_prim = mydf.nIP_Prim
    nao_prim = nao // ncell
    
    #### allocate buffer ####
    
    offset          = 0
    DM_RgRg_complex = np.ndarray((nIP_prim,nIP_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    DM_RgRg_real    = np.ndarray((nIP_prim,nIP_prim*ncell),         dtype=np.float64,    buffer=mydf.jk_buffer, offset=offset)
    
    offset += (nIP_prim * nIP_prim * ncell_complex) * DM_RgRg_complex.itemsize
    DM_complex = np.ndarray((nao_prim,nao_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    DM_real = np.ndarray((nao_prim,nao), dtype=np.float64, buffer=mydf.jk_buffer, offset=offset)
    DM_real.ravel()[:] = dm[:nao_prim, :].ravel()[:]
    offset += (nao_prim * nao_prim * ncell_complex) * DM_complex.itemsize
    
    #### get D ####
    
    #_get_DM_RgRg_real(mydf, DM_real, DM_complex, DM_RgRg_real, DM_RgRg_complex, offset)
    
    fn1 = getattr(libpbc, "_FFT_Matrix_Col_InPlace", None)
    assert fn1 is not None
    
    buf_fft = np.ndarray((nao_prim, nao_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
        
    fn1(
        DM_real.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao_prim),
        ctypes.c_int(nao_prim),
        kmesh.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    buf_A = np.ndarray((nao_prim, nao_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    offset2 = offset + (nao_prim * nao_prim) * buf_A.itemsize
    buf_B = np.ndarray((nao_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset2)
    
    offset3 = offset2 + (nao_prim * nIP_prim) * buf_B.itemsize
    buf_C = np.ndarray((nao_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset3)
    
    offset4 = offset3 + (nao_prim * nIP_prim) * buf_C.itemsize
    buf_D = np.ndarray((nIP_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset4)
    
    aoRg_FFT = mydf.aoRg_FFT
    
    if isinstance(aoRg_FFT, list): 
        for i in range(ncell_complex):
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            buf_A[:] = DM_complex[:, k_begin:k_end]
            # buf_B[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim]
            buf_B = aoRg_FFT[i]
        
            lib.dot(buf_A, buf_B, c=buf_C)
            lib.dot(buf_B.T.conj(), buf_C, c=buf_D)
        
            k_begin = i * nIP_prim
            k_end   = (i + 1) * nIP_prim
        
            DM_RgRg_complex[:, k_begin:k_end] = buf_D
    else:
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
    
    buf_fft = np.ndarray((nIP_prim, nIP_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    fn2 = getattr(libpbc, "_iFFT_Matrix_Col_InPlace", None)
    assert fn2 is not None
    
    fn2(
        DM_RgRg_complex.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_prim),
        ctypes.c_int(nIP_prim),
        kmesh.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    # inplace multiplication
    
    lib.cwise_mul(mydf.W, DM_RgRg_real, out=DM_RgRg_real)
    
    offset = nIP_prim * nIP_prim * ncell_complex * DM_RgRg_complex.itemsize
    
    buf_fft = np.ndarray((nIP_prim, nIP_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    fn1(
        DM_RgRg_real.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_prim),
        ctypes.c_int(nIP_prim),
        kmesh.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
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
        
            buf_A.ravel()[:] = DM_RgRg_complex[:, k_begin:k_end].ravel()[:]
            # buf_B.ravel()[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim].ravel()[:]
            buf_B = aoRg_FFT[i]
        
            lib.dot(buf_A, buf_B.T.conj(), c=buf_C)
            lib.dot(buf_B, buf_C, c=buf_D)
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            K_complex_buf[:, k_begin:k_end] = buf_D
    else:
        
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
    
    buf_fft = np.ndarray((nao_prim, nao_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    K_complex_buf *= (ngrid / vol)
    
    #print("K_complex_buf = ", K_complex_buf)
    
    if in_real_space:
        
        fn2(
            K_complex_buf.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_prim),
            ctypes.c_int(nao_prim),
            kmesh.ctypes.data_as(ctypes.c_void_p),
            buf_fft.ctypes.data_as(ctypes.c_void_p)
        )
    
        K = pack_JK(K_real_buf, kmesh, nao_prim, output=None)
    
    else:
    
        K_complex_buf = K_complex_buf.conj().copy()  ### NOTE: convention problem   
        K = pack_JK_in_FFT_space(K_complex_buf, kmesh, nao_prim, output=None)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    _benchmark_time(t1, t2, "_contract_k_dm", mydf)
    
    return K
   
def _get_k_kSym_direct(mydf, _dm, use_mpi=False):
    
    if use_mpi:
        assert mydf.direct == True
        from pyscf.pbc.df.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast, reduce
        size = comm.Get_size()
    
    t1 = (logger.process_clock(), logger.perf_counter())
    
    ############# preprocess #############
    
    dm, in_real_space = _preprocess_dm(mydf, _dm)
    if in_real_space:
        if np.prod(mydf.kmesh) == 1:
            in_real_space = False
    assert not in_real_space
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]
        
    aoR  = mydf.aoR
    aoRg = mydf.aoRg    
    
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
    nao  = cell.nao
    nao_prim  = mydf.nao_prim
    aux_basis = mydf.aux_basis
    kmesh     = np.array(mydf.kmesh, dtype=np.int32)
    nkpts     = np.prod(kmesh)
    
    grid_ordering = mydf.grid_ID_ordered 
    
    if hasattr(mydf, "coulG") == False:
        if mydf.omega is not None:
            assert mydf.omega >= 0.0
        # mydf.coulG = tools.get_coulG(cell, mesh=mesh, omega=mydf.omega)
        raise NotImplementedError("coulG is not implemented yet.")
    
    coulG = mydf.coulG
    coulG_real = coulG.reshape(*mesh)[:, :, :mesh[2]//2+1].reshape(-1).copy()
    
    mydf.allocate_k_buffer()
    build_k_buf  = mydf.build_k_buf
    build_VW_buf = mydf.build_VW_in_k_buf
    
    group = mydf.group
    assert len(group) == len(aux_basis)
    
    ######### allocate buffer ######### 
        
    Density_RgAO_buf = mydf.Density_RgAO_buf
    
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
    
    K1 = np.zeros((nao_prim, nao), dtype=np.float64) # contribution from V matrix
    K2 = np.zeros((nao_prim, nao), dtype=np.float64) # contribution from W matrix
    
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
                
    ######### finally delete the buffer #########
    
    if use_mpi:
        comm.Barrier()
    
    if use_mpi:
        K1 = reduce(K1, root = 0)
        K2 = reduce(K2, root = 0)
        if rank == 0:
            # K = K1 + K1.T - K2
            K1 = pack_JK(K1, kmesh, nao_prim)
            K2 = pack_JK(K2, kmesh, nao_prim)
            K  = K1 + K1.T - K2
        else:
            K = None
        K = bcast(K, root = 0)
    else:
        # K = K1 + K1.T - K2 
        K1 = pack_JK(K1, kmesh, nao_prim)
        K2 = pack_JK(K2, kmesh, nao_prim)
        K  = K1 + K1.T - K2
    
    del K1
    del K2
    
    ############ transform back to K ############
        
    # print("K = ", K[0])
        
    K = _RowCol_FFT_bench(K[:nao_prim, :], kmesh, inv=True, TransBra=False, TransKet=True)
    K*= nkpts
    K*= ngrid / vol
    
    Res = []
    for i in range(np.prod(kmesh)):
        Res.append(K[:, i*nao_prim:(i+1)*nao_prim])
    K = np.array(Res)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    #if mydf.verbose:
    _benchmark_time(t1, t2, "_contract_k_dm_quadratic_direct", mydf)
    
    # return K * ngrid / vol
    return K
   
def get_jk_dm_translation_symmetry(mydf, dm, hermi=1, kpt=np.zeros(3),
                                    kpts_band=None, with_j=True, with_k=True, omega=None, 
                                   **kwargs):
    
    '''JK for given k-point'''
    
    direct = mydf.direct
    use_mpi = mydf.use_mpi
    
    if use_mpi :
        raise NotImplementedError("ISDF does not support use_mpi")
    
    if len(dm.shape) == 3:
        assert dm.shape[0] <= 2
        #dm = dm[0]
    else:
        assert dm.ndim == 2
        dm = dm.reshape(1, dm.shape[0], dm.shape[1])

    if hasattr(mydf, 'kmesh') and mydf.kmesh is not None:
        from pyscf.pbc.df.isdf.isdf_tools_densitymatrix import symmetrize_dm
        dm = symmetrize_dm(dm, mydf.kmesh)
    else:
        if hasattr(mydf, 'kmesh') and mydf.kmesh is not None:
            from pyscf.pbc.df.isdf.isdf_tools_densitymatrix import symmetrize_dm
            dm = symmetrize_dm(dm, mydf.kmesh)

    if use_mpi:
        dm = bcast(dm, root=0)

    nset = dm.shape[0]

    #### perform the calculation ####

    if "exxdiv" in kwargs:
        exxdiv = kwargs["exxdiv"]
    else:
        exxdiv = None

    #vj = vk = None
    vj = np.zeros_like(dm)
    vk = np.zeros_like(dm)

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

    for iset in range(nset):
        if with_j:
            vj[iset] = _contract_j_dm_k_ls(mydf, dm[iset], use_mpi)  
            sys.stdout.flush()
        if with_k:
            if mydf.direct:
                raise NotImplementedError
            else:
                if mydf.with_robust_fitting:
                    vk[iset] = _get_k_kSym_robust_fitting_fast(mydf, dm[iset])
                else:
                    vk[iset] = _get_k_kSym(mydf, dm[iset])
            if exxdiv == 'ewald':
                print("WARNING: ISDF does not support ewald")

    if exxdiv == 'ewald':
        if np.allclose(kpt, np.zeros(3)):
            # from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0, _format_dms, _format_kpts_band, _format_jks
            kpts = kpt.reshape(1,3)
            kpts = np.asarray(kpts)
            #dm_kpts = dm.reshape(-1, dm.shape[0], dm.shape[1]).copy()
            dm_kpts = dm.copy()
            dm_kpts = lib.asarray(dm_kpts, order='C')
            dms     = _format_dms(dm_kpts, kpts)
            nset, nkpts, nao = dms.shape[:3]
            assert nset <= 2
            kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
            nband = len(kpts_band)
            assert nband == 1
            if is_zero(kpts_band) and is_zero(kpts):
                vk = vk.reshape(nset,nband,nao,nao)
            else:
                raise NotImplementedError("ISDF does not support kpts_band != 0")
            _ewald_exxdiv_for_G0(mydf.cell, kpts, dms, vk, kpts_band=kpts_band)
            #vk = vk[0,0]
            vk = vk.reshape(nset,nao,nao)
        else:
            logger.warn(mydf, 'get_jk_dm_k_quadratic: Exxdiv for k-point is not supported')

    t1 = log.timer('sr jk', *t1)

    return vj, vk