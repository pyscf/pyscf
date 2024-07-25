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
import scipy
import ctypes, sys

############ pyscf module ############

from pyscf import lib
from pyscf.pbc.gto import Cell
from pyscf.pbc import tools
from pyscf.gto.mole import *
libpbc = lib.load_library('libpbc')

############ isdf utils ############

from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time
import pyscf.pbc.df.isdf.isdf_fast as ISDF
from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto
import pyscf.pbc.df.isdf.isdf_tools_local as ISDF_Local_Utils
import pyscf.pbc.df.isdf.isdf_linear_scaling_jk as ISDF_LinearScalingJK

##### all the involved algorithm in ISDF based on aoR_Holder ##### 

USE_SCIPY_QR = False  ## true for single-thread mode to compare with Kori's code

############ subroutines --- select IP ############

############ ls refers to linear scaling ############

def select_IP_atm_ls(mydf, c:int, m:int, first_natm=None, 
                     rela_cutoff = 0.0, 
                     no_retriction_on_nIP = False,
                     use_mpi=False):

    if use_mpi:
        from pyscf.pbc.df.isdf.isdf_tools_mpi import rank, comm, comm_size
    else:
        rank = 0
        comm = None
        comm_size = 1

    assert isinstance(mydf.aoR, list)
    assert isinstance(mydf.partition, list)
    
    ### determine the largest grids point of one atm ###

    log = lib.logger.Logger(mydf.cell.stdout, mydf.cell.verbose)

    natm      = mydf.cell.natm
    nao       = mydf.nao
    naux_max  = 0

    nao_per_atm = np.zeros((natm), dtype=np.int32)
    for i in range(mydf.nao):
        atm_id = mydf.ao2atomID[i]
        nao_per_atm[atm_id] += 1

    for nao_atm in nao_per_atm:
        naux_max = max(naux_max, int(np.sqrt(c*nao_atm)) + m)

    nthread = lib.num_threads()

    ### loop over atm ###
    
    coords  = mydf.coords
    assert coords is not None

    results = []

    fn_colpivot_qr = getattr(libpbc, "ColPivotQRRelaCut", None)
    assert(fn_colpivot_qr is not None)
    fn_ik_jk_ijk = getattr(libpbc, "NP_d_ik_jk_ijk", None)
    assert(fn_ik_jk_ijk is not None)

    weight = np.sqrt(mydf.cell.vol / coords.shape[0])

    if first_natm is None:
        first_natm = natm
    
    group_begin, group_end = ISDF_Local_Utils._range_partition(first_natm, rank, comm_size, use_mpi)
    
    for i in range(first_natm):
        results.append(None)
    
    aoR_atm1 = None
    aoR_atm2 = None
    aoPairBuffer = None
    R = None
    thread_buffer = None
    global_buffer = None
    
    log.debug4("-------------------------------------------")
    
    for atm_id in range(group_begin, group_end):
        
        aoR = mydf.aoR[atm_id]
        if aoR is None:  # it is used to split the task when using MPI
            continue

        grid_ID = mydf.partition[atm_id] 
        aoR_atm = mydf.aoR[atm_id].aoR
        nao_tmp = aoR_atm.shape[0]
        
        # create buffer for this atm

        dtypesize = aoR_atm.dtype.itemsize
        nao_atm   = nao_per_atm[atm_id]
        naux_now  = int(np.sqrt(c*nao_atm)) + m
        naux2_now = naux_now * naux_now

        R = np.ndarray((naux2_now, grid_ID.shape[0]), dtype=np.float64)

        aoR_atm1 = np.ndarray((naux_now, grid_ID.shape[0]), dtype=np.float64)
        aoR_atm2 = np.ndarray((naux_now, grid_ID.shape[0]), dtype=np.float64)

        aoPairBuffer = np.ndarray(
            (naux_now*naux_now, grid_ID.shape[0]), dtype=np.float64)

        G1 = np.random.rand(nao_tmp, naux_now)
        G1, _ = numpy.linalg.qr(G1)
        G1    = G1.T
        G2 = np.random.rand(nao_tmp, naux_now)
        G2, _ = numpy.linalg.qr(G2)
        G2    = G2.T

        lib.dot(G1, aoR_atm, c=aoR_atm1)
        lib.dot(G2, aoR_atm, c=aoR_atm2)

        fn_ik_jk_ijk(aoR_atm1.ctypes.data_as(ctypes.c_void_p),
                     aoR_atm2.ctypes.data_as(ctypes.c_void_p),
                     aoPairBuffer.ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(naux_now),
                     ctypes.c_int(naux_now),
                     ctypes.c_int(grid_ID.shape[0]))
        if no_retriction_on_nIP:
            max_rank = min(naux2_now, grid_ID.shape[0])
            log.debug4("In select_IP_atm_ls, no_retriction_on_nIP")
        else:
            max_rank  = min(naux2_now, grid_ID.shape[0], nao_atm * c + m)
            log.debug4("In select_IP_atm_ls, retriction_on_nIP")
        npt_find      = ctypes.c_int(0)
        pivot         = np.arange(grid_ID.shape[0], dtype=np.int32)
        thread_buffer = np.ndarray((nthread+1, grid_ID.shape[0]+1), dtype=np.float64)
        global_buffer = np.ndarray((1, grid_ID.shape[0]), dtype=np.float64)
        
        log.debug4("In select_IP_atm_ls, max_rank      = %d" % (max_rank))
        log.debug4("In select_IP_atm_ls, naux2_now     = %d" % (naux2_now))
        log.debug4("In select_IP_atm_ls, grid_ID.shape = %s" % (grid_ID.shape))
        log.debug4("In select_IP_atm_ls, rela_cutoff   = %e" % (rela_cutoff))
        
        if USE_SCIPY_QR:
            R, pivot = scipy.linalg.qr(aoPairBuffer, pivoting=True, mode='r', check_finite=False, overwrite_a=True)
            npt_find = nao_atm * c + m 
        else:
            fn_colpivot_qr(aoPairBuffer.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(naux2_now),
                            ctypes.c_int(grid_ID.shape[0]),
                            ctypes.c_int(max_rank),
                            ctypes.c_double(1e-14),
                            ctypes.c_double(rela_cutoff),
                            pivot.ctypes.data_as(ctypes.c_void_p),
                            R.ctypes.data_as(ctypes.c_void_p),
                            ctypes.byref(npt_find),
                            thread_buffer.ctypes.data_as(ctypes.c_void_p),
                            global_buffer.ctypes.data_as(ctypes.c_void_p))
            npt_find = npt_find.value
            cutoff = abs(R[npt_find-1, npt_find-1])
            log.debug4("ngrid = %d, npt_find = %d, cutoff = %12.6e" % (grid_ID.shape[0], npt_find, cutoff))
                        
        pivot = pivot[:npt_find]
        pivot.sort()
        
        atm_IP = grid_ID[pivot]
        atm_IP = np.array(atm_IP, dtype=np.int32)
        atm_IP.sort()
        results[atm_id] = atm_IP
        
        log.debug4("In select_IP_atm_ls, npt_find      = %d" %(npt_find))
        log.debug4("-------------------------------------------")

    del aoR_atm1
    del aoR_atm2
    del aoPairBuffer
    del R
    del thread_buffer
    del global_buffer

    if use_mpi:
        results = ISDF_Local_Utils._sync_list(results, first_natm)

    assert len(results) == first_natm

    return results

def select_IP_group_ls(mydf, aoRg_possible, c:int, m:int, group=None, atm_2_IP_possible = None):
        
    assert isinstance(aoRg_possible, list)
    assert isinstance(group, list) or isinstance(group, np.ndarray)
    assert isinstance(atm_2_IP_possible, list)
    
    assert len(aoRg_possible) == len(atm_2_IP_possible)
    # assert len(aoRg_possible) == mydf.natm
    
    log = lib.logger.Logger(mydf.cell.stdout, mydf.cell.verbose)
    
    if group is None:
        raise ValueError("group must be specified")

    #if mydf.verbose:
    #     print("In select_IP, num_threads = ", lib.num_threads())
        
    nthread = lib.num_threads()
    
    coords = mydf.coords
        
    fn_colpivot_qr = getattr(libpbc, "ColPivotQRRelaCut", None)
    assert(fn_colpivot_qr is not None)
    fn_ik_jk_ijk = getattr(libpbc, "NP_d_ik_jk_ijk", None)
    assert(fn_ik_jk_ijk is not None)

    weight = np.sqrt(mydf.cell.vol / coords.shape[0])
    
    #### perform QRCP ####

    nao_group = 0
    for atm_id in group:
        shl_begin = mydf.shl_atm[atm_id][0]
        shl_end   = mydf.shl_atm[atm_id][1]
        nao_atm = mydf.aoloc_atm[shl_end] - mydf.aoloc_atm[shl_begin]
        nao_group += nao_atm
    
    ##### random projection #####

    nao = mydf.nao
    
    # aoR_atm = ISDF_eval_gto(mydf.cell, coords=coords[IP_possible]) * weight
    
    aoRg_unpacked = []
    for atm_id in group:
        aoRg_unpacked.append(aoRg_possible[atm_id])
    if len(aoRg_unpacked) == 1:
        aoRg_packed = aoRg_unpacked[0].aoR
    else:
        aoRg_packed = ISDF_Local_Utils._pack_aoR_holder(aoRg_unpacked, nao).aoR
    
    nao = aoRg_packed.shape[0]

    log.debug4("In select_IP_group_ls, nao_group   = %d" % (nao_group))
    log.debug4("In select_IP_group_ls, nao         = %d" % (nao))    
    log.debug4("In select_IP_group_ls, c           = %d, m = %d" % (c, m))
    log.debug4("In select_IP_group_ls, rela_cutoff = %e" % (mydf.rela_cutoff_QRCP))
    
    # naux_now = int(np.sqrt(c*nao)) + m # seems to be too large
    naux_now = int(np.sqrt(c*nao_group)) + m
    G1 = np.random.rand(nao, naux_now)
    G1, _ = numpy.linalg.qr(G1)
    G1 = G1.T
    
    G2 = np.random.rand(nao, naux_now)
    G2, _ = numpy.linalg.qr(G2)
    G2    = G2.T 
    # naux_now = nao
        
    aoR_atm1 = lib.ddot(G1, aoRg_packed)
    naux_now1 = aoR_atm1.shape[0]
    aoR_atm2 = lib.ddot(G2, aoRg_packed)
    naux_now2 = aoR_atm2.shape[0]
    
    naux2_now = naux_now1 * naux_now2
    
    R = np.ndarray((naux2_now, aoRg_packed.shape[1]), dtype=np.float64)

    aoPairBuffer = np.ndarray((naux2_now, aoRg_packed.shape[1]), dtype=np.float64)

    fn_ik_jk_ijk(aoR_atm1.ctypes.data_as(ctypes.c_void_p),
                 aoR_atm2.ctypes.data_as(ctypes.c_void_p),
                 aoPairBuffer.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux_now1),
                 ctypes.c_int(naux_now2),
                 ctypes.c_int(aoRg_packed.shape[1]))

    aoR_atm1 = None
    aoR_atm2 = None
    del aoR_atm1
    del aoR_atm2

    IP_possible = []
    for atm_id in group:
        if atm_2_IP_possible[atm_id] is None:
            continue
        IP_possible.extend(atm_2_IP_possible[atm_id])
    IP_possible = np.array(IP_possible, dtype=np.int32)

    if mydf.no_restriction_on_nIP:
        max_rank = min(naux2_now, IP_possible.shape[0])
        log.debug4("In select_IP_group_ls, no_restriction_on_nIP")
    else:
        max_rank  = min(naux2_now, IP_possible.shape[0], nao_group * c)  
        log.debug4("In select_IP_group_ls, restriction_on_nIP")
        
    log.debug4("In select_IP_group_ls, naux2_now         = %d, max_rank = %d" % (naux2_now, max_rank))
    log.debug4("In select_IP_group_ls, IP_possible.shape = %s" % (IP_possible.shape))
    log.debug4("In select_IP_group_ls, nao_group         = %d" % (nao_group))
    log.debug4("In select_IP_group_ls, c = %d" % (c))
    log.debug4("In select_IP_group_ls, nao_group * c = %d" % (nao_group * c))
    
    npt_find = ctypes.c_int(0)
    pivot    = np.arange(IP_possible.shape[0], dtype=np.int32)

    thread_buffer = np.ndarray((nthread+1, IP_possible.shape[0]+1), dtype=np.float64)
    global_buffer = np.ndarray((1, IP_possible.shape[0]), dtype=np.float64)


    if not USE_SCIPY_QR:
        fn_colpivot_qr(aoPairBuffer.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_int(naux2_now),
                       ctypes.c_int(IP_possible.shape[0]),
                       ctypes.c_int(max_rank),
                       ctypes.c_double(1e-14),
                       ctypes.c_double(mydf.rela_cutoff_QRCP),
                       pivot.ctypes.data_as(ctypes.c_void_p),
                       R.ctypes.data_as(ctypes.c_void_p),
                       ctypes.byref(npt_find),
                       thread_buffer.ctypes.data_as(ctypes.c_void_p),
                       global_buffer.ctypes.data_as(ctypes.c_void_p))
        npt_find = npt_find.value
        cutoff   = abs(R[npt_find-1, npt_find-1])
        log.debug4("ngrid = %d, npt_find = %d, cutoff = %12.6e" % (IP_possible.shape[0], npt_find, cutoff))
    else:
        # pivot, rankc = scipy.linalg.lapack.dpstrf(aoPairBuffer)[1:3]
        # pivot = pivot[:rankc]-1
        # npt_find = nao_group * c 
        R, pivot = scipy.linalg.qr(aoPairBuffer, pivoting=True, mode='r', check_finite=False, overwrite_a=True)
        npt_find = nao_group * c 
    
    log.debug4("In select_IP_group_ls, npt_find = %d" % (npt_find))
    
    pivot = pivot[:npt_find]
    pivot.sort()
    results = list(IP_possible[pivot])
    results = np.array(results, dtype=np.int32)
    
    ### clean up ###
    
    del aoPairBuffer
    del R
    del thread_buffer
    del global_buffer
    del G1
    del G2
    del aoRg_packed
    del IP_possible
    aoRg_packed = None
    IP_possible = None
    aoPairBuffer = None
    R = None
    pivot = None
    thread_buffer = None
    global_buffer = None
    
    return results

def select_IP_local_ls_drive(mydf, c, m, IP_possible_atm, group, use_mpi=False):
    
    if use_mpi:
        from pyscf.pbc.df.isdf.isdf_tools_mpi import rank, comm, comm_size
    else:
        rank = 0
        comm = None
        comm_size = 1   
    
    IP_group  = []
    
    aoRg_possible = mydf.aoRg_possible

    ######### allocate buffer #########

    natm = mydf.natm
    
    for i in range(len(group)):
        IP_group.append(None)

    if len(group) < natm:
        
        if use_mpi == False:
            for i in range(len(group)):
                IP_group[i] = select_IP_group_ls(mydf, aoRg_possible, c, m, group=group[i], atm_2_IP_possible=IP_possible_atm)
        else:
            group_begin, group_end = ISDF_Local_Utils._range_partition(len(group), rank, comm_size, use_mpi)
            for i in range(group_begin, group_end):
                IP_group[i] = select_IP_group_ls(mydf, aoRg_possible, c, m, group=group[i], atm_2_IP_possible=IP_possible_atm)
            # allgather(IP_group)
            
            IP_group = ISDF_Local_Utils._sync_list(IP_group, len(group))

    else:
        IP_group = IP_possible_atm 

    mydf.IP_group = IP_group
    
    mydf.IP_flat = []
    mydf.IP_segment = [0]
    nIP_now = 0
    for x in IP_group:
        mydf.IP_flat.extend(x)
        nIP_now += len(x)
        mydf.IP_segment.append(nIP_now)
    mydf.IP_flat = np.array(mydf.IP_flat, dtype=np.int32)
    mydf.naux = mydf.IP_flat.shape[0]
    
    gridID_2_atmID = mydf.gridID_2_atmID
    
    partition_IP = []
    for i in range(natm):
        partition_IP.append([])
    
    for _ip_id_ in mydf.IP_flat:
        atm_id = gridID_2_atmID[_ip_id_]
        partition_IP[atm_id].append(_ip_id_)
    
    for i in range(natm):
        partition_IP[i] = np.array(partition_IP[i], dtype=np.int32)
        partition_IP[i].sort()  
    
    mydf.partition_IP = partition_IP

    ### build ### 
    
    if len(group) < natm:
        
        coords = mydf.coords
        weight = np.sqrt(mydf.cell.vol / mydf.coords.shape[0])
    
        del mydf.aoRg_possible
        mydf.aoRg_possible = None
    
        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        mydf.aoRg = mydf._construct_build_aoRg(partition_IP, group)
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, "build_aoRg", mydf)
    
    else:
        if use_mpi:
            mydf.aoRg = mydf.aoRg_possible
        else:
            mydf.aoRg = mydf.aoRg_possible
    
    if rank == 0:
        memory = ISDF_Local_Utils._get_aoR_holders_memory(mydf.aoRg)
        log    = lib.logger.Logger(mydf.cell.stdout, mydf.cell.verbose)
        log.info("memory to store aoRg is %d " %(memory))
        
    return IP_group

############ subroutines --- build aux bas ############

def find_common_elements_positions(arr1, arr2):
    position1 = []
    position2 = []
    i, j = 0, 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            i += 1
        elif arr1[i] > arr2[j]:
            j += 1
        else:
            # positions.append(((i, arr1[i]), (j, arr2[j])))
            position1.append(i)
            position2.append(j)
            i += 1
            j += 1
    return np.array(position1, dtype=np.int32), np.array(position2, dtype=np.int32)

def build_aux_basis_ls(mydf, group, IP_group, debug=True, use_mpi=False):


    log = lib.logger.Logger(mydf.cell.stdout, mydf.cell.verbose)

    if use_mpi:
        from pyscf.pbc.df.isdf.isdf_tools_mpi import rank, comm, comm_size
    else:
        rank = 0
        comm = None
        comm_size = 1
    
    ###### split task ######
    
    ngroup = len(group)
    nthread = lib.num_threads()
    assert len(IP_group) == ngroup
        
    group_begin, group_end = ISDF_Local_Utils._range_partition(ngroup, rank, comm_size, use_mpi)
    
    ngroup_local = group_end - group_begin
    
    if ngroup_local == 0:
        log.warn(" WARNING : rank = %d, ngroup_local = 0" % rank)
    
    mydf.group_begin = group_begin
    mydf.group_end = group_end
        
    ###### build grid_ID_local ###### 
    
    coords = mydf.coords

    ###### build aux basis ######
    
    mydf.aux_basis = []
    
    for i in range(ngroup):
        mydf.aux_basis.append(None)
        
    fn_cholesky = getattr(libpbc, "Cholesky", None)
    assert (fn_cholesky is not None)
    
    fn_build_aux = getattr(libpbc, "Solve_LLTEqualB_Parallel", None)
    assert(fn_build_aux is not None)
    
    for i in range(group_begin, group_end):
        
        aoRg_unpacked = []
        aoR_unpacked = []
        
        for atm_id in group[i]:
            aoRg_unpacked.append(mydf.aoRg[atm_id])
            aoR_unpacked.append(mydf.aoR[atm_id])
        
        aoRg1 = ISDF_Local_Utils._pack_aoR_holder(aoRg_unpacked, mydf.nao)
        aoR1 = ISDF_Local_Utils._pack_aoR_holder(aoR_unpacked, mydf.nao)
        
        if aoRg1.aoR.shape[0] == aoR1.aoR.shape[0]:
            aoRg1 = aoRg1.aoR
            aoR1 = aoR1.aoR
        else:
            pos1, pos2 = find_common_elements_positions(aoRg1.ao_involved, aoR1.ao_involved)
            assert len(pos1) == aoRg1.aoR.shape[0]
            aoRg1 = aoRg1.aoR
            aoR1 = aoR1.aoR[pos2,:]
        
            
        A = lib.ddot(aoRg1.T, aoRg1)
        lib.square_inPlace(A)
        grid_ID = mydf.partition_group_to_gridID[i]
        B = lib.ddot(aoRg1.T, aoR1)
        lib.square_inPlace(B)
                    
        fn_cholesky = getattr(libpbc, "Cholesky", None)
        assert(fn_cholesky is not None)
        fn_cholesky(
            A.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(A.shape[0]),
        ) 
        nThread = lib.num_threads()
        bunchsize = B.shape[1]//nThread
        fn_build_aux(
            ctypes.c_int(B.shape[0]),
            A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(B.shape[1]),
            ctypes.c_int(bunchsize)
        )
        
        mydf.aux_basis[i] = B.copy()

    ### sync aux_basis ###
    
    if use_mpi:
        mydf.aux_basis = ISDF_Local_Utils._sync_list(mydf.aux_basis, ngroup)
    
    del A 
    A = None
    del B
    B = None
    del aoRg1
    aoRg1 = None
    del aoR1
    aoR1 = None
  
def build_auxiliary_Coulomb_local_bas_wo_robust_fitting(mydf, debug=True, use_mpi=False):
    
    if use_mpi:
        raise NotImplementedError("use_mpi = True is not supported")
        #### NOTE: one should bcast aux_basis first! ####

    
    t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    cell = mydf.cell
    mesh = cell.mesh
    mesh_int32         = np.array(mesh, dtype=np.int32)
    
    naux = mydf.naux
    
    ncomplex = mesh[0] * mesh[1] * (mesh[2] // 2 + 1) * 2 
    
    group_begin = mydf.group_begin
    group_end = mydf.group_end
    ngroup = len(mydf.group)
    
    grid_ordering = mydf.grid_ID_ordered 
    
    if mydf.omega is not None:
        assert mydf.omega >= 0.0
    
    coulG = mydf.coulG.copy()
    coulG_real         = coulG.reshape(*mesh)[:, :, :mesh[2]//2+1].reshape(-1).copy()
    
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
        
    ####### allocate buf for V ########
    
    nThread = lib.num_threads()
    bufsize_per_thread = (coulG_real.shape[0] * 2 + mesh[0] * mesh[1] * mesh[2])
    buf = np.zeros((nThread, bufsize_per_thread), dtype=np.double)
    
    assert len(mydf.aux_basis) == ngroup
    
    naux_local = 0
    max_naux_bunch = 0
    for i in range(group_begin, group_end):
        naux_local += mydf.aux_basis[i].shape[0]    
        max_naux_bunch = max(max_naux_bunch, mydf.aux_basis[i].shape[0])
    
    if hasattr(mydf, "grid_pnt_near_atm"):
        max_naux_bunch = max(max_naux_bunch, len(mydf.grid_pnt_near_atm))
        if use_mpi == False or (use_mpi and rank == comm_size - 1):
            naux_local += len(mydf.grid_pnt_near_atm)
    
    V = np.zeros((max_naux_bunch, np.prod(mesh_int32)), dtype=np.double)
    
    naux = mydf.naux
    
    W = np.zeros((naux_local, naux), dtype=np.double)
    
    aux_row_loc = 0
    
    if hasattr(mydf, "grid_pnt_near_atm"):
        grid_ID_near_atm = mydf.grid_pnt_near_atm
    else:
        grid_ID_near_atm = []
        grid_ID_near_atm = np.array(grid_ID_near_atm, dtype=np.int32)
    for i in range(group_begin, group_end):
        
        aux_basis_now = mydf.aux_basis[i]
        naux_bra = aux_basis_now.shape[0]
        grid_ID = mydf.partition_group_to_gridID[i]
        
        construct_V(aux_basis_now, buf, V, grid_ID, grid_ordering)
        
        grid_shift = 0
        aux_col_loc = 0
        for j in range(0, ngroup):
            grid_ID_now = mydf.partition_group_to_gridID[j]
            aux_bas_ket = mydf.aux_basis[j]
            naux_ket = aux_bas_ket.shape[0]
            ngrid_now = grid_ID_now.size
            W[aux_row_loc:aux_row_loc+naux_bra, aux_col_loc:aux_col_loc+naux_ket] = lib.ddot(V[:naux_bra, grid_shift:grid_shift+ngrid_now], aux_bas_ket.T)
            grid_shift += ngrid_now
            aux_col_loc += naux_ket
        print("aux_row_loc = %d, aux_col_loc = %d" % (aux_row_loc, aux_col_loc))
        print("V.shape = ", V[:naux_bra,:].shape)
        W[aux_row_loc:aux_row_loc+naux_bra, aux_col_loc:] = V[:naux_bra, grid_shift:]
        aux_row_loc += aux_basis_now.shape[0]
    
    if (use_mpi == False or (use_mpi and rank == comm_size - 1)) and len(grid_ID_near_atm) != 0:
        ### construct the final row ### 
        grid_ID = grid_ID_near_atm
        aux_basis_now = np.identity(len(grid_ID), dtype=np.double)
        construct_V(aux_basis_now, buf, V, grid_ID, grid_ordering)
        grid_shift = 0
        aux_col_loc = 0
        naux_bra = len(grid_ID)
        for j in range(0, ngroup):
            grid_ID_now = mydf.partition_group_to_gridID[j]
            aux_bas_ket = mydf.aux_basis[j]
            naux_ket = aux_bas_ket.shape[0]
            ngrid_now = grid_ID_now.size
            W[aux_row_loc:aux_row_loc+naux_bra, aux_col_loc:aux_col_loc+naux_ket] = lib.ddot(V[:naux_bra, grid_shift:grid_shift+ngrid_now], aux_bas_ket.T)
            grid_shift += ngrid_now
            aux_col_loc += naux_ket
        assert aux_row_loc == aux_col_loc
        W[aux_row_loc:, aux_col_loc:] = V[:naux_bra, grid_shift:]
    
    del buf
    buf = None
    del V
    V = None
    
    mydf.W = W
    
    if use_mpi:
        comm.Barrier()
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    if mydf.verbose > 0:
        _benchmark_time(t0, t1, 'build_auxiliary_Coulomb', mydf)

def build_auxiliary_Coulomb_local_bas(mydf, debug=True, use_mpi=False):
    
    if hasattr(mydf, "grid_pnt_near_atm") and len(mydf.grid_pnt_near_atm) != 0 :
        raise NotImplementedError("grid_pnt_near_atm is not supported")
    
    log = lib.logger.Logger(mydf.cell.stdout, mydf.cell.verbose)
    
    t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    cell = mydf.cell
    mesh = cell.mesh
        
    naux = mydf.naux
    
    ncomplex = mesh[0] * mesh[1] * (mesh[2] // 2 + 1) * 2 
    
    group_begin = mydf.group_begin
    group_end = mydf.group_end
    
    grid_ordering = mydf.grid_ID_ordered
    
    def construct_V_CCode(aux_basis:list[np.ndarray], mesh, coul_G):
        
        coulG_real         = coul_G.reshape(*mesh)[:, :, :mesh[2]//2+1].reshape(-1).copy()
        nThread            = lib.num_threads()
        bufsize_per_thread = int((coulG_real.shape[0] * 2 + mesh[0] * mesh[1] * mesh[2]) * 1.1)
        bufsize_per_thread = (bufsize_per_thread + 15) // 16 * 16
        
        buf = np.zeros((nThread, bufsize_per_thread), dtype=np.double)
        
        # nAux               = aux_basis.shape[0]
        
        nAux = 0
        for x in aux_basis:
            nAux += x.shape[0]
        
        ngrids             = mesh[0] * mesh[1] * mesh[2]
        mesh_int32         = np.array(mesh, dtype=np.int32)
        V                  = np.zeros((nAux, ngrids), dtype=np.double)
        
        fn = getattr(libpbc, "_construct_V_local_bas", None)
        assert(fn is not None)
        
        shift_row = 0
        ngrid_now = 0
        for i in range(len(aux_basis)):
            
            aux_basis_now = aux_basis[i]
            grid_ID = mydf.partition_group_to_gridID[group_begin+i]
            assert aux_basis_now.shape[1] == grid_ID.size 
            ngrid_now += grid_ID.size
        
            fn(mesh_int32.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(aux_basis_now.shape[0]),
                ctypes.c_int(aux_basis_now.shape[1]),
                grid_ID.ctypes.data_as(ctypes.c_void_p),
                aux_basis_now.ctypes.data_as(ctypes.c_void_p),
                coulG_real.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(shift_row),
                V.ctypes.data_as(ctypes.c_void_p),
                grid_ordering.ctypes.data_as(ctypes.c_void_p),
                buf.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(bufsize_per_thread))
        
            shift_row += aux_basis_now.shape[0]

        del buf
        buf = None

        return V
    
    ########### construct V ###########

    if mydf.omega is not None:
        assert mydf.omega >= 0.0
    coulG = mydf.coulG.copy()
    V = construct_V_CCode(mydf.aux_basis, mesh, coulG)

    if use_mpi:

        ############# the only communication #############
    
        grid_segment = mydf.grid_segment 
        assert len(grid_segment) == comm_size + 1
    
        t0_comm = (lib.logger.process_clock(), lib.logger.perf_counter())
    
        sendbuf = []
        for i in range(comm_size):
            p0 = grid_segment[i]
            p1 = grid_segment[i+1]
            sendbuf.append(V[:, p0:p1])
        del V
        V = None
        V_fullrow = np.vstack(alltoall(sendbuf, split_recvbuf=True))
        del sendbuf
        sendbuf = None
    
        mydf.V_R = V_fullrow
    
        t1_comm = (lib.logger.process_clock(), lib.logger.perf_counter()) 
        t_comm = t1_comm[1] - t0_comm[1]
    
        if mydf.verbose > 0:
            log.info("rank = %d, t_comm = %12.6e" % (rank, t_comm))
    else:
        t_comm = 0.0
        mydf.V_R = V

    ########### construct W ###########
    
    aux_group_shift = [0]
    naux_now = 0
    for i in range(len(mydf.IP_group)):
        IP_group_now = mydf.IP_group[i]
        naux_now += len(IP_group_now)
        aux_group_shift.append(naux_now)
    
    mydf.W = np.zeros((mydf.naux, mydf.naux), dtype=np.float64) 
    
    grid_shift = 0
    for i in range(group_begin, group_end):
        aux_begin = aux_group_shift[i]
        aux_end   = aux_group_shift[i+1]
        ngrid_now = mydf.partition_group_to_gridID[i].size
        sys.stdout.flush()
        mydf.W[:, aux_begin:aux_end] = lib.ddot(mydf.V_R[:, grid_shift:grid_shift+ngrid_now], mydf.aux_basis[i-group_begin].T)
        grid_shift += ngrid_now
    
    if use_mpi:
        comm.Barrier()
        
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    if mydf.verbose > 0:
        _benchmark_time(t0, t1, 'build_auxiliary_Coulomb', mydf)
    
    sys.stdout.flush()

    
class PBC_ISDF_Info_Quad(ISDF.PBC_ISDF_Info):
    
    ''' Interpolative separable density fitting (ISDF) for periodic systems.
    The locality is explored! 
    
    Fitting aux basis is linear scaling!
    
    Quad stands for quadratic scaling for constructing V and W matrix as well as build K matrix!
    
    Examples:

    >>> pbc_isdf = PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, aoR_cutoff=1e-8, direct=False, use_occ_RI_K=False)
    >>> pbc_isdf.build_IP_local(c=C, m=5)
    >>> pbc_isdf.build_auxiliary_Coulomb()
    >>> from pyscf.pbc import scf
    >>> mf = scf.RHF(cell)
    >>> pbc_isdf.direct_scf = mf.direct_scf
    >>> mf.with_df = pbc_isdf
    >>> mf.verbose = 0
    >>> mf.kernel()
    
    '''
    
    # group_partition refer to the group of atoms to perform local fitting
    # if not set then each atom is treated as a group 
    
    def __init__(self, mol:Cell, 
                 with_robust_fitting=True,
                 kmesh              =None,
                 verbose            =None,
                 rela_cutoff_QRCP   =None,
                 aoR_cutoff         =1e-8,
                 direct             =False,
                 use_occ_RI_K       =True,
                 limited_memory     =False,
                 build_K_bunchsize  =None,
                 ):
        
        if verbose is None:
            verbose = mol.verbose
        
        super().__init__(
            mol=mol,
            aoR=None,
            with_robust_fitting=with_robust_fitting,
            kmesh=kmesh,
            get_partition=False,
            verbose=verbose
        )
        
        self.cell = mol.copy()
        cell = self.cell

        #### get other info #### 
        
        shl_atm = []
        
        for i in range(self.natm):
            shl_atm.append([None, None])
        
        for i in range(cell.nbas):
            atm_id = cell.bas_atom(i)
            if shl_atm[atm_id][0] is None:
                shl_atm[atm_id][0] = i
            shl_atm[atm_id][1] = i+1
        
        self.shl_atm = shl_atm
        self.aoloc_atm = cell.ao_loc_nr() 
        
        self.use_mpi = False

        self.aoR_cutoff = aoR_cutoff
        
        if rela_cutoff_QRCP is None:
            self.no_restriction_on_nIP = False
            self.rela_cutoff_QRCP = 0.0
        else:
            self.no_restriction_on_nIP = True
            self.rela_cutoff_QRCP = rela_cutoff_QRCP
    
        self.aoR = None
        self.partition = None

        self.V_W_cutoff = None

        self.direct = direct # whether to use direct method to calculate J and K, if True, the memory usage will be reduced, V W will not be stored
        if self.direct:
            self.with_robust_fitting = True

        self.with_translation_symmetry = False
        self.kmesh = None
        
        ######### default setting for range separation #########
        
        # WARNING: not a good design pattern to write this code here! 
        
        self.omega           = None
        self.use_aft_ao      = False
        self.ke_cutoff_pp    = self.cell.ke_cutoff
        self.ke_cutoff_ft_ao = self.cell.ke_cutoff
        self.ft_ao_mesh      = self.mesh.copy()
        self.rsjk            = None
        self.cell_rsjk       = None
            
        ########## coul kernel ##########
        
        self.get_coulG()
        self.ovlp = self.cell.pbc_intor('int1e_ovlp')
        self.occ_tol = 1e-9
        self.occ_RI_K  = use_occ_RI_K

        ########## limited memory ##########
        
        self._limited_memory    = limited_memory
        self._build_K_bunchsize = build_K_bunchsize
        if build_K_bunchsize is None:
            if limited_memory:
                from pyscf.pbc.df.isdf._isdf_local_K_direct import K_DIRECT_NAUX_BUNCHSIZE
                self._build_K_bunchsize = K_DIRECT_NAUX_BUNCHSIZE
            else:
                self._build_K_bunchsize = 10000 * 10000 # infinite in practice

    @property
    def first_natm(self):
        if self.kmesh is not None:
            return self.cell.natm // np.prod(self.kmesh)
        else:
            return self.cell.natm

    def build_partition_aoR(self, Ls):
        
        if self.aoR is not None and self.partition is not None:
            return
            
        ##### build cutoff info #####   
        
        self.distance_matrix = ISDF_Local_Utils.get_cell_distance_matrix(self.cell)
        weight = np.sqrt(self.cell.vol / self.coords.shape[0])
        precision = self.aoR_cutoff
        rcut = ISDF_Local_Utils._estimate_rcut(self.cell, self.coords.shape[0], precision)
        rcut_max = np.max(rcut)
        atm2_bas = ISDF_Local_Utils._atm_to_bas(self.cell)
        self.AtmConnectionInfo = []
        
        for i in range(self.cell.natm):
            tmp = ISDF_Local_Utils.AtmConnectionInfo(self.cell, i, self.distance_matrix, precision, rcut, rcut_max, atm2_bas)
            self.AtmConnectionInfo.append(tmp)
        
        ##### build partition #####
        
        if Ls is None:
            lattice_x = self.cell.lattice_vectors()[0][0]
            lattice_y = self.cell.lattice_vectors()[1][1]
            lattice_z = self.cell.lattice_vectors()[2][2]
            
            Ls = [int(lattice_x)/3+6, int(lattice_y)/3+6, int(lattice_z)/3+6]

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if self.rsjk is not None and self.cell_rsjk is not None:
            self.partition = ISDF_Local_Utils.get_partition(self.cell_rsjk, self.coords, self.AtmConnectionInfo, 
                                                                  Ls, 
                                                                  self.with_translation_symmetry,
                                                                  self.kmesh,
                                                                  self.use_mpi)
        else:
            self.partition = ISDF_Local_Utils.get_partition(self.cell, self.coords, self.AtmConnectionInfo, 
                                                                  Ls, 
                                                                  self.with_translation_symmetry,
                                                                  self.kmesh,
                                                                  self.use_mpi)
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        if not self.use_mpi:
            rank = 0
        else:
            from pyscf.pbc.df.isdf.isdf_tools_mpi import rank
        
        if rank == 0:
            _benchmark_time(t1, t2, "build_partition", self)
        
        for i in range(self.natm):
            self.partition[i] = np.array(self.partition[i], dtype=np.int32)
            self.partition[i].sort()
        
        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        sync_aoR = False
        if self.direct:
            sync_aoR = True
            
        ## deal with translation symmetry ##
        
        first_natm = self.first_natm
        
        ####################################
        
        for x in range(self.natm):
            # print("len of partition[%d] = %d" % (x, len(self.partition[x])))
            logger.debug4(self, "len of partition[%d] = %d" % (x, len(self.partition[x])))
        
        if self.use_aft_ao:
            self.aoR = ISDF_Local_Utils.get_aoR_analytic(self.cell, self.coords, self.partition,
                                                         None,
                                                         first_natm,
                                                         self.group,
                                                         self.distance_matrix, 
                                                         self.AtmConnectionInfo, 
                                                         self.use_mpi, self.use_mpi, sync_aoR)                                                          
        else:
            assert self.rsjk is None and self.cell_rsjk is None
            self.aoR = ISDF_Local_Utils.get_aoR(self.cell, self.coords, self.partition, 
                                                      None,
                                                      first_natm,
                                                      self.group,
                                                      self.distance_matrix, 
                                                      self.AtmConnectionInfo, 
                                                      self.use_mpi, self.use_mpi, sync_aoR)
    
        memory = ISDF_Local_Utils._get_aoR_holders_memory(self.aoR)
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        if rank == 0:
            _benchmark_time(t1, t2, "build_aoR", self)
    
    def _allocate_jk_buffer(self, datatype, ngrids_local):
        pass
    
    @property
    def max_nao_involved(self):
        return np.max([aoR_holder.aoR.shape[0] for aoR_holder in self.aoR if aoR_holder is not None])
    
    @property
    def max_ngrid_involved(self):
        return np.max([aoR_holder.aoR.shape[1] for aoR_holder in self.aoR if aoR_holder is not None])
    
    @property
    def max_nIP_involved(self):
        return np.max([aoR_holder.aoR.shape[1] for aoR_holder in self.aoRg if aoR_holder is not None])
    
    @property
    def maxsize_group_naux(self):
        maxsize_group_naux = 0
        for group_id, atm_ids in enumerate(self.group):
            naux_tmp = 0
            for atm_id in atm_ids:
                naux_tmp += self.aoRg[atm_id].aoR.shape[1]
            maxsize_group_naux = max(maxsize_group_naux, naux_tmp)
        return maxsize_group_naux
    
    def deallocate_k_buffer(self):
        if hasattr(self, "build_k_buf") and self.build_k_buf is not None:
            del self.build_k_buf
            self.build_k_buf = None
        if hasattr(self, "build_VW_in_k_buf") and self.build_VW_in_k_buf is not None:
            del self.build_VW_in_k_buf
            self.build_VW_in_k_buf = None
    
    def allocate_k_buffer(self): 
        
        log = lib.logger.Logger(self.cell.stdout, self.cell.verbose)
        
        ### TODO: split grid again to reduce the size of buf when robust fitting is true! 
        # TODO: try to calculate the size when direct is true
        
        max_nao_involved   = self.max_nao_involved
        max_ngrid_involved = self.max_ngrid_involved
        max_nIP_involved   = self.max_nIP_involved
        maxsize_group_naux = self.maxsize_group_naux
        
        allocated = False
        
        if self.direct:
            if hasattr(self, "build_k_buf") and self.build_k_buf is not None:
                if hasattr(self, "build_VW_in_k_buf") and self.build_VW_in_k_buf is not None:
                    allocated = True
        else:
            if hasattr(self, "build_k_buf") and self.build_k_buf is not None:
                allocated = True
    
        if allocated:
            pass
        else:
            
            if self.direct:
                
                if self._limited_memory:
                    build_K_bunchsize = min(maxsize_group_naux, self._build_K_bunchsize)
                else:
                    build_K_bunchsize = maxsize_group_naux
                
                #### compare build_K_bunchsize with those buf used for W matrix ####
                
                size1 = maxsize_group_naux * self.nao
                size2 = maxsize_group_naux * max_nao_involved
                self.Density_RgAO_buf = np.zeros((size1+size2,), dtype=np.float64)

                #### allocate build_VW_in_k_buf ####      
                          
                mesh = self.cell.mesh
                ngrid= np.prod(mesh)
                ncomplex = mesh[0] * mesh[1] * (mesh[2]//2+1)
                nthread = lib.num_threads()
                
                build_K_bunchsize = max(maxsize_group_naux * self.naux//ngrid+2, build_K_bunchsize)
                build_K_bunchsize = max(maxsize_group_naux * max_nIP_involved//max_ngrid_involved+2, build_K_bunchsize)
                self._build_K_bunchsize = build_K_bunchsize
                
                size0 = (np.prod(self.cell.mesh) + 2 * ncomplex) * nthread
                size1 = build_K_bunchsize * np.prod(self.cell.mesh) 
                size2 = maxsize_group_naux * self.naux
                self.build_VW_in_k_buf = np.zeros((size0+size1+size2,), dtype=np.float64)
                
                #### allocate build_k_buf ####
                
                size1 = build_K_bunchsize * np.prod(self.cell.mesh) # density RgR 
                size2 = build_K_bunchsize * max_ngrid_involved      # ddot_res_RgR
                size3 = maxsize_group_naux * self.nao               # K1_tmp1
                #size4 = max_ngrid_involved * max_nao_involved      # K1_tmp1_ddot_res
                size4 = maxsize_group_naux * self.nao  
                #size5 = max_ngrid_involved * max_ngrid_involved
                size5 = 0
                size6 = self.nao * self.nao               # K1_final_ddot
                
                size = size1 + size2 + size3 + size4 + size5 + size6
                
                self.build_k_buf = np.zeros((size,), dtype=np.float64)
                
                log.info("In allocate_k_buffer, Density_RgAO_buf  memory = %d"%(self.Density_RgAO_buf.nbytes))
                log.info("In allocate_k_buffer, build_VW_in_k_buf memory = %d"%(self.build_VW_in_k_buf.nbytes))
                log.info("In allocate_k_buffer, build_k_buf       memory = %d"%(self.build_k_buf.nbytes))
                
            else:
                            
                self.Density_RgAO_buf = np.zeros((self.naux, self.nao), dtype=np.float64)
                max_dim = max(max_nao_involved, max_ngrid_involved, self.nao)
                
                ### size0 in getting W part of K ###
            
                size0 = self.naux * max_nIP_involved + self.naux * max_nao_involved + self.naux * max(max_nIP_involved, max_nao_involved)
                
                ### size1 in getting Density Matrix ### 
            
                size11 = self.nao * max_nIP_involved + self.nao * self.nao
                size1  = self.naux * self.nao + self.naux * max_dim + self.nao * self.nao
                size1 += self.naux * max_nao_involved     
                size1 = max(size1, size11)
            
                ### size2 in getting K ### 
                
                size2 = self.naux * max_nao_involved
                if self.with_robust_fitting:
                    size2 += self.naux * max_ngrid_involved + self.naux * max_nao_involved
                    size2 += self.naux * max_ngrid_involved
                self.build_k_buf = np.zeros((max(size0, size1, size2)), dtype=np.float64)
            
    def _construct_build_aoRg(self, IP_group, group=None):
        
        if group is None:
            group = []
            for i in range(self.natm):
                group.append([i])
        for i in range(len(group)):
            group[i] = np.array(group[i], dtype=np.int32)
            group[i].sort()
        assert self.natm == len(IP_group) 
        
        aoR_holders_res = []
        for i in range(self.natm):
            aoR_holders_res.append(None)

        assert hasattr(self, "partition")
        assert hasattr(self, "aoR")
        
        atm_ordering = []
        for i in range(len(group)):
            atm_ordering.extend(group[i]) 
        
        IP_ID_NOW = 0 
        GRID_ID_NOW = 0
        
        IP_loc_in_ordered_grids = []
        
        for atm_id in atm_ordering:
            aoR_holder = self.aoR[atm_id]
            if aoR_holder is None:
                if IP_group[atm_id] is None:
                    continue
                else:
                    IP_ID_NOW += len(IP_group[atm_id])
                    continue
            nIP = len(IP_group[atm_id])
            
            idx = np.searchsorted(self.partition[atm_id], IP_group[atm_id])
            
            ao_involved = aoR_holder.ao_involved.copy()
            aoR = aoR_holder.aoR[:, idx].copy()
            aoR_holders_res[atm_id] = ISDF_Local_Utils.aoR_Holder(aoR, ao_involved, IP_ID_NOW, IP_ID_NOW+nIP, IP_ID_NOW, IP_ID_NOW+nIP)
            
            IP_loc_in_ordered_grids.extend(idx+GRID_ID_NOW)

            IP_ID_NOW   += nIP
            GRID_ID_NOW += len(self.partition[atm_id])
            
        self.IP_loc_in_ordered_grids = np.array(IP_loc_in_ordered_grids, dtype=np.int32)
        assert self.IP_loc_in_ordered_grids.ndim == 1
        
        return aoR_holders_res
            
    def _determine_c(self):
        '''
        called in build_IP_local when c is not set
        empirical rule to determine c
        '''
        
        DEFAULT = 15
        SEGMENT = [1e-2, 1e-3, 1e-4, 1e-5]
        C       = [10, 20, 30, 35, 40]
        
        if self.rela_cutoff_QRCP is None:
            return DEFAULT
        else:
            if self.rela_cutoff_QRCP > SEGMENT[0]:
                return C[0]
            else:
                for i in range(1, len(SEGMENT)):
                    if self.rela_cutoff_QRCP > SEGMENT[i]:
                        return C[i]
                return C[-1]
        
    def build_IP_local(self, c=None, m=5, first_natm=None, group=None, Ls = None, debug=True):
        
        if c is None:
            c = self._determine_c()
        
        if first_natm is None:
            first_natm = self.natm
        
        if group == None:
            group = []
            for i in range(natm):
                group.append([i])
        
        self.group = group
        
        for i in range(len(group)):
            group[i] = np.array(group[i], dtype=np.int32)
            group[i].sort()
        
        # build partition and aoR # 
        
        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        self.build_partition_aoR(Ls)
        
        ao2atomID = self.ao2atomID
        partition = self.partition
        aoR  = self.aoR
        natm = self.natm
        nao  = self.nao
        
        self.partition_atmID_to_gridID = partition
        
        self.partition_group_to_gridID = []
        for i in range(len(group)):
            self.partition_group_to_gridID.append([])
            for atm_id in group[i]:
                self.partition_group_to_gridID[i].extend(partition[atm_id])
            self.partition_group_to_gridID[i] = np.array(self.partition_group_to_gridID[i], dtype=np.int32)
            # self.partition_group_to_gridID[i].sort()
        
        ngrids = self.coords.shape[0]
        
        gridID_2_atmID = np.zeros((ngrids), dtype=np.int32)
        
        for atm_id in range(natm):
            gridID_2_atmID[partition[atm_id]] = atm_id
        
        self.gridID_2_atmID  = gridID_2_atmID
        self.grid_ID_ordered = ISDF_Local_Utils._get_grid_ordering(self.partition, self.group, self.use_mpi)
        
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        if self.verbose and debug:
            _benchmark_time(t1, t2, "build_partition_aoR", self)
        
        t1 = t2
        
        if len(group) < first_natm:
            IP_Atm = select_IP_atm_ls(self, c+1, m, first_natm, 
                                      rela_cutoff=self.rela_cutoff_QRCP,
                                      no_retriction_on_nIP=self.no_restriction_on_nIP,
                                      use_mpi=self.use_mpi)
        else:
            IP_Atm = select_IP_atm_ls(self, c, m, first_natm, 
                                      rela_cutoff=self.rela_cutoff_QRCP,
                                      no_retriction_on_nIP=self.no_restriction_on_nIP,
                                      use_mpi=self.use_mpi)
        t3 = (lib.logger.process_clock(), lib.logger.perf_counter())
                
        self.aoRg_possible = self._construct_build_aoRg(IP_Atm, None)
        
        t4 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if self.verbose and debug:
            _benchmark_time(t3, t4, "build_aoRg_possible", self)
        
        select_IP_local_ls_drive(self, c, m, IP_Atm, group, use_mpi=self.use_mpi)
        
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        if self.verbose and debug:
            _benchmark_time(t1, t2, "select_IP", self)
        
        t1 = t2
        
        build_aux_basis_ls(self, group, self.IP_group, debug=debug, use_mpi=self.use_mpi)
        
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        if self.verbose and debug:
            _benchmark_time(t1, t2, "build_aux_basis", self)
        
        sys.stdout.flush()

    def get_nuc(self, kpts=None):
        if hasattr(self, "gdf"):
            return self.gdf.get_nuc(kpts)
        else:
            super().get_nuc(kpts)   

    def get_coulG(self):
        if hasattr(self, "rsjk") and self.rsjk is not None:
            
            ##### construct coulG_LR , copy from rsjk.py #####
    
            if self.rsjk.cell.dimension!=3:
                raise NotImplementedError('3D only')

            _, _, kws = self.rsjk.cell.get_Gv_weights(self.mesh)
            coulG_SR_at_G0 = np.pi/self.rsjk.omega**2 * kws
            kpt = np.zeros(3)
            with lib.temporary_env(self.rsjk.cell, dimension=3):
                coulG_SR = self.rsjk.weighted_coulG_SR(kpt, False, self.mesh)
            G0_idx = 0
            coulG_SR[G0_idx] += coulG_SR_at_G0
            coulG_full = self.rsjk.weighted_coulG(kpt, None, self.mesh, omega=0.0)
            self.coulG = coulG_full - coulG_SR
            
            coulG_bench = tools.get_coulG(self.cell_rsjk, mesh=self.cell_rsjk.mesh, omega=0.0)
            
            ### find coulG_full with values larger than 1e-6 ###
            
            idx = np.where(np.abs(coulG_full) > 1e-6)
            
            G1 = coulG_full[idx].copy()
            G2 = coulG_bench[idx].copy()
            ratio = G2/G1
            fac = ratio[0]
            assert fac == 1.0/kws 
            assert np.allclose(ratio, fac)
            self.coulG *= fac
            
        else:
            self.coulG = tools.get_coulG(self.cell, mesh=self.cell.mesh)
            
    def diag_dm(self, dm, linear_dep_threshold=1e-16):
        '''Solver for generalized eigenvalue problem

        .. math:: HC = SCE
        
            used only for occ-RI-K, better not merge into PySCF first!
        
        '''
        # print("ovlp = ", self.ovlp)
        
        # diagonalize overlap matrix
        e, v = scipy.linalg.eigh(self.ovlp)
        
        mask = e > linear_dep_threshold * e[-1]
        e = e[mask]
        v = v[:,mask]
        v*= np.sqrt(e)
        
        dm_new_basis = np.dot(v.T, np.dot(dm, v))
        
        mo_occ, mo_coeff = scipy.linalg.eigh(dm_new_basis)
        
        mo_coeff = np.dot(v, mo_coeff) # SC = mocoeff
        
        v /= np.sqrt(e) 
        
        mo_coeff = np.dot(v.T, mo_coeff)
        mo_coeff = (1.0/e).reshape(-1,1) * mo_coeff
        mo_coeff = np.dot(v, mo_coeff)
        
        return mo_occ[::-1], mo_coeff[:,::-1]

    def build_auxiliary_Coulomb(self, debug=True):
        
        if self.direct == True:
            return # do nothing
        
        ### the cutoff based on distance for V and W is used only for testing now ! ###
        
        distance_max = np.max(self.distance_matrix)
        if self.V_W_cutoff is not None and self.V_W_cutoff > distance_max:
            logger.warn(self, "WARNING : V_W_cutoff is larger than the maximum distance in the cell")
            self.V_W_cutoff = None # no cutoff indeed 
        if self.V_W_cutoff is not None:
            logger.debug4(self, "PBC_ISDF_Info_Quad:->build_auxiliary_Coulomb: V_W_cutoff   = %12.6e" % self.V_W_cutoff)
            logger.debug4(self, "PBC_ISDF_Info_Quad:->build_auxiliary_Coulomb: distance_max = %12.6e" % distance_max)
        
        if self.with_robust_fitting:
            build_auxiliary_Coulomb_local_bas(self, debug=debug, use_mpi=self.use_mpi)
        else:
            build_auxiliary_Coulomb_local_bas_wo_robust_fitting(self, debug=debug, use_mpi=self.use_mpi)
                
        if self.V_W_cutoff is not None:
            
            if hasattr(self, "V_R"):
                V = self.V_R
                
                bra_loc = 0
                for atm_i, aoRg_holder in enumerate(self.aoRg):
                    nbra = aoRg_holder.aoR.shape[1]
                    ket_loc = 0
                    for atm_j, aoR_holder in enumerate(self.aoR):
                        nket = aoR_holder.aoR.shape[1]
                        if self.distance_matrix[atm_i, atm_j] > self.V_W_cutoff:
                            V[bra_loc:bra_loc+nbra, ket_loc:ket_loc+nket] = 0.0
                        ket_loc += nket
                    bra_loc += nbra
                    
                self.V_R = V

            W = self.W
            
            bra_loc = 0
            for atm_i, aoRg_holder_bra in enumerate(self.aoRg):
                nbra = aoRg_holder.aoR.shape[1]
                ket_loc = 0
                for atm_j, aoRg_holder_ket in enumerate(self.aoRg):
                    nket = aoRg_holder.aoR.shape[1]
                    if self.distance_matrix[atm_i, atm_j] > self.V_W_cutoff:
                        W[bra_loc:bra_loc+nbra, ket_loc:ket_loc+nket] = 0.0
                    ket_loc += nket
                bra_loc += nbra
            
            self.W = W

    get_jk = ISDF_LinearScalingJK.get_jk_dm_quadratic
        
    def aoR_RangeSeparation(self, CompactAO):
        
        self.CompactAOList = np.array(CompactAO, dtype=np.int32)
        DiffuseAO = []
        for i in range(self.nao):
            if i not in CompactAO:
                DiffuseAO.append(i)
        self.DiffuseAOList = np.array(DiffuseAO, dtype=np.int32)
        
        IsCompact = np.zeros((self.nao), dtype=bool)
        IsCompact[CompactAO] = True
        IsCompact[DiffuseAO] = False
        self.IsCompact = IsCompact
        
        for aoR in self.aoR:
            aoR.RangeSeparation(IsCompact)
        for aoRg in self.aoRg:
            aoRg.RangeSeparation(IsCompact)
    
    def aoRg_full(self):
        
        fn_pack = getattr(libpbc, "_Pack_Matrix_SparseRow_DenseCol", None)
        assert fn_pack is not None

        partition = []
    
        res = np.zeros((self.nao, self.naux), dtype=np.float64)
        for i in range(self.natm):
            aoRg_i            = self.aoRg[i]
            ao_involved_i     = aoRg_i.ao_involved
            nao_i             = aoRg_i.aoR.shape[0]
            global_IP_begin_i = aoRg_i.global_gridID_begin
            nIP_i             = aoRg_i.aoR.shape[1]
                    
            fn_pack(
                res.ctypes.data_as(ctypes.c_void_p), 
                ctypes.c_int(res.shape[0]),
                ctypes.c_int(res.shape[1]),
                aoRg_i.aoR.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_i),
                ctypes.c_int(nIP_i),
                ao_involved_i.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(global_IP_begin_i),
                ctypes.c_int(global_IP_begin_i+nIP_i)
            )
            
            partition.append([global_IP_begin_i, global_IP_begin_i+nIP_i])
            
        return res, partition
      
    ### LS_THC fit ###
    
    def LS_THC_recompression(self, X:np.ndarray, force_LS_THC=True): 
        
        from pyscf.pbc.df.isdf.isdf_ao2mo import LS_THC 
        
        if force_LS_THC:
            self.with_robust_fitting = False
            self.force_LS_THC        = True
            self.W    = LS_THC(self, X) / (self.ngrids/self.cell.vol)
            self.aoRg = X
            self.aoR  = None
            self.V_R  = None
        else:
            self.force_LS_THC        = False
            self.W2    = LS_THC(self, X) / (self.ngrids/self.cell.vol)
            self.aoRg2 = X

    ### check aoR value ###
    
    def check_aoR(self):
        for aoR_holder in self.aoR:
            max_abs_index = np.unravel_index(np.argmax(np.abs(aoR_holder.aoR)), aoR_holder.aoR.shape)
            value = aoR_holder.aoR[max_abs_index[0]][max_abs_index[1]]
            ao_indx = aoR_holder.ao_involved[max_abs_index[0]]
            print("max_abs_value = ", value, " with indx = ", ao_indx, max_abs_index[1]+aoR_holder.global_gridID_begin)
            

if __name__ == '__main__':
    
    C = 15
    from pyscf.lib.parameters import BOHR
    from pyscf.pbc.df.isdf.isdf_tools_cell import build_supercell, build_supercell_with_partition
    import pyscf.pbc.gto as pbcgto
    
    verbose = 10
        
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
    KE_CUTOFF = 70
    # basis = 'unc-gth-cc-tzvp'
    # pseudo = "gth-hf"  
    basis = 'gth-dzvp'
    pseudo = "gth-pade"   
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=KE_CUTOFF, basis=basis, pseudo=pseudo)    
    prim_partition = [[0,1],[2,3],[4,5],[6,7]]
    # prim_partition = [[0], [1], [2], [3], [4], [5], [6], [7]]
    # prim_partition = [[0,1,2,3,4,5,6,7]]
    # prim_partition = [[0,1,2,3],[4,5,6,7]]
    
    prim_mesh = prim_cell.mesh

    Ls = [1, 1, 2]
    # Ls = [2, 2, 2]
    Ls = np.array(Ls, dtype=np.int32)
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    # mesh = None  ### NOTE: magically, use None will be much slower ? 
    
    cell, group_partition = build_supercell_with_partition(atm, prim_a, mesh=mesh, 
                                                     Ls=Ls,
                                                     basis=basis, 
                                                     pseudo=pseudo,
                                                     partition=prim_partition, ke_cutoff=KE_CUTOFF, verbose=verbose)
    print("group_partition = ", group_partition)
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    # pbc_isdf_info = PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, aoR_cutoff=1e-8, direct=False, use_occ_RI_K=False)
    pbc_isdf_info = PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, 
                                       aoR_cutoff=1e-8, 
                                       direct=True, limited_memory=True, build_K_bunchsize=32,
                                       use_occ_RI_K=False, rela_cutoff_QRCP=3e-3)
    pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition)
    pbc_isdf_info.build_auxiliary_Coulomb()    
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    _benchmark_time(t1, t2, "build isdf", pbc_isdf_info)
    
    # pbc_isdf_info.check_aoR()
    # exit(1)
    
    from pyscf.pbc import scf

    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 6
    mf.conv_tol = 1e-7
    
    mf.kernel()
    
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    _benchmark_time(t1, t2, "scf", pbc_isdf_info)
    sys.stdout.flush()
        
    ### compare without occ-RI-K ###
    
    pbc_isdf_info.direct = False
    pbc_isdf_info.build_auxiliary_Coulomb()    
    
    pbc_isdf_info.occ_RI_K = True
    
    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 8
    mf.conv_tol = 1e-7
    mf.kernel()
        
    ###### test LS_THC ######
    
    _myisdf = PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, aoR_cutoff=1e-8, direct=False, use_occ_RI_K=False)
    _myisdf.build_IP_local(c=13, m=5, group=group_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
    X,partition = _myisdf.aoRg_full() 
    print("X = ", X.shape)
    print("partition = ", partition)
    pbc_isdf_info.LS_THC_recompression(X)
    
    pbc_isdf_info.occ_RI_K = False
    
    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df   = pbc_isdf_info
    mf.max_cycle = 8
    mf.conv_tol  = 1e-7
    mf.kernel()