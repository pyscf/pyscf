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
import pyscf.pbc.df.isdf.isdf_ao2mo as isdf_ao2mo
import pyscf.pbc.df.isdf.isdf_jk as isdf_jk

from pyscf.pbc.df.isdf.isdf_fast import PBC_ISDF_Info

import pyscf.pbc.df.isdf.isdf_outcore as ISDF_outcore
import pyscf.pbc.df.isdf.isdf_fast as ISDF
import pyscf.pbc.df.isdf.isdf_split_grid as ISDF_split_grid
import pyscf.pbc.df.isdf.isdf_k as ISDF_K

from pyscf.pbc.df.isdf.isdf_fast import rank, comm, comm_size, allgather, bcast, reduce, gather, alltoall, _comm_bunch

from pyscf.pbc.df.isdf.isdf_fast_mpi import get_jk_dm_mpi

import ctypes, sys

from multiprocessing import Pool

from memory_profiler import profile

libpbc = lib.load_library('libpbc')

from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto
from pyscf.pbc.df.isdf.isdf_k import build_supercell
import pyscf.pbc.df.isdf.isdf_linear_scaling_base as ISDF_LinearScalingBase
import pyscf.pbc.df.isdf.isdf_linear_scaling_jk as ISDF_LinearScalingJK

##### all the involved algorithm in ISDF based on aoR_Holder ##### 

############ select IP ############

# ls stands for linear scaling

def select_IP_atm_ls(mydf, c:int, m:int, first_natm=None, 
                  rela_cutoff = 0.0, 
                  no_retriction_on_nIP = False,
                  use_mpi=False):

    # bunchsize = lib.num_threads()

    assert isinstance(mydf.aoR, list)
    assert isinstance(mydf.partition, list)

    if mydf.verbose:
        print("In select_IP, num_threads = ", lib.num_threads())

    ### determine the largest grids point of one atm ###

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

    # buf_size = mydf.get_buffer_size_in_IP_selection(c, m)

    # if hasattr(mydf, "IO_buf"):
    #     buf = mydf.IO_buf
    # else:
    #     buf = np.zeros((buf_size), dtype=np.float64)
    #     mydf.IO_buf = buf
    # if buf.size < buf_size:
    #     # reallocate
    #     mydf.IO_buf = np.zeros((buf_size), dtype=np.float64)
    #     # print("reallocate buf of size = ", buf_size)
    #     buf = mydf.IO_buf
        
    # buf_tmp = np.ndarray((buf_size), dtype=np.float64)

    ### loop over atm ###
    
    coords  = mydf.coords
    assert coords is not None

    results = []

    # fn_colpivot_qr = getattr(libpbc, "ColPivotQR", None)
    fn_colpivot_qr = getattr(libpbc, "ColPivotQRRelaCut", None)
    assert(fn_colpivot_qr is not None)
    fn_ik_jk_ijk = getattr(libpbc, "NP_d_ik_jk_ijk", None)
    assert(fn_ik_jk_ijk is not None)

    weight = np.sqrt(mydf.cell.vol / coords.shape[0])

    if first_natm is None:
        first_natm = natm
    
    aoR_atm1 = None
    aoR_atm2 = None
    aoPairBuffer = None
    R = None
    thread_buffer = None
    global_buffer = None
    
    for atm_id in range(first_natm):
        
        aoR = mydf.aoR[atm_id]
        if aoR is None:
            continue

        # buf_tmp[:] = 0.0

        # grid_ID = np.where(mydf.partition == atm_id)[0]
        grid_ID = mydf.partition[atm_id] 

        # offset  = 0
        # aoR_atm = np.ndarray((nao, grid_ID.shape[0]), dtype=np.complex128, buffer=buf_tmp, offset=offset)
        # aoR_atm = ISDF_eval_gto(mydf.cell, coords=coords[grid_ID], out=aoR_atm) * weight, # evaluate aoR should take weight into account
        aoR_atm = mydf.aoR[atm_id].aoR
            
        nao_tmp = aoR_atm.shape[0]
        
        # create buffer for this atm

        dtypesize = aoR_atm.dtype.itemsize

        # offset += nao_tmp*grid_ID.shape[0] * dtypesize

        nao_atm  = nao_per_atm[atm_id]
        naux_now = int(np.sqrt(c*nao_atm)) + m
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
        else:
            max_rank  = min(naux2_now, grid_ID.shape[0], nao_atm * c + m)
        npt_find      = ctypes.c_int(0)
        pivot         = np.arange(grid_ID.shape[0], dtype=np.int32)
        thread_buffer = np.ndarray((nthread+1, grid_ID.shape[0]+1), dtype=np.float64)
        global_buffer = np.ndarray((1, grid_ID.shape[0]), dtype=np.float64)

        # print("thread_buffer.shape = ", thread_buffer.shape)
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
            
        # aoPairBuffer, R, pivot, npt_find = colpivot_qr(aoPairBuffer, max_rank)
            
        cutoff = abs(R[npt_find-1, npt_find-1])
        print("ngrid = %d, npt_find = %d, cutoff = %12.6e" % (grid_ID.shape[0], npt_find, cutoff))
        pivot = pivot[:npt_find]
        pivot.sort()
        # results.extend(list(grid_ID[pivot]))
        
        atm_IP = grid_ID[pivot]
        atm_IP = np.array(atm_IP, dtype=np.int32)
        atm_IP.sort()
        results.append(atm_IP)

    # print("results = ", results)

    if use_mpi:
        comm.Barrier()
        results = allgather(results)
        # results.sort()
    # results.sort()
    
    if mydf.verbose:
        print("In select_IP, num_threads = ", lib.num_threads())

    del aoR_atm1
    del aoR_atm2
    del aoPairBuffer
    del R
    del thread_buffer
    del global_buffer
    # del buf_size

    return results

def select_IP_group_ls(mydf, aoRg_possible, c:int, m:int, group=None, atm_2_IP_possible = None):
    
    assert isinstance(aoRg_possible, list)
    assert isinstance(group, list)
    assert isinstance(atm_2_IP_possible, list)
    
    assert len(aoRg_possible) == len(atm_2_IP_possible)
    assert len(aoRg_possible) == mydf.natm
    
    
    if group is None:
        raise ValueError("group must be specified")

    if mydf.verbose:
        print("In select_IP, num_threads = ", lib.num_threads())
        
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
        aoRg_packed = ISDF_LinearScalingBase._pack_aoR_holder(aoRg_unpacked, nao).aoR
    
    nao = aoRg_packed.shape[0]

    # print("nao_group = ", nao_group)
    # print("nao = ", nao)    
    # print("c = %d, m = %d" % (c, m))

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
        IP_possible.extend(atm_2_IP_possible[atm_id])
    IP_possible = np.array(IP_possible, dtype=np.int32)

    if mydf.no_restriction_on_nIP:
        max_rank = min(naux2_now, IP_possible.shape[0])
    else:
        max_rank  = min(naux2_now, IP_possible.shape[0], nao_group * c)  
    # print("naux2_now = %d, max_rank = %d" % (naux2_now, max_rank))
    # print("IP_possible.shape = ", IP_possible.shape)
    # print("nao_group = ", nao_group)
    # print("c = ", c)
    # print("nao_group * c = ", nao_group * c)
    
    npt_find = ctypes.c_int(0)
    pivot    = np.arange(IP_possible.shape[0], dtype=np.int32)

    thread_buffer = np.ndarray((nthread+1, IP_possible.shape[0]+1), dtype=np.float64)
    global_buffer = np.ndarray((1, IP_possible.shape[0]), dtype=np.float64)

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
    # print("ngrid = %d, npt_find = %d, cutoff = %12.6e" % (IP_possible.shape[0], npt_find, cutoff))
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
    
    IP_group  = []
    
    aoRg_possible = mydf.aoRg_possible

    ######### allocate buffer #########

    natm = mydf.natm
    
    for i in range(len(group)):
        IP_group.append(None)

    if len(group) < natm:
        for i in range(len(group)):
            if use_mpi == False or ((use_mpi == True) and (i % comm_size == rank)):
                IP_group[i] = select_IP_group_ls(mydf, aoRg_possible, c, m, group=group[i], atm_2_IP_possible=IP_possible_atm)
    else:
        # print("use possible IP, do not perform select_IP_group_ls")
        # print("IP_possible_atm = ", IP_possible_atm)
        # IP_group = copy.deepcopy(IP_possible_atm)
        IP_group = IP_possible_atm

    if use_mpi:
        comm.Barrier()
        for i in range(len(group)):
            # if i % comm_size == rank:
            #     print("rank = %d, group[%d] = " % (rank, i), group[i])
            #     print("rank = %d, IP_group[%d] = " % (rank, i), IP_group[i])
            IP_group[i] = bcast(IP_group[i], root=i % comm_size)

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
        # print("partition_IP[%d] = " % i, partition_IP[i])
    
    ### build ### 
    
    if len(group) < natm:
        
        coords = mydf.coords
        weight = np.sqrt(mydf.cell.vol / mydf.coords.shape[0])
    
        del mydf.aoRg_possible
        mydf.aoRg_possible = None
    
        # mydf.aoRg = ISDF_eval_gto(mydf.cell, coords=coords[mydf.IP_flat]) * weight

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        mydf.aoRg = ISDF_LinearScalingBase.get_aoR(mydf.cell, mydf.coords, 
                                                   partition_IP, 
                                                   mydf.distance_matrix,
                                                   mydf.AtmConnectionInfo, 
                                                   False, False)
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, "build_aoRg")
    
    else:
        mydf.aoRg = mydf.aoRg_possible
    
    print("IP_segment = ", mydf.IP_segment)
    
    memory = ISDF_LinearScalingBase._get_aoR_holders_memory(mydf.aoRg)
    print("memory to store aoRg is ", memory)
        
    return IP_group

############ build aux bas ############

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
    
    ###### split task ######
    
    ngroup = len(group)
    nthread = lib.num_threads()
    # assert len(IP_group) == ngroup + 1
    assert len(IP_group) == ngroup
    
    group_bunchsize = ngroup
    
    if use_mpi:
        if ngroup % comm_size == 0 :
            group_bunchsize = ngroup // comm_size
        else:
            group_bunchsize = ngroup // comm_size + 1
    
    if use_mpi:
        group_begin = min(ngroup, rank * group_bunchsize)
        group_end = min(ngroup, (rank+1) * group_bunchsize)
    else:
        group_begin = 0
        group_end = ngroup
    
    ngroup_local = group_end - group_begin
    
    if ngroup_local == 0:
        print(" WARNING : rank = %d, ngroup_local = 0" % rank)
    
    mydf.group_begin = group_begin
    mydf.group_end = group_end
    
    ###### calculate grid segment ######
    
    grid_segment = [0]
    
    if use_mpi:
        for i in range(comm_size):
            p0 = min(ngroup, i * group_bunchsize)
            p1 = min(ngroup, (i+1) * group_bunchsize)
            size_now = 0
            for j in range(p0, p1):
                size_now += mydf.partition_group_to_gridID[j].size
            grid_segment.append(grid_segment[-1] + size_now)
    else:
        grid_segment.append(mydf.ngrids)
    
    if use_mpi == False or (use_mpi and rank == 0):
        print("grid_segment = ", grid_segment)
    
    mydf.grid_segment = grid_segment
    
    ###### build grid_ID_local ###### 
    
    coords = mydf.coords
    grid_ID_local = []
    for i in range(group_begin, group_end):
        grid_ID_local.extend(mydf.partition_group_to_gridID[i])
    
    grid_ID_local = np.array(grid_ID_local, dtype=np.int32)
    
    mydf.ngrids_local = grid_ID_local.size
    mydf.grid_ID_local = grid_ID_local
    # mydf._allocate_jk_buffer(mydf.aoRg.dtype, mydf.ngrids_local)

    ###### build aux basis ######
    
    mydf.aux_basis = []
    
    grid_loc_now = 0
    
    fn_cholesky = getattr(libpbc, "Cholesky", None)
    assert (fn_cholesky is not None)
    
    fn_build_aux = getattr(libpbc, "Solve_LLTEqualB_Parallel", None)
    assert(fn_build_aux is not None)
    
    for i in range(group_begin, group_end):
            
        IP_loc_begin = mydf.IP_segment[i]
        IP_loc_end   = mydf.IP_segment[i+1]
            
        # aoRg1 = mydf.aoRg[:, IP_loc_begin:IP_loc_end] 
        
        aoRg_unpacked = []
        aoR_unpacked = []
        
        for atm_id in group[i]:
            aoRg_unpacked.append(mydf.aoRg[atm_id])
            aoR_unpacked.append(mydf.aoR[atm_id])
        
        aoRg1 = ISDF_LinearScalingBase._pack_aoR_holder(aoRg_unpacked, mydf.nao)
        aoR1 = ISDF_LinearScalingBase._pack_aoR_holder(aoR_unpacked, mydf.nao)
        # assert aoRg1.shape[0] == aoR1.shape[0]
        
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
        # e, h = np.linalg.eigh(A)
        # print("e = ", e)
        grid_ID = mydf.partition_group_to_gridID[i]
        grid_loc_end = grid_loc_now + grid_ID.size
        B = lib.ddot(aoRg1.T, aoR1)
        grid_loc_now = grid_loc_end
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
        
        mydf.aux_basis.append(B.copy())
                    
    del A 
    A = None
    del B
    B = None
    del aoRg1
    aoRg1 = None
    del aoR1
    aoR1 = None
    
from pyscf.pbc.df.isdf.isdf_split_grid import build_auxiliary_Coulomb_local_bas_wo_robust_fitting, build_auxiliary_Coulomb_local_bas
    
class PBC_ISDF_Info_Quad(ISDF.PBC_ISDF_Info):
    
    # Quad stands for quadratic scaling
    
    def __init__(self, mol:Cell, 
                 # aoR: np.ndarray = None,
                 with_robust_fitting=True,
                 Ls=None,
                 # get_partition=True,
                 verbose = 1,
                 rela_cutoff_QRCP = None,
                 aoR_cutoff = 1e-8,
                 ):
        
        super().__init__(
            mol=mol,
            aoR=None,
            with_robust_fitting=with_robust_fitting,
            Ls=Ls,
            get_partition=False,
            verbose=verbose
        )
        
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

    def build_partition_aoR(self, Ls):
        
        if self.aoR is not None and self.partition is not None:
            return
            
        
        ##### build cutoff info #####   
        
        self.distance_matrix = ISDF_LinearScalingBase.get_cell_distance_matrix(self.cell)
        weight = np.sqrt(self.cell.vol / self.coords.shape[0])
        precision = self.aoR_cutoff
        rcut = ISDF_LinearScalingBase._estimate_rcut(self.cell, self.coords.shape[0], precision)
        rcut_max = np.max(rcut)
        atm2_bas = ISDF_LinearScalingBase._atm_to_bas(self.cell)
        self.AtmConnectionInfo = []
        
        for i in range(self.cell.natm):
            tmp = ISDF_LinearScalingBase.AtmConnectionInfo(self.cell, i, self.distance_matrix, precision, rcut, rcut_max, atm2_bas)
            self.AtmConnectionInfo.append(tmp)
        
        ##### build partition #####
        
        if Ls is None:
            lattice_x = self.cell.lattice_vectors()[0][0]
            lattice_y = self.cell.lattice_vectors()[1][1]
            lattice_z = self.cell.lattice_vectors()[2][2]
            
            Ls = [int(lattice_x/3)+1, int(lattice_y/3)+1, int(lattice_z/3)+1]

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        self.partition = ISDF_LinearScalingBase.get_partition(self.cell, self.coords, self.AtmConnectionInfo, 
                                                              Ls, self.use_mpi)
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, "build_partition")
        
        for i in range(self.natm):
            self.partition[i] = np.array(self.partition[i], dtype=np.int32)
            self.partition[i].sort()
        
        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        self.aoR = ISDF_LinearScalingBase.get_aoR(self.cell, self.coords, self.partition, self.distance_matrix, 
                                                  self.AtmConnectionInfo, 
                                                  self.use_mpi, self.use_mpi)
    
        memory = ISDF_LinearScalingBase._get_aoR_holders_memory(self.aoR)
        print("memory to store aoR is ", memory)
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, "build_aoR")
    
        ### check aoR ###
        
        # for i in range(self.natm):
        #     benchmark = ISDF_eval_gto(self.cell, coords=self.coords[self.partition[i]]) * weight
        #     aoR_test = self.aoR[i].todense(self.nao)
        #     np.allclose(benchmark, aoR_test)
    
    def _allocate_jk_buffer(self, datatype, ngrids_local):
        pass
    
    def allocate_k_buffer(self, max_nao_involved, max_nIP_involved, max_ngrid_involved): ### TODO: split grid again to reduce the size of buf when robust fitting is true! 
        if hasattr(self, "build_k_buf"):
            return
        else:
            
            print("In allocate_k_buffer ")
            print("max_nao_involved   = ", max_nao_involved)
            print("max_ngrid_involved = ", max_ngrid_involved)
            
            self.Density_RgAO_buf = np.zeros((self.naux, self.nao), dtype=np.float64)
            # self.Density_RgRg_buf = np.zeros((self.naux, self.naux), dtype=np.float64)
            
            # print("self.Density_RgRg_buf shape = ", self.Density_RgRg_buf.shape)
            
            # if self.with_robust_fitting:
            #     self.Density_RgR_buf = np.zeros((self.naux, np.prod(self.cell.mesh)), dtype=np.float64)
            #     print("self.Density_RgR_buf shape = ", self.Density_RgR_buf.shape)
            # else:
            #     self.Density_RgR_buf = None
                
            max_dim = max(max_nao_involved, max_ngrid_involved, self.nao)
                
            ### size0 in getting W part of K ###
            
            size0 = self.naux * max_nIP_involved + self.naux * max_nao_involved + self.naux * max(max_nIP_involved, max_nao_involved)
                
            ### size1 in getting Density Matrix ### 
            
            size11 = self.nao * max_nIP_involved + self.nao * self.nao
            
            size1  = self.naux * self.nao + self.naux * max_dim + self.nao * self.nao
            size1 += self.naux * max_nao_involved 
            
            size1 = max(size1, size11)
            
            print("max_dim = ", max_dim)
            print("size1 = ", size1)
            
            ### size2 in getting K ### 
                
            size2 = self.naux * max_nao_involved
            if self.with_robust_fitting:
                size2 += self.naux * max_ngrid_involved + self.naux * max_nao_involved
                size2 += self.naux * max_ngrid_involved
            
            print("size2 = ", size2)
                
            # self.ddot_k_buf = np.zeros((self.naux, max_nao_involved), dtype=np.float64)
            self.build_k_buf = np.zeros((max(size0, size1, size2)), dtype=np.float64)

            print("build_k_buf shape = ", self.build_k_buf.shape)
            
    
    def build_IP_local(self, c=5, m=5, first_natm=None, group=None, Ls = None, debug=True):
        
        if first_natm is None:
            first_natm = self.natm
        
        if group == None:
            group = []
            for i in range(natm):
                group.append([i])
        
        self.group = group
        
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
        
        self.gridID_2_atmID = gridID_2_atmID
        
        grid_ID_ordered = []
        # for i in range(len(group)):
        #     print("group_gridID = ", self.partition_group_to_gridID[i])
        #     grid_ID_ordered.extend(self.partition_group_to_gridID[i])
        for i in range(natm):  
            grid_ID_ordered.extend(partition[i])
        grid_ID_ordered = np.array(grid_ID_ordered, dtype=np.int32)
        self.grid_ID_ordered = grid_ID_ordered
        
        
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        if self.verbose and debug:
            _benchmark_time(t1, t2, "build_partition_aoR")
        
        t1 = t2
        
        if len(group) < natm:
            IP_Atm = select_IP_atm_ls(self, c+1, m, first_natm, 
                                      rela_cutoff=self.rela_cutoff_QRCP,
                                      no_retriction_on_nIP=self.no_restriction_on_nIP,
                                      use_mpi=self.use_mpi)
        else:
            IP_Atm = select_IP_atm_ls(self, c, 0, first_natm, 
                                      rela_cutoff=self.rela_cutoff_QRCP,
                                      no_retriction_on_nIP=self.no_restriction_on_nIP,
                                      use_mpi=self.use_mpi)
        t3 = (lib.logger.process_clock(), lib.logger.perf_counter())
        self.aoRg_possible = ISDF_LinearScalingBase.get_aoR(self.cell, self.coords, 
                                                            IP_Atm, self.distance_matrix, 
                                                            self.AtmConnectionInfo, self.use_mpi, self.use_mpi)
        t4 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t3, t4, "build_aoRg_possible")
        
        select_IP_local_ls_drive(self, c, m, IP_Atm, group, use_mpi=self.use_mpi)
        
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        if self.verbose and debug:
            _benchmark_time(t1, t2, "select_IP")
        
        t1 = t2
        
        build_aux_basis_ls(self, group, self.IP_group, debug=debug, use_mpi=self.use_mpi)
        
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        if self.verbose and debug:
            _benchmark_time(t1, t2, "build_aux_basis")
        
        sys.stdout.flush()
        

    def build_auxiliary_Coulomb(self, debug=True):
        
        ### the cutoff based on distance for V and W is used only for testing now ! ###
        
        distance_max = np.max(self.distance_matrix)
        if self.V_W_cutoff is not None and self.V_W_cutoff > distance_max:
            print("WARNING : V_W_cutoff is larger than the maximum distance in the cell")
            self.V_W_cutoff = None # no cutoff indeed 
        if self.V_W_cutoff is not None:
            print("V_W_cutoff   = ", self.V_W_cutoff)
            print("distance_max = ", distance_max)
        
        if self.with_robust_fitting:
            build_auxiliary_Coulomb_local_bas(self, debug=debug, use_mpi=self.use_mpi)
        else:
            build_auxiliary_Coulomb_local_bas_wo_robust_fitting(self, debug=debug, use_mpi=self.use_mpi)
        
        print("self.V_W_cutoff = ", self.V_W_cutoff)
        
        if self.V_W_cutoff is not None:
            
            if hasattr(self, "V_R"):
                V = self.V_R
                
                bra_loc = 0
                for atm_i, aoRg_holder in enumerate(self.aoRg):
                    nbra = aoRg_holder.aoR.shape[1]
                    ket_loc = 0
                    for atm_j, aoR_holder in enumerate(self.aoR):
                        nket = aoR_holder.aoR.shape[1]
                        # print("distance between %d and %d is %12.6e" % (atm_i, atm_j, self.distance_matrix[atm_i, atm_j]))
                        if self.distance_matrix[atm_i, atm_j] > self.V_W_cutoff:
                            # print("cutoff V_R between %d and %d" % (atm_i, atm_j))
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
                        # print("cutoff W between %d and %d" % (atm_i, atm_j))
                        W[bra_loc:bra_loc+nbra, ket_loc:ket_loc+nket] = 0.0
                    ket_loc += nket
                bra_loc += nbra
            
            self.W = W

    get_jk = ISDF_LinearScalingJK.get_jk_dm_quadratic
        
C = 7

from pyscf.lib.parameters import BOHR
from pyscf.pbc.df.isdf.isdf_split_grid import build_supercell_with_partition

if __name__ == '__main__':
    
    verbose = 4
    if rank != 0:
        verbose = 0
        
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
    
#     prim_a = np.array(
#                     [[14.572056092/2, 0.000000000, 0.000000000],
#                      [0.000000000, 14.572056092/2, 0.000000000],
#                      [0.000000000, 0.000000000,  6.010273939],]) * BOHR
#     atm = [
# ['Cu1',	(1.927800,	1.927800,	1.590250)],
# ['O1',	(1.927800,	0.000000,	1.590250)],
# ['O1',	(0.000000,	1.927800,	1.590250)],
# ['Ca',	(0.000000,	0.000000,	0.000000)],
#     ]
#     basis = {
#         'Cu1':'ecpccpvdz', 'Cu2':'ecpccpvdz', 'O1': 'ecpccpvdz', 'Ca':'ecpccpvdz'
#     }
#     pseudo = {'Cu1': 'gth-pbe-q19', 'Cu2': 'gth-pbe-q19', 'O1': 'gth-pbe', 'Ca': 'gth-pbe'}
#     ke_cutoff = 128 
#     prim_cell = ISDF_K.build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=ke_cutoff, basis=basis, pseudo=pseudo)
#     prim_mesh = prim_cell.mesh
    
    KE_CUTOFF = 70
    # KE_CUTOFF = 128
        
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=KE_CUTOFF)
    prim_mesh = prim_cell.mesh
    # prim_partition = [[0], [1], [2], [3], [4], [5], [6], [7]]
    prim_partition = [[0, 1, 2, 3, 4, 5, 6, 7]]
    # prim_partition = [[0,1],[2,3],[4,5],[6,7]]
    
    # prim_partition = [[0], [1], [2], [3]]
    
    Ls = [1, 1, 6]
    Ls = np.array(Ls, dtype=np.int32)
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    cell, group_partition = build_supercell_with_partition(atm, prim_a, mesh=mesh, 
                                                     Ls=Ls,
                                                     # basis=basis, pseudo=pseudo,
                                                     partition=prim_partition, ke_cutoff=KE_CUTOFF, verbose=verbose)
    print("group_partition = ", group_partition)
    pbc_isdf_info = PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, aoR_cutoff=1e-8)
    pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
    # pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition, Ls=[Ls[0]*3, Ls[1]*3, Ls[2]*3])
    pbc_isdf_info.Ls = Ls
    pbc_isdf_info.build_auxiliary_Coulomb(debug=True)
    
    
    
    from pyscf.pbc import scf

    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 12
    mf.conv_tol = 1e-7
    
    mf.kernel()
    
    from pyscf.pbc.df.isdf.isdf_densitymatrix_tool import analysis_dm, analysis_dm_on_grid
    
    dm = mf.make_rdm1()
    
    # analysis_dm(cell, dm, pbc_isdf_info.distance_matrix)
    # analysis_dm_on_grid(pbc_isdf_info, dm, pbc_isdf_info.distance_matrix)
    
    # pp = pbc_isdf_info.get_pp()
    # mf = scf.RHF(cell)
    # pbc_isdf_info.direct_scf = mf.direct_scf
    # # mf.with_df = pbc_isdf_info
    # mf.with_df.get_pp = lambda *args, **kwargs: pp
    # mf.max_cycle = 1
    # mf.conv_tol = 1e-7
    # mf.kernel()
    
    dm = mf.make_rdm1()
    
    pbc_isdf_info.V_W_cutoff = 13.0 
    pbc_isdf_info.build_auxiliary_Coulomb(debug=True)
    
    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 12
    mf.conv_tol = 1e-7
    
    mf.kernel(dm)
    