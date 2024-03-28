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
import pyscf.pbc.df.isdf.isdf_linear_scaling_base as ISDF_LinearScalingBase

##### all the involved algorithm in ISDF based on aoR_Holder ##### 

############ select IP ############

# ls stands for linear scaling

def select_IP_atm_ls(mydf, c:int, m:int, first_natm=None, global_IP_selection=True, 
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

    buf_size_per_thread = mydf.get_buffer_size_in_IP_selection(c, m)
    buf_size            = buf_size_per_thread

    if hasattr(mydf, "IO_buf"):
        buf = mydf.IO_buf
    else:
        buf = np.zeros((buf_size), dtype=np.float64)
        mydf.IO_buf = buf

    if buf.size < buf_size:
        # reallocate
        mydf.IO_buf = np.zeros((buf_size), dtype=np.float64)
        # print("reallocate buf of size = ", buf_size)
        buf = mydf.IO_buf
    buf_tmp = np.ndarray((buf_size), dtype=np.float64, buffer=buf)

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

        buf_tmp[:buf_size_per_thread] = 0.0

        # grid_ID = np.where(mydf.partition == atm_id)[0]
        grid_ID = mydf.partition[atm_id] 

        # offset  = 0
        # aoR_atm = np.ndarray((nao, grid_ID.shape[0]), dtype=np.complex128, buffer=buf_tmp, offset=offset)
        # aoR_atm = ISDF_eval_gto(mydf.cell, coords=coords[grid_ID], out=aoR_atm) * weight, # evaluate aoR should take weight into account
        aoR_atm = mydf.aoR[atm_id].aoR
            
        nao_tmp = aoR_atm.shape[0]
        
        # create buffer for this atm

        dtypesize = buf.dtype.itemsize

        offset += nao_tmp*grid_ID.shape[0] * dtypesize

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
        # print("ngrid = %d, npt_find = %d, cutoff = %12.6e" % (grid_ID.shape[0], npt_find, cutoff))
        pivot = pivot[:npt_find]
        pivot.sort()
        results.extend(list(grid_ID[pivot]))

    # print("results = ", results)

    if use_mpi:
        comm.Barrier()
        results = allgather(results)
        # results.sort()
    results.sort()
    
    if mydf.verbose:
        print("In select_IP, num_threads = ", lib.num_threads())

    del aoR_atm1
    del aoR_atm2
    del aoPairBuffer
    del R
    del thread_buffer
    del global_buffer

    return results

def select_IP_group_ls(mydf, aoRg_possible, c:int, m:int, group=None, group_2_IP_possible = None):
    
    assert isinstance(aoRg_possible, list)
    assert isinstance(group, list)
    assert isinstance(group_2_IP_possible, dict)
    
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
    aoRg_packed = ISDF_LinearScalingBase._pack_aoR_holder(aoRg_unpacked, nao).aoR
    

    # print("nao_group = ", nao_group)
    # print("nao = ", nao)    
    # print("c = %d, m = %d" % (c, m))

    naux_now = int(np.sqrt(c*nao)) + m
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
        IP_possible.extend(group_2_IP_possible[atm_id])
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

############ build aux bas ############

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
    mydf._allocate_jk_buffer(mydf.aoRg.dtype, mydf.ngrids_local)

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
        
        aoRg1 = ISDF_LinearScalingBase._pack_aoR_holder(aoRg_unpacked, mydf.nao).aoR
        aoR1 = ISDF_LinearScalingBase._pack_aoR_holder(aoR_unpacked, mydf.nao).aoR
        assert aoRg1.shape[0] == B.shape[0]
            
        A = lib.ddot(aoRg1.T, aoRg1)
        lib.square_inPlace(A)
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