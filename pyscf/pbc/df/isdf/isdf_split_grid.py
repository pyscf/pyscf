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

from pyscf.pbc.df.isdf.isdf_fast import rank, comm, comm_size, allgather, bcast, reduce, gather, alltoall, _comm_bunch

from pyscf.pbc.df.isdf.isdf_fast_mpi import get_jk_dm_mpi

import ctypes, sys

from multiprocessing import Pool

from memory_profiler import profile

libpbc = lib.load_library('libpbc')

from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto

########## WARNING: ABANDON, THIS IDEA DOES NOT WORK ! !!! ##########

############ select IP ############

def _select_IP_given_group(mydf, c:int, m:int, group=None, IP_possible = None):
    
    if group is None:
        raise ValueError("group must be specified")

    if mydf.verbose:
        print("In select_IP, num_threads = ", lib.num_threads())
        
    nthread = lib.num_threads()
    
    coords = mydf.coords
    
    weight = np.sqrt(mydf.cell.vol / coords.shape[0])
    
    # fn_colpivot_qr = getattr(libpbc, "ColPivotQR", None)
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

    ### do not deal with this problem right now
    # buf_size_per_thread = mydf.get_buffer_size_in_IP_selection_given_atm(c, m)
    # buf_size = buf_size_per_thread
    # buf = mydf.IO_buf
    # buf_tmp = np.ndarray((buf_size), dtype=buf.dtype, buffer=buf)
    # buf_tmp[:buf_size_per_thread] = 0.0 
    
    ##### random projection #####

    nao = mydf.nao
    
    aoR_atm = ISDF_eval_gto(mydf.cell, coords=coords[IP_possible]) * weight

    if hasattr(mydf, "aoR_cutoff"):
        max_row = np.max(np.abs(aoR_atm), axis=1)
        # print("max_row = ", max_row)
        # print("max_row.shape = ", max_row.shape)
        where = np.where(max_row > mydf.aoR_cutoff)[0]
        print("before cutoff aoR_atm.shape = ", aoR_atm.shape)
        aoR_atm = aoR_atm[where]
        print("after  cutoff aoR_atm.shape = ", aoR_atm.shape)
        nao = aoR_atm.shape[0]

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
        
    aoR_atm1 = lib.ddot(G1, aoR_atm)
    
    naux_now1 = aoR_atm1.shape[0]
    aoR_atm2 = lib.ddot(G2, aoR_atm)
    naux_now2 = aoR_atm2.shape[0]
    
    naux2_now = naux_now1 * naux_now2
    
    R = np.ndarray((naux2_now, IP_possible.shape[0]), dtype=np.float64)

    aoPairBuffer = np.ndarray((naux2_now, IP_possible.shape[0]), dtype=np.float64)

    fn_ik_jk_ijk(aoR_atm1.ctypes.data_as(ctypes.c_void_p),
                 aoR_atm2.ctypes.data_as(ctypes.c_void_p),
                 aoPairBuffer.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux_now1),
                 ctypes.c_int(naux_now2),
                 ctypes.c_int(IP_possible.shape[0]))

    aoR_atm1 = None
    aoR_atm2 = None

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
    
    aoPairBuffer = None
    R = None
    pivot = None
    thread_buffer = None
    global_buffer = None
    
    return results

def select_IP_local_drive(mydf, c, m, IP_possible_group, group, use_mpi=False):
    
    IP_group  = []

    ######### allocate buffer #########

    natm = mydf.natm
    
    for i in range(len(group)):
        IP_group.append(None)

    for i in range(len(group)):
        if use_mpi == False or ((use_mpi == True) and (i % comm_size == rank)):
            IP_group[i] = _select_IP_given_group(mydf, c, m, group=group[i], IP_possible=IP_possible_group[i])

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
    
    ### build ### 
    
    coords = mydf.coords
    weight = np.sqrt(mydf.cell.vol / mydf.coords.shape[0])
    mydf.aoRg = ISDF_eval_gto(mydf.cell, coords=coords[mydf.IP_flat]) * weight
    # mydf.aoRg1 = np.zeros_like(mydf.aoRg2)
    mydf.naux = mydf.aoRg.shape[1]
    
    print("IP_segment = ", mydf.IP_segment)
    
    return IP_group

############ build aux bas ############

def build_aux_basis_0(mydf, group, IP_group, debug=True):

    natm = mydf.natm

    aux_basis = []
    
    mydf.group_begin = 0
    mydf.group_end = len(group)
    
    print("mydf.naux = ", mydf.naux)
    print("mydf.ngrids = ", mydf.ngrids)

    aoRg = mydf.aoRg

    for i in range(len(group)):

        IP_loc_begin = mydf.IP_segment[i]
        IP_loc_end   = mydf.IP_segment[i+1]
        
        aoRg1 = aoRg[:,IP_loc_begin:IP_loc_end]
        
        A = lib.ddot(aoRg1.T, aoRg1)
        lib.square_inPlace(A)
        grid_ID = mydf.partition_group_to_gridID[i]
        B = lib.ddot(aoRg1.T, mydf.aoR[:, grid_ID])
        B = lib.square(B)
        
        with lib.threadpool_controller.limit(limits=lib.num_threads(), user_api='blas'):
            e, h = scipy.linalg.eigh(A)
        
        print("condition number = ", e[-1]/e[0])
        where = np.where(e > e[-1]*1e-16)[0]
        e = e[where]
        h = h[:,where]
        
        B = lib.ddot(h.T, B)
        lib.d_i_ij_ij(1.0/e, B, out=B)
        aux_tmp = lib.ddot(h, B)
        aux_basis.append(lib.ddot(h, B))
        # aux_basis[IP_loc_begin:IP_loc_end, grid_ID] = aux_tmp
        
        del e
        e = None
        del h
        h = None
        B = None
        A = None
        aux_tmp = None
    
    del aux_tmp
    aux_tmp = None
    del B
    B = None
    del A
    A = None
    
    # mydf.aux_basis = aux_basis
    # mydf.naux = mydf.aux_basis.shape[0]
    mydf.naux = mydf.aoRg.shape[1]
    # print("aux_basis.shape = ", mydf.aux_basis.shape)

def build_aux_basis_fast(mydf, group, IP_group, debug=True, use_mpi=False):
    
    ###### split task ######
    
    ngroup = len(group)
    nthread = lib.num_threads()
    
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
    
    ###### build aoR ###### 
    
    coords = mydf.coords
    weight = np.sqrt(mydf.cell.vol / mydf.coords.shape[0])
    grid_ID_local = []
    for i in range(group_begin, group_end):
        grid_ID_local.extend(mydf.partition_group_to_gridID[i])
    
    grid_ID_local = np.array(grid_ID_local, dtype=np.int32)
    
    if mydf.with_robust_fitting:
        aoR = ISDF_eval_gto(mydf.cell, coords=coords[grid_ID_local]) * weight
        mydf.aoR = aoR
    else:
        mydf.aoR = None
    mydf.ngrids_local = grid_ID_local.size
    
    mydf.grid_ID_local = grid_ID_local
    
    mydf._allocate_jk_buffer(mydf.aoRg.dtype, mydf.ngrids_local)

    ###### build aux basis ######
    
    aoRg = mydf.aoRg
    mydf.aux_basis = []
    
    grid_loc_now = 0
    
    for i in range(group_begin, group_end):
            
        IP_loc_begin = mydf.IP_segment[i]
        IP_loc_end   = mydf.IP_segment[i+1]
            
        aoRg1 = mydf.aoRg[:, IP_loc_begin:IP_loc_end]
            
        A = lib.ddot(aoRg1.T, aoRg1)
        lib.square_inPlace(A)
        grid_ID = mydf.partition_group_to_gridID[i]
        grid_loc_end = grid_loc_now + grid_ID.size
        if mydf.with_robust_fitting:
            B = lib.ddot(aoRg1.T, mydf.aoR[:, grid_loc_now:grid_loc_end])
        else:
            B = ISDF_eval_gto(mydf.cell, coords=coords[grid_ID_local[grid_loc_now:grid_loc_end]]) * weight
            B = lib.ddot(aoRg1.T, B)
        grid_loc_now = grid_loc_end
        lib.square_inPlace(B)
            
        with lib.threadpool_controller.limit(limits=lib.num_threads(), user_api='blas'):
            e, h = scipy.linalg.eigh(A)
            
        print("block %d condition number = " % i, e[-1]/e[0])
            
        where = np.where(e > e[-1]*1e-16)[0]
        e = e[where]
        h = h[:,where]
            
        B = lib.ddot(h.T, B)
        lib.d_i_ij_ij(1.0/e, B, out=B)
        aux_tmp = lib.ddot(h, B)
            
        mydf.aux_basis.append(aux_tmp)
        
    del A 
    A = None
    del B
    B = None
    del e
    e = None
    del h
    h = None
            
def build_auxiliary_Coulomb_local_bas_wo_robust_fitting(mydf, debug=True, use_mpi=False):
    
    if use_mpi:
        raise NotImplementedError("use_mpi = True is not supported")
        #### NOTE: one should bcast aux_basis first! ####

    
    t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    cell = mydf.cell
    mesh = cell.mesh
    mesh_int32         = np.array(mesh, dtype=np.int32)
    mydf._allocate_jk_buffer(mydf.aoRg.dtype, mydf.ngrids_local)
    
    naux = mydf.naux
    
    ncomplex = mesh[0] * mesh[1] * (mesh[2] // 2 + 1) * 2 
    
    group_begin = mydf.group_begin
    group_end = mydf.group_end
    ngroup = len(mydf.group)
    
    grid_ordering = mydf.grid_ID_ordered 
    
    coulG = tools.get_coulG(cell, mesh=mesh)
    mydf.coulG = coulG.copy()
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
    
    V = np.zeros((max_naux_bunch, np.prod(mesh_int32)), dtype=np.double)
    
    naux = mydf.naux
    
    W = np.zeros((naux_local, naux), dtype=np.double)
    
    aux_row_loc = 0
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
        aux_row_loc += aux_basis_now.shape[0]
    
    del buf
    buf = None
    del V
    V = None
    
    mydf.W = W
    
    if use_mpi:
        comm.Barrier()
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    if mydf.verbose > 0:
        _benchmark_time(t0, t1, 'build_auxiliary_Coulomb')

def build_auxiliary_Coulomb_local_bas(mydf, debug=True, use_mpi=False):
    
    t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    cell = mydf.cell
    mesh = cell.mesh
    
    mydf._allocate_jk_buffer(mydf.aoR.dtype, mydf.ngrids_local)
    
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

        # print("V.shape = ", V.shape)
        # # print("aux_basis.shape = ", aux_basis.shape)
        # print("self.jk_buffer.size    = ", mydf.jk_buffer.size)
        # print("self.jk_buffer.shape   = ", mydf.jk_buffer.shape)
        # sys.stdout.flush()
        # print("len(aux_bas) = ", len(aux_basis))
        
        shift_row = 0
        ngrid_now = 0
        for i in range(len(aux_basis)):
            
            aux_basis_now = aux_basis[i]
            grid_ID = mydf.partition_group_to_gridID[group_begin+i]
            assert aux_basis_now.shape[1] == grid_ID.size 
            ngrid_now += grid_ID.size
            # print("i = ", i)
            # print("shift_row = ", shift_row) 
            # print("aux_bas_now = ", aux_basis_now.shape)
            # print("ngrid_now = ", ngrid_now)
            # print("buf = ", buf.shape)
            # print("ngrid_ordering = ", grid_ordering.size)
            # sys.stdout.flush()
        
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

    coulG = tools.get_coulG(cell, mesh=mesh)
    mydf.coulG = coulG.copy()
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
            print("rank = %d, t_comm = %12.6e" % (rank, t_comm))
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
    
    mydf.W = np.zeros((mydf.naux, mydf.naux), dtype=mydf.aoRg.dtype) 
    
    grid_shift = 0
    for i in range(group_begin, group_end):
        aux_begin = aux_group_shift[i]
        aux_end   = aux_group_shift[i+1]
        ngrid_now = mydf.partition_group_to_gridID[i].size
        # print("aux_begin = %d, aux_end = %d, ngrid_now = %d" % (aux_begin, aux_end, ngrid_now))
        # print("grid_shift = %d" % grid_shift)
        # print("shape 1 = ", mydf.V_R[:,grid_shift:grid_shift+ngrid_now].shape)
        # print("shape 2 = ", mydf.aux_basis[i-group_begin].T.shape)
        sys.stdout.flush()
        mydf.W[:, aux_begin:aux_end] = lib.ddot(mydf.V_R[:, grid_shift:grid_shift+ngrid_now], mydf.aux_basis[i-group_begin].T)
        grid_shift += ngrid_now
    
    if use_mpi:
        comm.Barrier()
    
    # print("W = ", mydf.W[:5,:5])
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    if mydf.verbose > 0:
        _benchmark_time(t0, t1, 'build_auxiliary_Coulomb')
    
    sys.stdout.flush()

class PBC_ISDF_Info_SplitGrid(ISDF.PBC_ISDF_Info):
    
    def __init__(self, mol:Cell, aoR: np.ndarray = None,
                 with_robust_fitting=True,
                 Ls=None,
                 get_partition=True,
                 verbose = 1,
                 rela_cutoff_QRCP = None,
                 ):
    
        super().__init__(
            mol=mol,
            aoR=aoR,
            with_robust_fitting=with_robust_fitting,
            Ls=Ls,
            get_partition=get_partition,
            verbose=verbose
        )
        
        shl_atm = []
        
        natm = self.natm
        cell = self.cell
        
        for i in range(natm):
            shl_atm.append([None, None])
        
        for i in range(cell.nbas):
            atm_id = cell.bas_atom(i)
            if shl_atm[atm_id][0] is None:
                shl_atm[atm_id][0] = i
            shl_atm[atm_id][1] = i+1
        
        self.shl_atm = shl_atm
        self.aoloc_atm = cell.ao_loc_nr() 
        
        self.use_mpi = False

        self.aoR_cutoff = 1e-8
        
        if rela_cutoff_QRCP is None:
            self.no_restriction_on_nIP = False
            self.rela_cutoff_QRCP = 0.0
        else:
            self.no_restriction_on_nIP = True
            self.rela_cutoff_QRCP = rela_cutoff_QRCP

    def _allocate_jk_buffer(self, datatype, ngrids_local):

        # print("In _allocate_jk_buffer")
        # print("ngrids_local = ", ngrids_local)
        
        if self.jk_buffer is None:
            
            nao = self.nao
            ngrids = ngrids_local
            naux = self.naux
            
            buffersize_k = nao * ngrids + naux * ngrids + naux * naux + nao * nao
            buffersize_j = nao * ngrids + ngrids + nao * naux + naux + naux + nao * nao
            
            nThreadsOMP   = lib.num_threads()
            size_ddot_buf = max((naux*naux)+2, ngrids) * nThreadsOMP 
            
            if self.with_robust_fitting == False:
                buffersize_k = 0
                buffersize_j = 0
                size_ddot_buf = max((nao*nao)+2, ngrids) * nThreadsOMP 
            
            # print("buffersize_k = ", buffersize_k)
            
            if hasattr(self, "IO_buf"):
                
                if self.with_robust_fitting == False:
                    self.IO_buf = None
                    self.jk_buffer = np.ndarray((1,), dtype=datatype)
                    self.ddot_buf = np.ndarray((nThreadsOMP, max((nao*nao)+2, ngrids)), dtype=datatype)
                else:
                    if self.IO_buf.size < (max(buffersize_k, buffersize_j) + size_ddot_buf):
                        self.IO_buf = np.zeros((max(buffersize_k, buffersize_j) + size_ddot_buf,), dtype=datatype)
                    self.jk_buffer = np.ndarray((max(buffersize_k, buffersize_j),),
                                                dtype=datatype, buffer=self.IO_buf, offset=0)
                    offset         = max(buffersize_k, buffersize_j) * self.jk_buffer.dtype.itemsize
                    self.ddot_buf  = np.ndarray((nThreadsOMP, max((naux*naux)+2, ngrids)),
                                                dtype=datatype, buffer=self.IO_buf, offset=offset)

            else:
                
                self.jk_buffer = np.ndarray((max(buffersize_k, buffersize_j),), dtype=datatype)
                self.ddot_buf = np.zeros((size_ddot_buf,), dtype=datatype)


        else:
            assert self.jk_buffer.dtype == datatype
            assert self.ddot_buf.dtype == datatype
            
    def get_buffer_size_in_IP_selection_given_atm(self, c, m):
        pass

    def build_IP_Local(self, c=5, m=5,
                       # global_IP_selection=True,
                       build_global_basis=True,
                       IP_ID=None,
                       group=None,
                       debug=True):
    
        # build partition

        ao2atomID = self.ao2atomID
        partition = self.partition
        aoR  = self.aoR
        natm = self.natm
        nao  = self.nao
        ao2atomID = self.ao2atomID
        partition = self.partition
        aoR  = self.aoR
        natm = self.natm
        nao  = self.nao
        
        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        possible_IP = None
        if IP_ID is None:
            IP_ID = ISDF._select_IP_direct(self, c+1, m, global_IP_selection=False, 
                                           aoR_cutoff=self.aoR_cutoff,
                                           rela_cutoff=self.rela_cutoff_QRCP,
                                           no_retriction_on_nIP=self.no_restriction_on_nIP,
                                           use_mpi=self.use_mpi) # get a little bit more possible IPs
            IP_ID.sort()
            IP_ID = np.array(IP_ID, dtype=np.int32)
        possible_IP = np.array(IP_ID, dtype=np.int32)
        
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if debug and self.verbose > 0:
            _benchmark_time(t1, t2, 'build_IP')

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

        coords = self.coords
        weight = np.sqrt(self.cell.vol / coords.shape[0])
        aoR_possible_IP = ISDF_eval_gto(self.cell, coords=coords[possible_IP]) * weight
        
        self.aoR_possible_IP = aoR_possible_IP
        self.possible_IP = possible_IP
        
        if group == None:
            group = []
            for i in range(natm):
                group.append([i])
        
        possible_IP_atm = []
        for i in range(natm):
            possible_IP_atm.append([])
        for ip_id in possible_IP:
            atm_id = self.partition[ip_id]
            possible_IP_atm[atm_id].append(ip_id)
        for i in range(natm):
            possible_IP_atm[i] = np.array(possible_IP_atm[i], dtype=np.int32)
            possible_IP_atm[i].sort()
        
        possible_IP_group = []
        for i in range(len(group)):
            possible_IP_group.append([])
            for atm_id in group[i]:
                # print("atm_id = ", atm_id)
                # print("possible_IP_atm[atm_id] = ", possible_IP_atm[atm_id])
                possible_IP_group[i].extend(possible_IP_atm[atm_id])
            possible_IP_group[i].sort()
            possible_IP_group[i] = np.array(possible_IP_group[i], dtype=np.int32)
        
        partition_atmID_to_gridID = []
        for i in range(natm):
            partition_atmID_to_gridID.append([])
        for i in range(len(partition)):
            partition_atmID_to_gridID[partition[i]].append(i) # this can be extremely slow
        for i in range(natm):
            partition_atmID_to_gridID[i] = np.array(partition_atmID_to_gridID[i], dtype=np.int32)
            partition_atmID_to_gridID[i].sort()
        self.partition_atmID_to_gridID = partition_atmID_to_gridID
        self.partition_group_to_gridID = []
        for i in range(len(group)):
            self.partition_group_to_gridID.append([])
            for atm_id in group[i]:
                self.partition_group_to_gridID[i].extend(partition_atmID_to_gridID[atm_id])
            self.partition_group_to_gridID[i] = np.array(self.partition_group_to_gridID[i], dtype=np.int32)
            self.partition_group_to_gridID[i].sort()
        
        grid_ID_ordered = []
        for i in range(len(group)):
            grid_ID_ordered.extend(self.partition_group_to_gridID[i])
        grid_ID_ordered = np.array(grid_ID_ordered, dtype=np.int32)
        self.grid_ID_ordered = grid_ID_ordered
        
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        if debug and self.verbose > 0:
            _benchmark_time(t1, t2, 'build_aux_info')
            
        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
            
        self.group = group
        select_IP_local_drive(self, c, m, possible_IP_group, group, use_mpi=self.use_mpi)

        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        if debug and self.verbose > 0:
            _benchmark_time(t1, t2, 'select_IP_local_drive')

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

        # if self.use_mpi == False:
        #     build_aux_basis(self, group, self.IP_group, debug=True)
        # else:
        build_aux_basis_fast(self, group, self.IP_group, debug=debug, use_mpi=self.use_mpi) 
            
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        if debug and self.verbose > 0:
            _benchmark_time(t1, t2, 'build_aux_basis')
            
        sys.stdout.flush()
            
    def check_AOPairError(self):
        
        print("In check_AOPairError")
        
        for i in range(len(self.group)):
        # for i in range(1):
            
            IP_begin = self.IP_segment[i]
            IP_end   = self.IP_segment[i+1]
                
            print("group[%d] = " % i, self.group[i])
            print("IP_segment[%d] = %d, %d" % (i, IP_begin, IP_end))
                
            for atm_id in self.group[i]:
            
                ao_loc_begin = self.aoloc_atm[self.shl_atm[atm_id][0]]
                ao_loc_end   = self.aoloc_atm[self.shl_atm[atm_id][1]]
            
                aoRg1 = self.aoRg1[ao_loc_begin:ao_loc_end, IP_begin:IP_end]
                aoRg2 = self.aoRg2[:,IP_begin:IP_end]            
                coeff = numpy.einsum('ik,jk->ijk', aoRg1, aoRg2).reshape(-1, IP_end-IP_begin)        
                basis = self.aux_basis[IP_begin:IP_end,:]
                aux_approx = coeff @ basis
            
                # aoPair = numpy.einsum('ik,jk->ijk', self.aoR[ao_loc_begin:ao_loc_end, :], np.vstack([self.aoR[:ao_loc_begin, :], self.aoR[ao_loc_end:, :]])).reshape(-1, self.aoR.shape[1])
                aoPair = numpy.einsum('ik,jk->ijk', self.aoR[ao_loc_begin:ao_loc_end, :], self.aoR).reshape(-1, self.aoR.shape[1])
            
                diff = aux_approx - aoPair
            
                diff_pair_abs_max = np.max(np.abs(diff), axis=1)
            
                for j in range(ao_loc_end-ao_loc_begin):
                    loc_now = 0
                    for k in range(self.nao):
                        loc = j * self.nao + k
                        print("diff_pair_abs_max[%d,%d] = %12.6e" % (ao_loc_begin+j,k, diff_pair_abs_max[loc])) 
            
                # for i in range(diff_pair_abs_max.shape[0]):
                #     print("diff_pair_abs_max[%d] = %12.6e" % (i, diff_pair_abs_max[i]))

    def build_auxiliary_Coulomb(self, debug=True):
        if self.with_robust_fitting:
            return build_auxiliary_Coulomb_local_bas(self, debug=debug, use_mpi=self.use_mpi)
        else:
            return build_auxiliary_Coulomb_local_bas_wo_robust_fitting(self, debug=debug, use_mpi=self.use_mpi)

    get_jk = isdf_jk.get_jk_dm


class PBC_ISDF_Info_SplitGrid_MPI(PBC_ISDF_Info_SplitGrid):
    
    def __init__(self, mol:Cell, 
                 with_robust_fitting=True,
                 Ls=None,
                 get_partition=True,
                 verbose = 1,
                 ):

        if rank != 0:
            verbose = 0
        
        super().__init__(
            mol=mol,
            with_robust_fitting=with_robust_fitting,
            Ls=Ls,
            get_partition=get_partition,
            verbose=verbose
        )
        
        self.use_mpi = True
    
    get_jk = get_jk_dm_mpi

from pyscf.pbc.df.isdf.isdf_k import build_supercell

def build_supercell_with_partition(prim_atm, 
                                   prim_a, 
                                   mesh=None, 
                                   Ls = [1,1,1],
                                   partition = None, 
                                   basis='gth-dzvp', 
                                   pseudo='gth-pade', 
                                   ke_cutoff=70, 
                                   max_memory=2000, 
                                   precision=1e-8,
                                   use_particle_mesh_ewald=True,
                                   verbose=4):

    cell = build_supercell(prim_atm, prim_a, mesh=mesh, Ls=Ls, basis=basis, pseudo=pseudo, ke_cutoff=ke_cutoff, max_memory=max_memory, precision=precision, use_particle_mesh_ewald=use_particle_mesh_ewald, verbose=verbose)

    natm_prim = len(prim_atm)
    
    if partition is None:
        partition = []
        for i in range(natm_prim):
            partition.append([i])

    partition_supercell = []

    for ix in range(Ls[0]):
        for iy in range(Ls[1]):
            for iz in range(Ls[2]):
                cell_id = ix * Ls[1] * Ls[2] + iy * Ls[2] + iz
                for sub_partition in partition:
                    partition_supercell.append([x + cell_id * natm_prim for x in sub_partition])

    return cell, partition_supercell

#### split over grid points ? ####

C = 35

if __name__ == '__main__':

    cell   = pbcgto.Cell()
    boxlen = 3.5668
    cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    # cell.atom = '''
    #                C     0.      0.      0.
    #                C     0.8917  0.8917  0.8917
    #                C     1.7834  1.7834  0.
    #                C     2.6751  2.6751  0.8917
    #                C     1.7834  0.      1.7834
    #                C     2.6751  0.8917  2.6751
    #                C     0.      1.7834  1.7834
    #                C     0.8917  2.6751  2.6751
    #             '''
    
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
    
    verbose = 4
    if rank != 0:
        verbose = 0
    
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=KE_CUTOFF)
    prim_mesh = prim_cell.mesh
    # cell = tools.super_cell(cell, [1, 1, 2])
    # prim_partition = [[0,1,2,3],[4,5,6,7]]
    # prim_partition = [[0,1],[2,3],[4,5],[6,7]]
    # prim_partition = [[0,1,2,3,4,5,6,7]]
    prim_partition = [[0],[1], [2], [3], [4], [5], [6], [7]]
    
    Ls = [1, 1, 1]
    Ls = np.array(Ls, dtype=np.int32)
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    cell, partition = build_supercell_with_partition(atm, prim_a, mesh=mesh, Ls=Ls, partition=prim_partition, ke_cutoff=KE_CUTOFF, verbose=verbose)
        
    ########## get aoR ##########
    
    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2, _eval_rhoG

    df_tmp = MultiGridFFTDF2(cell)

    grids  = df_tmp.grids
    coords = np.asarray(grids.coords).reshape(-1,3)
    nx = grids.mesh[0]

    mesh   = grids.mesh
    ngrids = np.prod(mesh)
    assert ngrids == coords.shape[0]

    aoR   = df_tmp._numint.eval_ao(cell, coords)[0].T  # the T is important
    aoR  *= np.sqrt(cell.vol / ngrids)
    
    ##############################
    
    if rank == 0:
        
        pbc_isdf_info = PBC_ISDF_Info_SplitGrid(cell, None, with_robust_fitting=False, rela_cutoff_QRCP=1e-5)
        pbc_isdf_info.build_IP_Local(build_global_basis=True, c=C, group=partition)
        #print(pbc_isdf_info.IP_group) 
    
        print("pbc_isdf_info.naux = ", pbc_isdf_info.naux) 
        print("effective c = ", float(pbc_isdf_info.naux) / pbc_isdf_info.nao) 
    
        # pbc_isdf_info.check_AOPairError()

        pbc_isdf_info.build_auxiliary_Coulomb()
    
        # exit(1)
    
        from pyscf.pbc import scf

        mf = scf.RHF(cell)
        pbc_isdf_info.direct_scf = mf.direct_scf
        mf.with_df = pbc_isdf_info
        mf.max_cycle = 100
        mf.conv_tol = 1e-7

        print("mf.direct_scf = ", mf.direct_scf)

        mf.kernel()
        
        del mf
        del pbc_isdf_info
    

    
    if comm_size > 1:
        comm.Barrier()
        ### test MPI version ###
        pbc_isdf_info = PBC_ISDF_Info_SplitGrid_MPI(cell,verbose=verbose)
        pbc_isdf_info.build_IP_Local(build_global_basis=True, c=C, group=partition)
        pbc_isdf_info.build_auxiliary_Coulomb()
        
        from pyscf.pbc import scf

        mf = scf.RHF(cell)
        pbc_isdf_info.direct_scf = mf.direct_scf
        mf.with_df = pbc_isdf_info
        mf.max_cycle = 100
        mf.conv_tol = 1e-7

        print("mf.direct_scf = ", mf.direct_scf)

        mf.kernel()
        
        del mf
        del pbc_isdf_info