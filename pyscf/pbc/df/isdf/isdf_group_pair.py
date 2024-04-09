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

import pyscf.pbc.df.isdf.isdf_fast as ISDF

import ctypes

from multiprocessing import Pool

from memory_profiler import profile

libpbc = lib.load_library('libpbc')

from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto

############ select IP ############

def _select_IP_given_atm(mydf, c:int, m:int, group=None, IP_possible = None, use_mpi=False):
    
    if group is None:
        raise ValueError("group must be specified")

    if mydf.verbose:
        print("In select_IP, num_threads = ", lib.num_threads())
        
    nthread = lib.num_threads()
    
    coords = mydf.coords
    
    fn_colpivot_qr = getattr(libpbc, "ColPivotQR", None)
    assert(fn_colpivot_qr is not None)
    fn_ik_jk_ijk = getattr(libpbc, "NP_d_ik_jk_ijk", None)
    assert(fn_ik_jk_ijk is not None)

    weight = np.sqrt(mydf.cell.vol / coords.shape[0])
    
    #### perform QRCP ####

    ### do not deal with this problem right now
    # buf_size_per_thread = mydf.get_buffer_size_in_IP_selection_given_atm(c, m)
    # buf_size = buf_size_per_thread
    # buf = mydf.IO_buf
    # buf_tmp = np.ndarray((buf_size), dtype=buf.dtype, buffer=buf)
    # buf_tmp[:buf_size_per_thread] = 0.0 
    
    ##### random projection #####

    nao = mydf.nao
    
    nao_group = 0
    for atm_id in group:
        shl_begin = mydf.shl_atm[atm_id][0]
        shl_end   = mydf.shl_atm[atm_id][1]
        nao_atm = mydf.aoloc_atm[shl_end] - mydf.aoloc_atm[shl_begin]
        nao_group += nao_atm

    aoR_atm1 = np.zeros((nao_group, mydf.aoR_possible_IP.shape[1]), dtype=np.float64)
    
    loc_now = 0
    for atm_id in group:
        print("atm_id = ", atm_id)
        shl_begin = mydf.shl_atm[atm_id][0]
        shl_end   = mydf.shl_atm[atm_id][1]
        aoloc_begin = mydf.aoloc_atm[shl_begin]
        aoloc_end   = mydf.aoloc_atm[shl_end]
        nao_now = aoloc_end - aoloc_begin
        aoR_atm1[loc_now:loc_now+nao_now, :] = mydf.aoR_possible_IP[aoloc_begin:aoloc_end, :]
        loc_now += nao_now

    # print("nao_group = ", nao_group)
    # print("nao = ", nao)    
    # print("c = %d, m = %d" % (c, m))

    naux_now = int(np.sqrt(c*nao_group)) + m
    # G1 = np.random.rand(nao_group, naux_now)
    # G1, _ = numpy.linalg.qr(G1)
    # G1 = G1.T
    
    G1 = np.identity(nao_group)
    
    # G2 = np.random.rand(nao, naux_now)
    # G2, _ = numpy.linalg.qr(G2)
    # G2    = G2.T 
    # naux_now = nao
    G2 = np.identity(nao)
        
    # aoR_atm1 = mydf.aoR_possible_IP[aoloc_begin:aoloc_end, :]
    
    # diff = aoR_atm1 - mydf.aoR_possible_IP
    # print("diff = ", np.max(np.abs(diff)))
    # assert np.allclose(aoR_atm1, mydf.aoR_possible_IP)
    
    aoR_atm1 = lib.ddot(G1, aoR_atm1)
    
    naux_now1 = aoR_atm1.shape[0]
    aoR_atm2 = lib.ddot(G2, mydf.aoR_possible_IP)
    # aoR_atm2 = np.vstack([mydf.aoR_possible_IP[:aoloc_begin,:], mydf.aoR_possible_IP[aoloc_end:, :]])
    naux_now = aoR_atm2.shape[0]
    
    naux2_now = naux_now1 * naux_now
    
    R = np.ndarray((naux2_now, IP_possible.shape[0]), dtype=np.float64)

    aoPairBuffer = np.ndarray((naux2_now, IP_possible.shape[0]), dtype=np.float64)

    fn_ik_jk_ijk(aoR_atm1.ctypes.data_as(ctypes.c_void_p),
                 aoR_atm2.ctypes.data_as(ctypes.c_void_p),
                 aoPairBuffer.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux_now1),
                 ctypes.c_int(naux_now),
                 ctypes.c_int(IP_possible.shape[0]))

    aoR_atm1 = None
    aoR_atm2 = None

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
    
    
    ### clean up ###
    
    aoPairBuffer = None
    R = None
    pivot = None
    thread_buffer = None
    global_buffer = None
    
    return results

def select_IP_local_drive(mydf, c, m, IP_possible, group, use_mpi=False):
    
    IP_group  = []

    ######### allocate buffer #########

    natm = mydf.natm
    
    for i in range(len(group)):
        IP_group.append(None)

    for i in range(len(group)):
        IP_group[i] = _select_IP_given_atm(mydf, c, m, group=group[i], IP_possible=IP_possible, use_mpi=use_mpi)

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
    mydf.aoRg2 = ISDF_eval_gto(mydf.cell, coords=coords[mydf.IP_flat]) * weight
    mydf.aoRg1 = np.zeros_like(mydf.aoRg2)
    
    for i in range(len(group)):
        IP_begin = mydf.IP_segment[i]
        IP_end   = mydf.IP_segment[i+1]
        
        for atm_id in group[i]:
            shl_begin = mydf.shl_atm[atm_id][0]
            shl_end = mydf.shl_atm[atm_id][1]
            
            ao_begin = mydf.aoloc_atm[shl_begin]
            ao_end   = mydf.aoloc_atm[shl_end]
        
            # mydf.aoRg1.append(ISDF_eval_gto(mydf.cell, coords=coords[mydf.IP_group[i]], shls_slice=(shl_begin,shl_end)) * weight)
            
            mydf.aoRg1[ao_begin:ao_end, IP_begin:IP_end] = ISDF_eval_gto(mydf.cell, coords=coords[mydf.IP_group[i]], shls_slice=(shl_begin, shl_end)) * weight
    
    print("IP_segment = ", mydf.IP_segment)
    
    return IP_group

############ build aux bas ############

def build_aux_basis(mydf, group, IP_group, debug=True, use_mpi=False):

    natm = mydf.natm

    aux_basis = []

    aoRg2_global = mydf.aoRg2
    aoRg1_global = mydf.aoRg1

    for i in range(len(group)):
        # ao_loc_begin = mydf.aoloc_atm[mydf.shl_atm[i][0]]
        # ao_loc_end   = mydf.aoloc_atm[mydf.shl_atm[i][1]]
        
        IP_loc_begin = mydf.IP_segment[i]
        IP_loc_end   = mydf.IP_segment[i+1]
        
        # aoRg1 = mydf.aoRg1[i]
        aoRg1 = aoRg1_global[:,IP_loc_begin:IP_loc_end]
        aoRg2 = aoRg2_global[:,IP_loc_begin:IP_loc_end]
        
        A1 = lib.ddot(aoRg1.T, aoRg1)
        A2 = lib.ddot(aoRg2.T, aoRg2)
        A = A1 * A2 
        A1 = None
        A2 = None
        # B1 = lib.ddot(aoRg1.T, mydf.aoR[ao_loc_begin:ao_loc_end,:])
        B1 = lib.ddot(aoRg1.T, mydf.aoR)
        B2 = lib.ddot(aoRg2.T, mydf.aoR)
        B = B1 * B2 
        B1 = None
        B2 = None
        
        with lib.threadpool_controller.limit(limits=lib.num_threads(), user_api='blas'):
            e, h = scipy.linalg.eigh(A)
        
        print("condition number = ", e[-1]/e[0])
        where = np.where(e > e[-1]*1e-16)[0]
        e = e[where]
        h = h[:,where]
        
        B = lib.ddot(h.T, B)
        lib.d_i_ij_ij(1.0/e, B, out=B)
        aux_basis.append(lib.ddot(h, B))
        
        e = None
        h = None
        B = None
        A1 = None
        A2 = None
        A = None
        B1 = None
        B2 = None
    
    mydf.aux_basis = np.vstack(aux_basis)
    
    print("aux_basis.shape = ", mydf.aux_basis.shape)
