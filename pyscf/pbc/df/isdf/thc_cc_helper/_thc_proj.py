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

from functools import reduce

import numpy as np
import ctypes
from pyscf.lib import logger
from pyscf import lib
libpbc = lib.load_library('libpbc')

def _thc_proj_2(Xo:np.ndarray,
                Xv:np.ndarray,
                tauo:np.ndarray,
                tauv:np.ndarray,
                partition,
                qr_cutoff = 1e-1):
    
    t1 = (logger.process_clock(), logger.perf_counter())
    
    nocc = Xo.shape[0]
    nvir = Xv.shape[0]
    nthc = Xo.shape[1]
    nlaplace = tauo.shape[1]
    
    pnt_pre_selected = []
    
    #### loop over each atm select potential points #### 
    
    fn_colpivot_qr = getattr(libpbc, "ColPivotQRRelaCut", None)
    assert(fn_colpivot_qr is not None)
    
    for iatm in range(len(partition)):
        
        id_begin, id_end = partition[iatm][0], partition[iatm][1]
    
        Xo_tmp = Xo[:, id_begin:id_end]
        Xv_tmp = Xv[:, id_begin:id_end]
    
        proj_full = np.einsum("iP,aP,iW,aW->iaPW", Xo_tmp, Xv_tmp, tauo, tauv, optimize=True)
    
        nrow = nocc * nvir 
        ncol = (id_end - id_begin) * nlaplace
        
        proj_full = proj_full.reshape(nrow, ncol)
    
        ##### prepare to perform the qr decomposition ##### 
    
        npt_find      = ctypes.c_int(0)
        pivot         = np.arange(ncol, dtype=np.int32)
        nthread       = lib.num_threads()
        thread_buffer = np.ndarray((nthread+1, ncol+1), dtype=np.float64)
        global_buffer = np.ndarray((1, ncol), dtype=np.float64)
        R = np.ndarray((nrow, ncol), dtype=np.float64)
    
        fn_colpivot_qr(proj_full.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_int(nrow),
                       ctypes.c_int(ncol),
                       ctypes.c_int(ncol),
                       ctypes.c_double(1e-14),
                       ctypes.c_double(qr_cutoff),
                       pivot.ctypes.data_as(ctypes.c_void_p),
                       R.ctypes.data_as(ctypes.c_void_p),
                       ctypes.byref(npt_find),
                       thread_buffer.ctypes.data_as(ctypes.c_void_p),
                       global_buffer.ctypes.data_as(ctypes.c_void_p))

        npt_find = npt_find.value
                        
        cutoff   = abs(R[npt_find-1, npt_find-1])
        pivot = pivot[:npt_find].copy()
        
        pivot = [ ((i // nlaplace) + id_begin) * nlaplace + (i % nlaplace) for i in pivot]
        pivot.sort()
        #print("pivot = ", pivot)
        print("_select_IP_direct: ncol = %d, npt_find = %d, cutoff = %12.6e" % (ncol, npt_find, cutoff))
        pnt_pre_selected.extend(pivot)
    
    #### global selection ####
    
    pnt_pre_selected.sort()
    
    pnt_pre_selected_partition = [] 
    for i in range(nlaplace):
        pnt_pre_selected_partition.append([])
    
    #print("pnt_pre_selected", pnt_pre_selected)
    #print("nlaplace = ", nlaplace)
    
    for i in pnt_pre_selected:
        #print(i, i % nlaplace)
        pnt_pre_selected_partition[i % nlaplace].append(i//nlaplace)
    
    id_flatted = []
    partition_laplace = [0]
    npnt = 0
    for i in range(nlaplace):
        #print("pnt_pre_selected_partition[%d]" % i, pnt_pre_selected_partition[i])
        pnt_pre_selected_partition[i].sort()
        npnt += len(pnt_pre_selected_partition[i])
        partition_laplace.append(npnt)
        for x in pnt_pre_selected_partition[i]:
            id_flatted.append((i, x))
            
    #print("partition_laplace", partition_laplace)
    #print("id_flatted", id_flatted)
    
    npnt_candidate = len(id_flatted)
    print("npnt_candidate = ", npnt_candidate)
    
    #### prepare to perform the qr decomposition ####
    
    proj_full = np.zeros((nocc, nvir, npnt_candidate))
    
    for i in range(nlaplace):
        proj_full[:,:,partition_laplace[i]:partition_laplace[i+1]] = np.einsum("iP,aP,i,a->iaP", Xo[:, pnt_pre_selected_partition[i]], Xv[:, pnt_pre_selected_partition[i]], tauo[:,i], tauv[:,i], optimize=True)

    #print(proj_full[0, :])

    proj_full = proj_full.reshape(nrow, npnt_candidate)
    ncol = npnt_candidate

    npt_find      = ctypes.c_int(0)
    pivot         = np.arange(ncol, dtype=np.int32)
    #print("pivot", pivot)
    nthread       = lib.num_threads()
    thread_buffer = np.ndarray((nthread+1, ncol+1), dtype=np.float64)
    global_buffer = np.ndarray((1, ncol), dtype=np.float64)
    R = np.ndarray((nrow, ncol), dtype=np.float64)
    
    fn_colpivot_qr(proj_full.ctypes.data_as(ctypes.c_void_p),
                   ctypes.c_int(nrow),
                   ctypes.c_int(ncol),
                   ctypes.c_int(ncol),
                   ctypes.c_double(1e-14),
                   ctypes.c_double(qr_cutoff),
                   pivot.ctypes.data_as(ctypes.c_void_p),
                   R.ctypes.data_as(ctypes.c_void_p),
                   ctypes.byref(npt_find),
                   thread_buffer.ctypes.data_as(ctypes.c_void_p),
                   global_buffer.ctypes.data_as(ctypes.c_void_p))

    npt_find = npt_find.value
                        
    cutoff = abs(R[npt_find-1, npt_find-1])
    pivot  = pivot[:npt_find].copy()

    #print("pivot = ", pivot)
    print("_select_IP_direct: ncol = %d, npt_find = %d, cutoff = %12.6e" % (ncol, npt_find, cutoff))
    
    selected = [id_flatted[i] for i in pivot]
    
    #print("selected", selected)
    
    selected_partition = []
    for i in range(nlaplace):
        selected_partition.append([])
    
    for i in selected:
        selected_partition[i[0]].append(i[1])
    
    for i in range(nlaplace):
        selected_partition[i].sort()
    
    #print("selected_partition", selected_partition)
    
    ######## build Xo_T2, Xv_T2, and the proj ########
    
    Xo_T2 = np.zeros((nocc, npt_find))
    Xv_T2 = np.zeros((nvir, npt_find))
    
    loc = 0
    for i in range(nlaplace):
        npnt_now = len(selected_partition[i])
        Xo_T2[:, loc:loc+npnt_now] = np.einsum("iP,i->iP", Xo[:, selected_partition[i]], tauo[:, i], optimize=True)
        Xv_T2[:, loc:loc+npnt_now] = np.einsum("aP,a->aP", Xv[:, selected_partition[i]], tauv[:, i], optimize=True)
        loc += npnt_now

    ################### build projector ###################
    
    t2 = (logger.process_clock(), logger.perf_counter()) 
    
    print("Time for THC PROJ: ", t2[1] - t1[1], "s", t2[0] - t1[0], "s")
    
    return Xo_T2, Xv_T2
    
    