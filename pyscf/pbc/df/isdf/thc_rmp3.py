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

import numpy
import pyscf
from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo.incore import iden_coeffs
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts_helper
from pyscf.lib import logger
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique
from pyscf import __config__
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv, _contract_compact
import pyscf.pbc.gto as pbcgto

import numpy as np
import ctypes
libpbc = lib.load_library('libpbc')

from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time
import pyscf.pbc.df.isdf.isdf_linear_scaling as ISDF
from pyscf.pbc.df.isdf.isdf_tools_cell import build_supercell, build_supercell_with_partition
from pyscf.pbc.df.isdf.isdf_ao2mo import LS_THC, LS_THC_eri, laplace_holder

from pyscf.pbc.df.isdf.isdf_posthf import _restricted_THC_posthf_holder 
from pyscf.pbc.df.isdf._thc._RMP3 import *
# from pyscf.pbc.df.isdf._thc._RMP2_forloop import * 
from pyscf.pbc.df.isdf._thc._RMP3_CC_forloop import *
from pyscf.pbc.df.isdf._thc._RMP3_CX_forloop import *
from pyscf.pbc.df.isdf._thc._RMP3_XX_forloop import *

SCHEDULE_TYPE_NAIVE   = 0
SCHEDULE_TYPE_C       = 1
SCHEDULE_TYPE_OPT_MEM = 2
SCHEDULE_TYPE_FORLOOP = 3
SCHEDULE_TYPE_FORLOOP_ENOUGH_MEMORY = 4

### function to determine the bunchsize ### 

def _print_memory(prefix, memory):
    return prefix + " = %12.2f MB" % (memory/1e6)

def _bunchsize_determination_driver_rmp3(
    _fn_bufsize,
    _task_name,
    _nocc, 
    _nvir,
    _n_laplace,
    _nthc_int,
    memory,
    dtype_size = 8
):
    ##### assume parallel over one THC and one vir block #####
    
    ## (1) check whether memory is too limited ## 
    
    buf1 = _fn_bufsize(_nvir, _nocc, _n_laplace, _nthc_int, 1, 1, 1, 1)
    buf1 = np.sum(buf1)
    buf2 = 0
    
    if (buf1+buf2) * dtype_size > memory:
        print(_print_memory("memory needed %s" % (_task_name), (buf1+buf2)*dtype_size))
        raise ValueError("memory is too limited")
    
    ## (2) check whether memory is too large ## 
    
    buf1 = _fn_bufsize(_nvir, _nocc, _n_laplace, _nthc_int, _nthc_int, _nthc_int, _n_laplace, _n_laplace)
    buf1 = np.sum(buf1)
    
    if buf1 * dtype_size < memory:
        return _nthc_int, _n_laplace
    
    ## (3) memory is neither too limited nor too large ##
    
    thc_bunchsize = 8
    
    n_laplace_size = _n_laplace
    niter_laplace = 1
    
    while True:
        buf1 = _fn_bufsize(_nvir, _nocc, _n_laplace, _nthc_int, thc_bunchsize, thc_bunchsize, n_laplace_size, n_laplace_size)
        buf1 = np.sum(buf1)
        
        if buf1 * dtype_size > memory:
            niter_laplace *= 2
            n_laplace_size = (_n_laplace // niter_laplace) + 1
            if n_laplace_size == 1:
                break
        else:
            break

    buf1 = _fn_bufsize(_nvir, _nocc, _n_laplace, _nthc_int, thc_bunchsize, thc_bunchsize, n_laplace_size, n_laplace_size)
    buf1 = np.sum(buf1)
    
    if buf1 * dtype_size > memory:
        thc_bunchsize = 1
        buf1 = _fn_bufsize(_nvir, _nocc, _n_laplace, _nthc_int, thc_bunchsize, thc_bunchsize, n_laplace_size, n_laplace_size)
        buf1 = np.sum(buf1)
    
    thc_bunchsize_0 = thc_bunchsize
    thc_bunchsize_1 = thc_bunchsize
    
    reach_maximal_memory = False
    vir_bunchsize_fixed  = False
    
    while True:
        
        if buf1 * dtype_size < memory:
            thc_bunchsize_0 = thc_bunchsize_1
            thc_bunchsize_1 *= 2
            thc_bunchsize_1 = min(thc_bunchsize_1, _nthc_int)
        else:
            reach_maximal_memory = True
            thc_bunchsize_1 = (thc_bunchsize_0 + thc_bunchsize_1) // 2
        
        buf1 = _fn_bufsize(_nvir, _nocc, _n_laplace, _nthc_int, thc_bunchsize_1, thc_bunchsize_1, n_laplace_size, n_laplace_size)
        buf1 = np.sum(buf1)
        
        if buf1 < memory and reach_maximal_memory and (thc_bunchsize_1)<thc_bunchsize_0+4:
            break
                    
    return thc_bunchsize_1, n_laplace_size
        

from pyscf.pbc.df.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast

class THC_RMP3(_restricted_THC_posthf_holder):
    
    def __init__(self, my_isdf, my_mf, X,
                 laplace_rela_err = 1e-7,
                 laplace_order    = 2,
                 memory           = 128 * 1000 * 1000,
                 with_mpi=False):
        
        super().__init__(my_isdf, my_mf, X,
                            laplace_rela_err = laplace_rela_err,
                            laplace_order    = laplace_order)
        
        self.memory = memory
        self.buffer = None
        self.with_mpi = with_mpi
    
    #### kernels for THC-RMP2 #### 
    
    def _kernel_C(self):
        
        if rank > 0:
            res = None
        else:
            mp3_CC = 8* RMP3_CC(self.Z,
                                self.X_o,
                                self.X_v,
                                self.tau_o,
                                self.tau_v)
        
        
            mp3_CX_1 = RMP3_CX_1(self.Z,
                                self.X_o,
                                self.X_v,
                                self.tau_o,
                                self.tau_v)

            mp3_CX_2 = RMP3_CX_2(self.Z,
                                self.X_o,
                                self.X_v,
                                self.tau_o,
                                self.tau_v)
        
            mp3_CX_3 = RMP3_CX_3(self.Z,
                                self.X_o,
                                self.X_v,
                                self.tau_o,
                                self.tau_v)
    
            mp3_CX = -4 * (mp3_CX_1 + mp3_CX_2 + mp3_CX_3) 
        
            mp3_XX_1 = RMP3_XX_1(self.Z,
                                self.X_o,
                                self.X_v,
                                self.tau_o,
                                self.tau_v)

            mp3_XX_2 = RMP3_XX_2(self.Z,
                                self.X_o,
                                self.X_v,
                                self.tau_o,
                                self.tau_v)

            mp3_XX_3 = RMP3_XX_3(self.Z,
                                self.X_o,
                                self.X_v,
                                self.tau_o,
                                self.tau_v)
        
            mp3_XX_4 = RMP3_XX_4(self.Z,
                                self.X_o,
                                self.X_v,
                                self.tau_o,
                                self.tau_v)
        
            mp3_XX_5 = RMP3_XX_5(self.Z,
                                self.X_o,
                                self.X_v,
                                self.tau_o,
                                self.tau_v)
        
            mp3_XX_6 = RMP3_XX_6(self.Z,
                                self.X_o,
                                self.X_v,
                                self.tau_o,
                                self.tau_v)
    
            mp3_XX_7 = RMP3_XX_7(self.Z,
                                self.X_o,
                                self.X_v,
                                self.tau_o,
                                self.tau_v)
        
            mp3_XX_8 = RMP3_XX_8(self.Z,
                                self.X_o,
                                self.X_v,
                                self.tau_o,
                                self.tau_v)
        
            mp3_XX = 2 * (mp3_XX_1 + mp3_XX_2 + mp3_XX_3 + mp3_XX_5 + mp3_XX_8)
            mp3_XX-= (mp3_XX_4 + mp3_XX_6)
            mp3_XX-= 4*mp3_XX_7
            
            res = mp3_CC+mp3_CX+mp3_XX
    
            print("E_corr(RMP3) = " + str(mp3_CC+mp3_CX+mp3_XX))
        
        if comm_size > 1:
            res = bcast(res)
            
        return res
    
    def _kernel_opt_mem(self, get_memory_only=False):
        
        if self.buffer is None:
            self.buffer = np.zeros((self.memory//8))
        
        ### first fetch the size of required memory ### 
        
        res = None
        
        if rank == 0:
            
            ### print the memory needed for THC-RMP3 ###
            
            buf = RMP3_CC_determine_bucket_size(self.nvir, self.nocc, self.n_laplace, self.nthc_int)
            bufsize = np.sum(buf)
            print("memory needed for RMP3_CC = %12.2f MB" % ((bufsize)*8/1e6))
        
            buf = RMP3_CX_1_determine_bucket_size(self.nvir, self.nocc, self.n_laplace, self.nthc_int)
            bufsize = np.sum(buf)
            print("memory needed for RMP3_CX_1 = %12.2f MB" % ((bufsize)*8/1e6))
            
            buf = RMP3_CX_2_determine_bucket_size(self.nvir, self.nocc, self.n_laplace, self.nthc_int)
            bufsize = np.sum(buf)
            print("memory needed for RMP3_CX_2 = %12.2f MB" % ((bufsize)*8/1e6))
            
            buf = RMP3_CX_3_determine_bucket_size(self.nvir, self.nocc, self.n_laplace, self.nthc_int)
            bufsize = np.sum(buf)
            print("memory needed for RMP3_CX_3 = %12.2f MB" % ((bufsize)*8/1e6))
        
            buf = RMP3_XX_1_determine_bucket_size(self.nvir, self.nocc, self.n_laplace, self.nthc_int)
            bufsize = np.sum(buf)
            print("memory needed for RMP3_XX_1 = %12.2f MB" % ((bufsize)*8/1e6))
            
            buf = RMP3_XX_2_determine_bucket_size(self.nvir, self.nocc, self.n_laplace, self.nthc_int)
            bufsize = np.sum(buf)
            print("memory needed for RMP3_XX_2 = %12.2f MB" % ((bufsize)*8/1e6))
            
            buf = RMP3_XX_3_determine_bucket_size(self.nvir, self.nocc, self.n_laplace, self.nthc_int)
            bufsize = np.sum(buf)
            print("memory needed for RMP3_XX_3 = %12.2f MB" % ((bufsize)*8/1e6))
            
            buf = RMP3_XX_4_determine_bucket_size(self.nvir, self.nocc, self.n_laplace, self.nthc_int)
            bufsize = np.sum(buf)
            print("memory needed for RMP3_XX_4 = %12.2f MB" % ((bufsize)*8/1e6))
            
            buf = RMP3_XX_5_determine_bucket_size(self.nvir, self.nocc, self.n_laplace, self.nthc_int)
            bufsize = np.sum(buf)
            print("memory needed for RMP3_XX_5 = %12.2f MB" % ((bufsize)*8/1e6))
            
            buf = RMP3_XX_6_determine_bucket_size(self.nvir, self.nocc, self.n_laplace, self.nthc_int)
            bufsize = np.sum(buf)
            print("memory needed for RMP3_XX_6 = %12.2f MB" % ((bufsize)*8/1e6))
            
            buf = RMP3_XX_7_determine_bucket_size(self.nvir, self.nocc, self.n_laplace, self.nthc_int)
            bufsize = np.sum(buf)
            print("memory needed for RMP3_XX_7 = %12.2f MB" % ((bufsize)*8/1e6))
            
            buf = RMP3_XX_8_determine_bucket_size(self.nvir, self.nocc, self.n_laplace, self.nthc_int)
            bufsize = np.sum(buf)
            print("memory needed for RMP3_XX_8 = %12.2f MB" % ((bufsize)*8/1e6))
        
            if get_memory_only:
                return 
        
            ### CC term 
        
            t1 = (logger.process_clock(), logger.perf_counter())
            mp3_CC = 8* RMP3_CC_opt_mem(self.Z,
                                        self.X_o,
                                        self.X_v,
                                        self.tau_o,
                                        self.tau_v,
                                        self.buffer)
            t2 = (logger.process_clock(), logger.perf_counter())
            _benchmark_time(t1, t2, "RMP2_J_opt_mem")
            
            print("RMP3_CC = " + str(mp3_CC))
            
            ### CX term
            
            mp3_CX_1 = RMP3_CX_1_opt_mem(self.Z,
                                         self.X_o,
                                         self.X_v,
                                         self.tau_o,
                                         self.tau_v,
                                         self.buffer)
            t3 = (logger.process_clock(), logger.perf_counter())
            _benchmark_time(t2, t3, "RMP3_CX_1_opt_mem")
            
            mp3_CX_2 = RMP3_CX_2_opt_mem(self.Z,
                                            self.X_o,
                                            self.X_v,
                                            self.tau_o,
                                            self.tau_v,
                                            self.buffer)
            t4 = (logger.process_clock(), logger.perf_counter())
            
            _benchmark_time(t3, t4, "RMP3_CX_2_opt_mem")
            
            mp3_CX_3 = RMP3_CX_3_opt_mem(self.Z,
                                            self.X_o,
                                            self.X_v,
                                            self.tau_o,
                                            self.tau_v,
                                            self.buffer)
            
            t5 = (logger.process_clock(), logger.perf_counter())
            
            _benchmark_time(t4, t5, "RMP3_CX_3_opt_mem")
            
            mp3_CX = -4 * (mp3_CX_1 + mp3_CX_2 + mp3_CX_3)
            
            print("RMP3_CX = " + str(mp3_CX))
            
            ### XX term
            
            mp3_XX_1 = RMP3_XX_1_opt_mem(self.Z,
                                         self.X_o,
                                         self.X_v,
                                         self.tau_o,
                                         self.tau_v,
                                         self.buffer)
            
            mp3_XX_2 = RMP3_XX_2_opt_mem(self.Z,
                                            self.X_o,
                                            self.X_v,
                                            self.tau_o,
                                            self.tau_v,
                                            self.buffer)
            
            mp3_XX_3 = RMP3_XX_3_opt_mem(self.Z,
                                            self.X_o,
                                            self.X_v,
                                            self.tau_o,
                                            self.tau_v,
                                            self.buffer)

            mp3_XX_4 = RMP3_XX_4_opt_mem(self.Z,
                                            self.X_o,
                                            self.X_v,
                                            self.tau_o,
                                            self.tau_v,
                                            self.buffer)
            
            mp3_XX_5 = RMP3_XX_5_opt_mem(self.Z,
                                            self.X_o,
                                            self.X_v,
                                            self.tau_o,
                                            self.tau_v,
                                            self.buffer)
        
            mp3_XX_6 = RMP3_XX_6_opt_mem(self.Z,
                                            self.X_o,
                                            self.X_v,
                                            self.tau_o,
                                            self.tau_v,
                                            self.buffer)
        
            mp3_XX_7 = RMP3_XX_7_opt_mem(self.Z,
                                            self.X_o,
                                            self.X_v,
                                            self.tau_o,
                                            self.tau_v,
                                            self.buffer)
            
            mp3_XX_8 = RMP3_XX_8_opt_mem(self.Z,
                                            self.X_o,
                                            self.X_v,
                                            self.tau_o,
                                            self.tau_v,
                                            self.buffer)
            
            mp3_XX = 2 * (mp3_XX_1 + mp3_XX_2 + mp3_XX_3 + mp3_XX_5 + mp3_XX_8)
            mp3_XX-= (mp3_XX_4 + mp3_XX_6)
            mp3_XX-= 4*mp3_XX_7
            
            print("RMP3_XX = " + str(mp3_XX))
            
            t6 = (logger.process_clock(), logger.perf_counter())
            _benchmark_time(t5, t6, "RMP3_XX_opt_mem")
            
            print("E_corr(RMP3) = " + str(mp3_CC+mp3_CX+mp3_XX))
            res = mp3_CC+mp3_CX+mp3_XX
            
        if comm_size > 1:
        
            res = bcast(res)
        
        return res

    def _kernel_forloop(self):
        
        if self.buffer is None:
            self.buffer = np.zeros((self.memory//8))
        
        # raise NotImplementedError("forloop version of THC-RMP3 is not implemented yet")

        ### first fetch the size of required memory ### 
        
        bunchsize = 8
        n_laplace = 2
        
        res = None
        
        #### CC term ####
        
        t0 = (logger.process_clock(), logger.perf_counter())
        
        if rank == 0:
        
            ### print the memory needed for THC-RMP3 ###
            
            t1 = (logger.process_clock(), logger.perf_counter())
            mp3_CC = 8 * RMP3_CC_forloop_(self.Z,
                                        self.X_o,
                                        self.X_v,
                                        self.tau_o,
                                        self.tau_v,
                                        self.buffer)
            t2 = (logger.process_clock(), logger.perf_counter())
            _benchmark_time(t1, t2, "RMP3_CC_forloop")
            
        else:
            mp3_CC = None
            
        if comm_size > 1:
            mp3_CC = bcast(mp3_CC)
        
        if rank == 0:
            print("RMP3_CC = " + str(mp3_CC))
        
        #### CX term ####
        
        t1 = (logger.process_clock(), logger.perf_counter())
        
        if rank == 0:
            bunchsize, n_laplace = _bunchsize_determination_driver_rmp3(
                RMP3_CX_1_forloop_Q_R_determine_bucket_size_forloop,
                "RMP3_CX_1",
                self.nocc,
                self.nvir,
                self.n_laplace,
                self.nthc_int,
                self.memory
            )
            print("bunchsize = %d, n_laplace = %d for CX 1" % (bunchsize, n_laplace))
        else:
            bunchsize = None
            n_laplace = None
        
        if self.with_mpi:
            bunchsize = bcast(bunchsize)
            n_laplace = bcast(n_laplace)

        mp3_CX_1 = RMP3_CX_1_forloop_Q_R_forloop_Q_R(
            self.Z,
            self.X_o,
            self.X_v,
            self.tau_o,
            self.tau_v,
            self.buffer,
            bunchsize,
            bunchsize,
            n_laplace,
            n_laplace,
            self.with_mpi
        )

        if rank == 0:
            bunchsize, n_laplace = _bunchsize_determination_driver_rmp3(
                RMP3_CX_2_forloop_P_R_determine_bucket_size_forloop,
                "RMP3_CX_2",
                self.nocc,
                self.nvir,
                self.n_laplace,
                self.nthc_int,
                self.memory
            )
            print("bunchsize = %d, n_laplace = %d for CX 2" % (bunchsize, n_laplace))
        else:
            bunchsize = None
            n_laplace = None
        
        if self.with_mpi:
            bunchsize = bcast(bunchsize)
            n_laplace = bcast(n_laplace)

        mp3_CX_2 = RMP3_CX_2_forloop_P_R_forloop_P_R(
            self.Z,
            self.X_o,
            self.X_v,
            self.tau_o,
            self.tau_v,
            self.buffer,
            bunchsize,
            bunchsize,
            n_laplace,
            n_laplace,
            self.with_mpi
        )

        if rank == 0:
            bunchsize, n_laplace = _bunchsize_determination_driver_rmp3(
                RMP3_CX_3_forloop_P_T_determine_bucket_size_forloop,
                "RMP3_CX_3",
                self.nocc,
                self.nvir,
                self.n_laplace,
                self.nthc_int,
                self.memory
            )
            print("bunchsize = %d, n_laplace = %d for CX 3" % (bunchsize, n_laplace))
        else:
            bunchsize = None
            n_laplace = None
        
        if self.with_mpi:
            bunchsize = bcast(bunchsize)
            n_laplace = bcast(n_laplace)
            
        mp3_CX_3 = RMP3_CX_3_forloop_P_T_forloop_P_T(
            self.Z,
            self.X_o,
            self.X_v,
            self.tau_o,
            self.tau_v,
            self.buffer,
            bunchsize,
            bunchsize,
            n_laplace,
            n_laplace,
            self.with_mpi
        )

        mp3_CX = -4 * (mp3_CX_1 + mp3_CX_2 + mp3_CX_3)

        t2 = (logger.process_clock(), logger.perf_counter())

        if rank == 0:
            print("RMP3_CX = " + str(mp3_CX))
            _benchmark_time(t1, t2, "RMP3_CX_forloop")

        #### XX term ####
        
        ####### 1 #######
        
        if rank == 0:
            bunchsize, n_laplace = _bunchsize_determination_driver_rmp3(
                RMP3_XX_1_forloop_P_R_determine_bucket_size_forloop,
                "RMP3_XX_1",
                self.nocc,
                self.nvir,
                self.n_laplace,
                self.nthc_int,
                self.memory
            )
            print("bunchsize = %d, n_laplace = %d for XX 1" % (bunchsize, n_laplace))
        else:
            bunchsize = None
            n_laplace = None
        
        if self.with_mpi:
            bunchsize = bcast(bunchsize)
            n_laplace = bcast(n_laplace)

        mp3_XX_1 = RMP3_XX_1_forloop_P_R_forloop_P_R(
            self.Z,
            self.X_o,
            self.X_v,
            self.tau_o,
            self.tau_v,
            self.buffer,
            bunchsize,
            bunchsize,
            n_laplace,
            n_laplace,
            self.with_mpi
        )
        
        ####### 2 #######
        
        if rank == 0:
            bunchsize, n_laplace = _bunchsize_determination_driver_rmp3(
                RMP3_XX_2_forloop_P_S_determine_bucket_size_forloop,
                "RMP3_XX_2",
                self.nocc,
                self.nvir,
                self.n_laplace,
                self.nthc_int,
                self.memory
            )
            print("bunchsize = %d, n_laplace = %d for XX 2" % (bunchsize, n_laplace))
        else:
            bunchsize = None
            n_laplace = None
        
        if self.with_mpi:
            bunchsize = bcast(bunchsize)
            n_laplace = bcast(n_laplace)

        mp3_XX_2 = RMP3_XX_2_forloop_P_S_forloop_P_S(
            self.Z,
            self.X_o,
            self.X_v,
            self.tau_o,
            self.tau_v,
            self.buffer,
            bunchsize,
            bunchsize,
            n_laplace,
            n_laplace,
            self.with_mpi
        )
        
        ####### 3 #######
        
        if rank == 0:
            bunchsize, n_laplace = _bunchsize_determination_driver_rmp3(
                RMP3_XX_3_forloop_P_S_determine_bucket_size_forloop,
                "RMP3_XX_3",
                self.nocc,
                self.nvir,
                self.n_laplace,
                self.nthc_int,
                self.memory
            )
            print("bunchsize = %d, n_laplace = %d for XX 3" % (bunchsize, n_laplace))
        else:
            bunchsize = None
            n_laplace = None
        
        if self.with_mpi:
            bunchsize = bcast(bunchsize)
            n_laplace = bcast(n_laplace)

        mp3_XX_3 = RMP3_XX_3_forloop_P_S_forloop_P_S(
            self.Z,
            self.X_o,
            self.X_v,
            self.tau_o,
            self.tau_v,
            self.buffer,
            bunchsize,
            bunchsize,
            n_laplace,
            n_laplace,
            self.with_mpi
        )
        
        ####### 4 #######
        
        if rank == 0:
            bunchsize, n_laplace = _bunchsize_determination_driver_rmp3(
                RMP3_XX_4_forloop_P_R_determine_bucket_size_forloop,
                "RMP3_XX_4",
                self.nocc,
                self.nvir,
                self.n_laplace,
                self.nthc_int,
                self.memory
            )
            print("bunchsize = %d, n_laplace = %d for XX 4" % (bunchsize, n_laplace))
        else:
            bunchsize = None
            n_laplace = None
        
        if self.with_mpi:
            bunchsize = bcast(bunchsize)
            n_laplace = bcast(n_laplace)

        mp3_XX_4 = RMP3_XX_4_forloop_P_R_forloop_P_R(
            self.Z,
            self.X_o,
            self.X_v,
            self.tau_o,
            self.tau_v,
            self.buffer,
            bunchsize,
            bunchsize,
            n_laplace,
            n_laplace,
            self.with_mpi
        )
        
        ####### 5 #######
        
        if rank == 0:
            bunchsize, n_laplace = _bunchsize_determination_driver_rmp3(
                RMP3_XX_5_forloop_P_S_determine_bucket_size_forloop,
                "RMP3_XX_5",
                self.nocc,
                self.nvir,
                self.n_laplace,
                self.nthc_int,
                self.memory
            )
            print("bunchsize = %d, n_laplace = %d for XX 5" % (bunchsize, n_laplace))
        else:
            bunchsize = None
            n_laplace = None
        
        if self.with_mpi:
            bunchsize = bcast(bunchsize)
            n_laplace = bcast(n_laplace)

        mp3_XX_5 = RMP3_XX_5_forloop_P_S_forloop_P_S(
            self.Z,
            self.X_o,
            self.X_v,
            self.tau_o,
            self.tau_v,
            self.buffer,
            bunchsize,
            bunchsize,
            n_laplace,
            n_laplace,
            self.with_mpi
        )
        
        ####### 6 ####### 

        if rank == 0:
            bunchsize, n_laplace = _bunchsize_determination_driver_rmp3(
                RMP3_XX_6_forloop_P_S_determine_bucket_size_forloop,
                "RMP3_XX_6",
                self.nocc,
                self.nvir,
                self.n_laplace,
                self.nthc_int,
                self.memory
            )
            print("bunchsize = %d, n_laplace = %d for XX 6" % (bunchsize, n_laplace))
        else:
            bunchsize = None
            n_laplace = None
        
        if self.with_mpi:
            bunchsize = bcast(bunchsize)
            n_laplace = bcast(n_laplace)

        mp3_XX_6 = RMP3_XX_6_forloop_P_S_forloop_P_S(
            self.Z,
            self.X_o,
            self.X_v,
            self.tau_o,
            self.tau_v,
            self.buffer,
            bunchsize,
            bunchsize,
            n_laplace,
            n_laplace,
            self.with_mpi
        )
        
        ####### 7 ####### 
        
        if rank == 0:
            bunchsize, n_laplace = _bunchsize_determination_driver_rmp3(
                RMP3_XX_7_forloop_P_R_determine_bucket_size_forloop,
                "RMP3_XX_7",
                self.nocc,
                self.nvir,
                self.n_laplace,
                self.nthc_int,
                self.memory
            )
            print("bunchsize = %d, n_laplace = %d for XX 7" % (bunchsize, n_laplace))
        else:
            bunchsize = None
            n_laplace = None
        
        if self.with_mpi:
            bunchsize = bcast(bunchsize)
            n_laplace = bcast(n_laplace)

        mp3_XX_7 = RMP3_XX_7_forloop_P_R_forloop_P_R(
            self.Z,
            self.X_o,
            self.X_v,
            self.tau_o,
            self.tau_v,
            self.buffer,
            bunchsize,
            bunchsize,
            n_laplace,
            n_laplace,
            self.with_mpi
        )
        
        ####### 8 ####### 

        if rank == 0:
            bunchsize, n_laplace = _bunchsize_determination_driver_rmp3(
                RMP3_XX_8_forloop_P_R_determine_bucket_size_forloop,
                "RMP3_XX_8",
                self.nocc,
                self.nvir,
                self.n_laplace,
                self.nthc_int,
                self.memory
            )
            print("bunchsize = %d, n_laplace = %d for XX 8" % (bunchsize, n_laplace))
        else:
            bunchsize = None
            n_laplace = None
        
        if self.with_mpi:
            bunchsize = bcast(bunchsize)
            n_laplace = bcast(n_laplace)

        mp3_XX_8 = RMP3_XX_8_forloop_P_R_forloop_P_R(
            self.Z,
            self.X_o,
            self.X_v,
            self.tau_o,
            self.tau_v,
            self.buffer,
            bunchsize,
            bunchsize,
            n_laplace,
            n_laplace,
            self.with_mpi
        )
        
        mp3_XX = 2 * (mp3_XX_1 + mp3_XX_2 + mp3_XX_3 + mp3_XX_5 + mp3_XX_8)
        mp3_XX-= (mp3_XX_4 + mp3_XX_6)
        mp3_XX-= 4*mp3_XX_7
        
        t3 = (logger.process_clock(), logger.perf_counter())
        
        if rank == 0:
            print("RMP3_XX = " + str(mp3_XX))
            _benchmark_time(t2, t3, "RMP3_XX_forloop")
            _benchmark_time(t0, t3, "RMP3_forloop")
        
        return mp3_CC+mp3_CX+mp3_XX

    #### driver #### 
    
    def kernel(self, schedule=SCHEDULE_TYPE_FORLOOP, get_memory_only=False):
        if schedule == SCHEDULE_TYPE_OPT_MEM:
            return self._kernel_opt_mem(get_memory_only)
        elif schedule == SCHEDULE_TYPE_FORLOOP:
            return self._kernel_forloop()
        elif schedule == SCHEDULE_TYPE_C:
            return self._kernel_C()
        else:
            raise ValueError("schedule type not recognized")

if __name__ == "__main__":
    
    c = 15 
    N = 2
    
    if rank == 0:
        cell   = pbcgto.Cell()
        boxlen = 3.5668
        cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]]) 
    
        cell.atom = [
                    ['C', (0.     , 0.     , 0.    )],
                    ['C', (0.8917 , 0.8917 , 0.8917)],
                    # ['C', (1.7834 , 1.7834 , 0.    )],
                    # ['C', (2.6751 , 2.6751 , 0.8917)],
                    # ['C', (1.7834 , 0.     , 1.7834)],
                    # ['C', (2.6751 , 0.8917 , 2.6751)],
                    # ['C', (0.     , 1.7834 , 1.7834)],
                    # ['C', (0.8917 , 2.6751 , 2.6751)],
                ] 

        cell.basis   = 'gth-szv'
        cell.pseudo  = 'gth-pade'
        cell.verbose = 4
        cell.ke_cutoff = 128
        cell.max_memory = 800  # 800 Mb
        cell.precision  = 1e-8  # integral precision
        cell.use_particle_mesh_ewald = True 
    
        verbose = 4
    
        prim_cell = build_supercell(cell.atom, cell.a, Ls = [1,1,1], ke_cutoff=cell.ke_cutoff, basis=cell.basis, pseudo=cell.pseudo)   
        #prim_partition = [[0,1,2,3], [4,5,6,7]]
        prim_partition = [[0,1]]
        prim_mesh = prim_cell.mesh
    
        Ls = [1, 1, N]
        Ls = np.array(Ls, dtype=np.int32)
        mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
        mesh = np.array(mesh, dtype=np.int32)
    
        cell, group_partition = build_supercell_with_partition(
                                        cell.atom, cell.a, mesh=mesh, 
                                        Ls=Ls,
                                        basis=cell.basis, 
                                        pseudo=cell.pseudo,
                                        partition=prim_partition, ke_cutoff=cell.ke_cutoff, verbose=verbose) 
    
    ####### isdf MP2 can perform directly! ####### 
    
    import numpy
    from pyscf.pbc import gto, scf, mp    
    
    if rank == 0:
        myisdf = ISDF.PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, aoR_cutoff=1e-8, direct=False, use_occ_RI_K=False)
        myisdf.build_IP_local(c=c, m=5, group=group_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
        myisdf.build_auxiliary_Coulomb(debug=True)
            
        mf_isdf = scf.RHF(cell)
        myisdf.direct_scf = mf_isdf.direct_scf
        mf_isdf.with_df = myisdf
        mf_isdf.max_cycle = 16
        mf_isdf.conv_tol = 1e-8
        mf_isdf.kernel()
     
        isdf_pt = mp.RMP2(mf_isdf)
        isdf_pt.kernel()
    else:
        myisdf = None
        mf_isdf = None
        isdf_pt = None
    
    ####### thc rmp2 #######
    
    if rank == 0:
        _myisdf = ISDF.PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, aoR_cutoff=1e-8, direct=False, use_occ_RI_K=False)
        _myisdf.build_IP_local(c=4, m=5, group=group_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
        X          = _myisdf.aoRg_full() 
    else:
        X = None
    
    thc_rmp3 = THC_RMP3(myisdf, mf_isdf, X, memory=800*1000*1000, with_mpi=True)
    thc_rmp3.kernel(SCHEDULE_TYPE_OPT_MEM, get_memory_only=False)
    thc_rmp3.kernel(SCHEDULE_TYPE_FORLOOP)
    #thc_rmp3.kernel(SCHEDULE_TYPE_C)