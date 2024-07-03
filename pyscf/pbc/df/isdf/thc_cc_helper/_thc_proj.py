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

from pyscf.lib import logger
from pyscf import lib
libpbc = lib.load_library('libpbc')

def _thc_proj_2(Xo:np.ndarray,
                Xv:np.ndarray,
                tauo:np.ndarray,
                tauv:np.ndarray,
                qr_cutoff = 1e-3):
    
    t1 = (logger.process_clock(), logger.perf_counter())
    
    proj_full = np.einsum("iP,aP,iW,aW->iaPW", Xo, Xv, tauo, tauv)
    
    nocc = Xo.shape[0]
    nvir = Xv.shape[0]
    nthc = Xo.shape[1]
    nlaplace = tauo.shape[1]
    
    nrow = nocc * nvir 
    ncol = nthc * nlaplace
    
    ##### prepare to perform the qr decomposition ##### 
    
    npt_find      = ctypes.c_int(0)
    pivot         = np.arange(grid_ID.shape[0], dtype=np.int32)
    nthread       = lib.num_threads()
    thread_buffer = np.ndarray((nthread+1, grid_ID.shape[0]+1), dtype=np.float64)
    global_buffer = np.ndarray((1, grid_ID.shape[0]), dtype=np.float64)
    R = np.ndarray((nrow, ncol), dtype=np.float64)
    
    
    t2 = (logger.process_clock(), logger.perf_counter()) 
    
    