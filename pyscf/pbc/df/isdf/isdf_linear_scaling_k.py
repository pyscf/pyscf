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

import ctypes
from multiprocessing import Pool
from memory_profiler import profile
libpbc = lib.load_library('libpbc')
from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto 

import pyscf.pbc.df.isdf.isdf_linear_scaling as ISDF_LinearScaling

class PBC_ISDF_Info_Quad_K(ISDF_LinearScaling.PBC_ISDF_Info_Quad):
    
    # Quad stands for quadratic scaling
    
    def __init__(self, mol:Cell, 
                 # aoR: np.ndarray = None,
                 with_robust_fitting=True,
                 Ls=None,
                 # get_partition=True,
                 verbose = 1,
                 rela_cutoff_QRCP = None,
                 aoR_cutoff = 1e-8,
                 direct=False
                 ):
        
        super().__init__(mol, with_robust_fitting, None, verbose, rela_cutoff_QRCP, aoR_cutoff, direct)
        
        self.Ls    = Ls
        self.kmesh = Ls
        
        assert self.mesh[0] % Ls[0] == 0
        assert self.mesh[1] % Ls[1] == 0
        assert self.mesh[2] % Ls[2] == 0
        
        #### information relating primitive cell and supercell
        
        self.meshPrim = np.array(self.mesh) // np.array(self.Ls)
        self.natm     = self.cell.natm
        self.natmPrim = self.cell.natm // np.prod(self.Ls)
        
        #### information dealing grids 