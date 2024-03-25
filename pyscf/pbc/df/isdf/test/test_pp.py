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
from pyscf.pbc.dft import multigrid

import ctypes
from multiprocessing import Pool
from memory_profiler import profile

if __name__ == '__main__':
    
    # from pyscf.pbc.df.isdf import isdf_fast_mpi

    cell   = pbcgto.Cell()
    
    boxlen = 4.2
    cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    cell.atom = '''
Li 0.0   0.0   0.0
Li 2.1   2.1   0.0
Li 0.0   2.1   2.1
Li 2.1   0.0   2.1
H  0.0   0.0   2.1
H  0.0   2.1   0.0
H  2.1   0.0   0.0
H  2.1   2.1   2.1
'''

    cell.basis   = 'gth-dzvp'
    # cell.basis   = 'gth-tzvp'
    cell.pseudo  = 'gth-pade'

    # cell.ke_cutoff  = 32   # kinetic energy cutoff in a.u.
    cell.ke_cutoff = 128
    cell.max_memory = 800  # 800 Mb
    cell.precision  = 1e-8  # integral precision
    cell.use_particle_mesh_ewald = True

    cell.build()

    cell = tools.super_cell(cell, [1, 1, 2])

    from pyscf.pbc import scf

    mf = scf.RHF(cell)
    # mf.with_df = pbc_isdf_info
    mf.max_cycle = 100
    mf.conv_tol = 1e-7

    pp1 = mf.with_df.get_pp()
    
    print(pp1)  
    
    mf.with_df = multigrid.MultiGridFFTDF2(cell)
    v_pp_loc2_nl = mf.with_df.get_pp(max_memory=cell.max_memory)
    v_pp_loc1_G = mf.with_df.vpplocG_part1
    v_pp_loc1 = multigrid.multigrid_pair._get_j_pass2(mf.with_df, v_pp_loc1_G)
    v_pp = v_pp_loc1 + v_pp_loc2_nl
    
    print(v_pp)
    
    print("pp1.shape = ", pp1.shape)
    print("v_pp.shape = ", v_pp.shape)
    
    diff = np.linalg.norm(v_pp - pp1)
    
    print("diff = ", diff/np.sqrt(v_pp.size))