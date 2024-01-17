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

import sys
import ctypes
import _ctypes

LiH_ATM = '''
Li 0.0   0.0   0.0
Li %f    %f    0.0
Li 0.0   %f    %f
Li %f    0.0   %f
H  0.0   0.0   %f
H  0.0   %f    0.0
H  %f    0.0   0.0
H  %f    %f    %f
'''

if __name__ == "__main__":
    # for length in [1.6,1.8,2.0,2.2,2.4]:
    for length in [4, 4.1, 4.2]:
        cell   = pbcgto.Cell()
        boxlen = length
        cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])

        cell.atom = LiH_ATM % (boxlen/2, boxlen/2, boxlen/2, boxlen/2, boxlen/2, boxlen/2, boxlen/2, boxlen/2, boxlen/2, boxlen/2, boxlen/2, boxlen/2)

        cell.basis   = 'gth-szv'
        cell.pseudo  = 'gth-pade'
        cell.verbose = 4

        cell.ke_cutoff = 128
        cell.max_memory = 800  # 800 Mb
        cell.precision  = 1e-8  # integral precision
        cell.use_particle_mesh_ewald = True

        cell.build()

        from pyscf.pbc import scf

        mf = scf.RHF(cell)
        mf.max_cycle = 100
        mf.conv_tol = 1e-8
        mf.kernel()