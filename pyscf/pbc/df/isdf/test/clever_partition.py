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

from multiprocessing import Pool

def _clever_partition(aoR:np.ndarray, atmID_2_AO, ratio = 0.8):

    natm = len(atmID_2_AO)
    ngrid = aoR.shape[1]

    aoMax = np.zeros((ngrid, natm))

    for atmID, AO_List in enumerate(atmID_2_AO):
        aoR_atm = np.abs(aoR[AO_List, :])
        aoR_atm_max = np.max(aoR_atm, axis=0)
        aoMax[:, atmID] = aoR_atm_max
    
    # for each grid point find the one with the largest value
        
    aoMax_arg = np.argmax(aoMax, axis=1)

    # print(aoMax_arg.shape)

    # print aoMax 
    
    # for id in range(ngrid):
    #     print("id = %d" % (id), end="")
    #     for iatm in range(natm):
    #         print("%15.8e " % aoMax[id, iatm], end="")
    #     print("")
    
    # for each point find the second largest 
        
    aoMax_sorted = np.sort(aoMax, axis=1)

    # for id in range(ngrid):
    #     print("id = %d" % (id), end="")
    #     for iatm in range(natm):
    #         print("%15.8e " % aoMax_sorted[id, iatm], end="")
    #     print("")

    pair_grid = 0

    partition_region = {

    }

    for x in range(natm):
        partition_region[(x)] = []
        for y in range(x):
            partition_region[(x, y)] = []

    for id in range(ngrid):
        second_largest = aoMax_sorted[id, -2]
        largest = aoMax_sorted[id, -1]
        if second_largest > largest * ratio:
        # find the loc 
            loc = np.where(aoMax[id, :] == second_largest)[0]
            # print("id = %d, loc = %s" % (id, loc))

            if len(loc) > 1:
                arg1 = loc[0]
                arg2 = loc[1]
                if arg1 < arg2:
                    arg1, arg2 = arg2, arg1
                partition_region[(arg1, arg2)].append(id)
            else:

                loc = loc[0]
                pair_grid += 1
                arg1 = aoMax_arg[id]
                arg2 = loc
                # print(aoMax[id, :])
                if arg1 < arg2:
                    arg1, arg2 = arg2, arg1
                partition_region[(arg1, arg2)].append(id)
        else:
            argmax = aoMax_arg[id]
            partition_region[(argmax)].append(id)
    
    print("pair_grid = %d" % pair_grid)
    for key in partition_region.keys():
        print("len(partition_region[%s]) = %d" % (str(key), len(partition_region[key])))
        partition_region[key] = np.asarray(partition_region[key], dtype=np.int32)
    return partition_region

if __name__ == "__main__":

    cell   = pbcgto.Cell()
    boxlen = 3.5668
    cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])

    cell.atom = '''
                   C     0.      0.      0.
                   C     0.8917  0.8917  0.8917
                   C     1.7834  1.7834  0.
                   C     2.6751  2.6751  0.8917
                   C     1.7834  0.      1.7834
                   C     2.6751  0.8917  2.6751
                   C     0.      1.7834  1.7834
                   C     0.8917  2.6751  2.6751
                '''

    cell.basis   = 'gth-szv'
    cell.pseudo  = 'gth-pade'
    cell.verbose = 4

    # cell.ke_cutoff  = 100   # kinetic energy cutoff in a.u.
    cell.ke_cutoff = 256
    cell.max_memory = 800  # 800 Mb
    cell.precision  = 1e-8  # integral precision
    cell.use_particle_mesh_ewald = True

    cell.build()

    cell = tools.super_cell(cell, [1, 1, 1])

    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2, _eval_rhoG

    df_tmp = MultiGridFFTDF2(cell)

    grids  = df_tmp.grids
    coords = np.asarray(grids.coords).reshape(-1,3)
    mesh   = grids.mesh
    ngrids = np.prod(mesh)
    assert ngrids == coords.shape[0]

    aoR   = df_tmp._numint.eval_ao(cell, coords)[0].T  # the T is important
    aoR  *= np.sqrt(cell.vol / ngrids)

    print("aoR.shape = ", aoR.shape)

    natm = cell.natm

    atmID_2_AO = []

    for i in range(natm):
        atmID_2_AO.append([])
    
    ao_loc = 0
    for i in range(cell._bas.shape[0]):
        atm_id = cell._bas[i, ATOM_OF]
        nctr   = cell._bas[i, NCTR_OF]
        angl   = cell._bas[i, ANG_OF]
        nao_now = nctr * (2 * angl + 1)
        atmID_2_AO[atm_id].extend(list(range(ao_loc, ao_loc+nao_now)))
        ao_loc += nao_now

    print(atmID_2_AO)

    _clever_partition(aoR, atmID_2_AO)