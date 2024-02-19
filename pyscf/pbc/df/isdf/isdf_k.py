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

import ctypes

from multiprocessing import Pool

from memory_profiler import profile

libpbc = lib.load_library('libpbc')

from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto

####################### Util Module #######################

def _extract_grid_primitive_cell(cell_a, mesh, Ls, coords):
    """
    Extract the primitive cell grid information from the supercell grid information
    """
    
    #print("In _extract_grid_primitive_cell")
    
    assert cell_a[0, 1] == 0.0
    assert cell_a[0, 2] == 0.0
    assert cell_a[1, 0] == 0.0
    assert cell_a[1, 2] == 0.0
    assert cell_a[2, 0] == 0.0
    assert cell_a[2, 1] == 0.0
    
    ngrids = np.prod(mesh)
    # print("ngrids = ", ngrids)

    assert ngrids == coords.shape[0]
    
    Lx = Ls[0]
    Ly = Ls[1]
    Lz = Ls[2]
    
    # print("Lx = ", Lx)
    # print("Ly = ", Ly)
    # print("Lz = ", Lz)
    
    print("Length supercell x = %15.6f , primitive cell x = %15.6f" % (cell_a[0, 0], cell_a[0, 0] / Lx))
    print("Length supercell y = %15.6f , primitive cell y = %15.6f" % (cell_a[1, 1], cell_a[1, 1] / Ly))
    print("Length supercell z = %15.6f , primitive cell z = %15.6f" % (cell_a[2, 2], cell_a[2, 2] / Lz))
    
    nx, ny, nz = mesh
    
    # print("nx = ", nx)
    # print("ny = ", ny)
    # print("nz = ", nz)
    
    coords = coords.reshape(nx, ny, nz, 3)
    
    assert nx % Lx == 0
    assert ny % Ly == 0
    assert nz % Lz == 0
    
    nx_prim = nx // Lx
    ny_prim = ny // Ly
    nz_prim = nz // Lz
    
    # print("nx_prim = ", nx_prim)
    # print("ny_prim = ", ny_prim)
    # print("nz_prim = ", nz_prim)
    
    ngrids_prim = nx_prim * ny_prim * nz_prim
    
    res_dict = {}
    
    # res = []
        
    prim_grid = coords[:nx_prim, :ny_prim, :nz_prim].reshape(-1, 3)
        
    for ix in range(Lx):
        for iy in range(Ly):
            for iz in range(Lz):
                x_0 = ix * nx_prim
                x_1 = (ix + 1) * nx_prim
                y_0 = iy * ny_prim
                y_1 = (iy + 1) * ny_prim
                z_0 = iz * nz_prim
                z_1 = (iz + 1) * nz_prim
                
                grid_tmp = coords[x_0:x_1, y_0:y_1, z_0:z_1].reshape(-1, 3)
                
                shift_bench = np.zeros((3), dtype=np.float64)
                shift_bench[0] = ix * cell_a[0, 0] / Lx
                shift_bench[1] = iy * cell_a[1, 1] / Ly
                shift_bench[2] = iz * cell_a[2, 2] / Lz
                
                shifts = grid_tmp - prim_grid
                
                # print("shifts = ", shifts)
                # print("shift_bench = ", shift_bench)
                
                for ID in range(shifts.shape[0]):
                    shift = shifts[ID]
                    # print("shift = ", shift)
                    if np.allclose(shift, shift_bench) == False:
                        tmp = shift - shift_bench
                        nx = round (tmp[0] / cell_a[0, 0])
                        ny = round (tmp[1] / cell_a[1, 1])
                        nz = round (tmp[2] / cell_a[2, 2])
                        # print(tmp)
                        # print(nx, ny, nz)
                        assert np.allclose(tmp[0], nx * cell_a[0, 0])
                        assert np.allclose(tmp[1], ny * cell_a[1, 1])
                        assert np.allclose(tmp[2], nz * cell_a[2, 2])
                        # grid_tmp[ID] = prim_grid[ID] + shift_bench, do not shift to avoid numerical error

                # res.append(grid_tmp)
                res_dict[(nx, ny, nz)] = grid_tmp
    
    return res_dict

def _split_partition(Voroini_partition, mesh, Ls):
    ngrids = np.prod(mesh)
    assert ngrids == coords.shape[0]
    
    Lx = Ls[0]
    Ly = Ls[1]
    Lz = Ls[2]

    nx, ny, nz = mesh
    
    Voroini_partition_reshaped = Voroini_partition.reshape(nx, ny, nz)
        
    assert nx % Lx == 0
    assert ny % Ly == 0
    assert nz % Lz == 0
    
    nx_prim = nx // Lx
    ny_prim = ny // Ly
    nz_prim = nz // Lz
    
    ngrids_prim = nx_prim * ny_prim * nz_prim
    
    res_dict = {}
    prim_grid = Voroini_partition_reshaped[:nx_prim, :ny_prim, :nz_prim].reshape(-1, 3)
        
    for ix in range(Lx):
        for iy in range(Ly):
            for iz in range(Lz):
                x_0 = ix * nx_prim
                x_1 = (ix + 1) * nx_prim
                y_0 = iy * ny_prim
                y_1 = (iy + 1) * ny_prim
                z_0 = iz * nz_prim
                z_1 = (iz + 1) * nz_prim
                
                grid_tmp               = Voroini_partition_reshaped[x_0:x_1, y_0:y_1, z_0:z_1].reshape(-1)
                res_dict[(nx, ny, nz)] = grid_tmp
    
    return res_dict
    
def build_supercell(prim_atm, 
                    prim_a, 
                    mesh=None, 
                    Ls = [1,1,1], 
                    basis='gth-dzvp', 
                    pseudo='gth-pade', 
                    ke_cutoff=70, 
                    max_memory=2000, 
                    precision=1e-8,
                    use_particle_mesh_ewald=True):
    
    Cell = pbcgto.Cell()
    
    assert prim_a[0, 1] == 0.0
    assert prim_a[0, 2] == 0.0
    assert prim_a[1, 0] == 0.0
    assert prim_a[1, 2] == 0.0
    assert prim_a[2, 0] == 0.0
    assert prim_a[2, 1] == 0.0
    
    Supercell_a = prim_a * np.array(Ls)
    Cell.a = Supercell_a
    
    atm = []
    
    for ix in range(Ls[0]):
        for iy in range(Ls[1]):
            for iz in range(Ls[2]):
                shift = [ix * prim_a[0, 0], iy * prim_a[1, 1], iz * prim_a[2, 2]]
                for atom in prim_atm:
                    atm.append([atom[0], (atom[1][0] + shift[0], atom[1][1] + shift[1], atom[1][2] + shift[2])])
    
    Cell.atom = atm
    Cell.basis = basis
    Cell.pseudo = pseudo
    Cell.ke_cutoff = ke_cutoff
    Cell.max_memory = max_memory
    Cell.precision = precision
    Cell.use_particle_mesh_ewald = use_particle_mesh_ewald
    Cell.verbose = 4
    Cell.unit = 'angstorm'
    
    Cell.build(mesh=mesh)
    
    return Cell

####################### Select IP #######################

def _get_possible_IP(pbc_isdf_info:PBC_ISDF_Info, Ls, coords):
    cell = pbc_isdf_info.cell
    ncell = np.prod(Ls)
    
    mesh = cell.mesh
    mesh_prim = np.array(mesh) // np.array(Ls)
    ngrid_prim = np.prod(mesh_prim)
    
    natm = cell.natm
    natm_in_partition = natm // ncell
        
    partition = pbc_isdf_info.partition
    partition_reshaped = partition.reshape(mesh)
    
    possible_primID_selected = np.zeros((ngrid_prim), dtype=np.int32)
    
    tmp = []
    
    for atmid in range(8): ### ????
        # print("atmid = ", atmid)
        where = np.where(partition == atmid)
        # print(where)
        for grid_id in where[0]:
            pnt_id = (grid_id // (mesh[1] * mesh[2]), (grid_id // mesh[2]) % mesh[1], grid_id % mesh[2])
            box_id = (pnt_id[0] // mesh_prim[0], pnt_id[1] // mesh_prim[1], pnt_id[2] // mesh_prim[2])
            pnt_prim_id = (pnt_id[0] % mesh_prim[0], pnt_id[1] % mesh_prim[1], pnt_id[2] % mesh_prim[2])
            pnt_prim_ravel_id = pnt_prim_id[0] * mesh_prim[1] * mesh_prim[2] + pnt_prim_id[1] * mesh_prim[2] + pnt_prim_id[2]
            if box_id[0] == 0 and box_id[1] == 0 and box_id[2] == 0:
                possible_primID_selected[pnt_prim_ravel_id] = 1
            tmp.append((grid_id, box_id, pnt_prim_ravel_id))
    
    # res = []
    
    possible_grid_ID = []
    
    for data in tmp:
        grid_id, box_id, pnt_prim_ravel_id = data
        
        if box_id[0] == 0 and box_id[1] == 0 and box_id[2] == 0:
            # res.append(coords[grid_id])
            possible_grid_ID.append(grid_id)
        else:
            if possible_primID_selected[pnt_prim_ravel_id] == 0: # not selected in the first box
                # res.append(coords[grid_id])
                possible_grid_ID.append(grid_id)
    
    possible_grid_ID.sort()
    print("possible_grid_ID = ", possible_grid_ID)
    
    return possible_grid_ID, np.array(coords[possible_grid_ID])

  
####################### build aux basis #######################

## Incore 



## Outcore

    
class PBC_ISDF_Info_kSym(ISDF_outcore.PBC_ISDF_Info_outcore):
    def __init__(self, mol:Cell, max_buf_memory:int, Ls=[1,1,1], outcore=True, with_robust_fitting=True, aoR=None):
        
        super().__init__(mol=mol, max_buf_memory=max_buf_memory, outcore=outcore, with_robust_fitting=with_robust_fitting, aoR=aoR)
        
        assert with_robust_fitting == False
        assert self.mesh[0] % Ls[0] == 0
        assert self.mesh[1] % Ls[1] == 0
        assert self.mesh[2] % Ls[2] == 0
        
        self.Ls = Ls
        
        if self.coords is None :
            from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2
            df_tmp = MultiGridFFTDF2(self.cell)
            self.coords = np.asarray(df_tmp.grids.coords).reshape(-1,3)

        print("self.cell.lattice_vectors = ", self.cell.lattice_vectors())
        self.ordered_grid_coords = _extract_grid_primitive_cell(self.cell.lattice_vectors(), self.mesh, self.Ls, self.coords)
    
    ################ select IP ################
    
    def select_IP(self, c:int, m:int):
        first_natm = self.cell.natm // np.prod(self.Ls)
        IP_GlobalID = ISDF._select_IP_direct(self, c, m, first_natm, True) # we do not have to perform selection IP over the whole supercell ! 
        
        # get primID
        
        mesh = self.cell.mesh
        mesh_prim = np.array(mesh) // np.array(self.Ls)
        ngrid_prim = np.prod(mesh_prim)
                
        possible_grid_ID = []
    
        for grid_id in IP_GlobalID:
            pnt_id = (grid_id // (mesh[1] * mesh[2]), (grid_id // mesh[2]) % mesh[1], grid_id % mesh[2])
            box_id = (pnt_id[0] // mesh_prim[0], pnt_id[1] // mesh_prim[1], pnt_id[2] // mesh_prim[2])
            pnt_prim_id = (pnt_id[0] % mesh_prim[0], pnt_id[1] % mesh_prim[1], pnt_id[2] % mesh_prim[2])
            pnt_prim_ravel_id = pnt_prim_id[0] * mesh_prim[1] * mesh_prim[2] + pnt_prim_id[1] * mesh_prim[2] + pnt_prim_id[2]
            # print("grid_id = %d, pnt_id = %s, box_id = %s, pnt_prim_id = %s" % (grid_id, pnt_id, box_id, pnt_prim_id))
            possible_grid_ID.append(pnt_prim_ravel_id)

        possible_grid_ID = list(set(possible_grid_ID))
        possible_grid_ID.sort()
        
        print("nIP = ", len(possible_grid_ID))
        
        return np.array(possible_grid_ID, dtype=np.int32)
    
    ################ construct aux basis ################
    
    
    ################ construct W ################
    
    ################ get jk ################

C = 2
M = 5

if __name__ == "__main__":
    
    boxlen = 3.5668
    prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    
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
    
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=8)
    prim_mesh = prim_cell.mesh
    print("prim_mesh = ", prim_mesh)
    
    Ls = [3,3,3]
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    
    cell = build_supercell(atm, prim_a, Ls = Ls, ke_cutoff=4, mesh=mesh)
    
    for i in range(cell.natm):
        print('%s %s  charge %f  xyz %s' % (cell.atom_symbol(i),
                                        cell.atom_pure_symbol(i),
                                        cell.atom_charge(i),
                                        cell.atom_coord(i)))

    print("Atoms' charges in a vector\n%s" % cell.atom_charges())
    print("Atoms' coordinates in an array\n%s" % cell.atom_coords())
    
    
    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2

    df_tmp = MultiGridFFTDF2(cell)

    grids  = df_tmp.grids
    coords = np.asarray(grids.coords).reshape(-1,3)
    nx = grids.mesh[0]

    mesh   = grids.mesh
    ngrids = np.prod(mesh)
    assert ngrids == coords.shape[0]

    aoR   = df_tmp._numint.eval_ao(cell, coords)[0].T  # the T is important
    aoR  *= np.sqrt(cell.vol / ngrids)

    print("aoR.shape = ", aoR.shape)

    pbc_isdf_info = PBC_ISDF_Info(cell, aoR)
    _, Possible_IP_coords = _get_possible_IP(pbc_isdf_info, Ls, coords)
    print("Possible_IP_coords = ", Possible_IP_coords)
    
    ############ construct ISDF object ############
    
    pbc_isdf_info_ksym = PBC_ISDF_Info_kSym(cell, 20 * 1000 * 1000, Ls=Ls, outcore=True, with_robust_fitting=False, aoR=None)
    
    ############ test select IP ############
    
    possible_IP = pbc_isdf_info_ksym.select_IP(C, M)
    print("possible_IP = ", possible_IP)