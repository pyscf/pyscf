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

import numpy as np
from pyscf import lib
from pyscf.pbc.lib.kpts import KPoints
from pyscf.gto.mole import *

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
    
    # print("Length supercell x = %15.6f , primitive cell x = %15.6f" % (cell_a[0, 0], cell_a[0, 0] / Lx))
    # print("Length supercell y = %15.6f , primitive cell y = %15.6f" % (cell_a[1, 1], cell_a[1, 1] / Ly))
    # print("Length supercell z = %15.6f , primitive cell z = %15.6f" % (cell_a[2, 2], cell_a[2, 2] / Lz))
    
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
    
    res = []
        
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

                res.append(grid_tmp)
                res_dict[(ix, iy, iz)] = grid_tmp
    res = np.array(res).reshape(-1, 3)
    return res, res_dict

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

def _RowCol_FFT_bench(input, Ls, inv=False, TransBra = True, TransKet = True):
    """
    A is a 3D array, (nbra, nket, ngrid_prim)
    """
    
    A = input
    ncell = np.prod(Ls)
    
    if TransKet:
        assert A.shape[1] % ncell == 0
    if TransBra:
        assert A.shape[0] % ncell == 0
    
    # print("A.shape = ", A.shape)
    # print("Ls = ", Ls)
    
    NPOINT_KET = A.shape[1] // ncell
    
    if TransKet:
        A = A.reshape(A.shape[0], -1, NPOINT_KET) # nbra, nBox, NPOINT
        A = A.transpose(0, 2, 1)                  # nbra, NPOINT, nBox
        shape_tmp = A.shape
        A = A.reshape(A.shape[0] * NPOINT_KET, *Ls)
        # perform 3d fft 
        if inv:
            A = np.fft.ifftn(A, axes=(1, 2, 3))
        else:
            A = np.fft.fftn(A, axes=(1, 2, 3))
        A = A.reshape(shape_tmp)
        A = A.transpose(0, 2, 1)
        A = A.reshape(A.shape[0], -1)
        print("finish transform ket")
    # transform bra
    NPOINT_BRA = A.shape[0] // ncell
    if TransBra:
        A = A.reshape(-1, NPOINT_BRA, A.shape[1])
        A = A.transpose(1, 2, 0)
        shape_tmp = A.shape
        A = A.reshape(-1, *Ls)
        if inv:
            A = np.fft.fftn(A, axes=(1, 2, 3))
        else:
            A = np.fft.ifftn(A, axes=(1, 2, 3))
        A = A.reshape(shape_tmp)
        A = A.transpose(2, 0, 1)
        A = A.reshape(-1, A.shape[2])
        print("finish transform bra")
    # print(A[:NPOINT, :NPOINT])
    return A

def _RowCol_FFT_ColFull_bench(input, Ls, mesh):
    """
    A is a 3D array, (nbra, nket, ngrid_prim)
    """
    A = input
    ncell = np.prod(Ls)
    nGrids = np.prod(mesh)
    assert A.shape[1] == nGrids
    assert A.shape[0] % ncell == 0
    A = A.reshape(A.shape[0], *mesh)
    # perform 3d fft 
    A = np.fft.fftn(A, axes=(1, 2, 3))
    A = A.reshape(A.shape[0], -1)
    print("finish transform ket")
    # transform bra
    NPOINT_BRA = A.shape[0] // ncell
    A = A.reshape(-1, NPOINT_BRA, A.shape[1])
    A = A.transpose(1, 2, 0)
    shape_tmp = A.shape
    A = A.reshape(-1, *Ls)
    A = np.fft.ifftn(A, axes=(1, 2, 3))
    A = A.reshape(shape_tmp)
    A = A.transpose(2, 0, 1)
    A = A.reshape(-1, A.shape[2])
    print("finish transform bra")
    return A

def _kmesh_to_Kpoints(cell, mesh):
    
    from pyscf.pbc.lib.kpts import KPoints 
    
    kpts = []
    
    for i in range(mesh[0]):
        for j in range(mesh[1]):
            for k in range(mesh[2]):
                kpts.append([1.0/float(mesh[0]) * float(i), 
                             1.0/float(mesh[1]) * float(j), 
                             1.0/float(mesh[2]) * float(k)])
    
    kpts = np.array(kpts)
    
    return KPoints(cell, kpts) 