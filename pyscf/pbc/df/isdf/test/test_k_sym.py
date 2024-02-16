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

import ctypes

from multiprocessing import Pool

from memory_profiler import profile

libpbc = lib.load_library('libpbc')

BASIS_CUTOFF               = 1e-18  # too small may lead to numerical instability
CRITERION_CALL_PARALLEL_QR = 256

from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto

def _extract_grid_primitive_cell(cell_a, mesh, Ls, coords):
    """
    Extract the primitive cell grid information from the supercell grid information
    """
    
    print("In _extract_grid_primitive_cell")
    
    assert cell_a[0, 1] == 0.0
    assert cell_a[0, 2] == 0.0
    assert cell_a[1, 0] == 0.0
    assert cell_a[1, 2] == 0.0
    assert cell_a[2, 0] == 0.0
    assert cell_a[2, 1] == 0.0
    
    ngrids = np.prod(mesh)
    print("ngrids = ", ngrids)

    assert ngrids == coords.shape[0]
    
    Lx = Ls[0]
    Ly = Ls[1]
    Lz = Ls[2]
    
    print("Lx = ", Lx)
    print("Ly = ", Ly)
    print("Lz = ", Lz)
    
    print("Length supercell x = %15.6f , primitive cell x = %15.6f" % (cell_a[0, 0], cell_a[0, 0] / Lx))
    print("Length supercell y = %15.6f , primitive cell y = %15.6f" % (cell_a[1, 1], cell_a[1, 1] / Ly))
    print("Length supercell z = %15.6f , primitive cell z = %15.6f" % (cell_a[2, 2], cell_a[2, 2] / Lz))
    
    nx, ny, nz = mesh
    
    print("nx = ", nx)
    print("ny = ", ny)
    print("nz = ", nz)
    
    coords = coords.reshape(nx, ny, nz, 3)
    
    assert nx % Lx == 0
    assert ny % Ly == 0
    assert nz % Lz == 0
    
    nx_prim = nx // Lx
    ny_prim = ny // Ly
    nz_prim = nz // Lz
    
    print("nx_prim = ", nx_prim)
    print("ny_prim = ", ny_prim)
    print("nz_prim = ", nz_prim)
    
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
                print("shift_bench = ", shift_bench)
                
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
                res_dict[(nx, ny, nz)] = grid_tmp
    
    return res, res_dict
                

if __name__ == '__main__':

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

    cell.basis   = 'gth-dzvp'
    # cell.basis   = 'gth-tzvp'
    cell.pseudo  = 'gth-pade'
    cell.verbose = 4

    # cell.ke_cutoff  = 256   # kinetic energy cutoff in a.u.
    cell.ke_cutoff = 4
    cell.max_memory = 800  # 800 Mb
    cell.precision  = 1e-8  # integral precision
    cell.use_particle_mesh_ewald = True

    cell.build()

    Ls   = [2, 3, 4]
    cell = tools.super_cell(cell, Ls)

    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2, _eval_rhoG

    df_tmp = MultiGridFFTDF2(cell)

    grids  = df_tmp.grids
    coords = np.asarray(grids.coords).reshape(-1,3)
    nx = grids.mesh[0]

    # for i in range(coords.shape[0]):
    #     print(coords[i])
    # exit(1)

    mesh   = grids.mesh
    ngrids = np.prod(mesh)
    assert ngrids == coords.shape[0]
    
    print("ngrids = ", ngrids)
    print("mesh   = ", mesh)
    print("cell.a = ", cell.a)
    
    grid_primitive, _ = _extract_grid_primitive_cell(cell.a, mesh, Ls, coords) 
    
    ngrid_prim = grid_primitive[0].size // 3
    print(ngrid_prim)
    
    # for i in range(len(grid_primitive)):
    #     print(grid_primitive[i].shape)
    #     print(grid_primitive[i])
    
    for i in range(len(grid_primitive)):
        # print("Line %d" % i)
        for j in range(len(grid_primitive)):
            # for k in range(ngrid_prim):
                # print(grid_primitive[i][k], grid_primitive[j][k])
                # print(grid_primitive[j][k] - grid_primitive[i][k])
            shift = grid_primitive[i] - grid_primitive[j]
            # print(shift[0])
            # print(shift)
            ## assert all the elements in shift is equal 
            # for k in range(3):
            #     assert np.allclose(shift[:,k], shift[0,k])
    
    NPOINT = 4
    
    # generate NPOINT random ints (no repeated) from [0, ngrid_prim)
    
    idx = np.random.choice(ngrid_prim, NPOINT, replace=False)
    idx.sort()
    print(idx)
    coord_ordered = []
    IP_coord = []
    for i in range(len(grid_primitive)):
        IP_coord.append(grid_primitive[i][idx])
        coord_ordered.append(grid_primitive[i])
    IP_coord = np.asarray(IP_coord)
    coord_ordered = np.asarray(coord_ordered)
    print(IP_coord.shape)
    print(coord_ordered.shape)
    IP_coord = IP_coord.reshape(-1,3)
    coord_ordered = coord_ordered.reshape(-1,3)
    
    # calculate aoRg 
    aoRg   = df_tmp._numint.eval_ao(cell, IP_coord)[0].T  # the T is important
    aoRg  *= np.sqrt(cell.vol / ngrids)
    aoR    = df_tmp._numint.eval_ao(cell, coord_ordered)[0].T
    aoR   *= np.sqrt(cell.vol / ngrids)
    
    A = np.asarray(lib.dot(aoRg.T, aoRg), order='C')
    print(A.shape)
    A = A ** 2
    
    print(A.shape)
    ncell = np.prod(Ls)
    
    # for i in range(ncell):
    #     for j in range(ncell):
    #         b_begin = i * NPOINT
    #         b_end   = (i + 1) * NPOINT
    #         k_begin = j * NPOINT
    #         k_end   = (j + 1) * NPOINT
    #         mat     = A[b_begin:b_end, k_begin:k_end]
    #         # print(mat)
    #         # print(mat- mat.T)
    #         assert np.allclose(mat, mat.T)  # fail not the case

    B = np.asarray(lib.dot(aoRg.T, aoR), order='C')
    B = B ** 2
    
    # transform ket 
    A = A.reshape(A.shape[0], -1, NPOINT) # nbra, nBox, NPOINT
    A = A.transpose(0, 2, 1)              # nbra, NPOINT, nBox
    shape_tmp = A.shape
    A = A.reshape(A.shape[0] * NPOINT, *Ls)
    # perform 3d fft 
    A = np.fft.fftn(A, axes=(1, 2, 3))
    A = A.reshape(shape_tmp)
    A = A.transpose(0, 2, 1)
    A = A.reshape(A.shape[0], -1)
    print("finish transform ket")
    # transform bra
    A = A.reshape(-1,NPOINT, A.shape[1])
    A = A.transpose(1, 2, 0)
    shape_tmp = A.shape
    A = A.reshape(-1, *Ls)
    A = np.fft.ifftn(A, axes=(1, 2, 3))
    A = A.reshape(shape_tmp)
    A = A.transpose(2, 0, 1)
    A = A.reshape(-1, A.shape[2])
    print("finish transform bra")
    # print(A[:NPOINT, :NPOINT])
    
    # exit(1)
    
    ### transform B 
    
    B = B.reshape(B.shape[0], -1, ngrid_prim) # nbra, nBox, NPOINT
    B = B.transpose(0, 2, 1)              # nbra, NPOINT, nBox
    shape_tmp = B.shape
    B = B.reshape(B.shape[0] * ngrid_prim, *Ls)
    # perform 3d fft
    B = np.fft.fftn(B, axes=(1, 2, 3))
    B = B.reshape(shape_tmp)
    B = B.transpose(0, 2, 1)
    B = B.reshape(B.shape[0], -1)
    print("finish transform ket")
    # transform bra
    B = B.reshape(-1,NPOINT, B.shape[1])
    B = B.transpose(1, 2, 0)
    shape_tmp = B.shape
    B = B.reshape(-1, *Ls)
    B = np.fft.ifftn(B, axes=(1, 2, 3))
    B = B.reshape(shape_tmp)
    B = B.transpose(2, 0, 1)
    B = B.reshape(-1, B.shape[2])
    print("finish transform bra")
    
    for i in range(ncell):
        b_begin = i * NPOINT
        b_end   = (i + 1) * NPOINT
        k_begin = i * NPOINT
        k_end   = (i + 1) * NPOINT
        mat     = A[b_begin:b_end, k_begin:k_end]

        mat_before = A[b_begin:b_end, :k_begin]
        assert np.allclose(mat_before, 0.0)  # block diagonal 
        mat_after = A[b_begin:b_end, k_end:]
        assert np.allclose(mat_after, 0.0)   # block diagonal 
    
        # test B 
        
        k_begin = i * ngrid_prim
        k_end   = (i + 1) * ngrid_prim
        mat     = B[b_begin:b_end, k_begin:k_end]
        
        mat_before = B[b_begin:b_end, :k_begin]
        assert np.allclose(mat_before, 0.0) # block diagonal
        mat_after = B[b_begin:b_end, k_end:]
        assert np.allclose(mat_after, 0.0)  # block diagonal
    
    ## a much more efficient way to construct A, B , only half FFT is needed
    
    A_test = np.zeros_like(A)
    B_test = np.zeros_like(B)
    
    # print(A[:NPOINT, :NPOINT])
    
    A2 = np.asarray(lib.dot(aoRg.T, aoRg), order='C')
    print(A2.shape)
    A2 = A2 ** 2
    
    A_tmp = A2[:NPOINT, :].copy()
    A_tmp = A_tmp.reshape(A_tmp.shape[0], -1, NPOINT) # nbra, nBox, NPOINT
    print(A_tmp.shape)
    A_tmp = A_tmp.transpose(0, 2, 1)              # nbra, NPOINT, nBox
    shape_tmp = A_tmp.shape
    A_tmp = A_tmp.reshape(A_tmp.shape[0] * NPOINT, *Ls)
    # perform 3d fft 
    A_tmp = np.fft.fftn(A_tmp, axes=(1, 2, 3))
    A_tmp = A_tmp.reshape(shape_tmp)
    A_tmp = A_tmp.transpose(0, 2, 1)
    # A_tmp = A_tmp.transpose(2, 0, 1)
    A_tmp = A_tmp.reshape(A_tmp.shape[0], -1)
    # A_tmp *= np.sqrt(np.prod(Ls))
    
    B2 = np.asarray(lib.dot(aoRg.T, aoR), order='C')
    print(B2.shape)
    B2 = B2 ** 2
    
    B_tmp = B2[:NPOINT, :].copy()
    B_tmp = B_tmp.reshape(B_tmp.shape[0], -1, ngrid_prim) # nbra, nBox, NPOINT
    B_tmp = B_tmp.transpose(0, 2, 1)              # nbra, NPOINT, nBox
    shape_tmp = B_tmp.shape
    B_tmp = B_tmp.reshape(B_tmp.shape[0] * ngrid_prim, *Ls)
    # perform 3d fft
    B_tmp = np.fft.fftn(B_tmp, axes=(1, 2, 3))
    B_tmp = B_tmp.reshape(shape_tmp)
    B_tmp = B_tmp.transpose(0, 2, 1)
    B_tmp = B_tmp.reshape(B_tmp.shape[0], -1)
    
    
    # diagonal block
    
    for i in range(ncell):
        
        # print("Compr cell %d" % i)
        
        b_begin = i * NPOINT
        b_end   = (i + 1) * NPOINT
        k_begin = i * NPOINT
        k_end   = (i + 1) * NPOINT
        mat     = A_tmp[:, k_begin:k_end]
        # mat     = A_tmp[i]
        # print(mat)
        # print(A[b_begin:b_end, k_begin:k_end])
        assert np.allclose(mat, A[b_begin:b_end, k_begin:k_end])
        # mat2 = A[b_begin:b_end, k_begin:k_end]
        # print(mat/mat2)
        
        k_begin = i * ngrid_prim
        k_end   = (i + 1) * ngrid_prim
        mat     = B_tmp[:, k_begin:k_end]
        assert np.allclose(mat, B[b_begin:b_end, k_begin:k_end])