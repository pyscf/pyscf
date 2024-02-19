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
                
# the following subroutine are all testing functions

def _RowCol_FFT_bench(input, Ls):
    """
    A is a 3D array, (nbra, nket, ngrid_prim)
    """
    A = input
    ncell = np.prod(Ls)
    assert A.shape[1] % ncell == 0
    assert A.shape[0] % ncell == 0
    NPOINT_KET = A.shape[1] // ncell
    A = A.reshape(A.shape[0], -1, NPOINT_KET) # nbra, nBox, NPOINT
    A = A.transpose(0, 2, 1)                  # nbra, NPOINT, nBox
    shape_tmp = A.shape
    A = A.reshape(A.shape[0] * NPOINT_KET, *Ls)
    # perform 3d fft 
    A = np.fft.fftn(A, axes=(1, 2, 3))
    A = A.reshape(shape_tmp)
    A = A.transpose(0, 2, 1)
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
    # print(A[:NPOINT, :NPOINT])
    return A

def _RowCol_FFT_Fast(input, Ls):
    A = input
    ncell = np.prod(Ls)
    assert A.shape[1] % ncell == 0
    
    NPOINT_KET = A.shape[1] // ncell
    A = A.reshape(A.shape[0], -1, NPOINT_KET) # nbra, nBox, NPOINT
    A = A.transpose(0, 2, 1)                  # nbra, NPOINT, nBox
    shape_tmp = A.shape
    A = A.reshape(A.shape[0] * NPOINT_KET, *Ls)
    # perform 3d fft
    A = np.fft.fftn(A, axes=(1, 2, 3))
    A = A.reshape(shape_tmp)
    A = A.transpose(0, 2, 1)
    A = A.reshape(A.shape[0], -1)
    
    return A

def _RowCol_rFFT_Fast(input, Ls):
    A = input
    ncell = np.prod(Ls)
    assert A.shape[1] % ncell == 0
    
    NPOINT_KET = A.shape[1] // ncell
    A = A.reshape(A.shape[0], -1, NPOINT_KET) # nbra, nBox, NPOINT
    A = A.transpose(0, 2, 1)                  # nbra, NPOINT, nBox
    shape_tmp = A.shape
    A = A.reshape(A.shape[0] * NPOINT_KET, *Ls)
    # perform 3d fft
    # print(A.shape)
    A = np.fft.rfftn(A, axes=(1, 2, 3)) # the input is real 
    # print(A.shape)
    nComplex = np.prod([Ls[0], Ls[1], Ls[2]//2+1])
    A = A.reshape((shape_tmp[0], shape_tmp[1], nComplex))
    A = A.transpose(0, 2, 1)
    A = A.reshape(A.shape[0], -1)
    
    return A

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
    
    NPOINT = 8
    
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
    
    B = np.asarray(lib.dot(aoRg.T, aoR), order='C')
    B = B ** 2
    
    A = _RowCol_FFT_bench(A, Ls)
    B = _RowCol_FFT_bench(B, Ls)
    
    for i in range(ncell):
        b_begin = i * NPOINT
        b_end   = (i + 1) * NPOINT
        k_begin = i * NPOINT
        k_end   = (i + 1) * NPOINT
        mat     = A[b_begin:b_end, k_begin:k_end]

        assert np.allclose(mat, mat.T.conj()) # block diagonal， A is Hermitian Conjugate and positive definite

        mat_before = A[b_begin:b_end, :k_begin]
        assert np.allclose(mat_before, 0.0)  # block diagonal 
        mat_after  = A[b_begin:b_end, k_end:]
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
    A_tmp = _RowCol_FFT_Fast(A_tmp, Ls)
    
    B2 = np.asarray(lib.dot(aoRg.T, aoR), order='C')
    print(B2.shape)
    B2 = B2 ** 2
    print(B2.dtype)
    
    B_tmp = B2[:NPOINT, :].copy()
    print(B_tmp)
    B_tmp = _RowCol_FFT_Fast(B_tmp, Ls)
    
    # diagonal block
    
    A_Diag = []
    B_Diag = []
    
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
        
        A_Diag.append(mat)
        
        k_begin = i * ngrid_prim
        k_end   = (i + 1) * ngrid_prim
        mat     = B_tmp[:, k_begin:k_end]
        assert np.allclose(mat, B[b_begin:b_end, k_begin:k_end])
    
        B_Diag.append(mat)
    
    ################################ test rFFT, both python and C ################################
    
    B_tmp2 = B2[:NPOINT, :].copy()
    print(B_tmp2)

    mesh_complex = np.asarray([Ls[0], Ls[1], Ls[2]//2+1], dtype=np.int32)
    mesh_real = np.asarray(Ls, dtype=np.int32)
    nMeshReal = np.prod(Ls)
    nMeshComplex = np.prod(mesh_complex)
    assert B_tmp2.shape[1] % nMeshReal == 0
    nPoint = B_tmp2.shape[1] // nMeshReal
    
    Matrix = np.zeros((B_tmp2.shape[0], nPoint, nMeshComplex), dtype=np.complex128, order='C')
    Matrix_real = np.ndarray((B_tmp2.shape[0], nPoint, nMeshReal), dtype=np.float64, order='C', buffer=Matrix)
    Matrix_real.ravel()[:] = B_tmp2.ravel()
    
    Buf = np.zeros((B_tmp2.shape[0], nPoint, nMeshComplex), dtype=np.complex128, order='C')
    
    fn = getattr(libpbc, "_FFT_Matrix_Col_InPlace", None)
    assert fn is not None
    
    fn(
        Matrix_real.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(B_tmp.shape[0]),
        ctypes.c_int(nPoint),
        mesh_real.ctypes.data_as(ctypes.c_void_p),
        Buf.ctypes.data_as(ctypes.c_void_p)
    )

    Matrix = Matrix.reshape(Matrix.shape[0], -1)    
    B_tmp2 = _RowCol_rFFT_Fast(B_tmp2, Ls)
    
    assert np.allclose(Matrix, B_tmp2) # we get the correct answer!!!
    
    ################################ solve AX=B ################################
    
    fn_cholesky = getattr(libpbc, "Complex_Cholesky", None)
    assert fn_cholesky is not None
    fn_solve = getattr(libpbc, "Solve_LLTEqualB_Complex_Parallel", None)
    assert fn_solve is not None
    
    # X_Diag = []
    # X_Diag_C = []
    

    for i in range(ncell):
        A = A_Diag[i].copy()
        B = B_Diag[i].copy()
        X = np.linalg.solve(A, B)
        # X_Diag.append(X)

        # print("A = ", A)
        # print("A.shape = ", A.shape)
        # print("A.dtype = ", A.dtype)
        # e, h = np.linalg.eigh(A)
        # print(e)
        
        fn_cholesky(
            A.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(A.shape[0])
        )
        
        nrhs = B.shape[1]
        num_threads = lib.num_threads()
        bunchsize = nrhs // num_threads
        
        fn_solve(
            ctypes.c_int(A.shape[0]),
            A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(B.shape[1]),
            ctypes.c_int(bunchsize)
        )
        
        # print(X)
        # print(B)
        
        assert np.allclose(X, B)
    
    