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

COND_CUTOFF = 1e-16

AUX_BASIS_DATASET     = 'aux_basis'
AUX_BASIS_FFT_DATASET = 'aux_basis_fft'
AOR_DATASET           = 'aoR'
B_MATRIX              = 'B'

BUNCHSIZE_IN_AUX_BASIS_OUTCORE = 10000

## TODO: with k-symmetry and robust fitting ! 

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

# @profile    
def build_supercell(prim_atm, 
                    prim_a, 
                    mesh=None, 
                    Ls = [1,1,1], 
                    basis='gth-dzvp', 
                    pseudo='gth-pade', 
                    ke_cutoff=70, 
                    max_memory=2000, 
                    precision=1e-8,
                    use_particle_mesh_ewald=True,
                    verbose=4):
    
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
    Cell.verbose = verbose
    Cell.unit = 'angstorm'
    
    Cell.build(mesh=mesh)
    
    return Cell

def build_primitive_cell(supercell:Cell, kmesh):
    
    Cell = pbcgto.Cell()
    
    # assert prim_a[0, 1] == 0.0
    # assert prim_a[0, 2] == 0.0
    # assert prim_a[1, 0] == 0.0
    # assert prim_a[1, 2] == 0.0
    # assert prim_a[2, 0] == 0.0
    # assert prim_a[2, 1] == 0.0
    
    prim_a = np.array( [supercell.a[0]/kmesh[0], supercell.a[1]/kmesh[1], supercell.a[2]/kmesh[2]], dtype=np.float64 )
    
    print("supercell.a = ", supercell.a)
    print("prim_a = ", prim_a)
    
    Cell.a = prim_a
    
    atm = supercell.atom[:supercell.natm//np.prod(kmesh)]
    
    Cell.atom = atm
    Cell.basis = supercell.basis
    Cell.pseudo = supercell.pseudo
    Cell.ke_cutoff = supercell.ke_cutoff
    Cell.max_memory = supercell.max_memory
    Cell.precision = supercell.precision
    Cell.use_particle_mesh_ewald = supercell.use_particle_mesh_ewald
    Cell.verbose = supercell.verbose
    Cell.unit = supercell.unit
    
    mesh = np.array(supercell.mesh) // np.array(kmesh)
    
    Cell.build(mesh=mesh)
    
    return Cell

####################### Select IP #######################

# @profile 
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
    # print("possible_grid_ID = ", possible_grid_ID)
    
    return possible_grid_ID, np.array(coords[possible_grid_ID])

####### global IP selection #######

def _select_IP_ksym_global_direct(mydf, c:int, m:int, first_natm=None):

    print("In _select_IP_ksym_global_direct")

    bunchsize = lib.num_threads()

    ### determine the largest grids point of one atm ###

    natm         = mydf.cell.natm
    nao          = mydf.nao
    naux_max     = 0

    nao_per_atm = np.zeros((natm), dtype=np.int32)
    for i in range(mydf.nao):
        atm_id = mydf.ao2atomID[i]
        nao_per_atm[atm_id] += 1

    for nao_atm in nao_per_atm:
        naux_max = max(naux_max, int(np.sqrt(c*nao_atm)) + m)

    nthread = lib.num_threads()

    buf_size_per_thread = mydf.get_buffer_size_in_IP_selection(c, m)
    # buf_size            = buf_size_per_thread
    buf_size = 0

    # print("nthread        = ", nthread)
    # print("buf_size       = ", buf_size)
    # print("buf_per_thread = ", buf_size_per_thread)

    if hasattr(mydf, "IO_buf"):
        buf = mydf.IO_buf
    else:
        buf = np.zeros((buf_size), dtype=np.float64)
        mydf.IO_buf = buf

    if buf.size < buf_size:
        # reallocate
        mydf.IO_buf = np.zeros((buf_size), dtype=np.float64)
        print("reallocate buf of size = ", buf_size)
        buf = mydf.IO_buf
    buf_tmp = np.ndarray((buf_size), dtype=np.float64, buffer=buf)

    ### loop over atm ###

    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2

    df_tmp  = MultiGridFFTDF2(mydf.cell)
    grids   = df_tmp.grids
    coords  = np.asarray(grids.coords).reshape(-1,3)
    assert coords is not None

    results = []

    fn_colpivot_qr = getattr(libpbc, "ColPivotQR", None)
    assert(fn_colpivot_qr is not None)
    fn_ik_jk_ijk = getattr(libpbc, "NP_d_ik_jk_ijk", None)
    assert(fn_ik_jk_ijk is not None)

    weight = np.sqrt(mydf.cell.vol / coords.shape[0])

    aoR = mydf.aoR
    ngrid_prim = mydf.nGridPrim
    
    if aoR is None:
        aoR = mydf._numint.eval_ao(mydf.cell, mydf.ordered_grid_coords[:ngrid_prim])[0].T * weight
    
    assert ngrid_prim == aoR.shape[1]
    assert nao == aoR.shape[0]

    results = list(range(ngrid_prim))
    results = np.array(results, dtype=np.int32)

    ### global IP selection, we can use this step to avoid numerical issue ###

    print("global IP selection")

    bufsize = mydf.get_buffer_size_in_global_IP_selection(ngrid_prim, c, m)

    if buf.size < bufsize:
        mydf.IO_buf = np.zeros((bufsize), dtype=np.float64)
        buf = mydf.IO_buf
        print("reallocate buf of size = ", bufsize)

    dtypesize = buf.dtype.itemsize

    buf_tmp = np.ndarray((bufsize), dtype=np.float64, buffer=buf)

    offset = 0
    aoRg   = np.ndarray((nao, len(results)), dtype=np.complex128, buffer=buf_tmp)
    aoRg = aoR
    
    offset += nao*len(results) * dtypesize

    naux_now  = int(np.sqrt(c*nao)) + m
    naux2_now = naux_now * naux_now

    print("naux_now = ", naux_now)
    print("naux2_now = ", naux2_now)

    R = np.ndarray((naux2_now, len(results)), dtype=np.float64, buffer=buf_tmp, offset=offset)
    offset += naux2_now*len(results) * dtypesize

    aoRg1 = np.ndarray((naux_now, len(results)), dtype=np.float64, buffer=buf_tmp, offset=offset)
    offset += naux_now*len(results) * dtypesize

    aoRg2 = np.ndarray((naux_now, len(results)), dtype=np.float64, buffer=buf_tmp, offset=offset)
    offset += naux_now*len(results) * dtypesize

    aoPairBuffer = np.ndarray(
            (naux_now*naux_now, len(results)), dtype=np.float64, buffer=buf_tmp, offset=offset)
    offset += naux_now*naux_now*len(results) * dtypesize

    G1 = np.random.rand(nao, naux_now)
    G1, _ = numpy.linalg.qr(G1)
    G1    = G1.T
    G2 = np.random.rand(nao, naux_now)
    G2, _ = numpy.linalg.qr(G2)
    G2    = G2.T

    lib.dot(G1, aoRg, c=aoRg1)
    lib.dot(G2, aoRg, c=aoRg2)

    fn_ik_jk_ijk(aoRg1.ctypes.data_as(ctypes.c_void_p),
                 aoRg2.ctypes.data_as(ctypes.c_void_p),
                 aoPairBuffer.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux_now),
                 ctypes.c_int(naux_now),
                 ctypes.c_int(len(results)))

    nao_first = np.sum(nao_per_atm[:first_natm])

    max_rank  = min(naux2_now, len(results), nao_first * c)

    print("max_rank = ", max_rank)

    npt_find      = ctypes.c_int(0)
    pivot         = np.arange(len(results), dtype=np.int32)
    thread_buffer = np.ndarray((nthread+1, len(results)+1), dtype=np.float64, buffer=buf_tmp, offset=offset)
    offset       += (nthread+1)*(len(results)+1) * dtypesize
    global_buffer = np.ndarray((1, len(results)), dtype=np.float64, buffer=buf_tmp, offset=offset)
    offset       += len(results) * dtypesize

    fn_colpivot_qr(aoPairBuffer.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(naux2_now),
                    ctypes.c_int(len(results)),
                    ctypes.c_int(max_rank),
                    ctypes.c_double(1e-14),
                    pivot.ctypes.data_as(ctypes.c_void_p),
                    R.ctypes.data_as(ctypes.c_void_p),
                    ctypes.byref(npt_find),
                    thread_buffer.ctypes.data_as(ctypes.c_void_p),
                    global_buffer.ctypes.data_as(ctypes.c_void_p))

    npt_find = npt_find.value
    cutoff   = abs(R[npt_find-1, npt_find-1])
    print("ngrid = %d, npt_find = %d, cutoff = %12.6e" % (len(results), npt_find, cutoff))
    pivot = pivot[:npt_find]

    pivot.sort()

    results = np.array(results, dtype=np.int32)
    results = list(results[pivot])

    return results

  
####################### build aux basis #######################

## Incore 

def _RowCol_FFT_bench(input, Ls, inv=False, TransBra = True, TransKet = True):
    """
    A is a 3D array, (nbra, nket, ngrid_prim)
    """
    
    A = input
    ncell = np.prod(Ls)
    
    assert A.shape[1] % ncell == 0
    assert A.shape[0] % ncell == 0
    
    print("A.shape = ", A.shape)
    print("Ls = ", Ls)
    
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

def _construct_aux_basis_benchmark(mydf:ISDF.PBC_ISDF_Info):

    aoRg = mydf.aoRg
    coords = mydf.coords
    weight = np.sqrt(mydf.cell.vol / mydf.ngrids)
    aoR_unordered = mydf._numint.eval_ao(mydf.cell, coords)[0].T * weight
    Ls = mydf.Ls
    mesh = mydf.mesh
    meshPrim = np.array(mesh) // np.array(Ls)
    
    ### test the blockdiag matrixstructure ### 
    
    ncell     = np.prod(mydf.Ls)
    mesh      = mydf.mesh
    mesh_prim = np.array(mesh) // np.array(mydf.Ls)
    nGridPrim = mydf.nGridPrim
    nIP_Prim  = mydf.nIP_Prim
    nGrids    = mydf.ngrids
    
    A = np.asarray(lib.ddot(aoRg.T, aoRg), order='C')
    lib.square_inPlace(A)
    
    mydf.aux_basis_bench = np.asarray(lib.ddot(aoRg.T, aoR_unordered), order='C')
    lib.square_inPlace(mydf.aux_basis_bench)
    
    print("mydf.aux_basis_bench = ", mydf.aux_basis_bench.shape)
    
    ### check symmetry ### 
    
    print("B.shape = ", mydf.aux_basis_bench.shape)
    print("nIP_prim = ", nIP_Prim)
    print("nGridPrim = ", nGridPrim)
        
    A = _RowCol_FFT_bench(A, Ls)
    mydf.aux_basis_bench = _RowCol_FFT_ColFull_bench(mydf.aux_basis_bench, Ls, mesh)
    mydf.aux_basis_bench = mydf.aux_basis_bench.reshape(-1, meshPrim[0], Ls[0], meshPrim[1],Ls[1], meshPrim[2], Ls[2])
    mydf.aux_basis_bench = mydf.aux_basis_bench.transpose(0, 2, 4, 6, 1, 3, 5)
    mydf.aux_basis_bench = mydf.aux_basis_bench.reshape(-1, np.prod(mesh))
    
    for i in range(ncell):
        
        b_begin = i * nIP_Prim
        b_end   = (i + 1) * nIP_Prim
        
        k_begin = i * nIP_Prim
        k_end   = (i + 1) * nIP_Prim
        
        matrix_before = A[b_begin:b_end, :k_begin]
        matrix_after  = A[b_begin:b_end, k_end:]
        
        assert np.allclose(matrix_before, 0.0)
        assert np.allclose(matrix_after, 0.0)   
        
        k_begin = i * nGridPrim
        k_end   = (i + 1) * nGridPrim
        
        matrix_before = mydf.aux_basis_bench[b_begin:b_end, :k_begin]
        matrix_after  = mydf.aux_basis_bench[b_begin:b_end, k_end:]
        
        assert np.allclose(matrix_before, 0.0)
        assert np.allclose(matrix_after, 0.0)
    
    A = np.asarray(lib.ddot(aoRg.T, aoRg), order='C')
    lib.square_inPlace(A)
            
    mydf.aux_basis_bench = np.asarray(lib.ddot(aoRg.T, aoR_unordered), order='C')
    lib.square_inPlace(mydf.aux_basis_bench)
    
    mydf.aux_basis_bench = np.linalg.solve(A, mydf.aux_basis_bench)
    
    # perform FFT 
    
    mydf.aux_basis_bench_Grid = mydf.aux_basis_bench.copy()
    
    mydf.aux_basis_bench = _RowCol_FFT_ColFull_bench(mydf.aux_basis_bench, Ls, mesh)
    
    mydf.aux_basis_bench = mydf.aux_basis_bench.reshape(-1, meshPrim[0], Ls[0], meshPrim[1],Ls[1], meshPrim[2], Ls[2])
    mydf.aux_basis_bench = mydf.aux_basis_bench.transpose(0, 2, 4, 6, 1, 3, 5)
    mydf.aux_basis_bench = mydf.aux_basis_bench.reshape(-1, np.prod(mesh))
    
    aux_basis_bench_res = np.zeros((nIP_Prim, nGrids), dtype=np.complex128)

    for icell in range(ncell):
        b_begin = icell * nIP_Prim
        b_end   = (icell + 1) * nIP_Prim
        k_begin = icell * nGridPrim
        k_end   = (icell + 1) * nGridPrim
        
        matrix_before = mydf.aux_basis_bench[b_begin:b_end, :k_begin]
        matrix_after  = mydf.aux_basis_bench[b_begin:b_end, k_end:]
        if np.allclose(matrix_before, 0.0) == False:
            print("Warning Cell %d, matrix_before is not zero" % icell)
        # print("matrix_after = ", matrix_after)
        # assert np.allclose(matrix_after, 0.0, atol=1e-7)
        if np.allclose(matrix_after, 0.0) == False:
            print("Warning Cell %d, matrix_after is not zero" % icell)
        
        aux_basis_bench_res[:, k_begin:k_end] = mydf.aux_basis_bench[b_begin:b_end, k_begin:k_end]

    fac = np.sqrt(np.prod(Ls) / np.prod(mesh)) # normalization factor 

    mydf.aux_basis_bench = aux_basis_bench_res * fac

# @profile 
def _construct_aux_basis_kSym(mydf:ISDF.PBC_ISDF_Info):

    #### get the buffer ####
    
    nGrids   = mydf.ngrids
    nGridPrim = mydf.nGridPrim
    nIP_Prim = mydf.nIP_Prim
    
    Mesh = mydf.mesh
    Mesh = np.array(Mesh, dtype=np.int32)
    Ls   = mydf.Ls
    Ls   = np.array(Ls, dtype=np.int32)
    MeshPrim = np.array(Mesh) // np.array(Ls)
    ncell_complex = Ls[0] * Ls[1] * (Ls[2]//2+1)

    print("nGrids        = ", nGrids)
    print("nGridPrim     = ", nGridPrim)
    print("nIP_Prim      = ", nIP_Prim)
    print("ncell_complex = ", ncell_complex)
    print("Mesh          = ", Mesh)
    print("Ls            = ", Ls)
    print("MeshPrim      = ", MeshPrim)

    naux = mydf.naux
    
    mydf._allocate_jk_buffer() #
    
    buffer1 = np.ndarray((nIP_Prim, ncell_complex*nIP_Prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=0)
    buffer1_real = np.ndarray((nIP_Prim, naux), dtype=np.double, buffer=mydf.jk_buffer, offset=0)
    offset  = nIP_Prim * ncell_complex*nIP_Prim * buffer1.itemsize
    buffer2 = np.ndarray((nIP_Prim, ncell_complex*nGridPrim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    offset += nIP_Prim * ncell_complex*nGridPrim * buffer2.itemsize
    buffer3 = np.ndarray((nIP_Prim, ncell_complex*nGridPrim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    buffer3_real = np.ndarray((nIP_Prim, nGrids), dtype=np.double, buffer=mydf.jk_buffer, offset=offset)
    offset += nIP_Prim * ncell_complex*nGridPrim * buffer3.itemsize
    nthread = lib.num_threads()
    buffer_final_fft = np.ndarray((nthread, nGridPrim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    offset += nthread * nGridPrim * buffer_final_fft.itemsize
    
    nao = mydf.nao
    aoR = mydf.aoR
    nao_prim = nao // np.prod(Ls) 
    
    #### do the work ####

    aoRg = mydf.aoRg[:, :nIP_Prim] # only the first box is used
    
    print("aoRg.shape         = ", aoRg.shape)
    print("mydf.aoRg          = ", mydf.aoRg.shape)
    print("buffer1_real.shape = ", buffer1_real.shape)

    A = np.asarray(lib.ddot(aoRg.T, mydf.aoRg, c=buffer1_real), order='C')
    lib.square_inPlace(A)
    
    # construct B 
    
    offset_aoR = nIP_Prim * ncell_complex*nIP_Prim * buffer1.itemsize
    buf_aoR = np.ndarray((nao, nGridPrim), dtype=np.double, buffer=mydf.jk_buffer, offset=offset_aoR)
    offset_aoR_ddot = offset_aoR + nao * nGridPrim * buf_aoR.itemsize
    buf_ddot_res = np.ndarray((nIP_Prim, nGridPrim), dtype=np.double, buffer=mydf.jk_buffer, offset=offset_aoR_ddot)
    ddot_buf = mydf.ddot_buf
    
    # mydf.aux_basis = buffer3
    # B              = np.asarray(lib.ddot(aoRg.T, mydf.aoR, c=buffer3_real), order='C')
    
    for ix in range(Ls[0]):
        for iy in range(Ls[1]):
            for iz in range(Ls[2]):
                
                ### pack aoR ### 
                
                for ix_row in range(Ls[0]):
                    for iy_row in range(Ls[1]):
                        for iz_row in range(Ls[2]):
                            
                            loc_row1 = ix_row * Ls[1] * Ls[2] + iy_row * Ls[2] + iz_row
                            
                            row_begin1 = loc_row1 * nao_prim
                            row_end1   = (loc_row1 + 1) * nao_prim
                            
                            ix2 = (ix_row - ix + Ls[0]) % Ls[0]
                            iy2 = (iy_row - iy + Ls[1]) % Ls[1]
                            iz2 = (iz_row - iz + Ls[2]) % Ls[2]

                            loc_row2 = ix2 * Ls[1] * Ls[2] + iy2 * Ls[2] + iz2

                            row_begin2 = loc_row2 * nao_prim
                            row_end2   = (loc_row2 + 1) * nao_prim
                            
                            buf_aoR[row_begin1:row_end1, :] = aoR[row_begin2:row_end2, :]
                
                # perform one dot 
                
                loc = ix * Ls[1] * Ls[2] + iy * Ls[2] + iz
                k_begin = loc * nGridPrim
                k_end   = (loc + 1) * nGridPrim
            
                lib.ddot_withbuffer(aoRg.T, buf_aoR, c=buf_ddot_res, buf=ddot_buf)
                buffer3_real[:, k_begin:k_end] = buf_ddot_res
    B = buffer3_real
    lib.square_inPlace(B)
            
    ##### FFT #####
    
    fn = getattr(libpbc, "_FFT_Matrix_Col_InPlace", None)
    assert fn is not None
    
    print("A.shape = ", A.shape)
    print("B.shape = ", B.shape)
    print("buffer2.shape = ", buffer2.shape)
    print("mesh = ", Mesh)
    
    fn(
        A.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_Prim),
        ctypes.c_int(nIP_Prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buffer2.ctypes.data_as(ctypes.c_void_p)
    )
    
    fn(
        B.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_Prim),
        ctypes.c_int(nGridPrim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buffer2.ctypes.data_as(ctypes.c_void_p)
    )
    
    print("before solve linear equation")
        
    ##### solve the linear equation #####
    
    A_complex = buffer1
    B_complex = buffer3
    
    mydf.aux_basis = np.zeros((nIP_Prim, nGrids), dtype=np.complex128)
    
    fn_cholesky = getattr(libpbc, "Complex_Cholesky", None)
    assert fn_cholesky is not None
    fn_solve = getattr(libpbc, "Solve_LLTEqualB_Complex_Parallel", None)
    assert fn_solve is not None
    
    nthread = lib.num_threads()
    bunchsize = nGridPrim // nthread
    
    ### after solve linear equation, we have to perform another FFT ### 
    
    # freq1 = np.array(range(MeshPrim[0]), dtype=np.float64)
    # freq2 = np.array(range(MeshPrim[1]), dtype=np.float64)
    # freq3 = np.array(range(MeshPrim[2]), dtype=np.float64)
    # freq_q = np.array(np.meshgrid(freq1, freq2, freq3, indexing='ij'))
    
    # freq1 = np.array(range(Ls[0]), dtype=np.float64)
    # freq2 = np.array(range(Ls[1]), dtype=np.float64)
    # freq3 = np.array(range(Ls[2]//2+1), dtype=np.float64)
    # freq_Q = np.array(np.meshgrid(freq1, freq2, freq3, indexing='ij'))
    
    # FREQ = np.einsum("ijkl,ipqs->ijklpqs", freq_Q, freq_q)
    # FREQ[0] /= (Ls[0] * MeshPrim[0])
    # FREQ[1] /= (Ls[1] * MeshPrim[1])
    # FREQ[2] /= (Ls[2] * MeshPrim[2])
    # FREQ = np.einsum("ijklpqs->jklpqs", FREQ)
    # FREQ  = FREQ.reshape(-1, np.prod(MeshPrim)).copy()
    # FREQ  = np.exp(-2.0j * np.pi * FREQ)  # this is the only correct way to construct the factor
    
    fn_FREQ = getattr(libpbc, "_FREQ", None)
    assert fn_FREQ is not None
    
    FREQ = np.zeros((ncell_complex, nGridPrim), dtype=np.complex128)
    
    fn_FREQ(
        FREQ.ctypes.data_as(ctypes.c_void_p),
        MeshPrim.ctypes.data_as(ctypes.c_void_p),
        Ls.ctypes.data_as(ctypes.c_void_p)
    )
    
    fn_final_fft = getattr(libpbc, "_FinalFFT", None)
    assert fn_final_fft is not None
    fn_permutation_conj = getattr(libpbc, "_PermutationConj", None)
    assert fn_permutation_conj is not None
    
    # def _permutation(nx, ny, nz, shift_x, shift_y, shift_z):
    #     res = np.zeros((nx*ny*nz), dtype=numpy.int32)
    #     loc_now = 0
    #     for ix in range(nx):
    #         for iy in range(ny):
    #             for iz in range(nz):
    #                 ix2 = (nx - ix - shift_x) % nx
    #                 iy2 = (ny - iy - shift_y) % ny
    #                 iz2 = (nz - iz - shift_z) % nz
    #                 
    #                 loc = ix2 * ny * nz + iy2 * nz + iz2
    #                 # res[loc_now] = loc
    #                 res[loc] = loc_now
    #                 loc_now += 1
    #     return res
    
    fn_permutation = getattr(libpbc, "_get_permutation", None)
    assert fn_permutation is not None
    
    permutation = np.zeros((8, nGridPrim), dtype=np.int32)
    # print("permutation.shape = ", permutation.shape)
    # permutation[0] = _permutation(MeshPrim[0], MeshPrim[1], MeshPrim[2], 0, 0, 0)
    # permutation[1] = _permutation(MeshPrim[0], MeshPrim[1], MeshPrim[2], 0, 0, 1)
    # permutation[2] = _permutation(MeshPrim[0], MeshPrim[1], MeshPrim[2], 0, 1, 0)
    # permutation[3] = _permutation(MeshPrim[0], MeshPrim[1], MeshPrim[2], 0, 1, 1)
    # permutation[4] = _permutation(MeshPrim[0], MeshPrim[1], MeshPrim[2], 1, 0, 0)
    # permutation[5] = _permutation(MeshPrim[0], MeshPrim[1], MeshPrim[2], 1, 0, 1)
    # permutation[6] = _permutation(MeshPrim[0], MeshPrim[1], MeshPrim[2], 1, 1, 0)
    # permutation[7] = _permutation(MeshPrim[0], MeshPrim[1], MeshPrim[2], 1, 1, 1)
    
    fn_permutation(
        MeshPrim.ctypes.data_as(ctypes.c_void_p),
        permutation.ctypes.data_as(ctypes.c_void_p)
    )
    
    fac = np.sqrt(np.prod(Ls) / np.prod(Mesh)) # normalization factor
    
    buf_A = np.ndarray((nIP_Prim, nIP_Prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    offset += nIP_Prim * nIP_Prim * buf_A.itemsize
    buf_B = np.ndarray((nIP_Prim, nGridPrim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    offset += nIP_Prim * nGridPrim * buf_B.itemsize
    
    i=0
    for ix in range(Ls[0]):
        for iy in range(Ls[1]):
            for iz in range(Ls[2]//2+1):
                
                buf_A.ravel()[:] = A_complex[:, i*nIP_Prim:(i+1)*nIP_Prim].ravel()[:]
                buf_B.ravel()[:] = B_complex[:, i*nGridPrim:(i+1)*nGridPrim].ravel()[:]
                
                fn_cholesky(
                    buf_A.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nIP_Prim)
                )
                
                # X = np.linalg.solve(A_tmp, B_tmp)
        
                fn_solve(
                    ctypes.c_int(nIP_Prim),
                    buf_A.ctypes.data_as(ctypes.c_void_p),
                    buf_B.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(buf_B.shape[1]),
                    ctypes.c_int(bunchsize)
                )
                
                # with lib.threadpool_controller.limit(limits=lib.num_threads(), user_api='blas'):
                #     e, h = scipy.linalg.eigh(buf_A)

                # print("e = ", e)
                # print(buf_B)
        
                # e, h = np.linalg.eigh(buf_A)  # we get back to this mode, because numerical issue ! 
                
                # print("condition number = ", e[-1]/e[0])
                # e_max = np.max(e)
                # e_min_cutoff = e_max * COND_CUTOFF
                # # throw away the small eigenvalues
                # where = np.where(abs(e) > e_min_cutoff)[0]
                # e = e[where]
                # h = h[:, where].copy()
                # buf_C = np.ndarray((e.shape[0], nGridPrim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
                # buf_C = np.asarray(lib.dot(h.T.conj(), buf_B, c=buf_C), order='C')
                # buf_C = (1.0/e).reshape(-1,1) * buf_C
                # lib.dot(h, buf_C, c=buf_B)
                
                # print("B_tmp = ", B_tmp[:5,:5])
                
                # B_tmp1 = B_tmp.copy()
                # B_tmp1 = B_tmp1 * FREQ[i]
                # B_tmp1 = B_tmp1.reshape(-1, *MeshPrim)
                # B_tmp1 = np.fft.fftn(B_tmp1, axes=(1, 2, 3)) # shit
                # B_tmp1 = B_tmp1.reshape(-1, np.prod(MeshPrim))
                
                # print("B_tmp = ", B_tmp[:5,:5])
                
                # print(buf_B)


                fn_final_fft(
                    buf_B.ctypes.data_as(ctypes.c_void_p),
                    FREQ[i].ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nIP_Prim),
                    ctypes.c_int(nGridPrim),
                    MeshPrim.ctypes.data_as(ctypes.c_void_p),
                    buffer_final_fft.ctypes.data_as(ctypes.c_void_p)
                )
                
                # print("aux_basis = ", buf_B)
                
                # print("B_tmp1 = ", B_tmp1[:5,:5])
                # print("B_tmp  = ", B_tmp[:5,:5])
                                
                # assert np.allclose(B_tmp1, B_tmp)
                
                #### perform the last FFT ####
                
                iloc = ix * Ls[1] * Ls[2] + iy * Ls[2] + iz
                mydf.aux_basis[:, iloc*nGridPrim:(iloc+1)*nGridPrim] = buf_B
                
                # perform the complex conjugate transpose
                
                ix2 = (Ls[0] - ix) % Ls[0]
                iy2 = (Ls[1] - iy) % Ls[1]
                iz2 = (Ls[2] - iz) % Ls[2]
                
                i+=1
                
                if ix2==ix and iy2==iy and iz2==iz:
                    print("skip the complex conjugate transpose for (ix,iy,iz) = ", ix, iy, iz)
                    continue
                
                perm_id = 0
                if ix != 0:
                    perm_id += 4
                if iy != 0:
                    perm_id += 2
                if iz != 0:
                    perm_id += 1
                
                fn_permutation_conj(
                    buf_B.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nIP_Prim),
                    ctypes.c_int(nGridPrim),
                    permutation[perm_id].ctypes.data_as(ctypes.c_void_p),
                    buffer_final_fft.ctypes.data_as(ctypes.c_void_p)
                )
                
                iloc2 = ix2 * Ls[1] * Ls[2] + iy2 * Ls[2] + iz2
                
                mydf.aux_basis[:, iloc2*nGridPrim:(iloc2+1)*nGridPrim] = buf_B
    
    mydf.aux_basis = mydf.aux_basis * fac
                     
## Outcore

def _construct_aux_basis_kSym_outcore(mydf:ISDF.PBC_ISDF_Info, IO_File:str, IO_buf:np.ndarray):
    
    #### preprocess ####
    
    if isinstance(IO_File, str):
        if h5py.is_hdf5(IO_File):
            f_aux_basis = h5py.File(IO_File, 'a')
            if AUX_BASIS_DATASET in f_aux_basis:
                del (f_aux_basis[AUX_BASIS_DATASET])
            if AOR_DATASET in f_aux_basis:
                del (f_aux_basis[AOR_DATASET])
            if B_MATRIX in f_aux_basis:
                del (f_aux_basis[B_MATRIX])
        else:
            f_aux_basis = h5py.File(IO_File, 'w')
    else:
        assert (isinstance(IO_File, h5py.Group))
        f_aux_basis = IO_File
    
    # mydf._allocate_jk_buffer(np.float64)
    
    nGrids   = mydf.ngrids
    nGridPrim = mydf.nGridPrim
    nIP_Prim = mydf.nIP_Prim
    
    Mesh = mydf.mesh
    Mesh = np.array(Mesh, dtype=np.int32)
    Ls   = mydf.Ls
    Ls   = np.array(Ls, dtype=np.int32)
    MeshPrim = np.array(Mesh) // np.array(Ls)
    ncell = np.prod(Ls)
    ncell_complex = Ls[0] * Ls[1] * (Ls[2]//2+1)
    naux = mydf.naux
    naoPrim = mydf.nao // np.prod(Ls)
    nao  = mydf.nao
    
    chunk_size1 = (1, nIP_Prim, nIP_Prim)
    chunk_size2 = (naoPrim, nGridPrim) 
    
    ### do the work ### 
    
    ###### 1. create asyc functions ######
    
    h5d_B         = f_aux_basis.create_dataset(B_MATRIX, (ncell_complex, nIP_Prim, nGridPrim), dtype='complex128', chunks=chunk_size1)
    h5d_aux_basis = f_aux_basis.create_dataset(AUX_BASIS_DATASET, (ncell_complex, nIP_Prim, nGridPrim), dtype='complex128', chunks=chunk_size1)
    h5d_aoR       = f_aux_basis.create_dataset(AOR_DATASET, (nao, nGridPrim), 'f8', chunks=chunk_size2)
    
    def save_B(col0, col1, buf:np.ndarray):
        if col0<col1:
            dest_sel   = np.s_[:, :, col0:col1]
            source_sel = np.s_[:, :, :]
            h5d_B.write_direct(buf, source_sel=source_sel, dest_sel=dest_sel)
        # h5d_B[:, :, col0:col1] = buf.transpose(1, 0, 2)

    def save_aoR(col0, col1, buf:np.ndarray):
        if col0<col1:
            # print("write ", col0, col1)
            # print("buf", buf.dtype)
            # print(buf.shape)
            dest_sel   = np.s_[:, col0:col1]
            source_sel = np.s_[:, :]
            h5d_aoR.write_direct(buf, source_sel=source_sel, dest_sel=dest_sel)
    
    
    ####### 2. construct A ########
    
    IO_buf_memory = IO_buf.size * IO_buf.dtype.itemsize
    print("IO_buf size = ", IO_buf.size)
    
    offset = 0
    buf_A = np.ndarray((nIP_Prim, nIP_Prim * ncell_complex), dtype=np.complex128, buffer=IO_buf, offset=offset)
    buf_A_real = np.ndarray((nIP_Prim, nIP_Prim * ncell), dtype=np.double, buffer=IO_buf, offset=offset)
    
    offset += nIP_Prim * nIP_Prim * ncell_complex * buf_A.itemsize
    # print("offset = ", offset//8)
    # print("allocate size = ", nIP_Prim * nIP_Prim * ncell_complex * buf_A.itemsize//8)
    buf_A_fft = np.ndarray((nIP_Prim, nIP_Prim * ncell_complex), dtype=np.complex128, buffer=IO_buf, offset=offset)
    
    # NOTE：size of IO_buf at least 2 * nIP_Prim * nIP_Prim * ncell_complex * 2
    
    aoRg = mydf.aoRg[:, :nIP_Prim] # only the first box is used
    A    = np.asarray(lib.ddot(aoRg.T, mydf.aoRg, c=buf_A_real), order='C')
    lib.square_inPlace(A)
    
    fn = getattr(libpbc, "_FFT_Matrix_Col_InPlace", None)
    assert fn is not None
    
    fn(
        A.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_Prim),
        ctypes.c_int(nIP_Prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_A_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    A = buf_A
    
    # print("A[0,0]=",A[0,0])
    
    ####### 3. diag A while construct B ########
    
    # block_A_e = []
    block_A_e = np.zeros((ncell_complex, nIP_Prim), dtype=np.double)
    
    ### we assume that IO_buf is large enough so that the bunchsize of construct B can be larger than nGridPrim // ncell_complex + 1

    coords_prim = mydf.ordered_grid_coords[:nGridPrim]

    bunchsize = nGridPrim // ncell_complex + 1
    
    offset_backup = offset
    
    offset_aoR_buf1 = offset
    AoR_Buf1 = np.ndarray((nao, bunchsize), dtype=np.complex128, buffer=IO_buf, offset=offset)
    offset += nao * bunchsize * AoR_Buf1.itemsize
    
    offset_aoR_buf2 = offset
    AoR_Buf2 = np.ndarray((nao, bunchsize), dtype=np.complex128, buffer=IO_buf, offset=offset)
    offset += nao * bunchsize * AoR_Buf2.itemsize
    
    offset_aoR_bufpack = offset
    AoR_BufPack        = np.ndarray((nao, bunchsize), dtype=np.float64, buffer=IO_buf, offset=offset)
    offset            += nao * bunchsize * AoR_BufPack.itemsize

    B_bunchsize = min(nIP_Prim, bunchsize) # more acceptable for memory
    if B_bunchsize < BUNCHSIZE_IN_AUX_BASIS_OUTCORE:
        B_bunchsize = BUNCHSIZE_IN_AUX_BASIS_OUTCORE
    sub_bunchsize = B_bunchsize // ncell
    sub_bunchsize = min(sub_bunchsize, bunchsize)

    offset_B_buf1 = offset
    B_Buf1 = np.ndarray((nIP_Prim, ncell_complex, sub_bunchsize), dtype=np.complex128, buffer=IO_buf, offset=offset)
    offset += nIP_Prim * sub_bunchsize * ncell_complex * B_Buf1.itemsize
    
    offset_B_buf2 = offset
    B_Buf2 = np.ndarray((nIP_Prim, ncell_complex, sub_bunchsize), dtype=np.complex128, buffer=IO_buf, offset=offset)
    offset += nIP_Prim * sub_bunchsize * ncell_complex * B_Buf2.itemsize
    
    offset_B_buf3 = offset
    B_Buf_transpose = np.ndarray((nIP_Prim, sub_bunchsize, ncell_complex), dtype=np.complex128, buffer=IO_buf, offset=offset)
    offset += nIP_Prim * sub_bunchsize * ncell_complex * B_Buf_transpose.itemsize
    
    offset_B_buf_fft = offset
    B_BufFFT = np.ndarray((nIP_Prim, sub_bunchsize*ncell_complex), dtype=np.complex128, buffer=IO_buf, offset=offset)
    offset += nIP_Prim * sub_bunchsize * ncell_complex * B_BufFFT.itemsize
    
    offset_ddot_res = offset
    buf_ddot_res = np.ndarray((nIP_Prim, sub_bunchsize), dtype=np.double, buffer=IO_buf, offset=offset_ddot_res)
    offset += nIP_Prim * sub_bunchsize * buf_ddot_res.itemsize
    ddot_buf = mydf.ddot_buf
    
    buf_A_diag = np.ndarray((nIP_Prim, nIP_Prim), dtype=np.complex128, buffer=IO_buf, offset=offset)
    
    
    weight  = np.sqrt(mydf.cell.vol / nGrids)
    
    print("weight = ", weight)
    
    fn = getattr(libpbc, "_FFT_Matrix_Col_InPlace", None)
    assert fn is not None
    
    # with lib.call_in_background(save_B) as async_write_B:
    for _ in range(1):
        # with lib.call_in_background(save_aoR) as async_write_aoR:
        for _ in range(1):
            
            async_write_aoR = save_aoR
            async_write_B = save_B
            
            iloc = 0
            for ix in range(Ls[0]):
                for iy in range(Ls[1]):
                    for iz in range(Ls[2]//2+1):
                                                
                        bunch_begin = iloc * bunchsize
                        bunch_end   = (iloc + 1) * bunchsize
                        bunch_begin = min(bunch_begin, nGridPrim)
                        bunch_end   = min(bunch_end, nGridPrim)
                        
                        # print("iloc = ", iloc)
                        # print("bunch_begin = ", bunch_begin)
                        # print("bunch_end   = ", bunch_end)
                        # print("offset_aOR_buf1 = ", offset_aoR_buf1)
                        
                        AoR_Buf1 = np.ndarray((nao, bunch_end-bunch_begin), dtype=np.complex128, buffer=IO_buf, offset=offset_aoR_buf1)
                        AoR_Buf1 = ISDF_eval_gto(mydf.cell, coords=coords_prim[bunch_begin:bunch_end], out=AoR_Buf1) * weight

                        # print("after AO")

                        ### asyc write aoR ###
                        
                        async_write_aoR(bunch_begin, bunch_end, AoR_Buf1)
                        
                        # print("get here after write aoR")
            
                        for p0, p1 in lib.prange(0, bunch_end-bunch_begin, sub_bunchsize):
                            
                            AoR_BufPack = np.ndarray((nao, p1-p0), dtype=np.float64, buffer=IO_buf, offset=offset_aoR_bufpack)

                            B_Buf1 = np.ndarray((nIP_Prim, ncell, p1-p0), dtype=np.float64, buffer=IO_buf, offset=offset_B_buf1)

                            for ix2 in range(Ls[0]):
                                for iy2 in range(Ls[1]):
                                    for iz2 in range(Ls[2]):
                                    
                                        # pack aoR #
                                    
                                        for ix_row in range(Ls[0]):
                                            for iy_row in range(Ls[1]):
                                                for iz_row in range(Ls[2]):
                                                    
                                                    loc_row1 = ix_row * Ls[1] * Ls[2] + iy_row * Ls[2] + iz_row
                                                    
                                                    row_begin1 = loc_row1 * naoPrim
                                                    row_end1   = (loc_row1 + 1) * naoPrim
                                                    
                                                    ix3 = (ix_row - ix2 + Ls[0]) % Ls[0]
                                                    iy3 = (iy_row - iy2 + Ls[1]) % Ls[1]
                                                    iz3 = (iz_row - iz2 + Ls[2]) % Ls[2]
                                                    
                                                    loc_row2 = ix3 * Ls[1] * Ls[2] + iy3 * Ls[2] + iz3
                                                    
                                                    row_begin2 = loc_row2 * naoPrim
                                                    row_end2   = (loc_row2 + 1) * naoPrim
                                                    
                                                    AoR_BufPack[row_begin1:row_end1, :] = AoR_Buf1[row_begin2:row_end2, p0:p1]
                                                    # AoR_BufPack[row_begin2:row_end2, :] = AoR_Buf1[row_begin1:row_end1, p0:p1]

                                        # perform one dot #
                                    
                                        loc = ix2 * Ls[1] * Ls[2] + iy2 * Ls[2] + iz2
                                        
                                        k_begin = loc * bunchsize
                                        k_end   = (loc + 1) * bunchsize
                                        # print("p1 = ", p1)
                                        # print("p0 = ", p0)
                                        # print("ddot_buf.shape = ", ddot_buf.shape)
                                        # print("aoRg.shape = ", aoRg.shape)
                                        
                                        buf_ddot_res = np.ndarray((nIP_Prim, p1-p0), dtype=np.float64, buffer=IO_buf, offset=offset_ddot_res)
                                        lib.ddot_withbuffer(aoRg.T, AoR_BufPack, c=buf_ddot_res, buf=ddot_buf)
                                        B_Buf1[:, loc, :] = buf_ddot_res
                                        
                            lib.square_inPlace(B_Buf1)
                            
                            # print("we get here!")
                            fn(
                                B_Buf1.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(nIP_Prim),
                                ctypes.c_int(p1-p0),
                                Ls.ctypes.data_as(ctypes.c_void_p),
                                B_BufFFT.ctypes.data_as(ctypes.c_void_p)
                            )
                            # print("we get here!")
                            
                            B_Buf1_complex = np.ndarray((nIP_Prim, ncell_complex, p1-p0), dtype=np.complex128, buffer=IO_buf, offset=offset_B_buf1)
                            
                            B_Buf_transpose = np.ndarray((ncell_complex,nIP_Prim, p1-p0), dtype=np.complex128, buffer=IO_buf, offset=offset_B_buf3)
                            B_Buf_transpose = B_Buf1_complex.transpose(1, 0, 2)
                            B_Buf1 = np.ndarray((ncell_complex,nIP_Prim, p1-p0), dtype=np.complex128, buffer=IO_buf, offset=offset_B_buf1)
                            B_Buf1.ravel()[:] = B_Buf_transpose.ravel()[:]
                            
                            async_write_B(p0+bunch_begin, p1+bunch_begin, B_Buf1)
                            offset_B_buf1, offset_B_buf2 = offset_B_buf2, offset_B_buf1
                            
                            # print("we get here!")
                        
                        #### diag A ####
                        
                        buf_A_diag = A[:, iloc*nIP_Prim:(iloc+1)*nIP_Prim]
                        
                        with lib.threadpool_controller.limit(limits=lib.num_threads(), user_api='blas'):
                            e, h = scipy.linalg.eigh(buf_A_diag)
                        block_A_e[iloc] = e
                        A[:, iloc*nIP_Prim:(iloc+1)*nIP_Prim] = h
                        
                        #### final swap buffer ####
                        
                        offset_aoR_buf1, offset_aoR_buf2 = offset_aoR_buf2, offset_aoR_buf1
                        
                        iloc += 1

    ####### 4. construct aux basis ########

    offset = offset_backup
    B_bunchsize = min(nIP_Prim, nGridPrim) # more acceptable for memory
    if B_bunchsize < BUNCHSIZE_IN_AUX_BASIS_OUTCORE:
        B_bunchsize = BUNCHSIZE_IN_AUX_BASIS_OUTCORE
    B_bunchsize = min(B_bunchsize, nGridPrim)

    offset_B_buf1 = offset
    B_Buf1 = np.ndarray((nIP_Prim, B_bunchsize), dtype=np.complex128, buffer=IO_buf, offset=offset)
    offset += nIP_Prim * B_bunchsize * B_Buf1.itemsize
    
    offset_B_buf2 = offset
    B_Buf2 = np.ndarray((nIP_Prim, B_bunchsize), dtype=np.complex128, buffer=IO_buf, offset=offset)
    offset += nIP_Prim * B_bunchsize * B_Buf2.itemsize
    
    offset_B_ddot = offset
    B_ddot = np.ndarray((nIP_Prim, B_bunchsize), dtype=np.complex128, buffer=IO_buf, offset=offset)
    offset += nIP_Prim * B_bunchsize * B_ddot.itemsize

    buf_A = np.ndarray((nIP_Prim, nIP_Prim), dtype=np.complex128, buffer=IO_buf, offset=offset)
    offset += nIP_Prim * nIP_Prim * buf_A.itemsize
    
    offset_A2 = offset
    buf_A2 = np.ndarray((nIP_Prim, nIP_Prim), dtype=np.complex128, buffer=IO_buf, offset=offset)
    
    def save_auxbasis(icell, col0, col1, buf:np.ndarray):
        dest_sel   = np.s_[icell, :, col0:col1]
        source_sel = np.s_[:, :]
        h5d_aux_basis.write_direct(buf, source_sel=source_sel, dest_sel=dest_sel)

    def load_B(icell, col0, col1, buf:np.ndarray):
        dest_sel   = np.s_[:, :]
        source_sel = np.s_[icell, :, col0:col1]
        f_aux_basis[B_MATRIX].read_direct(buf, source_sel=source_sel, dest_sel=dest_sel)

    def load_B2(icell, col0, col1, buf:np.ndarray):
        dest_sel   = np.s_[:, :]
        source_sel = np.s_[icell, :, col0:col1]
        f_aux_basis[B_MATRIX].read_direct(buf, source_sel=source_sel, dest_sel=dest_sel)

    # with lib.call_in_background(save_auxbasis) as async_write:
    #     with lib.call_in_background(load_B) as async_loadB: 

    for _ in range(1):
        for _ in range(1):
            async_write = save_auxbasis
            async_loadB = load_B

            for icell in range(ncell_complex):
        
                begin = icell       * nIP_Prim
                end   = (icell + 1) * nIP_Prim
        
                buf_A = A[:, begin:end]
                e = block_A_e[icell]

                e_max = np.max(e)
                e_min_cutoff = e_max * COND_CUTOFF
                # throw away the small eigenvalues
                # where = np.where(abs(e) > e_min_cutoff)[0]
                where1 = np.where(e < e_min_cutoff)[0]
                e1 = e[where1]
                print("eigenvalues indicate numerical instability")
                for loc, x in enumerate(e1):
                    print("e1[%3d] = %15.6e" % (loc, x))
                where = np.where(e > e_min_cutoff)[0]
                e = e[where]
                
                buf_A2 = np.ndarray((nIP_Prim, e.shape[0]), dtype=np.complex128, buffer=IO_buf, offset=offset_A2)
                buf_A2.ravel()[:] = buf_A[:, where].ravel()[:]

                B_Buf1 = np.ndarray((nIP_Prim, B_bunchsize), dtype=np.complex128, buffer=IO_buf, offset=offset_B_buf1)
                load_B2(icell, 0, B_bunchsize, B_Buf1)
                
                # print("B[%d,0,0] = "%icell, B_Buf1[0,0])
                
                for p0, p1 in lib.prange(0, nGridPrim, B_bunchsize): 
                    
                    B_Buf1 = np.ndarray((nIP_Prim, p1-p0), dtype=np.complex128, buffer=IO_buf, offset=offset_B_buf1)
                    
                    p2 = p1 + B_bunchsize
                    p2 = min(p2, nGridPrim)
                    B_Buf2 = np.ndarray((nIP_Prim, p2-p1), dtype=np.complex128, buffer=IO_buf, offset=offset_B_buf2)
                    
                    if p1 < p2:
                        async_loadB(icell, p1, p2, B_Buf2)
                    
                    ### do the work ### 
                    
                    B_ddot = np.ndarray((e.shape[0], p1-p0), dtype=np.complex128, buffer=IO_buf, offset=offset_B_ddot)
                    lib.dot(buf_A2.T.conj(), B_Buf1, c=B_ddot)
                    B_ddot = (1.0/e).reshape(-1,1) * B_ddot
                    lib.dot(buf_A2, B_ddot, c=B_Buf1)
                    
                    ## async write ## 
                    
                    async_write(icell, p0, p1, B_Buf1)
                    offset_B_buf2, offset_B_buf1 = offset_B_buf1, offset_B_buf2
        
        # the final FFT is postponed to the next step # 
        
       
def _aux_basis_FFT_outcore(mydf:ISDF.PBC_ISDF_Info, IO_File:str, IO_buf:np.ndarray):
    
    t1 = (logger.process_clock(), logger.perf_counter())
    
    #### preprocess ####
    
    if isinstance(IO_File, str):
        if h5py.is_hdf5(IO_File):
            f_aux_basis = h5py.File(IO_File, 'a')
            assert AUX_BASIS_DATASET in f_aux_basis
            if AUX_BASIS_FFT_DATASET in f_aux_basis:
                del (f_aux_basis[AUX_BASIS_FFT_DATASET])
        else:
            raise ValueError("IO_File must be a h5py.File object")
    else:
        assert (isinstance(IO_File, h5py.Group))
        f_aux_basis = IO_File

    nGridPrim = mydf.nGridPrim
    nIP_Prim = mydf.nIP_Prim
    mesh = mydf.mesh
    meshPrim = np.array(mesh) // np.array(mydf.Ls)
    meshPrim = np.array(meshPrim, dtype=np.int32)
    Ls = np.array(mydf.Ls, dtype=np.int32)
    ncell = np.prod(Ls)
    ncell_complex = Ls[0] * Ls[1] * (Ls[2]//2+1)
    
    itask_2_xyz = np.zeros((ncell_complex, 3), dtype=np.int32)
    
    loc = 0
    for ix in range(Ls[0]):
        for iy in range(Ls[1]):
            for iz in range(Ls[2]//2+1):
                itask_2_xyz[loc, 0] = ix
                itask_2_xyz[loc, 1] = iy
                itask_2_xyz[loc, 2] = iz
                loc += 1

    # get freq #
    
    freq1 = np.array(range(meshPrim[0]), dtype=np.float64)
    freq2 = np.array(range(meshPrim[1]), dtype=np.float64)
    freq3 = np.array(range(meshPrim[2]), dtype=np.float64)
    freq_q = np.array(np.meshgrid(freq1, freq2, freq3, indexing='ij'))
    freq1 = np.array(range(Ls[0]), dtype=np.float64)
    freq2 = np.array(range(Ls[1]), dtype=np.float64)
    freq3 = np.array(range(Ls[2]//2+1), dtype=np.float64)
    freq_Q = np.array(np.meshgrid(freq1, freq2, freq3, indexing='ij'))
    FREQ = np.einsum("ijkl,ipqs->ijklpqs", freq_Q, freq_q)
    FREQ[0] /= (Ls[0] * meshPrim[0])
    FREQ[1] /= (Ls[1] * meshPrim[1])
    FREQ[2] /= (Ls[2] * meshPrim[2])
    FREQ = np.einsum("ijklpqs->jklpqs", FREQ)
    FREQ  = FREQ.reshape(-1, np.prod(meshPrim)).copy()
    FREQ  = np.exp(-2.0j * np.pi * FREQ)
    
    print("FREQ.shape = ", FREQ.shape)
    
    def _permutation(nx, ny, nz, shift_x, shift_y, shift_z):
        
        res = np.zeros((nx*ny*nz), dtype=numpy.int32)
        
        loc_now = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    ix2 = (nx - ix - shift_x) % nx
                    iy2 = (ny - iy - shift_y) % ny
                    iz2 = (nz - iz - shift_z) % nz
                    
                    loc = ix2 * ny * nz + iy2 * nz + iz2
                    # res[loc_now] = loc
                    res[loc] = loc_now
                    loc_now += 1
        return res
    
    permutation = np.zeros((8, nGridPrim), dtype=np.int32)
    # print("permutation.shape = ", permutation.shape)
    permutation[0] = _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 0, 0, 0)
    permutation[1] = _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 0, 0, 1)
    permutation[2] = _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 0, 1, 0)
    permutation[3] = _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 0, 1, 1)
    permutation[4] = _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 1, 0, 0)
    permutation[5] = _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 1, 0, 1)
    permutation[6] = _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 1, 1, 0)
    permutation[7] = _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 1, 1, 1)
    
    fac = np.sqrt(np.prod(Ls) / np.prod(mesh))
    
    # allocate buffer #

    IO_Buf_Size = IO_buf.size 
    
    nthread = lib.num_threads()
    
    
    bunchsize = (IO_Buf_Size - nthread * nGridPrim * 2) // (nGridPrim * 2 * 4)
    
    if bunchsize < 1:
        raise ValueError("IO_buf is too small") 
    
    # if bunchsize > nIP_Prim // 5: # for test only
    #     bunchsize = nIP_Prim // 5
    
    IBunch = nIP_Prim // bunchsize + 1
    bunchsize_now = nIP_Prim // IBunch + 1
    bunchsize_now = min(bunchsize_now, nIP_Prim)
    bunchsize = bunchsize_now
    
    print("itask = ", ncell_complex)
    print("Ibunch = ", IBunch)
    print("bunchsize = ", bunchsize_now) 
    
    offset = 0
    buffer_final_fft = np.ndarray((nthread, nGridPrim), dtype=np.complex128, buffer=IO_buf, offset=offset)
    offset += nthread * nGridPrim * buffer_final_fft.itemsize
    
    offset_buf1 = offset
    aux_FFT_buf1 = np.ndarray((bunchsize_now, nGridPrim), dtype=np.complex128, buffer=IO_buf, offset=offset)
    offset += bunchsize_now * nGridPrim * aux_FFT_buf1.itemsize
    
    offset_buf2 = offset
    aux_FFT_buf2 = np.ndarray((bunchsize_now, nGridPrim), dtype=np.complex128, buffer=IO_buf, offset=offset)
    offset += bunchsize_now * nGridPrim * aux_FFT_buf2.itemsize
    
    offset_buf3 = offset
    auf_FFT_conj_buf1 = np.ndarray((bunchsize_now, nGridPrim), dtype=np.complex128, buffer=IO_buf, offset=offset)
    offset += bunchsize_now * nGridPrim * auf_FFT_conj_buf1.itemsize
    
    offset_buf4 = offset
    auf_FFT_conj_buf2 = np.ndarray((bunchsize_now, nGridPrim), dtype=np.complex128, buffer=IO_buf, offset=offset)
    offset += bunchsize_now * nGridPrim * auf_FFT_conj_buf2.itemsize
    
    #aux_basis_dataset = f_aux_basis[AUX_BASIS_DATASET]
    h5d_aux_basis = f_aux_basis[AUX_BASIS_DATASET]
    h5d_aux_FFT   = f_aux_basis.create_dataset(AUX_BASIS_FFT_DATASET, (ncell, nIP_Prim, nGridPrim), dtype='complex128', chunks=(1, nIP_Prim, nIP_Prim))

    print("h5d_aux_basis.dtype = ", h5d_aux_basis.dtype)

    def load_aux_basis(icell, row0, row1, buf:np.ndarray):
        if row0<row1:
            dest_sel   = np.s_[:, :]
            source_sel = np.s_[icell, row0:row1, :]
            h5d_aux_basis.read_direct(buf, source_sel=source_sel, dest_sel=dest_sel)
    
    def load_aux_basis2(icell, row0, row1, buf:np.ndarray):
        if row0<row1:
            dest_sel   = np.s_[:, :]
            source_sel = np.s_[icell, row0:row1, :]
            h5d_aux_basis.read_direct(buf, source_sel=source_sel, dest_sel=dest_sel)

    def save_aux_FFT(icell, row0, row1, buf:np.ndarray):
        if row0<row1:
            source_sel = np.s_[:, :]
            dest_sel   = np.s_[icell, row0:row1, :]
            h5d_aux_FFT.write_direct(buf, source_sel=source_sel, dest_sel=dest_sel)
    
    def save_aux_FFT2(icell, row0, row1, buf:np.ndarray):
        if row0<row1:
            source_sel = np.s_[:, :]
            dest_sel   = np.s_[icell, row0:row1, :]
            h5d_aux_FFT.write_direct(buf, source_sel=source_sel, dest_sel=dest_sel)
    
    def save_aux_FFT3(icell, row0, row1, buf:np.ndarray):
        if row0<row1:
            source_sel = np.s_[:, :]
            dest_sel   = np.s_[icell, row0:row1, :]
            h5d_aux_FFT.write_direct(buf, source_sel=source_sel, dest_sel=dest_sel)
    
    nTask_tot = ncell_complex * IBunch
    
    fn_final_fft = getattr(libpbc, "_FinalFFT", None)
    assert fn_final_fft is not None
    fn_permutation_conj = getattr(libpbc, "_PermutationConj", None)
    assert fn_permutation_conj is not None
                
    # with lib.call_in_background(save_aux_FFT) as async_write_aux_FFT:
    #     with lib.call_in_background(load_aux_basis) as async_load_aux_basis:
    
    for _ in range(1):
        for _ in range(1):
            
            async_write_aux_FFT = save_aux_FFT
            async_load_aux_basis = load_aux_basis
           
            icell = 0
            row0  = 0
            row1  = bunchsize_now   
            
            aux_FFT_buf1 = np.ndarray((bunchsize_now, nGridPrim), dtype=np.complex128, buffer=IO_buf, offset=offset_buf1)
            load_aux_basis2(icell, row0, row1, aux_FFT_buf1)
            
            for itask in range(nTask_tot):
                                
                t3 = (logger.process_clock(), logger.perf_counter())
                
                icell = itask // IBunch
                p0 = (itask % IBunch) * bunchsize_now
                p1 = p0 + bunchsize_now
                p1 = min(p1, nIP_Prim)
                
                aux_FFT_buf1 = np.ndarray((p1-p0, nGridPrim), dtype=np.complex128, buffer=IO_buf, offset=offset_buf1)
                    
                ### preload the next task ###
                
                itask_next = itask + 1
                icell_next = itask_next // IBunch
                p0_next = itask_next % IBunch * bunchsize_now
                p1_next = p0_next + bunchsize_now
                p1_next = min(p1_next, nIP_Prim)
                    
                if itask_next == nTask_tot:
                    p0_next = 0
                    p1_next = 0
                aux_FFT_buf2 = np.ndarray((p1_next-p0_next, nGridPrim), dtype=np.complex128, buffer=IO_buf, offset=offset_buf2)
                async_load_aux_basis(icell_next, p0_next, p1_next, aux_FFT_buf2)
                
                ### do the work ###
                
                ix, iy, iz = itask_2_xyz[icell]

                fn_final_fft(
                    aux_FFT_buf1.ctypes.data_as(ctypes.c_void_p),
                    FREQ[icell].ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(p1-p0),
                    ctypes.c_int(nGridPrim),
                    meshPrim.ctypes.data_as(ctypes.c_void_p),
                    buffer_final_fft.ctypes.data_as(ctypes.c_void_p)
                )
                
                aux_FFT_buf1 *= fac
                    
                real_cell_loc = ix * Ls[1] * (Ls[2]) + iy * (Ls[2]) + iz
                # async_write_aux_FFT(real_cell_loc, p0, p1, aux_FFT_buf1) ???? Why not use this?   
                save_aux_FFT3(real_cell_loc, p0, p1, aux_FFT_buf1)
                    
                    ### complex conjugate ###
                    
                    ### we don't even need this ###

                    # ix2 = (Ls[0] - ix) % Ls[0]
                    # iy2 = (Ls[1] - iy) % Ls[1]
                    # iz2 = (Ls[2] - iz) % Ls[2]
                
                    # if ix2!=ix or iy2!=iy or iz2!=iz:
                        
                    # perm_id = 0
                    # if ix != 0:
                    #     perm_id += 4
                    # if iy != 0:
                    #     perm_id += 2
                    # if iz != 0:
                    #     perm_id += 1
                    
                    # p0_prime = p0
                    # p1_prime = p1
                    # if ix2!=ix or iy2!=iy or iz2!=iz:
                    #     aux_FFT_conj_buf1 = np.ndarray((p1-p0, nGridPrim), dtype=np.complex128, buffer=IO_buf, offset=offset_buf3)
                    #     aux_FFT_conj_buf1.ravel()[:] = aux_FFT_buf1.ravel()[:]
                    #     fn_permutation_conj(
                    #         aux_FFT_conj_buf1.ctypes.data_as(ctypes.c_void_p),
                    #         ctypes.c_int(p1-p0),
                    #         ctypes.c_int(nGridPrim),
                    #         permutation[perm_id].ctypes.data_as(ctypes.c_void_p),
                    #         buffer_final_fft.ctypes.data_as(ctypes.c_void_p)
                    #     )
                    # else:
                    #     # not work to be done actually
                    #     aux_FFT_conj_buf1 = None
                    #     p0_prime = 0
                    #     p1_prime = 0
                    # real_cell_loc2 = ix2 * Ls[1] * (Ls[2]) + iy2 * (Ls[2]) + iz2
                    # async_write_aux_FFT2(real_cell_loc2, p0_prime, p1_prime, aux_FFT_conj_buf1)
                    # # save_aux_FFT3(real_cell_loc2, p0, p1, aux_FFT_conj_buf1)
                    # offset_buf3, offset_buf4 = offset_buf4, offset_buf3
                
                    ## final swtich ### 
                    
                offset_buf1, offset_buf2 = offset_buf2, offset_buf1
                                    
                t4 = (logger.process_clock(), logger.perf_counter())
                
                print("itask %5d wall time: %12.6f CPU time: %12.6f" % (itask, t4[1] - t3[1], t4[0] - t3[0]))

    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "aux_basis_FFT_outcore")


####################### construct W #######################
    
def _construct_W_benchmark_grid(cell, aux_basis:np.ndarray, Ls, mesh):
    
    def constrcuct_V_CCode(aux_basis:np.ndarray, mesh, coul_G):
        
        coulG_real         = coul_G.reshape(*mesh)[:, :, :mesh[2]//2+1].reshape(-1)
        nThread            = lib.num_threads()
        nAux               = aux_basis.shape[0]
        bunchsize          = nAux // (2*nThread)
        bufsize_per_thread = bunchsize * coulG_real.shape[0] * 2
        bufsize_per_thread = (bufsize_per_thread + 15) // 16 * 16
        ngrids             = aux_basis.shape[1]
        mesh_int32         = np.array(mesh, dtype=np.int32)

        V                  = np.zeros((nAux, ngrids), dtype=np.double)
        buffer             = np.zeros((nThread, bufsize_per_thread), dtype=np.double)

        fn = getattr(libpbc, "_construct_V", None)
        assert(fn is not None)

        print("V.shape = ", V.shape)
        print("aux_basis.shape = ", aux_basis.shape)
        # print("self.jk_buffer.size    = ", self.jk_buffer.size)
        # print("self.jk_buffer.shape   = ", self.jk_buffer.shape)

        fn(mesh_int32.ctypes.data_as(ctypes.c_void_p),
           ctypes.c_int(nAux),
           aux_basis.ctypes.data_as(ctypes.c_void_p),
           coulG_real.ctypes.data_as(ctypes.c_void_p),
           V.ctypes.data_as(ctypes.c_void_p),
           ctypes.c_int(bunchsize),
           buffer.ctypes.data_as(ctypes.c_void_p),
           ctypes.c_int(bufsize_per_thread))

        return V
    
    mesh  = cell.mesh
    coulG = tools.get_coulG(cell, mesh=mesh)
    
    V_R = constrcuct_V_CCode(aux_basis, mesh, coulG)
    
    naux = aux_basis.shape[0]
    
    W = np.zeros((naux,naux))
    lib.ddot(a=aux_basis, b=V_R.T, c=W, beta=1.0)
    
    meshPrim = np.array(mesh) // np.array(Ls)
    V_R = V_R.reshape(-1, Ls[0], meshPrim[0], Ls[1], meshPrim[1], Ls[2], meshPrim[2])
    V_R = V_R.transpose(0, 1, 3, 5, 2, 4, 6).reshape(V_R.shape[0], -1)
    
    return V_R, W
    
## Incore

# @profile 
def _construct_W_incore(mydf:ISDF.PBC_ISDF_Info):
    
    mydf._allocate_jk_buffer()
    
    mesh  = mydf.mesh
    coulG = tools.get_coulG(mydf.cell, mesh=mesh)
    Ls    = mydf.Ls
    Ls    = np.array(Ls, dtype=np.int32)
    mesh_prim = np.array(mesh) // np.array(Ls)
    coulG = coulG.reshape(mesh_prim[0], Ls[0], mesh_prim[1], Ls[1], mesh_prim[2], Ls[2])
    coulG = coulG.transpose(1, 3, 5, 0, 2, 4).reshape(-1, np.prod(mesh_prim)).copy()
    
    nIP_prim      = mydf.nIP_Prim
    nGrid_prim    = mydf.nGridPrim
    ncell         = np.prod(Ls)
    ncell_complex = Ls[0] * Ls[1] * (Ls[2]//2+1)
    
    #### allocate buffer ####
    
    W = np.zeros((nIP_prim, nIP_prim*ncell), dtype=np.float64)
    
    print("nIP_prim = ", nIP_prim)
    print("nGrid_prim = ", nGrid_prim)
    
    offset  = 0
    A_buf   = np.ndarray((nIP_prim, nGrid_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=0)
    offset += nIP_prim * nGrid_prim * A_buf.itemsize
    print("offset = ", offset//8)
    B_buf   = np.ndarray((nIP_prim, nGrid_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    offset += nIP_prim * nGrid_prim * B_buf.itemsize
    print("offset = ", offset//8)
    W_buf   = np.ndarray((nIP_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    offset += nIP_prim * nIP_prim * W_buf.itemsize
    print("offset = ", offset//8)
    print("ncell_complex = ", ncell_complex)
    print(mydf.jk_buffer.size)
    W_buf2  = np.ndarray((nIP_prim, nIP_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)   
    W_buf3  = np.ndarray((nIP_prim, nIP_prim*ncell), dtype=np.float64, buffer=mydf.jk_buffer, offset=offset)
    offset += nIP_prim * nIP_prim * ncell_complex * W_buf2.itemsize
    W_buf_fft = np.ndarray((nIP_prim, nIP_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)

    fn = getattr(libpbc, "_construct_W_multiG", None)
    assert(fn is not None)

    # for i in range(ncell):
    
    loc = 0
    
    for ix in range(Ls[0]):
        for iy in range(Ls[1]):
            for iz in range(Ls[2]//2+1):
                
                i = ix * Ls[1] * (Ls[2]) + iy * (Ls[2]) + iz
                
                k_begin = i * nGrid_prim
                k_end   = (i + 1) * nGrid_prim
        
                A_buf[:] = mydf.aux_basis[:, k_begin:k_end]

                B_buf[:] = A_buf[:]

                fn(
                    ctypes.c_int(nIP_prim),
                    ctypes.c_int(0),
                    ctypes.c_int(nGrid_prim),
                    B_buf.ctypes.data_as(ctypes.c_void_p),
                    coulG[i].ctypes.data_as(ctypes.c_void_p)
                )
        
                # print("A_buf.shape = ", A_buf.shape)
                # print("B_buf.shape = ", B_buf.shape)
                # print("W_buf.shape = ", W_buf.shape)
                # print("coulG = ", coulG[i])
                # print("A_buf = ", A_buf)
                # print("B_buf = ", B_buf)
            
                # lib.dot(B_buf, A_buf.T.conj(), c=W_buf)
                lib.dot(A_buf, B_buf.T.conj(), c=W_buf)
                
                # print("W_buf = ", W_buf)

                k_begin = loc * nIP_prim
                k_end   = (loc + 1) * nIP_prim

                W_buf2[:, k_begin:k_end] = W_buf
            
                loc += 1
    
    fn = getattr(libpbc, "_iFFT_Matrix_Col_InPlace", None)
    assert(fn is not None)
    
    fn(
        W_buf2.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_prim),
        ctypes.c_int(nIP_prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        W_buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    
    W.ravel()[:] = W_buf3.ravel()[:]
    mydf.W = W

    ### used in get J ### 
    
    W0 = np.zeros((nIP_prim, nIP_prim), dtype=np.float64)
    
    for i in range(ncell):
            
            k_begin = i * nIP_prim
            k_end   = (i + 1) * nIP_prim
            
            W0 += W[:, k_begin:k_end]

    mydf.W0 = W0

    return W


## Outcore

def _construct_W_outcore(mydf:ISDF.PBC_ISDF_Info, IO_File:str, IO_buf:np.ndarray):
    
    mydf._allocate_jk_buffer()
    
    #### preprocess ####
    
    if isinstance(IO_File, str):
        if h5py.is_hdf5(IO_File):
            f_aux_basis = h5py.File(IO_File, 'a')
            assert AUX_BASIS_FFT_DATASET in f_aux_basis
        else:
            raise ValueError("IO_File must be a h5py.File object")
    else:
        assert (isinstance(IO_File, h5py.Group))
        f_aux_basis = IO_File
        
    nGridPrim = mydf.nGridPrim
    nIP_Prim = mydf.nIP_Prim
    mesh = mydf.mesh
    meshPrim = np.array(mesh) // np.array(mydf.Ls)
    meshPrim = np.array(meshPrim, dtype=np.int32)
    Ls = np.array(mydf.Ls, dtype=np.int32)
    ncell = np.prod(Ls)
    ncell_complex = Ls[0] * Ls[1] * (Ls[2]//2+1)
    
    h5d_aux_basis_fft = f_aux_basis[AUX_BASIS_FFT_DATASET]
    
    def load_aux_basis_fft(icell, col0, col1, buf:np.ndarray):
        if col0<col1:
            dest_sel   = np.s_[:, :]
            source_sel = np.s_[icell, :, col0:col1]
            h5d_aux_basis_fft.read_direct(buf, source_sel=source_sel, dest_sel=dest_sel)
    
    def load_aux_basis_fft2(icell, col0, col1, buf:np.ndarray):
        if col0<col1:
            dest_sel   = np.s_[:, :]
            source_sel = np.s_[icell, :, col0:col1]
            h5d_aux_basis_fft.read_direct(buf, source_sel=source_sel, dest_sel=dest_sel)

    #### allocate buffer ####
    
    W = np.zeros((nIP_Prim, nIP_Prim*ncell_complex), dtype=np.complex128)
    W_real = np.ndarray((nIP_Prim, nIP_Prim*ncell), dtype=np.float64, buffer=W, offset = 0)

    IO_buf_size = IO_buf.size
    print("size IO Buf = ", IO_buf.size)
    bunchsize = (IO_buf_size - nIP_Prim * nIP_Prim * 2 - nIP_Prim * nIP_Prim * ncell_complex * 2) // (3 * 2 * nIP_Prim)
    bunchsize = min(bunchsize, nGridPrim)

    if nGridPrim % bunchsize == 0:
        nBunch = nGridPrim//bunchsize
    else:
        nBunch = nGridPrim // bunchsize + 1
        bunchsize = nGridPrim // nBunch + 1
        bunchsize = min(bunchsize, nGridPrim)
    
    print("bunchsize = ", bunchsize)
    print("nBunch = ", nBunch)

    if bunchsize == 0:
        raise ValueError("IO_buf is too small")

    # if nBunch == 1:
    #     nBUnch = 10
    #     bunchsize = nIP_Prim // nBunch + 1

    offset = 0
    W_buf  =  np.ndarray((nIP_Prim, nIP_Prim), dtype=np.complex128, buffer=IO_buf, offset=offset)
    offset+= nIP_Prim * nIP_Prim * W_buf.itemsize
    
    aux_Buf1_offset = offset
    print("aux_Buf1_offset = ", aux_Buf1_offset//8)
    print("offset          = ", offset//8)
    print("nIP_Prim        = ", nIP_Prim)
    print("bunchsize       = ", bunchsize)
    aux_FFT_Buf1 = np.ndarray((nIP_Prim, bunchsize), dtype=np.complex128, buffer=IO_buf, offset=offset)
    offset += nIP_Prim * bunchsize * aux_FFT_Buf1.itemsize
    
    aux_Buf2_offset = offset
    print("aux_Buf2_offset = ", aux_Buf2_offset//8)
    aux_FFT_Buf2 = np.ndarray((nIP_Prim, bunchsize), dtype=np.complex128, buffer=IO_buf, offset=offset)
    offset += nIP_Prim * bunchsize * aux_FFT_Buf2.itemsize
    
    aux_Buf3_offset = offset
    print("aux_Buf3_offset = ", aux_Buf3_offset//8)
    print("offset          = ", offset//8)
    print("nIP_Prim        = ", nIP_Prim)
    print("bunchsize       = ", bunchsize)
    print("buffersize      = ", mydf.jk_buffer.size)
    aux_FFT_Buf3 = np.ndarray((nIP_Prim, bunchsize), dtype=np.complex128, buffer=IO_buf, offset=offset)
    offset += nIP_Prim * bunchsize * aux_FFT_Buf3.itemsize
    
    W_buf_fft = np.ndarray((nIP_Prim, nIP_Prim*ncell_complex), dtype=np.complex128, buffer=IO_buf, offset=offset)
    
    coulG = tools.get_coulG(mydf.cell, mesh=mesh)
    coulG = coulG.reshape(meshPrim[0], Ls[0], meshPrim[1], Ls[1], meshPrim[2], Ls[2])
    coulG = coulG.transpose(1, 3, 5, 0, 2, 4).reshape(-1, np.prod(meshPrim)).copy()
    
    fn = getattr(libpbc, "_construct_W_multiG", None)
    assert(fn is not None)
    
    icell_2_xyz = np.zeros((ncell_complex, 3), dtype=np.int32)
    
    loc = 0
    for ix in range(Ls[0]):
        for iy in range(Ls[1]):
            for iz in range(Ls[2]//2+1):
                icell_2_xyz[loc, 0] = ix
                icell_2_xyz[loc, 1] = iy
                icell_2_xyz[loc, 2] = iz
                loc += 1
    
    
    # with lib.call_in_background(load_aux_basis_fft) as prefetch:
    
    for _ in range(1):
        prefetch = load_aux_basis_fft
    
        for i in range(ncell_complex):
            
            t1 = (logger.process_clock(), logger.perf_counter())
            
            ix, iy, iz = icell_2_xyz[i]
            icell = ix * Ls[1] * (Ls[2]) + iy * (Ls[2]) + iz
            
            aux_FFT_Buf1 = np.ndarray((nIP_Prim, bunchsize), dtype=np.complex128, buffer=IO_buf, offset=aux_Buf1_offset)
            load_aux_basis_fft2(icell, 0, bunchsize, aux_FFT_Buf1)
            
            W_buf.ravel()[:] = 0.0 # reset W_buf
            
            
            
            for p0, p1 in lib.prange(0, nGridPrim, bunchsize):
                
                aux_FFT_Buf1 = np.ndarray((nIP_Prim, p1-p0), dtype=np.complex128, buffer=IO_buf, offset=aux_Buf1_offset)
                
                p2 = p1 + bunchsize
                p2 = min(p2, nGridPrim)
                aux_FFT_Buf2 = np.ndarray((nIP_Prim, p2-p1), dtype=np.complex128, buffer=IO_buf, offset=aux_Buf2_offset)
                
                prefetch(icell, p1, p2, aux_FFT_Buf2)
                
                ### do the work ###
                
                aux_buf3 = np.ndarray((nIP_Prim, p1-p0), dtype=np.complex128, buffer=IO_buf, offset=aux_Buf3_offset)
                
                # copy 1 to 3
                
                aux_buf3.ravel()[:] = aux_FFT_Buf1.ravel()[:]
                                
                fn(
                    ctypes.c_int(nIP_Prim),
                    ctypes.c_int(p0),
                    ctypes.c_int(p1),
                    aux_buf3.ctypes.data_as(ctypes.c_void_p),
                    coulG[icell].ctypes.data_as(ctypes.c_void_p)
                )
                                
                lib.dot(aux_FFT_Buf1, aux_buf3.T.conj(), beta=1, c=W_buf)
                
                aux_Buf2_offset, aux_Buf1_offset = aux_Buf1_offset, aux_Buf2_offset
                
            k_begin = i * nIP_Prim
            k_end   = (i + 1) * nIP_Prim
                
            W[:, k_begin:k_end] = W_buf
            
            t2 = (logger.process_clock(), logger.perf_counter())
            
            print("cell %5d wall time: %12.6f CPU time: %12.6f" % (i, t2[1] - t1[1], t2[0] - t1[0]))
    
    fn = getattr(libpbc, "_iFFT_Matrix_Col_InPlace", None)
    assert(fn is not None)
    
    fn(
        W.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_Prim),
        ctypes.c_int(nIP_Prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        W_buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    W_real = np.ndarray((nIP_Prim, nIP_Prim*ncell), dtype=np.float64, buffer=W, offset = 0)
    
    mydf.W = W_real

    ### used in get J ### 
    
    W0 = np.zeros((nIP_Prim, nIP_Prim), dtype=np.float64)
    
    for i in range(ncell):
            
            k_begin = i * nIP_Prim
            k_end   = (i + 1) * nIP_Prim
            
            W0 += W_real[:, k_begin:k_end]

    mydf.W0 = W0

    return W_real

####################### construct V W #######################

def _construct_V_W_incore(mydf:ISDF.PBC_ISDF_Info):
    
    mydf._allocate_jk_buffer()
    
    mesh  = mydf.mesh
    coulG = tools.get_coulG(mydf.cell, mesh=mesh)
    Ls    = mydf.Ls
    Ls    = np.array(Ls, dtype=np.int32)
    mesh_prim = np.array(mesh) // np.array(Ls)
    mesh_prim = np.array(mesh_prim, dtype=np.int32)
    coulG = coulG.reshape(mesh_prim[0], Ls[0], mesh_prim[1], Ls[1], mesh_prim[2], Ls[2])
    coulG = coulG.transpose(1, 3, 5, 0, 2, 4).reshape(-1, np.prod(mesh_prim)).copy()
    
    nIP_prim      = mydf.nIP_Prim
    nGrid_prim    = mydf.nGridPrim
    ncell         = np.prod(Ls)
    ncell_complex = Ls[0] * Ls[1] * (Ls[2]//2+1)
    
    #### allocate buffer ####
    
    W = np.zeros((nIP_prim, nIP_prim*ncell), dtype=np.float64)
    V2 = np.zeros((nIP_prim, nGrid_prim*ncell_complex), dtype=np.complex128)
    V = np.ndarray((nIP_prim, nGrid_prim*ncell), dtype=np.float64, buffer=V2)
    V_fftbuf = np.zeros((nIP_prim, nGrid_prim*ncell_complex), dtype=np.complex128)
    
    # print("nIP_prim = ", nIP_prim)
    # print("nGrid_prim = ", nGrid_prim)
    
    offset  = 0
    A_buf   = np.ndarray((nIP_prim, nGrid_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=0)
    offset += nIP_prim * nGrid_prim * A_buf.itemsize
    # print("offset = ", offset//8)
    B_buf   = np.ndarray((nIP_prim, nGrid_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    offset += nIP_prim * nGrid_prim * B_buf.itemsize
    # print("offset = ", offset//8)
    W_buf   = np.ndarray((nIP_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    offset += nIP_prim * nIP_prim * W_buf.itemsize
    # print("offset = ", offset//8)
    # print("ncell_complex = ", ncell_complex)
    # print(mydf.jk_buffer.size)
    W_buf2  = np.ndarray((nIP_prim, nIP_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)   
    W_buf3  = np.ndarray((nIP_prim, nIP_prim*ncell), dtype=np.float64, buffer=mydf.jk_buffer, offset=offset)
    offset += nIP_prim * nIP_prim * ncell_complex * W_buf2.itemsize
    W_buf_fft = np.ndarray((nIP_prim, nIP_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)

    fn = getattr(libpbc, "_construct_W_multiG", None)
    assert(fn is not None)

    # for i in range(ncell):
    
    fn_FREQ = getattr(libpbc, "_FREQ", None)
    assert fn_FREQ is not None
    
    FREQ = np.zeros((ncell_complex, nGrid_prim), dtype=np.complex128)
    
    # print("mesh_prim = ", mesh_prim)
    
    fn_FREQ(
        FREQ.ctypes.data_as(ctypes.c_void_p),
        mesh_prim.ctypes.data_as(ctypes.c_void_p),
        Ls.ctypes.data_as(ctypes.c_void_p)
    )
    
    # print("mesh_prim = ", mesh_prim)
    
    fn_final_ifft = getattr(libpbc, "_FinaliFFT", None)
    assert fn_final_ifft is not None
    
    nthread = lib.num_threads()
    buffer_final_fft = np.ndarray((nthread, nGrid_prim), dtype=np.complex128)
    
    loc = 0
    for ix in range(Ls[0]):
        for iy in range(Ls[1]):
            for iz in range(Ls[2]//2+1):
                
                i = ix * Ls[1] * (Ls[2]) + iy * (Ls[2]) + iz
                
                k_begin = i * nGrid_prim
                k_end   = (i + 1) * nGrid_prim
        
                A_buf[:] = mydf.aux_basis[:, k_begin:k_end]

                B_buf[:] = A_buf[:]

                fn(
                    ctypes.c_int(nIP_prim),
                    ctypes.c_int(0),
                    ctypes.c_int(nGrid_prim),
                    B_buf.ctypes.data_as(ctypes.c_void_p),
                    coulG[i].ctypes.data_as(ctypes.c_void_p)
                )
        
                # print("A_buf.shape = ", A_buf.shape)
                # print("B_buf.shape = ", B_buf.shape)
                # print("W_buf.shape = ", W_buf.shape)
                # print("coulG = ", coulG[i])
                # print("A_buf = ", A_buf)
                # print("B_buf = ", B_buf)
            
                # lib.dot(B_buf, A_buf.T.conj(), c=W_buf)
                lib.dot(A_buf, B_buf.T.conj(), c=W_buf)
                
                # print("W_buf = ", W_buf)

                k_begin = loc * nIP_prim
                k_end   = (loc + 1) * nIP_prim

                W_buf2[:, k_begin:k_end] = W_buf

                # print("B_Buf = ", B_buf)
                
                # print("mesh_prim = ", mesh_prim)
                fn_final_ifft(
                    B_buf.ctypes.data_as(ctypes.c_void_p),
                    FREQ[loc].ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nIP_prim),
                    ctypes.c_int(nGrid_prim),
                    mesh_prim.ctypes.data_as(ctypes.c_void_p),
                    buffer_final_fft.ctypes.data_as(ctypes.c_void_p)
                )
                
                # print("B_Buf = ", B_buf)
                
                k_begin = loc * nGrid_prim
                k_end   = (loc + 1) * nGrid_prim
                
                V2[:, k_begin:k_end] = B_buf
            
                loc += 1
    
    fn = getattr(libpbc, "_iFFT_Matrix_Col_InPlace", None)
    assert(fn is not None)
    
    fn(
        W_buf2.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_prim),
        ctypes.c_int(nIP_prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        W_buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    W_buf_fft = None
    
    fn(
        V2.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_prim),
        ctypes.c_int(nGrid_prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        V_fftbuf.ctypes.data_as(ctypes.c_void_p)
    )
    
    V_fftbuf = None
    
    W.ravel()[:] = W_buf3.ravel()[:]
    mydf.W = W
    mydf.V_R = V.copy() * np.sqrt(np.prod(mesh_prim))
    V = None
    V2 = None

    ### used in get J ### 
    
    W0 = np.zeros((nIP_prim, nIP_prim), dtype=np.float64)
    V0 = np.zeros((nIP_prim, nGrid_prim), dtype=np.float64)
    for i in range(ncell):
            
        k_begin = i * nIP_prim
        k_end   = (i + 1) * nIP_prim
            
        W0 += W[:, k_begin:k_end]
        V0 += mydf.V_R[:, i*nGrid_prim:(i+1)*nGrid_prim]

    mydf.W0 = W0
    mydf.V0 = V0

    return W


####################### get_j #######################
    
def _get_j_with_Wgrid(mydf:ISDF.PBC_ISDF_Info, W_grid=None, dm=None):
    
    if W_grid is None:
        if hasattr(mydf, "W_grid"):
            W_grid = mydf.W_grid
        else:
            mydf.construct_auxbasis_benchmark()
            aux_basis = mydf.aux_basis_bench_Grid
            _, W_grid = _construct_W_benchmark_grid(mydf.cell, aux_basis, mydf.Ls, mydf.mesh)
            mydf.W_grid = W_grid
    print("W_grid.shape = ", W_grid.shape)
    
    W_backup = mydf.W
    mydf.W   = W_grid
    J = ISDF_outcore._get_j_dm_wo_robust_fitting(mydf, dm)
    mydf.W   = W_backup
    return J

def _get_k_with_Wgrid(mydf:ISDF.PBC_ISDF_Info, W_grid=None, dm=None):
    
    # W_backup = mydf.W
    # mydf.W   = W_grid
    # K = ISDF_outcore._get_k_dm_wo_robust_fitting(mydf, dm)
    # mydf.W   = W_backup
    
    if W_grid is None:
        W_grid = mydf.W_grid
    
    t1 = (logger.process_clock(), logger.perf_counter())
    
    aoRg = mydf.aoRg
    
    density_RgRg = lib.dot(dm, aoRg)
    density_RgRg = lib.dot(aoRg.T, density_RgRg)
    lib.cwise_mul(W_grid, density_RgRg, out=density_RgRg)
    K = lib.dot(density_RgRg, aoRg.T)
    K = lib.dot(aoRg, K)
    
    ngrid = np.prod(mydf.cell.mesh)
    vol = mydf.cell.vol
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    _benchmark_time(t1, t2, "_contract_k_dm")
    
    return K * ngrid / vol
   
# @profile  
def _pack_JK(input_mat:np.ndarray, Ls, nao_prim, output=None):
    
    assert input_mat.dtype == np.float64    
    ncell = np.prod(Ls)
    # print("ncell = ", ncell)
    # print("Ls = ", Ls)  
    # print("nao_prim = ", nao_prim)
    # print("input_mat.shape = ", input_mat.shape)
    assert input_mat.shape[0] == nao_prim
    assert input_mat.shape[1] == nao_prim * ncell
    
    if output is None:
        output = np.zeros((ncell*nao_prim, ncell*nao_prim), dtype=np.float64)  
    else:
        assert output.shape == (ncell*nao_prim, ncell*nao_prim)  
    
    for ix_row in range(Ls[0]):
        for iy_row in range(Ls[1]):
            for iz_row in range(Ls[2]):
                
                loc_row = ix_row * Ls[1] * Ls[2] + iy_row * Ls[2] + iz_row
                
                b_begin = loc_row * nao_prim
                b_end   = (loc_row + 1) * nao_prim
                
                for ix_col in range(Ls[0]):
                    for iy_col in range(Ls[1]):
                        for iz_col in range(Ls[2]):
                            
                            loc_col = ix_col * Ls[1] * Ls[2] + iy_col * Ls[2] + iz_col
                            
                            k_begin = loc_col * nao_prim
                            k_end   = (loc_col + 1) * nao_prim
                            
                            ix = (ix_col - ix_row) % Ls[0]
                            iy = (iy_col - iy_row) % Ls[1]
                            iz = (iz_col - iz_row) % Ls[2]
                            
                            loc_col2 = ix * Ls[1] * Ls[2] + iy * Ls[2] + iz
                            
                            k_begin2 = loc_col2 * nao_prim
                            k_end2   = (loc_col2 + 1) * nao_prim
                            
                            output[b_begin:b_end, k_begin:k_end] = input_mat[:, k_begin2:k_end2]
                            
    return output
    
# @profile    
def _get_j_kSym(mydf:ISDF.PBC_ISDF_Info, dm):
    
    ### preprocess
    
    mydf._allocate_jk_buffer(dm.dtype)
    
    t1 = (logger.process_clock(), logger.perf_counter())
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]
    
    nao  = dm.shape[0]
    cell = mydf.cell
    assert cell.nao == nao
    ngrid = np.prod(cell.mesh)
    vol = cell.vol
    
    W         = mydf.W
    aoRg      = mydf.aoRg
    aoRg_Prim = mydf.aoRg_Prim
    # aoR_Prim  = mydf.aoR_Prim
    naux      = aoRg.shape[1]
    
    Ls = mydf.Ls
    mesh = mydf.mesh
    meshPrim = np.array(mesh) // np.array(Ls)
    nGridPrim = mydf.nGridPrim
    ncell = np.prod(Ls)
    ncell_complex = Ls[0] * Ls[1] * (Ls[2]//2+1)
    nIP_prim = mydf.nIP_Prim
    nao_prim = nao // ncell
    
    ### allocate buffer
    
    assert dm.dtype == np.float64
    
    buffer  = mydf.jk_buffer
    
    buffer1 = np.ndarray((nao,nIP_prim),  dtype=dm.dtype, buffer=buffer, offset=0) 
    buffer2 = np.ndarray((nIP_prim),      dtype=dm.dtype, buffer=buffer, offset=(nao * nIP_prim) * dm.dtype.itemsize)
    
    offset  = (nao * nIP_prim + nIP_prim) * dm.dtype.itemsize
    buffer3 = np.ndarray((nao_prim,nao),   dtype=np.float64, buffer=buffer, offset=offset)
    
    offset += (nao_prim * nao) * dm.dtype.itemsize
    bufferW = np.ndarray((nIP_prim,1), dtype=np.float64, buffer=buffer, offset=offset)
    
    offset += (nIP_prim) * dm.dtype.itemsize
    bufferJ_block = np.ndarray((nao_prim, nao_prim), dtype=np.float64, buffer=buffer, offset=offset)
    
    offset += (nao_prim * nao_prim) * dm.dtype.itemsize
    bufferi = np.ndarray((nao_prim,nIP_prim), dtype=np.float64, buffer=buffer, offset=offset)
    
    offset += (nao_prim * nIP_prim) * dm.dtype.itemsize
    bufferj = np.ndarray((nao_prim,nIP_prim), dtype=np.float64, buffer=buffer, offset=offset)
    
    offset += (nao_prim * nIP_prim) * dm.dtype.itemsize
    buffer4 = np.ndarray((nao_prim,nIP_prim), dtype=np.float64, buffer=buffer, offset=offset)
    
    lib.ddot(dm, aoRg_Prim, c=buffer1)
    tmp1 = buffer1
    density_Rg = np.asarray(lib.multiply_sum_isdf(aoRg_Prim, tmp1, out=buffer2), order='C')
    
    ### check the symmetry of density_Rg
    
    W_0 = mydf.W0
    
    lib.ddot(W_0, density_Rg.reshape(-1,1), c=bufferW)
    bufferW = bufferW.reshape(-1)
    buffer_J = buffer3
    
    for ix_q in range(Ls[0]):
        for iy_q in range(Ls[1]):
            for iz_q in range(Ls[2]):
                
                bufferJ_block.ravel()[:] = 0.0 # set to zero
    
                ### loop over the blocks
                
                for ix in range(Ls[0]):
                    for iy in range(Ls[1]):
                        for iz in range(Ls[2]):
                            
                            ipx = (Ls[0] - ix) % Ls[0]
                            ipy = (Ls[1] - iy) % Ls[1]
                            ipz = (Ls[2] - iz) % Ls[2]
                            
                            loc_p = ipx * Ls[1] * Ls[2] + ipy * Ls[2] + ipz
                            
                            begin = loc_p * nao_prim
                            end   = (loc_p + 1) * nao_prim
                            
                            buffer_i = aoRg_Prim[begin:end, :]
                            
                            iqx = (ix_q - ix + Ls[0]) % Ls[0]
                            iqy = (iy_q - iy + Ls[1]) % Ls[1]
                            iqz = (iz_q - iz + Ls[2]) % Ls[2]
                            
                            loc_q = iqx * Ls[1] * Ls[2] + iqy * Ls[2] + iqz
                            
                            begin = loc_q * nao_prim
                            end   = (loc_q + 1) * nao_prim
                            
                            buffer_j = aoRg_Prim[begin:end, :]
                            tmp = np.asarray(lib.d_ij_j_ij(buffer_j, bufferW, out=buffer4), order='C')
                            lib.ddot_withbuffer(buffer_i, tmp.T, c=bufferJ_block, beta=1, buf=mydf.ddot_buf)
                
                ### set ### 
                
                loc_q = ix_q * Ls[1] * Ls[2] + iy_q * Ls[2] + iz_q
                
                begin = loc_q * nao_prim
                end   = (loc_q + 1) * nao_prim
                
                buffer_J[:, begin:end] = bufferJ_block
    
    buffer_J = buffer_J * (ngrid / vol)
    
    J = _pack_JK(buffer_J, Ls, nao_prim, output=None)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    _benchmark_time(t1, t2, "_contract_j_dm")
    
    return J
    

def _get_j_kSym_robust_fitting(mydf:ISDF.PBC_ISDF_Info, dm):
    
    ### preprocess
    
    mydf._allocate_jk_buffer(dm.dtype)
    
    t1 = (logger.process_clock(), logger.perf_counter())
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]
    
    nao  = dm.shape[0]
    cell = mydf.cell
    assert cell.nao == nao
    ngrid = np.prod(cell.mesh)
    vol = cell.vol
    
    W         = mydf.W
    V_R       = mydf.V_R
    aoRg      = mydf.aoRg
    aoRg_Prim = mydf.aoRg_Prim
    aoR_Prim  = mydf.aoR
    naux      = aoRg.shape[1]
    
    Ls = mydf.Ls
    mesh = mydf.mesh
    meshPrim = np.array(mesh) // np.array(Ls)
    nGridPrim = mydf.nGridPrim
    ncell = np.prod(Ls)
    ncell_complex = Ls[0] * Ls[1] * (Ls[2]//2+1)
    nIP_prim = mydf.nIP_Prim
    nao_prim = nao // ncell
    
    ### allocate buffer
    
    assert dm.dtype == np.float64
    
    # buffer  = mydf.jk_buffer
    # buffer1 = np.ndarray((nao,nIP_prim),  dtype=dm.dtype, buffer=buffer, offset=0) 
    # buffer2 = np.ndarray((nIP_prim),      dtype=dm.dtype, buffer=buffer, offset=(nao * nIP_prim) * dm.dtype.itemsize)
    # offset  = (nao * nIP_prim + nIP_prim) * dm.dtype.itemsize
    # buffer3 = np.ndarray((nao_prim,nao),   dtype=np.float64, buffer=buffer, offset=offset)
    # offset += (nao_prim * nao) * dm.dtype.itemsize
    # bufferW = np.ndarray((nIP_prim,1), dtype=np.float64, buffer=buffer, offset=offset)
    # offset += (nIP_prim) * dm.dtype.itemsize
    # bufferJ_block = np.ndarray((nao_prim, nao_prim), dtype=np.float64, buffer=buffer, offset=offset)
    bufferJ_block = np.zeros((nao_prim, nao_prim), dtype=np.float64)
    # offset += (nao_prim * nao_prim) * dm.dtype.itemsize
    # bufferi = np.ndarray((nao_prim,nIP_prim), dtype=np.float64, buffer=buffer, offset=offset)
    # offset += (nao_prim * nIP_prim) * dm.dtype.itemsize
    # bufferj = np.ndarray((nao_prim,nIP_prim), dtype=np.float64, buffer=buffer, offset=offset)
    # offset += (nao_prim * nIP_prim) * dm.dtype.itemsize
    # buffer4 = np.ndarray((nao_prim,nIP_prim), dtype=np.float64, buffer=buffer, offset=offset)
    
    # lib.ddot(dm, aoRg_Prim, c=buffer1)
    # tmp1 = buffer1
    # density_Rg = np.asarray(lib.multiply_sum_isdf(aoRg_Prim, tmp1, out=buffer2), order='C')
    
    density_Rg = lib.ddot(dm, aoRg_Prim)
    density_Rg = np.asarray(lib.multiply_sum_isdf(aoRg_Prim, density_Rg), order='C')
    
    density_R  = lib.ddot(dm, aoR_Prim)
    density_R = np.asarray(lib.multiply_sum_isdf(aoR_Prim, density_R), order='C')
    
    J = np.zeros((nao_prim,nao), dtype=np.float64)
    
    ### construct J1 part 
    
    V_0 = mydf.V0
    bufferV  = lib.ddot(V_0.T, density_Rg.reshape(-1,1)).reshape(-1)
    buffer_J = np.zeros((nao_prim,nao), dtype=np.float64)
    
    for ix_q in range(Ls[0]):
        for iy_q in range(Ls[1]):
            for iz_q in range(Ls[2]):
                
                bufferJ_block.ravel()[:] = 0.0 # set to zero
    
                ### loop over the blocks
                
                for ix in range(Ls[0]):
                    for iy in range(Ls[1]):
                        for iz in range(Ls[2]):
                            
                            ipx = (Ls[0] - ix) % Ls[0]
                            ipy = (Ls[1] - iy) % Ls[1]
                            ipz = (Ls[2] - iz) % Ls[2]
                            
                            loc_p = ipx * Ls[1] * Ls[2] + ipy * Ls[2] + ipz
                            
                            begin = loc_p * nao_prim
                            end   = (loc_p + 1) * nao_prim
                            
                            buffer_i = aoR_Prim[begin:end, :]
                            
                            iqx = (ix_q - ix + Ls[0]) % Ls[0]
                            iqy = (iy_q - iy + Ls[1]) % Ls[1]
                            iqz = (iz_q - iz + Ls[2]) % Ls[2]
                            
                            loc_q = iqx * Ls[1] * Ls[2] + iqy * Ls[2] + iqz
                            
                            begin = loc_q * nao_prim
                            end   = (loc_q + 1) * nao_prim
                            
                            buffer_j = aoR_Prim[begin:end, :]
                            tmp = np.asarray(lib.d_ij_j_ij(buffer_j, bufferV), order='C')
                            lib.ddot(buffer_i, tmp.T, c=bufferJ_block, beta=1)
                            tmp = None
                            buffer_j = None
                            buffer_i = None
                
                ### set ### 
                
                loc_q = ix_q * Ls[1] * Ls[2] + iy_q * Ls[2] + iz_q
                
                begin = loc_q * nao_prim
                end   = (loc_q + 1) * nao_prim
                
                buffer_J[:, begin:end] = bufferJ_block
    
    J += buffer_J * (ngrid / vol)
    
    ### construct J2 - W part 
    
    W_0 = mydf.W0
    V_0 = mydf.V0
    bufferW  = lib.ddot(W_0, density_Rg.reshape(-1,1)).reshape(-1)
    bufferV  = lib.ddot(V_0, density_R.reshape(-1,1)).reshape(-1)
    bufferW -= bufferV
    # bufferW = -bufferW
    buffer_J = np.zeros((nao_prim,nao), dtype=np.float64)
    
    for ix_q in range(Ls[0]):
        for iy_q in range(Ls[1]):
            for iz_q in range(Ls[2]):
                
                bufferJ_block.ravel()[:] = 0.0 # set to zero
    
                ### loop over the blocks
                
                for ix in range(Ls[0]):
                    for iy in range(Ls[1]):
                        for iz in range(Ls[2]):
                            
                            ipx = (Ls[0] - ix) % Ls[0]
                            ipy = (Ls[1] - iy) % Ls[1]
                            ipz = (Ls[2] - iz) % Ls[2]
                            
                            loc_p = ipx * Ls[1] * Ls[2] + ipy * Ls[2] + ipz
                            
                            begin = loc_p * nao_prim
                            end   = (loc_p + 1) * nao_prim
                            
                            buffer_i = aoRg_Prim[begin:end, :]
                            
                            iqx = (ix_q - ix + Ls[0]) % Ls[0]
                            iqy = (iy_q - iy + Ls[1]) % Ls[1]
                            iqz = (iz_q - iz + Ls[2]) % Ls[2]
                            
                            loc_q = iqx * Ls[1] * Ls[2] + iqy * Ls[2] + iqz
                            
                            begin = loc_q * nao_prim
                            end   = (loc_q + 1) * nao_prim
                            
                            buffer_j = aoRg_Prim[begin:end, :]
                            tmp = np.asarray(lib.d_ij_j_ij(buffer_j, bufferW), order='C')
                            # lib.ddot_withbuffer(buffer_i, tmp.T, c=bufferJ_block, beta=1, buf=mydf.ddot_buf)
                            lib.ddot(buffer_i, tmp.T, c=bufferJ_block, beta=1)
                            tmp = None
                            buffer_j = None
                            buffer_i = None
                            
                
                ### set ### 
                
                loc_q = ix_q * Ls[1] * Ls[2] + iy_q * Ls[2] + iz_q
                
                begin = loc_q * nao_prim
                end   = (loc_q + 1) * nao_prim
                
                buffer_J[:, begin:end] = bufferJ_block
    
    J -= buffer_J * (ngrid / vol)
    
    J = _pack_JK(J, Ls, nao_prim, output=None)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    _benchmark_time(t1, t2, "_contract_j_dm")
    
    return J
 

def _get_DM_RgRg_benchmark(mydf:ISDF.PBC_ISDF_Info, dm):
    
    aoRg = mydf.aoRg
    
    tmp1 = lib.ddot(dm, aoRg, c=None)
    
    return lib.ddot(aoRg.T, tmp1, c=None)
    
def _get_DM_RgRg_real(mydf:ISDF.PBC_ISDF_Info, dm_real, dm_complex, dm_RgRg_real, dm_RgRg_complex, offset):
    
    nao   = mydf.nao
    Ls    = np.array(mydf.Ls, dtype=np.int32)
    ncell = np.prod(Ls)
    ncell_complex = Ls[0] * Ls[1] * (Ls[2]//2+1)
    nao_prim  = nao // ncell
    nIP_prim  = mydf.nIP_Prim
    mesh      = np.array(mydf.mesh, dtype=np.int32)  
    meshPrim  = np.array(mesh) // np.array(Ls)
    nGridPrim = mydf.nGridPrim
    
    fn1 = getattr(libpbc, "_FFT_Matrix_Col_InPlace", None)
    assert fn is not None
    
    buf_fft = np.ndarray((nao_prim, nao_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
        
    fn1(
        dm_real.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao_prim),
        ctypes.c_int(nao_prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    buf_A = np.ndarray((nao_prim, nao_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    offset2 = offset + (nao_prim * nao_prim) * buf_A.itemsize
    buf_B = np.ndarray((nao_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset2)
    
    offset3 = offset2 + (nao_prim * nIP_prim) * buf_B.itemsize
    buf_C = np.ndarray((nao_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset3)
    
    offset4 = offset3 + (nao_prim * nIP_prim) * buf_C.itemsize
    buf_D = np.ndarray((nIP_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset4)
    
    aoRg_FFT = mydf.aoRg_FFT
    
    for i in range(ncell_complex):
        
        k_begin = i * nao_prim
        k_end   = (i + 1) * nao_prim
        
        buf_A[:] = dm_complex[:, k_begin:k_end]
        buf_B[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim]
        
        lib.dot(buf_A, buf_B, c=buf_C)
        lib.dot(buf_B.T.conj(), buf_C, c=buf_D)
        
        k_begin = i * nIP_prim
        k_end   = (i + 1) * nIP_prim
        
        dm_RgRg_complex[:, k_begin:k_end] = buf_D
    
    buf_fft = np.ndarray((nIP_prim, nIP_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    fn2 = getattr(libpbc, "_iFFT_Matrix_Col_InPlace", None)
    assert fn is not None
    
    fn2(
        dm_RgRg_complex.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_prim),
        ctypes.c_int(nIP_prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    # print("dm_RgRg_complex = ", dm_RgRg_complex[:5,:5])
    
    return dm_RgRg_real

# @profile     
def _get_k_kSym(mydf:ISDF.PBC_ISDF_Info, dm):
 
    #### preprocess ####  
    
    mydf._allocate_jk_buffer(dm.dtype)
    t1 = (logger.process_clock(), logger.perf_counter())
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]
    
    nao  = dm.shape[0]
    cell = mydf.cell    
    assert cell.nao == nao
    ngrid = np.prod(cell.mesh)
    vol = cell.vol
    
    W         = mydf.W
    # aoRg      = mydf.aoRg
    # aoRg_Prim = mydf.aoRg_Prim
    # naux      = aoRg.shape[1]
    naux = mydf.naux
    
    Ls = np.array(mydf.Ls, dtype=np.int32)
    mesh = mydf.mesh
    meshPrim = np.array(mesh) // np.array(Ls)
    nGridPrim = mydf.nGridPrim
    ncell = np.prod(Ls)
    ncell_complex = Ls[0] * Ls[1] * (Ls[2]//2+1)
    nIP_prim = mydf.nIP_Prim
    nao_prim = nao // ncell
    
    #### allocate buffer ####
    
    offset = 0
    DM_RgRg_complex = np.ndarray((nIP_prim,nIP_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    DM_RgRg_real = np.ndarray((nIP_prim,nIP_prim*ncell), dtype=np.float64, buffer=mydf.jk_buffer, offset=offset)
    
    offset += (nIP_prim * nIP_prim * ncell_complex) * DM_RgRg_complex.itemsize
    DM_complex = np.ndarray((nao_prim,nao_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    DM_real = np.ndarray((nao_prim,nao), dtype=np.float64, buffer=mydf.jk_buffer, offset=offset)
    DM_real.ravel()[:] = dm[:nao_prim, :].ravel()[:]
    offset += (nao_prim * nao_prim * ncell_complex) * DM_complex.itemsize
    
    #### get D ####
    
    #_get_DM_RgRg_real(mydf, DM_real, DM_complex, DM_RgRg_real, DM_RgRg_complex, offset)
    
    fn1 = getattr(libpbc, "_FFT_Matrix_Col_InPlace", None)
    assert fn1 is not None
    
    buf_fft = np.ndarray((nao_prim, nao_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
        
    fn1(
        DM_real.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao_prim),
        ctypes.c_int(nao_prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    buf_A = np.ndarray((nao_prim, nao_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    offset2 = offset + (nao_prim * nao_prim) * buf_A.itemsize
    buf_B = np.ndarray((nao_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset2)
    
    offset3 = offset2 + (nao_prim * nIP_prim) * buf_B.itemsize
    buf_C = np.ndarray((nao_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset3)
    
    offset4 = offset3 + (nao_prim * nIP_prim) * buf_C.itemsize
    buf_D = np.ndarray((nIP_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset4)
    
    aoRg_FFT = mydf.aoRg_FFT
    
    if isinstance(aoRg_FFT, list): 
        for i in range(ncell_complex):
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            buf_A[:] = DM_complex[:, k_begin:k_end]
            # buf_B[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim]
            buf_B = aoRg_FFT[i]
        
            lib.dot(buf_A, buf_B, c=buf_C)
            lib.dot(buf_B.T.conj(), buf_C, c=buf_D)
        
            k_begin = i * nIP_prim
            k_end   = (i + 1) * nIP_prim
        
            DM_RgRg_complex[:, k_begin:k_end] = buf_D
    else:
        for i in range(ncell_complex):
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            buf_A[:] = DM_complex[:, k_begin:k_end]
            buf_B[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim]
        
            lib.dot(buf_A, buf_B, c=buf_C)
            lib.dot(buf_B.T.conj(), buf_C, c=buf_D)
        
            k_begin = i * nIP_prim
            k_end   = (i + 1) * nIP_prim
        
            DM_RgRg_complex[:, k_begin:k_end] = buf_D
    
    buf_fft = np.ndarray((nIP_prim, nIP_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    fn2 = getattr(libpbc, "_iFFT_Matrix_Col_InPlace", None)
    assert fn2 is not None
    
    fn2(
        DM_RgRg_complex.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_prim),
        ctypes.c_int(nIP_prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    # inplace multiplication
    
    lib.cwise_mul(mydf.W, DM_RgRg_real, out=DM_RgRg_real)
    
    offset = nIP_prim * nIP_prim * ncell_complex * DM_RgRg_complex.itemsize
    
    buf_fft = np.ndarray((nIP_prim, nIP_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    fn1(
        DM_RgRg_real.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_prim),
        ctypes.c_int(nIP_prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    K_complex_buf = np.ndarray((nao_prim, nao_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    K_real_buf    = np.ndarray((nao_prim, nao_prim*ncell), dtype=np.float64, buffer=mydf.jk_buffer, offset=offset)
    offset += (nao_prim * nao_prim * ncell_complex) * K_complex_buf.itemsize
    offset_now = offset    
    
    buf_A = np.ndarray((nIP_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_now)
    offset_now += (nIP_prim * nIP_prim) * buf_A.itemsize
    buf_B = np.ndarray((nao_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_now)
    offset_now += (nao_prim * nIP_prim) * buf_B.itemsize
    buf_C = np.ndarray((nIP_prim, nao_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_now)
    offset_now += (nIP_prim * nao_prim) * buf_C.itemsize
    buf_D = np.ndarray((nao_prim, nao_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_now)
    
    if isinstance(aoRg_FFT, list): 
        
        for i in range(ncell_complex):
        
            k_begin = i * nIP_prim
            k_end   = (i + 1) * nIP_prim
        
            buf_A.ravel()[:] = DM_RgRg_complex[:, k_begin:k_end].ravel()[:]
            # buf_B.ravel()[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim].ravel()[:]
            buf_B = aoRg_FFT[i]
        
            lib.dot(buf_A, buf_B.T.conj(), c=buf_C)
            lib.dot(buf_B, buf_C, c=buf_D)
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            K_complex_buf[:, k_begin:k_end] = buf_D
    else:
        
        for i in range(ncell_complex):
        
            k_begin = i * nIP_prim
            k_end   = (i + 1) * nIP_prim
        
            buf_A.ravel()[:] = DM_RgRg_complex[:, k_begin:k_end].ravel()[:]
            buf_B.ravel()[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim].ravel()[:]
        
            lib.dot(buf_A, buf_B.T.conj(), c=buf_C)
            lib.dot(buf_B, buf_C, c=buf_D)
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            K_complex_buf[:, k_begin:k_end] = buf_D
    
    buf_fft = np.ndarray((nao_prim, nao_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    fn2(
        K_complex_buf.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao_prim),
        ctypes.c_int(nao_prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    K_real_buf *= (ngrid / vol)
    
    K = _pack_JK(K_real_buf, Ls, nao_prim, output=None)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    _benchmark_time(t1, t2, "_contract_k_dm")
    
    return K
    
    # return DM_RgRg_real # temporary return for debug

def _get_k_kSym_robust_fitting(mydf:ISDF.PBC_ISDF_Info, dm):
    
    '''
    this is a slow version, abandon ! 
    '''
 
    #### preprocess ####  
    
    mydf._allocate_jk_buffer(dm.dtype)
    t1 = (logger.process_clock(), logger.perf_counter())
    
    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]
    
    nao  = dm.shape[0]
    cell = mydf.cell    
    assert cell.nao == nao
    ngrid = np.prod(cell.mesh)
    vol = cell.vol
    
    W         = mydf.W
    # aoRg      = mydf.aoRg
    # aoRg_Prim = mydf.aoRg_Prim
    # naux      = aoRg.shape[1]
    naux = mydf.naux
    
    Ls = np.array(mydf.Ls, dtype=np.int32)
    mesh = mydf.mesh
    meshPrim = np.array(mesh) // np.array(Ls)
    nGridPrim = mydf.nGridPrim
    ncell = np.prod(Ls)
    ncell_complex = Ls[0] * Ls[1] * (Ls[2]//2+1)
    nIP_prim = mydf.nIP_Prim
    nao_prim = nao // ncell
    
    #### allocate buffer ####
     
    
    offset = 0
    DM_RgRg_complex = np.ndarray((nIP_prim,nIP_prim*ncell_complex), dtype=np.complex128)
    DM_RgRg_real = np.ndarray((nIP_prim,nIP_prim*ncell), dtype=np.float64, buffer=DM_RgRg_complex)
    
    offset += (nIP_prim * nIP_prim * ncell_complex) * DM_RgRg_complex.itemsize
    DM_complex = np.ndarray((nao_prim,nao_prim*ncell_complex), dtype=np.complex128)
    DM_real = np.ndarray((nao_prim,nao), dtype=np.float64, buffer=DM_complex, offset=0)
    DM_real.ravel()[:] = dm[:nao_prim, :].ravel()[:]
    offset += (nao_prim * nao_prim * ncell_complex) * DM_complex.itemsize
    
    #### get D ####
    
    #_get_DM_RgRg_real(mydf, DM_real, DM_complex, DM_RgRg_real, DM_RgRg_complex, offset)
    
    fn1 = getattr(libpbc, "_FFT_Matrix_Col_InPlace", None)
    assert fn1 is not None
    
    fn_packcol2 = getattr(libpbc, "_buildK_packcol2", None)
    assert fn_packcol2 is not None
    
    buf_fft = np.ndarray((nao_prim, nao_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    t3 = (logger.process_clock(), logger.perf_counter())
    
    fn1(
        DM_real.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao_prim),
        ctypes.c_int(nao_prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "_fft1")
    
    buf_A = np.ndarray((nao_prim, nao_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    offset2 = offset + (nao_prim * nao_prim) * buf_A.itemsize
    buf_B = np.ndarray((nao_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset2)
    
    offset3 = offset2 + (nao_prim * nIP_prim) * buf_B.itemsize
    buf_C = np.ndarray((nao_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset3)
    
    offset4 = offset3 + (nao_prim * nIP_prim) * buf_C.itemsize
    buf_D = np.ndarray((nIP_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset4)
    
    aoRg_FFT = mydf.aoRg_FFT
    
    t3 = (logger.process_clock(), logger.perf_counter())
    
    if isinstance(aoRg_FFT, list):
        
        for i in range(ncell_complex):
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            # buf_A[:] = DM_complex[:, k_begin:k_end]
            fn_packcol2(
                buf_A.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_prim),
                ctypes.c_int(2*nao_prim),
                DM_complex.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(DM_complex.shape[0]),
                ctypes.c_int(2*DM_complex.shape[1]),
                ctypes.c_int(2*k_begin),
                ctypes.c_int(2*k_end)
            )
            # buf_B[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim]
            buf_B.ravel()[:] = aoRg_FFT[i].ravel()[:]
        
            lib.dot(buf_A, buf_B, c=buf_C)
            lib.dot(buf_B.T.conj(), buf_C, c=buf_D)
        
            k_begin = i * nIP_prim
            k_end   = (i + 1) * nIP_prim
        
            DM_RgRg_complex[:, k_begin:k_end] = buf_D
            
    else:
    
        for i in range(ncell_complex):
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            buf_A[:] = DM_complex[:, k_begin:k_end]
            buf_B[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim]
        
            lib.dot(buf_A, buf_B, c=buf_C)
            lib.dot(buf_B.T.conj(), buf_C, c=buf_D)
        
            k_begin = i * nIP_prim
            k_end   = (i + 1) * nIP_prim
        
            DM_RgRg_complex[:, k_begin:k_end] = buf_D
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "DM_RgRg_complex")
    
    t3 = t4
    
    buf_fft = np.ndarray((nIP_prim, nIP_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    fn2 = getattr(libpbc, "_iFFT_Matrix_Col_InPlace", None)
    assert fn2 is not None
    
    DM_RgRg_complex2 = DM_RgRg_complex.copy()
    
    fn2(
        DM_RgRg_complex.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_prim),
        ctypes.c_int(nIP_prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "DM_RgRg_complex 2")
    t3 = t4
    
    # inplace multiplication
    
    lib.cwise_mul(mydf.W, DM_RgRg_real, out=DM_RgRg_real)
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "lib.cwise_mul 2")
    t3 = t4
    
    offset = nIP_prim * nIP_prim * ncell_complex * DM_RgRg_complex.itemsize
    
    buf_fft = np.ndarray((nIP_prim, nIP_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    fn1(
        DM_RgRg_real.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_prim),
        ctypes.c_int(nIP_prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "DM_RgRg_real")
    t3 = t4
    
    K_complex_buf = np.ndarray((nao_prim, nao_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    K_real_buf    = np.ndarray((nao_prim, nao_prim*ncell), dtype=np.float64, buffer=mydf.jk_buffer, offset=offset)
    offset += (nao_prim * nao_prim * ncell_complex) * K_complex_buf.itemsize
    offset_now = offset    
    
    buf_A = np.ndarray((nIP_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_now)
    offset_now += (nIP_prim * nIP_prim) * buf_A.itemsize
    buf_B = np.ndarray((nao_prim, nIP_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_now)
    offset_now += (nao_prim * nIP_prim) * buf_B.itemsize
    buf_C = np.ndarray((nIP_prim, nao_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_now)
    offset_now += (nIP_prim * nao_prim) * buf_C.itemsize
    buf_D = np.ndarray((nao_prim, nao_prim), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset_now)
    
    if isinstance(aoRg_FFT, list):
        for i in range(ncell_complex):
        
            k_begin = i * nIP_prim
            k_end   = (i + 1) * nIP_prim
        
            # buf_A.ravel()[:] = DM_RgRg_complex[:, k_begin:k_end].ravel()[:]
            fn_packcol2(
                buf_A.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nIP_prim),
                ctypes.c_int(2*nIP_prim),
                DM_RgRg_complex.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(DM_RgRg_complex.shape[0]),
                ctypes.c_int(2*DM_RgRg_complex.shape[1]),
                ctypes.c_int(2*k_begin),
                ctypes.c_int(2*k_end)
            )
            # buf_B.ravel()[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim].ravel()[:]
            buf_B.ravel()[:] = aoRg_FFT[i].ravel()[:]
        
            lib.dot(buf_A, buf_B.T.conj(), c=buf_C)
            lib.dot(buf_B, buf_C, c=buf_D)
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            K_complex_buf[:, k_begin:k_end] = buf_D
            
    else:
        for i in range(ncell_complex):
        
            k_begin = i * nIP_prim
            k_end   = (i + 1) * nIP_prim
        
            buf_A.ravel()[:] = DM_RgRg_complex[:, k_begin:k_end].ravel()[:]
            buf_B.ravel()[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim].ravel()[:]
        
            lib.dot(buf_A, buf_B.T.conj(), c=buf_C)
            lib.dot(buf_B, buf_C, c=buf_D)
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            K_complex_buf[:, k_begin:k_end] = buf_D
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "K_complex_buf")
    t3 = t4
    
    buf_fft = np.ndarray((nao_prim, nao_prim*ncell_complex), dtype=np.complex128, buffer=mydf.jk_buffer, offset=offset)
    
    fn2(
        K_complex_buf.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao_prim),
        ctypes.c_int(nao_prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "K_real_buf")
    t3 = t4
    
    K_real_buf *= (ngrid / vol)
    
    K = -_pack_JK(K_real_buf, Ls, nao_prim, output=None) # "-" due to robust fitting
    
    # return -K
    
    ########### do the same thing on V ###########
    
    DM_RgR_complex = np.ndarray((nIP_prim,nGridPrim*ncell_complex), dtype=np.complex128)
    DM_RgR_real = np.ndarray((nIP_prim,nGridPrim*ncell), dtype=np.float64, buffer=DM_RgR_complex)
    
    aoR_FFT = mydf.aoR_FFT
    
    buf_A = np.ndarray((nao_prim, nao_prim), dtype=np.complex128)
    buf_B = np.ndarray((nao_prim, nGridPrim), dtype=np.complex128)
    buf_B2 = np.ndarray((nao_prim, nIP_prim), dtype=np.complex128)
    buf_C = np.ndarray((nao_prim, nGridPrim), dtype=np.complex128)
    buf_D = np.ndarray((nIP_prim, nGridPrim), dtype=np.complex128)
    
    if isinstance(aoRg_FFT, list):
        assert isinstance(aoR_FFT, list)
        
        for i in range(ncell_complex):
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            # buf_A[:] = DM_complex[:, k_begin:k_end]
            fn_packcol2(
                buf_A.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_prim),
                ctypes.c_int(2*nao_prim),
                DM_complex.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(DM_complex.shape[0]),
                ctypes.c_int(2*DM_complex.shape[1]),
                ctypes.c_int(2*k_begin),
                ctypes.c_int(2*k_end)
            )
            # buf_B[:] = aoR_FFT[:, i*nGridPrim:(i+1)*nGridPrim]
            # buf_B2[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim]
            buf_B.ravel()[:] = aoR_FFT[i].ravel()[:]
            buf_B2.ravel()[:] = aoRg_FFT[i].ravel()[:]
        
            lib.dot(buf_A, buf_B, c=buf_C)
            lib.dot(buf_B2.T.conj(), buf_C, c=buf_D)
        
            k_begin = i * nGridPrim
            k_end   = (i + 1) * nGridPrim
        
            DM_RgR_complex[:, k_begin:k_end] = buf_D
    else:
        for i in range(ncell_complex):
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            buf_A[:] = DM_complex[:, k_begin:k_end]
            buf_B[:] = aoR_FFT[:, i*nGridPrim:(i+1)*nGridPrim]
            buf_B2[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim]
        
            lib.dot(buf_A, buf_B, c=buf_C)
            lib.dot(buf_B2.T.conj(), buf_C, c=buf_D)
        
            k_begin = i * nGridPrim
            k_end   = (i + 1) * nGridPrim
        
            DM_RgR_complex[:, k_begin:k_end] = buf_D
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "DM_RgR_complex")
    t3 = t4
    
    # DM_RgRg1 = DM_RgR_complex[:, mydf.IP_ID]
    # DM_RgRg2 = DM_RgRg_complex2[:, :nIP_prim]
    # diff = np.linalg.norm(DM_RgRg1 - DM_RgRg2)
    # print("diff = ", diff)
    # for i in range(10):
    #     for j in range(10):
    #         print(DM_RgRg1[i,j], DM_RgRg2[i,j])
    # assert np.allclose(DM_RgRg1, DM_RgRg2)
    
    buf_A = None
    buf_B = None
    buf_B2 = None
    buf_C = None
    buf_D = None
    
    buf_fft = np.ndarray((nIP_prim, nGridPrim*ncell_complex), dtype=np.complex128)
    
    fn2(
        DM_RgR_complex.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_prim),
        ctypes.c_int(nGridPrim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
        
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "DM_RgR_real")
    t3 = t4
        
    # inplace multiplication
    
    # print("DM_RgR_complex = ", DM_RgR_complex[:5,:5])
    # print("mydf.V_R       = ", mydf.V_R[:5,:5])
    
    lib.cwise_mul(mydf.V_R, DM_RgR_real, out=DM_RgR_real)
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "cwise_mul")
    t3 = t4
    
    # buf_fft = np.ndarray((nIP_prim, nGridPrim*ncell_complex), dtype=np.complex128)
    
    fn1(
        DM_RgR_real.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nIP_prim),
        ctypes.c_int(nGridPrim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "DM_RgR_complex 2")
    t3 = t4
    
    # print("DM_RgR_complex = ", DM_RgR_complex[:5,:5])
    
    buf_fft = None
    
    K_complex_buf = np.ndarray((nao_prim, nao_prim*ncell_complex), dtype=np.complex128)
    K_real_buf    = np.ndarray((nao_prim, nao_prim*ncell), dtype=np.float64, buffer=K_complex_buf)
    
    buf_A = np.ndarray((nIP_prim, nGridPrim), dtype=np.complex128)
    buf_B = np.ndarray((nao_prim, nGridPrim), dtype=np.complex128)
    buf_B2 = np.ndarray((nao_prim, nIP_prim), dtype=np.complex128)
    buf_C = np.ndarray((nIP_prim, nao_prim), dtype=np.complex128)
    buf_D = np.ndarray((nao_prim, nao_prim), dtype=np.complex128)
    
    if isinstance(aoRg_FFT, list):
        
        for i in range(ncell_complex):
        
            k_begin = i * nGridPrim
            k_end   = (i + 1) * nGridPrim
        
            # buf_A.ravel()[:] = DM_RgR_complex[:, k_begin:k_end].ravel()[:]
            fn_packcol2(
                buf_A.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nIP_prim),
                ctypes.c_int(2*nGridPrim),
                DM_RgR_complex.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(DM_RgR_complex.shape[0]),
                ctypes.c_int(2*DM_RgR_complex.shape[1]),
                ctypes.c_int(2*k_begin),
                ctypes.c_int(2*k_end)
            )
            # print("buf_A = ", buf_A[:5,:5])
            # buf_B.ravel()[:] = aoR_FFT[:, i*nGridPrim:(i+1)*nGridPrim].ravel()[:]
            # print("buf_B = ", buf_B[:5,:5])
            # buf_B2.ravel()[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim].ravel()[:]  
            # print("buf_B2 = ", buf_B2[:5,:5]) 
            
            buf_B.ravel()[:] = aoR_FFT[i].ravel()[:]
            buf_B2.ravel()[:] = aoRg_FFT[i].ravel()[:]
        
            lib.dot(buf_A, buf_B.T.conj(), c=buf_C)
            lib.dot(buf_B2, buf_C, c=buf_D)
        
            # print("buf_D = ", buf_D[:5,:5])
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            K_complex_buf[:, k_begin:k_end] = buf_D
    else:
        for i in range(ncell_complex):
        
            k_begin = i * nGridPrim
            k_end   = (i + 1) * nGridPrim
        
            buf_A.ravel()[:] = DM_RgR_complex[:, k_begin:k_end].ravel()[:]
            # print("buf_A = ", buf_A[:5,:5])
            buf_B.ravel()[:] = aoR_FFT[:, i*nGridPrim:(i+1)*nGridPrim].ravel()[:]
            # print("buf_B = ", buf_B[:5,:5])
            buf_B2.ravel()[:] = aoRg_FFT[:, i*nIP_prim:(i+1)*nIP_prim].ravel()[:]  
            # print("buf_B2 = ", buf_B2[:5,:5]) 
        
            lib.dot(buf_A, buf_B.T.conj(), c=buf_C)
            lib.dot(buf_B2, buf_C, c=buf_D)
        
            # print("buf_D = ", buf_D[:5,:5])
        
            k_begin = i * nao_prim
            k_end   = (i + 1) * nao_prim
        
            K_complex_buf[:, k_begin:k_end] = buf_D
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "K_complex_buf 1")
    t3 = t4
    
    buf_A = None
    buf_B = None
    buf_B2 = None
    buf_C = None
    buf_D = None
    
    buf_fft = np.ndarray((nao_prim, nao_prim*ncell_complex), dtype=np.complex128)
    
   #  print("K_complex_buf = ", K_complex_buf[:5,:5])
    
    fn2(
        K_complex_buf.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao_prim),
        ctypes.c_int(nao_prim),
        Ls.ctypes.data_as(ctypes.c_void_p),
        buf_fft.ctypes.data_as(ctypes.c_void_p)
    )
    
    t4 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t3, t4, "K_complex_buf 2")
    t3 = t4
    
    buf_fft = None
    
    K_real_buf *= (ngrid / vol)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    _benchmark_time(t1, t2, "_contract_k_dm")
    
    t1 = t2
    
    K2 = _pack_JK(K_real_buf, Ls, nao_prim, output=None)
    
    t2 = (logger.process_clock(), logger.perf_counter())
    
    _benchmark_time(t1, t2, "_pack_JK")
    
    # print("K2 = ", K2[:5,:5])
    # print("K = ", -K[:5,:5])
    
    K += K2 + K2.T
    
    # print("K = ", K[:5,:5])
    
    DM_RgR_complex = None
    DM_RgR_real = None
    
    return K
    
    # return DM_RgRg_real # temporary return for debug


# @profile 
def _symmetrize_dm(dm, Ls):
    '''
    
    generate translation symmetrized density matrix (by average)
    
    Args :
        dm : np.ndarray, density matrix, shape = (nao, nao)
        Ls : list, supercell dimension, shape = (3,), or kmesh in k-sampling

    Returns :
        dm_symm : np.ndarray, symmetrized density matrix, shape = (nao, nao)
    '''
    
        
    ncell = np.prod(Ls)
    nao = dm.shape[0]
    nao_prim = nao // ncell
    dm_symm = np.zeros((nao,nao), dtype=dm.dtype)
        
    for i in range(Ls[0]):
        for j in range(Ls[1]):
            for k in range(Ls[2]):
                
                dm_symmized_buf = np.zeros((nao_prim,nao_prim), dtype=dm.dtype)
                
                for i_row in range(Ls[0]):
                    for j_row in range(Ls[1]):
                        for k_row in range(Ls[2]):
                            
                            loc_row = i_row * Ls[1] * Ls[2] + j_row * Ls[2] + k_row
                            loc_col = ((i + i_row) % Ls[0]) * Ls[1] * Ls[2] + ((j + j_row) % Ls[1]) * Ls[2] + (k + k_row) % Ls[2]
                            
                            b_begin = loc_row * nao_prim
                            b_end   = (loc_row + 1) * nao_prim
                            
                            k_begin = loc_col * nao_prim
                            k_end   = (loc_col + 1) * nao_prim
                            
                            dm_symmized_buf += dm[b_begin:b_end, k_begin:k_end]
        
                dm_symmized_buf /= ncell
                
                for i_row in range(Ls[0]):
                    for j_row in range(Ls[1]):
                        for k_row in range(Ls[2]):
                            
                            loc_row = i_row * Ls[1] * Ls[2] + j_row * Ls[2] + k_row
                            loc_col = ((i + i_row) % Ls[0]) * Ls[1] * Ls[2] + ((j + j_row) % Ls[1]) * Ls[2] + (k + k_row) % Ls[2]
                            
                            b_begin = loc_row * nao_prim
                            b_end   = (loc_row + 1) * nao_prim
                            
                            k_begin = loc_col * nao_prim
                            k_end   = (loc_col + 1) * nao_prim
                            
                            dm_symm[b_begin:b_end, k_begin:k_end] = dm_symmized_buf        
        
    return dm_symm

# @profile 
def get_jk_dm_kSym(mydf, dm, hermi=1, kpt=np.zeros(3),
                   kpts_band=None, with_j=True, with_k=True, omega=None, **kwargs):
    '''JK for given k-point'''

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    assert with_j is True and with_k is True

    #### explore the linearity of J K with respect to dm ####

    #### perform the calculation ####

    if mydf.jk_buffer is None:  # allocate the buffer for get jk
        # mydf._allocate_jk_buffer(mydf, datatype=dm.dtype)
        mydf._allocate_jk_buffer(datatype=dm.dtype)

    if "exxdiv" in kwargs:
        exxdiv = kwargs["exxdiv"]
    else:
        exxdiv = None

    vj = vk = None

    if kpts_band is not None and abs(kpt-kpts_band).sum() > 1e-9:
        raise NotImplementedError("ISDF does not support kpts_band != kpt")

    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

    j_real = gamma_point(kpt)
    k_real = gamma_point(kpt) and not np.iscomplexobj(dm)

    assert j_real
    assert k_real

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, (mydf.max_memory - mem_now))

    log.debug1('max_memory = %d MB (%d in use)', max_memory, mem_now)

    dm = _symmetrize_dm(dm, mydf.Ls)

    if mydf.outcore and mydf.with_robust_fitting == True:
        raise NotImplementedError("outcore robust fitting has bugs and is extremely slow.")
        # vj, vk = _get_jk_dm_outcore(mydf, dm)
    else:
        if mydf.with_robust_fitting == True:
            # vj = _get_j_dm_wo_robust_fitting(mydf, dm)
            # vk = _get_k_dm_wo_robust_fitting(mydf, dm)
            # raise NotImplementedError
            vj = _get_j_kSym_robust_fitting(mydf, dm)
            vk = _get_k_kSym_robust_fitting(mydf, dm)
        else:
            # print(dm[:10, :10])
            vj1 = _get_j_kSym(mydf, dm)
            vk2 = _get_k_kSym(mydf, dm)
            # vj = _get_j_with_Wgrid(mydf, dm=dm)
            # vk = _get_k_with_Wgrid(mydf, dm=dm)
            
            vj = vj1
            vk = vk2

    t1 = log.timer('sr jk', *t1)

    return vj, vk

class PBC_ISDF_Info_kSym(ISDF_outcore.PBC_ISDF_Info_outcore):
    
    # @profile 
    def __init__(self, mol:Cell, max_buf_memory:int, Ls=[1,1,1], outcore=True, with_robust_fitting=True, aoR=None):
        
        print("Ls = ", Ls)
        
        super().__init__(mol=mol, max_buf_memory=max_buf_memory, outcore=outcore, with_robust_fitting=with_robust_fitting, aoR=aoR, Ls=Ls)
        
        # assert with_robust_fitting == False
        assert self.mesh[0] % Ls[0] == 0
        assert self.mesh[1] % Ls[1] == 0
        assert self.mesh[2] % Ls[2] == 0
        
        self.Ls = Ls
        
        if self.coords is None:
            from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2
            df_tmp = MultiGridFFTDF2(self.cell)
            self.coords = np.asarray(df_tmp.grids.coords).reshape(-1,3).copy()

        print("self.cell.lattice_vectors = ", self.cell.lattice_vectors())
        self.ordered_grid_coords, self.ordered_grid_coords_dict = _extract_grid_primitive_cell(self.cell.lattice_vectors(), self.mesh, self.Ls, self.coords)

        #### symmetrize the partition ####

        self.partition = np.array(self.partition, dtype=np.int32)
        print("self.partition.shape = ", self.partition.shape)
        assert self.partition.shape[0] == np.prod(self.coords.shape[0]) // np.prod(self.Ls)

        self.meshPrim = np.array(self.mesh) // np.array(self.Ls)
        print("meshprim = ", self.meshPrim)
        self.natm     = self.cell.natm
        self.natmPrim = self.cell.natm // np.prod(self.Ls)

        # partition = self.partition.reshape(self.Ls[0], self.meshPrim[0], self.Ls[1], self.meshPrim[1], self.Ls[2], self.meshPrim[2])
        # partition = partition.transpose(0,2,4,1,3,5).reshape(-1, np.prod(self.meshPrim))
        # partition = self.partition.reshape(self.meshPrim[0], self.meshPrim[1], self.meshPrim[2])
        partition = np.zeros((np.prod(Ls), np.prod(self.meshPrim)), dtype=np.int32)
        nmeshPrim = np.prod(self.meshPrim)

        # for ix in range(self.Ls[0]):
        #     for iy in range(self.Ls[1]):
        #         for iz in range(self.Ls[2]):
        #             loc = ix * self.Ls[1] * self.Ls[2] + iy * self.Ls[2] + iz
        #             shift = loc * self.natmPrim
        
        partition[0] = self.partition
        for i in range(nmeshPrim):
            atm_id = partition[0][i]
            atm_id = (atm_id % self.natmPrim)
            partition[0][i] = atm_id
        for i in range(1, np.prod(Ls)):
            partition[i] = self.natmPrim * i + partition[0]
        
        partition = partition.reshape(self.Ls[0], self.Ls[1], self.Ls[2], self.meshPrim[0], self.meshPrim[1], self.meshPrim[2])
        partition = partition.transpose(0,3,1,4,2,5).reshape(-1)
        self.partition = partition

        # self.aoR_Prim = self._numint.eval_ao(self.cell, self.ordered_grid_coords_dict[(0,0,0)])[0].T

        if outcore is False:
            weight   = np.sqrt(self.cell.vol / self.ngrids)
            ngrim_prim = self.ngrids // np.prod(self.Ls)        
            self.aoR = self._numint.eval_ao(self.cell, self.ordered_grid_coords[:ngrim_prim])[0].T * weight # the T is important, TODO: reduce it!
        else:
            self.aoR = None
    
    ################ test function ################ 
    
    def check_data(self, criterion=1e12):
        print("check data")
        not_correct = False
        if hasattr(self, "W"):
            for i in range(self.W.shape[0]):
                for j in range(self.W.shape[1]):
                    if abs(self.W[i,j]) > criterion:
                        # print("W[{},{}] = {}".format(i,j,self.W[i,j]))
                        not_correct = True
        if hasattr(self, "W0"):
            for i in range(self.W0.shape[0]):
                for j in range(self.W0.shape[1]):
                    if abs(self.W0[i,j]) > criterion:
                        # print("W0[{},{}] = {}".format(i,j,self.W0[i,j]))
                        not_correct = True
        if hasattr(self, "aoRg"):
            for i in range(self.aoRg.shape[0]):
                for j in range(self.aoRg.shape[1]):
                    if abs(self.aoRg[i,j]) > criterion:
                        # print("aoRg[{},{}] = {}".format(i,j,self.aoRg[i,j]))
                        not_correct = True
        if hasattr(self, "aoRg_FFT"):
            for i in range(self.aoRg_FFT.shape[0]):
                for j in range(self.aoRg_FFT.shape[1]):
                    if abs(self.aoRg_FFT[i,j]) > criterion:
                        # print("aoRg_FFT[{},{}] = {}".format(i,j,self.aoRg_FFT[i,j]))
                        not_correct = True
        if not_correct:
            raise ValueError("data is not correct")
        print("end check data")
        
    # @profile 
    def construct_auxbasis_benchmark(self):
        _construct_aux_basis_benchmark(self)
    
    ################ allocate buffer ################ 
    
    def _get_bufsize_get_j(self):
        
        # if self.with_robust_fitting == False:
        if True:
            
            naux       = self.naux
            nao        = self.nao
            nIP_Prim   = self.nIP_Prim
            nao_prim   = self.nao // np.prod(self.Ls)
            
            size_buf3  = nao * naux + naux + naux + nao * nao
            size_buf4  = nao * nIP_Prim
            size_buf4 += nIP_Prim
            size_buf4 += nao_prim * nao
            size_buf4 += nIP_Prim
            size_buf4 += nao_prim * nao_prim
            size_buf4 += nao_prim * nIP_Prim * 3
            
            return max(size_buf3, size_buf4)
            
        # else:
        #     raise NotImplementedError

    def _get_bufsize_get_k(self):
        
        # if self.with_robust_fitting == False:
        if True:
            
            naux     = self.naux
            nao      = self.nao
            nIP_Prim = self.nIP_Prim
            nao_prim = self.nao // np.prod(self.Ls)
            ncell_complex = self.Ls[0] * self.Ls[1] * (self.Ls[2]//2+1)
            
            size_buf5  = nIP_Prim * nIP_Prim * ncell_complex * 2
            size_buf5 += nao_prim * nao_prim * 2
            size_buf5 += nIP_Prim * nIP_Prim * ncell_complex * 2
            
            size_buf6  = nIP_Prim * nIP_Prim * ncell_complex * 2
            size_buf6 += nIP_Prim * nIP_Prim * ncell_complex * 2
            size_buf6 += nao_prim * nao_prim * ncell_complex * 2
            size_buf6 += nIP_Prim * nIP_Prim  * 2
            size_buf6 += nao_prim * nIP_Prim  * 2 * 2
            size_buf6 += nao_prim * nao_prim  * 2
        
            return max(size_buf5, size_buf6)
        
        # else:
            
        #     raise NotImplementedError
    
    # @profile 
    def _allocate_jk_buffer(self, dtype=np.float64):
        
        if self.jk_buffer is not None:
            return
            
        num_threads = lib.num_threads()
        
        nIP_Prim = self.nIP_Prim
        nGridPrim = self.nGridPrim
        ncell_complex = self.Ls[0] * self.Ls[1] * (self.Ls[2]//2+1)
        nao_prim  = self.nao // np.prod(self.Ls)
        naux       = self.naux
        nao        = self.nao
        ngrids = nGridPrim * self.Ls[0] * self.Ls[1] * self.Ls[2]
        ncell  = np.prod(self.Ls)
        
        if self.outcore is False:
            
            ### in build aux basis ###
            
            size_buf1 = nIP_Prim * ncell_complex*nIP_Prim * 2
            size_buf1+= nIP_Prim * ncell_complex*nGridPrim * 2 * 2
            size_buf1+= num_threads * nGridPrim * 2
            size_buf1+= nIP_Prim * nIP_Prim * 2
            size_buf1+= nIP_Prim * nGridPrim * 2 * 2
            
            ### in construct W ###
            
            # print("nIP_Prim = ", nIP_Prim)
            # print("ncell_complex = ", ncell_complex)    
            
            size_buf2  = nIP_Prim * nIP_Prim * 2
            size_buf2 += nIP_Prim * nGridPrim * 2 * 2
            size_buf2 += nIP_Prim * nIP_Prim *  ncell_complex * 2 * 2
            
            # print("size_buf2 = ", size_buf2)
            
            ### in get_j ###
                    
            buf_J = self._get_bufsize_get_j()
            
            ### in get_k ### 
        
            buf_K = self._get_bufsize_get_k()
            
            ### ddot_buf ###
            
            # size_ddot_buf = max(naux*naux+2,ngrids)*num_threads
            size_ddot_buf = (nIP_Prim*nIP_Prim+2)*num_threads
            
            # print("size_buf1 = ", size_buf1)
            # print("size_buf2 = ", size_buf2)
            # print("size_buf3 = ", size_buf3)
            # print("size_buf4 = ", size_buf4)
            # print("size_buf5 = ", size_buf5)
            
            size_buf = max(size_buf1,size_buf2,buf_J,buf_K)
            
            # print("size_buf = ", size_buf)
            
            if hasattr(self, "IO_buf"):
                if self.IO_buf.size < (size_buf+size_ddot_buf):
                    self.IO_buf = np.zeros((size_buf+size_ddot_buf), dtype=np.float64)
                self.jk_buffer = np.ndarray((size_buf), dtype=np.float64, buffer=self.IO_buf, offset=0)
                self.ddot_buf  = np.ndarray((size_ddot_buf), dtype=np.float64, buffer=self.IO_buf, offset=size_buf)

            else:

                self.jk_buffer = np.ndarray((size_buf), dtype=np.float64)
                self.ddot_buf  = np.zeros((size_ddot_buf), dtype=np.float64)
            
        else:
            # raise NotImplementedError
    
            buf_J = self._get_bufsize_get_j()
            buf_K = self._get_bufsize_get_k()
    
            ### in build aux basis ###
            
            size_buf1 = nIP_Prim * ncell_complex * nIP_Prim * 2 * 2 # store A 
            
            bunchsize = nGridPrim // ncell_complex + 1
            size_buf2 = nIP_Prim * ncell_complex * nGridPrim * 2 # store B
            size_buf2+= nao * bunchsize * 5 # AoR Buf
            
            B_bunchsize = min(nIP_Prim, bunchsize) # more acceptable for memory
            if B_bunchsize < BUNCHSIZE_IN_AUX_BASIS_OUTCORE:
                B_bunchsize = BUNCHSIZE_IN_AUX_BASIS_OUTCORE
            sub_bunchsize = B_bunchsize // ncell
            sub_bunchsize = min(sub_bunchsize, bunchsize)
            
            size_buf2 += sub_bunchsize * ncell_complex * nIP_Prim * 2 * 4 # store B
            size_buf2 += nIP_Prim * sub_bunchsize * 2 

            size_buf2 += nIP_Prim * nIP_Prim * 2 # buf_A_diag
            
            # the solve equation 
            
            size_buf3 =  nIP_Prim * ncell_complex * nGridPrim * 2 
            B_bunchsize = min(nIP_Prim, nGridPrim) # more acceptable for memory
            if B_bunchsize < BUNCHSIZE_IN_AUX_BASIS_OUTCORE:
                B_bunchsize = BUNCHSIZE_IN_AUX_BASIS_OUTCORE
            B_bunchsize = min(B_bunchsize, nGridPrim)
            
            size_buf3 += nIP_Prim * B_bunchsize * 2 * 3 # store B
            size_buf3 += nIP_Prim * nIP_Prim * 2 # buf_A_diag
            size_buf3 += nIP_Prim * nIP_Prim * 2 # buf_A2_diag
            
            size_buf = max(size_buf1, size_buf2, size_buf3, buf_J, buf_K)
            
            size1 = sub_bunchsize * nao
            size_ddot_buf = (max(size1,nIP_Prim*nIP_Prim)+2)*num_threads
            
            if hasattr(self, "IO_buf"):
                if self.IO_buf.size < size_buf:
                    print("reallocate of size = ", size_buf)
                    self.IO_buf = np.zeros((size_buf), dtype=np.float64)
            else:
                self.IO_buf = np.zeros((size_buf), dtype=np.float64)

            self.jk_buffer = np.ndarray((size_buf), dtype=np.float64, buffer=self.IO_buf, offset=0)
            self.ddot_buf  = np.ndarray((size_ddot_buf), dtype=np.float64)
    
    ################ select IP ################
    
    # @profile 
    def select_IP(self, c:int, m:int, possible_grid_ID=None, global_selection=False):
        first_natm = self.cell.natm // np.prod(self.Ls)
        
        print("first_natm = ", first_natm)
        print("c = ", c)
        print("m = ", m)
        
        # get primID
        
        mesh = self.cell.mesh
        mesh_prim = np.array(mesh) // np.array(self.Ls)
        ngrid_prim = np.prod(mesh_prim)
        grid_primitive = self.ordered_grid_coords_dict[(0,0,0)]
        # self.IP_coords = grid_primitive[possible_grid_ID]
        weight         = np.sqrt(self.cell.vol / self.ngrids)
        self.nGridPrim = grid_primitive.shape[0]
        
        
        if possible_grid_ID is None:
            if global_selection:
                IP_GlobalID = _select_IP_ksym_global_direct(self, c, m, first_natm)
            else:
                IP_GlobalID = ISDF._select_IP_direct(self, c, m, first_natm, True) # we do not have to perform selection IP over the whole supercell ! 
            print("len of IP_GlobalID = ", len(IP_GlobalID))
    
            possible_grid_ID = []

            if global_selection == False:
                for grid_id in IP_GlobalID:
                    pnt_id = (grid_id // (mesh[1] * mesh[2]), (grid_id // mesh[2]) % mesh[1], grid_id % mesh[2])
                    box_id = (pnt_id[0] // mesh_prim[0], pnt_id[1] // mesh_prim[1], pnt_id[2] // mesh_prim[2])
                    pnt_prim_id = (pnt_id[0] % mesh_prim[0], pnt_id[1] % mesh_prim[1], pnt_id[2] % mesh_prim[2])
                    pnt_prim_ravel_id = pnt_prim_id[0] * mesh_prim[1] * mesh_prim[2] + pnt_prim_id[1] * mesh_prim[2] + pnt_prim_id[2]
                    # print("grid_id = %d, pnt_id = %s, box_id = %s, pnt_prim_id = %s" % (grid_id, pnt_id, box_id, pnt_prim_id))
                    possible_grid_ID.append(pnt_prim_ravel_id)
            else:
                possible_grid_ID = IP_GlobalID.copy()

            possible_grid_ID = list(set(possible_grid_ID))
            possible_grid_ID.sort()
            
        
        print("nIP = ", len(possible_grid_ID))
        print("possible_grid_ID = ", possible_grid_ID)
        
        ordered_IP_coords = []
        
        # print("self.ordered_grid_coords_dict = ", self.ordered_grid_coords_dict.keys())
        
        for ix in range(self.Ls[0]):
            for iy in range(self.Ls[1]):
                for iz in range(self.Ls[2]):
                    ordered_IP_coords.append(self.ordered_grid_coords_dict[(ix,iy,iz)][possible_grid_ID]) # enforce translation symmetry
        
        self.ordered_IP_coords = np.array(ordered_IP_coords).reshape(-1,3).copy()
    
        self.aoRg      = self._numint.eval_ao(self.cell, self.ordered_IP_coords)[0].T * weight
        self.aoRg_Prim = self.aoRg[:, :len(possible_grid_ID)].copy()
        self.nIP_Prim  = len(possible_grid_ID)
        
        nao_prim = self.nao // np.prod(self.Ls)
        Ls       = np.array(self.Ls, dtype=np.int32)    
        ncell_complex = self.Ls[0] * self.Ls[1] * (self.Ls[2]//2+1)
        
        # self.aoRg_FFT  = self.aoRg[:nao_prim,:].copy()
        self.aoRg_FFT  = np.zeros((nao_prim, ncell_complex*self.nIP_Prim), dtype=np.complex128)
        self.aoRg_FFT_real = np.ndarray((nao_prim, np.prod(Ls)*self.nIP_Prim), dtype=np.double, buffer=self.aoRg_FFT, offset=0)
        self.aoRg_FFT_real.ravel()[:] = self.aoRg[:nao_prim,:].ravel()
        
        nthread        = lib.num_threads()
        buffer         = np.ndarray((nao_prim, ncell_complex*self.nIP_Prim), dtype=np.complex128, buffer=self.jk_buffer, offset=0)
        
        fn = getattr(libpbc, "_FFT_Matrix_Col_InPlace", None)
        assert fn is not None
        
        print("self.aoRg_FFT.shape = ", self.aoRg_FFT.shape)
        
        fn(
            self.aoRg_FFT_real.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_prim),
            ctypes.c_int(self.nIP_Prim),
            Ls.ctypes.data_as(ctypes.c_void_p),
            buffer.ctypes.data_as(ctypes.c_void_p)
        ) # no normalization factor ! 
                
        if self.with_robust_fitting:
            
            cell = self.cell
            
            shl_loc_begin = None
            shl_loc_end = None
    
            for i in range(cell.nbas):
                if cell.bas_atom(i) < first_natm:
                    if shl_loc_begin is None:
                        shl_loc_begin = i
                    shl_loc_end = i+1
            
            print("shl_loc_begin = ", shl_loc_begin)
            print("shl_loc_end = ", shl_loc_end)
    
            # aoR_FFT_eval = ISDF_eval_gto(cell=self.cell, coords=self.ordered_grid_coords, shls_slice=(shl_loc_begin, shl_loc_end)) * weight
            aoR_FFT_eval = self._numint.eval_ao(self.cell, self.ordered_grid_coords, shls_slice=(shl_loc_begin, shl_loc_end))[0].T * weight
            print("aoR_FFT_eval.shape = ", aoR_FFT_eval.shape)
            self.aoR_FFT = np.ndarray((nao_prim, ncell_complex*self.nGridPrim), dtype=np.complex128)
            self.aoR_FFT_real = np.ndarray((nao_prim, np.prod(Ls)*self.nGridPrim), dtype=np.double, buffer=self.aoR_FFT, offset=0)
            self.aoR_FFT_real.ravel()[:] = aoR_FFT_eval.ravel()[:]
            
            buffer = np.ndarray((nao_prim, ncell_complex*self.nGridPrim), dtype=np.complex128)
            fn(self.aoR_FFT_real.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_prim),
                ctypes.c_int(self.nGridPrim),
                Ls.ctypes.data_as(ctypes.c_void_p),
                buffer.ctypes.data_as(ctypes.c_void_p)
            )
            buffer = None
            aoR_FFT_eval = None
                
        return np.array(possible_grid_ID, dtype=np.int32)
        
    ################ construct W ################

    ################ driver for build ################
    
    # @profile 
    def build_IP_auxbasis(self, IO_File:str = None, c:int = 5, m:int = 5, IP_ID=None):
        
        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        # if IP_ID is not None:
        #     self.IP_ID = IP_ID
        # else:
        self.IP_ID = self.select_IP(c, m, possible_grid_ID=IP_ID)  # prim_gridID
        self.IP_ID = np.asarray(self.IP_ID, dtype=np.int32)
        print("IP_ID = ", self.IP_ID)
        print("len(IP_ID) = ", len(self.IP_ID))
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, "select IP")

        if IO_File is None:
            # generate a random file name start with tmp_
            import random
            import string
            IO_File = "tmp_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=8)) + ".hdf5"

        print("IO_File = ", IO_File)

        # construct aoR

        if self.coords is None:
            from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2
            df_tmp = MultiGridFFTDF2(self.cell)
            self.coords = np.asarray(df_tmp.grids.coords).reshape(-1,3).copy()

        # t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        # coords_IP = self.coords[self.IP_ID]
        # weight    = np.sqrt(self.cell.vol / self.ngrids)
        # self.aoRg = self._numint.eval_ao(self.cell, coords_IP)[0].T * weight  # the T is important
        # t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        # _benchmark_time(t1, t2, "construct aoR") # built in select_IP

        self.naux = self.aoRg.shape[1]
        print("naux = ", self.naux)
        self.c    = c
        
        # print("naux = ", self.naux)
        # self.chunk_size, self.nRow_IO_V, self.blksize_aux, self.bunchsize_readV, self.grid_bunchsize, self.blksize_W, self.use_large_chunk_W  = _determine_bunchsize(
        #     self.nao, self.naux, self.mesh, self.IO_buf.size, self.saveAoR)

        # construct aux basis

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if self.outcore:
            # print("construct aux basis in outcore mode")
            # _construct_aux_basis_IO(self, IO_File, self.IO_buf)
            self._allocate_jk_buffer()
            print("after allocate buffer")
            _construct_aux_basis_kSym_outcore(self, IO_File, self.IO_buf)
            # raise NotImplementedError   
        else:
            print("construct aux basis in incore mode")
            _construct_aux_basis_kSym(self)
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, "construct aux basis")

        self.IO_FILE = IO_File
    
    # @profile 
    def build_auxiliary_Coulomb(self):

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

        if self.outcore:
            # raise NotImplementedError

            print("construct W in outcore mode")

            if self.with_robust_fitting:
                raise NotImplementedError
            else:
                _aux_basis_FFT_outcore(self, self.IO_FILE, self.IO_buf)
                _construct_W_outcore(self, self.IO_FILE, self.IO_buf)
        
        else:
            
            print("construct W in incore mode")
            
            if self.with_robust_fitting:
                _construct_V_W_incore(self)
            else:
                _construct_W_incore(self)
        
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        _benchmark_time(t1, t2, "construct W")
    
    ################ get jk ################

    get_jk = get_jk_dm_kSym

C = 2
M = 5

if __name__ == "__main__":
    
    boxlen = 3.5668
    prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    
    KE_CUTOFF = 8
    
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
    
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=KE_CUTOFF)
    prim_mesh = prim_cell.mesh
    print("prim_mesh = ", prim_mesh)
    
    # Ls = [2, 2, 2]
    # Ls = [2, 1, 3]
    Ls = [1, 1, 3]
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    
    cell = build_supercell(atm, prim_a, Ls = Ls, ke_cutoff=KE_CUTOFF, mesh=mesh)
    
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

    # pbc_isdf_info = PBC_ISDF_Info(cell, aoR)
    # _, Possible_IP_coords = _get_possible_IP(pbc_isdf_info, Ls, coords)
    # print("Possible_IP_coords = ", Possible_IP_coords)
    
    ############ construct ISDF object ############
    
    pbc_isdf_info_ksym = PBC_ISDF_Info_kSym(cell, 8 * 1000 * 1000, Ls=Ls, outcore=False, with_robust_fitting=True, aoR=None)
    
    ############ test select IP ############
    
    possible_IP = pbc_isdf_info_ksym.select_IP(C, M)
    print("possible_IP = ", possible_IP)
    
    pbc_isdf_info_ksym.build_IP_auxbasis(c=C, m=M)
    pbc_isdf_info_ksym.construct_auxbasis_benchmark()
    
    basis1 = pbc_isdf_info_ksym.aux_basis
    basis2 = pbc_isdf_info_ksym.aux_basis_bench
    
    print("basis1.shape = ", basis1.shape)
    print("basis2.shape = ", basis2.shape)
    
    # print(basis1[:10,:10])
    # print(basis2[:10,:10])
    # print(basis1[:10,:10]/basis2[:10,:10])
    
    # print(basis1[-10:, -10:])
    # print(basis2[-10:, -10:])
    # print(basis1[-10:, -10:]/basis2[-10:, -10:])
    
    for ix in range(Ls[0]):
        for iy in range(Ls[1]):
            for iz in range(Ls[2]):
                icell = ix * Ls[1] * Ls[2] + iy * Ls[2] + iz
                
                k_begin = icell * pbc_isdf_info_ksym.nGridPrim
                k_end   = (icell + 1) * pbc_isdf_info_ksym.nGridPrim
                
                basis1 = pbc_isdf_info_ksym.aux_basis[:, k_begin:k_end]
                basis2 = pbc_isdf_info_ksym.aux_basis_bench[:, k_begin:k_end]
                
                print(basis1[:3,:3])
                print(basis2[:3,:3])
                print(basis1[:3,:3]/basis2[:3,:3])
                
                for i in range(basis1.shape[1]):
                    print("%3d %15.8e %15.8e %15.8e %15.8e" % (i, basis1[0,i].real, basis1[0,i].imag, basis2[0,i].real, basis2[0,i].imag))
                
                assert np.allclose(basis1, basis2) # we get the same result, correct !
                
                print("Cell %d pass" % icell)
    
    assert np.allclose(basis1, basis2) # we get the same result, correct ! 
    
    basis3 = pbc_isdf_info_ksym.aux_basis_bench_Grid
    print("basis3.shape = ", basis3.shape)
    print("basis3.dtype = ", basis3.dtype)
    
    V_R_grid, W_grid = _construct_W_benchmark_grid(cell, basis3, Ls, mesh)
    
    # print("W_grid = ", W_grid[:4,:4])
    
    print("W_grid.shape = ", W_grid.shape)
    # W_grid = W_grid.reshape(-1, mesh[0], mesh[1], mesh[2])
    # print("W_grid.shape = ", W_grid.shape)
    # W_grid = np.fft.fftn(W_grid, axes=(1, 2, 3))
    # W_grid = W_grid.reshape(-1, np.prod(mesh))
    # W_grid = W_grid.reshape(mesh[0], mesh[1], mesh[2], -1)
    # W_grid = W_grid.transpose(3, 0, 1, 2)
    # W_grid = np.fft.ifftn(W_grid, axes=(1, 2, 3))
    # W_grid = W_grid.transpose(1, 2, 3, 0)
    # W_grid = W_grid.reshape(-1, np.prod(mesh))
    
    # W_grid = _RowCol_FFT_bench(W_grid, Ls)
    
    nIP_prim = pbc_isdf_info_ksym.nIP_Prim
    nGrim_prim = pbc_isdf_info_ksym.nGridPrim

    # print(W_grid[:nIP_prim, nIP_prim:2*nIP_prim])
    # print(W_grid[nIP_prim:2*nIP_prim,:nIP_prim])
    
    
    W_fft  = _RowCol_FFT_bench(W_grid, Ls)
    V_R_fft = _RowCol_FFT_bench(V_R_grid, Ls)
    
    # print(W_fft[:nIP_prim, :nIP_prim])
    
    ## check W_fft's diagonal structure ##
    
    ncell = np.prod(Ls)
    
    W_fft_packed = np.zeros((nIP_prim, nIP_prim*ncell), dtype=np.complex128)

    
    for icell in range(ncell):
        b_begin = icell * nIP_prim
        b_end   = (icell + 1) * nIP_prim
        k_begin = icell * nIP_prim
        k_end   = (icell + 1) * nIP_prim
        
        matrix_before = W_fft[b_begin:b_end, :k_begin]
        matrix_after  = W_fft[b_begin:b_end, k_end:]
        assert np.allclose(matrix_before, 0.0)
        assert np.allclose(matrix_after, 0.0)
        
        mat = W_fft[b_begin:b_end, k_begin:k_end]
        assert np.allclose(mat, mat.T.conj())
        
        k_begin = icell * nGrim_prim
        k_end   = (icell + 1) * nGrim_prim
        matrix_before = V_R_fft[b_begin:b_end, :k_begin]
        matrix_after = V_R_fft[b_begin:b_end, k_end:]
        # assert np.allclose(matrix_before, 0.0)
        # assert np.allclose(matrix_after, 0.0)
        
        if np.allclose(matrix_before, 0.0) == False:
            print("norm of matrix_before = ", np.linalg.norm(matrix_before))
        if np.allclose(matrix_after, 0.0) == False:
            print("norm of matrix_after = ", np.linalg.norm(matrix_after))
        
        W_fft_packed[:, k_begin:k_end] = W_fft[b_begin:b_end, k_begin:k_end]
    
    pbc_isdf_info_ksym.build_auxiliary_Coulomb()
    
    W = pbc_isdf_info_ksym.W
    W_bench = W_grid[:nIP_prim, :]
    
    # print(W_fft_packed.shape)
    # print(W.shape)
    
    W1 = W[:4, :4]
    W2 = W_bench[:4, :4]
    
    print(W1)
    print(W2)
    print(W1/W2)
    
    assert np.allclose(W, W_bench)
    
    V_R = pbc_isdf_info_ksym.V_R
    V_R_bench = V_R_grid[:nIP_prim, :]
    
    V1 = V_R[:4, :4]
    V2 = V_R_bench[:4, :4]
    
    print(V1)
    print(V2)
    print(V1/V2)
    
    assert np.allclose(V_R, V_R_bench)
    
    # exit(1)
    
    from pyscf.pbc import dft as pbcdft
    mf=pbcdft.RKS(cell)
    mf.xc = "PBE,PBE"
    mf.init_guess='atom'  # atom guess is fast
    # mf.with_df = multigrid.MultiGridFFTDF2(cell)
    # mf.with_df.ngrids = 4  # number of sets of grid points ? ? ? 

    dm1 = mf.get_init_guess(cell, 'atom')
    
    J_bench = _get_j_with_Wgrid(pbc_isdf_info_ksym, W_grid, dm1)
    
    J2      = _get_j_kSym(pbc_isdf_info_ksym, dm1)
    
    print(J_bench[:4, :4])
    print(J2[:4, :4])
    
    assert np.allclose(J_bench, J2)  # we get the correct answer!
    
    ## check the symmetry of dm1 ## 
    
    dm_packed = _RowCol_FFT_bench(dm1, Ls)
    
    ncell = np.prod(Ls)
    nao   = cell.nao
    nao_prim = nao // ncell
    
    for icell in range(ncell):
        b_begin = icell * nao_prim
        b_end   = (icell + 1) * nao_prim
        k_begin = icell * nao_prim
        k_end   = (icell + 1) * nao_prim
        
        matrix_before = dm_packed[b_begin:b_end, :k_begin]
        matrix_after  = dm_packed[b_begin:b_end, k_end:]
        assert np.allclose(matrix_before, 0.0)
        assert np.allclose(matrix_after, 0.0)
        
        mat = dm_packed[b_begin:b_end, k_begin:k_end]
        assert np.allclose(mat, mat.T.conj())
    
    # DM_bencharmk = _get_DM_RgRg_benchmark(pbc_isdf_info_ksym, dm1)
    # DM_test2 = _get_k_kSym(pbc_isdf_info_ksym, dm1)
    # print(DM_bencharmk[:4, :4])
    # print(DM_test2[:4, :4])
    
    # assert np.allclose(DM_bencharmk[:nIP_prim,:], DM_test2)
    
    K_bench = _get_k_with_Wgrid(pbc_isdf_info_ksym, W_grid, dm1)
    K2      = _get_k_kSym(pbc_isdf_info_ksym, dm1)
    
    # print(K_bench[:4, :4])
    # print(K2[:4, :4])
    # print(K_bench[:4, :4]/K2[:4, :4])
    
    assert np.allclose(K_bench, K2)  # we get the correct answer!
    
    ### do the SCF ### 
    
    boxlen = 3.5668
    prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    
    KE_CUTOFF = 70
    
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
    
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=KE_CUTOFF)
    prim_mesh = prim_cell.mesh
    print("prim_mesh = ", prim_mesh)
    
    C = 15
    
    # Ls = [2, 2, 2]
    Ls = [1, 2, 2]
    Ls = np.array(Ls, dtype=np.int32)
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    cell = build_supercell(atm, prim_a, Ls = Ls, ke_cutoff=KE_CUTOFF, mesh=mesh)
    
    pbc_isdf_info = PBC_ISDF_Info_kSym(cell, 80 * 1000 * 1000, Ls=Ls, outcore=False, with_robust_fitting=True, aoR=None)
    pbc_isdf_info.build_IP_auxbasis(c=C, m=M)
    pbc_isdf_info.build_auxiliary_Coulomb()
    
    from pyscf.pbc import scf
    
    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 16
    mf.conv_tol = 1e-7

    print("mf.direct_scf = ", mf.direct_scf)

    mf.kernel()
    
    exit(1)
        
    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    # mf.with_df = pbc_isdf_info
    mf.max_cycle = 64
    mf.conv_tol = 1e-7

    print("mf.direct_scf = ", mf.direct_scf)

    mf.kernel()
    