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

import pyscf.pbc.df.isdf.isdf_k as ISDF_K

C = 2
M = 5

if __name__ == "__main__":
    
    boxlen = 3.5668
    prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    
    KE_CUTOFF = 16
    
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
    
    prim_cell = ISDF_K.build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=KE_CUTOFF)
    prim_mesh = prim_cell.mesh
    print("prim_mesh = ", prim_mesh)

    Ls = [2, 2, 2]
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    
    cell = ISDF_K.build_supercell(atm, prim_a, Ls = Ls, ke_cutoff=KE_CUTOFF, mesh=mesh)
    
    ncell = np.prod(Ls)
    ncell_complex = Ls[0] * Ls[1] * (Ls[2]//2+1)
    
    memory = 8 * 1000 * 1000 # 8 MB
    pbc_isdf_info_ksym = ISDF_K.PBC_ISDF_Info_kSym(cell, 8 * 1000 * 1000, Ls=Ls, outcore=True, with_robust_fitting=False, aoR=None) 
    pbc_isdf_info_ksym.build_IP_auxbasis(c=C, m=M)
    
    ### check ### 
    
    ngrim_prim = pbc_isdf_info_ksym.ngrids // np.prod(pbc_isdf_info_ksym.Ls) 
    coord_grim = pbc_isdf_info_ksym.ordered_grid_coords[:ngrim_prim]
    weight     = np.sqrt(pbc_isdf_info_ksym.cell.vol / pbc_isdf_info_ksym.ngrids)
    pbc_isdf_info_ksym.aoR = pbc_isdf_info_ksym._numint.eval_ao(pbc_isdf_info_ksym.cell, coord_grim)[0].T * weight
    
    pbc_isdf_info_ksym.outcore = False
    pbc_isdf_info_ksym.jk_buffer = None
    pbc_isdf_info_ksym._allocate_jk_buffer()
    ISDF_K._construct_aux_basis_kSym(pbc_isdf_info_ksym)
    
    aux_benchmark = pbc_isdf_info_ksym.aux_basis
    
    IO_File = pbc_isdf_info_ksym.IO_FILE
    f_aux_basis = h5py.File(IO_File, 'r')
    
    aux_basis = f_aux_basis[ISDF_K.AUX_BASIS_DATASET][:]
    print("aux_basis = ", aux_basis.shape)
    # print("aux_basis = ", aux_basis)
    
    aux_basis = aux_basis.reshape(ncell_complex, -1, prim_mesh[0] * prim_mesh[1] * prim_mesh[2])
    # aux_basis = aux_basis.transpose(3, 0, 1, 2, 4, 5, 6).reshape(-1, prim_mesh[0], prim_mesh[1], prim_mesh[2])
    
    # get the frequency #
    
    freq1 = np.array(range(prim_mesh[0]), dtype=np.float64)
    freq2 = np.array(range(prim_mesh[1]), dtype=np.float64)
    freq3 = np.array(range(prim_mesh[2]), dtype=np.float64)
    freq_q = np.array(np.meshgrid(freq1, freq2, freq3, indexing='ij'))
    
    freq1 = np.array(range(Ls[0]), dtype=np.float64)
    freq2 = np.array(range(Ls[1]), dtype=np.float64)
    freq3 = np.array(range(Ls[2]//2+1), dtype=np.float64)
    freq_Q = np.array(np.meshgrid(freq1, freq2, freq3, indexing='ij'))
    
    FREQ = np.einsum("ijkl,ipqs->ijklpqs", freq_Q, freq_q)
    FREQ[0] /= (Ls[0] * prim_mesh[0])
    FREQ[1] /= (Ls[1] * prim_mesh[1])
    FREQ[2] /= (Ls[2] * prim_mesh[2])
    FREQ = np.einsum("ijklpqs->jklpqs", FREQ)
    FREQ  = FREQ.reshape(-1, np.prod(prim_mesh)).copy()
    FREQ  = np.exp(-2.0j * np.pi * FREQ)  # this is the only correct way to construct the factor
    
    fn_final_fft = getattr(libpbc, "_FinalFFT", None)
    assert fn_final_fft is not None
    fn_permutation_conj = getattr(libpbc, "_PermutationConj", None)
    assert fn_permutation_conj is not None
    
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
    
    permutation = np.zeros((8, ngrim_prim), dtype=np.int32)
    print("permutation.shape = ", permutation.shape)
    permutation[0] = _permutation(prim_mesh[0], prim_mesh[1], prim_mesh[2], 0, 0, 0)
    permutation[1] = _permutation(prim_mesh[0], prim_mesh[1], prim_mesh[2], 0, 0, 1)
    permutation[2] = _permutation(prim_mesh[0], prim_mesh[1], prim_mesh[2], 0, 1, 0)
    permutation[3] = _permutation(prim_mesh[0], prim_mesh[1], prim_mesh[2], 0, 1, 1)
    permutation[4] = _permutation(prim_mesh[0], prim_mesh[1], prim_mesh[2], 1, 0, 0)
    permutation[5] = _permutation(prim_mesh[0], prim_mesh[1], prim_mesh[2], 1, 0, 1)
    permutation[6] = _permutation(prim_mesh[0], prim_mesh[1], prim_mesh[2], 1, 1, 0)
    permutation[7] = _permutation(prim_mesh[0], prim_mesh[1], prim_mesh[2], 1, 1, 1)
    
    fac = np.sqrt(np.prod(Ls) / np.prod(mesh))
    
    # print("fac = ", fac)
    # print(1/fac)
    
    i=0
    for ix in range(Ls[0]):
        for iy in range(Ls[1]):
            for iz in range(Ls[2]//2+1):
                # final FFT # 
                basis_now = aux_basis[i] 
                basis_now = basis_now * FREQ[i].reshape(1,-1)
                basis_now = basis_now.reshape(-1, prim_mesh[0], prim_mesh[1], prim_mesh[2])
                basis_now = np.fft.fftn(basis_now, axes=(1,2,3))
                basis_now = basis_now.reshape(-1, np.prod(prim_mesh))
                basis_now*=fac
                
                i_begin = i * ngrim_prim
                i_end = (i+1) * ngrim_prim
                
                basis_bench = aux_benchmark[:, i_begin:i_end]
                
                # print(basis_now[:4,:4])
                # print(basis_bench[:4,:4])
                # print(basis_now[:4,:4]/basis_bench[:4,:4])
                
                assert np.allclose(basis_now, basis_bench)