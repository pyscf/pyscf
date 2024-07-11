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

import sys

import numpy
import numpy as np

from pyscf.pbc.gto import Cell
import pyscf.pbc.gto as pbcgto

def symmetrize_dm(dm:np.ndarray, Ls):
    '''
    
    generate translation symmetrized density matrix (by average)
    
    Args :
        dm : np.ndarray, density matrix, shape = (nao, nao)
        Ls : list, supercell dimension, shape = (3,), or kmesh in k-sampling

    Returns :
        dm_symm : np.ndarray, symmetrized density matrix, shape = (nao, nao)
    '''
    
    is_single_dm = False
    
    if dm.ndim == 2:
        is_single_dm = True
        dm = dm.reshape(1, dm.shape[0], dm.shape[1])
        
    ncell = np.prod(Ls)
    nao   = dm.shape[1]
    nset  = dm.shape[0]
    nao_prim = nao // ncell
    dm_symm = np.zeros((nset,nao,nao), dtype=dm.dtype)
        
    for i in range(Ls[0]):
        for j in range(Ls[1]):
            for k in range(Ls[2]):
                
                dm_symmized_buf = np.zeros((nset,nao_prim,nao_prim), dtype=dm.dtype)
                
                for i_row in range(Ls[0]):
                    for j_row in range(Ls[1]):
                        for k_row in range(Ls[2]):
                            
                            loc_row = i_row * Ls[1] * Ls[2] + j_row * Ls[2] + k_row
                            loc_col = ((i + i_row) % Ls[0]) * Ls[1] * Ls[2] + ((j + j_row) % Ls[1]) * Ls[2] + (k + k_row) % Ls[2]
                            
                            b_begin = loc_row * nao_prim
                            b_end   = (loc_row + 1) * nao_prim
                            
                            k_begin = loc_col * nao_prim
                            k_end   = (loc_col + 1) * nao_prim
                            
                            dm_symmized_buf += dm[:,b_begin:b_end, k_begin:k_end]
        
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
                            
                            dm_symm[:,b_begin:b_end, k_begin:k_end] = dm_symmized_buf        
    
    if is_single_dm:
        return dm_symm[0]
    else:
        return dm_symm        

def pack_JK(input_mat:np.ndarray, Ls, nao_prim, output=None):
    
    '''
    pack matrix in real space
    '''
    
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
 
def pack_JK_in_FFT_space(input_mat:np.ndarray, kmesh, nao_prim, output=None):
    
    '''
    pack matrix in k-space
    '''
    
    ncomplex = kmesh[0] * kmesh[1] * (kmesh[2] // 2 + 1)
    assert input_mat.dtype == np.complex128
    assert input_mat.shape[0] == nao_prim
    #print("input_mat.shape = ", input_mat.shape)
    #print("nao_prim = ", nao_prim)
    #print("ncomplex = ", ncomplex)
    assert input_mat.shape[1] == nao_prim * ncomplex
    
    nkpts = np.prod(kmesh)
    
    if output is None:
        output = np.zeros((nao_prim, nao_prim*nkpts), dtype=np.complex128)
    else:
        assert output.shape == (nao_prim, nao_prim*nkpts) or output.shape == (nkpts, nao_prim, nao_prim)
    
    output = output.reshape(nkpts, nao_prim, nao_prim)
    
    loc = 0
    
    for ix in range(kmesh[0]):
        for iy in range(kmesh[1]):
            for iz in range(kmesh[2] // 2 + 1):
                loc1 = ix * kmesh[1] * kmesh[2] + iy * kmesh[2] + iz
                #loc2 = ix * kmesh[1] * kmesh[2] + iy * kmesh[2] + (kmesh[2] - iz) % kmesh[2]
                loc2 = (kmesh[0] - ix) % kmesh[0] * kmesh[1] * kmesh[2] + (kmesh[1] - iy) % kmesh[1] * kmesh[2] + (kmesh[2] - iz) % kmesh[2]
                if loc1 == loc2:
                    output[loc1] = input_mat[:, loc*nao_prim:(loc+1)*nao_prim]
                    imag_part = np.imag(output[loc1])
                    if np.max(np.abs(imag_part)) > 1e-8:
                        print("Warning: max abs of imag_part = ", np.max(np.abs(imag_part)))
                else:
                    output[loc1] = input_mat[:, loc*nao_prim:(loc+1)*nao_prim]
                    output[loc2] = input_mat[:, loc*nao_prim:(loc+1)*nao_prim].conj()
                loc += 1
                
    return output

