# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

from __future__ import print_function, division
import numpy as np
from ctypes import POINTER, c_float, c_int64
from pyscf.nao.m_libnao import libnao

libnao.c_csr_bloch_mat.argtypes = (
  POINTER(c_int64),   # r2s     ! row -> start index in data and cols arrays
  POINTER(c_int64),   # nrows   ! number of rows (number of orbitals in unit cell)
  POINTER(c_int64),   # i2col   ! index -> column
  POINTER(c_float),   # i2dat   ! index -> data element
  POINTER(c_float),   # i2xyz   ! index -> coordinate difference
  POINTER(c_int64),   # nnz     ! number of non-zero matrix elements (maximal index)
  POINTER(c_int64),   # orb_sc2orb_uc ! orbital in Super Cell -> orbital in Unit cell correspondence
  POINTER(c_int64),   # norbs_sc      ! size of the previous array
  POINTER(c_float),   # kvec          ! Cartesian coordinates of a k-vector
  POINTER(c_float*2)) # cmat          ! Bloch matrix (nrows,nrows)

def siesta_hsx_bloch_mat(csr, hsx, kvec=np.array([0.0,0.0,0.0])):
  assert csr.nnz==hsx.x4.shape[0]
  assert hsx.norbs==len(csr.indptr)-1
  
  r2s   = np.require( csr.indptr, dtype=np.int64, requirements='C')
  i2col = np.require( csr.indices, dtype=np.int64, requirements='C')
  i2dat = np.require( csr.data, dtype=np.float32, requirements='C')
  i2xyz = np.require( hsx.x4, dtype=np.float32, requirements='C')
  osc2o = np.require( hsx.orb_sc2orb_uc, dtype=np.int64, requirements='C')
  kvecc = np.require( kvec, dtype=np.float32, requirements='C')
  cmat  = np.require( np.zeros((hsx.norbs,hsx.norbs), dtype=np.complex64), requirements='CW')
  
  libnao.c_csr_bloch_mat( r2s.ctypes.data_as(POINTER(c_int64)),
    c_int64(hsx.norbs),
    i2col.ctypes.data_as(POINTER(c_int64)),
    i2dat.ctypes.data_as(POINTER(c_float)),
    i2xyz.ctypes.data_as(POINTER(c_float)),
    c_int64(csr.nnz),
    osc2o.ctypes.data_as(POINTER(c_int64)),
    c_int64(hsx.norbs_sc),
    kvecc.ctypes.data_as(POINTER(c_float)),
    cmat.ctypes.data_as(POINTER(c_float*2))
  )
  return cmat

#
#
#
def siesta_hsx_bloch_mat_py(csr, hsx, kvec=[0.0,0.0,0.0], prec=64):
  assert(type(prec)==int)
  assert csr.nnz==hsx.x4.shape[0]
  assert hsx.norbs==len(csr.indptr)-1

  caux = np.exp(1.0j*np.dot(hsx.x4,kvec)) * csr.data

  den_bloch = np.zeros((hsx.norbs,hsx.norbs), dtype='complex'+str(2*prec))
  for row in range(hsx.norbs):
    for ind in range(csr.indptr[row], csr.indptr[row+1]):
      col = hsx.orb_sc2orb_uc[csr.indices[ind]]
      den_bloch[col,row]=den_bloch[col,row]+caux[ind]

  return den_bloch
