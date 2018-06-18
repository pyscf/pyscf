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

import numpy as np
from numpy import empty 

#
#
#
def _siesta2blanko_csr(orb2m, mat, orb_sc2orb_uc=None):
  #print(' m    ',  len(orb2m))
  #print(' data ',  len(mat.data))
  #print(' col  ',  len(mat.indices))
  #print(' ptr  ',  len(mat.indptr))
  n = len(orb2m)
  if(n!=len(mat.indptr)-1): raise SystemError('!number of orbitals should be equal to number of rows?')
  
  ptr2ph = np.zeros_like(mat.data, dtype = np.float64)
  
  if orb_sc2orb_uc is None:
    orb_sc2m = orb2m
  else:
    orb_sc2m = np.zeros_like(orb_sc2orb_uc)
    for orb_sc,orb_uc in enumerate(orb_sc2orb_uc): orb_sc2m[orb_sc] = orb2m[orb_uc]
  
  for row in range(n):
    ph1, s, f = (-1.0)**orb2m[row], mat.indptr[row], mat.indptr[row+1]
    ptr2ph[s:f] = ph1 * (-1.0)**orb_sc2m[mat.indices[s:f]]

  mat.data = mat.data*ptr2ph
  return 0

#
#
#
def _siesta2blanko_csr_slow(orb2m, mat, orb_sc2orb_uc=None):
  #print(' m    ',  len(orb2m))
  #print(' data ',  len(mat.data))
  #print(' col  ',  len(mat.indices))
  #print(' ptr  ',  len(mat.indptr))
  n = len(orb2m)
  if(n!=len(mat.indptr)-1): raise SystemError('!mat')

  for row in range(n):
    m1  = orb2m[row]
    col_data = mat.data[mat.indptr[row]:mat.indptr[row+1]]
    col_data = col_data * (-1)**m1
    
    col_phase = numpy.empty((len(col_data)), dtype='int8')
    for ind in range(mat.indptr[row],mat.indptr[row+1]):
      icol = mat.indices[ind]
      if(icol>=n): icol=orb_sc2orb_uc[icol]
      m2 = orb2m[icol]
      col_phase[ind-mat.indptr[row]] = (-1.0)**m2
    
    col_data = col_data*col_phase
    mat.data[mat.indptr[row]:mat.indptr[row+1]] = col_data
  return 0
