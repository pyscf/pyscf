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

from __future__ import division, print_function
import numpy as np
try:
  import numba
  use_numba = True
except:
  use_numba = False

"""
well described here  http://www.netlib.org/lapack/lug/node123.html
"""

#
#
#
def pack2den_u(pack):
  """ Unpacks a packed format to dense format """
  dim = size2dim(len(pack))
  den = np.zeros((dim,dim))
  for i in range(dim):
    for j in range(i+1):
      den[i,j] = pack[ij2pack_u(i,j)]
      den[j,i] = den[i,j]
  return den

#
#
#
def pack2den_l(pack):
  """ Unpacks a packed format to dense format """
  dim = size2dim(len(pack))
  den = np.zeros((dim,dim))
  for j in range(dim):
    for i in range(j,dim):
      den[j,i] = pack[ij2pack_l(i,j,dim)]
      den[i,j] = den[j,i]
  return den

#
#
#
def size2dim(pack_size):
  """
    Computes dimension of the matrix by it's packed size. Packed size is n*(n+1)//2
  """
  rdim = (np.sqrt(8.0*pack_size+1.0) - 1.0)/2.0
  ndim = int(rdim)
  if ndim != rdim : raise SystemError('!ndim/=rdim')
  if ndim*(ndim+1)//2 != pack_size: SystemError('!pack_size')
  return ndim


def ij2pack_u(i,j):
  ma = max(i,j)
  return ma*(ma+1)//2+min(i,j)
  

def ij2pack_l(i,j,dim):
  mi = min(i,j)
  return max(i,j)+((2*dim-mi-1)*mi)//2


if use_numba:
  @numba.jit(nopython=True)
  def cp_block_pack_u(bmat, s1, f1, s2, f2, gmat, add=False):
    if add:
      for i in range(s1,f1):
        for j in range(s2,f2):
          if j>i: continue
          ma = max(i,j)
          gmat[ma*(ma+1)//2+min(i,j)] += bmat[i-s1, j-s2]
    else:  
      for i in range(s1,f1):
        for j in range(s2,f2):
          ma = max(i,j)
          gmat[ma*(ma+1)//2+min(i,j)] = bmat[i-s1, j-s2]  
else:
  def cp_block_pack_u(bmat, s1, f1, s2, f2, gmat, add=False):
    if add:
      for i in range(s1,f1):
        for j in range(s2,f2):
          if j>i: continue
          ma = max(i,j)
          gmat[ma*(ma+1)//2+min(i,j)] += bmat[i-s1, j-s2]
    else:  
      for i in range(s1,f1):
        for j in range(s2,f2):
          ma = max(i,j)
          gmat[ma*(ma+1)//2+min(i,j)] = bmat[i-s1, j-s2]  
