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
from numpy import zeros,extract, array
from scipy.sparse import csr_matrix
from timeit import default_timer as timer

#
def lsofcsr(coo3, dtype=float, shape=None, axis=0):
  """
    Generate a list of csr matrices out of a 3-dimensional coo format 
    Args:
      coo3  : must be a tuple (data, (i1,i2,i3)) in analogy to the tuple (data, (rows,cols)) for a common coo format
      shape : a tuple of dimensions if they are known or cannot be guessed correctly from the data
      axis  : index (0,1 or 2) along which to construct the list of sparse arrays
    Returns:
      list of csr matrices
  """
  (d, it) = coo3
  assert len(it)==3
  for ia in it: assert len(d)==len(ia)
  shape = [max(ia)+1 for ia in it] if shape is None else shape
  #print( len(d) )
  #print( shape )
  
  iir = [i for i in range(len(shape)) if i!=axis]
  #print(__name__, iir)
  #print(__name__, type(it), type(it[0]==0))
  #print(__name__, it[0]==0)
  
  lsofcsr = [0] * shape[axis]
  sh = [shape[i] for i in iir]
  for i in range(shape[axis]):
    mask = it[axis]==i
    csrm = csr_matrix( (extract(mask,d), (extract(mask,it[iir[0]]),extract(mask,it[iir[1]]) )), shape=sh, dtype=dtype)
    csrm.eliminate_zeros()
    lsofcsr[i] = csrm
    
  #print(__name__, 'ttot', ttot)
  
  return lsofcsr

#
#
#
class lsofcsr_c():

  def __init__(self, coo3, dtype=float, shape=None, axis=0):
    self.lsofcsr = lsofcsr(coo3, dtype=dtype, shape=shape, axis=axis)
    self.shape = shape
    self.axis = axis
    self.dtype = dtype

  def __getitem__(self, i):
    return self.lsofcsr[i] 
  
  def __len__(self):
    return len(self.lsofcsr)
         
  def toarray(self):
    vab_arr = zeros(self.shape, dtype=self.dtype)
    if self.axis==2 :
      for b,m in enumerate(self): vab_arr[:,:,b]=m.toarray()
    elif self.axis==1:
      for b,m in enumerate(self): vab_arr[:,b,:]=m.toarray()
    elif self.axis==0:
      for b,m in enumerate(self): vab_arr[b,:,:]=m.toarray()
    else:
      print('??? self.axis ', self.axis)
      raise RuntimeError('!impl')

    return vab_arr
