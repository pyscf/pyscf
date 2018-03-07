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
from ctypes import POINTER, c_int64, c_float, c_char_p, create_string_buffer
from pyscf.nao.m_libnao import libnao

# interfacing with fortran subroutines 
libnao.siesta_hsx_size.argtypes = (c_char_p, POINTER(c_int64), POINTER(c_int64), POINTER(c_int64), POINTER(c_int64))
libnao.siesta_hsx_read.argtypes = (c_char_p, POINTER(c_int64), POINTER(c_float),
                                    POINTER(c_int64), POINTER(c_int64), 
                                    POINTER(c_int64), POINTER(c_int64), 
                                    POINTER(c_int64))
# END of interfacing with fortran subroutines 

import numpy as np
import sys
from scipy.sparse import csr_matrix
from numpy import empty 

#
#
#
def siesta_hsx_read(fname, force_gamma=None): 

  fname = create_string_buffer(fname.encode())
  if force_gamma is None: 
    ft = c_int64(-1)
  elif force_gamma: 
    ft = c_int64(1)
  elif not force_gamma: 
    ft = c_int64(2)
  
  bufsize, row_ptr_size, col_ind_size = c_int64(), c_int64(), c_int64()
  libnao.siesta_hsx_size(fname, ft, bufsize, row_ptr_size, col_ind_size)
  if bufsize.value<=0 or row_ptr_size.value <= 0 or col_ind_size.value <= 0: return None
  
  dat = empty(bufsize.value, dtype=np.float32)
  dimensions = empty(4, dtype=np.int64)
  row_ptr = empty(row_ptr_size.value, dtype=np.int64)
  col_ind = empty(col_ind_size.value, dtype=np.int64)

  libnao.siesta_hsx_read(fname, ft, dat.ctypes.data_as(POINTER(c_float)),
      row_ptr.ctypes.data_as(POINTER(c_int64)), row_ptr_size,
      col_ind.ctypes.data_as(POINTER(c_int64)), col_ind_size,
      dimensions.ctypes.data_as(POINTER(c_int64)))
  return dat, row_ptr, col_ind, dimensions

#
#
#
class siesta_hsx_c():
  def __init__(self, fname='siesta.HSX', force_gamma=None):
    
    self.fname = fname
    dat, row_ptr, col_ind, dimensions = siesta_hsx_read(fname, force_gamma)
    if dat is None or row_ptr is None or col_ind is None:
      raise RuntimeError('file HSX not found '+ fname)

    self.norbs, self.norbs_sc, self.nspin, self.nnz = dimensions
    i = 0
    self.is_gamma = (dat[i]>0); i=i+1;
    self.nelec    =  int(dat[i]); i=i+1;
    self.telec    =  dat[i]; i=i+1;
    self.h4 = np.reshape(dat[i:i+self.nnz*self.nspin], (self.nspin,self.nnz)); i=i+self.nnz*self.nspin;
    self.s4 = dat[i:i+self.nnz]; i = i + self.nnz;
    self.x4 = np.reshape(dat[i:i+self.nnz*3], (self.nnz,3)); i = i + self.nnz*3;
    self.spin2h4_csr = []
    for s in range(self.nspin):
      self.spin2h4_csr.append(csr_matrix((self.h4[s,:], col_ind, row_ptr), dtype=np.float32))
    self.s4_csr = csr_matrix((self.s4, col_ind, row_ptr), dtype=np.float32)

    self.orb_sc2orb_uc=None
    if(i<len(dat)):
      if(self.is_gamma): raise SystemError('i<len(dat) && gamma')
      self.orb_sc2orb_uc = np.array(dat[i:i+self.norbs_sc]-1, dtype='int'); i = i + self.norbs_sc
    if(i!=len(dat)): raise SystemError('i!=len(dat)')  

  def deallocate(self):
    del self.h4
    del self.s4
    del self.x4
    del self.spin2h4_csr
    del self.s4_csr
