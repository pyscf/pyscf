from __future__ import print_function, division
from ctypes import POINTER, c_int64, c_float, c_char_p, create_string_buffer
from pyscf.nao.m_libnao import libnao

# interfacing with fortran subroutines 
libnao.siesta_hsx_size.argtypes = (c_char_p, POINTER(c_int64), POINTER(c_int64))
libnao.siesta_hsx_read.argtypes = (c_char_p, POINTER(c_int64), POINTER(c_float))
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
  
  bufsize = c_int64()
  libnao.siesta_hsx_size(fname, ft, bufsize)
  if bufsize.value<=0 : return None
  dat = empty(bufsize.value, dtype=np.float32)
  libnao.siesta_hsx_read(fname, ft, dat.ctypes.data_as(POINTER(c_float)))
  return dat

#
#
#
class siesta_hsx_c():
  def __init__(self, fname='siesta.HSX', force_gamma=None):
    
    self.fname = fname
    dat = siesta_hsx_read(fname, force_gamma)
    if dat is None: raise RuntimeError('file HSX not found '+ fname)
    i = 0
    self.norbs    =  int(dat[i]); i=i+1;
    self.norbs_sc =  int(dat[i]); i=i+1;
    self.nspin    =  int(dat[i]); i=i+1;
    self.nnz      =  int(dat[i]); i=i+1;
    self.is_gamma = (dat[i]>0); i=i+1;
    self.nelec    =  dat[i]; i=i+1;
    self.telec    =  dat[i]; i=i+1;
    self.h4 = np.reshape(dat[i:i+self.nnz*self.nspin], (self.nspin,self.nnz)); i=i+self.nnz*self.nspin;
    self.s4 = dat[i:i+self.nnz]; i = i + self.nnz;
    self.x4 = np.reshape(dat[i:i+self.nnz*3], (self.nnz,3)); i = i + self.nnz*3;
    self.row_ptr = np.array(dat[i:i+self.norbs+1]-1, dtype='int'); i = i + self.norbs+1;
    self.col_ind = np.array(dat[i:i+self.nnz]-1, dtype='int'); i = i + self.nnz;
    self.spin2h4_csr = []
    for s in range(self.nspin):
      self.spin2h4_csr.append(csr_matrix((self.h4[s,:], self.col_ind, self.row_ptr), dtype=np.float32))
    self.s4_csr = csr_matrix((self.s4, self.col_ind, self.row_ptr), dtype=np.float32)

    self.orb_sc2orb_uc=None
    if(i<len(dat)):
      if(self.is_gamma): raise SystemError('i<len(dat) && gamma')
      self.orb_sc2orb_uc = np.array(dat[i:i+self.norbs_sc]-1, dtype='int'); i = i + self.norbs_sc
    if(i!=len(dat)): raise SystemError('i!=len(dat)')  
