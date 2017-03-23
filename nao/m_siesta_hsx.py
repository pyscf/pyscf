from ctypes import POINTER, c_int64, c_float, c_char_p, create_string_buffer
import os
from pyscf.lib import misc 

dll = misc.load_library("libnao")

# interfacing with fortran subroutines 
dll.siesta_hsx_size.argtypes = (c_char_p, POINTER(c_int64), POINTER(c_int64))
dll.siesta_hsx_read.argtypes = (c_char_p, POINTER(c_int64), POINTER(c_float))
# END of interfacing with fortran subroutines 

import numpy
import sys
from numpy import empty 

#
#
#
def siesta_hsx_read(label='siesta', force_type=-1): 

  fname = create_string_buffer(label+'.HSX')
  ft = c_int64(force_type)
  bufsize = c_int64()
  dll.siesta_hsx_size(fname, ft, bufsize)
    
  dat = empty(bufsize.value, dtype="float32")
  dll.siesta_hsx_read(fname, ft, dat.ctypes.data_as(POINTER(c_float)))
  return dat
  

class siesta_hsx_c():
  def __init__(self, label, force_type):
    dat = siesta_hsx_read(label, force_type)
    i = 0
    self.norbs    =  int(dat[i]); i=i+1;
    self.norbs_sc =  int(dat[i]); i=i+1;
    self.nspin    =  int(dat[i]); i=i+1;
    self.nnz      =  int(dat[i]); i=i+1;     
    self.is_gamma = (dat[i]>0); i=i+1;
    self.Ne       =  dat[i]; i=i+1;
    self.Te       =  dat[i]; i=i+1;
    self.H4 = numpy.reshape(dat[i:i+self.nnz*self.nspin], (self.nnz,self.nspin), order='F'); i=i+self.nnz*self.nspin;
    self.S4 = dat[i:i+self.nnz]; i = i + self.nnz;
    self.X4 = numpy.reshape(dat[i:i+self.nnz*3], (3,self.nnz), order='F'); i = i + self.nnz*3;
    self.row_ptr = numpy.array(dat[i:i+self.norbs+1], dtype='int'); i = i + self.norbs+1;
    self.col_ind = numpy.array(dat[i:i+self.nnz], dtype='int'); i = i + self.nnz;
    if(i<len(dat)):
	  if(self.is_gamma): raise SystemError('i<len(dat) && gamma')
	  self.orb_sc2orb_uc = numpy.array(dat[i:i+self.norbs_sc], dtype='int'); i = i + self.norbs_sc
    if(i!=len(dat)): raise SystemError('i!=len(dat)')  
