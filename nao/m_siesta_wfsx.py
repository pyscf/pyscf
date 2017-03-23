from ctypes import POINTER, c_int64, c_float, c_double, c_char_p, c_int, create_string_buffer
import os
import sys
import numpy
import sys
from numpy import empty 
from pyscf.lib import misc


dll = misc.load_library("libnao")

# interfacing with fortran subroutines 
dll.siesta_wfsx_book_size.argtypes = (c_char_p, POINTER(c_int64))
dll.siesta_wfsx_book_read.argtypes = (c_char_p, POINTER(c_int))
dll.siesta_wfsx_dread.argtypes = (c_char_p, POINTER(c_double))
dll.siesta_wfsx_sread.argtypes = (c_char_p, POINTER(c_float))

# END of interfacing with fortran subroutines 

#
#
#
def siesta_wfsx_book_read_py(label='siesta'): 

  name = create_string_buffer(label)
  bufsize = c_int64()
  dll.siesta_wfsx_book_size(name, bufsize)
    
  idat = empty(bufsize.value, dtype="int32")
  dll.siesta_wfsx_book_read(name, idat.ctypes.data_as(POINTER(c_int)))
  return idat

#
#
#
def siesta_wfsx_dread(w):
  ddata = numpy.empty(w.nkpoints*w.nspin*w.norbs + + w.nkpoints*3, 'float64')
  dll.siesta_wfsx_dread(create_string_buffer(w.label), ddata.ctypes.data_as(POINTER(c_double)))
  return ddata

#
#
#
def siesta_wfsx_sread(w, sdata):
  name = create_string_buffer(w.label)
  bufsize = w.nkpoints*w.nspin*w.norbs**2*w.nreim
  dll.siesta_wfsx_sread(name, sdata.ctypes.data_as(POINTER(c_float)))

  
class siesta_wfsx_c():
  def __init__(self, label):

    self.label    = label
    ### Read integer data
    idat = siesta_wfsx_book_read_py(label)
    i = 0
    self.nkpoints = idat[i]; i=i+1
    self.nspin    = idat[i]; i=i+1
    self.norbs    = idat[i]; i=i+1
    self.gamma    = idat[i]>0; i=i+1
    self.orb2atm  = idat[i:i+self.norbs]; i=i+self.norbs
    self.orb2ao   = idat[i:i+self.norbs]; i=i+self.norbs
    self.orb2n    = idat[i:i+self.norbs]; i=i+self.norbs
    if(self.gamma) : self.nreim = 1;
    else: self.nreim = 2;
    

    splen    = idat[i]; i=i+1
    self.orb2strspecie = []
    for j in range(self.norbs):
      splabel = ''
      for k in range(splen):
        splabel = splabel + chr(idat[i]); i=i+1
      self.orb2strspecie.append(splabel.strip())

    self.sp2strspecie = []
    for strsp in self.orb2strspecie:
      if strsp not in self.sp2strspecie:
        self.sp2strspecie.append(strsp)

    symlen   = idat[i]; i=i+1
    self.orb2strsym = []
    for j in range(self.norbs):
      symlabel = ''
      for k in range(symlen):
        symlabel = symlabel + chr(idat[i]); i=i+1
      self.orb2strsym.append(symlabel.strip())

    ### Read double precision data
    ddata = siesta_wfsx_dread(self)

    self.E = numpy.empty((self.norbs,self.nspin,self.nkpoints), dtype='float64', order='F')
    self.kpoints = numpy.empty((3,self.nkpoints), dtype='float64', order='F')
    i = 0
    for k in range(self.nkpoints):
      for s in range(self.nspin):
        for j in range(self.norbs):
          self.E[j,s,k] = ddata[i]; i=i+1

    for k in range(self.nkpoints):
      for j in range(3):
          self.kpoints[j,k] = ddata[i]; i=i+1

    ### Read single precision data
    self.X = numpy.empty((self.nreim,self.norbs,self.norbs,self.nspin,self.nkpoints), dtype='float32', order='F')
    siesta_wfsx_sread(self, self.X)
