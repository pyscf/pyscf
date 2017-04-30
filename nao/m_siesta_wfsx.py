from ctypes import POINTER, c_int64, c_float, c_double, c_char_p, c_int, create_string_buffer
import os
import sys
import numpy
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

  name = create_string_buffer(label.encode())
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
  dll.siesta_wfsx_dread(create_string_buffer(w.label.encode()), ddata.ctypes.data_as(POINTER(c_double)))
  return ddata

#
#
#
def siesta_wfsx_sread(w, sdata):
  name = create_string_buffer(w.label.encode())
  bufsize = w.nkpoints*w.nspin*w.norbs**2*w.nreim
  dll.siesta_wfsx_sread(name, sdata.ctypes.data_as(POINTER(c_float)))


  
class siesta_wfsx_c():
  def __init__(self, label='siesta'):

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
    
    # list of caracter that could be used to split the psf file name
    splen    = idat[i]; i=i+1
    self.orb2strspecie = []
    for j in range(self.norbs):
      splabel = ''
      for k in range(splen):
        splabel = splabel + chr(idat[i]); i=i+1
      splabel = splabel.replace(" ", "")

      ch = splabel
      
      self.orb2strspecie.append(ch)

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

    self.ksn2e = numpy.empty((self.nkpoints,self.nspin,self.norbs), dtype='float64', order='F')
    self.k2xyz = numpy.empty((self.nkpoints,3), dtype='float64', order='F')
    i = 0
    for k in range(self.nkpoints):
      for s in range(self.nspin):
        for n in range(self.norbs):
          self.ksn2e[k,s,n] = ddata[i]; i=i+1

    for k in range(self.nkpoints):
      for j in range(3):
          self.k2xyz[k,j] = ddata[i]; i=i+1

    ### Read single precision data
    self.X = numpy.empty((self.nreim,self.norbs,self.norbs,self.nspin,self.nkpoints), dtype='float32', order='F')
    siesta_wfsx_sread(self, self.X)
