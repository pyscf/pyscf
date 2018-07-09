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
from ctypes import POINTER, c_int64, c_float, c_double, c_char_p, create_string_buffer
import os
import sys
import numpy as np
from numpy import zeros, empty 
from pyscf.nao.m_libnao import libnao

# interfacing with fortran subroutines 
libnao.siesta_wfsx_book_size.argtypes = (c_char_p, POINTER(c_int64), POINTER(c_int64), POINTER(c_int64))
libnao.siesta_wfsx_book_read.argtypes = (c_char_p, POINTER(c_int64), POINTER(c_int64), POINTER(c_int64))
libnao.siesta_wfsx_dread.argtypes = (c_char_p, POINTER(c_int64), POINTER(c_double), POINTER(c_int64))
libnao.siesta_wfsx_sread.argtypes = (c_char_p, POINTER(c_int64), POINTER(c_float), POINTER(c_int64))
# END of interfacing with fortran subroutines 

#
#
#
def siesta_wfsx_book_read_py(fname, nreim):
  """ Creates buffer for integer data from .WFSX files """
  name = create_string_buffer(fname.encode())
  bufsize = c_int64(-999)
  ios = c_int64(22)
  libnao.siesta_wfsx_book_size(name, c_int64(nreim), bufsize, ios)
  if ios.value!=0 : return None
  idat = empty(bufsize.value, dtype=np.int64)
  libnao.siesta_wfsx_book_read(name, c_int64(nreim), idat.ctypes.data_as(POINTER(c_int64)), ios)
  if ios.value!=0 : return None
  return idat

#
#
#
def siesta_wfsx_dread(w, nreim):
  ddata = empty(w.nkpoints*w.nspin*w.norbs + + w.nkpoints*3)
  ios = c_int64(-999)
  libnao.siesta_wfsx_dread(create_string_buffer(w.fname.encode()), c_int64(nreim), ddata.ctypes.data_as(POINTER(c_double)), ios)
  if ios.value!=0 : raise RuntimeError('ios!=0 %d'%(ios.value))
  return ddata

#
#
#
def siesta_wfsx_sread(w, sdata, nreim):
  name = create_string_buffer(w.fname.encode())
  bufsize = w.nkpoints*w.nspin*w.norbs**2*w.nreim
  ios = c_int64(-999)
  libnao.siesta_wfsx_sread(name, c_int64(nreim), sdata.ctypes.data_as(POINTER(c_float)), ios)
  if ios.value!=0 : raise RuntimeError('ios!=0 %d'%(ios.value))


class siesta_wfsx_c():
  def __init__(self, label='siesta', chdir='.', fname=None, force_gamma=None):

    nreim = -999
    if force_gamma is not None:
      if force_gamma : nreim = 1
    
    if fname is None :
      self.label = label
      ends = ['fullBZ.WFSX', 'WFSX']
      for end in ends:
        fname = chdir+'/'+label+'.'+end
        idat = siesta_wfsx_book_read_py(fname, nreim)
        if idat is None :
          print(fname, ' skip') 
          continue
        self.fname = fname
        break
    else:
      self.fname = fname
      idat = siesta_wfsx_book_read_py(fname, nreim)
                
    if idat is None :  raise RuntimeError('No .WFSX file found')
    
    i = 0
    self.nkpoints = idat[i]; i=i+1
    self.nspin    = idat[i]; i=i+1
    self.norbs    = idat[i]; i=i+1
    self.gamma    = idat[i]>0 if force_gamma is None else force_gamma; i=i+1
    self.orb2atm  = idat[i:i+self.norbs]; i=i+self.norbs
    self.orb2ao   = idat[i:i+self.norbs]; i=i+self.norbs
    self.orb2n    = idat[i:i+self.norbs]; i=i+self.norbs
    if(self.gamma) :
      self.nreim = 1;
    else: 
      self.nreim = 2;
    
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
      symlabel = '' # make oneliner here (to oneline)
      for k in range(symlen):
        symlabel = symlabel + chr(idat[i]); i=i+1
      self.orb2strsym.append(symlabel.strip())

    ### Read double precision data
    ddata = siesta_wfsx_dread(self, self.nreim)

    self.ksn2e = empty((self.nkpoints,self.nspin,self.norbs))
    self.k2xyz = empty((self.nkpoints,3))
    i = 0
    for k in range(self.nkpoints):
      for s in range(self.nspin):
        for n in range(self.norbs):
          self.ksn2e[k,s,n] = ddata[i]; i=i+1

    for k in range(self.nkpoints):
      for j in range(3):
          self.k2xyz[k,j] = ddata[i]; i=i+1

    ### Read single precision data
    
    self.x = np.require(zeros((self.nkpoints,self.nspin,self.norbs,self.norbs,self.nreim), dtype=np.float32), requirements='CW')
    siesta_wfsx_sread(self, self.x, self.nreim)
