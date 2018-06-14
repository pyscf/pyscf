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
from ctypes import c_int, sizeof

#
#
#
class openmx_mat_c():

  def __init__(self, natoms, Total_NumOrbs, FNAN, natn):
    """
      Bookkeeper for the OpenMX matrices (costom sparse format from OpenMX (and Fireball?)) 
      natoms : number of atoms in the unit cell
      Total_NumOrbs : array of size (natoms+1) in which we store number of atomic orbitals per each atom. The array is one-based?
      FNAN : array of size (natoms+1). the number of first nearest neighbouring atoms per atom. The array is one-based ?
      natn : array of size (natoms+1,max(FNAN)+1) grobal index of neighboring atoms of an atom ct_AN
    """
    assert natoms>0
    self.natoms = natoms
    assert len(Total_NumOrbs)==natoms+1
    self.Total_NumOrbs = Total_NumOrbs
    self.Total_NumOrbs_mx = max(Total_NumOrbs)
    assert len(FNAN)==natoms+1
    self.FNAN = FNAN
    self.FNAN_mx = max(FNAN)
    assert natn.shape==(natoms+1, max(FNAN)+1)
    self.natn = natn

  
  def get_dims(self):
    return [self.natoms+1, self.FNAN_mx+1, self.Total_NumOrbs_mx, self.Total_NumOrbs_mx]
  
  def fromfile(self, f, out=None, dtype=np.float):
    """ Read from an open file f """
    if out is None:
      res = np.zeros((self.natoms+1, self.FNAN_mx+1, self.Total_NumOrbs_mx, self.Total_NumOrbs_mx), dtype=dtype)
    else :
      res = out

    for ct_AN in range(1,self.natoms+1):
      for h_AN in range(0,self.FNAN[ct_AN]+1):
        for i in range(self.Total_NumOrbs[ct_AN]):
          c = self.Total_NumOrbs[self.natn[ct_AN,h_AN]]
          res[ct_AN,h_AN,i,0:c] = np.fromfile(f, count=c)
    
    return res

  
  
