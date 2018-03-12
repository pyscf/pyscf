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
import scipy as sp
from pyscf.nao.m_thrj import thrj
#from sympy.physics.quantum.cg import Wigner3j as w3j
#from sympy.physics.wigner import gaunt as gau

#
#
#
class gaunt_c():
  """ Computation of Gaunt coefficient (precompute then return) """
  def __init__(self, jmax=7):
    
    self.jmax = jmax
    
    self.njm = (jmax+1)**2
    nptr = self.njm**2
    
    ncef = 0
    for j1 in range(jmax+1):
      for j2 in range(jmax+1):
        ncef = ncef + ((j2+j1)-abs(j1-j2)+1)*(2*j1+1)*(2*j2+1)
    
    self._gaunt_data = np.zeros(ncef)
    self._gaunt_iptr = np.zeros(nptr+1, dtype=np.int64)

    ptr = 0
    for j1 in range(jmax+1):
      for m1 in range(-j1,j1+1):
        i1 = j1*(j1+1)+m1
        for j2 in range(jmax+1):
          for m2 in range(-j2,j2+1):
            i2 = j2*(j2+1)+m2
            ind = i1*self.njm+i2
            ptr = ptr + ((j2+j1)-abs(j1-j2)+1)
            self._gaunt_iptr[ind+1] = ptr

    for j1 in range(jmax+1):
      for m1 in range(-j1,j1+1):
        i1 = j1*(j1+1)+m1
        for j2 in range(jmax+1):
          for m2 in range(-j2,j2+1):
            i2 = j2*(j2+1)+m2
            ind = i1*self.njm+i2
            s = self._gaunt_iptr[ind]
            for j3ind,j3 in enumerate(range(abs(j1-j2), j1+j2+1)):
              #self._gaunt_data[s+j3ind] = np.sqrt( (2*j1+1.0)*(2*j2+1.0)*(2*j3+1.0)/(4.0*np.pi) ) * \
              #  w3j(j1,0,j2,0,j3,0).doit()*w3j(j1,m1,j2,m2,j3,-m1-m2).doit() # slow and accurate

              #self._gaunt_data[s+j3ind] = gau(j1,j2,j3,m1,m2,-m1-m2,prec=1e-10) # slow and wrong

              self._gaunt_data[s+j3ind] = np.sqrt( (2*j1+1.0)*(2*j2+1.0)*(2*j3+1.0)/(4.0*np.pi) ) * \
                thrj(j1,j2,j3,0,0,0)*thrj(j1,j2,j3,m1,m2,-m1-m2) # fast and accurate

  #
  def get_gaunt(self,j1,m1,j2,m2):

    i1 = j1*(j1+1)+m1
    i2 = j2*(j2+1)+m2
    ind = i1*self.njm+i2
    s,f = self._gaunt_iptr[ind],self._gaunt_iptr[ind+1]
    return self._gaunt_data[s:f]
