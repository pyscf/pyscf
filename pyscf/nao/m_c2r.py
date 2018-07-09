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
from pyscf.nao.m_fact import sgn

#
#
#
class c2r_c():
  """ Conversion from complex to real harmonics """
  def __init__(self, j):
    self._j = j
    self._c2r = np.zeros( (2*j+1, 2*j+1), dtype=np.complex128)
    self._c2r[j,j]=1.0
    for m in range(1,j+1):
      self._c2r[m+j, m+j] = sgn[m] * np.sqrt(0.5) 
      self._c2r[m+j,-m+j] = np.sqrt(0.5) 
      self._c2r[-m+j,-m+j]= 1j*np.sqrt(0.5)
      self._c2r[-m+j, m+j]= -sgn[m] * 1j * np.sqrt(0.5)
    
    self._hc_c2r = np.conj(self._c2r).transpose()
    self._conj_c2r = np.conjugate(self._c2r) # what is the difference ? conj and conjugate
    self._tr_c2r = np.transpose(self._c2r)
    
    #print(abs(self._hc_c2r.conj().transpose()-self._c2r).sum())

  #
  #
  #
  def c2r_moo(self, j, mab_c, mu2info):
    """ Transform tensor m, orb, orb given in complex spherical harmonics to real spherical harmonic"""
    no = mab_c.shape[1]
    mab_r = np.zeros((2*j+1, no, no)) # result
    
    xww1 = np.zeros((2*j+1, no, no), dtype=np.complex128)
    xww2 = np.zeros(       (no, no), dtype=np.complex128)
    xww3 = np.zeros(       (no, no), dtype=np.complex128)
    _j = self._j
    for m in range(-j,j+1):
      for m1 in range(-abs(m),abs(m)+1,2*abs(m) if m!=0 else 1):
        xww1[j+m,:,:]=xww1[j+m,:,:]+self._hc_c2r[_j+m1,_j+m]*mab_c[j+m1,:,:] # _c2r or _conj_c2r

    for m in range(-j,j+1):
      xww2.fill(0.0)
      for mu1,j1,s1,f1 in mu2info:
        for m1 in range(-j1,j1+1):
          for n1 in range(-abs(m1),abs(m1)+1,2*abs(m1) if m1!=0 else 1):
            xww2[s1+m1+j1,:]=xww2[s1+m1+j1,:]+self._c2r[m1+_j,n1+_j] * xww1[j+m,s1+n1+j1,:]

      xww3.fill(0.0)
      for mu2,j2,s2,f2 in mu2info:
        for m2 in range(-j2,j2+1):
          for n2 in range(-abs(m2),abs(m2)+1,2*abs(m2) if m2!=0 else 1):
            xww3[:,s2+m2+j2]=xww3[:,s2+m2+j2]+self._c2r[m2+_j,n2+_j] * xww2[:,s2+n2+j2]
      
      mab_r[j+m,:,:] = xww3[:,:].real
    return mab_r

  #
  #
  #
  def c2r_(self, j1,j2, jm,cmat,rmat,mat):

    assert(type(mat[0,0])==np.complex128)
    mat.fill(0.0)
    rmat.fill(0.0)
    for mm1 in range(-j1,j1+1):
      for mm2 in range(-j2,j2+1):
        if mm2 == 0 :
          mat[mm1+jm,mm2+jm] = cmat[mm1+jm,mm2+jm]*self._tr_c2r[mm2+self._j,mm2+self._j]
        else :
          mat[mm1+jm,mm2+jm] = \
            (cmat[mm1+jm,mm2+jm]*self._tr_c2r[mm2+self._j,mm2+self._j] + \
             cmat[mm1+jm,-mm2+jm]*self._tr_c2r[-mm2+self._j,mm2+self._j])
        #if j1==2 and j2==1:
        #  print( mm1,mm2, mat[mm1+jm,mm2+jm] )
          
    for mm2 in range(-j2,j2+1):
      for mm1 in range(-j1,j1+1):
        if mm1 == 0 :
          rmat[mm1+jm, mm2+jm] = (self._conj_c2r[mm1+self._j,mm1+self._j]*mat[mm1+jm,mm2+jm]).real
        else :
          rmat[mm1+jm, mm2+jm] = \
            (self._conj_c2r[mm1+self._j,mm1+self._j] * mat[mm1+jm,mm2+jm] + \
             self._conj_c2r[mm1+self._j,-mm1+self._j] * mat[-mm1+jm,mm2+jm]).real
        #if j1==2 and j2==1:
        #  print( mm1,mm2, rmat[mm1+jm,mm2+jm] )
