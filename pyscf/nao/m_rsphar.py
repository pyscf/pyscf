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

from __future__ import division
import numpy as np
from pyscf.nao.m_fact import sgn, onedivsqrt4pi,rttwo

lmx = 7
l2lmhl = np.array( [0]+[np.sqrt((l-0.5)/l) for l in range(1,lmx+1) ])
l2tlm1 = np.array( [0]+[np.sqrt(2*l-1.0) for l in range(1,lmx+1) ])
l2tlp1 = np.array( [np.sqrt(2*l+1.0) for l in range(lmx+1) ])
lm2aa = np.zeros(((lmx+1)**2))
lm2bb = np.zeros(((lmx+1)**2))

for l in range(lmx+1):
  for m, ind in zip(range(-l,l+1), range(l*(l+1)-l,l*(l+1)+l+1)):
    lm2aa[ind] = np.sqrt(1.0*l**2-m**2)
    lm2bb[ind] = 1.0/np.sqrt(1.0*(l+1)**2-m**2)

#
#
#
def rsphar(r,lmax,res):
  """
    Computes (all) real spherical harmonics up to the angular momentum lmax
    Args:
      r : Cartesian coordinates defining correct theta and phi angles for spherical harmonic
      lmax : Integer, maximal angular momentum
    Result:
      1-d numpy array of float64 elements with all spherical harmonics stored in order 0,0; 1,-1; 1,0; 1,+1 ... lmax,lmax, althogether 0 : (lmax+1)**2 elements.
  """
  
  xxpyy = r[0]**2 + r[1]**2
  dd=np.sqrt(xxpyy + r[2]**2)

  if dd < 1e-10:
    res.fill(0.0);
    res[0]=onedivsqrt4pi
    return 0

  if r[0]==0.0:
    phi = 0.5*np.pi if r[1]<0.0 else -0.5*np.pi
  else:
    phi = np.arctan( r[1]/r[0] ) if r[0]>=0.0 else np.arctan( r[1]/r[0] )+np.pi

  res[0]=onedivsqrt4pi
  if lmax==0: return 0

  ss=np.sqrt(xxpyy)/dd 
  cc=r[2]/dd

  for l in range(1,lmax+1): 
    twol,l2 = l+l,l*l
    il1,il2 = l2-1,l2+twol
    res[il2]=-ss*l2lmhl[l]*res[il1] 
    res[il2-1]=cc*l2tlm1[l]*res[il1]

  if lmax>=2:
    for m in range(lmax-1):
      if m<lmax:
        for l in range(m+1,lmax):
          ind=l*(l+1)+m
          zz=(l+l+1)*cc*res[ind]-lm2aa[ind]*res[ind-l-l] 
          res[ind+l+l+2]=zz*lm2bb[ind]

  for l in range(lmax+1):
    ll2=l*(l+1)
    res[ll2] = res[ll2]*l2tlp1[l]
    for m in range(1,l+1):
      cs,cc,P = np.sin(m*phi), np.cos(m*phi), res[ll2+m]*sgn[m]*l2tlp1[l]*rttwo
      res[ll2+m]=cc*P
      res[ll2-m]=cs*P
  return 0
