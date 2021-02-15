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

#
# Marc: numba would be nice here
#
def xjl(xx,lc):
  '''
  Spherical bessel functions
  Computes a table of j_l(x) for fixed xx, Eq. (39)
  Args:
    xx : float
    lc : integer angular momentum
  Result:
    xj[0:lc] : float
  '''
  assert(lc>-1)
  
  xj = np.zeros((lc+1), dtype='float64')
  
  if abs(xx)<1.0e-10:
    xj[0] = 1.0
    return(xj)

  sin_xx_div_xx = np.sin(xx)/xx

  if xx < 0.75*lc :
     aam,aa,bbm,bb,sa,qqm = 1.0, (2*lc+1)/xx, 0.0, 1.0, -1.0, 1e10
     for k in range(1,51):
        sb = (2*(lc+k)+1)/xx
        aap,bbp = sb*aa+sa*aam,sb*bb+sa*bbm
        aam,bbm = aa,bb
        aa,bb   = aap,bbp
        qq      = aa/bb
        if abs(qq-qqm)<1.0e-15 : break
        qqm = qq
  
     xj[lc] = 1.0
     if lc > 0 : xj[lc-1] = qq
     if lc > 1 :
        for l in range(lc-1,0,-1):
          xj[l-1] = (2*l+1)*xj[l]/xx-xj[l+1]

     cc = sin_xx_div_xx/xj[0]
     for l in range(lc+1): xj[l] = cc*xj[l]

  else :
     xj[0] = sin_xx_div_xx
     if lc > 0: xj[1] = xj[0]/xx-np.cos(xx)/xx
     if lc > 1:
        for l in range(1,lc): xj[l+1] = (2*l+1)*xj[l]/xx-xj[l-1]

  return(xj)
