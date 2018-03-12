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
from ctypes import POINTER, c_double, c_int
from pyscf.nao.m_libnao import libnao

libnao.csphar_talman.argtypes = (
  POINTER(c_double),  # rvec(3)
  POINTER(2*c_double),  # ylm(0:(lmax+1)**2)
  POINTER(c_int) )    # lmax
  
#
#
#
def csphar_talman_libnao(rvec, lmax):
  assert len(rvec)==3
  assert lmax>-1
  
  rvec_sp = np.require(rvec, dtype=float, requirements='C')
  ylm = np.require( np.zeros(((lmax+1)**2), dtype=np.complex128), dtype=np.complex128, requirements='C')

  libnao.csphar_talman(rvec_sp.ctypes.data_as(POINTER(c_double)), 
                       ylm.ctypes.data_as(POINTER(2*c_double)),
                       c_int(lmax) )
  return ylm

#
#
#
def talman2world(ylm_t):

  lmax = int(np.sqrt(len(ylm_t)))-1
  ylm_w = np.zeros(len(ylm_t), dtype=np.complex128)
    
  i = -1
  for l in range(lmax+1):
    for m in range(-l,l+1):
      i = i + 1
      ylm_w[i] = ylm_t[i]*np.sqrt((2*l+1)/(4*np.pi))
      
  return ylm_w
  
