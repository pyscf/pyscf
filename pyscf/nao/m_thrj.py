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
from pyscf.nao.m_fact import fact as fac, sgn
from ctypes import POINTER, c_double, c_int32, c_int, byref

lmax = 20
uselibnao = True

def comp_number_of3j(lmax):
  """ Computes number of 3j coefficients irreducible by symmetry (as implemented in thrj(l1,l2,l3,m1,m2,m3) )"""
  n3j=0
  for l1 in range(lmax+1):
    for l2 in range(l1+1):
      for m2 in range(-l2,l2+1):
        #for l3 in range(l2+1): n3j=n3j + (2*l3+1)
        n3j=n3j + (l2+1)**2
  return n3j

""" Storage of 3-j coeffcients irreducible by symmetry"""
ixxa = np.zeros(lmax+1, c_int32)
ixxb = np.zeros(lmax+1, c_int32)
ixxc = np.zeros(lmax+1, c_int32)
no3j = comp_number_of3j(lmax)
aa   = np.zeros(no3j, c_double)

if uselibnao :
  from pyscf.nao.m_libnao import libnao

  libnao.init_thrj.argtypes = (
    POINTER(c_int32),  # lmax
    POINTER(c_int32),  # ixxa
    POINTER(c_int32),  # ixxb
    POINTER(c_int32),  # ixxc 
   POINTER(c_double),  # aa
    POINTER(c_int32))  # na

  libnao.init_thrj(c_int32(lmax), 
    ixxa.ctypes.data_as(POINTER(c_int32)),
    ixxb.ctypes.data_as(POINTER(c_int32)),
    ixxc.ctypes.data_as(POINTER(c_int32)),
    aa.ctypes.data_as(POINTER(c_double)),
    c_int32(no3j)) # call library function

else:

  for ii in range(lmax+1):
    ixxa[ii]=ii*(ii+1)*(ii+2)*(2*ii+3)*(3*ii-1)/60
    ixxb[ii]=ii*(ii+1)*(3*ii**2+ii-1)/6
    ixxc[ii]=(ii+1)**2

  ic=-1
  yyx = 0.0
  for l1 in range(lmax+1):
    for l2 in range(l1+1):
      for m2 in range(-l2,l2+1):
        for l3 in range(l2+1):
          for m3 in range(-l3,l3+1):
            m1=-m2-m3
            if l3>=l1-l2 and abs(m1)<=l1:
              lg=l1+l2+l3
              xx=fac[lg-2*l1]*fac[lg-2*l2]*fac[lg-2*l3]/fac[lg+1]
              xx=xx*fac[l3+m3]*fac[l3-m3]/(fac[l1+m1]*fac[l1-m1]*fac[l2+m2]*fac[l2-m2]) 
              itmin=max(0,l1-l2+m3)
              itmax=min(l3-l2+l1,l3+m3)
              ss=0.0
              for it in range(itmin,itmax+1):
                ss = ss + sgn[it]*fac[l3+l1-m2-it]*fac[l2+m2+it]/(fac[l3+m3-it]*fac[it+l2-l1-m3]*fac[it]*fac[l3-l2+l1-it]) 
              yyx=sgn[l2+m2]*np.sqrt(xx)*ss 
            ic=ic+1
            aa[ic]=yyx

# if uselibnao

def thrj(l1i,l2i,l3i,m1i,m2i,m3i):
  """ Wigner3j symbol. Written by James Talman. """

  if abs(m1i)>l1i or abs(m2i)>l2i or abs(m3i)>l3i: return 0.0
  if m1i+m2i+m3i != 0: return 0.0

  l1,l2,l3,m1,m2,m3=l1i,l2i,l3i,m1i,m2i,m3i
  ph = 1.0
  if l1<l2 :
     l2,l1,m2,m1,ph=l1,l2,m1,m2,ph*sgn[l1+l2+l3]

  if l2<l3 :
     l2,l3,m2,m3,ph=l3,l2,m3,m2,ph*sgn[l1+l2+l3]

  if l1<l2 :
     l1,l2,m1,m2,ph=l2,l1,m2,m1,ph*sgn[l1+l2+l3]

  if l1>lmax: raise RuntimeError('thrj: 3-j coefficient out of range')

  if l1>l2+l3: return 0.0
   
  icc=ixxa[l1]+ixxb[l2]+ixxc[l2]*(l2+m2)+ixxc[l3]-l3+m3
  return ph*aa[icc-1]


def thrj_nobuf(l1,l2,l3,m1,m2,m3):
  """ Wigner3j symbol without buffer. Written by James Talman. """
  from pyscf.nao.m_libnao import libnao
  from ctypes import POINTER, c_double, c_int, byref

  libnao.thrj_subr.argtypes = (
    POINTER(c_int),  # l1
    POINTER(c_int),  # l2
    POINTER(c_int),  # l3
    POINTER(c_int),  # m1
    POINTER(c_int),  # m2
    POINTER(c_int),  # m3 
   POINTER(c_double))  # thrj

  aa = c_double()
  libnao.thrj_subr( c_int(l1),c_int(l2),c_int(l3),c_int(m1),c_int(m2),c_int(m3), byref(aa)) # call library function
  return aa.value
