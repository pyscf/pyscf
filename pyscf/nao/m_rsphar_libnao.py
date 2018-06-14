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

from __future__ import division, print_function
import numpy as np
from ctypes import POINTER, c_double, c_int64
from pyscf.nao.m_libnao import libnao

libnao.rsphar.argtypes = (POINTER(c_double), POINTER(c_int64), POINTER(c_double))
libnao.rsphar_vec.argtypes = (POINTER(c_double), POINTER(c_int64), POINTER(c_int64), POINTER(c_double))
libnao.rsphar_exp_vec.argtypes = (POINTER(c_double), POINTER(c_int64), POINTER(c_int64), POINTER(c_double))

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
  assert r.shape[-1]==3
  
  r_cp = np.require(r,  dtype=float, requirements='C')
  res = np.require(res,  dtype=float, requirements='CW')
  
  libnao.rsphar(r_cp.ctypes.data_as(POINTER(c_double)), c_int64(lmax), res.ctypes.data_as(POINTER(c_double)))
  return 0

#
#
#
def rsphar_vec(rvs,lmax):
  """
    Computes (all) real spherical harmonics up to the angular momentum lmax
    Args:
      rvs : Cartesian coordinates defining correct theta and phi angles for spherical harmonic
      lmax : Integer, maximal angular momentum
    Result:
      1-d numpy array of float64 elements with all spherical harmonics stored in order 0,0; 1,-1; 1,0; 1,+1 ... lmax,lmax, althogether 0 : (lmax+1)**2 elements.
  """
  assert rvs.shape[-1]==3
  r_cp = np.require(rvs,  dtype=float, requirements='C')
  nc = len(rvs)
  res = np.require( np.zeros((nc, (lmax+1)**2)), dtype=float, requirements='CW')
  libnao.rsphar_vec(r_cp.ctypes.data_as(POINTER(c_double)), c_int64(nc), c_int64(lmax), res.ctypes.data_as(POINTER(c_double)))
  
  #for irv,rvec in enumerate(rvs): rsphar(rvec,lmax,res[irv,:])
  return res

#
#
#
def rsphar_exp_vec(rvs,lmax):
  """
    Computes (all) real spherical harmonics up to the angular momentum lmax
    Args:
      rvs : Cartesian coordinates defining correct theta and phi angles for spherical harmonic
      lmax : Integer, maximal angular momentum
    Result:
      1-d numpy array of float64 elements with all spherical harmonics stored in order 0,0; 1,-1; 1,0; 1,+1 ... lmax,lmax, althogether 0 : (lmax+1)**2 elements.
  """  
  assert rvs.shape[0]==3
  r_cp = np.require(rvs,  dtype=np.float64, requirements='C')
  nc = rvs[0,...].size
  res = np.require( np.zeros(((lmax+1)**2,nc)), dtype=np.float64, requirements='CW')
  libnao.rsphar_exp_vec(r_cp.ctypes.data_as(POINTER(c_double)), c_int64(nc), c_int64(lmax), res.ctypes.data_as(POINTER(c_double)))
  
  #for irv,rvec in enumerate(rvs): rsphar(rvec,lmax,res[irv,:])
  return res
