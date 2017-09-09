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
