from __future__ import division, print_function
import numpy as np
from ctypes import POINTER, c_double, c_int
from pyscf.nao.m_libnao import libnao

libnao.rsphar.argtypes = (POINTER(c_double), POINTER(c_int), POINTER(c_double))

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
  r_copy = np.require(r,  dtype='float64', requirements='C')
  res = np.require(res,  dtype='float64', requirements='CW')
  
  libnao.rsphar(r_copy.ctypes.data_as(POINTER(c_double)), c_int(lmax), res.ctypes.data_as(POINTER(c_double)))
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
  #r_copy = np.require(rvs,  dtype=float, requirements='C')
  #nc = len(rvs)
  #res = np.require(np.zeros((nc,3)),  dtype=float, requirements='CW')
  #libnao.rsphar_vec(r_copy.ctypes.data_as(POINTER(c_double)), c_int64(nc), c_int64(lmax), res.ctypes.data_as(POINTER(c_double)))

  res = np.zeros((rvs.shape[0], (lmax+1)**2))
  for irv,rvec in enumerate(rvs): rsphar(rvec,lmax,res[irv,:])
  return res
