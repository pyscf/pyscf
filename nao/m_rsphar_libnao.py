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
  libnao.rsphar(r.ctypes.data_as(POINTER(c_double)), c_int(lmax), res.ctypes.data_as(POINTER(c_double)))

  return 0
