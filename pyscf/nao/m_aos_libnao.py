from __future__ import print_function
import numpy as np
from ctypes import POINTER, c_double, c_int64
from pyscf.nao.m_libnao import libnao

libnao.aos_libnao.argtypes = (
  POINTER(c_int64),  # ncoords
  POINTER(c_double), # coords
  POINTER(c_int64),  # norbs
  POINTER(c_double), # res[icoord, orb]
  POINTER(c_int64))  # ldres leading dimension (fastest changing dimension) of res (norbs)


""" The purpose of this is to evaluate the atomic orbitals at a given set of atomic coordinates """
def aos_libnao(coords, norbs):
  assert len(coords.shape) == 2
  assert coords.shape[1] == 3
  assert norbs>0

  ncoords = coords.shape[0]
  co2val = np.require( np.zeros((ncoords,norbs)), dtype=c_double, requirements='C')
  
  crd_copy = np.require(coords, dtype=c_double, requirements='C')
  libnao.aos_libnao(
    c_int64(ncoords),
    crd_copy.ctypes.data_as(POINTER(c_double)),
    c_int64(norbs),
    co2val.ctypes.data_as(POINTER(c_double)),
    c_int64(norbs))

  return co2val




