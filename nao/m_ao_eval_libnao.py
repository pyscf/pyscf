from __future__ import print_function
import numpy as np
from ctypes import POINTER, c_double, c_int64
from pyscf.nao.m_libnao import libnao

libnao.ao_eval.argtypes = (
  POINTER(c_int64), 
  POINTER(c_double), 
  POINTER(c_int64),
  POINTER(c_double),
  POINTER(c_double), 
  POINTER(c_int64), 
  POINTER(c_int64), 
  POINTER(c_double), 
  POINTER(c_int64), 
  POINTER(c_double),
  POINTER(c_int64), 
  POINTER(c_double),
  POINTER(c_int64))

#
#
#
def ao_eval_libnao_(ao, ra, isp, coords, res):
  """
    Compute the values of atomic orbitals on given grid points
    Args:
      ao  : instance of ao_log_c class
      ra  : vector where the atomic orbitals from "ao" are centered
      isp : specie index for which we compute
      coords: coordinates on which we compute
    Returns:
      res[norbs,ncoord] : array of atomic orbital values
  """
  libnao.ao_eval(c_int64(ao.sp2nmult[isp]), 
    ao.psi_log_rl[isp].ctypes.data_as(POINTER(c_double)),
    c_int64(ao.nr),
    c_double(ao.interp_rr.gammin_jt),
    c_double(ao.interp_rr.dg_jt),
    ao.sp_mu2j[isp].ctypes.data_as(POINTER(c_int64)), 
    ao.sp_mu2s[isp].ctypes.data_as(POINTER(c_int64)), 
    ra.ctypes.data_as(POINTER(c_double)), 
    c_int64(coords.shape[0]), 
    coords.ctypes.data_as(POINTER(c_double)), 
    c_int64(ao.sp2norbs[isp]), 
    res.ctypes.data_as(POINTER(c_double)), 
    c_int64(res.shape[1]))
    
  return 0

#
# See above
#
def ao_eval_libnao(ao, ra, isp, coords):
  res = np.zeros((ao.sp2norbs[isp],coords.shape[0]), dtype='float64')
  ao_eval_libnao_(ao, ra, isp, coords, res)
  return res
