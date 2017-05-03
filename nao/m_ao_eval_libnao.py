from __future__ import print_function
import numpy as np
from ctypes import POINTER, c_double, c_int64
from pyscf.nao.m_libnao import libnao

libnao.ao_eval.argtypes = (
  POINTER(c_int64),  # nmult
  POINTER(c_double), # psi_log_rl
  POINTER(c_int64),  # nr
  POINTER(c_double), # rhomin_jt
  POINTER(c_double), # dr_jt 
  POINTER(c_int64),  # mu2j
  POINTER(c_int64),  # mu2s
  POINTER(c_double), # mu2rcut
  POINTER(c_double), # rvec_atom_center 
  POINTER(c_int64),  # ncoords
  POINTER(c_double), # coords
  POINTER(c_int64),  # norbs
  POINTER(c_double), # res[orb, icoord]
  POINTER(c_int64))  # ldres leading dimension of res (ncoords)

#
#
#
def ao_eval_libnao_(ao, rat, isp, crds, res):
  """
    Compute the values of atomic orbitals on given grid points
    Args:
      ao  : instance of ao_log_c class
      rat : vector where the atomic orbitals from "ao" are centered
      isp : specie index for which we compute
      coords: coordinates on which we compute
    Returns:
      res[norbs,ncoord] : array of atomic orbital values
  """
  #print(res_copy.flags)
  rat_copy = np.require(rat,  dtype='float64', requirements='C')
  crd_copy = np.require(crds, dtype='float64', requirements='C')
  res_copy = np.require(res,  dtype='float64', requirements='CW')
  
  libnao.ao_eval(
    c_int64(ao.sp2nmult[isp]), 
    ao.psi_log_rl[isp].ctypes.data_as(POINTER(c_double)),
    c_int64(ao.nr),
    c_double(ao.interp_rr.gammin_jt),
    c_double(ao.interp_rr.dg_jt),
    ao.sp_mu2j[isp].ctypes.data_as(POINTER(c_int64)), 
    ao.sp_mu2s[isp].ctypes.data_as(POINTER(c_int64)),
    ao.sp_mu2rcut[isp].ctypes.data_as(POINTER(c_double)),
    rat_copy.ctypes.data_as(POINTER(c_double)), 
    c_int64(crd_copy.shape[0]), 
    crd_copy.ctypes.data_as(POINTER(c_double)), 
    c_int64(ao.sp2norbs[isp]), 
    res_copy.ctypes.data_as(POINTER(c_double)), 
    c_int64(res.shape[1])  )
  res = res_copy
  return 0

#
# See above
#
def ao_eval_libnao(ao, ra, isp, coords):
  res = np.zeros((ao.sp2norbs[isp],coords.shape[0]), dtype='float64')
  ao_eval_libnao_(ao, ra, isp, coords, res)
  return res


if __name__ == '__main__':
  from pyscf.nao.m_system_vars import system_vars_c
  from pyscf.nao.m_ao_eval import ao_eval
  from pyscf.nao.m_ao_eval_libnao import ao_eval_libnao
  
  sv  = system_vars_c()
  ra = np.array([0.3, -0.5, 0.77], dtype='float64')
  #coords = np.array([[0.07716887, 2.82933578, 3.73214881]])
  coords = np.random.rand(35580,3)*5.0
  
  print('ao_val2 (reference)')
  ao_val1 = ao_eval(sv.ao_log, ra, 0, coords)

  print('ao_val2_libnao')
  ao_val2 = ao_eval_libnao(sv.ao_log, ra, 0, coords)
  
  print(np.allclose(ao_val1,ao_val2))
  for iorb,[oo1,oo2] in enumerate(zip(ao_val1,ao_val2)):
    print(iorb, abs(oo1-oo2).argmax(), abs(oo1-oo2).max(), coords[abs(oo1-oo2).argmax(),:])


