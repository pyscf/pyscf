from __future__ import print_function, division
import numpy as np
from pyscf.nao.m_rsphar_libnao import rsphar

#
#
#
def ao_eval_(ao, ra, isp, coords, res):
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
  jmx_sp = ao.sp_mu2j[isp].max()
  rsh = np.zeros((jmx_sp+1)**2)

  for ic,coord in enumerate(coords-ra):
    rsphar(coord, jmx_sp, rsh)
    #comp_coeff_m2p3_k(r2d, a%interp_a, coeff, k)

    for j,ff in zip(ao.sp_mu2j[isp],ao.psi_log[isp]):
      print(j,ff)
    
  return 1.0
