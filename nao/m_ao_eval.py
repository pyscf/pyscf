from __future__ import print_function, division
import numpy as np
from pyscf.nao.m_rsphar_libnao import rsphar
#from pyscf.nao.m_rsphar import rsphar
from pyscf.nao.m_log_interp import comp_coeffs_

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
  coeffs = np.zeros((6))
  res.fill(0.0)
  rcutmx = ao.sp2rcut[isp]
  for icrd,coord in enumerate(coords-ra):
    rsphar(coord, jmx_sp, rsh)
    r = np.sqrt((coord**2).sum())
    if r>rcutmx: continue
    ir = comp_coeffs_(ao.interp_rr, r, coeffs)
    for j,ff,s,f in zip(ao.sp_mu2j[isp],ao.psi_log_rl[isp],ao.sp_mu2s[isp],ao.sp_mu2s[isp][1:]):
      fval = (ff[ir:ir+6]*coeffs).sum() if j==0 else (ff[ir:ir+6]*coeffs).sum()*r**j
      res[s:f,icrd] = fval * rsh[j*(j+1)-j:j*(j+1)+j+1]
  
  return 0

#
# See above
#
def ao_eval(ao, ra, isp, coords):
  res = np.zeros((ao.sp2norbs[isp],coords.shape[0]), dtype='float64')
  ao_eval_(ao, ra, isp, coords, res)
  return res
