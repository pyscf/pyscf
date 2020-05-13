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
from scipy.spatial.distance import cdist

from pyscf.nao.m_rsphar_libnao import rsphar
from pyscf.nao.m_log_interp import comp_coeffs_
#
#
#
def ao_eval(ao, ra, isp, coords):
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
  dd = cdist(ra.reshape((1,3)), coords).reshape(-1)
  mu_c2pao = ao.interp_rr.interp_csr(ao.psi_log_rl[isp], dd)

  res = np.zeros((ao.sp2norbs[isp],coords.shape[0]))
  jmx_sp = ao.sp_mu2j[isp].max()
  rsh = np.zeros((jmx_sp+1)**2)
  rcutmx = ao.sp2rcut[isp]
  for icrd,coord in enumerate(coords-ra):
    rsphar(coord, jmx_sp, rsh)
    r = dd[icrd]
    if r>rcutmx: continue
    
    for mu,(j,s,f) in enumerate(zip(ao.sp_mu2j[isp],ao.sp_mu2s[isp],ao.sp_mu2s[isp][1:])):
      fval = mu_c2pao[mu,icrd] if j==0 else mu_c2pao[mu,icrd]*r**j
      res[s:f,icrd] = fval * rsh[j*(j+1)-j:j*(j+1)+j+1]

  return res
