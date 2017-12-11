from __future__ import print_function, division
import numpy as np

try:
    from pyscf.nao.m_numba_utils import ls_contributing_numba
    use_numba = True
except:
    use_numba = False

def ls_contributing(pb, sp12, ra12):
  """
  List of contributing centers 
  prod_basis_c : instance of the prod_basis_c containing parameters .ac_rcut_ratio and .ac_npc_max and .sv providing instance of system_vars_c which provides the coordinates, unit cell vectors, species etc. and .prod_log prividing information on the cutoffs for each specie.
  sp12 : a couple of species
  ra12 : a couple of coordinates, correspondingly
  """
  ra3 = 0.5*(ra12[0,:] + ra12[1,:])
  ia2dist = np.zeros(pb.sv.natoms)

  if use_numba:
      ls_contributing_numba(pb.sv.atom2coord-ra3, ia2dist)
  else:
      for ia,rvec in enumerate(pb.sv.atom2coord-ra3): ia2dist[ia] = np.sqrt(np.dot(rvec, rvec))
  
  return np.argsort(ia2dist)[0:pb.ac_npc_max]
