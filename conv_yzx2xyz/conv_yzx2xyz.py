from __future__ import print_function
import numpy as np

#
#
#
def conv_yzx2xyz_1d(mol, mat_yzx):
  from pyscf import gto
  from copy import deepcopy

  n_mat = mat_yzx.shape[0]
  mat_xyz = deepcopy(mat_yzx)
  
  lm_yzx2m_xyz = []
  jmx = max(mol._bas[:,gto.ANG_OF])
  for l in range(jmx+1):
    if(l==1) :
      lm_yzx2m_xyz.append([1,2,0])
    else:
      lm_yzx2m_xyz.append(range(2*l+1))

  ost = 0
  for ib in range(mol.nbas):
    l  = mol._bas[ib,gto.ANG_OF]
    for i in range(mol._bas[ib,gto.NCTR_OF]):
      for mn in range(2*l+1):
        mn_xyz = lm_yzx2m_xyz[l][mn]
        mat_xyz[ost+mn_xyz] = mat_yzx[ost+mn]
      ost = ost + 2*l+1

  return mat_xyz

#
#
#
def conv_yzx2xyz_2d(mol, mat_yzx):
  o_xyz = conv_yzx2xyz_1d(mol, mat_yzx)
  o_xyz = conv_yzx2xyz_1d(mol, o_xyz.transpose())
  return o_xyz

#
#
#
def conv_yzx2xyz_4d(mol, mat_yzx):
  o_xyz = conv_yzx2xyz_1d(mol, mat_yzx)
  o_xyz = conv_yzx2xyz_1d(mol, np.einsum('abcd->bcda', o_xyz))
  o_xyz = conv_yzx2xyz_1d(mol, np.einsum('bcda->cdab', o_xyz))
  o_xyz = conv_yzx2xyz_1d(mol, np.einsum('cdab->dabc', o_xyz))
  return np.einsum('dabc->abcd', o_xyz)
