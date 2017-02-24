from __future__ import print_function
import numpy

#
#
#
def conv_yzx2xyz_1d(mol, mat_yzx):
  from pyscf import gto
  from copy import deepcopy

  n_mol = mol.nao_nr()
  n_mat = mat_yzx.shape[0]
  if(n_mol!=n_mat): raise SystemError('n_mol!=n_mat')

  mat_xyz = deepcopy(mat_yzx)
  
  lm_yzx2m_xyz = []
  jmx = max(mol._bas[:,gto.ANG_OF])
  for l in range(jmx+1):
    if(l==1) :
      lm_yzx2m_xyz.append([1,2,0])
    else:
      lm_yzx2m_xyz.append(range(2*l+1))
    
  #o = -1
  ost = 0
  #mu = -1
  for ib in range(mol.nbas):
    l  = mol.bas_angular(ib)
    for i in range(mol.bas_nctr(ib)):
      #mu = mu + 1
      for mn in range(2*l+1):
        #o = o + 1
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

