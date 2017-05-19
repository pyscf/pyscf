from __future__ import print_function
import numpy as np
from pyscf import gto

#
#
#
class conv_yzx2xyz_c():
  '''
  A class to organize a permutation of l=1 matrix elements
  '''
  def __init__(self, mol):
    self.mol = mol
    self.norbs = ((mol._bas[:,gto.ANG_OF]*2+1) * mol._bas[:,gto.NCTR_OF]).sum()
    self.m_yzx2m_xyz = np.array([1,2,0], dtype='int32')
    self.m_xyz2m_yzx = np.array([2,0,1], dtype='int32')
    
  def conv_yzx2xyz_1d(self, mat_yzx, m_yzx2m_xyz):
    """ Permutation of first index in a possibly multi-dimensional array.
        This would convert nao->pyscf convention"""
    from copy import deepcopy
    
    if(self.norbs!=mat_yzx.shape[0]): raise SystemError('n_mol!=n_mat')
    mat_xyz = deepcopy(mat_yzx)
    ost = 0
    for ib,[l,nctr] in enumerate(zip(self.mol._bas[:,gto.ANG_OF],self.mol._bas[:,gto.NCTR_OF])):
      if l!=1:
        ost = ost + (2*l+1)*nctr
        continue
      for i in range(nctr):
        for mn,mn_xyz in enumerate(m_yzx2m_xyz):
          mat_xyz[ost+mn_xyz] = mat_yzx[ost+mn]
        ost = ost + 3
    return mat_xyz

  def conv_yzx2xyz_2d(self, mat_yzx, direction='nao2pyscf'):
    if direction.lower()=='nao2pyscf':
      m2m = self.m_yzx2m_xyz
    elif direction.lower()=='pyscf2nao':
      m2m = self.m_xyz2m_yzx
    else:
      raise RuntimeError('!direction')
      
    o_xyz = self.conv_yzx2xyz_1d(mat_yzx, m2m)
    o_xyz = self.conv_yzx2xyz_1d(o_xyz.transpose(), m2m)
    return o_xyz

  def conv_yzx2xyz_4d(self, mat_yzx, direction='nao2pyscf'):

    if direction.lower()=='nao2pyscf':
      m2m = self.m_yzx2m_xyz
    elif direction.lower()=='pyscf2nao':
      m2m = self.m_xyz2m_yzx
    else:
      raise RuntimeError('!direction')

    o_xyz = self.conv_yzx2xyz_1d(mat_yzx, m2m)
    o_xyz = self.conv_yzx2xyz_1d(np.einsum('abcd->bcda', o_xyz), m2m)
    o_xyz = self.conv_yzx2xyz_1d(np.einsum('bcda->cdab', o_xyz), m2m)
    o_xyz = self.conv_yzx2xyz_1d(np.einsum('cdab->dabc', o_xyz), m2m)
    return np.einsum('dabc->abcd', o_xyz)

