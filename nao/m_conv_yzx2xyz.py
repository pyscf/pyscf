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
    
  def conv_yzx2xyz_1d(self, mat_yzx, m_yzx2m_xyz, sh_slice=None):
    """ Permutation of first index in a possibly multi-dimensional array.
        This would convert nao->pyscf convention"""
    from copy import deepcopy
    _bas = self.mol._bas
    (bs,bf) = (0,len(_bas)) if sh_slice is None else (sh_slice[0],sh_slice[1])
    assert(bs<bf and bf<=len(_bas))
    no = sum([(2*l+1)*nctr for l,nctr in zip(_bas[bs:bf,gto.ANG_OF],_bas[bs:bf,gto.NCTR_OF])]) 
    if(no!=mat_yzx.shape[0]):
      print(no, ' mat_yzx.shape ', mat_yzx.shape) 
      raise SystemError('n_mol!=n_mat')
    mat_xyz = deepcopy(mat_yzx)
    ost = 0
    for ib,[l,nctr] in enumerate(zip(_bas[bs:bf,gto.ANG_OF],_bas[bs:bf,gto.NCTR_OF])):
      if l!=1:
        ost = ost + (2*l+1)*nctr
        continue
      for i in range(nctr):
        for mn,mn_xyz in enumerate(m_yzx2m_xyz):
          mat_xyz[ost+mn_xyz] = mat_yzx[ost+mn]
        ost = ost + 3
    return mat_xyz

  def conv_yzx2xyz_2d(self, mat_yzx, direction='nao2pyscf', ss=None):
    if direction.lower()=='nao2pyscf':
      m2m = self.m_yzx2m_xyz
    elif direction.lower()=='pyscf2nao':
      m2m = self.m_xyz2m_yzx
    else:
      raise RuntimeError('!direction')

    n = len(self.mol._bas)
    sh = [[0,n],[0,n]] if ss is None else [[ss[0],ss[1]],[ss[2],ss[3]]]
      
    o_xyz = self.conv_yzx2xyz_1d(mat_yzx, m2m, sh[0])
    o_xyz = self.conv_yzx2xyz_1d(o_xyz.transpose(), m2m, sh[1])
    return o_xyz

  def conv_yzx2xyz_4d(self, mat_yzx, direction='nao2pyscf', ss=None):

    if direction.lower()=='nao2pyscf':
      m2m = self.m_yzx2m_xyz
    elif direction.lower()=='pyscf2nao':
      m2m = self.m_xyz2m_yzx
    else:
      raise RuntimeError('!direction')

    n = len(self.mol._bas)
    sh = [[0,n],[0,n],[0,n],[0,n]] if ss is None else [[ss[0],ss[1]],[ss[2],ss[3]],[ss[4],ss[5]],[ss[6],ss[7]]]

    o_xyz = self.conv_yzx2xyz_1d(mat_yzx, m2m, sh[0])
    o_xyz = self.conv_yzx2xyz_1d(np.einsum('abcd->bcda', o_xyz), m2m, sh[1])
    o_xyz = self.conv_yzx2xyz_1d(np.einsum('bcda->cdab', o_xyz), m2m, sh[2])
    o_xyz = self.conv_yzx2xyz_1d(np.einsum('cdab->dabc', o_xyz), m2m, sh[3])
    return np.einsum('dabc->abcd', o_xyz)

