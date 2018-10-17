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

  def conv_yzx2xyz_2d(self, mat_yzx, direction='pyscf2nao', ss=None):
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

  def conv_yzx2xyz_4d(self, mat_yzx, direction='pyscf2nao', ss=None):

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


if __name__=='__main__':
  """  Computes coulomb overlaps (ab|cd) between 4 orbitals sitting on the same atom with GTO and compares to NAO """
  from pyscf.nao import system_vars_c, ao_matelem_c
  from pyscf.nao.prod_log import prod_log as prod_log_c
  import numpy as np
  from timeit import default_timer as timer
  from scipy.sparse import csr_matrix

  mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvtz') # coordinates in Angstrom!
  sv = system_vars_c(gto=mol)
  t1s = timer()
  prod_log = prod_log_c(sv.ao_log, tol=1e-5)
  print(timer() - t1s)

  t1s = timer()
  me = ao_matelem_c(prod_log)
  print(timer() - t1s)

  m1 = gto.Mole_pure()

  for ia,sp in enumerate(sv.atom2sp):
    pab2v=prod_log.sp2vertex[sp]
    n = pab2v.shape[1]
    pab_shape = [pab2v.shape[0], pab2v.shape[1]*pab2v.shape[2]]
    pab2v_csr = csr_matrix(pab2v.reshape(pab_shape))
    print(pab2v_csr.getnnz(), pab_shape[0]*pab_shape[1])
    t1s = timer()
    coul=me.coulomb_am(sp, [0.0,0.0,0.0], sp, [0.0,0.0,0.0])
    t1s = timer()
    #fci1c = np.einsum('abq,qcd->abcd', np.einsum('pab,pq->abq', pab2v, coul), pab2v)
    qab2tci = coul*pab2v_csr
    fci1c = (qab2tci.transpose()*pab2v_csr).reshape([n,n,n,n])

    print(timer() - t1s)
    m1.build(atom=[mol._atom[ia]], basis=mol.basis)
    eri = m1.intor('cint2e_sph').reshape(fci1c.shape)
    eri = conv_yzx2xyz_c(m1).conv_yzx2xyz_4d(eri, 'pyscf2nao')

    print(fci1c.shape, coul.shape, abs(fci1c-eri).sum()/eri.size, abs(fci1c-eri).max())
