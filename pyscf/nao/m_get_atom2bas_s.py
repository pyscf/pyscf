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

def get_atom2bas_s(_bas):
  """
    For a given _bas list (see mole and mole_pure  from pySCF)
    constructs a list of atom --> start shell
    The list is natoms+1 long, i.e. larger than number of atoms 
    The list can be used to get the start,finish indices of pySCF's multipletts^*
    This is useful to compose shls_slice arguments for the pySCF integral evaluators .intor(...)
      ^* pySCF multipletts can be "repeated" number of contractions times
  """
  natoms = max([bb[0] for bb in _bas])+1
  atom2bas_s = np.array([len(_bas)]*(natoms+1), dtype=np.int32)
  for ib,[at,l,ngto,nctr,a,b,c,d] in enumerate(_bas): atom2bas_s[at] = min(atom2bas_s[at],ib)
  return atom2bas_s


if __name__ == '__main__':
  """
    Compute only bilocal part of the four-orbitals, two-center Coulomb integrals
  """
  from pyscf import gto
  from pyscf.nao.m_system_vars import system_vars_c
  from pyscf.nao.m_conv_yzx2xyz import conv_yzx2xyz_c

  tol = 1e-5
  
  mol = gto.M(atom='O 0 0 0; H 0 -0.1 1; H 0 0.1 -1', basis='ccpvdz')
  sv = system_vars_c(gto=mol)
  na = sv.natm
  for ia1,n1 in zip(range(na), sv.atom2s[1:]-sv.atom2s[0:na]):
    for ia2,n2 in zip(range(ia1+1,sv.natm+1), sv.atom2s[ia1+2:]-sv.atom2s[ia1+1:na]):
      mol2 = gto.Mole_pure(atom=[mol._atom[ia1], mol._atom[ia2]], basis=mol.basis).build()
      bs = get_atom2bas_s(mol2._bas)
      ss = (bs[0],bs[1], bs[1],bs[2], bs[0],bs[1], bs[1],bs[2])
      eri = mol2.intor('cint2e_sph', shls_slice=ss).reshape([n1,n2,n1,n2])

      eri = conv_yzx2xyz_c(mol2).conv_yzx2xyz_4d(eri, 'pyscf2nao', ss).reshape([n1*n2,n1*n2])
      ee,xx = np.linalg.eigh(eri)

      nlinindep = list(ee>tol).count(True)
      print(' ia1, ia2, n1, n2: ', ia1, ia2, n1, n2, eri.shape, n1*n2, nlinindep, n1*n2/nlinindep)

