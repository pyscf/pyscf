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
import unittest, numpy as np
from pyscf import gto
from pyscf.nao import nao, conv_yzx2xyz_c

mol = gto.M( verbose = 1,
    atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''', basis = 'cc-pvdz',)
conv = conv_yzx2xyz_c(mol)
sv = nao(gto=mol)

class KnowValues(unittest.TestCase):

  def test_verify(self):
      print(mol.spin, __file__)

  def test_gto2sv(self):
    """ Test transformation of the radial orbitals from GTO to NAO type"""
    psi_log = sv.ao_log.psi_log
    self.assertEqual((sv.natoms,sv.norbs,len(psi_log)), (3,24,2))
    rr = sv.ao_log.rr
    self.assertEqual(len(rr), 1024)
    dr = np.log(rr[1]/rr[0])
    for mu2ff in psi_log:
      for ff in mu2ff:
        norm = (ff**2*rr**3).sum()*dr
        self.assertAlmostEqual(norm, 1.0)

  def test_atom2sv(self):
    """ Test costructing a skeleton for later use to define spherical grid with pySCF """
    dl = [ [1, [1.0, 0.44, 2.0]], [8, [0.0, 0.0, 0.1]], [1, [0.0, 0.0, -2.0]]]
    sv = nao(xyz_list=dl)
    
    self.assertEqual(sv.natoms, len(dl))
    for ia,a in enumerate(dl): 
      self.assertEqual(sv.sp2charge[sv.atom2sp[ia]], a[0])
      self.assertTrue(np.all(sv.atom2coord[ia,:]==a[1]))
    self.assertTrue(sv.atom2s is None)

  def test_overlap_gto_vs_nao(self):
    """ Test computation of overlaps computed between NAOs against overlaps computed between GTOs"""
    from pyscf.nao.m_overlap_am import overlap_am
    oref = conv.conv_yzx2xyz_2d(mol.intor_symmetric('cint1e_ovlp_sph'))
    over = sv.overlap_coo(funct=overlap_am).toarray()
    self.assertTrue(abs(over-oref).sum()<5e-9)

  def test_laplace_gto_vs_nao(self):
    """ Test computation of kinetic energy between NAOs against those computed between GTOs"""
    from pyscf.nao.m_laplace_am import laplace_am
    tref = conv.conv_yzx2xyz_2d(mol.intor_symmetric('int1e_kin'))
    tkin = (-0.5*sv.overlap_coo(funct=laplace_am)).toarray()
    self.assertTrue(abs(tref-tkin).sum()/len(tkin)<5e-9)

  def test_energy_nuc_gto_vs_nao(self):
    """ Test computation of matrix elements of nuclear-electron attraction """
    sv = nao(gto=mol)
    e_nao = sv.energy_nuc()
    e_gto = mol.energy_nuc()
    self.assertAlmostEqual(e_nao, e_gto)

if __name__ == "__main__": 
    unittest.main()

