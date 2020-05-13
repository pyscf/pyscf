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
from pyscf import gto, scf
from pyscf.nao import nao, scf as scf_nao, conv_yzx2xyz_c

mol = gto.M( verbose = 1,
    atom = '''
        H     0    0        0
        H     0    0.757    0.587''', basis = 'cc-pvdz',)
conv = conv_yzx2xyz_c(mol)
gto_hf = scf.RHF(mol)
gto_hf.kernel()
rdm1 = conv.conv_yzx2xyz_2d(gto_hf.make_rdm1())

class KnowValues(unittest.TestCase):

  def test_kmat_gto_vs_nao(self):
    """ Test computation of Fock exchange between NAOs against this computed between GTOs"""
    vh_gto,k_gto = gto_hf.get_jk()
    k_gto = conv.conv_yzx2xyz_2d(k_gto)
    mf = scf_nao(mf=gto_hf, gto=mol)
    k_nao = mf.get_k(dm=rdm1)
    self.assertTrue(abs(k_nao-k_gto).sum()/k_gto.size<2.5e-5)
    
  def test_overlap_gto_vs_nao(self):
    """ Test computation of overlaps between NAOs against overlaps computed between GTOs"""
    from pyscf.nao.m_overlap_am import overlap_am
    oref = conv.conv_yzx2xyz_2d(mol.intor_symmetric('cint1e_ovlp_sph'))
    sv = nao(gto=mol)
    over = sv.overlap_coo(funct=overlap_am).toarray()
    self.assertTrue(abs(over-oref).sum()<5e-9)

  def test_laplace_gto_vs_nao(self):
    """ Test computation of kinetic energy between NAOs against those computed between GTOs"""
    from pyscf.nao.m_laplace_am import laplace_am
    tref = conv.conv_yzx2xyz_2d(mol.intor_symmetric('int1e_kin'))
    sv = nao(gto=mol)
    tkin = (-0.5*sv.overlap_coo(funct=laplace_am)).toarray()
    self.assertTrue(abs(tref-tkin).sum()/len(tkin)<5e-9)

  def test_vhartree_gto_vs_nao(self):
    """ Test computation of Hartree potential between NAOs against this computed between GTOs"""
    vh_gto = conv.conv_yzx2xyz_2d(gto_hf.get_j())
    scf = scf_nao(mf=gto_hf, gto=mol)
    vh_nao = scf.vhartree_coo(dm=rdm1)
    self.assertTrue(abs(vh_nao-vh_gto).sum()/vh_gto.size<1e-5)

  def test_vne_gto_vs_nao(self):
    """ Test computation of matrix elements of nuclear-electron attraction """
    vne = mol.intor_symmetric('int1e_nuc')
    vne_gto = conv.conv_yzx2xyz_2d(vne)
    sv = nao(gto=mol)
    vne_nao = sv.vnucele_coo_coulomb(level=1)
    #print('a,b,c', (vne_nao).sum(), (vne_gto).sum(), abs(vne_nao-vne_gto).sum()/vne_gto.size)
    self.assertTrue(abs(vne_nao-vne_gto).sum()/vne_gto.size<5e-6)

  def test_energy_nuc_gto_vs_nao(self):
    """ Test computation of matrix elements of nuclear-electron attraction """
    sv = nao(gto=mol)
    e_nao = sv.energy_nuc()
    e_gto = mol.energy_nuc()
    self.assertAlmostEqual(e_nao, e_gto)
    
if __name__ == "__main__": unittest.main()
