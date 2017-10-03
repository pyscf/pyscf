from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import system_vars_c, conv_yzx2xyz_c

mol = gto.M( verbose = 1,
    atom = '''
        H     0    0        0
        H     0    0.757    0.587''', basis = 'cc-pvdz',)
conv = conv_yzx2xyz_c(mol)
gto_hf = scf.RHF(mol)
gto_hf.kernel()
rdm1 = conv.conv_yzx2xyz_2d(gto_hf.make_rdm1())

class KnowValues(unittest.TestCase):
    
  def test_overlap_gto_vs_nao(self):
    """ Test computation of overlaps between NAOs against overlaps computed between GTOs"""
    from pyscf.nao.m_overlap_am import overlap_am
    oref = conv.conv_yzx2xyz_2d(mol.intor_symmetric('cint1e_ovlp_sph'))
    sv = system_vars_c().init_pyscf_gto(mol)
    over = sv.overlap_coo(funct=overlap_am).toarray()
    self.assertTrue(abs(over-oref).sum()<5e-9)

  def test_laplace_gto_vs_nao(self):
    """ Test computation of kinetic energy between NAOs against those computed between GTOs"""
    from pyscf.nao.m_laplace_am import laplace_am
    tref = conv.conv_yzx2xyz_2d(mol.intor_symmetric('int1e_kin'))
    sv = system_vars_c().init_pyscf_gto(mol)
    tkin = (0.5*sv.overlap_coo(funct=laplace_am)).toarray()
    self.assertTrue(abs(tref-tkin).sum()/len(tkin)<5e-9)

  def test_vhartree_gto_vs_nao(self):
    """ Test computation of Hartree potential between NAOs against this computed between GTOs"""
    vh_gto = conv.conv_yzx2xyz_2d(gto_hf.get_j())
    sv = system_vars_c().init_pyscf_gto(mol)
    vh_nao = sv.vhartree_coo(dm=rdm1)
    self.assertTrue(abs(vh_nao-vh_gto).sum()/vh_gto.size<1e-5)

  def test_kmat_gto_vs_nao(self):
    """ Test computation of Fock exchange between NAOs against this computed between GTOs"""
    vh_gto,k_gto = gto_hf.get_jk()
    k_gto = conv.conv_yzx2xyz_2d(k_gto)
    sv = system_vars_c().init_pyscf_gto(mol)
    k_nao = sv.kmat_den(dm=rdm1)
    self.assertTrue(abs(k_nao-k_gto).sum()/k_gto.size<5e-5)

  def test_vne_gto_vs_nao(self):
    """ Test computation of matrix elements of nuclear-electron attraction """
    vne = mol.intor_symmetric('int1e_nuc')
    vne_gto = conv.conv_yzx2xyz_2d(vne)
    sv = system_vars_c().init_pyscf_gto(mol)
    vne_nao = sv.vnucele_coo_coulomb(level=1)
    #print('a,b,c', (vne_nao).sum(), (vne_gto).sum(), abs(vne_nao-vne_gto).sum()/vne_gto.size)
    self.assertTrue(abs(vne_nao-vne_gto).sum()/vne_gto.size<5e-6)

  def test_energy_nuc_gto_vs_nao(self):
    """ Test computation of matrix elements of nuclear-electron attraction """
    sv = system_vars_c().init_pyscf_gto(mol)
    e_nao = sv.energy_nuc()
    e_gto = mol.energy_nuc()
    self.assertAlmostEqual(e_nao, e_gto)
    
if __name__ == "__main__":
  print("Test of computation of Hartree potential, Fock exchange and Kinetic energy and the nuclear-electron attr integrals")
  unittest.main()
