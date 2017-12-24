from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import scf as nao_scf

mol = gto.M( verbose = 1, atom = '''Mn 0 0 0;''', basis = 'cc-pvdz', spin = 5, )
#gto_mf_rhf = scf.RHF(mol)
#gto_mf_rhf.kernel()
gto_mf_uhf = scf.UHF(mol)
gto_mf_uhf.kernel()

class KnowValues(unittest.TestCase):

  def test_mn_uhf_0084(self):
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    """ Spin-resolved case """
    #print(__name__, dir(gto_mf_uhf))
    #print(set(dir(gto_mf_uhf))-set(dir(gto_mf_rhf)))
    scf = nao_scf(mf=gto_mf_uhf, gto=mol, verbosity=1)
    self.assertEqual(scf.nspin, 2)
    ne_occ = fermi_dirac_occupations(scf.telec, scf.mo_energy, scf.fermi_energy).sum()
    self.assertAlmostEqual(ne_occ, 25.0)

    #e_tot = scf.kernel_scf()

    self.assertEqual(scf.nspin, 2)
    ne_occ = fermi_dirac_occupations(scf.telec, scf.mo_energy, scf.fermi_energy).sum()
    self.assertAlmostEqual(ne_occ, 25.0)

    #o = scf.overlap_coo().toarray()
    #dm = scf.make_rdm1()
    
    #print(__name__, e_tot, gto_mf_uhf.e_tot)
    print(scf.mo_occ)
    print(scf.mo_energy)
    print(scf.nelectron)
    print(scf.mo_occ.sum())
    
if __name__ == "__main__": unittest.main()
