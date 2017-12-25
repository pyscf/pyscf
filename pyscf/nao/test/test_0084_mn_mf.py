from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import mf as nao_mf

mol = gto.M( verbose = 1, atom = '''Mn 0 0 0;''', basis = 'cc-pvdz', spin = 5, )
#gto_mf_rhf = scf.RHF(mol)
#gto_mf_rhf.kernel()
gto_mf_uhf = scf.UHF(mol)
gto_mf_uhf.kernel()

class KnowValues(unittest.TestCase):

  def test_mn_mean_field_0084(self):
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    """ Spin-resolved case """
    #print(__name__, dir(gto_mf_uhf))
    #print(set(dir(gto_mf_uhf))-set(dir(gto_mf_rhf)))
    mf = nao_mf(mf=gto_mf_uhf, gto=mol, verbosity=0)
    self.assertEqual(mf.nspin, 2)
    ne_occ = fermi_dirac_occupations(mf.telec, mf.mo_energy, mf.fermi_energy).sum()
    self.assertAlmostEqual(ne_occ, 25.0)
    o = mf.overlap_coo().toarray()
    dm = mf.make_rdm1()
    
    #print((dm[0,0,:,:,0]*o).sum())
    #print((dm[0,1,:,:,0]*o).sum())
    #mf.diag_check()
    
    #dos = mf.dos(np.arange(-1.4, 1.0, 0.01)+1j*0.02)
    #print(mf.norbs)
    #print(mf.nspin)
    #print(mf.fermi_energy)
    #print(mf.mo_occ)
    #print(mf.mo_energy)
    #print(mf.nelectron)
    #print((mf.mo_occ).sum())
    
if __name__ == "__main__": unittest.main()
