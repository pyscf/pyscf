from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import scf as nao_scf

mol = gto.M( verbose = 1, atom = '''Mn 0 0 0;''', basis = 'cc-pvdz', spin = 5, )
#gto_mf_rhf = scf.RHF(mol)
#gto_mf_rhf.kernel()
gto_mf_uhf = scf.UHF(mol)
gto_mf_uhf.kernel()
dm = gto_mf_uhf.make_rdm1()
#print(__name__, dm.shape, type(dm))
jg,kg = gto_mf_uhf.get_jk()
#print(jg.shape, kg.shape)
#print(__name__, (jg*dm).sum()/2.0)
#print(__name__, (kg*dm).sum()/2.0)
#print('vh0-vh1', (jg[0]-jg[1]).sum())

class KnowValues(unittest.TestCase):

  def test_mn_scf_0085(self):
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    """ Spin-resolved case redoing SCF procedure. """
    #print(__name__, dir(gto_mf_uhf))
    #print(set(dir(gto_mf_uhf))-set(dir(gto_mf_rhf)))
    scf = nao_scf(mf=gto_mf_uhf, gto=mol, verbosity=0)
    self.assertEqual(scf.nspin, 2)
    
    jn,kn = scf.get_jk()
    for d,dr in zip(jg.shape, jn.shape): self.assertEqual(d, dr)
    for d,dr in zip(kg.shape, kn.shape): self.assertEqual(d, dr)
    #print(__name__, jn.shape, kn.shape)

    dm_nao = scf.make_rdm1()
    Ehartree = (jn*dm_nao.reshape(scf.nspin,scf.norbs,scf.norbs)).sum()/2.0
    Ex = (kn*dm_nao.reshape(scf.nspin,scf.norbs,scf.norbs)).sum()/2.0
    self.assertAlmostEqual(Ehartree, 248.461304275)
    self.assertAlmostEqual(Ex, 50.9912877484)    
    ne_occ = fermi_dirac_occupations(scf.telec, scf.mo_energy, scf.fermi_energy).sum()
    self.assertAlmostEqual(ne_occ, 25.0)

    ## Do unrestricted Hartree-Fock SCF with numerical atomic orbitals
    e_tot = scf.kernel_scf()
    self.assertEqual(scf.nspin, 2)
    self.assertAlmostEqual(e_tot, -1149.86757123)

    ne_occ = fermi_dirac_occupations(scf.telec, scf.mo_energy, scf.fermi_energy).sum()
    self.assertAlmostEqual(ne_occ, 25.0)

    o,dm = scf.overlap_coo().toarray(), scf.make_rdm1()
    for nes, dms in zip(scf.nelec, dm[0,:,:,:,0]):
      #print(__name__, (dms*o).sum(), nes) 
      self.assertAlmostEqual((dms*o).sum(), nes, 4)
      
    #print(__name__, e_tot, gto_mf_uhf.e_tot)
    #print(scf.mo_occ)
    #print(scf.mo_energy)
    #print(scf.nelectron)
    #print(scf.mo_occ.sum())
    
if __name__ == "__main__": unittest.main()
