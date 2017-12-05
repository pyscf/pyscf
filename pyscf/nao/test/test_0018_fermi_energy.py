from __future__ import print_function, division
import os,unittest,numpy as np
from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
from pyscf.nao.m_fermi_energy import fermi_energy as get_fermi_energy

class KnowValues(unittest.TestCase):
  
  def test_fermi_energy_spin_saturated(self):
    """ This is to test the determination of Fermi level"""
    ee = np.arange(-10.13, 100.0, 0.1)
    #print('0: ', ee.shape)
    nelec = 5.0
    telec = 0.01
    fermi_energy = get_fermi_energy(ee, nelec, telec)
    occ = 2.0*fermi_dirac_occupations(telec, ee, fermi_energy)
    self.assertAlmostEqual(occ.sum(), 5.0)
    self.assertAlmostEqual(fermi_energy, -9.93)
    
    #print(occ)
    #print(occ.sum())
    #print(fermi_energy)

  def test_fermi_energy_spin_resolved_spin1(self):
    """ This is to test the determination of Fermi level"""
    ee = np.linspace(-10.13, 99.97, 1102).reshape((1,1102))
    #print('1: ', ee.shape)
    nelec = 5.0
    telec = 0.01
    fermi_energy = get_fermi_energy(ee, nelec, telec)
    occ = 2.0*fermi_dirac_occupations(telec, ee, fermi_energy)
    self.assertAlmostEqual(occ.sum(), 5.0)
    self.assertAlmostEqual(fermi_energy, -9.93)
    
    #print(occ)
    #print(occ.sum())
    #print(fermi_energy)

  def test_fermi_energy_spin_resolved(self):
    """ This is to test the determination of Fermi level in spin-resolved case"""
    ee = np.row_stack((np.linspace(-10.3, 100.0, 1003), np.linspace(-10.0, 100.0, 1003)))
    nelec = 11.0
    telec = 0.02
    #print(ee)
    fermi_energy = get_fermi_energy(ee, nelec, telec)
    occ = fermi_dirac_occupations(telec, ee, fermi_energy)

    self.assertAlmostEqual(occ.sum(), 11.0)
    self.assertAlmostEqual(fermi_energy, -9.60016955367)

    #print(occ)
    #print(occ.sum())
    #print(fermi_energy)

  def test_fermi_energy_spin_resolved_even(self):
    """ This is to test the determination of Fermi level in spin-resolved case"""
    ee = np.row_stack((np.linspace(-10.3, 100.0, 1003), np.linspace(-10.0, 100.0, 1003)))
    nelec = 20.0
    telec = 0.02
    #print(ee)
    fermi_energy = get_fermi_energy(ee, nelec, telec)
    occ = fermi_dirac_occupations(telec, ee, fermi_energy)

    self.assertAlmostEqual(occ.sum(), 20.0)
    self.assertAlmostEqual(fermi_energy, -9.10544404859)

    #print(occ)
    #print(occ.sum())
    #print(fermi_energy)

  def test_fermi_energy_spin_resolved_even_kpoints(self):
    """ This is to test the determination of Fermi level in spin-resolved case"""
    ee = np.row_stack((np.linspace(-10.1, 100.0, 1003), 
                       np.linspace(-10.2, 100.0, 1003),
                       np.linspace(-10.3, 100.0, 1003),
                       np.linspace(-10.4, 100.0, 1003))).reshape((4,1,1003))
    nelec = 20.0
    telec = 0.02
    nkpts = ee.shape[0]
    nspin = ee.shape[-2]
    #print(ee)
    fermi_energy = get_fermi_energy(ee, nelec, telec)
    occ = (3.0-nspin)*fermi_dirac_occupations(telec, ee, fermi_energy)

    #print(occ)
    #print(occ.sum()/nkpts)
    #print(fermi_energy)

    self.assertAlmostEqual(occ.sum()/nkpts, 20.0)
    self.assertAlmostEqual(fermi_energy, -9.2045998319213016)

  def test_fermi_energy_spin_resolved_even_kpoints_spin2(self):
    """ This is to test the determination of Fermi level in spin-resolved case"""
    ee = np.row_stack((np.linspace(-10.1, 100.0, 1003), 
                       np.linspace(-10.2, 100.0, 1003),
                       np.linspace(-10.3, 100.0, 1003),
                       np.linspace(-10.4, 100.0, 1003))).reshape((2,2,1003))
    nelec = 20.0
    telec = 0.02
    nkpts = ee.shape[0]
    nspin = ee.shape[-2]
    #print(ee)
    fermi_energy = get_fermi_energy(ee, nelec, telec)
    occ = (3.0-nspin)*fermi_dirac_occupations(telec, ee, fermi_energy)

    #print(occ)
    #print(occ.sum()/nkpts)
    #print(fermi_energy)

    self.assertAlmostEqual(occ.sum()/nkpts, 20.0)
    self.assertAlmostEqual(fermi_energy, -9.2045998319213016)

    
if __name__ == "__main__" : unittest.main()
