from __future__ import print_function, division
import os,unittest,numpy as np
from pyscf.nao.m_fermi_dirac import fermi_dirac, fermi_dirac_occupations

class KnowValues(unittest.TestCase):
  
  def test_fermi_dirac(self):
    """ This is to test the Fermi-Dirac occupations """
    ksn2e = np.zeros([1,1,100], dtype=np.float32)
    ksn2e[0,0,:] = np.linspace(-1000.0, 100.0, 100)
    fermi_energy = -100.0
    telec = 0.1
    ksn2f = fermi_dirac_occupations(telec, ksn2e, fermi_energy)
    self.assertAlmostEqual(ksn2f.sum(), 81.5)
    
if __name__ == "__main__":
  unittest.main()
