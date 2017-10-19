from __future__ import print_function, division
import unittest
from pyscf import gto

mol = gto.M(
    verbose = 1,
    atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)

class KnowValues(unittest.TestCase):

  def test_ao_log_sp2ion(self):
    """ This is for initializing with SIESTA radial orbitals """
    from pyscf.nao.m_siesta_ion_xml import siesta_ion_xml
    from pyscf.nao import ao_log_c
    import os
    dname = os.path.dirname(os.path.abspath(__file__))
    sp2ion = []
    sp2ion.append(siesta_ion_xml(dname+'/H.ion.xml'))
    sp2ion.append(siesta_ion_xml(dname+'/O.ion.xml'))
    ao = ao_log_c().init_ao_log_ion(sp2ion, nr=512, rmin=0.0025)
    self.assertEqual(ao.nr, 512)
    self.assertAlmostEqual(ao.rr[0], 0.0025)
    self.assertAlmostEqual(ao.rr[-1], 11.105004591662)
    self.assertAlmostEqual(ao.pp[-1], 63.271890905445957)
    self.assertAlmostEqual(ao.pp[0], 0.014244003769469984)
    self.assertEqual(len(ao.sp2nmult), 2)
    self.assertEqual(len(ao.sp_mu2j[1]), 5)
    self.assertEqual(ao.sp2charge[0], 1)

  def test_ao_log_gto(self):
    """ This is indeed for initializing with auxiliary basis set"""
    from pyscf.nao import ao_log_c, system_vars_c
    sv = system_vars_c().init_pyscf_gto(mol)
    ao = ao_log_c().init_ao_log_gto_lm(gto=mol, nao=sv, lm=sv.ao_log)
    

if __name__ == "__main__":
  unittest.main()
