# Author: Peter Koval

from __future__ import print_function, division
import unittest
from pyscf import gto
from pyscf.nao import log_mesh_c

mol = gto.M(
    verbose = 1,
    atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)

class KnowValues(unittest.TestCase):

  def test_log_mesh_gto(self):
    """ Test construction  of log mesh for GTOs"""
    lm = log_mesh_c().init_log_mesh_gto(mol, rmin=1e-6)
    self.assertEqual(lm.nr, 1024)
    self.assertAlmostEqual(lm.rr[0], 1e-6)
    self.assertAlmostEqual(lm.rr[-1], 11.494152344675497)
    self.assertAlmostEqual(lm.pp[-1], 644.74911990708938)
    self.assertAlmostEqual(lm.pp[0], 5.6093664027844639e-05)

  def test_log_mesh(self):
    """ Test construction of log mesh with predefined grids"""
    from pyscf.nao.m_log_mesh import log_mesh
    rr,pp=log_mesh(1024, 1e-3, 15.0)
    lm = log_mesh_c().init_log_mesh(rr,pp)
    self.assertEqual(lm.nr, 1024)
    self.assertAlmostEqual(lm.rr[0], 1e-3)
    self.assertAlmostEqual(lm.rr[-1], 15.0)
    self.assertAlmostEqual(lm.pp[-1], 318.3098861837907)
    self.assertAlmostEqual(lm.pp[0], 0.021220659078919384)
  
  def test_log_mesh_ion(self):
    from pyscf.nao.m_siesta_ion_xml import siesta_ion_xml
    #sp2ion = []
    #sp2ion.append(siesta_ion_xml('O.ion.xml'))
    #sp2ion.append(siesta_ion_xml('H.ion.xml'))
    
if __name__ == "__main__":
  unittest.main()

