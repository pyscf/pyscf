#
# Author: Peter Koval
#

from __future__ import print_function, division
import unittest
import numpy as np
from pyscf import gto
from pyscf.nao import system_vars_c

mol = gto.M(
    verbose = 1,
    atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)

class KnowValues(unittest.TestCase):

  def test_gto2sv(self):
    """ Test transformation of the radial orbitals from GTO to NAO type"""
    sv = system_vars_c(gto=mol)
    self.assertEqual(sv.natoms, 3)
    self.assertEqual(sv.norbs, 24)
    self.assertEqual(len(sv.ao_log.psi_log), 2)
    rr = sv.ao_log.rr
    self.assertEqual(len(rr), 1024)
    dr = np.log(rr[1]/rr[0])
    for mu2ff in sv.ao_log.psi_log:
      for ff in mu2ff:
        norm = (ff**2*sv.ao_log.rr**3).sum()*dr
        self.assertAlmostEqual(norm, 1.0)
    
  def test_overlap_gto_vs_nao(self):
    """ Test computation of overlaps computed between NAOs against overlaps computed between GTOs"""
    from pyscf.nao import conv_yzx2xyz_c, overlap_am
    sv = system_vars_c(gto=mol)
    oref = conv_yzx2xyz_c(mol).conv_yzx2xyz_2d(mol.intor_symmetric('cint1e_ovlp_sph'), direction='pyscf2nao')
    over = sv.comp_overlap_coo(funct=overlap_am).todense()
    self.assertTrue(abs(over-oref).sum()<5e-9)


if __name__ == "__main__":
  print("Full Tests for system_vars_c")
  unittest.main()

