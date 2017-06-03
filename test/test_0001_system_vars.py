from __future__ import print_function, division
#
# Author: 
#

import unittest
import numpy
import scipy.linalg
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
  def test_gto2nao(self):
    sv = system_vars_c(gto=mol)
    print(sv.natoms)
    self.assertEqual(len(sv.ao_log.rr), 1024)

if __name__ == "__main__":
    print("Full Tests for df")
    unittest.main()

