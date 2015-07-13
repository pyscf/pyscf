#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
import scipy.linalg
import tempfile
from pyscf import gto
from pyscf import scf
from pyscf import fci

class KnowValues(unittest.TestCase):
    def test_davidson(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [['H', (0,0,i)] for i in range(7)]
        mol.basis = {'H': 'sto-3g'}
        mol.spin = 1
        mol.build()
        mf = scf.RHF(mol)
        mf.scf()
        myfci = fci.FCI(mol, mf.mo_coeff)
        myfci.max_memory = .001
        myfci.max_cycle = 400
        e = myfci.kernel()[0]
        self.assertAlmostEqual(e, -7.0560297718867258, 9)

if __name__ == "__main__":
    print("Full Tests for linalg_helper")
    unittest.main()
