import unittest
import numpy as np
from pyscf import lib
from pyscf.pbc import gto, scf
from pyscf.pbc.ci import cisd

def setUpModule():
    global cell, kmf, kci, eris
    cell = gto.Cell()
    cell.a = np.eye(3) * 2.5
    cell.mesh = [11] * 3
    cell.atom = '''He    0.    2.       1.5
                   He    1.    1.       1.'''
    cell.basis = {'He': [(0, (1.5, 1)), (0, (1., 1))]}
    cell.build()

def tearDownModule():
    global cell
    del cell

class KnownValues(unittest.TestCase):
    def test_cisd(self):
        mf = scf.RHF(cell).run()
        myci = mf.CISD().run()
        self.assertTrue(isinstance(myci, cisd.RCISD))
        self.assertAlmostEqual(myci.e_corr, -0.0107155147353, 9)

        umf = mf.to_uhf()
        myci = umf.CISD().run()
        self.assertTrue(isinstance(myci, cisd.UCISD))
        self.assertAlmostEqual(myci.e_corr, -0.0107155147353, 9)

        gmf = mf.to_ghf()
        myci = gmf.CISD().run()
        self.assertTrue(isinstance(myci, cisd.GCISD))
        self.assertAlmostEqual(myci.e_corr, -0.0107155147353, 9)


if __name__ == "__main__":
    print("Full Tests for gamma-point CISD")
    unittest.main()
