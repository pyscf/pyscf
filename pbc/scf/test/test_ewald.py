import unittest
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc.scf import hf as pbchf

class KnowValues(unittest.TestCase):
    def test_ewald(self):
        cell = pbcgto.Cell()

        cell.unit = 'B'
        Lx = Ly = Lz = 5.
        cell.h = np.diag([Lx,Ly,Lz])
        cell.gs = np.array([20,20,20])
        cell.nimgs = [1,1,1]

        cell.atom.extend([['He', (2, 0.5*Ly, 0.5*Lz)],
                          ['He', (3, 0.5*Ly, 0.5*Lz)]])
        # these are some exponents which are not hard to integrate
        cell.basis = {'He': [[0, (1.0, 1.0)]]}

        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()

        ew_cut = (20,20,20)
        self.assertAlmostEqual(pbchf.ewald(cell, .05, ew_cut), -0.468640671931, 9)
        self.assertAlmostEqual(pbchf.ewald(cell, 0.1, ew_cut), -0.468640671931, 9)
        self.assertAlmostEqual(pbchf.ewald(cell, 0.2, ew_cut), -0.468640671931, 9)
        self.assertAlmostEqual(pbchf.ewald(cell, 1  , ew_cut), -0.468640671931, 9)

        def check(precision, eta_ref, ewald_ref):
            ew_eta0, ew_cut0 = cell.get_ewald_params(precision)
            self.assertAlmostEqual(ew_eta0, eta_ref)
            self.assertAlmostEqual(pbchf.ewald(cell, ew_eta0, ew_cut0), ewald_ref, 9)
        check(0.001, 4.78124933741, -0.469112631739)
        check(1e-05, 3.70353981157, -0.468642153932)
        check(1e-07, 3.1300624293 , -0.468640678042)
        check(1e-09, 2.76045559201, -0.468640671959)

if __name__ == '__main__':
    print("Full Tests for ewald")
    unittest.main()
