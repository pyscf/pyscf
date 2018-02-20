#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import lib


mol = gto.M(atom='''
H 0. 0. 0.
H 8. 0. 0.
''', basis='ccpvqz')
numpy.random.seed(1)
r = numpy.random.random((100,3)) * 2

class KnownValues(unittest.TestCase):
    def test_cart(self):
        val = mol.eval_gto('GTOval', r)
        self.assertEqual(val.shape, (100,60))
        self.assertAlmostEqual(lib.finger(val), -3.0283379087553808, 9)

        val = mol.eval_gto('GTOval_spinor', r)
        self.assertEqual(val.shape, (2,100,120))
        self.assertTrue(val.dtype == numpy.complex)
        self.assertAlmostEqual(lib.finger(val), -2.029604328041736+0.80571570114202995j, 9)

        val = mol.eval_gto('GTOval_ip', r)
        self.assertEqual(val.shape, (3,100,60))
        self.assertAlmostEqual(lib.finger(val), -14.526634330008513, 9)

        val = mol.eval_gto('GTOval_ig', r)
        self.assertEqual(val.shape, (3,100,60))
        self.assertAlmostEqual(lib.finger(val), -9.8708097471236808e-08, 9)

        val = mol.eval_gto('GTOval_sph_deriv3', r)
        self.assertEqual(val.shape, (20,100,60))
        self.assertAlmostEqual(lib.finger(val), 207.32371064250577, 9)

        val = mol.eval_gto('GTOval_cart_deriv3', r)
        self.assertEqual(val.shape, (20,100,70))
        self.assertAlmostEqual(lib.finger(val), -93.310347533061787, 9)

        val = mol.eval_gto('GTOval_spinor_deriv3', r)
        self.assertEqual(val.shape, (2,20,100,120))
        self.assertTrue(val.dtype == numpy.complex)
        self.assertAlmostEqual(lib.finger(val), -2470.5090409822333+64.324145522087917j, 8)

        mol1 = mol.copy()
        mol1.cart = True
        val = mol1.eval_gto('GTOval_ip', r)
        self.assertEqual(val.shape, (3,100,70))
        self.assertAlmostEqual(lib.finger(val), -15.411590075004403, 9)

if __name__ == '__main__':
    print("Full Tests for eval_gto")
    unittest.main()

