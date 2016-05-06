#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import grad
from pyscf.qmmm import itrf

mol = gto.M(
    verbose = 5,
    output = '/dev/null',
    atom = ''' H                 -0.00000000   -0.000    0.
 H                 -0.00000000   -0.000    1.
 H                 -0.00000000   -0.82    0.
 H                 -0.91000000   -0.020    0.''',
    basis = 'cc-pvdz')

class KnowValues(unittest.TestCase):
    def test_energy(self):
        coords = [(0.0,0.1,0.0)]
        charges = [1.00]
        mf = itrf.mm_charge(scf.RHF(mol), coords, charges)
        self.assertAlmostEqual(mf.kernel(), 2.0042702472400196, 9)

    def test_energy(self):
        coords = [(0.0,0.1,0.0)]
        charges = [1.00]
        mf = itrf.mm_charge(scf.RHF(mol), coords, charges).run()
        hfg = itrf.mm_charge_grad(grad.RHF(mf), coords, charges).run()
        self.assertAlmostEqual(numpy.linalg.norm(hfg.de), 29.462654284449247, 9)


if __name__ == "__main__":
    print("Full Tests for qmmm")
    unittest.main()

