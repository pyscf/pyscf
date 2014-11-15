#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import lib
from pyscf import dft

# for cgto
mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [[2, (0.,0.,0.)], ]
mol.basis = {"He": 'cc-pvdz'}
mol.grids = {"He": (50, 86),}
mol.build()
method = dft.RKS(mol)

class KnowValues(unittest.TestCase):
    def test_nr_lda(self):
        method.xc = 'lda, vwn_rpa'
        self.assertAlmostEqual(method.scf(), -2.8641551944479797, 9)

    def test_nr_pw91pw91(self):
        method.xc = 'pw91, pw91'
        self.assertAlmostEqual(method.scf(), -2.8914066891997789, 9)

    def test_nr_b88vwn(self):
        method.xc = 'b88, vwn'
        self.assertAlmostEqual(method.scf(), -2.9670729653770858, 9)

    def test_nr_xlyp(self):
        method.xc = 'xlyp'
        self.assertAlmostEqual(method.scf(), -2.904573834954614, 9)

    def test_nr_b3lyp(self):
        method.xc = 'b3lyp'
        self.assertAlmostEqual(method.scf(), -2.9070540971435919, 9)


if __name__ == "__main__":
    print("Full Tests for He")
    unittest.main()

