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
mol.build()
method = dft.RKS(mol)

class KnowValues(unittest.TestCase):
    def test_nr_lda(self):
        method.xc = 'lda, vwn_rpa'
        self.assertAlmostEqual(method.scf(), -2.8641551904776055, 9)

    def test_nr_pw91pw91(self):
        method.xc = 'pw91, pw91'
        self.assertAlmostEqual(method.scf(), -2.8914066724838849, 9)

    def test_nr_b88vwn(self):
        method.xc = 'b88, vwn'
        self.assertAlmostEqual(method.scf(), -2.9670729652962606, 9)

    def test_nr_xlyp(self):
        method.xc = 'xlyp'
        self.assertAlmostEqual(method.scf(), -2.9045738259332161, 9)

    def test_nr_b3lyp(self):
        method.xc = 'b3lyp'
        self.assertAlmostEqual(method.scf(), -2.9070540942168002, 9)


if __name__ == "__main__":
    print("Full Tests for He")
    unittest.main()

