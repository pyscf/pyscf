#
# File: test_dft_he.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
import gto
import lib.jacobi
from scf.dft import *

# for cgto
mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom.extend([[2, (0.,0.,0.)], ])
mol.basis = {"He": 'cc-pvdz'}
mol.grids = {"He": (50, 86),}
mol.build()

class KnowValues(unittest.TestCase):
    def test_nr_lda(self):
        method = RKS(mol)
        method.xc_func(XC_LDA_X,XC_LDA_C_VWN_RPA)
        self.assertAlmostEqual(method.scf(), -2.8641551906405587, 9)

    def test_nr_b88p86(self):
        method = RKS(mol)
        method.xc_func(XC_GGA_X_B88,XC_GGA_C_P86)
        self.assertAlmostEqual(method.scf(), -2.8981220396936074, 9)

    def test_nr_xlyp(self):
        method = RKS(mol)
        method.xc_func(XC_GGA_XC_XLYP)
        self.assertAlmostEqual(method.scf(), -2.904573834954344, 9)

    def test_nr_b3lyp(self):
        method = RKS(mol)
        method.xc_func(XC_HYB_GGA_XC_B3LYP)
        self.assertAlmostEqual(method.scf(), -2.9070540971435919, 9)


if __name__ == "__main__":
    print "Full Tests for He"
    unittest.main()

