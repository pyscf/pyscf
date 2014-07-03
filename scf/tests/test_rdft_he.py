#
# File: test_rdft_he.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
from pyscf.import gto
from pyscf.import lib
from pyscf.scf.rdft import *

# for cgto
mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom.extend([[2, (0.,0.,0.)], ])
mol.basis = {"He": 'cc-pvdz'}
mol.grids = {"He": (50, 86),}
mol.build()

class KnowValues(unittest.TestCase):
    def test_unrestricted_dks_lda(self):
        with lib.quite_run():
            method = UKS(mol)
        method.xc_func(XC_LDA_X,XC_LDA_C_VWN_RPA)
        self.assertAlmostEqual(method.scf(), -2.8642731420433396, 9)

    def test_unrestricted_dks_b88p86(self):
        with lib.quite_run():
            method = UKS(mol)
        method.xc_func(XC_GGA_X_B88,XC_GGA_C_P86)
        self.assertAlmostEqual(method.scf(), -2.8982458551148484, 9)

    def test_unrestricted_dks_xlyp(self):
        with lib.quite_run():
            method = UKS(mol)
        method.xc_func(XC_GGA_XC_XLYP)
        self.assertAlmostEqual(method.scf(), -2.9046989955363243, 9)

    def test_unrestricted_dks_b3lyp(self):
        with lib.quite_run():
            method = UKS(mol)
        method.xc_func(XC_HYB_GGA_XC_B3LYP)
        self.assertAlmostEqual(method.scf(), -2.9071785673037955, 9)

    def test_restricted_dks_lda(self):
        with lib.quite_run():
            method = RKS(mol)
        method.xc_func(XC_LDA_X,XC_LDA_C_VWN_RPA)
        self.assertAlmostEqual(method.scf(), -2.8642731420157901, 9)


if __name__ == "__main__":
    print "Full Tests of relativistic DKS for H2O"
    unittest.main()

