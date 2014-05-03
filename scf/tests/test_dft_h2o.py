#
# File: test_dft_h2o.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import gto
from scf.dft import *

h2o = gto.Mole()
h2o.verbose = 0
h2o.output = None#"out_h2o"
h2o.atom.extend([
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ])

h2o.grids = {"H": (50, 86),
             "O": (50, 86),}
h2o.basis = {"H": '6-31g',
             "O": '6-31g',}
h2o.build()


class KnowValues(unittest.TestCase):
    def test_nr_lda(self):
        method = RKS(h2o)
        method.xc_func(XC_LDA_X,XC_LDA_C_VWN_RPA)
        self.assertAlmostEqual(method.scf(), -76.013394314311597, 9)

    def test_nr_b88p86(self):
        method = RKS(h2o)
        method.xc_func(XC_GGA_X_B88,XC_GGA_C_P86)
        self.assertAlmostEqual(method.scf(), -76.385101833802537, 9)

    def test_nr_xlyp(self):
        method = RKS(h2o)
        method.xc_func(XC_GGA_XC_XLYP)
        self.assertAlmostEqual(method.scf(), -76.417563864361369, 9)

    def test_nr_b3lyp(self):
        method = RKS(h2o)
        method.xc_func(XC_HYB_GGA_XC_B3LYP)
        self.assertAlmostEqual(method.scf(), -76.384995400609824, 9)


if __name__ == "__main__":
    print "Full Tests for H2O"
    unittest.main()
