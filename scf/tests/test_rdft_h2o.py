#
# File: test_rdft_h2o.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
from pyscf import gto
from pyscf import lib
from pyscf.scf.rdft import *

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
h2o.build_moleinfo()


class KnowValues(unittest.TestCase):
    def test_unrestricted_dks_lda(self):
        with lib.quite_run():
            method = UKS(h2o)
        method.xc_func(XC_LDA_X,XC_LDA_C_VWN_RPA)
        self.assertAlmostEqual(method.scf(), -76.068061721154891, 8)

    def test_unrestricted_dks_b88p86(self):
        with lib.quite_run():
            method = UKS(h2o)
        method.xc_func(XC_GGA_X_B88,XC_GGA_C_P86)
        self.assertAlmostEqual(method.scf(), -76.439843608758949, 8)

    def test_unrestricted_dks_xlyp(self):
        with lib.quite_run():
            method = UKS(h2o)
        method.xc_func(XC_GGA_XC_XLYP)
        self.assertAlmostEqual(method.scf(), -76.47234258410711, 8)

    def test_unrestricted_dks_b3lyp(self):
        with lib.quite_run():
            method = UKS(h2o)
        method.xc_func(XC_HYB_GGA_XC_B3LYP)
        self.assertAlmostEqual(method.scf(), -76.439721041419972, 8)

    def test_restricted_dks_lda(self):
        with lib.quite_run():
            method = RKS(h2o)
        method.xc_func(XC_LDA_X,XC_LDA_C_VWN_RPA)
        self.assertAlmostEqual(method.scf(), -76.068061720166284, 8)



if __name__ == "__main__":
    print "Full Tests of relativistic DKS for H2O"
    unittest.main()
