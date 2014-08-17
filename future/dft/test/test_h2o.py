#!/usr/bin/env python

import unittest
from pyscf import gto
from pyscf import lib
from pyscf.future import dft

h2o = gto.Mole()
h2o.verbose = 0
h2o.output = None#"out_h2o"
h2o.atom.extend([
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ])

h2o.grids = {"H": (50, 194),
             "O": (50, 194),}
h2o.basis = {"H": '6-31g',
             "O": '6-31g',}
h2o.build()
method = dft.RKS(h2o)


class KnowValues(unittest.TestCase):
    def test_nr_lda(self):
        method.xc = 'lda, vwn_rpa'
        self.assertAlmostEqual(method.scf(), -76.013333366968084, 9)

    def test_nr_pw91pw91(self):
        method.xc = 'pw91, pw91'
        self.assertAlmostEqual(method.scf(), -76.35533525774612, 9)

    def test_nr_b88vwn(self):
        method.xc = 'b88, vwn'
        self.assertAlmostEqual(method.scf(), -76.690272093988938, 9)

    def test_nr_xlyp(self):
        method.xc = 'xlyp'
        self.assertAlmostEqual(method.scf(), -76.417511868433252, 9)

    def test_nr_b3lyp(self):
        method.xc = 'b3lyp'
        self.assertAlmostEqual(method.scf(), -76.384948371890346, 9)


if __name__ == "__main__":
    print "Full Tests for H2O"
    unittest.main()
