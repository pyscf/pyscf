#!/usr/bin/env python

import unittest
from pyscf import gto
from pyscf import lib
from pyscf import dft

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


class KnowValues(unittest.TestCase):
    def test_nr_lda(self):
        method = dft.RKS(h2o)
        method.prune_scheme = dft.gen_grid.treutler_prune
        method.xc = 'lda, vwn_rpa'
        self.assertAlmostEqual(method.scf(), -76.01330948329084, 9)

    def test_nr_pw91pw91(self):
        method = dft.RKS(h2o)
        method.prune_scheme = dft.gen_grid.treutler_prune
        method.xc = 'pw91, pw91'
        self.assertAlmostEqual(method.scf(), -76.355310330095563, 9)

    def test_nr_b88vwn(self):
        method = dft.RKS(h2o)
        method.prune_scheme = dft.gen_grid.treutler_prune
        method.xc = 'b88, vwn'
        self.assertAlmostEqual(method.scf(), -76.690247578608236, 9)

    def test_nr_xlyp(self):
        method = dft.RKS(h2o)
        method.prune_scheme = dft.gen_grid.treutler_prune
        method.xc = 'xlyp'
        self.assertAlmostEqual(method.scf(), -76.4174879445209, 9)

    def test_nr_b3lyp(self):
        method = dft.RKS(h2o)
        method.prune_scheme = dft.gen_grid.treutler_prune
        method.xc = 'b3lyp'
        self.assertAlmostEqual(method.scf(), -76.384928891413438, 9)

    def test_nr_b3lyp_direct(self):
        method = dft.RKS(h2o)
        method.prune_scheme = dft.gen_grid.treutler_prune
        method.radi_method = dft.radi.gauss_chebyshev
        method.xc = 'b3lyp'
        method.max_memory = 0
        method.direct_scf = True
        self.assertAlmostEqual(method.scf(), -76.384928891413438, 9)
        method.direct_scf = False
        self.assertAlmostEqual(method.scf(), -76.384928891413438, 9)


if __name__ == "__main__":
    print("Full Tests for H2O")
    unittest.main()
