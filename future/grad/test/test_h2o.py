#!/usr/bin/env python

import unittest
from pyscf import scf
from pyscf import gto
from pyscf import grad

h2o = gto.Mole()
h2o.verbose = 0
h2o.output = None#"out_h2o"
h2o.atom.extend([
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ])

h2o.basis = {"H": '6-31g',
             "O": '6-31g',}
h2o.build()


def finger(mat):
    return abs(mat).sum()

class KnowValues(unittest.TestCase):
    def test_nr_rhf(self):
        rhf = scf.RHF(h2o)
        rhf.scf()
        g = grad.hf.RHF(rhf)
        self.assertAlmostEqual(finger(g.grad_elec()), 10.126405944938071, 7)

    def test_r_uhf(self):
        uhf = scf.dhf.UHF(h2o)
        uhf.scf()
        g = grad.dhf.UHF(uhf)
        self.assertAlmostEqual(finger(g.grad_elec()), 10.126445561598123, 7)

#    def test_nr_uhf(self):
#        uhf = scf.UHF(h2o)
#        self.assertAlmostEqual(uhf.scf(g.grad_elec()), 0, 9)

    def test_energy_nuc(self):
        rhf = scf.RHF(h2o)
        rhf.scf()
        g = grad.hf.RHF(rhf)
        self.assertAlmostEqual(finger(g.grad_nuc()), 10.086972893020102, 9)


if __name__ == "__main__":
    print("Full Tests for H2O")
    unittest.main()

