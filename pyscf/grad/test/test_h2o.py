#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto, scf, dft
from pyscf import grad
from pyscf.grad import rks as rks_grad

h2o = gto.Mole()
h2o.verbose = 5
h2o.output = '/dev/null'
h2o.atom.extend([
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ])

h2o.basis = {"H": '6-31g',
             "O": '6-31g',}
h2o.build()


def finger(mat):
    return abs(mat).sum()

class KnownValues(unittest.TestCase):
    def test_nr_rhf(self):
        rhf = scf.RHF(h2o)
        rhf.conv_tol = 1e-14
        rhf.kernel()
        g = grad.RHF(rhf)
        self.assertAlmostEqual(finger(g.grad_elec()), 10.126405944938071, 7)

    def test_r_uhf(self):
        uhf = scf.dhf.UHF(h2o)
        uhf.conv_tol_grad = 1e-6
        uhf.kernel()
        g = grad.DHF(uhf)
        self.assertAlmostEqual(finger(g.grad_elec()), 10.126445612578864, 7)

    def test_nr_uhf(self):
        uhf = scf.UHF(h2o)
        uhf.conv_tol = 1e-14
        uhf.kernel()
        g = grad.UHF(uhf)
        self.assertAlmostEqual(finger(g.grad_elec()), 10.126405944938071, 7)

    def test_energy_nuc(self):
        rhf = scf.RHF(h2o)
        g = grad.RHF(rhf)
        self.assertAlmostEqual(finger(g.grad_nuc()), 10.086972893020102, 9)

    def test_ccsd(self):
        from pyscf import cc
        rhf = scf.RHF(h2o)
        rhf.kernel()
        mycc = cc.CCSD(rhf)
        mycc.kernel()
        mycc.solve_lambda()
        g1 = grad.ccsd.kernel(mycc)
        self.assertAlmostEqual(finger(g1), 0.065802850540912422, 8)

    def test_rks_lda(self):
        mf = dft.RKS(h2o)
        mf.grids.prune = None
        mf.run(conv_tol=1e-15, xc='lda,vwn')
        g = rks_grad.Grad(mf)
        g1 = g.grad()
        self.assertAlmostEqual(finger(g1), 0.098438461959390822, 7)

    def test_rks_bp86(self):
        mf = dft.RKS(h2o)
        mf.grids.prune = None
        mf.run(conv_tol=1e-15, xc='b88,p86')
        g = rks_grad.Grad(mf)
        g1 = g.grad()
        self.assertAlmostEqual(finger(g1), 0.10362532283229957, 7)

    def test_rks_b3lypg(self):
        mf = dft.RKS(h2o)
        mf.grids.prune = None
        mf.run(conv_tol=1e-15, xc='b3lypg')
        g = rks_grad.Grad(mf)
        g1 = g.grad()
        self.assertAlmostEqual(finger(g1), 0.066541921001296467, 7)


if __name__ == "__main__":
    print("Full Tests for H2O")
    unittest.main()

