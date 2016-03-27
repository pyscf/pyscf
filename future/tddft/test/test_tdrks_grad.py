#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
import unittest
import numpy
from pyscf import gto, dft
from pyscf import tddft
from pyscf.tddft import rks_grad

mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [
    ['H' , (0. , 0. , 1.804)],
    ['F' , (0. , 0. , 0.)], ]
mol.unit = 'B'
mol.basis = '631g'
mol.build()

def finger(a):
    w = numpy.cos(numpy.arange(len(a)))
    return numpy.dot(w, a)

class KnowValues(unittest.TestCase):
    def test_tda_lda(self):
        mf = dft.RKS(mol)
        mf.xc = 'LDA'
        mf.grids.prune = False
        mf.scf()
        td = tddft.TDA(mf).run(nstates=3)
        tdg = rks_grad.Gradients(td)
        g1 = tdg.kernel(state=2)
        self.assertAlmostEqual(g1[0,2], -9.23916667e-02, 9)

    def test_tda_b88(self):
        mf = dft.RKS(mol)
        mf.xc = 'b88'
        mf.grids.prune = False
        mf.scf()
        td = tddft.TDA(mf).run(nstates=3)
        tdg = rks_grad.Gradients(td)
        g1 = tdg.kernel(state=2)
        self.assertAlmostEqual(g1[0,2], -9.32506535e-02, 9)

    def test_tddft_lda(self):
        mf = dft.RKS(mol)
        mf.xc = 'LDA'
        mf.grids.prune = False
        mf.scf()
        td = tddft.TDDFT(mf).run(nstates=3)
        tdg = rks_grad.Gradients(td)
        g1 = tdg.kernel(state=2)
        self.assertAlmostEqual(g1[0,2], -1.31315477e-01, 9)

    def test_tddft_b3lyp(self):
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        mf.grids.prune = False
        mf.scf()
        td = tddft.TDDFT(mf).run(nstates=3)
        tdg = rks_grad.Gradients(td)
        g1 = tdg.kernel(state=2)
        self.assertAlmostEqual(g1[0,2], -1.55778110e-01, 9)


if __name__ == "__main__":
    print("Full Tests for TD-RKS gradients")
    unittest.main()

