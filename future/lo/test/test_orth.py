#!/usr/bin/env python

import unittest
from functools import reduce
import numpy
from pyscf import gto
from pyscf import scf
from pyscf.lo import orth

mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = '''
     O    0.   0.       0
     1    0.   -0.757   0.587
     1    0.   0.757    0.587'''

mol.basis = 'cc-pvdz'
mol.build()
mf = scf.RHF(mol)

class KnowValues(unittest.TestCase):
    def test_orth(self):
        numpy.random.seed(10)
        n = 100
        a = numpy.random.random((n,n))
        s = numpy.dot(a.T, a)
        c = orth.lowdin(s)
        self.assertTrue(numpy.allclose(reduce(numpy.dot, (c.T, s, c)),
                                       numpy.eye(n)))
        x1 = numpy.dot(a, c)
        x2 = orth.vec_lowdin(a)
        d = numpy.dot(x1.T,x2)
        d[numpy.diag_indices(n)] = 0
        self.assertAlmostEqual(numpy.linalg.norm(d), 0, 9)
        self.assertAlmostEqual(numpy.linalg.norm(c), 36.56738258719514, 9)
        self.assertAlmostEqual(abs(c).sum(), 2655.5580057303964, 7)

    def test_schmidt(self):
        numpy.random.seed(10)
        n = 100
        a = numpy.random.random((n,n))
        s = numpy.dot(a.T, a)
        c = orth.schmidt(s)
        self.assertTrue(numpy.allclose(reduce(numpy.dot, (c.T, s, c)),
                                       numpy.eye(n)))
        x1 = numpy.dot(a, c)
        x2 = orth.vec_schmidt(a)
        d = numpy.dot(x1.T,x2)
        d[numpy.diag_indices(n)] = 0
        self.assertAlmostEqual(numpy.linalg.norm(d), 0, 9)
        self.assertAlmostEqual(numpy.linalg.norm(c), 36.56738258719514, 9)
        self.assertAlmostEqual(abs(c).sum(), 1123.2089785000373, 7)

    def test_weight_orth(self):
        numpy.random.seed(10)
        n = 100
        a = numpy.random.random((n,n))
        s = numpy.dot(a.T, a)
        weight = numpy.random.random(n)
        c = orth.weight_orth(s, weight)
        self.assertTrue(numpy.allclose(reduce(numpy.dot, (c.T, s, c)),
                                       numpy.eye(n)))
        self.assertAlmostEqual(numpy.linalg.norm(c), 36.56738258719514, 9)
        self.assertAlmostEqual(abs(c).sum(), 1908.8535852660757, 7)

    def test_orth_ao(self):
        c0 = orth.pre_orth_ao(mol, method='scf')
        self.assertAlmostEqual(numpy.linalg.norm(c0), 5.7742626195362039, 9)
        self.assertAlmostEqual(abs(c0).sum(), 33.461490657433551, 8)
        c = orth.orth_ao(mol, 'lowdin', c0)
        self.assertAlmostEqual(numpy.linalg.norm(c), 8.9823854843222257, 9)
        self.assertAlmostEqual(abs(c).sum(), 94.933979307106767, 8)
        c = orth.orth_ao(mol, 'meta_lowdin', c0)
        self.assertAlmostEqual(numpy.linalg.norm(c), 8.9823854843222257, 9)
        self.assertAlmostEqual(abs(c).sum(), 93.029386338534394, 8)

    def test_ghost_atm_meta_lowdin(self):
        mol = gto.Mole()
        mol.atom = [["O" , (0. , 0.     , 0.)],
                    ['ghost'   , (0. , -0.757, 0.587)],
                    [1   , (0. , 0.757 , 0.587)] ]
        mol.spin = 1
        mol.basis = {'O':'ccpvdz', 'H':'ccpvdz',
                     'GHOST': gto.basis.load('631g','H')}
        mol.build()
        c = orth.orth_ao(mol, method='meta_lowdin')
        self.assertAlmostEqual(numpy.linalg.norm(c), 7.9067188905237256, 9)


if __name__ == "__main__":
    print("Test orth")
    unittest.main()


