#!/usr/bin/env python

import unittest
from functools import reduce
import numpy
from pyscf import gto
from pyscf import scf
from pyscf.lo import nao

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
mf.scf()

class KnowValues(unittest.TestCase):
    def test_pre_nao(self):
        c = nao.prenao(mol, mf.make_rdm1())
        self.assertAlmostEqual(numpy.linalg.norm(c), 7.2617698799320358, 9)
        self.assertAlmostEqual(abs(c).sum(), 40.032576928168687, 8)

    def test_nao(self):
        c = nao.nao(mol, mf)
        s = mf.get_ovlp()
        self.assertTrue(numpy.allclose(reduce(numpy.dot, (c.T, s, c)),
                                       numpy.eye(s.shape[0])))
        self.assertAlmostEqual(numpy.linalg.norm(c), 10.967144073462256, 9)
        self.assertAlmostEqual(abs(c).sum(), 110.03099712555559, 7)


if __name__ == "__main__":
    print("Test orth")
    unittest.main()


