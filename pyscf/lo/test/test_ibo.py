#!/usr/bin/env python

import unittest
from functools import reduce
import numpy
from pyscf import gto
from pyscf import lib
from pyscf import scf
from pyscf.lo import iao, ibo, orth

mol = gto.Mole()
mol.atom = '''
     O    0.   0.       0
     h    0.   -0.757   0.587
     h    0.   0.757    0.587'''
mol.basis = 'unc-sto3g'
mol.verbose = 5
mol.output = '/dev/null'
mol.build()

class KnownValues(unittest.TestCase):
    def test_ibo(self):
        mf = scf.RHF(mol).run()
        b = ibo.ibo(mol, mf.mo_coeff[:,mf.mo_occ>0], exponent=4)
        s_b = reduce(numpy.dot, (b.T, mf.get_ovlp(), b))
        self.assertAlmostEqual(lib.finger(b), -0.059680435404993903, 5)
        self.assertTrue(abs(s_b.diagonal() - 1).max() < 1e-9)

        b = ibo.ibo(mol, mf.mo_coeff[:,mf.mo_occ>0], exponent=2)
        s_b = reduce(numpy.dot, (b.T, mf.get_ovlp(), b))
        self.assertAlmostEqual(lib.finger(b), -0.47003453325391631, 5)
        self.assertTrue(abs(s_b.diagonal() - 1).max() < 1e-9)


if __name__ == "__main__":
    print("TODO: Test ibo")
    unittest.main()



