#!/usr/bin/env python

import unittest
from functools import reduce
import numpy
from pyscf import gto
from pyscf import lib
from pyscf import scf
from pyscf.lo import iao, ibo, orth, pipek

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
        self.assertTrue(abs(s_b.diagonal() - 1).max() < 1e-9)
        pop = pipek.atomic_pops(mol, b)
        z = numpy.einsum('xii,xii->', pop, pop)
        self.assertAlmostEqual(z, 4.0663610846219127, 6)
        self.assertAlmostEqual(lib.finger(b), 0.50200322891732885, 6)

        b = ibo.ibo(mol, mf.mo_coeff[:,mf.mo_occ>0], exponent=2)
        s_b = reduce(numpy.dot, (b.T, mf.get_ovlp(), b))
        self.assertTrue(abs(s_b.diagonal() - 1).max() < 1e-9)
        pop = pipek.atomic_pops(mol, b)
        z = numpy.einsum('xii,xii->', pop, pop)
        self.assertAlmostEqual(z, 4.0663609732471571, 6)
        self.assertAlmostEqual(lib.finger(b), 0.50200217429285976, 6)

    def test_ibo_PM(self):
        mf = scf.RHF(mol).run()
        b = ibo.PM(mol, mf.mo_coeff[:,mf.mo_occ>0], exponent=4).kernel()
        pop = pipek.atomic_pops(mol, b)
        z = numpy.einsum('xii,xii->', pop, pop)
        self.assertAlmostEqual(z, 3.9206879872618576, 6)
        self.assertAlmostEqual(lib.finger(b), -1.5634357606843325, 6)

        b = ibo.PM(mol, mf.mo_coeff[:,mf.mo_occ>0], exponent=2).kernel()
        pop = pipek.atomic_pops(mol, b)
        z = numpy.einsum('xii,xii->', pop, pop)
        self.assertAlmostEqual(z, 3.9206882147236133, 6)
        self.assertAlmostEqual(lib.finger(b), -1.5634350790965224, 6)


if __name__ == "__main__":
    print("Full tests for ibo")
    unittest.main()



