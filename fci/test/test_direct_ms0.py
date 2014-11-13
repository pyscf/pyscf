#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci

mol = gto.Mole()
mol.verbose = 0
mol.output = None#"out_h2o"
mol.atom = [
    ['H', ( 1.,-1.    , 0.   )],
    ['H', ( 0.,-1.    ,-1.   )],
    ['H', ( 0.,-0.5   ,-0.   )],
    ['H', ( 0.,-0.    ,-1.   )],
    ['H', ( 1.,-0.5   , 0.   )],
    ['H', ( 0., 1.    , 1.   )],
]

mol.basis = {'H': '6-31g'}
mol.build()

m = scf.RHF(mol)
ehf = m.scf()

norb = m.mo_coeff.shape[1]
nelec = mol.nelectron
h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
g2e = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)
na = fci.cistring.num_strings(norb, nelec/2)
numpy.random.seed(15)
ci0 = numpy.random.random((na,na))
ci0 = ci0
ci0 /= numpy.linalg.norm(ci0)
ci1 = numpy.random.random((na,na))
ci1 = ci1
ci1 /= numpy.linalg.norm(ci1)

class KnowValues(unittest.TestCase):
    def test_contract(self):
        ci1 = fci.direct_ms0.contract_1e(h1e, ci0, norb, nelec)
        self.assertAlmostEqual(numpy.linalg.norm(ci1), 9.4276430359009442, 10)
        ci1 = fci.direct_ms0.contract_2e(g2e, ci0, norb, nelec)
        self.assertAlmostEqual(numpy.linalg.norm(ci1), 15.40298630382985, 10)

    def test_kernel(self):
        e, c = fci.direct_ms0.kernel(h1e, g2e, norb, nelec)
        self.assertAlmostEqual(e, -9.1491239692, 8)

    def test_hdiag(self):
        hdiag = fci.direct_ms0.make_hdiag(h1e, g2e, norb, nelec)
        self.assertAlmostEqual(numpy.linalg.norm(hdiag), 996.50750750276575, 10)

    def test_rdm1(self):
        dm1 = fci.direct_ms0.make_rdm1(ci0, norb, nelec)
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 2.5141474316952714, 10)

    def test_rdm12(self):
        dm1, dm2 = fci.direct_ms0.make_rdm12(ci0, norb, nelec)
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 2.5141474316952732, 10)
        self.assertAlmostEqual(numpy.linalg.norm(dm2), 7.0301864978453006, 10)

    def test_trans_rdm1(self):
        dm1 = fci.direct_ms0.trans_rdm1(ci0, ci1, norb, nelec)
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 2.2335958337613193, 10)
        dm0 = fci.direct_ms0.make_rdm1(ci0, norb, nelec)
        dm1 = fci.direct_ms0.trans_rdm1(ci0, ci0, norb, nelec)
        self.assertTrue(numpy.allclose(dm1,dm0))

    def test_trans_rdm12(self):
        dm1, dm2 = fci.direct_ms0.trans_rdm12(ci0, ci1, norb, nelec)
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 2.2335958337613198, 10)
        self.assertAlmostEqual(numpy.linalg.norm(dm2), 6.7750795961899541, 10)
        _,dm0 = fci.direct_ms0.make_rdm12(ci0, norb, nelec)
        _,dm2 = fci.direct_ms0.trans_rdm12(ci0, ci0, norb, nelec)
        self.assertTrue(numpy.allclose(dm2,dm0))


if __name__ == "__main__":
    print "Full Tests for ms0"
    unittest.main()

