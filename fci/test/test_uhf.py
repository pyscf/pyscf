#!/usr/bin/env python

import unittest
import numpy
import gto
import scf
import ao2mo
import fci

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

mol.basis = {'H': 'sto-3g'}
mol.charge = 1
mol.build()

m = scf.UHF(mol)
ehf = m.scf()

mo_a, mo_b = m.mo_coeff
norb = mo_a.shape[1]
nelecr = ((mol.nelectron+1)/2, (mol.nelectron+1)/2)
h1er = reduce(numpy.dot, (mo_a.T, m.get_hcore()[0], mo_a))
g2er = ao2mo.incore.general(m._eri, (mo_a,)*4, compact=False)
h1es = (h1er, h1er)
g2es = (g2er, g2er, g2er)
na = fci.cistring.num_strings(norb, nelecr[0])
nb = fci.cistring.num_strings(norb, nelecr[1])
numpy.random.seed(15)
ci0 = numpy.random.random((na,nb))
ci1 = numpy.random.random((na,nb))

neleci = ((mol.nelectron+1)/2, (mol.nelectron-1)/2)
na = fci.cistring.num_strings(norb, neleci[0])
nb = fci.cistring.num_strings(norb, neleci[1])
h1ei = (reduce(numpy.dot, (mo_a.T, m.get_hcore()[0], mo_a)),
        reduce(numpy.dot, (mo_b.T, m.get_hcore()[0], mo_b)))
g2ei = (ao2mo.incore.general(m._eri, (mo_a,)*4, compact=False),
        ao2mo.incore.general(m._eri, (mo_a,mo_a,mo_b,mo_b), compact=False),
        ao2mo.incore.general(m._eri, (mo_b,)*4, compact=False))
numpy.random.seed(15)
ci2 = numpy.random.random((na,nb))
ci3 = numpy.random.random((na,nb))

class KnowValues(unittest.TestCase):
    def test_contract(self):
        ci1ref = fci.direct_spin1.contract_1e(h1er, ci0, norb, nelecr)
        ci1 = fci.direct_uhf.contract_1e(h1es, ci0, norb, nelecr)
        self.assertTrue(numpy.allclose(ci1, ci1ref))
        ci1ref = fci.direct_spin1.contract_2e(g2er, ci0, norb, nelecr)
        ci1 = fci.direct_uhf.contract_2e(g2es, ci0, norb, nelecr)
        self.assertTrue(numpy.allclose(ci1, ci1ref))
        self.assertAlmostEqual(numpy.linalg.norm(ci1), 198.313713069129, 10)
        ci3 = fci.direct_uhf.contract_2e(g2ei, ci2, norb, neleci)
        self.assertAlmostEqual(numpy.linalg.norm(ci3), 122.589931616116, 10)

    def test_kernel(self):
        eref, cref = fci.direct_spin1.kernel(h1er, g2er, norb, nelecr)
        e, c = fci.direct_uhf.kernel(h1es, g2es, norb, nelecr)
        self.assertAlmostEqual(e, eref, 8)
        self.assertAlmostEqual(e, -8.9347029192929, 8)
        e, c = fci.direct_uhf.kernel(h1es, g2es, norb, neleci)
        self.assertAlmostEqual(e, -8.7498253981782, 8)

    def test_hdiag(self):
        hdiagref = fci.direct_spin1.make_hdiag(h1er, g2er, norb, nelecr)
        hdiag = fci.direct_uhf.make_hdiag(h1es, g2es, norb, nelecr)
        self.assertTrue(numpy.allclose(hdiag, hdiagref))
        self.assertAlmostEqual(numpy.linalg.norm(hdiag), 133.98845707428, 10)
        hdiag = fci.direct_uhf.make_hdiag(h1es, g2es, norb, neleci)
        self.assertAlmostEqual(numpy.linalg.norm(hdiag), 113.87136636936, 10)

    def test_rdm1(self):
        dm1ref = fci.direct_spin1.make_rdm1(ci0, norb, nelecr)
        dm1 = fci.direct_uhf.make_rdm1(ci0, norb, nelecr)
        self.assertTrue(numpy.allclose(dm1ref, dm1))
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 393.03762428630, 10)
        dm1 = fci.direct_uhf.make_rdm1(ci2, norb, neleci)
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 242.33237916212, 10)

    def test_rdm12(self):
        dm1ref, dm2ref = fci.direct_spin1.make_rdm12(ci0, norb, nelecr)
        dm1, dm2 = fci.direct_uhf.make_rdm12s(ci0, norb, nelecr)
        dm1 = dm1[0] + dm1[1]
        dm2 = dm2[0] + dm2[1] + dm2[1].transpose(2,3,0,1) + dm2[2]
        self.assertTrue(numpy.allclose(dm1ref, dm1))
        self.assertTrue(numpy.allclose(dm2ref, dm2))
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 393.0376242863019, 10)
        self.assertAlmostEqual(numpy.linalg.norm(dm2), 1155.413506052811, 10)
        dm1, dm2 = fci.direct_uhf.make_rdm12s(ci2, norb, neleci)
        self.assertAlmostEqual(numpy.linalg.norm(dm1[0]), 143.05770559808, 10)
        self.assertAlmostEqual(numpy.linalg.norm(dm1[1]), 109.30195472840, 10)
        self.assertAlmostEqual(numpy.linalg.norm(dm2[0]), 258.07143130273, 10)
        self.assertAlmostEqual(numpy.linalg.norm(dm2[1]), 172.41469868799, 10)
        self.assertAlmostEqual(numpy.linalg.norm(dm2[2]), 149.76371060734, 10)

    def test_trans_rdm1(self):
        dm1ref = fci.direct_spin1.trans_rdm1(ci0, ci1, norb, nelecr)
        dm1 = fci.direct_uhf.trans_rdm1(ci0, ci1, norb, nelecr)
        self.assertTrue(numpy.allclose(dm1ref, dm1))
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 294.40681527414, 10)
        dm0 = fci.direct_uhf.make_rdm1(ci0, norb, nelecr)
        dm1 = fci.direct_uhf.trans_rdm1(ci0, ci0, norb, nelecr)
        self.assertTrue(numpy.allclose(dm1, dm0))
        dm1 = fci.direct_uhf.trans_rdm1(ci3, ci2, norb, neleci)
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 193.703051323676, 10)

    def test_trans_rdm12(self):
        dm1ref, dm2ref = fci.direct_spin1.trans_rdm12(ci0, ci1, norb, nelecr)
        dm1, dm2 = fci.direct_uhf.trans_rdm12s(ci0, ci1, norb, nelecr)
        dm1 = dm1[0] + dm1[1]
        dm2 = dm2[0] + dm2[1] + dm2[2] + dm2[3]
        self.assertTrue(numpy.allclose(dm1ref, dm1))
        self.assertTrue(numpy.allclose(dm2ref, dm2))
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 294.4068152741418, 10)
        self.assertAlmostEqual(numpy.linalg.norm(dm2), 949.0343056904616, 10)
        _,dm0 = fci.direct_uhf.make_rdm12s(ci0, norb, nelecr)
        _,dm2 = fci.direct_uhf.trans_rdm12s(ci0, ci0, norb, nelecr)
        self.assertTrue(numpy.allclose(dm2[0], dm0[0]))
        self.assertTrue(numpy.allclose(dm2[1], dm0[1]))
        self.assertTrue(numpy.allclose(dm2[3], dm0[2]))
        dm1, dm2 = fci.direct_uhf.trans_rdm12s(ci3, ci2, norb, neleci)
        self.assertAlmostEqual(numpy.linalg.norm(dm1[0]+dm1[1]), 193.703051323676, 10)
        self.assertAlmostEqual(numpy.linalg.norm(dm1[0]), 112.85954124885, 10)
        self.assertAlmostEqual(numpy.linalg.norm(dm1[1]), 92.827695172359, 10)
        self.assertAlmostEqual(numpy.linalg.norm(sum(dm2)), 512.111790469461, 10)
        self.assertAlmostEqual(numpy.linalg.norm(dm2[0]), 228.750384383495, 10)
        self.assertAlmostEqual(numpy.linalg.norm(dm2[1]), 155.324543159155, 10)
        self.assertAlmostEqual(numpy.linalg.norm(dm2[2]), 155.324543159155, 10)
        self.assertAlmostEqual(numpy.linalg.norm(dm2[3]), 141.269867535222, 10)


if __name__ == "__main__":
    print "Full Tests for uhf-based fci"
    unittest.main()

