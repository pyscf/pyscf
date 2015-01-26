from functools import reduce
import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci

norb = 6
nelec = 6
na = fci.cistring.num_strings(norb, nelec//2)
numpy.random.seed(1)
ci0 = numpy.random.random((na,na))
rdm1, rdm2 = fci.direct_spin1.make_rdm12(ci0, norb, nelec)

class KnowValues(unittest.TestCase):
    def test_rdm3(self):
        dm3ref = fci.rdm.make_dm3_o0(ci0, norb, nelec)
        dm3 = fci.rdm.make_dm3(ci0, norb, nelec)
        self.assertTrue(numpy.allclose(dm3ref, dm3))

        dm3 = fci.rdm.reorder_rdm3(rdm1, rdm2, dm3)
        fac = 1. / (nelec-2)
        self.assertTrue(numpy.allclose(rdm2, numpy.einsum('ijklmm->ijkl',dm3)*fac))
        self.assertTrue(numpy.allclose(rdm2, numpy.einsum('ijmmkl->ijkl',dm3)*fac))
        self.assertTrue(numpy.allclose(rdm2, numpy.einsum('mmijkl->ijkl',dm3)*fac))

    def test_dm4(self):
        dm4ref = fci.rdm.make_dm4_o0(ci0, norb, nelec)
        dm4 = fci.rdm.make_dm4(ci0, norb, nelec)
        self.assertTrue(numpy.allclose(dm4ref, dm4))

    def test_tdm2(self):
        dm1 = numpy.einsum('ij,ijkl->kl', ci0, fci.rdm._trans1(ci0, norb, nelec))
        self.assertTrue(numpy.allclose(rdm1, dm1))

        dm2 = numpy.einsum('ij,ijklmn->klmn', ci0, fci.rdm._trans2(ci0, norb, nelec))
        dm2 = fci.rdm.reorder_rdm(rdm1, dm2)[1]
        self.assertTrue(numpy.allclose(rdm2,dm2))

        na = ci0.shape[0]
        numpy.random.seed(1)
        ci = numpy.random.random((na,na))
        ci1 = numpy.random.random((na,na))
        dm1, dm2 = fci.direct_spin1.trans_rdm12(ci, ci1, norb, nelec)
        numpy.random.seed(2)
        self.assertAlmostEqual(numpy.dot(dm2.flatten(),numpy.random.random(dm2.size)),
                               3790.8867819690477, 7)
        self.assertTrue(numpy.allclose(dm2, dm2.transpose(2,3,0,1)))

        t1 = fci.rdm._trans1(ci1, norb, nelec)
        t2 = fci.rdm._trans2(ci1, norb, nelec)
        dm1a = numpy.einsum('ij,ijpq->pq', ci, t1)
        dm2a = numpy.einsum('ij,ijpqrs->pqrs', ci, t2)
        self.assertTrue(numpy.allclose(dm1a, dm1))
        dm1a, dm2a = fci.rdm.reorder_rdm(dm1a, dm2a)
        self.assertTrue(numpy.allclose(dm2a,dm2a.transpose(2,3,0,1)))

if __name__ == "__main__":
    print("Full Tests for fci.rdm")
    unittest.main()


