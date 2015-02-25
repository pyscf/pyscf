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
ci0 = ci0 + ci0.T
rdm1, rdm2 = fci.direct_spin1.make_rdm12(ci0, norb, nelec)

class KnowValues(unittest.TestCase):
    def test_rdm3(self):
        dm3ref = make_dm3_o0(ci0, norb, nelec)
        dm3 = fci.rdm.make_dm123('FCI3pdm_kern_spin0', ci0, ci0, norb, nelec)[2]
        self.assertTrue(numpy.allclose(dm3ref, dm3))

        dm3 = fci.rdm.reorder_rdm3(rdm1, rdm2, dm3)
        fac = 1. / (nelec-2)
        self.assertTrue(numpy.allclose(rdm2, numpy.einsum('ijklmm->ijkl',dm3)*fac))
        self.assertTrue(numpy.allclose(rdm2, numpy.einsum('ijmmkl->ijkl',dm3)*fac))
        self.assertTrue(numpy.allclose(rdm2, numpy.einsum('mmijkl->ijkl',dm3)*fac))

        dm3 = fci.rdm.make_dm123('FCI3pdm_kern_sf', ci0, ci0, norb, nelec)[2]
        dm2 = fci.direct_spin1.make_rdm12(ci0, norb, nelec, reorder=False)[1]
        self.assertTrue(numpy.allclose(dm2, numpy.einsum('mmijkl->ijkl',dm3)/nelec))

        numpy.random.seed(2)
        na = fci.cistring.num_strings(norb, 5)
        nb = fci.cistring.num_strings(norb, 3)
        ci1 = numpy.random.random((na,nb))
        dm3ref = make_dm3_o0(ci1, norb, (5,3))
        dm3 = fci.rdm.make_dm123('FCI3pdm_kern_sf', ci1, ci1, norb, (5,3))[2]
        self.assertTrue(numpy.allclose(dm3ref, dm3))

    def test_dm4(self):
        dm4ref = make_dm4_o0(ci0, norb, nelec)
        dm4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf', ci0, ci0, norb, nelec)[3]
        self.assertTrue(numpy.allclose(dm4ref, dm4))

        numpy.random.seed(2)
        na = fci.cistring.num_strings(norb, 5)
        nb = fci.cistring.num_strings(norb, 3)
        ci1 = numpy.random.random((na,nb))
        dm4ref = make_dm4_o0(ci1, norb, (5,3))
        dm4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf', ci1, ci1, norb, (5,3))[3]
        self.assertTrue(numpy.allclose(dm4ref, dm4))

    def test_tdm2(self):
        dm1 = numpy.einsum('ij,ijkl->kl', ci0, _trans1(ci0, norb, nelec))
        self.assertTrue(numpy.allclose(rdm1, dm1))

        dm2 = numpy.einsum('ij,ijklmn->klmn', ci0, _trans2(ci0, norb, nelec))
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

        t1 = _trans1(ci1, norb, nelec)
        t2 = _trans2(ci1, norb, nelec)
        dm1a = numpy.einsum('ij,ijpq->pq', ci, t1)
        dm2a = numpy.einsum('ij,ijpqrs->pqrs', ci, t2)
        self.assertTrue(numpy.allclose(dm1a, dm1))
        dm1a, dm2a = fci.rdm.reorder_rdm(dm1a, dm2a)
        self.assertTrue(numpy.allclose(dm2a,dm2a.transpose(2,3,0,1)))

# (6o,6e)   ~ 4MB
# (8o,8e)   ~ 153MB
# (10o,10e) ~ 4.8GB
# t2(*,ij,kl) = E_i^j E_k^l|0>
def _trans2(fcivec, norb, nelec):
    if isinstance(nelec, int):
        neleca = nelecb = nelec//2
    else:
        neleca, nelecb = nelec
    link_indexa = fci.cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = fci.cistring.gen_linkstr_index(range(norb), nelecb)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    fcivec = fcivec.reshape(na,nb)
    t1 = _trans1(fcivec, norb, nelec)
    t2 = numpy.zeros((na,nb,norb,norb,norb,norb))
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t2[str1,:,a,i] += sign * t1[str0]
    for k in range(na):
        for str0, tab in enumerate(link_indexb):
            for a, i, str1, sign in tab:
                t2[k,str1,a,i] += sign * t1[k,str0]
    return t2
def _trans1(fcivec, norb, nelec):
    if isinstance(nelec, int):
        neleca = nelecb = nelec//2
    else:
        neleca, nelecb = nelec
    link_indexa = fci.cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = fci.cistring.gen_linkstr_index(range(norb), nelecb)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    fcivec = fcivec.reshape(na,nb)
    t1 = numpy.zeros((na,nb,norb,norb))
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1[str1,:,a,i] += sign * fcivec[str0]
    for k in range(na):
        for str0, tab in enumerate(link_indexb):
            for a, i, str1, sign in tab:
                t1[k,str1,a,i] += sign * fcivec[k,str0]
    return t1

#
# NOTE: this rdm3 is defined as
# rdm3(p,q,r,s,t,u) = <p^+ q r^+ s t^+ u>
def make_dm3_o0(fcivec, norb, nelec):
    # <0|p^+ q r^+ s|i> <i|t^+ u|0>
    t1 = _trans1(fcivec, norb, nelec)
    t2 = _trans2(fcivec, norb, nelec)
    na, nb = t1.shape[:2]
    rdm3 = numpy.dot(t1.reshape(na*nb,-1).T, t2.reshape(na*nb,-1))
    return rdm3.reshape((norb,)*6).transpose(1,0,2,3,4,5)

def make_dm4_o0(fcivec, norb, nelec):
    # <0|p^+ q r^+ s|i> <i|t^+ u|0>
    t2 = _trans2(fcivec, norb, nelec)
    na, nb = t2.shape[:2]
    rdm4 = numpy.dot(t2.reshape(na*nb,-1).T, t2.reshape(na*nb,-1))
    return rdm4.reshape((norb,)*8).transpose(3,2,1,0,4,5,6,7)


if __name__ == "__main__":
    print("Full Tests for fci.rdm")
    unittest.main()


