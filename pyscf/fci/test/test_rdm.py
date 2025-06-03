#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import reduce
import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf import mcscf

def setUpModule():
    global ci0, rdm1, rdm2, norb, nelec
    norb = 6
    nelec = 6
    na = fci.cistring.num_strings(norb, nelec//2)
    numpy.random.seed(1)
    ci0 = numpy.random.random((na,na))
    ci0 = ci0 + ci0.T
    rdm1, rdm2 = fci.direct_spin1.make_rdm12(ci0, norb, nelec)

def tearDownModule():
    global ci0, rdm1, rdm2
    del ci0, rdm1, rdm2

class KnownValues(unittest.TestCase):
    def test_rdm3(self):
        dm3ref = make_dm3_o0(ci0, norb, nelec)
        dm1, dm2, dm3 = fci.rdm.make_dm123('FCI3pdm_kern_spin0', ci0, ci0, norb, nelec)
        self.assertTrue(numpy.allclose(dm3ref, dm3))

        dm3a = reorder_dm123_o0(dm1, dm2, dm3, False)[2]
        dm3b = fci.rdm.reorder_dm123(dm1, dm2, dm3, False)[2]
        self.assertTrue(numpy.allclose(dm3a, dm3b))
        dm3 = dm3b
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

    def test_rdm3s(self):
        dm1, dm2, dm3 = fci.direct_spin1.make_rdm123(ci0, norb, (nelec//2,nelec//2))

        (dm1a, dm1b), (dm2aa, dm2ab, dm2bb), (dm3aaa, dm3aab, dm3abb, dm3bbb) = \
        fci.direct_spin1.make_rdm123s(ci0, norb, (nelec//2,nelec//2))

        self.assertTrue(numpy.allclose(dm1a+dm1b, dm1))
        self.assertTrue(numpy.allclose(dm2aa+dm2bb+dm2ab+dm2ab.transpose(2,3,0,1), dm2))
        self.assertTrue(numpy.allclose(dm3aaa+dm3bbb+dm3aab+dm3aab.transpose(0,1,4,5,2,3)+\
        dm3aab.transpose(4,5,0,1,2,3)+dm3abb+dm3abb.transpose(2,3,0,1,4,5)+dm3abb.transpose(2,3,4,5,0,1), dm3))

        (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2bb) = \
        fci.direct_spin1.make_rdm12s(ci0, norb, (nelec//2,nelec//2))

        self.assertTrue(numpy.allclose(rdm1a, dm1a))
        self.assertTrue(numpy.allclose(rdm1b, dm1b))

        fac = 1. / (nelec//2-1)
        self.assertTrue(numpy.allclose(rdm1a, numpy.einsum('ijmm->ij',dm2aa)*fac))
        self.assertTrue(numpy.allclose(rdm1a, numpy.einsum('mmij->ij',dm2aa)*fac))
        self.assertTrue(numpy.allclose(rdm1b, numpy.einsum('ijmm->ij',dm2bb)*fac))
        self.assertTrue(numpy.allclose(rdm1b, numpy.einsum('mmij->ij',dm2bb)*fac))

        fac = 1. / (nelec//2) # dm2ab has a prefactor N * N instead of N(N-1)
        self.assertTrue(numpy.allclose(rdm1a, numpy.einsum('ijmm->ij',dm2ab)*fac))
        self.assertTrue(numpy.allclose(rdm1b, numpy.einsum('mmij->ij',dm2ab)*fac))

        fac = 1. / (nelec//2-2)
        self.assertTrue(numpy.allclose(rdm2aa, numpy.einsum('ijklmm->ijkl',dm3aaa)*fac))
        self.assertTrue(numpy.allclose(rdm2aa, numpy.einsum('ijmmkl->ijkl',dm3aaa)*fac))
        self.assertTrue(numpy.allclose(rdm2aa, numpy.einsum('mmijkl->ijkl',dm3aaa)*fac))

        self.assertTrue(numpy.allclose(rdm2bb, numpy.einsum('ijklmm->ijkl',dm3bbb)*fac))
        self.assertTrue(numpy.allclose(rdm2bb, numpy.einsum('ijmmkl->ijkl',dm3bbb)*fac))
        self.assertTrue(numpy.allclose(rdm2bb, numpy.einsum('mmijkl->ijkl',dm3bbb)*fac))

        fac = 1. / (nelec//2) # dm3aab/abb has a prefactor N * N * (N-1) instead of N(N-1)(N-2)
        self.assertTrue(numpy.allclose(rdm2aa, numpy.einsum('ijklmm->ijkl',dm3aab)*fac))
        self.assertTrue(numpy.allclose(rdm2bb, numpy.einsum('mmijkl->ijkl',dm3abb)*fac))

        fac = 1. / (nelec//2-1)
        self.assertTrue(numpy.allclose(rdm2ab, numpy.einsum('ijmmkl->ijkl',dm3aab)*fac))
        self.assertTrue(numpy.allclose(rdm2ab, numpy.einsum('mmijkl->ijkl',dm3aab)*fac))
        self.assertTrue(numpy.allclose(rdm2ab, numpy.einsum('ijklmm->ijkl',dm3abb)*fac))
        self.assertTrue(numpy.allclose(rdm2ab, numpy.einsum('ijmmkl->ijkl',dm3abb)*fac))

    def test_dm4(self):
        dm4ref = make_dm4_o0(ci0, norb, nelec)
        dm4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf', ci0, ci0, norb, nelec)[3]
        self.assertTrue(numpy.allclose(dm4ref, dm4))

        numpy.random.seed(2)
        na = fci.cistring.num_strings(norb, 5)
        nb = fci.cistring.num_strings(norb, 3)
        ci1 = numpy.random.random((na,nb))
        dm4ref = make_dm4_o0(ci1, norb, (5,3))
        dm1, dm2, dm3, dm4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf', ci1, ci1, norb, (5,3))
        self.assertTrue(numpy.allclose(dm4ref, dm4))
        self.assertTrue(numpy.allclose(dm3, numpy.einsum('ppmnijkl->mnijkl',dm4)/8))
        self.assertTrue(numpy.allclose(dm3, numpy.einsum('mnppijkl->mnijkl',dm4)/8))
        self.assertTrue(numpy.allclose(dm3, numpy.einsum('mnijppkl->mnijkl',dm4)/8))
        self.assertTrue(numpy.allclose(dm3, numpy.einsum('mnijklpp->mnijkl',dm4)/8))

        dm3a, dm4a = reorder_dm1234_o0(dm1, dm2, dm3, dm4, False)[2:]
        dm4b = fci.rdm.reorder_dm1234(dm1, dm2, dm3, dm4, False)[3]
        self.assertTrue(numpy.allclose(dm4a, dm4b))
        self.assertTrue(numpy.allclose(dm3a, numpy.einsum('ppmnijkl->mnijkl',dm4b)/5))
        self.assertTrue(numpy.allclose(dm3a, numpy.einsum('mnppijkl->mnijkl',dm4b)/5))
        self.assertTrue(numpy.allclose(dm3a, numpy.einsum('mnijppkl->mnijkl',dm4b)/5))
        self.assertTrue(numpy.allclose(dm3a, numpy.einsum('mnijklpp->mnijkl',dm4b)/5))

    def test_rdm4s(self):
        dm1, dm2, dm3, dm4 = fci.direct_spin1.make_rdm1234(ci0, norb, (nelec//2,nelec//2))

        (dm1a, dm1b), (dm2aa, dm2ab, dm2bb), (dm3aaa, dm3aab, dm3abb, dm3bbb), (dm4aaaa, dm4aaab, dm4aabb, dm4abbb, dm4bbbb) = \
        fci.direct_spin1.make_rdm1234s(ci0, norb, (nelec//2,nelec//2))

        self.assertTrue(numpy.allclose(dm1a+dm1b, dm1))
        self.assertTrue(numpy.allclose(dm2aa+dm2bb+dm2ab+dm2ab.transpose(2,3,0,1), dm2))
        self.assertTrue(numpy.allclose(dm3aaa+dm3bbb+dm3aab+dm3aab.transpose(0,1,4,5,2,3)+\
        dm3aab.transpose(4,5,0,1,2,3)+dm3abb+dm3abb.transpose(2,3,0,1,4,5)+dm3abb.transpose(2,3,4,5,0,1), dm3))
        self.assertTrue(numpy.allclose(
            dm4aaaa
            + dm4bbbb
            + dm4aaab
            + dm4aaab.transpose(0, 1, 2, 3, 6, 7, 4, 5)
            + dm4aaab.transpose(0, 1, 6, 7, 2, 3, 4, 5)
            + dm4aaab.transpose(6, 7, 0, 1, 2, 3, 4, 5)
            + dm4aabb
            + dm4aabb.transpose(0, 1, 4, 5, 2, 3, 6, 7)
            + dm4aabb.transpose(4, 5, 0, 1, 2, 3, 6, 7)
            + dm4aabb.transpose(0, 1, 4, 5, 6, 7, 2, 3)
            + dm4aabb.transpose(4, 5, 0, 1, 6, 7, 2, 3)
            + dm4aabb.transpose(4, 5, 6, 7, 0, 1, 2, 3)
            + dm4abbb
            + dm4abbb.transpose(2, 3, 0, 1, 4, 5, 6, 7)
            + dm4abbb.transpose(2, 3, 4, 5, 0, 1, 6, 7)
            + dm4abbb.transpose(2, 3, 4, 5, 6, 7, 0, 1),
            dm4,
        ))

        (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2bb), (rdm3aaa, rdm3aab, rdm3abb, rdm3bbb) = \
        fci.direct_spin1.make_rdm123s(ci0, norb, (nelec//2,nelec//2))

        self.assertTrue(numpy.allclose(rdm1a, dm1a))
        self.assertTrue(numpy.allclose(rdm1b, dm1b))

        fac = 1. / (nelec//2-1)
        self.assertTrue(numpy.allclose(rdm1a, numpy.einsum('ijmm->ij',dm2aa)*fac))
        self.assertTrue(numpy.allclose(rdm1a, numpy.einsum('mmij->ij',dm2aa)*fac))
        self.assertTrue(numpy.allclose(rdm1b, numpy.einsum('ijmm->ij',dm2bb)*fac))
        self.assertTrue(numpy.allclose(rdm1b, numpy.einsum('mmij->ij',dm2bb)*fac))

        fac = 1. / (nelec//2) # dm2ab has a prefactor N * N instead of N(N-1)
        self.assertTrue(numpy.allclose(rdm1a, numpy.einsum('ijmm->ij',dm2ab)*fac))
        self.assertTrue(numpy.allclose(rdm1b, numpy.einsum('mmij->ij',dm2ab)*fac))

        fac = 1. / (nelec//2-2)
        self.assertTrue(numpy.allclose(rdm2aa, numpy.einsum('ijklmm->ijkl',dm3aaa)*fac))
        self.assertTrue(numpy.allclose(rdm2aa, numpy.einsum('ijmmkl->ijkl',dm3aaa)*fac))
        self.assertTrue(numpy.allclose(rdm2aa, numpy.einsum('mmijkl->ijkl',dm3aaa)*fac))

        self.assertTrue(numpy.allclose(rdm2bb, numpy.einsum('ijklmm->ijkl',dm3bbb)*fac))
        self.assertTrue(numpy.allclose(rdm2bb, numpy.einsum('ijmmkl->ijkl',dm3bbb)*fac))
        self.assertTrue(numpy.allclose(rdm2bb, numpy.einsum('mmijkl->ijkl',dm3bbb)*fac))

        fac = 1. / (nelec//2) # dm3aab/abb has a prefactor N * N * (N-1) instead of N(N-1)(N-2)
        self.assertTrue(numpy.allclose(rdm2aa, numpy.einsum('ijklmm->ijkl',dm3aab)*fac))
        self.assertTrue(numpy.allclose(rdm2bb, numpy.einsum('mmijkl->ijkl',dm3abb)*fac))

        fac = 1. / (nelec//2-1)
        self.assertTrue(numpy.allclose(rdm2ab, numpy.einsum('ijmmkl->ijkl',dm3aab)*fac))
        self.assertTrue(numpy.allclose(rdm2ab, numpy.einsum('mmijkl->ijkl',dm3aab)*fac))
        self.assertTrue(numpy.allclose(rdm2ab, numpy.einsum('ijklmm->ijkl',dm3abb)*fac))
        self.assertTrue(numpy.allclose(rdm2ab, numpy.einsum('ijmmkl->ijkl',dm3abb)*fac))

        # NOTE: Tests skipeed because system only contains 3 alpha and beta electrons
        # fac = 1. / (nelec//2-3)
        # self.assertTrue(numpy.allclose(rdm3aaa, numpy.einsum('ijklmnoo->ijklmn',dm4aaaa)*fac))
        # self.assertTrue(numpy.allclose(rdm3aaa, numpy.einsum('ijkloomn->ijklmn',dm4aaaa)*fac))
        # self.assertTrue(numpy.allclose(rdm3aaa, numpy.einsum('ijooklmn->ijklmn',dm4aaaa)*fac))
        # self.assertTrue(numpy.allclose(rdm3aaa, numpy.einsum('ooijklmn->ijklmn',dm4aaaa)*fac))
        #
        # self.assertTrue(numpy.allclose(rdm3bbb, numpy.einsum('ijklmnoo->ijklmn',dm4bbbb)*fac))
        # self.assertTrue(numpy.allclose(rdm3bbb, numpy.einsum('ijkloomn->ijklmn',dm4bbbb)*fac))
        # self.assertTrue(numpy.allclose(rdm3bbb, numpy.einsum('ijooklmn->ijklmn',dm4bbbb)*fac))
        # self.assertTrue(numpy.allclose(rdm3bbb, numpy.einsum('ooijklmn->ijklmn',dm4bbbb)*fac))

        fac = 1. / (nelec//2-2)
        self.assertTrue(numpy.allclose(rdm3aab, numpy.einsum('ooijklmn->ijklmn',dm4aaab)*fac))
        self.assertTrue(numpy.allclose(rdm3aab, numpy.einsum('ijooklmn->ijklmn',dm4aaab)*fac))
        self.assertTrue(numpy.allclose(rdm3aab, numpy.einsum('ijkloomn->ijklmn',dm4aaab)*fac))

        self.assertTrue(numpy.allclose(rdm3abb, numpy.einsum('ijklmnoo->ijklmn',dm4abbb)*fac))
        self.assertTrue(numpy.allclose(rdm3abb, numpy.einsum('ijkloomn->ijklmn',dm4abbb)*fac))
        self.assertTrue(numpy.allclose(rdm3abb, numpy.einsum('ijooklmn->ijklmn',dm4abbb)*fac))

        fac = 1. / (nelec//2-1)
        self.assertTrue(numpy.allclose(rdm3aab, numpy.einsum('ijklmnoo->ijklmn',dm4aabb)*fac))
        self.assertTrue(numpy.allclose(rdm3aab, numpy.einsum('ijkloomn->ijklmn',dm4aabb)*fac))

        self.assertTrue(numpy.allclose(rdm3abb, numpy.einsum('ijooklmn->ijklmn',dm4aabb)*fac))
        self.assertTrue(numpy.allclose(rdm3abb, numpy.einsum('ooijklmn->ijklmn',dm4aabb)*fac))

        fac = 1. / (nelec//2)
        self.assertTrue(numpy.allclose(rdm3bbb, numpy.einsum('ooijklmn->ijklmn',dm4abbb)*fac))
        self.assertTrue(numpy.allclose(rdm3aaa, numpy.einsum('ijklmnoo->ijklmn',dm4aaab)*fac))

    def test_tdm2(self):
        dm1 = numpy.einsum('ij,ijkl->lk', ci0, _trans1(ci0, norb, nelec))
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
        dm1a = numpy.einsum('ij,ijpq->qp', ci, t1)
        dm2a = numpy.einsum('ij,ijpqrs->pqrs', ci, t2)
        self.assertTrue(numpy.allclose(dm1a, dm1))
        dm1a, dm2a = fci.rdm.reorder_rdm(dm1a, dm2a)
        self.assertTrue(numpy.allclose(dm2a,dm2a.transpose(2,3,0,1)))

    def test_full_alpha(self):
        nelec = (6,3)
        norb = 6
        npair = norb*(norb+1)//2
        numpy.random.seed(12)
        h1 = numpy.random.random((norb,norb))
        h1 = h1 + h1.T
        h2 = numpy.random.random((npair,npair)) * .1
        h2 = h2 + h2.T
        cis = fci.direct_spin1.FCI()
        e, c = cis.kernel(h1, h2, norb, nelec, verbose=5)
        dm1s, dm2s = cis.make_rdm12s(c, norb, nelec)
        self.assertAlmostEqual(abs(dm1s[0]).sum(), 6, 9)
        self.assertAlmostEqual(dm1s[1].trace(), 3, 9)
        self.assertAlmostEqual(abs(dm2s[0]).sum(), 60, 9)
        self.assertAlmostEqual(abs(numpy.einsum('iijk->jk', dm2s[1])/6-dm1s[1]).sum(), 0, 9)
        self.assertAlmostEqual(abs(numpy.einsum('iijk->jk', dm2s[2])/2-dm1s[1]).sum(), 0, 9)

    def test_0beta(self):
        nelec = (3,0)
        norb = 6
        npair = norb*(norb+1)//2
        numpy.random.seed(12)
        h1 = numpy.random.random((norb,norb))
        h1 = h1 + h1.T
        h2 = numpy.random.random((npair,npair)) * .1
        h2 = h2 + h2.T
        cis = fci.direct_spin1.FCI()
        e, c = cis.kernel(h1, h2, norb, nelec, verbose=5)
        dm1s, dm2s = cis.make_rdm12s(c, norb, nelec)
        self.assertAlmostEqual(dm1s[0].trace(), 3, 9)
        self.assertAlmostEqual(abs(dm1s[1]).sum(), 0, 9)
        self.assertAlmostEqual(abs(numpy.einsum('iijk->jk', dm2s[0])/2-dm1s[0]).sum(), 0, 9)
        self.assertAlmostEqual(abs(dm2s[1]).sum(), 0, 9)
        self.assertAlmostEqual(abs(dm2s[2]).sum(), 0, 9)

# (6o,6e)   ~ 4MB
# (8o,8e)   ~ 153MB
# (10o,10e) ~ 4.8GB
# t2(*,ij,kl) = E_i^j E_k^l|0>
def _trans2(fcivec, norb, nelec):
    if isinstance(nelec, (int, numpy.integer)):
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
    if isinstance(nelec, (int, numpy.integer)):
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

# <p^+ q r^+ s t^+ u> => <p^+ r^+ t^+ u s q>
# rdm2 is <p^+ q r^+ s>
def reorder_dm123_o0(rdm1, rdm2, rdm3, inplace=True):
    rdm1, rdm2 = fci.rdm.reorder_rdm(rdm1, rdm2, inplace)
    if not inplace:
        rdm3 = rdm3.copy()
    norb = rdm1.shape[0]
    for p in range(norb):
        for q in range(norb):
            for s in range(norb):
                rdm3[p,q,q,s] += -rdm2[p,s]
            for u in range(norb):
                rdm3[p,q,:,:,q,u] += -rdm2[p,u]
            for s in range(norb):
                rdm3[p,q,:,s,s,:] += -rdm2[p,q]
    for q in range(norb):
        for s in range(norb):
            rdm3[:,q,q,s,s,:] += -rdm1
    return rdm1, rdm2, rdm3

# <p^+ q r^+ s t^+ u w^+ v> => <p^+ r^+ t^+ w^+ v u s q>
# rdm2, rdm3 are the (reordered) standard 2-pdm and 3-pdm
def reorder_dm1234_o0(rdm1, rdm2, rdm3, rdm4, inplace=True):
    rdm1, rdm2, rdm3 = fci.rdm.reorder_dm123(rdm1, rdm2, rdm3, inplace)
    if not inplace:
        rdm4 = rdm4.copy()
    norb = rdm1.shape[0]
    delta = numpy.eye(norb)
    rdm4 -= numpy.einsum('qv,pwrstu->pqrstuvw', delta, rdm3)
    rdm4 -= numpy.einsum('sv,pqrwtu->pqrstuvw', delta, rdm3)
    rdm4 -= numpy.einsum('uv,pqrstw->pqrstuvw', delta, rdm3)
    rdm4 -= numpy.einsum('qt,pursvw->pqrstuvw', delta, rdm3)
    rdm4 -= numpy.einsum('st,pqruvw->pqrstuvw', delta, rdm3)
    rdm4 -= numpy.einsum('qr,pstuvw->pqrstuvw', delta, rdm3)
    rdm4 -= numpy.einsum('qr,sv,pwtu', delta, delta, rdm2)
    rdm4 -= numpy.einsum('qr,uv,pstw', delta, delta, rdm2)
    rdm4 -= numpy.einsum('qt,uv,pwrs', delta, delta, rdm2)
    rdm4 -= numpy.einsum('qt,sv,purw', delta, delta, rdm2)
    rdm4 -= numpy.einsum('st,qv,pwru', delta, delta, rdm2)
    rdm4 -= numpy.einsum('st,uv,pqrw', delta, delta, rdm2)
    rdm4 -= numpy.einsum('qr,st,puvw', delta, delta, rdm2)
    rdm4 -= numpy.einsum('qr,st,uv,pw->pqrstuvw', delta, delta, delta, rdm1)
    return rdm1, rdm2, rdm3, rdm4


if __name__ == "__main__":
    print("Full Tests for fci.rdm")
    unittest.main()
