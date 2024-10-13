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

import unittest
from functools import reduce
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf.fci import fci_slow

def setUpModule():
    global h1e, h2e, norb, nelec, na, nb
    nelec = (3,4)
    norb = 7
    numpy.random.seed(10)
    h1e = numpy.random.random((norb,norb))
    h2e = numpy.random.random((norb,norb,norb,norb))
    h2e = h2e + h2e.transpose(2,3,0,1)
    na = fci.cistring.num_strings(norb, nelec[0])
    nb = fci.cistring.num_strings(norb, nelec[1])

def tearDownModule():
    global h1e, h2e
    del h1e, h2e

class KnownValues(unittest.TestCase):
    def test_contract(self):
        ci0 = numpy.random.random((na,nb))
        ci1ref = fci_slow.contract_1e(h1e, ci0, norb, nelec)
        ci1 = fci.direct_nosym.contract_1e(h1e, ci0, norb, nelec)
        self.assertTrue(numpy.allclose(ci1ref, ci1))

        ci1ref = fci_slow.contract_2e(h2e, ci0, norb, nelec)
        ci1 = fci.direct_nosym.contract_2e(h2e, ci0, norb, nelec)
        self.assertTrue(numpy.allclose(ci1ref, ci1))

    def test_contract_complex(self):
        ci0 = numpy.random.random((na,nb)) + 1j * numpy.random.random((na,nb))
        ci1ref = fci_slow.contract_1e(h1e, ci0, norb, nelec)
        ci1 = fci.direct_nosym.contract_1e(h1e, ci0.real, norb, nelec).astype(complex)
        ci1 += 1j * fci.direct_nosym.contract_1e(h1e, ci0.imag, norb, nelec)
        self.assertTrue(numpy.allclose(ci1ref, ci1))

        ci1ref = fci_slow.contract_2e(h2e, ci0, norb, nelec)
        ci1 = fci.direct_nosym.contract_2e(h2e, ci0.real, norb, nelec).astype(complex)
        ci1 += 1j * fci.direct_nosym.contract_2e(h2e, ci0.imag, norb, nelec)
        self.assertTrue(numpy.allclose(ci1ref, ci1))

        ci1 = fci.direct_nosym.contract_2e(h2e, ci0, norb, nelec).astype(complex)
        self.assertAlmostEqual(abs(ci1ref - ci1).max(), 0, 12)

    def test_absorb_h1e(self):
        href = fci_slow.absorb_h1e(h1e, h2e, norb, nelec)
        h1 = fci.direct_nosym.absorb_h1e(h1e, h2e, norb, nelec)
        self.assertTrue(numpy.allclose(href, h1))

    def test_absorb_h1e_complex(self):
        href = fci_slow.absorb_h1e(h1e.astype(complex), h2e, norb, nelec)
        h1 = fci.direct_nosym.absorb_h1e(h1e.astype(complex), h2e, norb, nelec)
        self.assertTrue(numpy.allclose(href, h1))

    def test_kernel(self):
        h1 = h1e + h1e.T
        eri = .5* ao2mo.restore(1, ao2mo.restore(8, h2e, norb), norb)
        h = fci.direct_spin1.pspace(h1, eri, norb, nelec, np=5000)[1]
        eref, c0 = numpy.linalg.eigh(h)

        sol = fci.direct_nosym.FCI()
        e, c1 = sol.kernel(h1, eri, norb, nelec, max_space=40)
        self.assertAlmostEqual(eref[0], e, 9)
        self.assertAlmostEqual(abs(c0[:,0].dot(c1.ravel())), 1, 9)


if __name__ == "__main__":
    print("Full Tests for spin1")
    unittest.main()
