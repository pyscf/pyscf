#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
import numpy
from pyscf import fci
from pyscf import lib

def setUpModule():
    global h1e, eri, ci0, ci2, norb, nelec

    numpy.random.seed(12)
    norb = 6
    nelec = 3
    h1 = numpy.random.random((norb, norb)) + 1j * numpy.random.random((norb, norb))
    h2 = numpy.random.random((norb, norb, norb, norb)) + 1j * numpy.random.random((norb, norb, norb, norb))
    h1 = h1 + h1.T.conj()
    h2 = h2 + h2.transpose(2, 3, 0, 1)
    h2 = h2 + h2.transpose(1, 0, 3, 2).conj()
    h2 = h2 + h2.transpose(3, 2, 1, 0).conj()

    h1e = h1
    eri = h2

    numpy.random.seed(15)
    na = fci.cistring.num_strings(norb, nelec)
    ci0 = numpy.random.random((na, )) + 1j * numpy.random.random((na, ))
    ci0 = ci0 / numpy.linalg.norm(ci0)
    ci2 = numpy.random.random((na, )) + 1j * numpy.random.random((na, ))

def tearDownModule():
    global h1e, eri, ci0, ci2, norb, nelec
    del h1e, eri, ci0, ci2, norb, nelec

class KnownValues(unittest.TestCase):
    def test_contract(self):
        h2e = fci.fci_dhf_slow.absorb_h1e(h1e, eri, norb, nelec, 0.5)
        ci1 = fci.fci_dhf_slow.contract_2e(h2e, ci0, norb, nelec)
        ci3 = fci.fci_dhf_slow.contract_2e(h2e, ci2, norb, nelec)
        self.assertAlmostEqual(numpy.linalg.norm(ci1), 13.282579160597688, 8)
        self.assertAlmostEqual(numpy.linalg.norm(ci3), 45.62248606747163, 8)

    def test_kernel(self):
        e, c = fci.fci_dhf_slow.kernel(h1e, eri, norb, nelec)
        self.assertAlmostEqual(e, -19.32160357236596, 8)
        self.assertAlmostEqual(abs(lib.fp(c)), 0.44753770280929384, 8)

    def test_solver(self):
        sol = fci.fci_dhf_slow.FCI()
        e, c = sol.kernel(h1e, eri, norb, nelec)
        self.assertAlmostEqual(e, -19.32160357236596, 8)
        self.assertAlmostEqual(abs(lib.fp(c)), 0.44753770280929384, 8)

    def test_hdiag(self):
        hdiag = fci.fci_dhf_slow.make_hdiag(h1e, eri, norb, nelec)
        h2e = fci.fci_dhf_slow.absorb_h1e(h1e, eri, norb, nelec, 0.5)
        numpy.random.seed(15)
        for _ in range(5):
            idx = numpy.random.randint(len(ci0))
            ci = numpy.zeros_like(ci0)
            ci[idx] = 1
            ci_out = fci.fci_dhf_slow.contract_2e(h2e, ci, norb, nelec)
            self.assertAlmostEqual(ci_out[idx], hdiag[idx], 8)

    def test_rdm1(self):
        dm1 = fci.fci_dhf_slow.make_rdm1(ci0, norb, nelec)
        dm1_slow = fci.fci_dhf_slow.make_rdm1_slow(ci0, norb, nelec)
        self.assertAlmostEqual(numpy.linalg.norm(dm1 - dm1_slow), 0.0, 12)
        self.assertAlmostEqual(abs(lib.fp(dm1)), 0.23702307574649528, 8)

    def test_rdm12(self):
        dm1, dm2 = fci.fci_dhf_slow.make_rdm12(ci0, norb, nelec)
        self.assertAlmostEqual(abs(lib.fp(dm1)), 0.23702307574649528, 8)
        self.assertAlmostEqual(abs(lib.fp(dm2)), 2.8089637439079698, 8)
        self.assertAlmostEqual(abs(numpy.einsum('iijk->jk', dm2) / 2 - dm1).sum(), 0, 8)

if __name__ == "__main__":
    print("Full Tests for dhf-based fci")
    unittest.main()
