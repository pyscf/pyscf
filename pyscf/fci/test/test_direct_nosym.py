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

nelec = (3,4)
norb = 8
h1e = numpy.random.random((norb,norb))
h2e = numpy.random.random((norb,norb,norb,norb))
h2e = h2e + h2e.transpose(2,3,0,1)
na = fci.cistring.num_strings(norb, nelec[0])
nb = fci.cistring.num_strings(norb, nelec[1])
ci0 = numpy.random.random((na,nb))

class KnownValues(unittest.TestCase):
    def test_contract(self):
        ci1ref = fci_slow.contract_1e(h1e, ci0, norb, nelec)
        ci1 = fci.direct_nosym.contract_1e(h1e, ci0, norb, nelec)
        self.assertTrue(numpy.allclose(ci1ref, ci1))

        ci1ref = fci_slow.contract_2e(h2e, ci0, norb, nelec)
        ci1 = fci.direct_nosym.contract_2e(h2e, ci0, norb, nelec)
        self.assertTrue(numpy.allclose(ci1ref, ci1))


if __name__ == "__main__":
    print("Full Tests for spin1")
    unittest.main()

