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

import os
import unittest
from functools import reduce
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf import lib

h1, h2 = numpy.load(os.path.realpath(os.path.join(__file__, '..', 'spin_op_hamiltonian.npy')))
h1 = lib.unpack_tril(h1)

norb = 10
nelec = (5,5)
na = fci.cistring.num_strings(norb, nelec[0])
c0 = numpy.zeros((na,na))
c0[0,0] = 1
c0[-1,-1] = 1e-4
e0, ci0 = fci.direct_spin0.kernel(h1, h2, norb, nelec, ci0=c0)


def tearDownModule():
    global h1, h2, c0
    del h1, h2, c0

class KnownValues(unittest.TestCase):
    def test_spin_squre(self):
        ss = fci.spin_op.spin_square(ci0, norb, nelec)
        self.assertAlmostEqual(ss[0], 6, 9)
        ss = fci.spin_op.spin_square0(ci0, norb, nelec)
        self.assertAlmostEqual(ss[0], 6, 9)

        numpy.random.seed(1)
        u,w,v = numpy.linalg.svd(numpy.random.random((norb,6)))
        u = u[:,:6]
        h1a = h1[:6,:6]
        h1b = reduce(numpy.dot, (v.T, h1a, v))
        h2aa = ao2mo.restore(1, h2, norb)[:6,:6,:6,:6]
        h2ab = lib.einsum('klpq,pi,qj->klij', h2aa, v, v)
        h2bb = lib.einsum('pqkl,pi,qj->ijkl', h2ab, v, v)
        e1, ci1 = fci.direct_uhf.kernel((h1a,h1b), (h2aa,h2ab,h2bb), 6, (3,2))
        ss = fci.spin_op.spin_square(ci1, 6, (3,2), mo_coeff=(numpy.eye(6),v))[0]
        self.assertAlmostEqual(ss, 3.75, 8)

    def test_contract_ss(self):
        self.assertAlmostEqual(e0, -25.4538751043, 9)
        nelec = (6,4)
        na = fci.cistring.num_strings(norb, nelec[0])
        nb = fci.cistring.num_strings(norb, nelec[1])
        c0 = numpy.zeros((na,nb))
        c0[0,0] = 1
        bak0 = fci.direct_spin0.contract_2e
        bak1 = fci.direct_spin1.contract_2e
        fci.addons.fix_spin_(fci.direct_spin1)
        e, ci0 = fci.direct_spin1.kernel(h1, h2, norb, nelec, ci0=c0)
        fci.direct_spin0.contract_2e = bak0
        fci.direct_spin1.contract_2e = bak1
        self.assertAlmostEqual(e, -25.4437866823, 9)
        self.assertAlmostEqual(fci.spin_op.spin_square0(ci0, norb, nelec)[0], 2, 9)

        # Note: OMP parallelization may change the results due to round-off
        # instability.
        nelec = (5,5)
        fci.addons.fix_spin_(fci.direct_spin0)
        na = fci.cistring.num_strings(norb, nelec[0])
        c0 = numpy.zeros((na,na))
        c0[0,0] = 1
        c0[-1,-1] = 1e-4
        e, ci0 = fci.direct_spin0.kernel(h1, h2, norb, nelec, ci0=c0)
        fci.direct_spin0.contract_2e = bak0
        fci.direct_spin1.contract_2e = bak1
        self.assertAlmostEqual(e, -25.4095560762, 7)
        self.assertAlmostEqual(fci.spin_op.spin_square0(ci0, norb, nelec)[0], 0, 7)


if __name__ == "__main__":
    print("Full Tests for fci.spin_op")
    unittest.main()




