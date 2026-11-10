#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
from pyscf.scf import cphf, ucphf


def near_dependent_problem(seed=3, n=12):
    # Like the nuclear-displacement RHS of a DFT Hessian: the translations make
    # some combinations of the RHS vanish up to the integration-grid error.
    numpy.random.seed(seed)
    a = numpy.random.random((n, n)) * .1
    a = a + a.T
    base = numpy.random.randn(4, n)
    mix = numpy.random.randn(2, 4)
    h1 = numpy.vstack([base, mix.dot(base) + 1e-7 * numpy.random.randn(2, n)])
    ref = -numpy.linalg.solve(numpy.eye(n) + a, h1.T).T
    def fvind(mo1):
        return mo1.reshape(-1, n).dot(a)
    return fvind, h1, ref


class KnownValues(unittest.TestCase):
    def test_rhf_near_dependent_rhs(self):
        fvind, h1, ref = near_dependent_problem()
        mo_energy = numpy.array([-.5] * 3 + [.5] * 4)
        mo_occ = numpy.array([2.] * 3 + [0.] * 4)
        mo1 = cphf.solve(lambda x: fvind(x).reshape(-1, 4, 3), mo_energy, mo_occ,
                         h1.reshape(-1, 4, 3), tol=1e-8)[0]
        self.assertAlmostEqual(abs(mo1.reshape(ref.shape) - ref).max(), 0, 9)

    def test_uhf_near_dependent_rhs(self):
        fvind, h1, ref = near_dependent_problem()
        mo_energy = (numpy.array([-.5] * 3 + [.5] * 2), numpy.array([-.5] * 2 + [.5] * 3))
        mo_occ = (numpy.array([1.] * 3 + [0.] * 2), numpy.array([1.] * 2 + [0.] * 3))
        h1 = (h1[:, :6].reshape(-1, 2, 3), h1[:, 6:].reshape(-1, 3, 2))
        mo1a, mo1b = ucphf.solve(fvind, mo_energy, mo_occ, h1, tol=1e-8)[0]
        mo1 = numpy.hstack([mo1a.reshape(-1, 6), mo1b.reshape(-1, 6)])
        self.assertAlmostEqual(abs(mo1 - ref).max(), 0, 9)


if __name__ == '__main__':
    print('Full Tests for CPHF and UCPHF')
    unittest.main()
