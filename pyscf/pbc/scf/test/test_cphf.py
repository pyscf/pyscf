# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

"""Periodic CPHF against explicitly solved small linear systems."""
import unittest

import numpy as np

from pyscf.pbc.scf import cphf


class KnownValues(unittest.TestCase):
    def setUp(self):
        self.energy = [np.array([-1.2, .4, -.7, .9]),
                       np.array([.6, -1., 1.1])]
        # Different occupied/virtual dimensions, with non-contiguous occupied
        # indices, catch incorrect slicing and offsets between k points.
        self.occ = [np.array([2, 0, 2, 0]), np.array([0, 2, 0])]

    def make_response(self, shapes, complex_):
        rng = np.random.default_rng(19)
        sizes = [np.prod(shape) for shape in shapes]
        offsets = np.cumsum([0] + sizes)
        size = offsets[-1]
        matrix = rng.normal(size=(size, size))
        if complex_:
            matrix = matrix + 1j * rng.normal(size=matrix.shape)
        matrix = .005 * (matrix + matrix.conj().T)

        def unpack(vector):
            return [vector[p:q].reshape(shape)
                    for p, q, shape in zip(offsets[:-1], offsets[1:], shapes)]

        def response(mo1):
            return unpack(matrix @ np.concatenate([x.ravel() for x in mo1]))

        h1 = rng.normal(size=size)
        if complex_:
            h1 = h1 + 1j * rng.normal(size=size)
        return matrix, unpack, response, h1

    def test_field_independent(self):
        for complex_ in (False, True):
            with self.subTest(complex=complex_):
                shapes = [(2, 2, 2), (2, 2, 1)]
                matrix, unpack, response, h1 = self.make_response(shapes, complex_)
                gaps = []
                for energy, occ, shape in zip(self.energy, self.occ, shapes):
                    gap = energy[occ == 0, None] - energy[occ > 0]
                    gaps.append(np.broadcast_to(gap, shape).ravel())
                operator = matrix + np.diag(np.concatenate(gaps))
                expected = np.linalg.solve(operator, -h1)
                mo1, e1 = cphf.solve(response, self.energy, self.occ,
                                     unpack(h1.copy()), tol=1e-12)
                result = np.concatenate([x.ravel() for x in mo1])
                # Krylov also stops at its default linear-dependence threshold
                # (1e-13 for the squared residual), irrespective of tol.
                np.testing.assert_allclose(result, expected, atol=5e-7, rtol=0)
                np.testing.assert_allclose(operator @ result, -h1, atol=5e-7, rtol=0)
                self.assertIsNone(e1)
                self.assertEqual([x.shape for x in mo1], shapes)

    def test_field_dependent(self):
        for complex_ in (False, True):
            with self.subTest(complex=complex_):
                shapes = [(2, 4, 2), (2, 3, 1)]
                matrix, unpack, response, h1 = self.make_response(shapes, complex_)
                s1 = .03 * h1[::-1].copy()
                fixed, gaps, occupied_energies = [], [], []
                for energy, occ, shape in zip(self.energy, self.occ, shapes):
                    fixed.append(np.broadcast_to((occ > 0)[None, :, None],
                                                 shape).ravel())
                    gaps.append(np.broadcast_to(energy[:, None] - energy[occ > 0],
                                                shape).ravel())
                    occupied_energies.append(np.broadcast_to(energy[occ > 0],
                                                             shape).ravel())
                fixed = np.concatenate(fixed)
                gaps = np.concatenate(gaps)
                hs = h1 - s1 * np.concatenate(occupied_energies)
                expected = -.5 * s1
                free = ~fixed
                operator = matrix + np.diag(gaps)
                rhs = -hs[free] - matrix[np.ix_(free, fixed)] @ expected[fixed]
                expected[free] = np.linalg.solve(operator[np.ix_(free, free)], rhs)
                mo1, e1 = cphf.solve(response, self.energy, self.occ,
                                     unpack(h1.copy()), unpack(s1.copy()), tol=1e-12)
                result = np.concatenate([x.ravel() for x in mo1])
                np.testing.assert_allclose(result, expected, atol=5e-7, rtol=0)
                residual = hs + operator @ result
                np.testing.assert_allclose(residual[free], 0, atol=5e-7)
                for actual, full, occ in zip(e1, unpack(residual), self.occ):
                    np.testing.assert_allclose(actual, full[:, occ > 0],
                                               atol=5e-7, rtol=0)

    def test_zero_response(self):
        h1 = [np.ones((2, 2)), np.ones((2, 1))]
        mo1, e1 = cphf.kernel(lambda x: [np.zeros_like(a) for a in x],
                              self.energy, self.occ, h1, hermi=True)
        for actual, energy, occ in zip(mo1, self.energy, self.occ):
            expected = -1 / (energy[occ == 0, None] - energy[occ > 0])
            np.testing.assert_allclose(actual, expected, atol=1e-14)
        self.assertIsNone(e1)


if __name__ == '__main__':
    unittest.main()
