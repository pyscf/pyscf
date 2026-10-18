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

"""Geometric invariants of the ring and fullerene generators."""
import unittest

import numpy as np

from pyscf.tools import c60struct, ring


class KnownValues(unittest.TestCase):
    def test_regular_rings(self):
        for size in (3, 4, 6, 12):
            for bond in (.7, 1.4):
                with self.subTest(size=size, bond=bond):
                    xyz = np.array(ring.make(size, bond))
                    self.assertEqual(xyz.shape, (size, 3))
                    np.testing.assert_allclose(xyz.mean(axis=0), 0, atol=1e-14)
                    np.testing.assert_array_equal(xyz[:, 2], 0)
                    edges = xyz - np.roll(xyz, 1, axis=0)
                    np.testing.assert_allclose(np.linalg.norm(edges, axis=1), bond,
                                               atol=1e-14)
                    radii = np.linalg.norm(xyz, axis=1)
                    np.testing.assert_allclose(radii, radii[0], atol=1e-14)

    def test_rotations(self):
        for rotate, axis in ((c60struct.rotmaty, [0, 1, 0]),
                             (c60struct.rotmatz, [0, 0, 1])):
            for angle in (0., .3, np.pi, 2 * np.pi):
                with self.subTest(axis=axis, angle=angle):
                    matrix = rotate(angle)
                    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-14)
                    np.testing.assert_allclose(matrix @ axis, axis, atol=1e-14)
                    self.assertAlmostEqual(np.linalg.det(matrix), 1., places=14)
                    np.testing.assert_allclose(matrix @ rotate(-angle), np.eye(3),
                                               atol=1e-14)
        self.assertAlmostEqual(c60struct.r2edge(np.pi / 3, 2.), 2., places=14)

    def test_polyhedra(self):
        for make, size, neighbors in ((c60struct.make12, 12, 5),
                                      (c60struct.make20, 20, 3)):
            for bond in (1., 1.4):
                with self.subTest(size=size, bond=bond):
                    xyz = make(bond)
                    self.assertEqual(xyz.shape, (size, 3))
                    np.testing.assert_allclose(xyz.mean(axis=0), 0, atol=1e-14)
                    radii = np.linalg.norm(xyz, axis=1)
                    np.testing.assert_allclose(radii, radii[0], atol=1e-14)
                    distance = np.linalg.norm(xyz[:, None] - xyz, axis=-1)
                    np.fill_diagonal(distance, np.inf)
                    np.testing.assert_allclose(np.sort(distance, axis=1)[:, :neighbors],
                                               bond, atol=1e-14)
                    self.assertTrue((np.sort(distance, axis=1)[:, neighbors] > bond).all())

    def test_c60_bonds(self):
        for b5, b6 in ((1.46, 1.38), (1., 1.)):
            with self.subTest(b5=b5, b6=b6):
                xyz = c60struct.make60(b5, b6)
                self.assertEqual(xyz.shape, (60, 3))
                np.testing.assert_allclose(xyz.mean(axis=0), 0, atol=1e-14)
                # Every carbon has two pentagon edges and one hexagon-only
                # edge; inversion maps each vertex onto another vertex.
                distance = np.linalg.norm(xyz[:, None] - xyz, axis=-1)
                np.fill_diagonal(distance, np.inf)
                bonds = np.sort(distance, axis=1)[:, :3]
                np.testing.assert_allclose(bonds, np.tile(sorted([b5, b5, b6]), (60, 1)),
                                           atol=1e-14)
                inverted_distance = np.linalg.norm(xyz[:, None] + xyz, axis=-1)
                np.testing.assert_allclose(inverted_distance.min(axis=1), 0, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
