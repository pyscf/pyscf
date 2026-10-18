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

"""Check every tabulated Lebedev rule without running a DFT calculation."""
import unittest

import numpy as np

from pyscf.dft import LebedevGrid


class KnownValues(unittest.TestCase):
    def test_polynomial_exactness(self):
        # The normalized integral of (u.r)**n on the unit sphere is
        # 1/(n+1) for even n and zero for odd n, for every unit vector u.
        # Oblique directions exercise mixed monomials as well as x**n.
        directions = np.array([[1., 0., 0.], [0., 1., 0.],
                               [1., 2., 3.], [-2., 3., 1.]])
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        for order, size in LebedevGrid.LEBEDEV_ORDER.items():
            if order == 0:
                continue
            with self.subTest(order=order, size=size):
                grid = LebedevGrid.MakeAngularGrid(size)
                self.assertEqual(grid.shape, (size, 4))
                xyz, weights = grid[:, :3], grid[:, 3]
                self.assertTrue(np.isfinite(grid).all())
                np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), 1.,
                                           rtol=0, atol=5e-15)
                self.assertAlmostEqual(weights.sum(), 1., places=14)
                projection = xyz @ directions.T
                power = np.ones_like(projection)
                for degree in range(order + 1):
                    expected = 0. if degree % 2 else 1. / (degree + 1)
                    integral = np.einsum('i,ij->j', weights, power)
                    np.testing.assert_allclose(integral, expected,
                                               rtol=3e-13, atol=3e-15,
                                               err_msg='degree %d' % degree)
                    power *= projection

    def test_single_point(self):
        for size in (0, 1):
            with self.subTest(size=size):
                np.testing.assert_array_equal(
                    LebedevGrid.MakeAngularGrid(size), [0., 0., 0., 1.])

    def test_unsupported_size(self):
        for size in (-1, 2, 7, 5811):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, 'Unsupported angular grids'):
                    LebedevGrid.MakeAngularGrid(size)


if __name__ == '__main__':
    unittest.main()
