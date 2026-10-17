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

"""Finite differences of LKO grid weights, including saturated partitions."""

import unittest
from unittest import mock
import numpy as np
from pyscf import gto
from pyscf.dft import gen_grid, radi
from pyscf.grad import rks


class KnownValues(unittest.TestCase):
    def setUp(self):
        # Unequal radii make the radii-adjustment cases nontrivial.
        self.mol = gto.M(atom='He 0 0 0; H 1.3 .2 -.4', unit='Bohr',
                         basis='sto-3g', spin=1, verbose=0)
        atoms = self.mol.atom_coords()
        # Include both nuclei, distant points, and ordinary points.
        self.coords = np.array([atoms[0], [-8., 0., 0.], [9., 0., 0.],
                                [.3, .1, .2], [.9, -.2, .1], atoms[1]])
        self.owners = np.array([0, 0, 1, 0, 1, 1], dtype=np.int32)
        self.volumes = np.linspace(1., 2., len(self.coords))

    def weights(self, mol, coords, adjust=None):
        partition_fn = rks._get_lko_partition_functions(
            mol, self.owners, adjust, radi.BRAGG_RADII)[0]
        partition = partition_fn(coords)
        owner_partition = partition[self.owners, np.arange(len(coords))]
        return self.volumes * owner_partition / partition.sum(axis=0)

    def response(self, adjust=None):
        weights = self.weights(self.mol, self.coords, adjust)
        return rks.get_dw_partition_sorted(
            self.mol, self.owners, self.coords, weights,
            adjust, radi.BRAGG_RADII)

    def displaced_weights(self, atom, axis, step, adjust):
        atoms = self.mol.atom_coords()
        atoms[atom, axis] += step
        moved = self.mol.set_geom_(atoms, unit='Bohr', inplace=False)
        # Grid points follow their owner atom during the displacement.
        coords = self.coords.copy()
        coords[self.owners == atom, axis] += step
        return self.weights(moved, coords, adjust)

    def check_finite_difference(self, adjust=None):
        derivative = self.response(adjust)
        step = 1e-5
        for atom, axis in np.ndindex(self.mol.natm, 3):
            w_plus = self.displaced_weights(atom, axis, step, adjust)
            w_minus = self.displaced_weights(atom, axis, -step, adjust)
            finite_difference = (w_plus - w_minus) / (2 * step)
            # LKO orders the derivative axes as (direction, atom, point).
            np.testing.assert_allclose(
                derivative[axis, atom], finite_difference,
                atol=1e-7, rtol=1e-6, err_msg=f'atom={atom}, axis={axis}')

    def test_saturated_response(self):
        weights = self.weights(self.mol, self.coords)
        derivative = self.response()
        # The first and last points sit on their owner nuclei.
        np.testing.assert_allclose(weights[[0, -1]], self.volumes[[0, -1]],
                                   atol=1e-14)
        np.testing.assert_allclose(derivative[:, :, [0, -1]], 0., atol=1e-14)
        self.assertTrue(np.isfinite(derivative).all())

    def test_translation_invariance(self):
        derivative = self.response()
        # Translating all atoms together must leave the weights unchanged.
        np.testing.assert_allclose(derivative.sum(axis=1), 0., atol=1e-12)

    def test_finite_difference(self):
        self.check_finite_difference()

    def test_finite_difference_treutler_radii(self):
        self.check_finite_difference(radi.treutler_atomic_radii_adjust)

    def test_finite_difference_becke_radii(self):
        self.check_finite_difference(radi.becke_atomic_radii_adjust)


if __name__ == '__main__':
    unittest.main()
