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

"""Grid response at nuclei and near saturated Becke/Stratmann partitions."""
import unittest
from unittest import mock

import numpy as np

from pyscf import gto
from pyscf.dft import gen_grid, radi
from pyscf.grad import rks


class KnownValues(unittest.TestCase):
    def check_saturated_partition_response(self, scheme, adjust):
        mol = gto.M(atom='He 0 0 0; H 1.3 .2 -.4', unit='Bohr',
                    basis='sto-3g', spin=1, verbose=0)
        atoms = mol.atom_coords()
        # Points follow their owner atom, including one exactly at its nucleus
        # and one close enough that a switching weight rounds to zero.
        offsets = np.array([[0., 0., 0.], [1e-5, 2e-5, -1e-5],
                            [.3, .1, .2], [9., 0., 0.]])
        volumes = np.linspace(1., 2., len(offsets))
        atom_grids = {symbol: (offsets, volumes) for symbol in ('He', 'H')}
        grids = gen_grid.Grids(mol)
        grids.becke_scheme = scheme
        grids.radii_adjust = adjust
        with mock.patch.object(grids, 'gen_atomic_grids', return_value=atom_grids):
            with np.errstate(divide='raise', invalid='raise', over='raise'):
                response = list(rks.grids_response_cc(grids))

        def displaced_weights(ia, axis, displacement):
            displaced = atoms.copy()
            displaced[ia, axis] += displacement
            moved = mol.set_geom_(displaced, unit='Bohr', inplace=False)
            _, weights = gen_grid.get_partition(
                moved, atom_grids, adjust, radi.BRAGG_RADII,
                scheme, concat=False)
            return np.asarray(weights)

        # Evaluate all owner grids together for each displaced coordinate.
        step = 1e-5
        finite_difference = np.empty((mol.natm, mol.natm, 3, len(offsets)))
        for ia, axis in np.ndindex(mol.natm, 3):
            plus = displaced_weights(ia, axis, step)
            minus = displaced_weights(ia, axis, -step)
            finite_difference[:, ia, axis] = (plus - minus) / (2 * step)

        for owner, (_, weight, dw) in enumerate(response):
            with self.subTest(owner=owner):
                self.assertTrue(np.isfinite(dw).all())
                np.testing.assert_allclose(weight[0], volumes[0], atol=1e-14)
                np.testing.assert_allclose(dw[:, :, 0], 0., atol=1e-14)
                np.testing.assert_allclose(dw.sum(axis=0), 0., atol=1e-12)
                np.testing.assert_allclose(dw, finite_difference[owner],
                                           atol=1e-7, rtol=1e-6)

    def test_becke_without_radii_adjustment(self):
        self.check_saturated_partition_response(gen_grid.original_becke, None)

    def test_becke_treutler_radii(self):
        self.check_saturated_partition_response(
            gen_grid.original_becke, radi.treutler_atomic_radii_adjust)

    def test_becke_becke_radii(self):
        self.check_saturated_partition_response(
            gen_grid.original_becke, radi.becke_atomic_radii_adjust)

    def test_stratmann_without_radii_adjustment(self):
        self.check_saturated_partition_response(gen_grid.stratmann, None)

    def test_stratmann_treutler_radii(self):
        self.check_saturated_partition_response(
            gen_grid.stratmann, radi.treutler_atomic_radii_adjust)

    def test_stratmann_becke_radii(self):
        self.check_saturated_partition_response(
            gen_grid.stratmann, radi.becke_atomic_radii_adjust)


if __name__ == '__main__':
    unittest.main()
