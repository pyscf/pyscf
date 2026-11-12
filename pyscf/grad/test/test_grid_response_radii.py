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

"""Grid response with atomic-size adjustment on ECP and ghost atoms.

The Becke partition looks atomic radii up by proton number. On an ECP atom
mol.atom_charges() is Z - n_core (Au: 19, potassium's radius) and on a ghost
atom it is 0, so a response that used atom_charges() differentiated a
different partition from the one the energy is integrated on.
"""
import unittest
from unittest import mock

import numpy as np

from pyscf import gto
from pyscf.dft import gen_grid, radi
from pyscf.grad import rks


def ecp_mol():
    # O, not a second H: with Becke radii the Au-H size adjustment saturates
    # at the +-0.5 clamp for both lookups, so only the Au-O pair tells them apart.
    return gto.M(atom='Au 0 0 0; H 0 0 2.9; O 2.1 .8 -1.4', unit='Bohr',
                 basis={'Au': 'def2-svp', 'H': 'sto-3g', 'O': 'sto-3g'},
                 ecp={'Au': 'def2-svp'}, verbose=0)


def ghost_mol():
    return gto.M(atom='O 0 0 0; ghost-O 0 0 2.6; H 1.7 .4 -.9', unit='Bohr',
                 basis='sto-3g', spin=1, verbose=0)


class KnownValues(unittest.TestCase):
    def check_partition_response(self, mol, scheme, adjust):
        atoms = mol.atom_coords()
        rng = np.random.default_rng(7)
        offsets = rng.uniform(-1.5, 1.5, (12, 3))
        volumes = np.linspace(1., 2., len(offsets))
        atom_grids = {mol.atom_symbol(ia): (offsets, volumes)
                      for ia in range(mol.natm)}
        grids = gen_grid.Grids(mol)
        grids.becke_scheme = scheme
        grids.radii_adjust = adjust
        with mock.patch.object(grids, 'gen_atomic_grids', return_value=atom_grids):
            response = list(rks.grids_response_cc(grids))

        def displaced_weights(ia, axis, displacement):
            displaced = atoms.copy()
            displaced[ia, axis] += displacement
            moved = mol.set_geom_(displaced, unit='Bohr', inplace=False)
            _, weights = gen_grid.get_partition(
                moved, atom_grids, adjust, radi.BRAGG_RADII,
                scheme, concat=False)
            return np.asarray(weights)

        step = 1e-5
        finite_difference = np.empty((mol.natm, mol.natm, 3, len(offsets)))
        for ia, axis in np.ndindex(mol.natm, 3):
            plus = displaced_weights(ia, axis, step)
            minus = displaced_weights(ia, axis, -step)
            finite_difference[:, ia, axis] = (plus - minus) / (2 * step)

        for owner, (_, _, dw) in enumerate(response):
            with self.subTest(owner=owner):
                np.testing.assert_allclose(dw, finite_difference[owner],
                                           atol=1e-7, rtol=1e-6)

    def test_ecp_becke_treutler_radii(self):
        self.check_partition_response(
            ecp_mol(), gen_grid.original_becke, radi.treutler_atomic_radii_adjust)

    def test_ecp_becke_becke_radii(self):
        self.check_partition_response(
            ecp_mol(), gen_grid.original_becke, radi.becke_atomic_radii_adjust)

    def test_ecp_stratmann_treutler_radii(self):
        self.check_partition_response(
            ecp_mol(), gen_grid.stratmann, radi.treutler_atomic_radii_adjust)

    def test_ghost_becke_treutler_radii(self):
        self.check_partition_response(
            ghost_mol(), gen_grid.original_becke, radi.treutler_atomic_radii_adjust)

    def test_ghost_stratmann_becke_radii(self):
        self.check_partition_response(
            ghost_mol(), gen_grid.stratmann, radi.becke_atomic_radii_adjust)


if __name__ == '__main__':
    unittest.main()
