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
import numpy as np
from pyscf import gto
from pyscf.geomopt import addons

class KnownValues(unittest.TestCase):
    def test_symmetrize(self):
        mol = gto.M(atom='''
            O  0.   0.       0.
            H  0.   -0.757   0.587
            H  0.   0.757    0.587
                    ''', symmetry=True)
        coords = mol.atom_coords()
        sym_coords = addons.symmetrize(mol, coords)
        self.assertAlmostEqual(abs(coords-sym_coords).max(), 0, 9)

    def test_harmonic_callback(self):
        mol = gto.M(atom='He 0 0 0', basis='sto-3g', verbose=0)
        def potential(molecule):
            coords = molecule.atom_coords()
            return .5*np.sum(coords**2), coords.copy()
        method = addons.as_pyscf_method(mol, potential)
        scanner = method.Gradients().as_scanner()
        moved = mol.set_geom_(np.array([[.2, -.3, .4]]), unit='Bohr', inplace=False)
        energy, force = scanner(moved)
        self.assertAlmostEqual(energy, .145, places=14)
        np.testing.assert_allclose(force, [[.2, -.3, .4]], atol=1e-14)
        self.assertEqual(scanner.e_tot, energy)
        self.assertTrue(scanner.converged)
        self.assertEqual(scanner(mol)[0], 0)

if __name__ == "__main__":
    print("Tests for addons")
    unittest.main()
