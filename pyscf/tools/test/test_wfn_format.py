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

"""Validate exported wavefunctions by reconstructing their orbital values."""
import io
import itertools
import unittest

import numpy as np

from pyscf import gto
from pyscf.tools import wfn_format


class KnownValues(unittest.TestCase):
    def test_orbitals(self):
        mol = gto.M(atom='He 0 0 0; He 0 0 1.2', verbose=0,
                    basis={'He': [[0, [2., .3], [.5, .7]], [1, [.8, 1.]],
                                  [2, [.6, 1.]]]})
        rng = np.random.default_rng(9)
        coeff = rng.normal(size=(mol.nao, 3))
        coords = rng.normal(size=(7, 3))
        energy = [-.7, -.2, .5]
        occ = [2., 1., 0.]
        # WFN Cartesian type ordering, through d functions.
        powers = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                           [2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0],
                           [1, 0, 1], [0, 1, 1]])
        for energies, occupations in ((None, None), (None, occ), (energy, occ)):
            with self.subTest(energy=energies, occ=occupations):
                output = io.StringIO()
                wfn_format.write_mo(output, mol, coeff, energies, occupations)
                lines = output.getvalue().splitlines()
                self.assertEqual(lines[0], 'From PySCF')
                self.assertIn('3 MOL ORBITALS', lines[1])
                centers, types, exponents, mos = [], [], [], []
                for line in lines[2 + mol.natm:]:
                    if line.startswith('CENTRE ASSIGNMENTS'):
                        centers.extend(map(int, line.split()[2:]))
                    elif line.startswith('TYPE ASSIGNMENTS'):
                        types.extend(map(int, line.split()[2:]))
                    elif line.startswith('EXPONENTS'):
                        exponents.extend(map(float, line.split()[1:]))
                    elif line.startswith(('CANMO', 'MO ')):
                        mos.append([])
                        if occupations is not None:
                            self.assertAlmostEqual(float(line.split('OCC NO =')[1].split()[0]),
                                                   occupations[len(mos) - 1])
                        if energies is not None:
                            self.assertAlmostEqual(float(line.split('ORB. ENERGY =')[1]),
                                                   energies[len(mos) - 1])
                    elif line == 'END DATA':
                        break
                    else:
                        mos[-1].extend(map(float, line.split()))
                centers = np.array(centers) - 1
                types = np.array(types) - 1
                self.assertEqual(len(centers), len(exponents))
                self.assertEqual(np.array(mos).shape, (3, len(centers)))
                dr = coords[:, None] - mol.atom_coords()[centers]
                radial = np.exp(-np.sum(dr**2, axis=-1) * exponents)
                primitives = radial * np.prod(dr**powers[types], axis=-1)
                reconstructed = primitives @ np.array(mos).T
                expected = mol.eval_gto('GTOval_sph', coords) @ coeff
                # WFN stores coefficients and exponents with 8 significant
                # decimal places, so the text round trip limits accuracy.
                np.testing.assert_allclose(reconstructed, expected, atol=2e-8, rtol=2e-8)
        mol.cart = True
        with self.assertRaises(NotImplementedError):
            wfn_format.write_mo(io.StringIO(), mol, coeff)

    def test_ci_determinants(self):
        ci = np.arange(-4., 5.).reshape(3, 3) / 10
        for nelec in (3, np.int64(3), (2, 1)):
            with self.subTest(nelec=nelec):
                output = io.StringIO()
                wfn_format.write_ci(output, ci, 3, nelec, ncore=2)
                lines = output.getvalue().splitlines()
                self.assertEqual(list(map(int, lines[1].split())), [3, 9, 2, 3])
                alpha = list(itertools.combinations(range(3, 6), 2))
                beta = list(range(3, 6))
                actual = {}
                weights = []
                for line in lines[3:]:
                    fields = line.split()
                    coefficient = float(fields[0])
                    orbitals = tuple(map(int, fields[1:]))
                    actual[orbitals] = coefficient
                    weights.append(abs(coefficient))
                expected = {a + (-b,): ci[i, j] for i, a in enumerate(alpha)
                            for j, b in enumerate(beta)}
                self.assertEqual(actual, expected)
                self.assertEqual(weights, sorted(weights, reverse=True))


if __name__ == '__main__':
    unittest.main()
