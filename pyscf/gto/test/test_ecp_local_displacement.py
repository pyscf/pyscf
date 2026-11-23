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


class KnownValues(unittest.TestCase):
    def test_displaced_tight_s(self):
        g, r = 0.15, 4.0
        for exponent in (14.0, 15.0, 30.0):
            with self.subTest(exponent=exponent):
                mol = gto.M(
                    atom=[('S', (0., 0., 0.)), ('H', (0., 0., r))],
                    unit='Bohr',
                    basis={'S': [[0, [1., 1.]]], 'H': [[0, [exponent, 1.]]]},
                    ecp={'S': [10, [[-1, [[], [], [[g, 1.]]]]]]},
                    spin=1,
                    verbose=0,
                )
                p = 2 * exponent
                exact = (p / (p + g))**1.5 * np.exp(-p * g * r**2 / (p + g))
                self.assertAlmostEqual(mol.intor('ECPscalar')[-1, -1], exact, 11)

    def test_two_displaced_s_centers(self):
        a, b, g = 28., 35., 0.15
        ra = np.array([0., .03, 4.])
        rb = np.array([.3, .1, 4.05])
        mol = gto.M(
            atom=[('S', (0., 0., 0.)), ('H1', ra), ('H2', rb)],
            unit='Bohr',
            basis={'S': [[0, [1., 1.]]], 'H1': [[0, [a, 1.]]], 'H2': [[0, [b, 1.]]]},
            ecp={'S': [10, [[-1, [[], [], [[g, -1.]]]]]]},
            verbose=0,
        )
        p = a + b
        center = (a * ra + b * rb) / p
        exact = -(4 * a * b)**.75 / (p + g)**1.5 * np.exp(
            -a * b / p * np.dot(ra - rb, ra - rb) - p * g / (p + g) * np.dot(center, center)
        )
        for name in ('ECPscalar_sph', 'ECPscalar_cart'):
            with self.subTest(integral=name):
                value = mol.intor_by_shell(name, (1, 2))[0, 0]
                self.assertAlmostEqual(value, exact, 11)


if __name__ == '__main__':
    unittest.main()
