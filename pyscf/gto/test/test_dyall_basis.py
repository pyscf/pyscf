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

"""Integrity checks for the relativistic basis catalogue."""
import importlib
import unittest

import numpy as np

from pyscf import gto
from pyscf.data.elements import ELEMENTS


class KnownValues(unittest.TestCase):
    def test_dyall_catalogue(self):
        # Check every element in every modern Dyall family. No expensive
        # relativistic SCF calculation is needed.
        aliases = {name: path for name, path in gto.basis.ALIAS.items()
                   if isinstance(path, str) and path.startswith('dyall-basis.')}
        self.assertTrue(aliases)
        for name, path in aliases.items():
            module = importlib.import_module('pyscf.gto.basis.' + path)
            symbols = [symbol for symbol in ELEMENTS[1:]
                       if hasattr(module, symbol)]
            self.assertTrue(symbols, name)
            self.assertEqual(gto.basis.load(name, symbols[0]),
                             getattr(module, symbols[0]))
            for symbol in symbols:
                with self.subTest(basis=name, element=symbol):
                    shells = getattr(module, symbol)
                    self.assertTrue(shells)
                    angular = [shell[0] for shell in shells]
                    self.assertEqual(angular, sorted(angular))
                    self.assertEqual(angular[0], 0)
                    np.testing.assert_array_equal([shell[1] for shell in shells], 0)
                    primitives = np.array([shell[2] for shell in shells])
                    self.assertEqual(primitives.shape, (len(shells), 2))
                    self.assertTrue(np.isfinite(primitives).all())
                    self.assertTrue((primitives[:, 0] > 0).all())
                    np.testing.assert_array_equal(primitives[:, 1], 1.)

    @unittest.expectedFailure
    def test_v_3zp_independent_primitives(self):
        # The table currently repeats the f exponent 2.21852717. Keep this
        # regression visible pending verification of the original 3zp data.
        shells = gto.basis.load('dyall-3zp', 'V')
        pairs = [(shell[0], shell[2][0]) for shell in shells]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_normalized_atomic_orbitals(self):
        for name in ('dyall-v2z', 'dyall-cv3z', 'dyall-ae4z', 'dyall-av2z'):
            with self.subTest(basis=name):
                mol = gto.M(atom='He', basis=name, verbose=0)
                overlap = mol.intor('int1e_ovlp')
                np.testing.assert_allclose(overlap.diagonal(), 1., atol=2e-14,
                                           rtol=0)
                self.assertGreater(np.linalg.eigvalsh(overlap).min(), 0.)


if __name__ == '__main__':
    unittest.main()
