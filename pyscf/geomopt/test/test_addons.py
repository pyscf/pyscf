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

if __name__ == "__main__":
    print("Tests for addons")
    unittest.main()
