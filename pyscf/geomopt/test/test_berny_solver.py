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
from pyscf import lib
from pyscf import gto, scf
from pyscf.geomopt import berny_solver

class KnownValues(unittest.TestCase):
    def test_as_pyscf_method(self):
        mol = gto.M(atom='''
            O  0.   0.       0.
            H  0.   -0.757   0.587
            H  0.   0.757    0.587
                    ''', symmetry=True, verbose=0)
        gs = scf.RHF(mol).nuc_grad_method().as_scanner()
        f = lambda mol: gs(mol)
        m = berny_solver.as_pyscf_method(mol, f)
        mol1 = berny_solver.optimize(m)
        self.assertAlmostEqual(lib.finger(mol1.atom_coords()),
                               3.039311839766823, 4)
        self.assertEqual(mol1.symmetry, 'C2v')

if __name__ == "__main__":
    print("Tests for berny_solver")
    unittest.main()

