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
from pyscf.geomopt import geometric_solver

class KnownValues(unittest.TestCase):
    def test_optimize(self):
        mol = gto.M(atom='''
            O  0.   0.       0.
            H  0.   -0.757   0.587
            H  0.   0.757    0.587
                    ''',
                    symmetry=True, verbose=0)
        mol1 = geometric_solver.optimize(scf.RHF(mol))
        self.assertAlmostEqual(lib.finger(mol1.atom_coords()),
                               3.038506469458414, 4)
        self.assertEqual(mol1.symmetry, 'C2v')

    def test_optimize_high_cost(self):
        mol = gto.M(
        atom = [
        ['H',(  0.000000,    2.484212,    0.000000)],
        ['H',(  0.000000,   -2.484212,    0.000000)],
        ['H',(  2.151390,    1.242106,    0.000000)],
        ['H',( -2.151390,   -1.242106,    0.000000)],
        ['H',( -2.151390,    1.242106,    0.000000)],
        ['H',(  2.151390,   -1.242106,    0.000000)],
        ['C',(  0.000000,    1.396792,    0.000000)],
        ['C',(  0.000000,   -1.396792,    0.000000)],
        ['C',(  1.209657,    0.698396,    0.000000)],
        ['C',( -1.209657,   -0.698396,    0.000000)],
        ['C',( -1.209657,    0.698396,    0.000000)],
        ['C',(  1.209657,   -0.698396,    0.000000)], ],
            symmetry = True,
        )
        mf = scf.RHF(mol)
        sol = geometric_solver.GeometryOptimizer(mf)
        sol.max_cycle = 5
        sol.kernel()
        self.assertTrue(sol.converged)

if __name__ == "__main__":
    print("Tests for geometric_solver")
    unittest.main()

