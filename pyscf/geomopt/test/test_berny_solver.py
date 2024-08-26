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

try:
    from pyscf.geomopt import berny_solver
except ImportError:
    berny_solver = False

def setUpModule():
    global mol
    mol = gto.M(atom='''
        O  0.   0.       0.
        H  0.   -0.757   0.587
        H  0.   0.757    0.587
                ''', symmetry=True, verbose=0)

def tearDownModule():
    global mol
    del mol

@unittest.skipIf(not berny_solver, "pyberny library not found.")
class KnownValues(unittest.TestCase):
    def test_as_pyscf_method(self):
        gs = scf.RHF(mol).nuc_grad_method().as_scanner()
        f = lambda mol: gs(mol)
        m = berny_solver.as_pyscf_method(mol, f)
        mol1 = berny_solver.optimize(m)
        self.assertAlmostEqual(lib.fp(mol1.atom_coords()), 2.20003359484436, 4)
        self.assertEqual(mol1.symmetry, 'C2v')

    def test_optimize(self):
        conv_params = {
            'gradientmax': 0.1e-3,
            'gradientrms': 0.1e-3,
        }
        mf = scf.RHF(mol)
        mol_eq = mf.Gradients().optimizer(solver='berny').kernel(params=conv_params)
        self.assertAlmostEqual(lib.fp(mol_eq.atom_coords()), 2.19943732625887, 4)

if __name__ == "__main__":
    print("Tests for berny_solver")
    unittest.main()
