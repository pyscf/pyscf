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
import numpy
from pyscf import gto, scf, lib
from pyscf import grad

mol = gto.Mole()
mol.verbose = 5
mol.output = '/dev/null'
mol.atom.extend([
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ])
mol.basis = '6-31g'
mol.charge = -1
mol.spin = 1
mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


def make_mol(atom_id, coords):
    mol1 = mol.copy()
    mol1.atom[atom_id] = [mol1.atom[atom_id][0], coords]
    mol1.build(0, 0)
    return mol1

class KnownValues(unittest.TestCase):
    def test_finite_diff_uhf_grad(self):
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-14
        e0 = mf.kernel()
        g = grad.UHF(mf).kernel()
        mf_scanner = mf.as_scanner()

        e1 = mf_scanner(make_mol(0, (0, 0, 1e-4)))
        self.assertAlmostEqual(g[0,2], (e1-e0)/1e-4*lib.param.BOHR, 4)

        e1 = mf_scanner(make_mol(0, (0, 1e-5, 0)))
        self.assertAlmostEqual(g[0,1], (e1-e0)/1e-5*lib.param.BOHR, 4)

        e1 = mf_scanner(make_mol(1, (0. , -0.7571 , 0.587)))
        self.assertAlmostEqual(g[1,1], (e0-e1)/1e-4*lib.param.BOHR, 4)

        e1 = mf_scanner(make_mol(1, (0. , -0.757 , 0.5871)))
        self.assertAlmostEqual(g[1,2], (e1-e0)/1e-4*lib.param.BOHR, 4)


if __name__ == "__main__":
    print("Full Tests for UHF Gradients")
    unittest.main()

