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
from functools import reduce
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import fci

def setUpModule():
    global mol, m, norb, nelec
    mol = gto.M(atom='Be 0 0 0; H -1.1 0 .23; H 1.1 0 .23',
                symmetry='C2v', verbose=0)
    m = scf.RHF(mol)
    m.kernel()
    norb = m.mo_energy.size
    nelec = mol.nelectron

def tearDownModule():
    global mol, m
    del mol, m

class KnownValues(unittest.TestCase):
    def test_symm_spin0(self):
        fs = fci.FCI(mol, m.mo_coeff, singlet=True)
        fs.wfnsym = 'B2'
        fs.nroots = 3
        e, c = fs.kernel()
        self.assertAlmostEqual(e[0], -19.286003160337+mol.energy_nuc(), 9)
        self.assertAlmostEqual(e[1], -18.812177419921+mol.energy_nuc(), 9)
        self.assertAlmostEqual(e[2], -18.786684534678+mol.energy_nuc(), 9)
        self.assertAlmostEqual(fci.spin_op.spin_square0(c[0], norb, nelec)[0], 0, 7)
        self.assertAlmostEqual(fci.spin_op.spin_square0(c[1], norb, nelec)[0], 6, 7)
        self.assertAlmostEqual(fci.spin_op.spin_square0(c[2], norb, nelec)[0], 0, 7)

    def test_symm_spin1(self):
        fs = fci.FCI(mol, m.mo_coeff, singlet=False)
        fs.wfnsym = 'B2'
        fs.nroots = 2
        e, c = fs.kernel()
        self.assertAlmostEqual(e[0], -19.303845373762+mol.energy_nuc(), 9)
        self.assertAlmostEqual(e[1], -19.286003160337+mol.energy_nuc(), 9)
        self.assertAlmostEqual(fci.spin_op.spin_square0(c[0], norb, nelec)[0], 2, 9)
        self.assertAlmostEqual(fci.spin_op.spin_square0(c[1], norb, nelec)[0], 0, 9)


if __name__ == "__main__":
    print("Full Tests for init_guess")
    unittest.main()
