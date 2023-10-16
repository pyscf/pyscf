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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
from pyscf import gto
from pyscf import scf
from pyscf.scf import dhf

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.build(
        verbose = 0,
        atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ],
        basis = 'cc-pvdz',
    )

def tearDownModule():
    global mol
    del mol


class KnownValues(unittest.TestCase):
    def test_nr_rhf(self):
        rhf = scf.RHF(mol)
        rhf.conv_tol = 1e-11
        self.assertAlmostEqual(rhf.scf(), -76.026765673119627, 9)

    def test_nr_rohf(self):
        mol = gto.Mole()
        mol.build(
            verbose = 0,
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
            basis = 'cc-pvdz',
            charge = 1,
            spin = 1,
        )
        mf = scf.rohf.ROHF(mol)
        mf.conv_tol = 1e-11
        self.assertAlmostEqual(mf.scf(), -75.627354109594179, 9)

    def test_nr_uhf(self):
        uhf = scf.UHF(mol)
        uhf.conv_tol = 1e-11
        self.assertAlmostEqual(uhf.scf(), -76.026765673119598, 9)

    def test_nr_df_rhf(self):
        rhf = scf.density_fit(scf.RHF(mol), 'weigend')
        rhf.conv_tol = 1e-11
        self.assertAlmostEqual(rhf.scf(), -76.025936299701982, 9)

    def test_nr_df_rohf(self):
        mol = gto.Mole()
        mol.build(
            verbose = 0,
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
            basis = 'cc-pvdz',
            charge = 1,
            spin = 1,
        )
        mf = scf.density_fit(scf.ROHF(mol), 'weigend')
        mf.conv_tol = 1e-11
        self.assertAlmostEqual(mf.scf(), -75.626515724371899, 9)

    def test_nr_df_uhf(self):
        uhf = scf.density_fit(scf.UHF(mol), 'weigend')
        uhf.conv_tol = 1e-11
        self.assertAlmostEqual(uhf.scf(), -76.025936299702096, 9)

    def test_nr_rhf_no_mem(self):
        rhf = scf.RHF(mol)
        rhf.conv_tol = 1e-11
        rhf.max_memory = 0
        self.assertAlmostEqual(rhf.scf(), -76.026765673120565, 9)

    def test_nr_uhf_no_mem(self):
        uhf = scf.UHF(mol)
        uhf.conv_tol = 1e-11
        uhf.max_memory = 0
        self.assertAlmostEqual(uhf.scf(), -76.02676567312075, 9)

    def test_nr_rhf_no_direct(self):
        rhf = scf.RHF(mol)
        rhf.conv_tol = 1e-11
        rhf.max_memory = 0
        rhf.direct_scf = False
        self.assertAlmostEqual(rhf.scf(), -76.02676567311957, 9)

    def test_nr_uhf_no_direct(self):
        uhf = scf.UHF(mol)
        uhf.conv_tol = 1e-11
        uhf.max_memory = 0
        uhf.direct_scf = False
        self.assertAlmostEqual(uhf.scf(), -76.02676567311958, 9)

    def test_r_uhf_high_cost(self):
        uhf = dhf.UHF(mol)
        uhf.conv_tol_grad = 1e-5
        self.assertAlmostEqual(uhf.scf(), -76.081567907064198, 6)

    def test_r_rhf_high_cost(self):
        uhf = dhf.RHF(mol)
        uhf.conv_tol_grad = 1e-5
        self.assertAlmostEqual(uhf.scf(), -76.081567907064198, 6)

    def test_level_shift_uhf(self):
        uhf = scf.UHF(mol)
        uhf.level_shift = .2
        self.assertAlmostEqual(uhf.scf(), -76.026765673118078, 9)

    def test_nr_rhf_symm(self):
        mol1 = mol.copy()
        mol1.symmetry = 1
        mol1.build()
        rhf = scf.hf.RHF(mol1)
        rhf.conv_tol = 1e-11
        self.assertAlmostEqual(rhf.scf(), -76.026765673119655, 9)

    def test_nr_rohf_symm(self):
        mol = gto.Mole()
        mol.build(
            verbose = 0,
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
            basis = 'cc-pvdz',
            charge = 1,
            spin = 1,
            symmetry = True,
        )
        mf = scf.hf_symm.ROHF(mol)
        mf.conv_tol = 1e-11
        self.assertAlmostEqual(mf.scf(), -75.627354109593952, 9)

    def test_nr_uhf_symm(self):
        mol1 = mol.copy()
        mol1.symmetry = 1
        mol1.build()
        uhf = scf.uhf_symm.UHF(mol1)
        uhf.conv_tol = 1e-11
        self.assertAlmostEqual(uhf.scf(), -76.026765673119584, 9)


if __name__ == "__main__":
    print("Full Tests for H2O vdz")
    unittest.main()
