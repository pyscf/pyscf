#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

from pyscf import dft
from pyscf import gto
from pyscf import mp
from pyscf import scf
from pyscf.dbbsc import ecmd


def setUpModule():
    global mol_r, mf_r
    global mol_u, mf_u, mf_rohf
    mol_r = gto.M(atom='''
        H   0.000000000   0.000000000   0.457870600
        F   0.000000000   0.000000000  -0.457870600
    ''', basis='cc-pvtz', verbose=0)
    mf_r = scf.UHF(mol_r).density_fit(auxbasis='cc-pvtz-jkfit').run()

    mol_u = gto.M(atom='''
        N   0.000000000   0.000000000   0.000000000
        O   0.000000000   0.000000000   1.150770000
    ''', basis='cc-pvtz', spin=1, verbose=0)
    mf_u = scf.UHF(mol_u).density_fit(auxbasis='cc-pvtz-jkfit').run()
    mf_rohf = scf.ROHF(mol_u).density_fit(auxbasis='cc-pvtz-jkfit').run()


def tearDownModule():
    global mol_r, mf_r
    global mol_u, mf_u, mf_rohf
    del mol_r, mf_r
    del mol_u, mf_u, mf_rohf


def _small_grid(mol):
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build(with_non0tab=True)
    return grids


class KnownValues(unittest.TestCase):
    ecmd_module = ecmd

    @property
    def ecmd(self):
        return self.ecmd_module

    def test_functional_resolution(self):
        ecmd = self.ecmd
        self.assertEqual(ecmd._resolve_functional('pbe').xc, ',pbe')
        self.assertEqual(ecmd._resolve_functional(',VWN').xc, ',VWN')
        self.assertEqual(ecmd._resolve_functional('GGA_C_LYP').xc, 'GGA_C_LYP')
        with self.assertRaises(ValueError):
            ecmd._resolve_functional('')
        with self.assertRaises(ValueError):
            ecmd._resolve_functional('pbe,pbe')
        with self.assertRaises(ValueError):
            ecmd._resolve_functional(',GGA_X_PBE')
        with self.assertRaises(ValueError):
            ecmd._resolve_functional('MGGA_XC_B97M_V')

    def test_no_correction_for_one_electron(self):
        mol = gto.M(atom='H 0 0 0', basis='cc-pvdz', spin=1, verbose=0)
        mf = scf.UHF(mol).run()
        e = self.ecmd.energy(mf, grids=_small_grid(mol))
        self.assertAlmostEqual(e, 0.0, 12)


    def test_no_correction_for_triplet(self):
        mol = gto.M(atom='He 0 0 0', basis='cc-pvdz', spin=2, verbose=0)
        mf = scf.UHF(mol).run()
        e = self.ecmd.energy(mf, grids=_small_grid(mol))
        self.assertAlmostEqual(e, 0.0, 12)

    def test_constructor_is_configuration_api(self):
        global mf_r, mol_r
        ecmd = self.ecmd
        grids = _small_grid(mol_r)
        driver = ecmd.ECMD(
            mf_r,
            grids=grids,
            functional='pbe',
            ontop_model='ueg',
            aux_basis=None,
            max_memory=mf_r.max_memory,
        )
        self.assertIsNone(driver.aux_basis)
        driver.kernel()
        e_driver = driver.e_dbbsc
        e_direct = ecmd.energy(mf_r, grids=grids, functional='pbe', aux_basis=None, max_memory=mf_r.max_memory)
        self.assertAlmostEqual(e_driver, e_direct, 12)

    def test_rhf_pbe_ueg(self):
        global mf_r, mol_r
        e = self.ecmd.energy(mf_r, grids=_small_grid(mol_r), functional='pbe')
        # MolPro DF-HF: -100.05806074
        # MolPro: -0.07905824
        self.assertAlmostEqual(e, -0.07905823838478548, 9)

    def test_uhf_pbe_ueg(self):
        global mf_u, mol_u
        e = self.ecmd.energy(mf_u, grids=_small_grid(mol_u), functional='pbe')
        self.assertAlmostEqual(e, -0.11656406303713202, 9)

    def test_rohf_pbe_ueg(self):
        global mf_rohf, mol_u
        e = self.ecmd.energy(mf_rohf, grids=_small_grid(mol_u), functional='pbe')
        # MolPro: -0.11690384
        self.assertAlmostEqual(e, -0.11690393711094767, 9)

    def test_dfrhf_pbe_ueg(self):
        global mf_r, mol_r
        e = self.ecmd.energy(mf_r, grids=_small_grid(mol_r), functional='pbe', aux_basis='cc-pvtz-ri')
        # MRCC: -0.079074725636
        self.assertAlmostEqual(e, -0.07907471651739936, 9)

    def test_dfuhf_pbe_ueg(self):
        global mf_u, mol_u
        e = self.ecmd.energy(mf_u, grids=_small_grid(mol_u), functional='pbe', aux_basis='cc-pvtz-ri')
        # MRCC: -0.116596699840
        self.assertAlmostEqual(e, -0.11659677555544214, 9)

    def test_frozen_orbitals(self):
        global mf_r, mol_r
        ecmd = self.ecmd
        grids = _small_grid(mol_r)
        pt_r = mp.MP2(mf_r, frozen=1)
        e_pt = ecmd.energy(pt_r, grids=grids, functional='pbe')
        e_frozen = ecmd.energy(mf_r, grids=grids, functional='pbe', frozen=1)
        e_chemcore = ecmd.energy(mf_r, grids=grids, functional='pbe', frozen='chemcore')
        e_none = ecmd.energy(pt_r, grids=grids, functional='pbe', frozen='none')
        e_all = ecmd.energy(mf_r, grids=grids, functional='pbe')
        self.assertAlmostEqual(e_pt, -0.0399105841921325, 9)
        self.assertAlmostEqual(e_none, -0.07905823838478555, 9)
        self.assertAlmostEqual(e_frozen, e_pt, 12)
        self.assertAlmostEqual(e_chemcore, e_pt, 12)
        self.assertAlmostEqual(e_none, e_all, 12)


if __name__ == "__main__":
    print("Full Tests for DBBSC ecmd")
    unittest.main()
