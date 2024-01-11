#!/usr/bin/env python
# Copyright 2021 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy as np
from pyscf import __config__
from pyscf import gto, scf, cc
from pyscf.cc.ccsd_t import kernel as CCSD_T


class Water(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 4
        mol.output = '/dev/null'
        mol.atom = '''
        O          0.00000        0.00000        0.11779
        H          0.00000        0.75545       -0.47116
        H          0.00000       -0.75545       -0.47116
        '''
        mol.basis = 'cc-pvdz'
        mol.build()
        mf = scf.RHF(mol).run()
        cls.mol = mol
        cls.mf = mf
    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()
        del cls.mol, cls.mf

    def kernel(self, CC, **kwargs):
        mcc = CC(self.mf, **kwargs)
        eris = mcc.ao2mo()
        mcc.kernel(eris=eris)
        et = CCSD_T(mcc, eris=eris)
        return mcc.e_corr, et

    def test_fno_by_thresh(self):
        threshs = [1e-2,1e-3,1e-4]
        refs = [
            [-0.1180210293, -0.0001530894],
            [-0.1903456906, -0.0006322229],
            [-0.2130000748, -0.0030061317],
        ]
        for thresh,ref in zip(threshs,refs):
            eccsd, et = self.kernel(cc.FNOCCSD, thresh=thresh)
            self.assertAlmostEqual(eccsd, ref[0], 6)
            self.assertAlmostEqual(et, ref[1], 6)

        eccsd0, et0 = self.kernel(cc.CCSD)
        eccsd, et = self.kernel(cc.FNOCCSD, thresh=1e-100)
        self.assertAlmostEqual(eccsd, eccsd0, 6)
        self.assertAlmostEqual(et, et0, 6)

    def test_fno_by_thresh_frozen(self):
        threshs = [1e-2,1e-3,1e-4]
        refs = [
            [-0.0786641196, -0.0000408096],
            [-0.1284412378, -0.0007116076],
            [-0.1352476169, -0.0018534310],
        ]
        frozen = [1,20,23]
        for thresh,ref in zip(threshs,refs):
            eccsd, et = self.kernel(cc.FNOCCSD, thresh=thresh, frozen=frozen)
            self.assertAlmostEqual(eccsd, ref[0], 6)
            self.assertAlmostEqual(et, ref[1], 6)

        eccsd0, et0 = self.kernel(cc.CCSD, frozen=frozen)
        eccsd, et = self.kernel(cc.FNOCCSD, thresh=1e-100, frozen=frozen)
        self.assertAlmostEqual(eccsd, eccsd0, 6)
        self.assertAlmostEqual(et, et0, 6)

    def test_fno_by_pct_pvir_nvir_frozen(self):
        tasks = [
            [{'pct_occ': 0.99}, [-0.1325843694, -0.0014600579]],
            [{'pvir_act': 0.8}, [-0.1343480739, -0.0016905495]],
            [{'nvir_act': 14},  [-0.1343480739, -0.0016905495]],
        ]
        frozen = [1,20,23]
        for kwargs,ref in tasks:
            eccsd, et = self.kernel(cc.FNOCCSD, **kwargs, frozen=frozen)
            self.assertAlmostEqual(eccsd, ref[0], 6)
            self.assertAlmostEqual(et, ref[1], 6)


class Water_density_fitting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 4
        mol.output = '/dev/null'
        mol.atom = '''
        O          0.00000        0.00000        0.11779
        H          0.00000        0.75545       -0.47116
        H          0.00000       -0.75545       -0.47116
        '''
        mol.basis = 'cc-pvdz'
        mol.build()
        mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit').run()
        cls.mol = mol
        cls.mf = mf
    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()
        del cls.mol, cls.mf

    def kernel(self, CC, **kwargs):
        mcc = CC(self.mf, **kwargs)
        eris = mcc.ao2mo()
        mcc.kernel(eris=eris)
        et = CCSD_T(mcc, eris=eris)
        return mcc.e_corr, et

    def test_fno_by_thresh(self):
        threshs = [1e-2,1e-3,1e-4]
        refs = [
            [-0.1180086745, -0.0001528637],
            [-0.1903723005, -0.0006323478],
            [-0.2130687280, -0.0030086849],
        ]
        for thresh,ref in zip(threshs,refs):
            eccsd, et = self.kernel(cc.FNOCCSD, thresh=thresh)
            self.assertAlmostEqual(eccsd, ref[0], 6)
            self.assertAlmostEqual(et, ref[1], 6)

        eccsd0, et0 = self.kernel(cc.CCSD)
        eccsd, et = self.kernel(cc.FNOCCSD, thresh=1e-100)
        self.assertAlmostEqual(eccsd, eccsd0, 6)
        self.assertAlmostEqual(et, et0, 6)

    def test_fno_by_thresh_frozen(self):
        threshs = [1e-2,1e-3,1e-4]
        refs = [
            [-0.0786512769, -0.0000407462],
            [-0.1284642121, -0.0007119538],
            [-0.1352886314, -0.0018552359],
        ]
        frozen = [1,20,23]
        for thresh,ref in zip(threshs,refs):
            eccsd, et = self.kernel(cc.FNOCCSD, thresh=thresh, frozen=frozen)
            self.assertAlmostEqual(eccsd, ref[0], 6)
            self.assertAlmostEqual(et, ref[1], 6)

        eccsd0, et0 = self.kernel(cc.CCSD, frozen=frozen)
        eccsd, et = self.kernel(cc.FNOCCSD, thresh=1e-100, frozen=frozen)
        self.assertAlmostEqual(eccsd, eccsd0, 6)
        self.assertAlmostEqual(et, et0, 6)


class Water_UHF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 4
        mol.output = '/dev/null'
        mol.atom = '''
        O          0.00000        0.00000        0.11779
        H          0.00000        0.75545       -0.47116
        H          0.00000       -0.75545       -0.47116
        '''
        mol.basis = 'cc-pvdz'
        mol.charge = 1
        mol.spin = 1
        mol.build()
        mf = scf.UHF(mol).run()
        cls.mol = mol
        cls.mf = mf
    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()
        del cls.mol, cls.mf

    def kernel(self, CC, **kwargs):
        mcc = CC(self.mf, **kwargs)
        mcc.kernel()
        return mcc.e_corr

    def test_fno_by_thresh(self):
        threshs = [1e-3,1e-4]
        refs = [
            -0.1387658769,
            -0.1691956216,
        ]
        for thresh,ref in zip(threshs,refs):
            eccsd = self.kernel(cc.FNOCCSD, thresh=thresh)
            self.assertAlmostEqual(eccsd, ref, 6)

        eccsd0 = self.kernel(cc.CCSD)
        eccsd = self.kernel(cc.FNOCCSD, thresh=1e-100)
        self.assertAlmostEqual(eccsd, eccsd0, 6)

    def test_fno_by_thresh_frozen(self):
        threshs = [1e-3,1e-4]
        refs = [
            -0.0598508583,
            -0.0776356337,
        ]
        frozen = [[1,20,23],[2,21]]
        for thresh,ref in zip(threshs,refs):
            eccsd = self.kernel(cc.FNOCCSD, thresh=thresh, frozen=frozen)
            self.assertAlmostEqual(eccsd, ref, 6)

        eccsd0 = self.kernel(cc.CCSD, frozen=frozen)
        eccsd = self.kernel(cc.FNOCCSD, thresh=1e-100, frozen=frozen)
        self.assertAlmostEqual(eccsd, eccsd0, 6)

    def test_fno_by_pct_pvir_nvir_frozen(self):
        tasks = [
            [{'pct_occ': 0.99}, -0.0799112209],
            [{'pvir_act': 0.8}, -0.0812895797],
            [{'nvir_act': [14,16]},  -0.0812895797],
        ]
        frozen = [[1,20,23],[2,21]]
        for kwargs,ref in tasks:
            eccsd = self.kernel(cc.FNOCCSD, **kwargs, frozen=frozen)
            self.assertAlmostEqual(eccsd, ref, 6)


if __name__ == "__main__":
    print("Full Tests for FNO-CCSD and FNO-CCSD(T)")
    unittest.main()
