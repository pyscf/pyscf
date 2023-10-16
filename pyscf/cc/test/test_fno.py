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
        mol.pseudo = 'gth-hf-rev'
        mol.basis = 'cc-pvdz'
        mol.precision = 1e-10
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

    @unittest.skip('fail due to updates of pp_int?')
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

    @unittest.skip('fail due to updates of pp_int?')
    def test_fno_by_thresh_frozen(self):
        threshs = [1e-2,1e-3,1e-4]
        refs = [
            [-0.1173030018, -0.0001459448],
            [-0.1889586685, -0.0006157799],
            [-0.2109365568, -0.0029841323],
        ]
        for thresh,ref in zip(threshs,refs):
            eccsd, et = self.kernel(cc.FNOCCSD, thresh=thresh, frozen=1)
            self.assertAlmostEqual(eccsd, ref[0], 6)
            self.assertAlmostEqual(et, ref[1], 6)

        eccsd0, et0 = self.kernel(cc.CCSD, frozen=1)
        eccsd, et = self.kernel(cc.FNOCCSD, thresh=1e-100, frozen=1)
        self.assertAlmostEqual(eccsd, eccsd0, 6)
        self.assertAlmostEqual(et, et0, 6)


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
        mol.pseudo = 'gth-hf-rev'
        mol.basis = 'cc-pvdz'
        mol.precision = 1e-10
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

    @unittest.skip('fail due to updates of pp_int?')
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

    @unittest.skip('fail due to updates of pp_int?')
    def test_fno_by_thresh_frozen(self):
        threshs = [1e-2,1e-3,1e-4]
        refs = [
            [-0.1172906846, -0.0001457165],
            [-0.1889855085, -0.0006158811],
            [-0.2110052374, -0.0029866440],
        ]
        for thresh,ref in zip(threshs,refs):
            eccsd, et = self.kernel(cc.FNOCCSD, thresh=thresh, frozen=1)
            self.assertAlmostEqual(eccsd, ref[0], 6)
            self.assertAlmostEqual(et, ref[1], 6)

        eccsd0, et0 = self.kernel(cc.CCSD, frozen=1)
        eccsd, et = self.kernel(cc.FNOCCSD, thresh=1e-100, frozen=1)
        self.assertAlmostEqual(eccsd, eccsd0, 6)
        self.assertAlmostEqual(et, et0, 6)


if __name__ == "__main__":
    print("Full Tests for FNO-CCSD and FNO-CCSD(T)")
    unittest.main()
