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
from pyscf.cc.uccsd_t import kernel as CCSD_T


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
        mf = scf.UHF(mol).run()
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
            [-0.1200179470,-0.0001506723],
            [-0.1925851371,-0.0006428496],
            [-0.2149635715,-0.0030270215],
        ]
        for thresh,ref in zip(threshs,refs):
            eccsd, et = self.kernel(cc.FNOCCSD, thresh=thresh)
            # print('[%s],' % (','.join([f'{x:.10f}' for x in [eccsd,et]])))
            self.assertAlmostEqual(eccsd, ref[0], 6)
            self.assertAlmostEqual(et, ref[1], 6)

        eccsd0, et0 = self.kernel(cc.CCSD)
        eccsd, et = self.kernel(cc.FNOCCSD, thresh=1e-100)
        self.assertAlmostEqual(eccsd, eccsd0, 6)
        self.assertAlmostEqual(et, et0, 6)

    def test_fno_by_thresh_frozen(self):
        threshs = [1e-2,1e-3,1e-4]
        refs = [
            [-0.0777291294,-0.0000381574],
            [-0.1409238993,-0.0004851219],
            [-0.1500184791,-0.0021645217],
        ]
        for thresh,ref in zip(threshs,refs):
            eccsd, et = self.kernel(cc.FNOCCSD, thresh=thresh, frozen=1)
            # print('[%s],' % (','.join([f'{x:.10f}' for x in [eccsd,et]])))
            self.assertAlmostEqual(eccsd, ref[0], 6)
            self.assertAlmostEqual(et, ref[1], 6)

        eccsd0, et0 = self.kernel(cc.UCCSD, frozen=1)
        eccsd, et = self.kernel(cc.FNOCCSD, thresh=1e-100, frozen=1)
        self.assertAlmostEqual(eccsd, eccsd0, 6)
        self.assertAlmostEqual(et, et0, 6)


if __name__ == "__main__":
    print("Full Tests for FNO-CCSD and FNO-CCSD(T)")
    unittest.main()
