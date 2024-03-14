#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Authors: Xing Zhang <zhangxing.nju@gmail.com>
#

import unittest
import numpy as np
from pyscf.pbc import gto, scf, mp

def setUpModule():
    global He, nk, kpts0, kpts, kmf0, kmf, kmp2ref
    L = 2.
    He = gto.Cell()
    He.verbose = 0
    He.a = np.eye(3)*L
    He.atom =[['He' , ( L/2+0., L/2+0., L/2+0.)],]
    He.basis = {'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
    He.space_group_symmetry=True
    He.output = '/dev/null'
    He.build()

    nk = [2,2,2]

    kpts0 = He.make_kpts(nk)
    kmf0 = scf.KRHF(He, kpts0, exxdiv=None).density_fit()
    kmf0.kernel()
    kmp2ref = mp.KMP2(kmf0)
    kmp2ref.kernel()

    kpts = He.make_kpts(nk,space_group_symmetry=True,time_reversal_symmetry=True)
    kmf = scf.KRHF(He, kpts, exxdiv=None).density_fit()
    kmf.kernel()

def tearDownModule():
    global He, nk, kpts0, kpts, kmf0, kmf, kmp2ref
    He.stdout.close()
    del He, nk, kpts0, kpts, kmf0, kmf, kmp2ref

class KnownValues(unittest.TestCase):
    def test_kmp2(self):
        kmp2 = mp.KMP2(kmf)
        kmp2.kernel()
        self.assertAlmostEqual(kmp2.e_corr, kmp2ref.e_corr, 10)

    def test_rdm1(self):
        dm1ref = kmp2ref.make_rdm1()

        kmp2 = mp.KMP2(kmf)
        kmp2.kernel()
        dm1 = kmp2.make_rdm1()
        for i, k in enumerate(kpts.ibz2bz):
            error = np.amax(np.absolute(dm1[i] - dm1ref[k]))
            self.assertAlmostEqual(error, 0., 10)

if __name__ == '__main__':
    print("Full Tests for MP2 with k-point symmetry")
    unittest.main()
