#!/usr/bin/env python
# Copyright 2022-2023 The PySCF Developers. All Rights Reserved.
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
from pyscf import lib
from pyscf.pbc import gto, scf, cc

def setUpModule():
    global kmf, kcc
    L = 2.
    He = gto.Cell()
    He.verbose = 5
    He.output = '/dev/null'
    He.a = np.eye(3)*L
    He.atom =[['He' , ( L/2+0., L/2+0., L/2+0.)],]
    He.basis = {'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
    He.space_group_symmetry = True
    He.build()

    nk = [2,2,2]
    kpts = He.make_kpts(nk,space_group_symmetry=True,time_reversal_symmetry=True)
    kmf = scf.KRHF(He, kpts, exxdiv=None).density_fit()
    kmf.kernel()
    kcc = cc.KsymAdaptedRCCSD(kmf)
    kcc.kernel()

def tearDownModule():
    global kmf, kcc
    del kmf, kcc

class KnownValues(unittest.TestCase):
    def test_krccsd_ksym(self):
        self.assertAlmostEqual(kcc.e_corr, -0.0073791230287231875, 8)

        t1 = kcc.t1.todense()
        self.assertAlmostEqual(lib.fp(t1).max(), 0.0009185224804818333, 7)

    def test_vs_krccsd(self):
        kmf0 = kmf.to_khf()
        kccref = cc.krccsd.KRCCSD(kmf0)
        kccref.kernel()

        t1 = kcc.t1.todense()
        self.assertAlmostEqual(abs(kccref.t1 - t1).max(), 0, 6)
        t2 = kcc.t2.todense()
        self.assertAlmostEqual(abs(kccref.t2 - t2).max(), 0, 6)

if __name__ == '__main__':
    print("Full Tests for CCSD with k-point symmetry")
    unittest.main()
