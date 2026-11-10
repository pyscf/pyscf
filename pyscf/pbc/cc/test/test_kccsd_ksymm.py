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



class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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

    @classmethod
    def tearDownClass(cls):
        global kmf, kcc
        del kmf, kcc

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


class SmallCellTests(unittest.TestCase):
    def test_outcore_symmetry_integrals(self):
        cell = gto.M(atom='He 0 0 0', a=np.eye(3)*3,
                     basis=[[0, [2., 1]], [0, [.5, 1]]],
                     space_group_symmetry=True, mesh=[17]*3,
                     precision=1e-7, verbose=0)
        kpts = cell.make_kpts([2, 1, 1], space_group_symmetry=True,
                             time_reversal_symmetry=True)
        mf = scf.KRHF(cell, kpts, exxdiv=None).density_fit(auxbasis='weigend')
        mf.run(conv_tol=1e-11)
        reference = cc.KRCCSD(mf.to_khf())
        reference.direct = False
        ref = reference.ao2mo()
        obj = cc.KsymAdaptedRCCSD(mf)
        obj.direct = False
        obj.eris_outcore = True
        out = obj.ao2mo()
        for name in ('oooo', 'ooov', 'oovv', 'ovov', 'voov', 'vovv', 'vvvv'):
            with self.subTest(block=name):
                np.testing.assert_allclose(getattr(out, name).todense(), getattr(ref, name), atol=1e-10)
        e, t1, t2 = obj.init_amps(out)
        eref, _, _ = reference.init_amps(ref)
        self.assertAlmostEqual(e, eref, places=10)

if __name__ == '__main__':
    print("Full Tests for CCSD with k-point symmetry")
    unittest.main()
