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
import numpy as np
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc import dft as pbcdft


def setUpModule():
    global cell
    L = 4.
    cell = pbcgto.Cell()
    cell.verbose = 0
    cell.a = np.eye(3)*L
    cell.atom =[['He' , ( L/2+0., L/2+0. ,   L/2+1.)],]
    cell.basis = {'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
    cell.build()

def tearDownModule():
    global cell
    del cell


class KnownValues(unittest.TestCase):
    def test_klda(self):
        cell = pbcgto.M(atom='H 0 0 0; H 1 0 0', a=np.eye(3)*2, basis=[[0, [1, 1]]])
        cell.build()
        mf = cell.KUKS(kpts=cell.make_kpts([2,2,1]))
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.3846075202893169, 7)

        mf.kpts = cell.make_kpts([2,2,1])
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.3846075202893169, 7)

    def test_klda8_cubic_kpt_222_high_cost(self):
        cell = pbcgto.Cell()
        cell.unit = 'A'
        cell.a = '''3.5668  0.      0.
                    0.      3.5668  0.
                    0.      0.      3.5668'''
        cell.mesh = np.array([17]*3)
        cell.atom ='''
C, 0.,  0.,  0.
C, 0.8917,  0.8917,  0.8917
C, 1.7834,  1.7834,  0.
C, 2.6751,  2.6751,  0.8917
C, 1.7834,  0.    ,  1.7834
C, 2.6751,  0.8917,  2.6751
C, 0.    ,  1.7834,  1.7834
C, 0.8917,  2.6751,  2.6751'''
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()

        kpts = cell.make_kpts((2,2,2), with_gamma_point=False)
        mf = pbcdft.KUKS(cell, kpts)
        mf.conv_tol = 1e-9
        mf.xc = 'lda,vwn'
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -45.42583489512954, 8)

    def test_rsh_fft(self):
        mf = pbcdft.KUKS(cell)
        mf.xc = 'hse06'
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.482418296326724, 7)

        mf = pbcdft.KUKS(cell)
        mf.xc = 'camb3lyp'
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.4745140703871877, 7)

    def test_rsh_df(self):
        mf = pbcdft.KUKS(cell).density_fit()
        mf.xc = 'wb97'
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.4916945546399165, 6)

        mf.xc = 'camb3lyp'
        mf.omega = .15
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.4766238116030683, 6)

    def test_to_hf(self):
        mf = pbcdft.KUKS(cell).density_fit()
        mf.with_df._j_only = True
        a_hf = mf.to_hf()
        self.assertTrue(a_hf.with_df._j_only)
        self.assertTrue(isinstance(a_hf, pbcscf.kuhf.KUHF))

        mf = pbcdft.KUKS(cell, kpts=cell.make_kpts([2,1,1])).density_fit()
        mf.with_df._j_only = True
        a_hf = mf.to_hf()
        self.assertTrue(not a_hf.with_df._j_only)
        self.assertTrue(isinstance(a_hf, pbcscf.kuhf.KUHF))

    # issue 2993
    def test_kuks_as_kuhf(self):
        cell = pbcgto.Cell()
        cell.atom = "He 0 0 0; He 1 1 1"
        cell.basis = [[0, [1, 1]], [0, [.5, 1]]]
        cell.spin = 2
        cell.a = np.eye(3) * 3
        cell.build()

        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh, time_reversal_symmetry=True)
        mf = pbcdft.KUKS(cell, kpts, xc="hf")
        mf.max_cycle = 1
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -4.213403459087, 9)

        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh)
        mf = pbcdft.KUKS(cell, kpts, xc="hf")
        mf.max_cycle = 1
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -4.213403459087, 9)


if __name__ == '__main__':
    print("Full Tests for pbc.dft.kuks")
    unittest.main()
