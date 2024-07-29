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
        mf.xc = 'camb3lyp'
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.4745140703871877, 7)

    def test_rsh_df(self):
        mf = pbcdft.KUKS(cell).density_fit()
        mf.xc = 'camb3lyp'
        mf.omega = .15
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.4766238116030683, 7)


if __name__ == '__main__':
    print("Full Tests for pbc.dft.kuks")
    unittest.main()
