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

from pyscf import gto
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
import pyscf.pbc
pyscf.pbc.DEBUG = False


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
    def test_pp_UKS(self):
        cell = pbcgto.Cell()

        cell.unit = 'A'
        cell.atom = '''
            Si    2.715348700    2.715348700    0.000000000;
            Si    2.715348700    0.000000000    2.715348700;
        '''
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'

        Lx = Ly = Lz = 5.430697500
        cell.a = np.diag([Lx,Ly,Lz])
        cell.mesh = np.array([17]*3)

        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()

        mf = pbcdft.UKS(cell)
        mf.xc = 'blyp'
        self.assertAlmostEqual(mf.scf(), -7.6058004283213396, 7)

        mf.xc = 'lda,vwn'
        self.assertAlmostEqual(mf.scf(), -7.6162130840535092, 7)

    def test_rsh_fft(self):
        mf = pbcdft.UKS(cell)
        mf.xc = 'wb97'
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.491686357528346, 7)

        mf.xc = 'camb3lyp'
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.4745140703871877, 7)

    def test_rsh_df(self):
        mf = pbcdft.UKS(cell).density_fit()
        mf.xc = 'hse06'
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.482426252, 7)

        mf.xc = 'camb3lyp'
        mf.omega = .15
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.476623986, 7)


if __name__ == '__main__':
    print("Full Tests for pbc.dft.uks")
    unittest.main()
