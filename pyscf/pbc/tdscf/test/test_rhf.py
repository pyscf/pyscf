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
from pyscf import lib
from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import RHF, KRHF
from pyscf.pbc import tdscf

def setUpModule():
    global cell
    cell = Cell()
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = {'C': [[0, (0.8, 1.0)],
                        [1, (1.0, 1.0)]]}
    # cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 0
    cell.build()

def tearDownModule():
    global cell
    del cell

class KnownValues(unittest.TestCase):
    def test_tda_gamma_point(self):
        mf = RHF(cell).run(conv_tol=1e-10)
        td_model = tdscf.TDA(mf)
        td_model.kernel()
        e1 = td_model.e

        kmf = KRHF(cell, cell.make_kpts([1, 1, 1])).run(conv_tol=1e-10)
        td_model = tdscf.KTDA(kmf)
        td_model.kernel()
        e2 = td_model.e
        self.assertAlmostEqual(abs(e1-e2).max(), 0, 6)
        self.assertAlmostEqual(abs(e1 - 1.0329858545904074).max(), 0, 5)

    def test_tdhf_gamma_point(self):
        mf = RHF(cell).run(conv_tol=1e-10)
        td_model = tdscf.TDHF(mf)
        td_model.kernel()
        e1 = td_model.e

        kmf = KRHF(cell, cell.make_kpts([1, 1, 1])).run(conv_tol=1e-10)
        td_model = tdscf.KTDHF(kmf)
        td_model.kernel()
        e2 = td_model.e
        self.assertAlmostEqual(abs(e1-e2).max(), 0, 5)
        self.assertAlmostEqual(abs(e1 - 1.0301736485136344).max(), 0, 5)

if __name__ == '__main__':
    print("Tests for pbc.tdscf.rhf")
    unittest.main()
