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
import pyscf.pbc.gto as pbcgto
from pyscf.pbc import dft as pdft
from pyscf.pbc import tools as ptools


def setUpModule():
    global cell
    cell = pbcgto.Cell()
    cell.atom = '''
    C 0              0              0
    C 1.685068785275 1.685068785275 1.685068785275'''
    cell.a = '''
    0.000000000, 3.370137571, 3.370137571
    3.370137571, 0.000000000, 3.370137571
    3.370137571, 3.370137571, 0.000000000
    '''
    cell.basis = 'gth-szv'
    cell.unit = 'B'
    cell.pseudo = 'gth-pade'
    cell.mesh = [25]*3
    cell.verbose = 0
    cell.build()

def tearDownModule():
    global cell
    del cell

class KnowValues(unittest.TestCase):
    def test_gamma_vs_ks_high_cost(self):
        mf = pdft.KRKS(cell)
        mf.kpts = cell.make_kpts([1,1,3])
        ek = mf.kernel()

        scell = ptools.super_cell(cell, [1,1,3])
        scell.mesh = [25,25,73]
        mf = pdft.RKS(scell)
        eg = mf.kernel()
        self.assertAlmostEqual(ek, eg/3, 5)


if __name__ == '__main__':
    print("Full Tests for gamma point vs k-points")
    unittest.main()
