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

import unittest
from pyscf import lib
from pyscf.pbc import scf, gto, grad

def finger(mat):
    return abs(mat).sum()

cell = gto.Cell()
cell.atom= [['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391]]]
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.basis = 'gth-szv'
cell.verbose= 4
cell.pseudo = 'gth-pade'
cell.unit = 'bohr'
cell.build()

kpts = cell.make_kpts([1,1,2])
disp = 1e-5


class KnownValues(unittest.TestCase):
    def test_kuhf_grad(self):
        g_scan = scf.KUHF(cell, kpts, exxdiv=None).nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(finger(g), 0.11476575559553441, 6)

        mfs = g_scan.base.as_scanner()
        e1 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391+disp/2.0]]])
        e2 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 6)

if __name__ == "__main__":
    print("Full Tests for KUHF Gradients")
    unittest.main()
