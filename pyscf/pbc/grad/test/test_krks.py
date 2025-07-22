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
from pyscf.pbc import dft, gto, grad

def setUpModule():
    global cell, kpts, disp
    cell = gto.Cell()
    cell.atom= [['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391]]]
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.basis = [[0, [1.3, 1]], [1, [0.8, 1]]]
    cell.verbose = 5
    cell.pseudo = 'gth-pade'
    cell.unit = 'bohr'
    cell.mesh = [13] * 3
    cell.output = '/dev/null'
    cell.build()

    kpts = cell.make_kpts([1,1,3])
    disp = 1e-3

def tearDownModule():
    global cell
    cell.stdout.close()
    del cell


class KnownValues(unittest.TestCase):

    def test_lda_grad(self):
        mf = dft.KRKS(cell, kpts)
        mf.xc = 'lda,vwn'
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(lib.fp(g), -0.22166962318360375, 6)
        mfs = g_scan.base.as_scanner()
        e1 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391+disp/2.0]]])
        e2 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 6)

    def test_gga_grad(self):
        mf = dft.KRKS(cell, kpts)
        mf.xc = 'pbe,pbe'
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(lib.fp(g), -0.21844074846755882, 6)

        mfs = g_scan.base.as_scanner()
        e1 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391+disp/2.0]]])
        e2 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 6)

    def test_hybrid_grad(self):
        mf = dft.KRKS(cell, kpts)
        mf.xc = 'b3lyp5'
        mf.exxdiv = None
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(lib.fp(g), -0.19544969829285652, 6)

        mfs = g_scan.base.as_scanner()
        e1 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391+disp/2.0]]])
        e2 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 6)

if __name__ == "__main__":
    print("Full Tests for KRKS Gradients")
    unittest.main()
