#!/usr/bin/env python
# Copyright 2023 The PySCF Developers. All Rights Reserved.
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
from pyscf import dft
from pyscf.pbc import gto as pbcgto
try:
    import mcfun
except ImportError:
    mcfun = None

def setUpModule():
    global cell
    cell = pbcgto.Cell()
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''
    cell.atom = 'He 0.,  0.,  0.; H 0.8917,  0.8917,  0.8917'
    cell.basis = [[0, [2, 1]], [1, [.5, 1]]]
    cell.spin = 1
    cell.build()

def tearDownModule():
    global cell
    del cell

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_collinear_gks_gga(self):
        mf = cell.GKS()
        mf.xc = 'pbe'
        mf.collinear = 'col'
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.7335936544788495, 7)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_gks_gga(self):
        mf = cell.GKS().density_fit()
        mf.xc = 'pbe'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.7331713336996268, 7)

    def test_ncol_x2c_gks_lda(self):
        mf = cell.GKS().x2c()
        mf.xc = 'lda,vwn'
        mf.collinear = 'ncol'
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.6093309656369732, 7)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_x2c_gks_lda(self):
        mf = cell.GKS().x2c()
        mf.xc = 'lda,vwn'
        mf._numint.spin_samples = 50
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.6093309656369732, 6)


if __name__ == '__main__':
    print("Full Tests for pbc.dft.gks")
    unittest.main()
