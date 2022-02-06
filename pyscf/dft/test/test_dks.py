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
import numpy
from pyscf import gto
from pyscf import lib
from pyscf.dft import dks

mol = gto.Mole()
mol.atom = '''
O    0    0    0
H    0.   -0.757   0.587
H    0.   0.757    0.587'''
mol.charge = 1
mol.spin = None
mol.basis = 'uncsto3g'
mol.verbose = 7
mol.output = '/dev/null'
mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_noncol_dks_lda(self):
        with lib.temporary_env(lib.param, LIGHT_SPEED=12):
            mf = dks.UDKS(mol)
            mf.xc = 'lda,vwn'
            eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -80.3600775975093, 8)

    def test_x2c_uks_lda(self):
        mf = dks.UDKS(mol).x2c()
        mf.xc = 'lda,'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -73.9972147814834, 8)

    def test_dks_lda_omega(self):
        mf = dks.UDKS(mol)
        mf.xc = 'lda + .2*HF'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.71593214601225, 8)

        mf = dks.UDKS(mol)
        mf.xc = 'lda + .2*HF'
        mf.omega = .5
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.26598450889301, 8)

    def test_collinear_dks_lda(self):
        mf = dks.UDKS(mol)
        mf.xc = 'lda,vwn'
        mf.collinear = True
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -74.60528573582566, 8)

    def test_collinear_dks_gga(self):
        mf = dks.UDKS(mol)
        mf.xc = 'pbe'
        mf.collinear = True
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.07046613221685, 8)

    def test_collinear_x2c_uks_gga(self):
        mf = dks.UDKS(mol).x2c()
        mf.xc = 'pbe'
        mf.collinear = True
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.06735429926894, 8)


if __name__ == "__main__":
    print("Test DKS")
    unittest.main()
