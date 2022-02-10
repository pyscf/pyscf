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

mol = gto.Mole()
mol.atom = '''
O    0    0    0
H    0.   -0.757   0.587
H    0.   0.757    0.587'''
mol.spin = None
mol.basis = 'sto3g'
mol.verbose = 7
mol.output = '/dev/null'
mol.build()

mol1 = gto.M(
    verbose = 0,
    atom = '''
O    0    0    0
H    0.   -0.757   0.587
H    0.   0.757    0.587''',
    charge = 1,
    spin = 1,
    basis = 'sto3g')

def tearDownModule():
    global mol, mol1
    mol.stdout.close()
    del mol, mol1

class KnownValues(unittest.TestCase):
    def test_ncol_dks_lda_omega(self):
        mf = mol.GKS()
        mf.xc = 'lda + .2*HF'
        mf.collinear = 'ncol'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.883375491657, 8)

        mf = mol.GKS()
        mf.xc = 'lda + .2*HF'
        mf.collinear = 'ncol'
        mf.omega = .5
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.38735765834898, 8)

    def test_ncol_dks_lda(self):
        mf = mol.GKS()
        mf.xc = 'lda,vwn'
        mf.collinear = 'ncol'
        eks4 = mf.kernel()
        self.assertAlmostEqual(lib.fp(mf.mo_energy), -26.54785447210512, 10)
        self.assertAlmostEqual(eks4, -74.73210527989738, 8)

        mf = mol1.GKS()
        mf.xc = 'lda,'
        mf.collinear = 'ncol'
        eks4 = mf.kernel()
        self.assertAlmostEqual(lib.fp(mf.mo_energy), -27.542272513714398, 8)
        self.assertAlmostEqual(eks4, -73.77115048625794, 8)

    def test_collinear_dks_lda(self):
        mf = mol.GKS()
        mf.xc = 'lda,vwn'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(lib.fp(mf.mo_energy), -26.54785447210512, 5)
        self.assertAlmostEqual(eks4, -74.73210527989738, 8)

        mf = mol1.GKS()
        mf.xc = 'lda,'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(lib.fp(mf.mo_energy), -27.542272513714398, 8)
        self.assertAlmostEqual(eks4, -73.77115048625794, 8)

    def test_mcol_dks_lda(self):
        mf = mol.GKS()
        mf.xc = 'lda,vwn'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        eks4 = mf.kernel()
        self.assertAlmostEqual(lib.fp(mf.mo_energy), -26.54785447210512, 8)
        self.assertAlmostEqual(eks4, -74.73210527989738, 8)

        mf = mol1.GKS()
        mf.xc = 'lda,'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 50
        eks4 = mf.kernel()
        self.assertAlmostEqual(lib.fp(mf.mo_energy), -27.542272513714398, 8)
        self.assertAlmostEqual(eks4, -73.75901895083638, 8)

    def test_collinear_dks_gga(self):
        mf = mol.GKS()
        mf.xc = 'pbe'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.22563990078712, 8)

        mf = mol1.GKS()
        mf.xc = 'pbe'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -74.86995486005469, 8)

    def test_mcol_dks_gga(self):
        mf = mol.GKS()
        mf.xc = 'pbe'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.22563990078712, 8)

        mf = mol1.GKS()
        mf.xc = 'pbe'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -74.85852580209466, 8)

    def test_collinear_dks_mgga(self):
        mf = mol.GKS()
        mf.xc = 'm06l'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.30536839893855, 8)

    def test_mcol_dks_mgga(self):
        mf = mol.GKS()
        mf.xc = 'm06l'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.30536839893855, 8)

    def test_ncol_x2c_uks_lda(self):
        mf = mol.GKS().x2c()
        mf.xc = 'lda,'
        mf.collinear = 'ncol'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -74.09933666072668, 8)

    def test_mcol_x2c_uks_lda(self):
        mf = mol.GKS().x2c()
        mf.xc = 'lda,'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -74.09933666072668, 8)

    def test_collinear_x2c_uks_gga(self):
        mf = mol.GKS().x2c()
        mf.xc = 'pbe'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.26499704046972, 8)


if __name__ == "__main__":
    print("Test GKS")
    unittest.main()
