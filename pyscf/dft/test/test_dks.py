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
try:
    import mcfun
except ImportError:
    mcfun = None

def setUpModule():
    global mol, mol1
    mol = gto.Mole()
    mol.atom = '''
O    0    0    0
H    0.   -0.757   0.587
H    0.   0.757    0.587'''
    mol.basis = 'uncsto3g'
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
        basis = 'uncsto3g')

def tearDownModule():
    global mol, mol1
    mol.stdout.close()
    del mol, mol1

class KnownValues(unittest.TestCase):
    def test_ncol_dks_lda_omega(self):
        mf = dks.UDKS(mol)
        mf.xc = 'lda + .2*HF'
        mf.collinear = 'ncol'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -76.1457647524492, 8)

        mf = dks.UDKS(mol)
        mf.xc = 'lda + .2*HF'
        mf.collinear = 'ncol'
        mf.omega = .5
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.65261312636814, 8)

    def test_ncol_dks_lda(self):
        mf = dks.UDKS(mol)
        mf.xc = 'lda,vwn'
        mf.collinear = 'ncol'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.02287138686222, 8)

        mf = mol1.DKS()
        mf.xc = 'lda,'
        mf.collinear = 'ncol'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -74.0009332386581, 8)

    def test_collinear_dks_lda(self):
        mf = dks.UDKS(mol)
        mf.xc = 'lda,vwn'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.02287138686222, 8)

        mf = mol1.DKS()
        mf.xc = 'lda,'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -74.00093287236554, 8)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_dks_lda(self):
        mf = dks.UDKS(mol)
        mf.xc = 'lda,vwn'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.02287138686222, 8)

        mf = mol1.DKS()
        mf.xc = 'lda,'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -74.00137998066057, 8)

    def test_collinear_dks_gga(self):
        mf = dks.UDKS(mol)
        mf.xc = 'pbe'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.49000045638684, 8)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_dks_gga_high_cost(self):
        mf = dks.UDKS(mol)
        mf.xc = 'pbe'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.49000045638684, 8)

        mf = mol1.DKS()
        mf.xc = 'pbe'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.08323754335314, 8)

    def test_collinear_dks_mgga(self):
        mf = dks.UDKS(mol)
        mf.xc = 'm06l'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.54644605761466, 8)

        mf = mol1.DKS()
        mf.xc = 'pbe'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.08263680935453, 8)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_dks_mgga_high_cost(self):
        mf = dks.UDKS(mol)
        mf.xc = 'm06l'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.54644605761466, 8)

    def test_ncol_x2c_uks_lda(self):
        mf = dks.UDKS(mol).x2c()
        mf.xc = 'lda,'
        mf.collinear = 'ncol'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -74.35294598734184, 8)

    def test_collinear_x2c_uks_gga(self):
        mf = dks.UDKS(mol).x2c()
        mf.xc = 'pbe'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.48678822885732, 8)


if __name__ == "__main__":
    print("Test DKS")
    unittest.main()
