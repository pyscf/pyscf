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
from pyscf import dft
from pyscf.dft import dks
try:
    import mcfun
except ImportError:
    mcfun = None

def setUpModule():
    global mol, mol1, dm4c_guess, dm2c_guess
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

    numpy.random.seed(1)
    dm4c_guess = dks.UDKS(mol).get_init_guess(mol, 'hcore')
    dm4c_guess = dm4c_guess + (numpy.random.random(dm4c_guess.shape) * .1 +
                               numpy.random.random(dm4c_guess.shape) * .1j)
    n2c = mol1.nao_2c()
    dm2c_guess = dm4c_guess[:n2c,:n2c]

def tearDownModule():
    global mol, mol1
    mol.stdout.close()
    del mol, mol1

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids


    def test_ncol_dks_lda_omega_high_cost(self):
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

    def test_ncol_dks_lda_high_cost(self):
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

    def test_ncol_dks_lda_veff(self):
        mf = dks.UDKS(mol)
        mf.collinear = 'ncol'
        veff = mf.get_veff(mol, dm4c_guess)
        self.assertAlmostEqual(lib.fp(veff), 6.831445865173151-28.252983015580064j, 8)

    def test_collinear_dks_lda_high_cost(self):
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

    def test_collinear_dks_lda_veff(self):
        mf = dks.UDKS(mol)
        veff = mf.get_veff(mol, dm4c_guess)
        self.assertAlmostEqual(lib.fp(veff), 6.0513153425666815-27.01477415630974j, 8)

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

    def test_collinear_dks_gga_high_cost(self):
        mf = dks.UDKS(mol)
        mf.xc = 'pbe'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.49000045638684, 8)

    def test_collinear_dks_gga_veff(self):
        mf = dks.UDKS(mol)
        mf.xc = 'pbe'
        veff = mf.get_veff(mol, dm4c_guess)
        self.assertAlmostEqual(lib.fp(veff), 5.974620910832044-27.039876829357134j, 8)

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

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_dks_gga_eff(self):
        mf = dks.UDKS(mol1)
        mf.xc = 'pbe'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        veff = mf.get_veff(mol1, dm4c_guess)
        self.assertAlmostEqual(lib.fp(veff), 28.752767129570167-62.57066597442416j, 8)

    def test_collinear_dks_mgga_high_cost(self):
        mf = dks.UDKS(mol)
        mf.xc = 'm06l'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.54644605761466, 8)

        mf = mol1.DKS()
        mf.xc = 'm06l'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.14078030073547, 8)

    def test_collinear_dks_mgga_veff(self):
        mf = dks.UDKS(mol)
        mf.xc = 'm06l'
        dm = mf.get_init_guess(mol, 'hcore')
        veff = mf.get_veff(mol, dm)
        self.assertAlmostEqual(lib.fp(veff), 1.333048765920303-4.881773770448651j, 8)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_dks_mgga_high_cost(self):
        mf = dks.UDKS(mol)
        mf.xc = 'm06l'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.54644605761466, 8)

    def test_ncol_x2c_uks_lda_high_cost(self):
        mf = dks.UDKS(mol).x2c()
        mf.xc = 'lda,'
        mf.collinear = 'ncol'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -74.35294598734184, 8)

    def test_ncol_x2c_uks_lda_fock(self):
        mf = dks.UDKS(mol).x2c()
        mf.xc = 'lda,'
        mf.collinear = 'ncol'
        fock = mf.get_fock(dm=dm2c_guess)
        self.assertAlmostEqual(lib.fp(fock), 118.5637024485895-12.104187121201711j, 8)

    def test_collinear_x2c_uks_gga_high_cost(self):
        mf = dks.UDKS(mol).x2c()
        mf.xc = 'pbe'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.48678822885732, 8)

    def test_collinear_x2c_uks_gga_fock(self):
        mf = dks.UDKS(mol).x2c()
        mf.xc = 'pbe'
        mf.collinear = 'col'
        fock = mf.get_fock(dm=dm2c_guess)
        self.assertAlmostEqual(lib.fp(fock), 118.90408541458346-12.3152299163913j, 8)

if __name__ == "__main__":
    print("Test DKS")
    unittest.main()
