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
from pyscf.dft import numint, numint2c, libxc, radi
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
    def test_ncol_gks_lda_omega(self):
        mf = mol.GKS()
        mf.xc = 'lda + .2*HF'
        mf.collinear = 'ncol'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.883375491657, 6)

        mf = mol.GKS()
        mf.xc = 'lda + .2*SR_HF(0.3)'
        mf.collinear = 'ncol'
        mf.omega = .5
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.38735765834898, 6)

    def test_ncol_gks_lda(self):
        mf = mol.GKS()
        mf.xc = 'lda,vwn'
        mf.collinear = 'ncol'
        eks4 = mf.kernel()
        self.assertAlmostEqual(lib.fp(mf.mo_energy), -26.54785447210512, 5)
        self.assertAlmostEqual(eks4, -74.73210527989738, 6)

        mf = mol1.GKS()
        mf.xc = 'lda,'
        mf.collinear = 'ncol'
        eks4 = mf.kernel()
        self.assertAlmostEqual(lib.fp(mf.mo_energy), -27.542272513714398, 6)
        self.assertAlmostEqual(eks4, -73.77115048625794, 6)

    def test_collinear_gks_lda(self):
        mf = mol.GKS()
        mf.xc = 'lda,vwn'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        #FIXME: Why does mo_energy have small difference to ncol_gks_lda?
        self.assertAlmostEqual(lib.fp(mf.mo_energy), -26.54785447210512, 5)
        self.assertAlmostEqual(eks4, -74.73210527989738, 6)

        mf = mol1.GKS()
        mf.xc = 'lda,'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(lib.fp(mf.mo_energy), -27.542272513714398, 6)
        self.assertAlmostEqual(eks4, -73.77115048625794, 6)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_gks_lda(self):
        mf = mol.GKS()
        mf.xc = 'lda,vwn'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        eks4 = mf.kernel()
        self.assertAlmostEqual(lib.fp(mf.mo_energy), -26.54785447210512, 5)
        self.assertAlmostEqual(eks4, -74.73210527989738, 6)

        mf = mol1.GKS()
        mf.xc = 'lda,'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 50
        eks4 = mf.kernel()
        self.assertAlmostEqual(lib.fp(mf.mo_energy), -27.542272513714398, 6)
        self.assertAlmostEqual(eks4, -73.77115048625794, 6)

    def test_collinear_gks_gga(self):
        mf = mol.GKS()
        mf.xc = 'pbe'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.22563990078712, 6)

        mf = mol1.GKS()
        mf.xc = 'pbe'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -74.86995486005469, 6)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_gks_gga(self):
        mf = mol.GKS()
        mf.xc = 'pbe'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.22563990078712, 6)

        mf = mol1.GKS()
        mf.xc = 'pbe'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -74.87069542226276, 6)

    def test_collinear_gks_mgga(self):
        mf = mol.GKS()
        mf.xc = 'm06l'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.30536839893855, 5)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_gks_mgga(self):
        mf = mol1.GKS()
        mf.xc = 'm06l'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -74.94902210438143, 6)

    def test_ncol_x2c_gks_lda(self):
        mf = mol.GKS().x2c()
        mf.xc = 'lda,'
        mf.collinear = 'ncol'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -74.09933666072668, 6)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_x2c_gks_lda(self):
        mf = mol.GKS().x2c()
        mf.xc = 'lda,'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -74.09933666072668, 6)

    def test_collinear_x2c_gks_gga(self):
        mf = mol.GKS().x2c()
        mf.xc = 'pbe'
        mf.collinear = 'col'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -75.26499704046972, 6)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_lda_vxc_mat(self):
        nao = mol.nao
        n2c = nao * 2
        ao_loc = mol.ao_loc
        numpy.random.seed(12)
        dm = numpy.random.rand(n2c, n2c) * .001
        dm += numpy.eye(n2c)
        dm = dm + dm.T
        ngrids = 8
        coords = numpy.random.rand(ngrids,3)
        weight = numpy.random.rand(ngrids)

        ao = numint.eval_ao(mol, coords, deriv=0)
        rho = numint2c.eval_rho(mol, ao, dm, xctype='LDA', hermi=1)
        ni = numint2c.NumInt2C()
        vxc = ni.eval_xc_eff('lda,', rho, deriv=1)[1]
        mask = numpy.ones((100, mol.nbas), dtype=numpy.uint8)
        shls_slice = (0, mol.nbas)
        v0 = numint2c._ncol_lda_vxc_mat(mol, ao, weight, rho, vxc.copy(), mask, shls_slice, ao_loc, 0)
        v1 = numint2c._ncol_lda_vxc_mat(mol, ao, weight, rho, vxc.copy(), mask, shls_slice, ao_loc, 1)
        v1 = v1 + v1.conj().T
        ref = v0
        self.assertAlmostEqual(abs(v0 - v1).max(), 0, 13)
        self.assertAlmostEqual(lib.fp(v0), 0.19683067215390423, 12)

        ni.collinear = 'mcol'
        eval_xc = ni.mcfun_eval_xc_adapter('lda,')
        vxc = eval_xc('lda,', rho, deriv=1)[1]
        mask = numpy.ones((100, mol.nbas), dtype=numpy.uint8)
        shls_slice = (0, mol.nbas)
        v0 = numint2c._mcol_lda_vxc_mat(mol, ao, weight, rho, vxc.copy(), mask, shls_slice, ao_loc, 0)
        v1 = numint2c._mcol_lda_vxc_mat(mol, ao, weight, rho, vxc.copy(), mask, shls_slice, ao_loc, 1)
        v1 = v1 + v1.conj().T
        self.assertAlmostEqual(abs(v0 - ref).max(), 0, 3)
        self.assertAlmostEqual(abs(v0 - v1).max(), 0, 13)

    def test_mcol_gga_vxc_mat(self):
        nao = mol.nao
        n2c = nao * 2
        ao_loc = mol.ao_loc
        numpy.random.seed(12)
        dm = numpy.random.rand(n2c, n2c) * .01
        dm += numpy.eye(n2c)
        dm = dm + dm.T
        ngrids = 8
        coords = numpy.random.rand(ngrids,3)
        weight = numpy.random.rand(ngrids)

        ao = numint.eval_ao(mol, coords, deriv=1)
        rho = numint2c.eval_rho(mol, ao, dm, xctype='GGA', hermi=1)
        vxc = numpy.random.rand(4, 4, ngrids)
        mask = numpy.ones((100, mol.nbas), dtype=numpy.uint8)
        shls_slice = (0, mol.nbas)
        v0 = numint2c._mcol_gga_vxc_mat(mol, ao, weight, rho, vxc.copy(), mask, shls_slice, ao_loc, 0)
        v1 = numint2c._mcol_gga_vxc_mat(mol, ao, weight, rho, vxc.copy(), mask, shls_slice, ao_loc, 1)
        v1 = v1 + v1.conj().T
        self.assertAlmostEqual(abs(v0 - v1).max(), 0, 13)
        self.assertAlmostEqual(lib.fp(v0), -0.889763561992794-0.013552640219244905j, 12)

    def test_mcol_mgga_vxc_mat(self):
        nao = mol.nao
        n2c = nao * 2
        ao_loc = mol.ao_loc
        numpy.random.seed(12)
        dm = numpy.random.rand(n2c, n2c) * .01
        dm += numpy.eye(n2c)
        dm = dm + dm.T
        ngrids = 8
        coords = numpy.random.rand(ngrids,3)
        weight = numpy.random.rand(ngrids)

        ao = numint.eval_ao(mol, coords, deriv=1)
        rho = numint2c.eval_rho(mol, ao, dm, xctype='MGGA', hermi=1, with_lapl=False)
        vxc = numpy.random.rand(4, 5, ngrids)
        mask = numpy.ones((100, mol.nbas), dtype=numpy.uint8)
        shls_slice = (0, mol.nbas)
        v0 = numint2c._mcol_mgga_vxc_mat(mol, ao, weight, rho, vxc.copy(), mask, shls_slice, ao_loc, 0)
        v1 = numint2c._mcol_mgga_vxc_mat(mol, ao, weight, rho, vxc.copy(), mask, shls_slice, ao_loc, 1)
        v1 = v1 + v1.conj().T
        self.assertAlmostEqual(abs(v0 - v1).max(), 0, 13)
        self.assertAlmostEqual(lib.fp(v0), 0.45641500123185696-0.11533144122332428j, 12)

if __name__ == "__main__":
    print("Test GKS")
    unittest.main()
