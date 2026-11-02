#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
import numpy as np
import scipy.linalg
from pyscf import gto
from pyscf import dft
from pyscf import lib
from pyscf.dft import numint2c
try:
    import mcfun
except ImportError:
    mcfun = None

def setUpModule():
    global mol, mf
    mol = gto.M(
        verbose = 0,
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
        basis = '6-31g')
    with lib.temporary_env(dft.radi, ATOM_SPECIFIC_TREUTLER_GRIDS=False):
        mf = mol.GKS()
        mf.grids.level = 3
        mf.grids.build()

def tearDownModule():
    global mol, mf
    del mol, mf

class KnownValues(unittest.TestCase):
    def test_vxc_col(self):
        ni = numint2c.NumInt2C()
        ni.collinear = 'c'
        dm = mf.get_init_guess(mol, 'minao')
        n, e, v = ni.nr_vxc(mol, mf.grids, 'B88,', dm)
        self.assertAlmostEqual(n, 9.984666945, 5)
        self.assertAlmostEqual(e, -8.8304689765, 6)
        self.assertAlmostEqual(lib.fp(v), -2.5543189495160217, 8)

    def test_vxc_ncol(self):
        ni = numint2c.NumInt2C()
        ni.collinear = 'n'
        dm = mf.get_init_guess(mol, 'minao')
        n, e, v = ni.nr_vxc(mol, mf.grids, 'LDA,', dm)
        self.assertAlmostEqual(n, 9.984666945, 5)
        self.assertAlmostEqual(e, -7.9647540011, 6)
        self.assertAlmostEqual(lib.fp(v), -2.3201735523227227+0j, 8)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_vxc_mcol(self):
        ni = numint2c.NumInt2C()
        ni.collinear = 'm'
        ni.spin_samples = 14
        dm = mf.get_init_guess(mol, 'minao')
        n, e, v = ni.nr_vxc(mol, mf.grids, 'LDA,', dm)
        self.assertAlmostEqual(n, 9.984666945, 6)
        self.assertAlmostEqual(e, -7.9647540011, 6)
        self.assertAlmostEqual(lib.fp(v), -2.3201735523227227+0j, 8)

        n, e, v = ni.nr_vxc(mol, mf.grids, 'B88,', dm)
        self.assertAlmostEqual(n, 9.984666945, 5)
        #?self.assertAlmostEqual(e, -8.8304689765, 6)
        self.assertAlmostEqual(e, -8.8337325415, 6)
        #?self.assertAlmostEqual(lib.fp(v), -2.5543189495160217, 8)
        self.assertAlmostEqual(lib.fp(v), -2.5920046321400507+0j, 8)

    def test_fxc_col(self):
        ni = numint2c.NumInt2C()
        ni.collinear = 'c'
        dm = mf.get_init_guess(mol, 'minao')
        np.random.seed(10)
        dm1 = np.random.random(dm.shape)
        v = ni.nr_fxc(mol, mf.grids, 'B88,', dm, dm1)
        self.assertAlmostEqual(lib.fp(v), 1.0325167632577212, 6)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_fxc_mcol(self):
        ni = numint2c.NumInt2C()
        ni.collinear = 'm'
        ni.spin_samples = 14
        dm = mf.get_init_guess(mol, 'minao')
        np.random.seed(10)
        dm1 = np.random.random(dm.shape)
        v = ni.nr_fxc(mol, mf.grids, 'LDA,', dm, dm1)
        self.assertAlmostEqual(lib.fp(v), 1.9147702820070673+0j, 6)

        v = ni.nr_fxc(mol, mf.grids, 'M06', dm, dm1)
        self.assertAlmostEqual(lib.fp(v), 0.7599330879272782+0j, 6)

    def test_get_rho(self):
        ni = numint2c.NumInt2C()
        ni.collinear = 'c'
        dm = mf.get_init_guess(mol, 'minao')
        rho = ni.get_rho(mol, dm, mf.grids)
        self.assertAlmostEqual(lib.fp(rho), -361.4682369790235, 8)

        ni.collinear = 'm'
        ni.spin_samples = 50
        rho = ni.get_rho(mol, dm, mf.grids)
        self.assertAlmostEqual(lib.fp(rho), -361.4682369790235, 8)

    def check_cache(self, xc, collinear='col'):
        mol = gto.M(atom='He 0 0 0', basis=[[0, [1., 1.]], [0, [.3, 1.]]], verbose=0)
        grids = dft.gen_grid.Grids(mol)
        # Sample a finite-density region: XC second derivatives diverge in vacuum.
        grids.coords = np.random.default_rng(3).uniform(-1., 1., (112, 3))
        grids.weights = np.ones(112) / 112
        coeff = np.eye(4)
        occ = np.array([.7, .3, .5, .2])
        dm = np.diag(occ)
        ni = numint2c.NumInt2C()
        ni.collinear = collinear
        expected = ni.cache_xc_kernel(mol, grids, xc, coeff, occ)
        actual = ni.cache_xc_kernel1(mol, grids, xc, dm)
        for index, (x, y) in enumerate(zip(actual, expected)):
            # Density-matrix MGGA densities include a Laplacian placeholder;
            # orbital densities omit it. Compare rho, gradients, and tau.
            if index == 0 and x.shape != y.shape:
                x = x[:, [0, 1, 2, 3, 5]]
            np.testing.assert_allclose(x, y, atol=1e-10, rtol=1e-10)
        if collinear == 'col':
            perturbation = np.diag([.1, -.05, -.08, .03])
            response = ni.nr_fxc(mol, grids, xc, dm, perturbation,
                                 hermi=1, rho0=actual[0], vxc=actual[1], fxc=actual[2])
            step = 1e-4
            vp = ni.nr_vxc(mol, grids, xc, dm + step*perturbation)[2]
            vm = ni.nr_vxc(mol, grids, xc, dm - step*perturbation)[2]
            np.testing.assert_allclose(response, (vp-vm)/(2*step), atol=1e-8, rtol=1e-6)

    def test_lda_cache_and_response(self):
        self.check_cache('LDA,VWN')

    def test_gga_cache_and_response(self):
        self.check_cache('PBE')

    def test_mgga_cache_and_response(self):
        self.check_cache('TPSS')

    def test_noncollinear_lda_cache(self):
        self.check_cache('LDA,VWN', 'ncol')

if __name__ == "__main__":
    print("Test numint")
    unittest.main()
