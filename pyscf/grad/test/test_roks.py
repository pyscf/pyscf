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
from pyscf import gto, scf, lib, dft
from pyscf import grad

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom = '''
        O     0.   0.       0.
        H     0.8  0.3      0.2
        H     0.   -0.757   0.587
        H     0.   0.757    0.587'''
    mol.charge = 0
    mol.spin = 3
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_finite_diff_roks_grad(self):
        mf = scf.ROKS(mol)
        mf.xc = 'b3lypg'
        mf.conv_tol = 1e-14
        e0 = mf.kernel()
        g = grad.ROKS(mf).kernel()
        mf_scanner = mf.as_scanner()

        e1 = mf_scanner('''O    0.   0.       0.
                        H     0.8  0.3      0.2
                        H    0.   -0.758   0.587
                        H    0.   0.757    0.587''')
        e2 = mf_scanner('''O    0.   0.       0.
                        H     0.8  0.3      0.2
                        H    0.   -0.756   0.587
                        H    0.   0.757    0.587''')
        self.assertAlmostEqual(g[2,1], (e2-e1)/2e-3*lib.param.BOHR, 4)

    def test_finite_diff_df_roks_grad(self):
        mf = scf.ROKS(mol).density_fit ()
        mf.xc = 'b3lypg'
        mf.conv_tol = 1e-14
        e0 = mf.kernel()
        mf_grad = mf.nuc_grad_method ()
        g = mf_grad.kernel()
        mf_scanner = mf.as_scanner()

        e1 = mf_scanner('''O    0.   0.       0.
                        H     0.8  0.3      0.2
                        H    0.   -0.758   0.587
                        H    0.   0.757    0.587''')
        e2 = mf_scanner('''O    0.   0.       0.
                        H     0.8  0.3      0.2
                        H    0.   -0.756   0.587
                        H    0.   0.757    0.587''')
        self.assertAlmostEqual(g[2,1], (e2-e1)/2e-3*lib.param.BOHR, 4)

    def test_roks_lda_grid_response(self):
        mol = gto.Mole()
        mol.atom = [
            ['O' , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. ,  0.757 , 0.587)] ]
        mol.basis = '631g'
        mol.charge = 1
        mol.spin = 1
        mol.build()
        mf = dft.ROKS(mol)
        mf.conv_tol = 1e-12
        #mf.grids.atom_grid = (20,86)
        e0 = mf.scf()
        g = mf.Gradients()
        g1 = g.kernel()
#[[  0.    0.               0.0529837158]
# [  0.    0.0673568416    -0.0264979200]
# [  0.   -0.0673568416    -0.0264979200]]
        self.assertAlmostEqual(g1[0, 2], 0.0529837158, 6)
        g.grid_response = True
        g1 = g.kernel()
#[[  0.    0.               0.0529917556]
# [  0.    0.0673570505    -0.0264958778]
# [  0.   -0.0673570505    -0.0264958778]]
        self.assertAlmostEqual(g1[0, 2], 0.0529917556, 6)

    def test_roks_gga_grid_response(self):
        mol = gto.Mole()
        mol.atom = [
            ['O' , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. ,  0.757 , 0.587)] ]
        mol.basis = '631g'
        mol.charge = 1
        mol.spin = 1
        mol.build()
        mf = dft.ROKS(mol)
        mf.conv_tol = 1e-12
        mf.xc = 'b88,p86'
        e0 = mf.scf()
        g = mf.Gradients()
        g1 = g.kernel()
#[[  0.    0.               0.0516999634]
# [  0.    0.0638666270    -0.0258541362]
# [  0.   -0.0638666270    -0.0258541362]]
        self.assertAlmostEqual(g1[0, 2], 0.0516999634, 6)
        g.grid_response = True
        g1 = g.kernel()
#[[  0.    0.               0.0516940546]
# [  0.    0.0638566430    -0.0258470273]
# [  0.   -0.0638566430    -0.0258470273]]
        self.assertAlmostEqual(g1[0, 2], 0.0516940546, 6)

        mf.xc = 'b3lypg'
        e0 = mf.scf()
        g = mf.Gradients()
        g1 = g.kernel()
#[[  0.    0.               0.0395990911]
# [  0.    0.0586841789    -0.0198038250]
# [  0.   -0.0586841789    -0.0198038250]]
        self.assertAlmostEqual(g1[0, 2], 0.0395990911, 6)

    def test_roks_grids_converge(self):
        mol = gto.Mole()
        mol.atom = [
            ['H' , (0. , 0. , 1.804)],
            ['F' , (0. , 0. , 0.   )], ]
        mol.unit = 'B'
        mol.basis = '631g'
        mol.charge = -1
        mol.spin = 1
        mol.build()

        mf = dft.ROKS(mol)
        mf.conv_tol = 1e-14
        mf.kernel()
        g1 = mf.Gradients().kernel()
# sum over z direction non-zero, due to meshgrid response
#[[ 0  0   -0.1479101538]
# [ 0  0    0.1479140846]]
        self.assertAlmostEqual(g1[0, 2], -0.1479101538, 6)
        self.assertAlmostEqual(g1[1, 2],  0.1479140846, 6)

        mf = dft.ROKS(mol)
        mf.grids.prune = None
        mf.grids.level = 6
        mf.conv_tol = 1e-14
        mf.kernel()
        g1 = mf.Gradients().kernel()
#[[ 0  0   -0.1479101105]
# [ 0  0    0.1479099093]]
        self.assertAlmostEqual(g1[0, 2], -0.1479101105, 6)
        self.assertAlmostEqual(g1[1, 2],  0.1479099093, 6)


if __name__ == "__main__":
    print("Full Tests for ROKS Gradients")
    unittest.main()
