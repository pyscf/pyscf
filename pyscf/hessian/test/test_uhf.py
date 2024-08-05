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
from pyscf import gto, scf, lib
from pyscf import grad, hessian
try:
    from pyscf.dispersion import dftd3, dftd4
except ImportError:
    dftd3 = dftd4 = None

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = '6-31g'
    mol.spin = 2
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_uhf_hess(self):
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-14
        e0 = mf.kernel()
        hobj = mf.Hessian()
        hobj.level_shift = .05
        hess = hobj.kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.20243405976628576, 4)

    def test_uhf_hess_atmlst(self):
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-14
        e0 = mf.kernel()

        atmlst = [0, 1]
        hess_1 = mf.Hessian().kernel()[atmlst][:, atmlst]
        hess_2 = mf.Hessian().kernel(atmlst=atmlst)
        self.assertAlmostEqual(abs(hess_1-hess_2).max(), 0.0, 4)

    def test_finite_diff_uhf_hess(self):
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-14
        e0 = mf.kernel()
        hess = mf.Hessian().kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.20243405976628576, 4)

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 4)

    @unittest.skipIf(dftd3 is None, "requires the dftd3 library")
    def test_finite_diff_uhf_d3_hess(self):
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-14
        mf.disp = 'd3bj'
        e0 = mf.kernel()
        hess = mf.Hessian().kernel()

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 4)

    @unittest.skipIf(dftd4 is None, "requires the dftd4 library")
    def test_finite_diff_uhf_d4_hess(self):
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-14
        mf.disp = 'd4'
        e0 = mf.kernel()
        hess = mf.Hessian().kernel()

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 4)

if __name__ == "__main__":
    print("Full Tests for UHF Hessian")
    unittest.main()
