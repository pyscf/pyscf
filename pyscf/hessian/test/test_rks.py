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
from pyscf import gto, dft, lib
from pyscf import grad, hessian

mol = gto.Mole()
mol.verbose = 5
mol.output = '/dev/null'
mol.atom.extend([
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ])
mol.basis = '6-31g'
mol.build()

def tearDownModule():
    global mol
    del mol

class KnownValues(unittest.TestCase):
    def test_finite_diff_lda_hess(self):
        mf = dft.RKS(mol)
        mf.conv_tol = 1e-14
        e0 = mf.kernel()
        hess = mf.Hessian().kernel()
        self.assertAlmostEqual(lib.finger(hess), -0.7828771346902333, 6)

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 3)

    def test_finite_diff_b3lyp_hess(self):
        mf = dft.RKS(mol)
        mf.conv_tol = 1e-14
        mf.xc = 'b3lyp'
        e0 = mf.kernel()
        hess = mf.Hessian().kernel()
        self.assertAlmostEqual(lib.finger(hess), -0.7590878171493624, 6)

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 3)

    def test_finite_diff_wb97x_hess(self):
        mf = dft.RKS(mol)
        mf.conv_tol = 1e-14
        mf.xc = 'wb97x'
        e0 = mf.kernel()
        hess = mf.Hessian().kernel()
        self.assertAlmostEqual(lib.finger(hess), -0.7637876979690904, 6)

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 2)


if __name__ == "__main__":
    print("Full Tests for RKS Hessian")
    unittest.main()

