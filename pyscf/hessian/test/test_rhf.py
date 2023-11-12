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

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = 'ccpvdz'
    mol.build()

def tearDownModule():
    global mol
    del mol

class KnownValues(unittest.TestCase):
    def test_finite_diff_x2c_rhf_hess(self):
        mf = scf.RHF(mol).x2c()
        mf.conv_tol = 1e-14
        e0 = mf.kernel()
        hess = hessian.RHF(mf).kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.7800532318291435, 6)

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 4)

#        e1 = g_scanner(pmol.set_geom_('O  0. 0.0001 0.; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
#        e2 = g_scanner(pmol.set_geom_('O  0. -.0001 0.; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
#        self.assertAlmostEqual(abs(hess[0,:,1] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 4)
#
#        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.7571 0.587; 1  0. 0.757 0.587'))[1]
#        e2 = g_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.7569 0.587; 1  0. 0.757 0.587'))[1]
#        self.assertAlmostEqual(abs(hess[1,:,1] - (e2-e1)/2e-4*lib.param.BOHR).max(), 0, 4)

        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.757 0.5871; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.757 0.5869; 1  0. 0.757 0.587'))[1]
        self.assertAlmostEqual(abs(hess[1,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 4)


    def test_finite_diff_rhf_hess(self):
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-14
        e0 = mf.kernel()
        hess = hessian.RHF(mf).kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.7816353049729151, 6)

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 4)

#        e1 = g_scanner(pmol.set_geom_('O  0. 0.0001 0.; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
#        e2 = g_scanner(pmol.set_geom_('O  0. -.0001 0.; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
#        self.assertAlmostEqual(abs(hess[0,:,1] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 4)
#
#        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.7571 0.587; 1  0. 0.757 0.587'))[1]
#        e2 = g_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.7569 0.587; 1  0. 0.757 0.587'))[1]
#        self.assertAlmostEqual(abs(hess[1,:,1] - (e2-e1)/2e-4*lib.param.BOHR).max(), 0, 4)
#
#        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.757 0.5871; 1  0. 0.757 0.587'))[1]
#        e2 = g_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.757 0.5869; 1  0. 0.757 0.587'))[1]
#        self.assertAlmostEqual(abs(hess[1,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 4)

    def test_ecp_hess(self):
        mol = gto.M(atom='Cu 0 0 0; H 0 0 1.5', basis='lanl2dz',
                    ecp={'Cu':'lanl2dz'}, verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        hess = hessian.RHF(mf).kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.20927804440983355, 6)

        mfs = mf.nuc_grad_method().as_scanner()
        e1 = mfs(mol.set_geom_('Cu 0 0  0.001; H 0 0 1.5'))[1]
        e2 = mfs(mol.set_geom_('Cu 0 0 -0.001; H 0 0 1.5'))[1]
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/0.002*lib.param.BOHR).max(), 0, 5)

#        mfs = mf.nuc_grad_method().as_scanner()
#        e1 = mfs(mol.set_geom_('Cu 0 0 0; H 0 0 1.5001'))[1]
#        e2 = mfs(mol.set_geom_('Cu 0 0 0; H 0 0 1.4999'))[1]
#        self.assertAlmostEqual(abs(hess[1,:,2] - (e1-e2)/0.0002*lib.param.BOHR).max(), 0, 5)


if __name__ == "__main__":
    print("Full Tests for RHF Hessian")
    unittest.main()
