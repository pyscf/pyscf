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
from pyscf import grad

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
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_rhf_grad(self):
        g_scan = scf.RHF(mol).nuc_grad_method().as_scanner()
        g = g_scan(mol)[1]
        self.assertAlmostEqual(lib.finger(g), 0.0055116240804341972, 9)

        mfs = g_scan.base.as_scanner()
        e1 = mfs('O  0.  0. -0.001; H  0.  -0.757  0.587; H  0.  0.757   0.587')
        e2 = mfs('O  0.  0.  0.001; H  0.  -0.757  0.587; H  0.  0.757   0.587')
        self.assertAlmostEqual(g[0,2], (e2-e1)/0.002*lib.param.BOHR, 6)

    def test_finite_diff_x2c_rhf_grad(self):
        mf = scf.RHF(mol).x2c()
        mf.conv_tol = 1e-14
        e0 = mf.kernel()
        g = grad.RHF(mf).kernel()
        self.assertAlmostEqual(lib.finger(g), 0.0056363502746766807, 6)

        mf_scanner = mf.as_scanner()
        pmol = mol.copy()
        e1 = mf_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

        #e1 = mf_scanner(pmol.set_geom_('O  0.  1e-5 0.; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        #e2 = mf_scanner(pmol.set_geom_('O  0. -1e-5 0.; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        #self.assertAlmostEqual(g[0,1], (e1-e2)/2e-5*lib.param.BOHR, 5)

        e1 = mf_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.7571 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.7569 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[1,1], (e2-e1)/2e-4*lib.param.BOHR, 5)

        #e1 = mf_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.757 0.5871; 1  0. 0.757 0.587'))
        #e2 = mf_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.757 0.5869; 1  0. 0.757 0.587'))
        #self.assertAlmostEqual(g[1,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

    def test_finite_diff_rhf_grad(self):
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-14
        e0 = mf.kernel()
        g = grad.RHF(mf).kernel(atmlst=range(mol.natm))
        self.assertAlmostEqual(lib.finger(g), 0.0055115512502467556, 6)

        mf_scanner = mf.as_scanner()
        pmol = mol.copy()
        e1 = mf_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

        #e1 = mf_scanner(pmol.set_geom_('O  0.  1e-5 0.; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        #e2 = mf_scanner(pmol.set_geom_('O  0. -1e-5 0.; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        #self.assertAlmostEqual(g[0,1], (e1-e2)/2e-5*lib.param.BOHR, 5)

        #e1 = mf_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.7571 0.587; 1  0. 0.757 0.587'))
        #e2 = mf_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.7569 0.587; 1  0. 0.757 0.587'))
        #self.assertAlmostEqual(g[1,1], (e2-e1)/2e-4*lib.param.BOHR, 5)

        #e1 = mf_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.757 0.5871; 1  0. 0.757 0.587'))
        #e2 = mf_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.757 0.5869; 1  0. 0.757 0.587'))
        #self.assertAlmostEqual(g[1,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

    def test_ecp_grad(self):
        mol = gto.M(atom='Cu 0 0 0; H 0 0 1.5', basis='lanl2dz',
                    ecp='lanl2dz', verbose=0)
        mf = scf.RHF(mol)
        g_scan = mf.nuc_grad_method().as_scanner().as_scanner()
        g = g_scan(mol.atom)[1]
        self.assertAlmostEqual(lib.finger(g), -0.012310573162997052, 9)

        mfs = mf.as_scanner()
        e1 = mfs(mol.set_geom_('Cu 0 0 -0.001; H 0 0 1.5'))
        e2 = mfs(mol.set_geom_('Cu 0 0  0.001; H 0 0 1.5'))
        self.assertAlmostEqual(g[0,2], (e2-e1)/0.002*lib.param.BOHR, 6)


if __name__ == "__main__":
    print("Full Tests for RHF Gradients")
    unittest.main()

