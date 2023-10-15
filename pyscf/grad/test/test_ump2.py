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
from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import mp
from pyscf import grad
from pyscf.grad import ump2 as ump2_grad

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.spin = 2
    mol.basis = '631g'
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol_grad = 1e-8
    mf.kernel()

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


class KnownValues(unittest.TestCase):
    def test_mp2_grad(self):
        pt = mp.MP2(mf)
        pt.kernel()
        g1 = pt.nuc_grad_method().kernel(pt.t2, atmlst=[0,1,2])
# O     0.0000000000    -0.0000000000     0.1436990190
# H    -0.0000000000     0.1097329294    -0.0718495095
# H    -0.0000000000    -0.1097329294    -0.0718495095
        self.assertAlmostEqual(lib.fp(g1), -0.2241809640361207, 6)

    def test_mp2_grad_finite_diff(self):
        mol = gto.M(
            verbose = 0,
            atom = 'H 0 0 0; H 0 0 1.706',
            basis = '631g',
            unit='Bohr')
        mp_scanner = scf.UHF(mol).set(conv_tol=1e-14).apply(mp.MP2).as_scanner()
        e0 = mp_scanner(mol)
        mol = gto.M(
            verbose = 0,
            atom = 'H 0 0 0; H 0 0 1.704',
            basis = '631g',
            unit='Bohr')
        e1 = mp_scanner(mol)
        mol = gto.M(
            verbose = 0,
            atom = 'H 0 0 0; H 0 0 1.705',
            basis = '631g',
            unit='Bohr')
        mp_scanner(mol)
        g1 = mp_scanner.nuc_grad_method().kernel()
        self.assertAlmostEqual(g1[0,2], (e1-e0)*500, 6)

    def test_frozen(self):
        pt = mp.MP2(mf)
        pt.frozen = [0,1,10,11,12]
        pt.max_memory = 1
        pt.kernel()
        g1 = ump2_grad.Gradients(pt).kernel(pt.t2)
# O    -0.0000000000    -0.0000000000     0.1454782514
# H     0.0000000000     0.1092558730    -0.0727391257
# H    -0.0000000000    -0.1092558730    -0.0727391257
        self.assertAlmostEqual(lib.fp(g1), -0.22437278030057645, 6)

    def test_as_scanner(self):
        pt = mp.MP2(mf)
        pt.frozen = [0,1,10,11,12]
        gscan = pt.nuc_grad_method().as_scanner()
        e, g1 = gscan(mol)
        self.assertTrue(gscan.converged)
        self.assertAlmostEqual(e, -75.759091758835353, 9)
        self.assertAlmostEqual(lib.fp(g1), -0.22437278030057645, 6)


if __name__ == "__main__":
    print("Tests for MP2 gradients")
    unittest.main()
