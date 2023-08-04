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
from pyscf import grad
from pyscf.ci import ucisd
from pyscf.grad import ucisd as ucisd_grad

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
    def test_cisd_grad(self):
        myci = ucisd.UCISD(mf)
        myci.max_memory = 1
        myci.conv_tol = 1e-10
        myci.kernel()
        g1 = ucisd_grad.Gradients(myci).kernel(myci.ci)
# O     0.0000000000    -0.0000000000     0.1456473095
# H    -0.0000000000     0.1107223084    -0.0728236548
# H     0.0000000000    -0.1107223084    -0.0728236548
        self.assertAlmostEqual(lib.fp(g1), -0.22651925227633429, 6)

    def test_cisd_grad_finite_diff(self):
        ci_scanner = scf.UHF(mol).set(conv_tol=1e-14).apply(ucisd.UCISD).as_scanner()
        ci_scanner(mol)
        g1 = ci_scanner.nuc_grad_method().kernel()

        mol1 = gto.M(
            verbose = 0,
            atom = '''
            O    0.   0.       0.001
            H    0.  -0.757    0.587
            H    0.   0.757    0.587
            ''',
            basis = '631g',
            spin = 2)
        e0 = ci_scanner(mol1)
        mol1 = gto.M(
            verbose = 0,
            atom = '''
            O    0.   0.      -0.001
            H    0.  -0.757    0.587
            H    0.   0.757    0.587
            ''',
            basis = '631g',
            spin = 2)
        e1 = ci_scanner(mol1)
        self.assertAlmostEqual(g1[0,2], (e0-e1)*500*lib.param.BOHR, 5)

    def test_frozen(self):
        myci = ucisd.UCISD(mf)
        myci.frozen = [0,1,10,11,12]
        myci.max_memory = 1
        myci.kernel()
        g1 = myci.Gradients().kernel(myci.ci)
# O    -0.0000000000    -0.0000000000     0.1540204772
# H     0.0000000000     0.1144196177    -0.0770102386
# H     0.0000000000    -0.1144196177    -0.0770102386
        self.assertAlmostEqual(lib.fp(g1), -0.23578589551312196, 6)


if __name__ == "__main__":
    print("Tests for UCISD gradients")
    unittest.main()
