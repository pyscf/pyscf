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

def setUpModule():
    global mol, mol1
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

    mol1 = gto.Mole()
    mol1.verbose = 5
    mol1.output = '/dev/null'
    mol1.atom = '''
    C              0.63540095    0.65803739   -0.00861418
    H              0.99205538   -0.35077261   -0.00861418
    H              0.99207379    1.16243558   -0.88226569
    H             -0.43459905    0.65805058   -0.00861418'''
    mol1.charge = 0
    mol1.spin = 1
    mol1.build()

def tearDownModule():
    global mol, mol1
    mol.stdout.close()
    mol1.stdout.close()
    del mol, mol1


class KnownValues(unittest.TestCase):
    def test_finite_diff_rohf_grad(self):
        mf = scf.ROHF(mol)
        mf.conv_tol = 1e-14
        e0 = mf.kernel()
        g = grad.ROHF(mf).kernel()
        mf_scanner = mf.as_scanner()

        e1 = mf_scanner('''O    0.   0.       0.
                        H     0.8  0.3      0.2
                        H    0.   -0.758   0.587
                        H    0.   0.757    0.587''')
        e2 = mf_scanner('''O    0.   0.       0.
                        H     0.8  0.3      0.2
                        H    0.   -0.756   0.587
                        H    0.   0.757    0.587''')
        self.assertAlmostEqual(g[2,1], (e2-e1)/2e-3*lib.param.BOHR, 5)

        e1 = mf_scanner('''O    0.   0.       0.
                        H     0.8  0.3      0.2
                        H    0.   -0.7571  0.587
                        H    0.   0.757    0.587''')
        e2 = mf_scanner('''O    0.   0.       0.
                        H     0.8  0.3      0.2
                        H    0.   -0.7569 0.587
                        H    0.   0.757    0.587''')
        self.assertAlmostEqual(g[2,1], (e2-e1)/2e-4*lib.param.BOHR, 7)

        mf = scf.ROHF(mol1)
        mf.conv_tol = 1e-14
        e0 = mf.kernel()
        g = grad.ROHF(mf).kernel()
        mf_scanner = mf.as_scanner()

        e1 = mf_scanner('''
C              0.63540095    0.65803739   -0.00861418
H              0.99205538   -0.35077261   -0.00861418
H              0.99207379    1.16143558   -0.88226569
H             -0.43459905    0.65805058   -0.00861418''')
        e2 = mf_scanner('''
C              0.63540095    0.65803739   -0.00861418
H              0.99205538   -0.35077261   -0.00861418
H              0.99207379    1.16343558   -0.88226569
H             -0.43459905    0.65805058   -0.00861418''')
        self.assertAlmostEqual(g[2,1], (e2-e1)/2e-3*lib.param.BOHR, 5)

        e1 = mf_scanner('''
C              0.63540095    0.65803739   -0.00861418
H              0.99205538   -0.35077261   -0.00861418
H              0.99207379    1.16233558   -0.88226569
H             -0.43459905    0.65805058   -0.00861418''')
        e2 = mf_scanner('''
C              0.63540095    0.65803739   -0.00861418
H              0.99205538   -0.35077261   -0.00861418
H              0.99207379    1.16253558   -0.88226569
H             -0.43459905    0.65805058   -0.00861418''')
        self.assertAlmostEqual(g[2,1], (e2-e1)/2e-4*lib.param.BOHR, 7)

    def test_finite_diff_df_rohf_grad(self):
        mf = scf.ROHF(mol).density_fit ()
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
        self.assertAlmostEqual(g[2,1], (e2-e1)/2e-3*lib.param.BOHR, 5)

        e1 = mf_scanner('''O    0.   0.       0.
                        H     0.8  0.3      0.2
                        H    0.   -0.7571  0.587
                        H    0.   0.757    0.587''')
        e2 = mf_scanner('''O    0.   0.       0.
                        H     0.8  0.3      0.2
                        H    0.   -0.7569 0.587
                        H    0.   0.757    0.587''')
        self.assertAlmostEqual(g[2,1], (e2-e1)/2e-4*lib.param.BOHR, 7)

        mf = scf.ROHF(mol1).density_fit ()
        mf.conv_tol = 1e-14
        e0 = mf.kernel()
        mf_grad = mf.nuc_grad_method ()
        g = mf_grad.kernel()
        mf_scanner = mf.as_scanner()

        e1 = mf_scanner('''
C              0.63540095    0.65803739   -0.00861418
H              0.99205538   -0.35077261   -0.00861418
H              0.99207379    1.16143558   -0.88226569
H             -0.43459905    0.65805058   -0.00861418''')
        e2 = mf_scanner('''
C              0.63540095    0.65803739   -0.00861418
H              0.99205538   -0.35077261   -0.00861418
H              0.99207379    1.16343558   -0.88226569
H             -0.43459905    0.65805058   -0.00861418''')
        self.assertAlmostEqual(g[2,1], (e2-e1)/2e-3*lib.param.BOHR, 5)

        e1 = mf_scanner('''
C              0.63540095    0.65803739   -0.00861418
H              0.99205538   -0.35077261   -0.00861418
H              0.99207379    1.16233558   -0.88226569
H             -0.43459905    0.65805058   -0.00861418''')
        e2 = mf_scanner('''
C              0.63540095    0.65803739   -0.00861418
H              0.99205538   -0.35077261   -0.00861418
H              0.99207379    1.16253558   -0.88226569
H             -0.43459905    0.65805058   -0.00861418''')
        self.assertAlmostEqual(g[2,1], (e2-e1)/2e-4*lib.param.BOHR, 7)

    def test_rohf_grad_same_to_rhf_grad(self):
        mol = gto.Mole()
        mol.atom = [
            ['O' , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ]
        mol.basis = '631g'
        mol.build()
        mf = scf.ROHF(mol)
        mf.conv_tol = 1e-14
        e0 = mf.scf()
        g1 = mf.Gradients().grad()
#[[ 0   0               -2.41134256e-02]
# [ 0   4.39690522e-03   1.20567128e-02]
# [ 0  -4.39690522e-03   1.20567128e-02]]
        self.assertAlmostEqual(lib.fp(g1), 0.0055116240804341972, 6)

    def test_rohf_grad(self):
        mol = gto.Mole()
        mol.atom = [
            ['O' , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ]
        mol.basis = '631g'
        mol.charge = 1
        mol.spin = 1
        mol.build()
        rhf = scf.ROHF(mol)
        rhf.conv_tol = 1e-14
        e0 = rhf.scf()
        g1 = rhf.Gradients().grad()
#[[ 0   0                0.0028164604]
# [ 0   4.28113397e-02  -0.0014082302]
# [ 0  -4.28113397e-02  -0.0014082302]]
        self.assertAlmostEqual(g1[0,2], 0.0028164604, 6)


if __name__ == "__main__":
    print("Full Tests for ROHF Gradients")
    unittest.main()
