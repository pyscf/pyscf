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


if __name__ == "__main__":
    print("Full Tests for ROKS Gradients")
    unittest.main()
