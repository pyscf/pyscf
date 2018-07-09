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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import unittest
from pyscf import gto
from pyscf import scf
from pyscf.x2c import x2c

mol = gto.M(
    verbose = 5,
    output = '/dev/null',
    atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)


class KnownValues(unittest.TestCase):
    def test_sfx2c1e(self):
        myx2c = scf.RHF(mol).sfx2c1e()
        myx2c.with_x2c.xuncontract = False
        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.081765429967618, 9)

        myx2c.with_x2c.xuncontract = True
        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.075429077955874, 9)

        myx2c.with_x2c.approx = 'ATOM1E'
        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.075429682026396, 9)

    def test_sfx2c1e_cart(self):
        pmol = mol.copy()
        pmol.cart = True
        myx2c = scf.RHF(pmol).sfx2c1e()
        myx2c.with_x2c.xuncontract = False
        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.081452837461342, 9)

    def test_x2c1e(self):
        myx2c = x2c.UHF(mol)
        myx2c.with_x2c.xuncontract = False
        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.08176796102066, 9)

        myx2c.with_x2c.xuncontract = True
        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.075431226329414, 9)

        myx2c.with_x2c.approx = 'ATOM1E'
        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.07543183416206, 9)


if __name__ == "__main__":
    print("Full Tests for x2c")
    unittest.main()


