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
from pyscf import lib
from pyscf import gto
from pyscf import ao2mo
mol = gto.Mole()
mol.verbose = 5
mol.output = '/dev/null'
mol.atom = [
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)]]
mol.basis = 'cc-pvdz'
mol.build()

def tearDownModule():
    global mol
    del mol

class KnownValues(unittest.TestCase):
    def test_r_outcore_eri(self):
        n2c = mol.nao_2c()
        numpy.random.seed(1)
        mo = numpy.random.random((n2c,n2c)) + numpy.random.random((n2c,n2c))*1j
        eri1 = ao2mo.kernel(mol, mo, intor='int2e_spinor', max_memory=10, ioblk_size=5)
        self.assertAlmostEqual(lib.finger(eri1), -3427.91441754+2484.76429481j, 8)

if __name__ == '__main__':
    print('Full Tests for r_outcore')
    unittest.main()

