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
from functools import reduce
import numpy
from pyscf import gto
from pyscf import lib
from pyscf import scf
from pyscf.lo import iao

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.atom = '''
         O    0.   0.       0
         h    0.   -0.757   0.587
         h    0.   0.757    0.587'''
    mol.basis = 'unc-sto3g'
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_fast_iao_mulliken_pop(self):
        mf = scf.RHF(mol).run()
        a = iao.iao(mol, mf.mo_coeff[:,mf.mo_occ>0])
        p,chg = iao.fast_iao_mullikan_pop(mol, mf.make_rdm1(), a)
        self.assertAlmostEqual(lib.finger(p), 0.56812564587009806, 5)

        mf = scf.UHF(mol).run()
        p,chg = iao.fast_iao_mullikan_pop(mol, mf.make_rdm1(), a)
        self.assertAlmostEqual(lib.finger(p[0]+p[1]), 0.56812564587009806, 5)


if __name__ == "__main__":
    print("TODO: Test iao")
    unittest.main()
