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
from pyscf import gto, scf
from pyscf.prop import efg

class KnownValues(unittest.TestCase):
    def test_dhf_nr_limit(self):
        mol = gto.M(atom='''
             H    .8    0.    0.
             H    0.    .5    0.''',
            basis='ccpvdz')
        with lib.temporary_env(lib.param, LIGHT_SPEED=5000):
            r = scf.DHF(mol).run().EFG()
            nr = scf.RHF(mol).run().EFG()
        self.assertAlmostEqual(abs(r - nr).max(), 0, 7)

if __name__ == "__main__":
    print("Full Tests for DHF EFGs")
    unittest.main()

