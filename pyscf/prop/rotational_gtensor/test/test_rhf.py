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
from pyscf import gto
from pyscf.data import nist
from pyscf.prop.rotational_gtensor import rhf

class KnowValues(unittest.TestCase):
    def test_nuc_contribution(self):
        mol = gto.M(atom='''H  ,  0.   0.   0.
                            F  ,  0.   0.   0.917
                         ''')
        nuc = rhf.nuc(mol)
        self.assertAlmostEqual(nuc[0,0], 0.972976229429035, 9)

        mol = gto.M(atom='''C  ,  0.   0.   0.
                            O  ,  0.   0.   1.1283
                         ''')
        nuc = rhf.nuc(mol)
        self.assertAlmostEqual(nuc[0,0], 0.503388273805359, 9)


if __name__ == "__main__":
    print("Full Tests of RHF rotational g-tensor")
    unittest.main()
