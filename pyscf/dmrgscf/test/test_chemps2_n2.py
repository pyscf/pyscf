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
from pyscf import scf
from pyscf import mcscf
from pyscf import dmrgscf

b = 1.4
mol = gto.Mole()
mol.build(
verbose = 0,
output = None,
atom = [
    ['N',(  0.000000,  0.000000, -b/2)],
    ['N',(  0.000000,  0.000000,  b/2)], ],
basis = {'N': 'ccpvdz', },
)
m = scf.RHF(mol)
m.conv_tol = 1e-9
m.scf()


class KnowValues(unittest.TestCase):
    def test_mc2step_4o4e(self):
        mc = mcscf.CASSCF(m, 4, 4)
        mc.fcisolver = dmrgscf.CheMPS2(mol)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)

    def test_mc2step_6o6e(self):
        mc = mcscf.CASSCF(m, 6, 6)
        mc.fcisolver = dmrgscf.CheMPS2(mol)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.980105451388, 7)

if __name__ == "__main__":
    print("Full Tests for N2")
    unittest.main()

