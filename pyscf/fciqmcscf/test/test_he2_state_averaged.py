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
from pyscf import fciqmcscf

b = 1.4
mol = gto.Mole()

mol.build(
        verbose = 0,
output = None,
atom = [['He',(  0.000000,  0.000000, -b/2)],
        ['He',(  0.000000,  0.000000,  b)]],
basis = {'He': 'cc-pvdz'},
symmetry = False,
#symmetry_subgroup = 'D2h',
)

m = scf.RHF(mol)
m.conv_tol = 1e-9
m.scf()

class KnowValues(unittest.TestCase):
    def test_mc2step_7o4e_fciqmc_4states(self):
        mc = mcscf.CASSCF(m, 7, 4)
        mc.max_cycle_macro = 10
#        mc.fcisolver = fciqmcscf.FCIQMCCI(mol)
#        mc.fcisolver.RDMSamples = 2000
#        mc.fcisolver.maxwalkers = 3000
        mc.fcisolver.state_weights = [1.00]

        emc = mc.mc2step()[0]
        print('Final energy:', emc)

if __name__ == "__main__":
    unittest.main()
