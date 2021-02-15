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
from pyscf import mrpt
dmrgscf.settings.MPIPREFIX = 'mpirun -n 4'

b = 1.4
mol = gto.M(verbose = 0,
atom = [
    ['N', (0, 0, -b/2)],
    ['N', (0, 0,  b/2)], ],
basis = '631g')
m = scf.RHF(mol)
m.conv_tol = 1e-12
m.scf()

mc = dmrgscf.dmrgci.DMRGSCF(m, 4, 4)
mc.kernel()


class KnowValues(unittest.TestCase):
#    def test_nevpt2_with_4pdm(self):
#        e = mrpt.NEVPT(mc).kernel()
#        self.assertAlmostEqual(e, -0.14058373193902649, 6)

    def test_nevpt2_without_4pdm(self):
        e = mrpt.NEVPT(mc).compress_approx(maxM=5000).kernel()
        self.assertAlmostEqual(e, -0.14058324516302972, 6)

if __name__ == "__main__":
    print("Full Tests for N2")
    unittest.main()


