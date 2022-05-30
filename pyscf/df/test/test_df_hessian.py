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

import os
import unittest
import tempfile
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import df
from pyscf import hessian

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.build(
        verbose = 0,
        atom = '''O     0    0.       0.
                  1     0    -0.757   0.587
                  1     0    0.757    0.587''',
        basis = '6-31g',
    )

def tearDownModule():
    global mol
    del mol


class KnownValues(unittest.TestCase):
    def test_rhf_hess(self):
        gref = scf.RHF(mol).run().Hessian().kernel()
        g1 = scf.RHF(mol).density_fit().run().Hessian().kernel()
        self.assertAlmostEqual(abs(gref - g1).max(), 0, 3)

    def test_uks_hess(self):
        gref = mol.UKS.run(xc='b3lyp').Hessian().kernel()
        g1 = mol.UKS.density_fit().run(xc='b3lyp').Hessian().kernel()
        self.assertAlmostEqual(abs(gref - g1).max(), 0, 3)
#
if __name__ == "__main__":
    print("Full Tests for df.hessian")
    unittest.main()

