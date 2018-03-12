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

import unittest
import numpy
import scipy.linalg
import tempfile
from pyscf import gto
from pyscf import scf
from pyscf import fci

class KnowValues(unittest.TestCase):
    def test_davidson(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [['H', (0,0,i)] for i in range(8)]
        mol.basis = {'H': 'sto-3g'}
        mol.build()
        mf = scf.RHF(mol)
        mf.scf()
        myfci = fci.FCI(mol, mf.mo_coeff)
        myfci.max_memory = .001
        myfci.max_cycle = 100
        e = myfci.kernel()[0]
        self.assertAlmostEqual(e, -11.579978414933732+mol.energy_nuc(), 9)

if __name__ == "__main__":
    print("Full Tests for linalg_helper")
    unittest.main()
