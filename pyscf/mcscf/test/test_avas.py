#!/usr/bin/env python
# Copyright 2021 The PySCF Developers. All Rights Reserved.
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
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf.mcscf import avas


class KnownValues(unittest.TestCase):
    def test_avas(self):
        mol = gto.M(
            atom = '''
               H    0.000000,  0.500000,  1.5
               O    0.000000,  0.000000,  1.
               O    0.000000,  0.000000, -1.
               H    0.000000, -0.500000, -1.5''',
            basis = '6-31g',
            spin = 2,
            verbose = 7,
            output = '/dev/null'
        )
        mf = scf.RHF(mol).run(conv_tol=1e-10)
        ncas, nelecas, mo = avas.kernel(mf, 'O 2p')
        self.assertAlmostEqual(lib.fp(abs(mo)), 2.0834371806990823, 4)

        ncas, nelecas, mo = avas.kernel(mf, 'O 2p', openshell_option=3)
        self.assertAlmostEqual(lib.fp(abs(mo)), 1.886278150191051, 4)

        ncas, nelecas, mo = avas.kernel(mf.to_uhf(), 'O 2p')
        self.assertAlmostEqual(lib.fp(abs(mo)), 2.0950187018846607, 4)
        mol.stdout.close()

if __name__ == "__main__":
    print("Full Tests for mcscf.avas")
    unittest.main()
