# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
from pyscf import gto
from pyscf import scf
from pyscf.sgx import sgx


class KnownValues(unittest.TestCase):
    def test_reset(self):
        mol = gto.M(atom='He')
        mol1 = gto.M(atom='C')
        mf = scf.RHF(mol).COSX()
        mf.reset(mol1)
        self.assertTrue(mf.mol is mol1)
        self.assertTrue(mf.with_df.mol is mol1)
        self.assertEqual(mf.undo_sgx().__class__.__name__, 'RHF')

    def test_sgx_scf(self):
        mol = gto.Mole()
        mol.build(
            atom = [["O" , (0. , 0.     , 0.)],
                    [1   , (0. , -0.757 , 0.587)],
                    [1   , (0. , 0.757  , 0.587)] ],
            basis = 'ccpvdz',
        )
        method = sgx.sgx_fit(scf.RHF(mol), 'weigend')
        method.with_df.dfj = True
        energy = method.scf()
        self.assertAlmostEqual(energy, -76.02686422219752, 9)

if __name__ == "__main__":
    print("Full Tests for SGX")
    unittest.main()
