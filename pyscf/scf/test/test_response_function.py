#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf.scf import _response_functions

class KnownValues(unittest.TestCase):
    def test_gks_nlc(self):
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
            charge = 1,
            spin = 1,
            basis = '631g')
        nao = mol.nao
        mf_uks = mol.UHF().run().to_uks()
        mf_uks.xc = 'wb97mv'
        mf_uks.nlcgrids.level = 0

        dm = mf_uks.make_rdm1()
        dm1 = np.random.rand(2, nao, nao)
        vind = mf_uks.gen_response(with_nlc=True)
        ref = scipy.linalg.block_diag(*vind(dm1))

        mf_gks = mf_uks.to_gks()
        vind = mf_gks.gen_response(with_nlc=True)
        v = vind(scipy.linalg.block_diag(*dm1))
        self.assertAlmostEqual(abs(v - ref).max(), 0, 12)

if __name__ == "__main__":
    print("Full Tests for response_functions")
    unittest.main()
