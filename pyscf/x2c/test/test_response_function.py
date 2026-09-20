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
from pyscf import gto
from pyscf.x2c import x2c, dft
from pyscf.x2c import _response_functions  # noqa

class KnownValues(unittest.TestCase):
    def _check_dm0_response(self, mf, dm1, with_nlc):
        dm0 = mf.make_rdm1()
        vind_mo = mf.gen_response(mf.mo_coeff, mf.mo_occ, with_nlc=with_nlc)
        vind_dm0 = mf.gen_response(dm0=dm0, with_nlc=with_nlc)
        self.assertAlmostEqual(abs(vind_mo(dm1) - vind_dm0(dm1)).max(), 0, 10)

    def test_x2chf_response_dm0(self):
        mol = gto.M(
            verbose = 0,
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
            charge = 1,
            spin = 1,
            basis = '631g')

        mf = x2c.UHF(mol).run()
        n2c = mf.mo_coeff.shape[0]
        dm1 = np.random.rand(n2c, n2c)
        dm1 = dm1 + dm1.T
        self._check_dm0_response(mf, dm1, with_nlc=True)

        mf = dft.UKS(mol, xc='pbe').run()
        n2c = mf.mo_coeff.shape[0]
        dm1 = np.random.rand(n2c, n2c)
        dm1 = dm1 + dm1.T
        self._check_dm0_response(mf, dm1, with_nlc=False)

if __name__ == "__main__":
    print("Full Tests for x2c response_functions")
    unittest.main()
