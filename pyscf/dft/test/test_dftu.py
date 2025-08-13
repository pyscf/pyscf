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
from pyscf.dft import rkspu, ukspu

class KnownValues(unittest.TestCase):
    def test_RKSpU_linear_response(self):
        mol = gto.M(atom='''
        O  0.   0.       0.
        H  0.   -0.757   0.587
        H  0.   0.757    0.587''', basis='6-31g')
        mf = rkspu.RKSpU(mol, xc='pbe', U_idx=['O 2p'], U_val=[3.5])
        mf.run()
        u0 = rkspu.linear_response_u(mf)
        assert abs(u0 - 5.8926) < 1e-2

    def test_UKSpU_linear_response(self):
        mol = gto.M(atom='''
        O  0.   0.       0.
        H  0.   -0.757   0.587
        H  0.   0.757    0.587''', basis='6-31g')
        mf = ukspu.UKSpU(mol, xc='pbe', U_idx=['O 2p'], U_val=[3.5])
        mf.run()
        u0 = ukspu.linear_response_u(mf)
        assert abs(u0 - 5.8926) < 1e-2

if __name__ == '__main__':
    print("Full Tests for dft+U")
    unittest.main()
