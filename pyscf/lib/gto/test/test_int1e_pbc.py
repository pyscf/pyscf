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

mol = gto.Mole()
mol.atom = '''
He     .5    .5      -.5
He    1.     .2       .3
He     .1   -.1       .1 '''
mol.basis = {'He': [(0, (.5, 1)),
                    (1, (.6, 1)),
                    (2, (.8, 1))]}
mol.build()


class KnowValues(unittest.TestCase):
    def test_cint1e_r2_origi(self):
        ref = mol.intor('cint1e_r2_origi_sph')
        dat = mol.intor('cint1e_pbc_r2_origi_sph')
        self.assertTrue(numpy.allclose(ref, dat))

    def test_cint1e_r4_origi(self):
        ref = mol.intor('cint1e_r4_origi_sph')
        dat = mol.intor('cint1e_pbc_r4_origi_sph')
        self.assertTrue(numpy.allclose(ref, dat))

    def test_cint3c1e_r2_origk(self):
        ref = mol.intor('cint3c1e_r2_origk_sph')
        dat = mol.intor('cint3c1e_pbc_r2_origk_sph')
        self.assertTrue(numpy.allclose(ref, dat))

    def test_cint3c1e_r4_origk(self):
        ref = mol.intor('cint3c1e_r4_origk_sph')
        dat = mol.intor('cint3c1e_pbc_r4_origk_sph')
        self.assertTrue(numpy.allclose(ref, dat))

    def test_cint3c1e_r6_origk(self):
        ref = mol.intor('cint3c1e_r6_origk_sph')
        dat = mol.intor('cint3c1e_pbc_r6_origk_sph')
        self.assertTrue(numpy.allclose(ref, dat))

if __name__ == '__main__':
    print('Full Tests for int1e_pbc')
    unittest.main()
