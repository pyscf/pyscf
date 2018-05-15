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
from pyscf import lib
from pyscf.dft import dks

class KnownValues(unittest.TestCase):
    def test_dks_lda(self):
        mol = gto.Mole()
        mol.atom = [['Ne',(0.,0.,0.)]]
        mol.basis = 'uncsto3g'
        mol.verbose = 7
        mol.output = '/dev/null'
        mol.build()
        mf = dks.DKS(mol)
        mf.xc = 'lda,'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -126.041808355268, 9)
        mol.stdout.close()

    def test_x2c_uks_lda(self):
        mol = gto.Mole()
        mol.atom = [['Ne',(0.,0.,0.)]]
        mol.basis = 'uncsto3g'
        mol.verbose = 7
        mol.output = '/dev/null'
        mol.build()
        mf = dks.DKS(mol).x2c()
        mf.xc = 'lda,'
        eks4 = mf.kernel()
        self.assertAlmostEqual(eks4, -126.03378903205831, 9)
        mol.stdout.close()


if __name__ == "__main__":
    print("Test DKS")
    unittest.main()

