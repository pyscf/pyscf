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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy as np

from pyscf import gto
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
import pyscf.pbc
pyscf.pbc.DEBUG = False


class KnowValues(unittest.TestCase):
    def test_pp_UKS(self):
        cell = pbcgto.Cell()

        cell.unit = 'A'
        cell.atom = '''
            Si    2.715348700    2.715348700    0.000000000;
            Si    2.715348700    0.000000000    2.715348700;
        '''
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'

        Lx = Ly = Lz = 5.430697500
        cell.a = np.diag([Lx,Ly,Lz])
        cell.mesh = np.array([17]*3)

        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()

        mf = pbcdft.UKS(cell)
        mf.xc = 'blyp'
        self.assertAlmostEqual(mf.scf(), -7.6058004283213396, 8)

        mf.xc = 'lda,vwn'
        self.assertAlmostEqual(mf.scf(), -7.6162130840535092, 8)

if __name__ == '__main__':
    print("Full Tests for pbc.dft.uks")
    unittest.main()
