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
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbchf
import pyscf.pbc.mp
import pyscf.pbc.mp.kmp2


cell = pbcgto.Cell()
cell.unit = 'B'
L = 7
cell.atom.extend([['Be', (L/2.,  L/2., L/2.)]])
cell.a = 7 * np.identity(3)
cell.a[1,0] = 5.0

cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade-q2'
cell.mesh = [12]*3
cell.verbose = 5
cell.output = '/dev/null'
cell.build()

def run_kcell(cell, nk):
    abs_kpts = cell.make_kpts(nk, wrap_around=True)
    kmf = pbchf.KRHF(cell, abs_kpts, exxdiv=None)
    kmf.conv_tol = 1e-12
    ekpt = kmf.scf()
    mp = pyscf.pbc.mp.kmp2.KMP2(kmf).run()
    return ekpt, mp.e_corr

class KnownValues(unittest.TestCase):
    def test_111(self):
        nk = (1, 1, 1)
        hf_111 = -0.79932851980207353
        mp_111 = -2.4124398409652723e-05
        escf, emp = run_kcell(cell,nk)
        self.assertAlmostEqual(escf, hf_111, 9)
        self.assertAlmostEqual(emp, mp_111, 6)

    def test_311_high_cost(self):
        nk = (3, 1, 1)
        hf_311 = -0.85656225114216422
        mp_311 = -8.3491016166387105e-06
        escf, emp = run_kcell(cell,nk)
        self.assertAlmostEqual(escf,hf_311, 9)
        self.assertAlmostEqual(emp, mp_311, 6)

if __name__ == '__main__':
    print("Full kpoint test")
    unittest.main()

