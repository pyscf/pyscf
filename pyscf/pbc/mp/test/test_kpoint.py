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
    kmf = pbchf.KRHF(cell, abs_kpts)
    kmf.conv_tol = 1e-12
    ekpt = kmf.scf()
    mp = pyscf.pbc.mp.kmp2.KMP2(kmf).run()
    return ekpt, mp.e_corr

def run_kcell_complex(cell, nk):
    abs_kpts = cell.make_kpts(nk, wrap_around=True)
    kmf = pbchf.KRHF(cell, abs_kpts)
    kmf.conv_tol = 1e-12
    ekpt = kmf.scf()
    kmf.mo_coeff = [kmf.mo_coeff[i].astype(np.complex128) for i in range(np.prod(nk))]
    mp = pyscf.pbc.mp.kmp2.KMP2(kmf).run()
    return ekpt, mp.e_corr

class KnownValues(unittest.TestCase):
    def test_111(self):
        nk = (1, 1, 1)
        escf, emp = run_kcell(cell,nk)
        self.assertAlmostEqual(escf, -1.2061049658473704, 9)
        self.assertAlmostEqual(emp, -5.44597932944397e-06, 9)
        escf, emp = run_kcell_complex(cell,nk)
        self.assertAlmostEqual(emp, -5.44597932944397e-06, 9)

    def test_311_high_cost(self):
        nk = (3, 1, 1)
        escf, emp = run_kcell(cell,nk)
        self.assertAlmostEqual(escf, -1.0585001200928885, 9)
        self.assertAlmostEqual(emp, -7.9832274354253814e-06, 9)

if __name__ == '__main__':
    print("Full kpoint test")
    unittest.main()

