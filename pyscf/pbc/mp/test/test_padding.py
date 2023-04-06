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
from pyscf.pbc import scf as pbcscf
import pyscf.pbc.mp
import pyscf.pbc.mp.kmp2
from pyscf.scf.addons import remove_linear_dep_


def setUpModule():
    global cell
    a0 = 1.78339987 * 0.96  # squeezed
    atom = "C 0 0 0; C %.10f %.10f %.10f" % (a0*0.5, a0*0.5, a0*0.5)
    a = np.asarray(
        [[0., a0, a0],
        [a0, 0., a0],
        [a0, a0, 0.]])
    basis = "gth-dzvp"
    pseudo = "gth-pade"
    ke_cutoff = 100
    cell = pbcgto.Cell(atom=atom, a=a, basis=basis, pseudo=pseudo,
                       ke_cutoff=ke_cutoff)
    cell.verbose = 4
    cell.output = '/dev/null'
    cell.mesh = [17] * 3
    cell.build()

def tearDownModule():
    global cell
    cell.stdout.close()
    del cell

def run_kcell(cell, nk):
    abs_kpts = cell.make_kpts(nk, wrap_around=True)
    kmf = pbcscf.KRHF(cell, abs_kpts)
    kmf = remove_linear_dep_(kmf, threshold=1e-5, lindep=1e-6)
    kmf.conv_tol = 1e-12
    ekpt = kmf.scf()
    mp = pyscf.pbc.mp.kmp2.KMP2(kmf).run()
    return ekpt, mp.e_corr

class KnownValues(unittest.TestCase):
    def test_211(self):
        cell1 = cell.copy()
        cell1.basis = 'gth-dzv'
        cell1.mesh = [17] * 3
        cell1.build()
        nk = [2, 1, 1]
        escf, emp = run_kcell(cell1, nk)
        self.assertAlmostEqual(escf, -10.551651367521986, 7)
        self.assertAlmostEqual(emp , -0.1514005714425273, 7)

    def test_221_high_cost(self):
        cell1 = cell.copy()
        cell1.basis = 'gth-dzv'
        cell1.mesh = [17] * 3
        cell1.build()

        nk = [2, 2, 1]
        abs_kpts = cell1.make_kpts(nk, wrap_around=True)
        kmf = pbcscf.KRHF(cell1, abs_kpts)
        kmf = remove_linear_dep_(kmf, threshold=1e-4, lindep=1e-6)
        ekpt = kmf.scf()
        mp = pyscf.pbc.mp.kmp2.KMP2(kmf).run()

        self.assertAlmostEqual(ekpt,      -10.835614361742607, 8)
        self.assertAlmostEqual(mp.e_corr, -0.1522294774708119, 7)

    def test_222_high_cost(self):
        nk = (2, 2, 2)
        escf, emp = run_kcell(cell,nk)
        self.assertAlmostEqual(escf, -11.0152342492995, 8)
        self.assertAlmostEqual(emp, -0.233433093787577, 7)


if __name__ == '__main__':
    print("padding test")
    unittest.main()
