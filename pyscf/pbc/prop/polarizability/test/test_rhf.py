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
from pyscf import lib
from pyscf.pbc import gto, scf
from pyscf.pbc.prop.polarizability import rhf

cell = gto.Cell()
cell.atom = """H  0.0 0.0 0.0
               F  0.9 0.0 0.0
            """
cell.basis = 'sto-3g'
cell.a = [[2.82, 0, 0], [0, 2.82, 0], [0, 0, 2.82]]
cell.dimension = 1
cell.precision = 1e-10
cell.output = '/dev/null'
cell.build()

kpts = cell.make_kpts([16,1,1])
kmf = scf.KRHF(cell, kpts=kpts, exxdiv="ewald").density_fit()
kmf.kernel()
polar = rhf.Polarizability(kmf, kpts)

def tearDownModule():
    global cell, kmf
    del cell, kmf

class KnownValues(unittest.TestCase):
    def test_dip_moment(self):
        dip = polar.dip_moment()
        self.assertAlmostEqual(dip[0], -0.571816066, 6)

    def test_polarizability(self):
        e2 = polar.polarizability()
        self.assertAlmostEqual(e2[0,0], 3.83895494, 6)
        self.assertAlmostEqual(e2[1,1], 2.62071624e-03, 6)
        self.assertAlmostEqual(e2[2,2], e2[1,1], 9)

    def test_polarizability_with_freq(self):
        e2_0 = polar.polarizability()
        e2 = polar.polarizability_with_freq(0.)
        self.assertTrue(np.allclose(e2_0, e2))
        e2_p = polar.polarizability_with_freq(0.1)
        e2_m = polar.polarizability_with_freq(-0.1)
        self.assertTrue(np.allclose(e2_p, e2_m))
        self.assertAlmostEqual(e2_p[0,0], 3.90004305, 6)
        self.assertAlmostEqual(e2_p[1,1], 2.70994061e-03, 6)
        self.assertAlmostEqual(e2_p[2,2], e2_p[1,1], 9)

    def test_hyper_polarizability(self):
        e3 = polar.hyper_polarizability()
        self.assertAlmostEqual(e3[0,0,0], 3.02691297, 6)
        self.assertAlmostEqual(e3[0,1,1], -1.97889278e-02, 6)
        self.assertAlmostEqual(e3[0,2,2], e3[0,1,1], 9)

if __name__ == "__main__":
    print("Full Tests for krhf polarizability")
    unittest.main()
