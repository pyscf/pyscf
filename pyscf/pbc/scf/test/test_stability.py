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
from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.scf import stability

def setUpModule():
    global cell, kpts
    L = 4
    n = 15
    cell = pgto.Cell()
    cell.build(unit = 'B',
               verbose = 5,
               output = '/dev/null',
               a = ((L,0,0),(0,L,0),(0,0,L)),
               mesh = [n,n,n],
               atom = [['He', (L/2.-.5,L/2.,L/2.-.5)],
                       ['He', (L/2.   ,L/2.,L/2.+.5)]],
               basis = { 'He': [[0, (0.8, 1.0)],
                                [0, (1.0, 1.0)],
                                [0, (1.2, 1.0)]]})

    numpy.random.seed(4)
    kpts = numpy.random.random((1,3))

def tearDownModule():
    global cell
    cell.stdout.close()
    del cell


class KnownValues(unittest.TestCase):
    def test_hf_stability(self):
        mf = pscf.RHF(cell, exxdiv='ewald').run(conv_tol=1e-12)
        mo_i, mo_e = mf.stability(internal=True, external=True)
        self.assertAlmostEqual(abs(mf.mo_coeff-mo_i).max(), 0, 9)

    def test_khf_stability(self):
        kmf = pscf.KRHF(cell, kpts, exxdiv='ewald').run(conv_tol=1e-12)
        mo_i, mo_e = kmf.stability(internal=True, external=True)
        self.assertAlmostEqual(abs(kmf.mo_coeff[0]-mo_i[0]).max(), 0, 9)

        hop2, hdiag2 = stability._gen_hop_rhf_external(kmf)
        self.assertAlmostEqual(lib.fp(hdiag2), 18.528134783454508, 6)
        self.assertAlmostEqual(lib.fp(hop2(hdiag2)), 108.99683506471919, 5)

    def test_uhf_stability(self):
        umf = pscf.UHF(cell, exxdiv='ewald').run(conv_tol=1e-12)
        mo_i, mo_e = umf.stability(internal=True, external=True)
        self.assertAlmostEqual(abs(umf.mo_coeff[0]-mo_i[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(umf.mo_coeff[1]-mo_i[1]).max(), 0, 9)

    def test_kuhf_stability(self):
        kumf = pscf.KUHF(cell, kpts, exxdiv='ewald').run(conv_tol=1e-12)
        mo_i, mo_e = kumf.stability(internal=True, external=True)
        self.assertAlmostEqual(abs(kumf.mo_coeff[0][0]-mo_i[0][0]).max(), 0, 9)
        self.assertAlmostEqual(abs(kumf.mo_coeff[1][0]-mo_i[1][0]).max(), 0, 9)

        hop2, hdiag2 = stability._gen_hop_uhf_external(kumf)
        self.assertAlmostEqual(lib.fp(hdiag2), 10.977759629315884, 7)
        self.assertAlmostEqual(lib.fp(hop2(hdiag2)), 86.425042652868, 5)

    def test_rotate_mo(self):
        numpy.random.seed(4)
        def occarray(nmo, nocc):
            occ = numpy.zeros(nmo)
            occ[:nocc] = 2
            return occ
        mo_coeff = [numpy.random.random((8,8)),
                    numpy.random.random((8,7)),
                    numpy.random.random((8,8))]
        mo_occ = [occarray(8, 3), occarray(7, 3), occarray(8, 2)]
        dx = numpy.random.random(15+12+12)
        mo1 = stability._rotate_mo(mo_coeff, mo_occ, dx)
        self.assertAlmostEqual(lib.fp(mo1[0]), 1.1090134286653903, 12)
        self.assertAlmostEqual(lib.fp(mo1[1]), 1.0665953580532537, 12)
        self.assertAlmostEqual(lib.fp(mo1[2]), -5.008202013953201, 12)

if __name__ == "__main__":
    print("Full Tests for stability")
    unittest.main()
