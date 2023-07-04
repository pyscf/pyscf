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
import scipy.linalg
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import scf
from pyscf.pbc import dft

def setUpModule():
    global cell, kmf, mycc, eris
    L = 2
    cell = gto.Cell()
    cell.unit = 'B'
    cell.a = np.diag([L,L,L])
    cell.mesh = np.array([11]*3)
    cell.atom = [['He', (L/2.,L/2.,L/2.)]]
    cell.basis = { 'He': [[0, (1.0, 1.0)]] }
    cell.precision = 1e-9
    cell.build()


def tearDownModule():
    global cell
    del cell

class KnownValues(unittest.TestCase):
    def test_band(self):
        mf = scf.RHF(cell).run()
        kpts = cell.make_kpts([5,1,1])
        bands = mf.get_bands(kpts)[0]
        bands_ref = []
        for kpt in kpts:
            fock = mf.get_hcore(kpt=kpt) + mf.get_veff(kpts_band=kpt)
            ovlp = mf.get_ovlp(kpt=kpt)
            bands_ref.append(mf.eig(fock, ovlp)[0])
        self.assertAlmostEqual(abs(np.array(bands_ref) - np.array(bands)).max(), 0, 9)
        self.assertAlmostEqual(lib.fp(bands), -0.69079067047363329, 8)

    def test_band_kscf(self):
        kpts = cell.make_kpts([2,1,1])
        kmf = dft.KRKS(cell, kpts=kpts).run()
        np.random.seed(11)
        kpts_band = np.random.random((4,3))
        bands = kmf.get_bands(kpts_band)[0]
        bands_ref = []
        h1 = kmf.get_hcore(kpts=kpts_band)
        s1 = kmf.get_ovlp(kpts=kpts_band)
        vhf = kmf.get_veff(kpts_band=kpts_band)
        for i, kpt in enumerate(kpts_band):
            fock = h1[i] + vhf[i]
            bands_ref.append(scipy.linalg.eigh(fock, s1[i])[0])
        self.assertAlmostEqual(abs(np.array(bands_ref) - np.array(bands)).max(), 0, 9)
        self.assertAlmostEqual(lib.fp(bands), -0.61562245312227049, 8)

# TODO: test get_bands for hf/uhf with/without DF

if __name__ == '__main__':
    print("Full Tests for kpt-bands")
    unittest.main()
