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
from pyscf.pbc import gto
from pyscf.pbc import scf
from pyscf.pbc import dft

L = 2
cell = gto.Cell()
cell.unit = 'B'
cell.a = np.diag([L,L,L])
cell.mesh = np.array([11]*3)
cell.atom = [['He', (L/2.,L/2.,L/2.)]]
cell.basis = { 'He': [[0, (1.0, 1.0)]] }
cell.build()

def finger(a):
    a = np.asarray(a)
    return np.dot(a.ravel(), np.cos(np.arange(a.size)))

class KnowValues(unittest.TestCase):
    def test_band(self):
        mf = scf.RHF(cell).run()
        kpts = cell.make_kpts([10,1,1])
        bands = []
        for kpt in kpts:
            fock = mf.get_hcore(kpt=kpt) + mf.get_veff(kpts_band=kpt)
            ovlp = mf.get_ovlp(kpt=kpt)
            bands.append(mf.eig(fock, ovlp)[0])
        self.assertAlmostEqual(finger(bands), 6.7327210318311597, 8)

    def test_band_kscf(self):
        kpts = cell.make_kpts([2,1,1])
        kmf = dft.KRKS(cell, kpts=kpts).run()
        bands = []
        h1 = kmf.get_hcore()
        s1 = kmf.get_ovlp()
        vhf = kmf.get_veff(kpts_band=kpts)
        for i, kpt in enumerate(kpts):
            fock = h1[i] + vhf[i]
            bands.append(scipy.linalg.eigh(fock, s1[i])[0])
        self.assertAlmostEqual(finger(bands), -0.76745129086774599, 8)


if __name__ == '__main__':
    print("Full Tests for kpt-bands")
    unittest.main()

