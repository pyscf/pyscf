#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Author: Linus Bjarne Dittmer <email:linus.dittmer@stud.uni-heidelberg.de>
#

import unittest
import numpy
import scipy.linalg
import tempfile
from pyscf import gto
from pyscf import scf
from pyscf import dft

class KnownValues(unittest.TestCase):

    def setUp(self):
        self.mol1 = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = "Li 0.0 0.0 0.0; H 1.0 0.0 0.0",
            spin = 0,
            basis = '6-31g')

        self.mol3 = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = "Li 0.0 0.0 0.0; H 1.0 0.0 0.0",
            basis = '6-31g',
            spin = 2,)


    def tearDown(self):
        self.mol1.stdout.close()
        self.mol3.stdout.close()
        del self.mol1, self.mol3

    def test_nr_rhf(self):
        mf = scf.RHF(self.mol1)
        m3 = scf.M3SOSCF(mf, 2)
        d = m3.converge()
        self.assertAlmostEqual(d[1], -7.871355587185268, 9)

    def test_nr_rohf(self):
        mf = scf.RHF(self.mol3)
        m3 = scf.M3SOSCF(mf, 2)
        d = m3.converge()
        self.assertAlmostEqual(d[1], -7.775069668379746, 9)


    def test_nr_uhf(self):
        mf = scf.UHF(self.mol3)
        m3 = scf.M3SOSCF(mf, 2)
        d = m3.converge()
        self.assertAlmostEqual(d[1], -7.775142720974782, 9)

    def test_nr_rks_lda(self):
        mf = dft.RKS(self.mol1)
        eref = mf.kernel()
        m3 = scf.M3SOSCF(mf, 2)
        d = m3.converge()
        self.assertAlmostEqual(d[1], eref, 9)

    def test_nr_rks_rsh(self):
        '''test range-separated Coulomb'''
        mf = dft.RKS(self.mol1)
        mf.xc = 'wb97x'
        eref = mf.kernel()
        m3 = scf.M3SOSCF(mf, 2)
        d = m3.converge()
        self.assertAlmostEqual(d[1], eref, 9)

    def test_nr_rks(self):
        mf = dft.RKS(self.mol1)
        mf.xc = 'b3lyp'
        eref = mf.kernel()
        m3 = scf.M3SOSCF(mf, 2)
        d = m3.converge()
        self.assertAlmostEqual(d[1], eref, 9)

    def test_nr_roks(self):
        mf = dft.RKS(self.mol1)
        mf.xc = 'b3lyp'
        eref = mf.kernel()
        m3 = scf.M3SOSCF(mf, 2)
        d = m3.converge()
        self.assertAlmostEqual(d[1], eref, 9)


    def test_nr_uks_lda(self):
        mf = dft.UKS(self.mol3)
        eref = mf.kernel()
        m3 = scf.M3SOSCF(mf, 2)
        d = m3.converge()
        self.assertAlmostEqual(d[1], eref, 9)

    def test_nr_uks_rsh(self):
        '''test range-separated Coulomb'''
        mf = dft.UKS(self.mol3)
        mf.xc = 'wb97x'
        eref = mf.kernel()
        m3 = scf.M3SOSCF(mf, 2)
        d = m3.converge()
        self.assertAlmostEqual(d[1], eref, 9)

    def test_nr_uks(self):
        mf = dft.UKS(self.mol3)
        mf.xc = 'b3lyp'
        eref = mf.kernel()
        m3 = scf.M3SOSCF(mf, 2)
        d = m3.converge()
        self.assertAlmostEqual(d[1], eref, 9)

if __name__ == "__main__":
    print("Relevant tests for M3SOSCF")
    unittest.main()
