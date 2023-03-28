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

def setUpModule():
    global h2o_z0, h2o_z1, h2o_z0_s, h2o_z1_s, h4_z0_s, h4_z1_s
    h2o_z0 = gto.M(
        verbose = 5,
        output = '/dev/null',
        atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ],
        basis = '6-31g')

    h2o_z1 = gto.M(
        verbose = 5,
        output = '/dev/null',
        atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ],
        basis = '6-31g',
        charge = 1,
        spin = 1,)

def tearDownModule():
    global h2o_z0, h2o_z1
    h2o_z0.stdout.close()
    h2o_z1.stdout.close()
    del h2o_z0, h2o_z1

class KnownValues(unittest.TestCase):
    def test_nr_rhf(self):
        mf = scf.RHF(h2o_z0)
        m3 = scf.M3SOSCF(mf, 2)
        d = m3.converge()
        self.assertAlmostEqual(d[1], -75.98394849812, 9)

    def test_nr_rohf(self):
        mf = scf.RHF(h2o_z1)
        m3 = scf.M3SOSCF(mf, 2)
        d = m3.converge()
        self.assertAlmostEqual(d[1], -75.5783963795897, 9)


    def test_nr_uhf(self):
        mf = scf.UHF(h2o_z1)
        nr = scf.M3SOSCF(mf, 2)
        d = m3.converge()
        self.assertAlmostEqual(d[1], -75.58051984397145, 9)

    def test_nr_rks_lda(self):
        mf = dft.RKS(h2o_z0)
        eref = mf.kernel()
        nr = scf.M3SOSCF(mf, 2)
        d = m3.converge()
        self.assertAlmostEqual(d[1], eref, 9)

    def test_nr_rks_rsh(self):
        '''test range-separated Coulomb'''
        mf = dft.RKS(h2o_z0)
        mf.xc = 'wb97x'
        eref = mf.kernel()
        m3 = scf.M3SOSCF(mf, 2)
        self.assertAlmostEqual(d[1], eref, 9)

    def test_nr_rks(self):
        mf = dft.RKS(h2o_z0)
        mf.xc = 'b3lyp'
        eref = mf.kernel()
        m3 = scf.M3SOSCF(mf, 2)
        d = m3.converge()
        self.assertAlmostEqual(d[1], eref, 9)

    def test_nr_roks(self):
        mf = dft.RKS(h2o_z1)
        mf.xc = 'b3lyp'
        eref = mf.kernel()
        m3 = scf.M3SOSCF(mf, 2)
        d = m3.converge()
        self.assertAlmostEqual(d[1], eref, 9)


    def test_nr_uks_lda(self):
        mf = dft.UKS(h2o_z1)
        eref = mf.kernel()
        m3 = scf.M3SOSCF(mf, 2)
        d = m3.converge()
        self.assertAlmostEqual(d[1], eref, 9)

    def test_nr_uks_rsh(self):
        '''test range-separated Coulomb'''
        mf = dft.UKS(h2o_z1)
        mf.xc = 'wb97x'
        eref = mf.kernel()
        m3 = scf.M3SOSCF(mf, 2)
        d = m3.converge()
        self.assertAlmostEqual(d[1], eref, 9)

    def test_nr_uks(self):
        mf = dft.UKS(h2o_z1)
        mf.xc = 'b3lyp'
        eref = mf.kernel()
        m3 = scf.M3SOSCF(mf, 2)
        d = m3.converge()
        self.assertAlmostEqual(d[1], eref, 9)

if __name__ == "__main__":
    print("Relevant tests for M3SOSCF")
    unittest.main()
