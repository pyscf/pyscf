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
import tempfile
import numpy
from pyscf import gto
from pyscf import scf
from pyscf.scf import diis

class KnownValues(unittest.TestCase):
    def test_addis_minimize(self):
        numpy.random.seed(1)
        ds = numpy.random.random((4,2,2))
        fs = numpy.random.random((4,2,2))
        es = numpy.random.random(4)
        v, x = diis.adiis_minimize(ds, fs, -1)
        self.assertAlmostEqual(v, -0.44797757916272785, 9)

    def test_eddis_minimize(self):
        numpy.random.seed(1)
        ds = numpy.random.random((4,2,2))
        fs = numpy.random.random((4,2,2))
        es = numpy.random.random(4)
        v, x = diis.ediis_minimize(es, ds, fs)
        self.assertAlmostEqual(v, 0.31551563100606295, 9)

    def test_input_diis(self):
        mol = gto.M(
            verbose = 7,
            output = '/dev/null',
            atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
            basis = '631g',
        )
        mf1 = scf.RHF(mol)
        mf1.DIIS = diis.EDIIS
        mf1.max_cycle = 4
        e = mf1.kernel()
        self.assertAlmostEqual(e, -75.983875341696987, 9)
        mol.stdout.close()

    def test_roll_back(self):
        mol = gto.M(
            verbose = 7,
            output = '/dev/null',
            atom = '''
        O     0    0        0
        H     0    -1.757   1.587
        H     0    1.757    1.587''',
            basis = '631g',
        )
        mf1 = scf.RHF(mol)
        mf1.diis_space = 4
        mf1.diis_space_rollback = True
        mf1.max_cycle = 10
        e = mf1.kernel()
        self.assertAlmostEqual(e, -75.446749864901321, 9)
        mol.stdout.close()

    def test_diis_restart(self):
        mol = gto.M(
            verbose = 7,
            output = '/dev/null',
            atom = '''
        O     0    0        0
        H     0    -1.757   1.587
        H     0    1.757    1.587''',
            basis = '631g',
        )
        tmpf = tempfile.NamedTemporaryFile()
        mf = scf.RHF(mol)
        mf.diis_file = tmpf.name
        eref = mf.kernel()
        self.assertAlmostEqual(eref, -75.44606939063496, 9)

        mf = scf.RHF(mol)
        mf.diis = scf.diis.DIIS().restore(tmpf.name)
        mf.max_cycle = 3
        e = mf.kernel()
        self.assertAlmostEqual(e, eref, 9)


if __name__ == "__main__":
    print("Full Tests for DIIS")
    unittest.main()

