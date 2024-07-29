#!/usr/bin/env python
# Copyright 2021 The PySCF Developers. All Rights Reserved.
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

import unittest
from pyscf import lib, gto, scf, tdscf

def setUpModule():
    global mol, mf, nstates
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        ['H' , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ]
    mol.basis = '631g'
    mol.build()
    mf = scf.RHF(mol).run()
    nstates = 5 # make sure first 3 states are converged

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf

class KnownValues(unittest.TestCase):
    def test_tda_singlet(self):
        td = mf.TDA().set(nstates=nstates)
        e = td.kernel()[0]
        ref = [11.90276464, 11.90276464, 16.86036434]
        self.assertAlmostEqual(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 5)

    def test_tda_triplet(self):
        td = mf.TDA().set(nstates=nstates)
        td.singlet = False
        e = td.kernel()[0]
        ref = [11.01747918, 11.01747918, 13.16955056]
        self.assertAlmostEqual(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 5)

    def test_tdhf_singlet(self):
        td = mf.TDHF().set(nstates=nstates)
        e = td.kernel()[0]
        ref = [11.83487199, 11.83487199, 16.66309285]
        self.assertAlmostEqual(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 5)

    def test_tdhf_triplet(self):
        td = mf.TDHF().set(nstates=nstates)
        td.singlet = False
        e = td.kernel()[0]
        ref = [10.8919234, 10.8919234, 12.63440705]
        self.assertAlmostEqual(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 5)


if __name__ == "__main__":
    print("Full Tests for rhf-TDA and rhf-TDHF")
    unittest.main()
