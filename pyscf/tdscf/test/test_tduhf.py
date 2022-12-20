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
    global mol, mol1, mf, mf1
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ['H' , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ]
    mol.basis = '631g'
    mol.build()
    mf = scf.UHF(mol).run(conv_tol=1e-10)

    mol1 = gto.Mole()
    mol1.verbose = 7
    mol1.output = '/dev/null'
    mol1.atom = [
        ['H' , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ]
    mol1.basis = '631g'
    mol1.spin = 2
    mol1.build()
    mf1 = scf.UHF(mol1).run(conv_tol=1e-10)

def tearDownModule():
    global mol, mol1, mf, mf1
    mol1.stdout.close()
    del mol, mol1, mf, mf1

class KnownValues(unittest.TestCase):
    def test_tda(self):
        td = mf.TDA()
        td.nstates = 5
        e = td.kernel()[0]
        ref = [11.01748568, 11.01748568, 11.90277134, 11.90277134, 13.16955369]
        self.assertAlmostEqual(abs(e * 27.2114 - ref).max(), 0, 4)

    def test_tdhf(self):
        td = mf.TDHF()
        td.nstates = 5
        td.singlet = False
        e = td.kernel()[0]
        ref = [10.89192986, 10.89192986, 11.83487865, 11.83487865, 12.6344099]
        self.assertAlmostEqual(abs(e * 27.2114 - ref).max(), 0, 4)

    def test_tda_triplet(self):
        td = mf1.TDA()
        td.nstates = 5
        e = td.kernel()[0]
        ref = [3.32113736, 18.55977052, 21.01474222, 21.61501962, 25.0938973]
        self.assertAlmostEqual(abs(e * 27.2114 - ref).max(), 0, 4)

    def test_tdhf_triplet(self):
        td = mf1.TDHF()
        td.nstates = 4
        e = td.kernel()[0]
        ref = [3.31267103, 18.4954748, 20.84935404, 21.54808392]
        self.assertAlmostEqual(abs(e * 27.2114 - ref).max(), 0, 4)


if __name__ == "__main__":
    print("Full Tests for uhf-TDA and uhf-TDHF")
    unittest.main()
