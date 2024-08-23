#!/usr/bin/env python
# Copyright 2023 The PySCF Developers. All Rights Reserved.
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
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf.cc import qcisd

def setUpModule():
    global mol, mf, mycc
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = '631g'
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    mycc = qcisd.QCISD(mf)
    mycc.kernel()

def tearDownModule():
    global mol, mf, mycc
    mol.stdout.close()
    del mol, mf, mycc


class KnownValues(unittest.TestCase):
    def test_qcisd_t(self):
        mol = gto.Mole()
        mol.atom = """C  0.000  0.000  0.000
                      H  0.637  0.637  0.637
                      H -0.637 -0.637  0.637
                      H -0.637  0.637 -0.637
                      H  0.637 -0.637 -0.637"""
        mol.basis = 'cc-pvdz'
        mol.verbose = 7
        mol.output = '/dev/null'
        mol.spin = 0
        mol.build()
        mf = scf.RHF(mol).run()

        mycc = qcisd.QCISD(mf, frozen=1)
        ecc, t1, t2 = mycc.kernel()
        self.assertAlmostEqual(mycc.e_tot, -40.3839884, 6)
        et = mycc.qcisd_t()
        self.assertAlmostEqual(mycc.e_tot+et, -40.38767969, 6)

    def test_qcisd_t_frozen(self):
        mol = gto.Mole()
        mol.atom = [['Ne', (0,0,0)]]
        mol.basis = 'cc-pvdz'
        mol.verbose = 7
        mol.output = '/dev/null'
        mol.spin = 0
        mol.build()
        mf = scf.RHF(mol).run()

        mycc = qcisd.QCISD(mf, frozen=1).as_scanner()
        mycc(mol)
        et = mycc.qcisd_t()
        self.assertAlmostEqual(mycc.e_tot+et, -128.6788843055109, 6)

if __name__ == "__main__":
    print("Full Tests for CCD")
    unittest.main()
