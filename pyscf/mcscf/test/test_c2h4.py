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
from pyscf import scf
from pyscf import gto
from pyscf import mcscf

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        ["C", (-0.65830719,  0.61123287, -0.00800148)],
        ["C", ( 0.73685281,  0.61123287, -0.00800148)],
        ["H", ( 1.43439081,  1.81898387, -0.00800148)],
        ["H", (-1.35568919,  1.81920887, -0.00868348)],
        ["H", (-1.20806619, -0.34108413, -0.00755148)],
        ["H", ( 1.28636081, -0.34128013, -0.00668648)],]

    mol.basis = {'H': 'cc-pvdz',
                 'C': 'cc-pvdz',}
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.scf()

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


class KnownValues(unittest.TestCase):
    def test_casci_4o4e(self):
        mc = mcscf.CASCI(mf, 4, 4)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -77.9734951776, 7)

    def test_casci_6o4e(self):
        mc = mcscf.CASCI(mf, 6, 4)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -77.9746683275, 7)

    def test_casci_6o6e(self):
        mc = mcscf.CASCI(mf, 6, 6)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -77.9804561351, 7)

    def test_mc2step_6o6e_high_cost(self):
        mc = mcscf.CASSCF(mf, 6, 6)
        mc.conv_tol = 1e-8
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -78.0390051207, 7)

    def test_mc1step_6o6e_high_cost(self):
        mc = mcscf.CASSCF(mf, 6, 6)
        mc.conv_tol = 1e-8
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -78.0390051207, 7)

    def test_mc2step_4o4e_high_cost(self):
        mc = mcscf.CASSCF(mf, 4, 4)
        mc.conv_tol = 1e-8
        emc = mc.mc2step()[0]
        #?self.assertAlmostEqual(emc, -78.0103838390, 6)
        self.assertAlmostEqual(emc, -77.9916207871, 6)

    def test_mc1step_4o4e_high_cost(self):
        mc = mcscf.CASSCF(mf, 4, 4)
        mcscf.mc1step.WITH_MICRO_SCHEDULER, bak = True, mcscf.mc1step.WITH_MICRO_SCHEDULER
        mc.conv_tol = 1e-8
        emc = mc.mc1step()[0]
        mcscf.mc1step.WITH_MICRO_SCHEDULER = bak
        self.assertAlmostEqual(emc, -78.0103838390, 6)


if __name__ == "__main__":
    print("Full Tests for C2H4")
    unittest.main()
