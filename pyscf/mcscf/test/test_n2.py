#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
from pyscf import gto
from pyscf import scf
from pyscf import lib
from pyscf import mcscf

def setUpModule():
    global mol, molsym, m, msym
    b = 1.4
    mol = gto.M(
    verbose = 5,
    output = '/dev/null',
    atom = [
        ['N',(  0.000000,  0.000000, -b/2)],
        ['N',(  0.000000,  0.000000,  b/2)], ],
    basis = {'N': 'ccpvdz', },
    )
    m = scf.RHF(mol)
    m.conv_tol = 1e-9
    m.scf()

    molsym = gto.M(
    verbose = 5,
    output = '/dev/null',
    atom = [
        ['N',(  0.000000,  0.000000, -b/2)],
        ['N',(  0.000000,  0.000000,  b/2)], ],
    basis = {'N': 'ccpvdz', },
    symmetry = True
    )
    msym = scf.RHF(molsym)
    msym.conv_tol = 1e-9
    msym.scf()

def tearDownModule():
    global mol, molsym, m, msym
    mol.stdout.close()
    molsym.stdout.close()
    del mol, molsym, m, msym


class KnownValues(unittest.TestCase):
    def test_mc1step_4o4e(self):
        mc = mcscf.CASSCF(m, 4, 4)
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.7015375913946591, 4)

    def test_mc1step_4o4e_internal_rotation(self):
        mc = mcscf.CASSCF(m, 4, 4)
        mc.internal_rotation = True
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.7015375913946591, 4)

    def test_mc2step_4o4e(self):
        mc = mcscf.CASSCF(m, 4, 4)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.7015375913946591, 4)

    def test_mc1step_6o6e_high_cost(self):
        mc = mcscf.CASSCF(m, 6, 6)
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.980105451388, 7)

    def test_mc2step_6o6e_high_cost(self):
        mc = mcscf.CASSCF(m, 6, 6)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.980105451388, 7)

    def test_mc1step_symm_4o4e(self):
        mc = mcscf.CASSCF(msym, 4, 4)
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.7015375913946591, 4)

    def test_mc2step_symm_4o4e(self):
        mc = mcscf.CASSCF(msym, 4, 4)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.7015375913946591, 4)

    def test_mc1step_symm_6o6e_high_cost(self):
        mc = mcscf.CASSCF(msym, 6, 6)
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.980105451388, 7)

    def test_mc2step_symm_6o6e_high_cost(self):
        mc = mcscf.CASSCF(msym, 6, 6)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.980105451388, 7)

    def test_casci_4o4e(self):
        mc = mcscf.CASCI(m, 4, 4)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -108.8896744464714, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.6910275883606078, 4)

    def test_casci_symm_4o4e(self):
        mc = mcscf.CASCI(msym, 4, 4)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -108.8896744464714, 7)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.6910275883606078, 4)

        mc.wfnsym = 'A2u'
        # raised by mc.fcisolver.guess_wfnsym
        self.assertRaises(RuntimeError, mc.kernel)

    def test_casci_from_uhf(self):
        mf = scf.UHF(mol)
        mf.scf()
        mc = mcscf.CASCI(mf, 4, 4)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -108.8896744464714, 6)
        self.assertAlmostEqual(numpy.linalg.norm(mc.analyze()),
                               2.6910275883606078, 4)

    def test_casci_from_uhf1(self):
        mf = scf.UHF(mol)
        mf.scf()
        mc = mcscf.CASSCF(mf, 4, 4)
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc, -108.913786407955, 7)

    def test_frozen1s(self):
        mc = mcscf.CASSCF(msym, 4, 4)
        mc.frozen = 3
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.91373646206542, 7)

    def test_frozenselect(self):
        mc = mcscf.CASSCF(msym, 4, 4)
        mc.frozen = [i-1 for i in [19, 20, 26, 27]]
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.91238513746941, 7)

    def test_wfnsym(self):
        mc = mcscf.CASSCF(msym, 4, (3,1))
        mc.fcisolver.wfnsym = 14
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.74508322877787, 7)

        mc.wfnsym = 'A2u'
        with self.assertRaises(lib.exceptions.WfnSymmetryError):
            mc.mc1step()
        mc.ci = None
        emc = mc.mc1step()[0]
        self.assertAlmostEqual(emc, -108.69019443475308, 7)

    def test_ucasci(self):
        mc = mcscf.UCASCI(msym, 4, (3,1))
        emc = mc.kernel()[0]
        self.assertAlmostEqual(emc, -108.77486560653847, 7)

    def test_ucasscf_high_cost(self):
        mc = mcscf.UCASSCF(msym, 4, (3,1))
        emc = mc.kernel()[0]
        self.assertAlmostEqual(emc, -108.80789718975041, 7)

    def test_newton_casscf(self):
        mc = mcscf.newton(mcscf.CASSCF(m, 4, 4)).run()
        self.assertAlmostEqual(mc.e_tot, -108.9137864132358, 8)

    def test_newton_casscf_symm(self):
        mc = mcscf.newton(mcscf.CASSCF(msym, 4, 4)).run()
        self.assertAlmostEqual(mc.e_tot, -108.9137864132358, 8)

if __name__ == "__main__":
    print("Full Tests for N2")
    unittest.main()
