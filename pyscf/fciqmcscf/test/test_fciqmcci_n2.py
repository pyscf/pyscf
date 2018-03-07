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
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import fciqmcscf

b = 1.4
mol = gto.Mole()
mol.build(
verbose = 0,
output = None,
atom = [
    ['N',(  0.000000,  0.000000, -b/2)],
    ['N',(  0.000000,  0.000000,  b/2)], ],
basis = {'N': 'ccpvdz', },
symmetry = True,
symmetry_subgroup = 'D2h'
)
m = scf.RHF(mol)
m.conv_tol = 1e-9
m.scf()


class KnowValues(unittest.TestCase):
    def test_mc2step_4o4e_fci(self):
        mc = mcscf.CASSCF(m, 4, 4)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc,-108.91378640707609, 7)

    def test_mc2step_6o6e_fci(self):
        mc = mcscf.CASSCF(m, 6, 6)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc,-108.98028859357791, 7)

    def test_mc2step_4o4e_fciqmc_wdipmom(self):
        #nelec is the number of electrons in the active space
        nelec = 4
        norb = 4
        mc = mcscf.CASSCF(m, norb, nelec)
        mc.max_cycle_macro = 10
        mc.fcisolver = fciqmcscf.FCIQMCCI(mol)
        mc.fcisolver.RDMSamples = 5000

        emc, e_ci, fcivec, casscf_mo, casscf_mo_e = mc.mc2step()
        self.assertAlmostEqual(emc,-108.91378666934476, 7)

        # Calculate dipole moment in casscf_mo basis
        # First find the 1RDM in the full space
        one_pdm = fciqmc.find_full_casscf_12rdm(mc.fcisolver, casscf_mo,
                'spinfree_TwoRDM.1', norb, nelec)[0]
        dipmom = fciqmc.calc_dipole(mol,casscf_mo,one_pdm)
        print('Dipole moments: ',dipmom)
        #self.assertAlmostEqual(dipmom[0],xxx,5)
        #self.assertAlmostEqual(dipmom[1],xxx,5)
        #self.assertAlmostEqual(dipmom[2],xxx,5)

    def test_mc2step_6o6e_fciqmc(self):
        mc = mcscf.CASSCF(m, 6, 6)
        mc.max_cycle_macro = 10
        mc.fcisolver = fciqmcscf.FCIQMCCI(mol)
        mc.fcisolver.RDMSamples = 5000
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc,-108.98028859357791, 7)

if __name__ == "__main__":
    print("Full Tests for FCIQMC-CASSCF N2")
    unittest.main()

