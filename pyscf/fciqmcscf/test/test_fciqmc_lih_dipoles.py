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
from pyscf.fciqmcscf.fciqmc import *

b = 1.4
mol = gto.Mole()

mol.build(
verbose = 5,
output = 'tests.out',
#output = None,
atom = [['Li',(  0.000000,  0.000000, 1.005436697)],
        ['H',(  0.000000,  0.000000,  0.0)]],
basis = {'H': 'sto-3g', 'Li': 'sto-3g'},
symmetry = True,
symmetry_subgroup = 'C2v',
)

m = scf.RHF(mol)
m.conv_tol = 1e-9
m.scf()

class KnowValues(unittest.TestCase):
    def test_dipoles_hfbasis(self):
        fciqmcci = FCIQMCCI(mol)
        fciqmcci.tau = 0.01
        fciqmcci.RDMSamples = 2000

        norb = m.mo_coeff.shape[1]
        energy = run_standalone(fciqmcci, m, m.mo_coeff)
        two_pdm = read_neci_two_pdm(fciqmcci, 'spinfree_TwoRDM.1', norb)
        one_pdm = one_from_two_pdm(two_pdm, mol.nelectron)
        dips, elec, nuc = calc_dipole(mol, m.mo_coeff, one_pdm)

        # This requires use of the dneci executable, running on 2 cores to be correct
        self.assertAlmostEqual(energy, -7.7871453481816, 5)
        self.assertAlmostEqual(dips[0], 0.0, 7)
        self.assertAlmostEqual(dips[1], 0.0, 7)
        self.assertAlmostEqual(dips[2], 1.8578755475683275, 4)

    def test_dipoles_casscfbasis(self):

        # There are only 6 orbitals and 4 electrons, so this is almost the full
        # space, giving close-to-exact NO basis.
        mc = mcscf.CASSCF(m,4,4)
        # Ensures that casscf_mo returns the natural orbital basis in the
        # active space.
        mc.natorb = True
        emc, e_ci, fcivec, casscf_mo, mo_energy = mc.mc2step(m.mo_coeff)

        fciqmcci = FCIQMCCI(mol)
        fciqmcci.tau = 0.01
        fciqmcci.RDMSamples = 2000
        norb = mc.mo_coeff.shape[1]
        # Run from CASSCF natural orbitals
        energy = run_standalone(fciqmcci, m, casscf_mo)
        two_pdm = read_neci_two_pdm(fciqmcci, 'spinfree_TwoRDM.1', norb)
        one_pdm = one_from_two_pdm(two_pdm, mol.nelectron)
        dips, elec, nuc = calc_dipole(mol, mc.mo_coeff, one_pdm)

        self.assertAlmostEqual(energy, -7.787146064428100, 5)
        self.assertAlmostEqual(dips[0], 0.0, 7)
        self.assertAlmostEqual(dips[1], 0.0, 7)
        self.assertAlmostEqual(dips[2], 1.85781390006, 4)

if __name__ == "__main__":
    print('Tests for dipole moments from standalone FCIQMC calculation in HF '
          'and natural orbital basis sets.')
    unittest.main()
