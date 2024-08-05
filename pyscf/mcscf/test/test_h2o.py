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
import tempfile
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import fci
from pyscf import lib

def setUpModule():
    global mol, molsym, m, msym, mc_ref
    mol = gto.M(
    verbose = 5,
    output = '/dev/null',
    atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. ,-0.757  , 0.587)],
        [1   , (0. , 0.757  , 0.587)]],
    basis = '631g',
    )
    m = scf.RHF(mol)
    m.conv_tol = 1e-10
    m.scf()

    molsym = gto.M(
    verbose = 5,
    output = '/dev/null',
    atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. ,-0.757  , 0.587)],
        [1   , (0. , 0.757  , 0.587)]],
    basis = '631g',
    symmetry = True
    )
    msym = scf.RHF(molsym)
    msym.conv_tol = 1e-10
    msym.scf()

    mc_ref = mcscf.CASSCF (m, 4, 4).state_average_([0.25,]*4)
    # SA-CASSCF may be stuck at a local minimum e_tot = -75.75381945 with the
    # default initial guess from HF orbitals. The initial guess below is closed
    # to the single state CASSCF orbitals which can lead to a lower SA-CASSCF
    # energy e_tot = -75.762754627
    mo = mc_ref.sort_mo([4,5,6,10], base=1)
    mc_ref.kernel (mo)

def tearDownModule():
    global mol, molsym, m, msym, mc_ref
    mol.stdout.close()
    molsym.stdout.close()
    del mol, molsym, m, msym, mc_ref

# 4 states, in order: 1^A1, 3^B2, 1^B2, 3^A1
# 3 distinct ways of using state_average_mix to specify these states
class KnownValues(unittest.TestCase):
    def test_nosymm_sa4_newton (self):
        mc = mcscf.CASSCF (m, 4, 4).state_average_([0.25,]*4).newton ()
        mo = mc.sort_mo([4,5,6,10], base=1)
        mc.kernel(mo)
        self.assertAlmostEqual (mc.e_tot, mc_ref.e_tot, 8)
        for e1, e0 in zip (mc.e_states, mc_ref.e_states):
            self.assertAlmostEqual (e1, e0, 5)

    def test_spin_sa4 (self):
        fcisolvers = [fci.solver (mol, singlet=not(bool(i)), symm=False) for i in range (2)]
        fcisolvers[0].nroots = fcisolvers[1].nroots = 2
        fcisolvers[1].spin = 2
        mc = mcscf.addons.state_average_mix (mcscf.CASSCF (m, 4, 4), fcisolvers, [0.25,]*4)
        mo = mc.sort_mo([4,5,6,10], base=1)
        mc.kernel(mo)
        self.assertAlmostEqual (mc.e_tot, mc_ref.e_tot, 8)
        for e1, e0 in zip (numpy.sort (mc.e_states), mc_ref.e_states):
            self.assertAlmostEqual (e1, e0, 5)

    def test_spin_sa4_newton (self):
        fcisolvers = [fci.solver (mol, singlet=not(bool(i)), symm=False) for i in range (2)]
        fcisolvers[0].nroots = fcisolvers[1].nroots = 2
        fcisolvers[1].spin = 2
        mc = mcscf.addons.state_average_mix (mcscf.CASSCF (m, 4, 4), fcisolvers, [0.25,]*4).newton ()
        mo = mc.sort_mo([4,5,6,10], base=1)
        mc.kernel(mo)
        self.assertAlmostEqual (mc.e_tot, mc_ref.e_tot, 8)
        for e1, e0 in zip (numpy.sort (mc.e_states), mc_ref.e_states):
            self.assertAlmostEqual (e1, e0, 5)

    def test_pointgroup_sa4 (self):
        fcisolvers = [fci.solver (molsym, symm=True, singlet=False) for i in range (2)]
        fcisolvers[0].nroots = fcisolvers[1].nroots = 2
        fcisolvers[0].wfnsym = 'A1'
        fcisolvers[1].wfnsym = 'B1'
        mc = mcscf.addons.state_average_mix (mcscf.CASSCF (msym, 4, 4), fcisolvers, [0.25,]*4)
        mo = mc.sort_mo([4,5,6,10], base=1)
        mc.kernel(mo)
        self.assertAlmostEqual (mc.e_tot, mc_ref.e_tot, 8)
        for e1, e0 in zip (numpy.sort (mc.e_states), mc_ref.e_states):
            self.assertAlmostEqual (e1, e0, 5)

    def test_pointgroup_sa4_newton (self):
        fcisolvers = [fci.solver (molsym, symm=True, singlet=False) for i in range (2)]
        fcisolvers[0].nroots = fcisolvers[1].nroots = 2
        fcisolvers[0].wfnsym = 'A1'
        fcisolvers[1].wfnsym = 'B1'
        mc = mcscf.addons.state_average_mix (mcscf.CASSCF (msym, 4, 4), fcisolvers, [0.25,]*4).newton ()
        mo = mc.sort_mo([4,5,6,10], base=1)
        mc.kernel(mo)
        self.assertAlmostEqual (mc.e_tot, mc_ref.e_tot, 8)
        for e1, e0 in zip (numpy.sort (mc.e_states), mc_ref.e_states):
            self.assertAlmostEqual (e1, e0, 5)

    def test_spin_and_pointgroup_sa4 (self):
        fcisolvers = [fci.solver (molsym, singlet = not(bool(i%2))) for i in range (4)]
        fcisolvers[0].wfnsym = fcisolvers[1].wfnsym = 'B1'
        fcisolvers[2].wfnsym = fcisolvers[3].wfnsym = 'A1'
        fcisolvers[1].spin = fcisolvers[3].spin = 2
        mc = mcscf.addons.state_average_mix (mcscf.CASSCF (msym, 4, 4), fcisolvers, [0.25,]*4)
        mo = mc.sort_mo([4,5,6,10], base=1)
        mc.kernel(mo)
        self.assertAlmostEqual (mc.e_tot, mc_ref.e_tot, 8)
        for e1, e0 in zip (numpy.sort (mc.e_states), mc_ref.e_states):
            self.assertAlmostEqual (e1, e0, 5)

    def test_spin_and_pointgroup_sa4_newton (self):
        fcisolvers = [fci.solver (molsym, singlet = not(bool(i%2))) for i in range (4)]
        fcisolvers[0].wfnsym = fcisolvers[1].wfnsym = 'B1'
        fcisolvers[2].wfnsym = fcisolvers[3].wfnsym = 'A1'
        fcisolvers[1].spin = fcisolvers[3].spin = 2
        mc = mcscf.addons.state_average_mix (mcscf.CASSCF (msym, 4, 4), fcisolvers, [0.25,]*4).newton ()
        mo = mc.sort_mo([4,5,6,10], base=1)
        mc.kernel(mo)
        self.assertAlmostEqual (mc.e_tot, mc_ref.e_tot, 8)
        for e1, e0 in zip (numpy.sort (mc.e_states), mc_ref.e_states):
            self.assertAlmostEqual (e1, e0, 5)

    def test_casci(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            ['O', ( 0., 0.    , 0.   )],
            ['H', ( 0., -0.757, 0.587)],
            ['H', ( 0., 0.757 , 0.587)],]

        mol.basis = {'H': 'sto-3g',
                     'O': '6-31g',}
        mol.build()

        m = scf.RHF(mol).run()
        mc = mcscf.CASCI(m, 4, 4)
        mc.fcisolver = fci.solver(mol)
        mc.natorb = 1
        emc = mc.kernel()[0]
        self.assertAlmostEqual(emc, -75.9624554777, 7)

        mc = mcscf.CASCI(m, 4, (3,1))
        mc.fcisolver = fci.solver(mol, False)
        emc = mc.casci()[0]
        self.assertAlmostEqual(emc, -75.439016172976, 6)

    def test_addons(self):
        mc = mcscf.CASSCF(msym, 4, 4)
        mc.fcisolver = fci.solver(molsym, False) # to mix the singlet and triplet
        mc = mc.state_average_((.64,.36))
        emc, e_ci, fcivec, mo, mo_energy = mc.mc1step()[:5]
        self.assertAlmostEqual(emc, -75.85387884606675, 8)
        mc = mcscf.CASCI(msym, 4, 4)
        emc = mc.casci(mo)[0]
        self.assertAlmostEqual(emc, -75.98341123168858, 8)

        mc = mcscf.CASSCF(msym, 4, 4)
        mc = mc.state_specific_(2)
        emc = mc.kernel()[0]
        self.assertAlmostEqual(emc, -75.59353002290788, 8)

    def test_chkfile_mixed(self):
        fcisolvers = [
            fci.solver(mol, singlet=not (bool(i)), symm=False) for i in range(2)
        ]
        fcisolvers[0].nroots = fcisolvers[1].nroots = 2
        fcisolvers[1].spin = 2
        mc = mcscf.addons.state_average_mix(
            mcscf.CASSCF(m, 4, 4),
            fcisolvers,
            [
                0.25,
            ]
            * 4,
        )
        mo = mc.sort_mo([4, 5, 6, 10], base=1)
        mc.chkfile = tempfile.NamedTemporaryFile().name
        mc.chk_ci = True
        mc.kernel(mo)
        self.assertAlmostEqual(mc.e_tot, mc_ref.e_tot, 8)
        for e1, e0 in zip(numpy.sort(mc.e_states), mc_ref.e_states):
            self.assertAlmostEqual(e1, e0, 5)

        for state, (cref, c) in enumerate(
            zip(mc.ci, lib.chkfile.load(mc.chkfile, "mcscf/ci"))
        ):
            with self.subTest(state=state):
                self.assertEqual(lib.fp(cref), lib.fp(c))

        self.assertEqual(
            lib.fp(mc.mo_coeff), lib.fp(lib.chkfile.load(mc.chkfile, "mcscf/mo_coeff"))
        )


if __name__ == "__main__":
    print("Full Tests for H2O")
    unittest.main()
