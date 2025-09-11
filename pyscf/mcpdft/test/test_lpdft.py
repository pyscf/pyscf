#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
# Author: Matthew Hennefarth <mhennefarth@uchicago.com>

import tempfile, h5py
import numpy as np
from pyscf import gto, scf, dft, fci, lib
from pyscf import mcpdft
import unittest


def get_lih (r, n_states=2, functional='ftLDA,VWN3', basis='sto3g'):
    mol = gto.M (atom='Li 0 0 0\nH {} 0 0'.format (r), basis=basis,
                 output='/dev/null', verbose=0)
    mf = scf.RHF (mol).run ()
    if n_states == 2:
        mc = mcpdft.CASSCF (mf, functional, 2, 2, grids_level=1)

    else:
        mc = mcpdft.CASSCF(mf, functional, 5, 2, grids_level=1)

    mc.fix_spin_(ss=0)
    weights = [1.0/float(n_states), ] * n_states

    mc = mc.multi_state(weights, "lin")
    mc = mc.run()
    return mc

def get_water(functional='tpbe', basis='6-31g'):
    mol = gto.M(atom='''
 O     0.    0.000    0.1174
 H     0.    0.757   -0.4696
 H     0.   -0.757   -0.4696
    ''',symmetry=True, basis=basis, output='/dev/null', verbose=0)

    mf = scf.RHF(mol).run()

    weights = [0.5, 0.5]
    solver1 = fci.direct_spin1_symm.FCI(mol)
    solver1.wfnsym = 'A1'
    solver1.spin = 0
    solver2 = fci.direct_spin1_symm.FCI(mol)
    solver2.wfnsym = 'A2'
    solver2.spin = 2

    mc = mcpdft.CASSCF(mf, functional, 4, 4, grids_level=1)
    mc.chkfile = tempfile.NamedTemporaryFile().name 
    # mc.chk_ci = True
    mc = mc.multi_state_mix([solver1, solver2], weights, "lin")
    mc.run()
    return mc

def get_water_triplet(functional='tPBE', basis="6-31G"):
    mol = gto.M(atom='''
    O     0.    0.000    0.1174
    H     0.    0.757   -0.4696
    H     0.   -0.757   -0.4696
       ''', symmetry=True, basis=basis, output='/dev/null', verbose=0)

    mf = scf.RHF(mol).run()

    weights = np.ones(3) / 3
    solver1 = fci.direct_spin1_symm.FCI(mol)
    solver1.spin = 2
    solver1 = fci.addons.fix_spin(solver1, shift=.2, ss=2)
    solver1.nroots = 1
    solver2 = fci.direct_spin0_symm.FCI(mol)
    solver2.spin = 0
    solver2.nroots = 2

    mc = mcpdft.CASSCF(mf, functional, 4, 4, grids_level=1)
    mc.chkfile = tempfile.NamedTemporaryFile().name 
    # mc.chk_ci = True
    mc = mc.multi_state_mix([solver1, solver2], weights, "lin")
    mc.run()
    return mc


def setUpModule():
    global lih, lih_4, lih_tpbe, lih_tpbe0, lih_mc23, water, t_water, original_grids
    original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False
    lih = get_lih(1.5)
    lih_4 = get_lih(1.5, n_states=4, basis="6-31G")
    lih_tpbe = get_lih(1.5, functional="tPBE")
    lih_tpbe0 = get_lih(1.5, functional="tPBE0")
    lih_mc23 = get_lih(1.5, functional="MC23")
    water = get_water()
    t_water = get_water_triplet()

def tearDownModule():
    global lih, lih_4, lih_tpbe0, lih_tpbe, t_water, water, original_grids, lih_mc23
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = original_grids
    lih.mol.stdout.close()
    lih_4.mol.stdout.close()
    lih_tpbe0.mol.stdout.close()
    lih_tpbe.mol.stdout.close()
    lih_mc23.mol.stdout.close()
    water.mol.stdout.close()
    t_water.mol.stdout.close()
    del lih, lih_4, lih_tpbe0, lih_tpbe, t_water, water, original_grids, lih_mc23

class KnownValues(unittest.TestCase):

    def assertListAlmostEqual(self, first_list, second_list, expected):
        self.assertTrue(len(first_list) == len(second_list))
        for first, second in zip(first_list, second_list):
            self.assertAlmostEqual(first, second, expected)

    def test_lih_2_states_adiabat(self):
        e_mcscf_avg = np.dot (lih.e_mcscf, lih.weights)
        hcoup = abs(lih.lpdft_ham[1,0])
        hdiag = lih.get_lpdft_diag()

        e_states = lih.e_states

        # Reference values from OpenMolcas v22.02, tag 177-gc48a1862b
        E_MCSCF_AVG_EXPECTED = -7.78902185

        # Below reference values from 
        #   - PySCF commit 71fc2a41e697fec76f7f9a5d4d10fd2f2476302c
        #   - mrh   commit c5fc02f1972c1c8793061f20ed6989e73638fc5e
        HCOUP_EXPECTED = 0.01663680
        HDIAG_EXPECTED = [-7.87848993, -7.72984482]

        E_STATES_EXPECTED = [-7.88032921, -7.72800554]

        self.assertAlmostEqual(e_mcscf_avg, E_MCSCF_AVG_EXPECTED, 7)
        self.assertAlmostEqual(abs(hcoup), HCOUP_EXPECTED, 7)
        self.assertListAlmostEqual(hdiag, HDIAG_EXPECTED, 7)
        self.assertListAlmostEqual(e_states, E_STATES_EXPECTED, 7)

    def test_lih_4_states_adiabat(self):
        e_mcscf_avg = np.dot(lih_4.e_mcscf, lih_4.weights)
        hdiag = lih_4.get_lpdft_diag()
        hcoup = lih_4.lpdft_ham[np.triu_indices(4, k=1)]
        e_states = lih_4.e_states

        # References values from
        #     - PySCF       commit 71fc2a41e697fec76f7f9a5d4d10fd2f2476302c
        #     - PySCF-forge commit 00183c314ebbf541f8461e7b7e5ee9e346fd6ff5
        E_MCSCF_AVG_EXPECTED = -7.88112386
        HDIAG_EXPECTED = [-7.99784259, -7.84720560, -7.80476518, -7.80476521]
        HCOUP_EXPECTED = [0.01479405,0,0,0,0,0]
        E_STATES_EXPECTED = [-7.99928176, -7.84576642, -7.80476519, -7.80476519]

        self.assertAlmostEqual(e_mcscf_avg, E_MCSCF_AVG_EXPECTED, 7)
        self.assertListAlmostEqual(hdiag, HDIAG_EXPECTED, 7)
        self.assertListAlmostEqual(list(map(abs, hcoup)), HCOUP_EXPECTED, 7)
        self.assertListAlmostEqual(e_states, E_STATES_EXPECTED, 7)


    def test_lih_hybrid_tPBE_adiabat(self):
        e_mcscf_tpbe_avg = np.dot(lih_tpbe.e_mcscf, lih_tpbe.weights)
        e_mcscf_tpbe0_avg = np.dot(lih_tpbe0.e_mcscf, lih_tpbe0.weights)

        hlpdft_ham = 0.75 * lih_tpbe.lpdft_ham
        idx = np.diag_indices_from(hlpdft_ham)
        hlpdft_ham[idx] += 0.25 * lih_tpbe.e_mcscf
        e_hlpdft, si_hlpdft = lih_tpbe._eig_si(hlpdft_ham)

        # References values from
        #     - PySCF       commit 8ae2bb2eefcd342c52639097517b1eda7ca5d1cd
        #     - PySCF-forge commit a7b8b3bb291e528088f9cefab007438d9e0f4701
        E_MCSCF_AVG_EXPECTED = -7.78902182
        E_TPBE_STATES_EXPECTED = [-7.93389909, -7.78171959]


        self.assertAlmostEqual(e_mcscf_tpbe_avg, E_MCSCF_AVG_EXPECTED, 7)
        self.assertAlmostEqual(e_mcscf_tpbe_avg, e_mcscf_tpbe0_avg, 9)
        self.assertListAlmostEqual(lih_tpbe.e_states, E_TPBE_STATES_EXPECTED, 7)
        self.assertListAlmostEqual(lih_tpbe0.e_states, e_hlpdft, 9)
        self.assertListAlmostEqual(hlpdft_ham.flatten(), lih_tpbe0.lpdft_ham.flatten(), 9)

    def test_lih_mc23_adiabat(self):
        e_mcscf_mc23_avg = np.dot(lih_mc23.e_mcscf, lih_mc23.weights)
        hcoup = abs(lih_mc23.lpdft_ham[1,0])
        hdiag = lih_mc23.get_lpdft_diag()

        # Reference values from 
        #     - PySCF       commit 9a0bb6ddded7049bdacdaf4cfe422f7ce826c2c7
        #     - PySCF-forge commit eb0ad96f632994d2d1846009ecce047193682526
        E_MCSCF_AVG_EXPECTED = -7.78902182
        E_MC23_EXPECTED = [-7.94539408, -7.80094952]
        HCOUP_EXPECTED = 0.01285147
        HDIAG_EXPECTED = [-7.94424147, -7.80210214]

        self.assertAlmostEqual(e_mcscf_mc23_avg, E_MCSCF_AVG_EXPECTED, 7)
        self.assertAlmostEqual(hcoup, HCOUP_EXPECTED, 7)
        self.assertAlmostEqual(lib.fp(hdiag), lib.fp(HDIAG_EXPECTED), 7)
        self.assertAlmostEqual(lib.fp(lih_mc23.e_states), lib.fp(E_MC23_EXPECTED), 7)

    def test_water_spatial_samix(self):
        e_mcscf_avg = np.dot(water.e_mcscf, water.weights)
        hdiag = water.get_lpdft_diag()
        e_states = water.e_states

        # References values from
        #     - PySCF       commit 8ae2bb2eefcd342c52639097517b1eda7ca5d1cd
        #     - PySCF-forge commit 2c75a59604c458069ebda550e84a866ec1be45dc
        E_MCSCF_AVG_EXPECTED = -75.81489195169507
        HDIAG_EXPECTED = [-76.29913074162732, -75.93502437481517]

        self.assertAlmostEqual(e_mcscf_avg, E_MCSCF_AVG_EXPECTED, 7)
        self.assertListAlmostEqual(hdiag, HDIAG_EXPECTED, 7)
        # The off-diagonal should be identical to zero because of symmetry
        self.assertListAlmostEqual(e_states, hdiag, 10)

    def test_water_spin_samix(self):
        e_mcscf_avg = np.dot(t_water.e_mcscf, t_water.weights)
        hdiag = t_water.get_lpdft_diag()
        e_states = t_water.e_states
        hcoup = abs(t_water.get_lpdft_ham()[1,2])

        # References values from
        #     - PySCF       commit 8ae2bb2eefcd342c52639097517b1eda7ca5d1cd
        #     - PySCF-forge commit 2c75a59604c458069ebda550e84a866ec1be45dc
        E_MCSCF_AVG_EXPECTED = -75.75552048294597
        HDIAG_EXPECTED = [-76.01218048502902, -76.31379141689696, -75.92134410312458]
        E_STATES_EXPECTED = [-76.01218048502898, -76.3168078608912, -75.91832765913041]
        HCOUP_EXPECTED = 0.03453830159471619

        self.assertAlmostEqual(e_mcscf_avg, E_MCSCF_AVG_EXPECTED, 7)
        self.assertListAlmostEqual(e_states, E_STATES_EXPECTED, 6)
        self.assertListAlmostEqual(hdiag, HDIAG_EXPECTED, 6)
        self.assertAlmostEqual(hcoup, HCOUP_EXPECTED, 6)

    def test_chkfile(self):
        for mc, case in zip([water, t_water], ["SA", "SA Mix"]):
            with self.subTest(case=case):
                self.assertTrue(h5py.is_hdf5(mc.chkfile))
                self.assertEqual(lib.fp(mc.mo_coeff), lib.fp(lib.chkfile.load(mc.chkfile, "pdft/mo_coeff")))
                self.assertEqual(mc.e_tot, lib.chkfile.load(mc.chkfile, "pdft/e_tot"))
                self.assertEqual(lib.fp(mc.e_mcscf), lib.fp(lib.chkfile.load(mc.chkfile, "pdft/e_mcscf")))
                self.assertEqual(lib.fp(mc.e_states), lib.fp(lib.chkfile.load(mc.chkfile, "pdft/e_states")))        

                # Requires PySCF version > 2.6.2 which is not available on pip currently
                # for state, (c_ref, c) in enumerate(zip(mc.ci, lib.chkfile.load(mc.chkfile, "pdft/ci"))):
                    # with self.subTest(state=state):
                        # self.assertEqual(lib.fp(c_ref), lib.fp(c))

if __name__ == "__main__":
    print("Full Tests for Linearized-PDFT")
    unittest.main()
