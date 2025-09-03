#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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
# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.com>

import numpy as np
from pyscf import gto, scf, dft, fci
from pyscf import mcpdft
import unittest

'''
In this unit-test, test the MCPDFT energies calculated for the LiH
molecule at the state-specific and state-average (2-states) using
1. Meta-GGA functional (tM06L)
2. Hybrid-meta-GGA functional tM06L0
3. MC23 Functional

Test the MCPDFT energies calculated for the triplet water molecule at the
4. Meta-GGA functional (M06L)
5. MC23 Functional

Note: The reference values are generated from
OpenMolcas v24.10, tag 682-gf74be507d

The OpenMolcas results were obtained with this grid settings
&SEWARD
Grid Input
RQuad=TA
NR=100
LMAX=41
NOPrun
NOSCreening
'''

# To be consistent with OpenMolcas Grid Settings. Grids_att is defined as below
# Source: pyscf/mcpdft/test/test_diatomic_energies.py

om_ta_alpha = [0.8, 0.9, # H, He
    1.8, 1.4, # Li, Be
    1.3, 1.1, 0.9, 0.9, 0.9, 0.9, # B - Ne
    1.4, 1.3, # Na, Mg
    1.3, 1.2, 1.1, 1.0, 1.0, 1.0, # Al - Ar
    1.5, 1.4, # K, Ca
    1.3, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1, # Sc - Zn
    1.1, 1.0, 0.9, 0.9, 0.9, 0.9] # Ga - Kr

def om_treutler_ahlrichs(n, chg, *args, **kwargs):
    '''
    "Treutler-Ahlrichs" as implemented in OpenMolcas
    '''
    r = np.empty(n)
    dr = np.empty(n)
    alpha = om_ta_alpha[chg-1]
    step = 2.0 / (n+1) # = numpy.pi / (n+1)
    ln2 = alpha / np.log(2)
    for i in range(n):
        x = (i+1)*step - 1 # = numpy.cos((i+1)*step)
        r [i] = -ln2*(1+x)**.6 * np.log((1-x)/2)
        dr[i] = (step #* numpy.sin((i+1)*step)
                * ln2*(1+x)**.6 *(-.6/(1+x)*np.log((1-x)/2)+1/(1-x)))
    return r[::-1], dr[::-1]

my_grids = {'atom_grid': (99,590),
        'radi_method': om_treutler_ahlrichs,
        'prune': False,
        'radii_adjust': None}

def get_lih (r, stateaverage=False, functional='tM06L', basis='sto3g'):
    mol = gto.M (atom='Li 0 0 0\nH {} 0 0'.format (r), basis=basis,
                 output='/dev/null', verbose=0)
    mf = scf.RHF (mol).run ()
    if functional == 'tM06L0':
        tM06L0 = 't' + mcpdft.hyb('M06L',0.25, hyb_type='average')
        mc = mcpdft.CASSCF(mf, tM06L0, 5, 2, grids_attr=my_grids)
    else:
        mc = mcpdft.CASSCF(mf, functional, 5, 2, grids_attr=my_grids)

    if stateaverage:
        mc = mc.state_average_([0.5, 0.5])

    mc.fix_spin_(ss=0)
    mc = mc.run()
    return mc

def get_water_triplet(functional='tM06L', basis='6-31G'):
    mol = gto.M(atom='''
 O     0.    0.000    0.1174
 H     0.    0.757   -0.4696
 H     0.   -0.757   -0.4696
    ''',basis=basis, spin=2,output='/dev/null', verbose=0)

    mf = scf.RHF(mol).run()

    mc = mcpdft.CASSCF(mf, functional, 2, 2, grids_attr=my_grids)
    solver1 = fci.direct_spin1.FCI(mol)
    solver1 = fci.addons.fix_spin(solver1, ss=2)
    mc.fcisolver = solver1
    mc = mc.run()
    return mc

def setUpModule():
    global get_lih, lih_tm06l, lih_tmc23, lih_tm06l_sa2, lih_tmc23_sa2,lih_tm06l0
    global get_water_triplet, water_tm06l, water_tmc23
    global lih_tmc23_2, lih_tmc23_sa2_2, water_tmc23_2
    global lih_tmc25, lih_tmc25_sa2, water_tmc25

    # register otfnal tMC23_2 which is identical to MC23
    mc232_preset = mcpdft.otfnal.OT_PRESET['MC23']
    mcpdft.otfnal.register_otfnal('MC23_2', mc232_preset)

    lih_tm06l = get_lih(1.5, functional='tM06L')
    lih_tmc23 = get_lih(1.5, functional='MC23')
    lih_tmc25 = get_lih(1.5, functional='MC25')
    lih_tmc23_2 = get_lih(1.5, functional='tMC23_2')
    lih_tm06l_sa2 = get_lih(1.5, stateaverage=True, functional='tM06L')
    lih_tmc23_sa2 = get_lih(1.5, stateaverage=True, functional='MC23')
    lih_tmc25_sa2 = get_lih(1.5, stateaverage=True, functional='MC25')
    lih_tmc23_sa2_2 = get_lih(1.5, stateaverage=True, functional='tmc23_2')
    lih_tm06l0 = get_lih(1.5, functional='tM06L0')
    water_tm06l = get_water_triplet()
    water_tmc23 = get_water_triplet(functional='MC23')
    water_tmc25 = get_water_triplet(functional='MC25')
    water_tmc23_2 = get_water_triplet(functional='TMc23_2')

def tearDownModule():
    global lih_tm06l, lih_tmc23, lih_tm06l_sa2, lih_tmc23_sa2
    global lih_tm06l0, water_tm06l, water_tmc23
    global lih_tmc23_2, lih_tmc23_sa2_2, water_tmc23_2

    lih_tm06l.mol.stdout.close()
    lih_tmc23.mol.stdout.close()
    lih_tmc23_2.mol.stdout.close()
    lih_tm06l_sa2.mol.stdout.close()
    lih_tmc23_sa2.mol.stdout.close()
    lih_tmc23_sa2_2.mol.stdout.close()
    lih_tm06l0.mol.stdout.close()
    water_tm06l.mol.stdout.close()
    water_tmc23.mol.stdout.close()
    water_tmc23_2.mol.stdout.close()

    mcpdft.otfnal.unregister_otfnal('tMC23_2')

    del lih_tm06l, lih_tmc23, lih_tm06l_sa2, lih_tmc23_sa2
    del lih_tm06l0, water_tm06l, water_tmc23
    del lih_tmc23_2, lih_tmc23_sa2_2, water_tmc23_2

class KnownValues(unittest.TestCase):

    def assertListAlmostEqual(self, first_list, second_list, expected):
        self.assertTrue(len(first_list) == len(second_list))
        for first, second in zip(first_list, second_list):
            self.assertAlmostEqual(first, second, expected)

    def test_tmgga(self):
        e_mcscf = lih_tm06l.e_mcscf
        epdft = lih_tm06l.e_tot

        sa_e_mcscf = lih_tm06l_sa2.e_mcscf
        sa_epdft = lih_tm06l_sa2.e_states

        # The CAS and MCPDFT reference values are generated using
        # OpenMolcas v24.10, tag 682-gf74be507d
        E_CASSCF_EXPECTED = -7.88214917
        E_MCPDFT_EXPECTED = -7.95814186
        SA_E_CASSCF_EXPECTED = [-7.88205449, -7.74391704]
        SA_E_MCPDFT_EXPECTED = [-7.95807682, -7.79920022]

        self.assertAlmostEqual(e_mcscf, E_CASSCF_EXPECTED, 6)
        self.assertAlmostEqual(epdft, E_MCPDFT_EXPECTED, 6)
        self.assertListAlmostEqual(sa_e_mcscf, SA_E_CASSCF_EXPECTED, 6)
        self.assertListAlmostEqual(sa_epdft, SA_E_MCPDFT_EXPECTED, 6)

    def test_t_hyb_mgga(self):
        e_mcscf = lih_tm06l0.e_mcscf
        epdft = lih_tm06l0.e_tot

        # The CAS and MCPDFT reference values are generated using
        # OpenMolcas v24.10, tag 682-gf74be507d
        E_CASSCF_EXPECTED = -7.88214917
        E_MCPDFT_EXPECTED = -7.93914369

        self.assertAlmostEqual(e_mcscf, E_CASSCF_EXPECTED, 6)
        self.assertAlmostEqual(epdft, E_MCPDFT_EXPECTED, 6)

    def test_tmc23(self):
        e_mcscf = lih_tmc23.e_mcscf
        epdft = lih_tmc23.e_tot

        sa_e_mcscf = lih_tmc23_sa2.e_mcscf
        sa_epdft = lih_tmc23_sa2.e_states

        # The CAS and MCPDFT reference values are generated using
        # OpenMolcas v24.10, tag 682-gf74be507d
        E_CASSCF_EXPECTED = -7.88214917
        E_MCPDFT_EXPECTED = -7.95098727
        SA_E_CASSCF_EXPECTED = [-7.88205449, -7.74391704]
        SA_E_MCPDFT_EXPECTED = [-7.95093826, -7.80604012]

        self.assertAlmostEqual(e_mcscf, E_CASSCF_EXPECTED, 6)
        self.assertAlmostEqual(epdft, E_MCPDFT_EXPECTED, 6)
        self.assertListAlmostEqual(sa_e_mcscf, SA_E_CASSCF_EXPECTED, 6)
        self.assertListAlmostEqual(sa_epdft, SA_E_MCPDFT_EXPECTED, 6)

    def test_tmc25(self):
        e_mcscf = lih_tmc25.e_mcscf
        epdft = lih_tmc25.e_tot

        sa_e_mcscf = lih_tmc25_sa2.e_mcscf
        sa_epdft = lih_tmc25_sa2.e_states

        # The CAS and MCPDFT reference values are generated using
        # OpenMolcas v25.06, tag 7-g86cf2a446
        E_CASSCF_EXPECTED = -7.88214917
        E_MCPDFT_EXPECTED = -7.95714954
        SA_E_CASSCF_EXPECTED = [-7.88205449, -7.74391704]
        SA_E_MCPDFT_EXPECTED = [-7.95708248, -7.81092529]

        self.assertAlmostEqual(e_mcscf, E_CASSCF_EXPECTED, 6)
        self.assertAlmostEqual(epdft, E_MCPDFT_EXPECTED, 6)
        self.assertListAlmostEqual(sa_e_mcscf, SA_E_CASSCF_EXPECTED, 6)
        self.assertListAlmostEqual(sa_epdft, SA_E_MCPDFT_EXPECTED, 6)

    def test_tmc23_2(self):
        e_mcscf = lih_tmc23_2.e_mcscf
        epdft = lih_tmc23_2.e_tot

        sa_e_mcscf = lih_tmc23_sa2_2.e_mcscf
        sa_epdft = lih_tmc23_sa2_2.e_states

        # The CAS and MCPDFT reference values are generated using
        # OpenMolcas v24.10, tag 682-gf74be507d
        E_CASSCF_EXPECTED = -7.88214917
        E_MCPDFT_EXPECTED = -7.95098727
        SA_E_CASSCF_EXPECTED = [-7.88205449, -7.74391704]
        SA_E_MCPDFT_EXPECTED = [-7.95093826, -7.80604012]

        self.assertAlmostEqual(e_mcscf, E_CASSCF_EXPECTED, 6)
        self.assertAlmostEqual(epdft, E_MCPDFT_EXPECTED, 6)
        self.assertListAlmostEqual(sa_e_mcscf, SA_E_CASSCF_EXPECTED, 6)
        self.assertListAlmostEqual(sa_epdft, SA_E_MCPDFT_EXPECTED, 6)

    def test_water_triplet_tm06l(self):
        e_mcscf = water_tm06l.e_mcscf
        epdft = water_tm06l.e_tot

        # The CAS and MCPDFT reference values are generated using
        # OpenMolcas v24.10, tag 682-gf74be507d
        E_CASSCF_EXPECTED = -75.72365496
        E_MCPDFT_EXPECTED = -76.07686505

        self.assertAlmostEqual(e_mcscf, E_CASSCF_EXPECTED, 6)
        self.assertAlmostEqual(epdft, E_MCPDFT_EXPECTED, 6)

    def test_water_triplet_tmc23(self):
        e_mcscf = water_tmc23.e_mcscf
        epdft = water_tmc23.e_tot

        # The CAS and MCPDFT reference values are generated using
        # OpenMolcas v24.10, tag 682-gf74be507d
        E_CASSCF_EXPECTED = -75.72365496
        E_MCPDFT_EXPECTED = -76.02630019

        self.assertAlmostEqual(e_mcscf, E_CASSCF_EXPECTED, 6)
        self.assertAlmostEqual(epdft, E_MCPDFT_EXPECTED, 6)

    def test_water_triplet_tmc25(self):
        e_mcscf = water_tmc25.e_mcscf
        epdft = water_tmc25.e_tot

        # The CAS and MCPDFT reference values are generated using
        # OpenMolcas v25.06, tag 7-g86cf2a446
        E_CASSCF_EXPECTED = -75.72365496
        E_MCPDFT_EXPECTED = -76.08619632

        self.assertAlmostEqual(e_mcscf, E_CASSCF_EXPECTED, 6)
        self.assertAlmostEqual(epdft, E_MCPDFT_EXPECTED, 6)

    def test_water_triplet_tmc23_2(self):
        e_mcscf = water_tmc23_2.e_mcscf
        epdft = water_tmc23_2.e_tot

        # The CAS and MCPDFT reference values are generated using
        # OpenMolcas v24.10, tag 682-gf74be507d
        E_CASSCF_EXPECTED = -75.72365496
        E_MCPDFT_EXPECTED = -76.02630019

        self.assertAlmostEqual(e_mcscf, E_CASSCF_EXPECTED, 6)
        self.assertAlmostEqual(epdft, E_MCPDFT_EXPECTED, 6)

if __name__ == "__main__":
    print("Full Tests for MGGAs, Hybrid-MGGAs, and MC23")
    unittest.main()
