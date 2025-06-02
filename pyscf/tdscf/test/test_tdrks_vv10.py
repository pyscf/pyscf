# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
import pyscf
from pyscf.dft import rks, uks

def setUpModule():
    global mol, unrestricted_mol, excitation_energy_threshold, dipole_threshold, oscillator_strength_threshold

    atom = '''
    O  0.0000  0.7375 -0.0528
    O  0.0000 -0.7375 -0.1528
    H  0.8190  0.8170  0.4220
    H -0.8190 -0.8170  0.4220
    '''
    basis = 'def2-svp'

    mol = pyscf.M(atom=atom, basis=basis, max_memory=32000,
                  output='/dev/null', verbose=1)

    unrestricted_mol = pyscf.M(atom=atom, charge=1, spin=1, basis=basis, max_memory=32000,
                               output='/dev/null', verbose=1)

    excitation_energy_threshold = 1e-6
    dipole_threshold = 2e-4
    oscillator_strength_threshold = 1e-6

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def make_mf(mol, restricted = True):
    if restricted:
        mf = rks.RKS(mol, xc = "wb97x-v")
    else:
        mf = uks.UKS(mol, xc = "wb97x-v")
    mf.grids.atom_grid = (99,590)
    mf.nlcgrids.atom_grid = (50,194)

    mf.conv_tol = 1e-13
    mf.direct_scf_tol = 1e-16
    # if density_fitting:
    #     mf = mf.density_fit(auxbasis = "def2-universal-jkfit")
    mf.kernel()
    assert mf.converged
    return mf

class KnownValues(unittest.TestCase):
    def test_wb97xv_tddft_high_cost(self):
        ### Q-Chem input
        # $rem
        # JOBTYPE       sp
        # METHOD        wb97x-v
        # BASIS         def2-svp
        # THRESH 16
        # SCF_CONVERGENCE 13
        # RPA TRUE
        # CIS_N_ROOTS          5
        # CIS_SINGLETS         TRUE
        # CIS_TRIPLETS         FALSE
        # XC_GRID 000099000590
        # NL_GRID 000050000194
        # SYMMETRY FALSE
        # SYM_IGNORE TRUE
        # MEM_STATIC 2000
        # MEM_TOTAL  20000
        # $end
        reference_ground_state_energy = -151.3641561221
        reference_excited_state_energy = np.array([-151.14843260, -151.10016934, -151.07876401, -151.04365404, -151.01453591])
        reference_excitation_energy = reference_excited_state_energy - reference_ground_state_energy

        mf = make_mf(mol)
        tddft = mf.TDDFT()
        tddft.exclude_nlc = False
        test_excitation_energy, test_state_vector = tddft.kernel(nstates = len(reference_excited_state_energy))

        assert np.linalg.norm(test_excitation_energy - reference_excitation_energy) < excitation_energy_threshold

        reference_transition_dipole = np.array([
            [-0.0027, -0.0099,  0.0163],
            [-0.0266,  0.1685,  0.0146],
            [-0.0040, -0.1075,  0.0390],
            [-0.0385, -0.0231, -0.1681],
            [ 0.0684,  0.0576, -0.2139],
        ])
        test_transition_dipole = tddft.transition_dipole()

        for i_dipole in range(reference_transition_dipole.shape[0]):
            assert np.linalg.norm(test_transition_dipole[i_dipole] - reference_transition_dipole[i_dipole]) < dipole_threshold \
                or np.linalg.norm(test_transition_dipole[i_dipole] + reference_transition_dipole[i_dipole]) < dipole_threshold

        reference_oscillator_strength = np.array([0.0000531401, 0.0051569656, 0.0024927814, 0.0064662878, 0.0125286115])
        test_oscillator_strength = tddft.oscillator_strength()

        assert np.linalg.norm(test_oscillator_strength - reference_oscillator_strength) < oscillator_strength_threshold

    def test_wb97xv_tda(self):
        # Same Q-Chem input as above, Q-Chem computes both TDA and TDDFT in the same run
        reference_ground_state_energy = -151.3641561221
        reference_excited_state_energy = np.array([-151.14537857, -151.09702586, -151.07806251, -151.04306837, -151.01364584])
        reference_excitation_energy = reference_excited_state_energy - reference_ground_state_energy

        mf = make_mf(mol)
        tda = mf.TDA()
        tda.exclude_nlc = False
        test_excitation_energy, test_state_vector = tda.kernel(nstates = len(reference_excited_state_energy))

        assert np.linalg.norm(test_excitation_energy - reference_excitation_energy) < excitation_energy_threshold

        reference_transition_dipole = np.array([
            [-0.0039, -0.0088, -0.0068],
            [-0.0100,  0.1746,  0.0147],
            [-0.0125, -0.1214,  0.0384],
            [-0.0405, -0.0270, -0.1656],
            [ 0.0706,  0.0609, -0.2241],
        ])
        test_transition_dipole = tda.transition_dipole()

        for i_dipole in range(reference_transition_dipole.shape[0]):
            assert np.linalg.norm(test_transition_dipole[i_dipole] - reference_transition_dipole[i_dipole]) < dipole_threshold \
                or np.linalg.norm(test_transition_dipole[i_dipole] + reference_transition_dipole[i_dipole]) < dipole_threshold

        reference_oscillator_strength = np.array([0.0000204074, 0.0054841178, 0.0031204297, 0.0063755735, 0.0137712931])
        test_oscillator_strength = tda.oscillator_strength()

        assert np.linalg.norm(test_oscillator_strength - reference_oscillator_strength) < oscillator_strength_threshold

    def test_wb97xv_tddft_triplet_high_cost(self):
        ### Q-Chem input
        # $rem
        # JOBTYPE       sp
        # METHOD        wb97x-v
        # BASIS         def2-svp
        # THRESH 16
        # SCF_CONVERGENCE 13
        # RPA TRUE
        # CIS_N_ROOTS          5
        # CIS_SINGLETS         FALSE
        # CIS_TRIPLETS         TRUE
        # XC_GRID 000099000590
        # NL_GRID 000050000194
        # SYMMETRY FALSE
        # SYM_IGNORE TRUE
        # MEM_STATIC 2000
        # MEM_TOTAL  20000
        # $end
        reference_ground_state_energy = -151.3641561221
        reference_excited_state_energy = np.array([-151.19587195, -151.15395771, -151.09548852, -151.07813338, -151.06169230])
        reference_excitation_energy = reference_excited_state_energy - reference_ground_state_energy
        mf = make_mf(mol)
        tddft = mf.TDDFT()
        tddft.singlet = False
        tddft.exclude_nlc = False
        test_excitation_energy, test_state_vector = tddft.kernel(nstates = len(reference_excited_state_energy))

        assert np.linalg.norm(test_excitation_energy - reference_excitation_energy) < excitation_energy_threshold

    def test_wb97xv_tda_triplet(self):
        # Same Q-Chem input as above, Q-Chem computes both TDA and TDDFT in the same run
        reference_ground_state_energy = -151.3641561221
        reference_excited_state_energy = np.array([-151.19274710, -151.14933133, -151.09446103, -151.06656613, -151.06072560])
        reference_excitation_energy = reference_excited_state_energy - reference_ground_state_energy

        mf = make_mf(mol)
        tda = mf.TDA()
        tda.singlet = False
        tda.exclude_nlc = False
        test_excitation_energy, test_state_vector = tda.kernel(nstates = len(reference_excited_state_energy))

        assert np.linalg.norm(test_excitation_energy - reference_excitation_energy) < excitation_energy_threshold

    def test_wb97xv_unrestricted_tddft_high_cost(self):
        ### Q-Chem input
        # $rem
        # JOBTYPE       sp
        # METHOD        wb97x-v
        # BASIS         def2-svp
        # THRESH 16
        # SCF_CONVERGENCE 13
        # RPA TRUE
        # CIS_N_ROOTS          5
        # CIS_SINGLETS         TRUE
        # CIS_TRIPLETS         FALSE
        # UNRESTRICTED TRUE
        # XC_GRID 000099000590
        # NL_GRID 000050000194
        # SYMMETRY FALSE
        # SYM_IGNORE TRUE
        # MEM_STATIC 2000
        # MEM_TOTAL  20000
        # $end
        reference_ground_state_energy = -150.9397884760
        reference_excited_state_energy = np.array([-150.90300494, -150.80988169, -150.76053699, -150.72460109, -150.71759201])
        reference_excitation_energy = reference_excited_state_energy - reference_ground_state_energy

        mf = make_mf(unrestricted_mol, restricted = False)
        tddft = mf.TDDFT()
        tddft.exclude_nlc = False
        test_excitation_energy, test_state_vector = tddft.kernel(nstates = len(reference_excited_state_energy))

        assert np.linalg.norm(test_excitation_energy - reference_excitation_energy) < excitation_energy_threshold

        reference_transition_dipole = np.array([
            [ 0.0380,  0.3958,  0.0204],
            [-0.0922, -0.3828, -0.0152],
            [ 0.0247,  0.0668, -0.0011],
            [ 0.0330,  0.0555,  0.0893],
            [-0.0258, -0.0626,  0.0036],
        ])
        test_transition_dipole = tddft.transition_dipole()

        for i_dipole in range(reference_transition_dipole.shape[0]):
            assert np.linalg.norm(test_transition_dipole[i_dipole] - reference_transition_dipole[i_dipole]) < dipole_threshold \
                or np.linalg.norm(test_transition_dipole[i_dipole] + reference_transition_dipole[i_dipole]) < dipole_threshold

        reference_oscillator_strength = np.array([0.0038865748, 0.0134450605, 0.0006057376, 0.0017409850, 0.0006821462])
        test_oscillator_strength = tddft.oscillator_strength()

        assert np.linalg.norm(test_oscillator_strength - reference_oscillator_strength) < oscillator_strength_threshold

    def test_wb97xv_unrestricted_tda(self):
        # Same Q-Chem input as above, Q-Chem computes both TDA and TDDFT in the same run
        reference_ground_state_energy = -150.9397884760
        reference_excited_state_energy = np.array([-150.88981193, -150.79604327, -150.75118183, -150.72292823, -150.71461300])
        reference_excitation_energy = reference_excited_state_energy - reference_ground_state_energy

        mf = make_mf(unrestricted_mol, restricted = False)
        tda = mf.TDA()
        tda.exclude_nlc = False
        test_excitation_energy, test_state_vector = tda.kernel(nstates = len(reference_excited_state_energy))

        assert np.linalg.norm(test_excitation_energy - reference_excitation_energy) < excitation_energy_threshold

        reference_transition_dipole = np.array([
            [ 0.0165,  0.3855,  0.0232],
            [-0.0912, -0.4214, -0.0170],
            [ 0.0289,  0.0663, -0.0011],
            [ 0.0284,  0.0425,  0.0964],
            [-0.0226, -0.0487, -0.0455],
        ])
        test_transition_dipole = tda.transition_dipole()

        for i_dipole in range(reference_transition_dipole.shape[0]):
            assert np.linalg.norm(test_transition_dipole[i_dipole] - reference_transition_dipole[i_dipole]) < dipole_threshold \
                or np.linalg.norm(test_transition_dipole[i_dipole] + reference_transition_dipole[i_dipole]) < dipole_threshold

        reference_oscillator_strength = np.array([0.0049780202, 0.0178406618, 0.0006579850, 0.0017216364, 0.0007431055])
        test_oscillator_strength = tda.oscillator_strength()

        assert np.linalg.norm(test_oscillator_strength - reference_oscillator_strength) < oscillator_strength_threshold


if __name__ == "__main__":
    print("Tests for TD-RKS and TD-UKS with vv10")
    unittest.main()
