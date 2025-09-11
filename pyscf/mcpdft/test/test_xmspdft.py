#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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
# Author: Matthew Hennefarth <mhennefarth@uchicago.edu>


import numpy as np
from pyscf import gto, scf, dft, mcpdft, lib
from pyscf.mcpdft import xmspdft

import unittest


def get_lih(r, weights=None, fnal="ftLDA,VWN3"):
    global mols
    mol = gto.M(
        atom="Li 0 0 0\nH {} 0 0".format(r),
        basis="sto3g",
        output="/dev/null",
        verbose=0,
    )
    mols.append(mol)
    mf = scf.RHF(mol).run()
    mc = mcpdft.CASSCF(mf, fnal, 2, 2, grids_level=1)
    mc.fix_spin_(ss=0)
    if weights is None:
        weights = [0.5, 0.5]
    mc = mc.multi_state(weights, "xms").run(conv_tol=1e-8)
    return mc


def setUpModule():
    global mols, original_grids
    mols = []
    original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False


def tearDownModule():
    global mols, original_grids
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = original_grids
    [m.stdout.close() for m in mols]
    del mols, original_grids


class KnownValues(unittest.TestCase):

    def test_lih_xms2ftlda(self):
        mc = get_lih(1.5)
        e_mcscf_avg = np.dot(mc.e_mcscf, mc.weights)
        hcoup = abs(mc.heff_mcscf[1, 0])
        ct_mcscf = abs(mc.si_mcscf[0, 0])

        # Reference values from
        # - PySCF        hash 8ae2bb2eefcd342c52639097517b1eda7ca5d1cd
        # - PySCF-forge  hash 5b8ab86a31917ca1a6b414f7a590c4046b9a8994
        #
        # Implementation with those hashes verified with OpenMolcas
        # (tag 462-g00b34a15f)

        HCOUP_EXPECTED = 0.01996004651860848
        CT_MCSCF_EXPECTED = 0.9886771524332543
        E_MCSCF_AVG_EXPECTED = -7.789021830554006
        E_STATES_EXPECTED = [-7.858628517291297, -7.69980510010583]

        with self.subTest("diabats"):
            self.assertAlmostEqual(e_mcscf_avg, E_MCSCF_AVG_EXPECTED, 9)
            self.assertAlmostEqual(hcoup, HCOUP_EXPECTED, 9)
            self.assertAlmostEqual(ct_mcscf, CT_MCSCF_EXPECTED, 9)

        with self.subTest("adiabats"):
            self.assertAlmostEqual(lib.fp(mc.e_states), lib.fp(E_STATES_EXPECTED), 9)

        with self.subTest("safock"):
            safock = xmspdft.make_fock_mcscf(mc, ci=mc.get_ci_adiabats(uci="MCSCF"))
            EXPECTED_SA_FOCK_DIAG = [-4.207598506457942, -3.88169762424571]
            EXPECTED_SA_FOCK_OFFDIAG = 0.05063053788053997

            self.assertAlmostEqual(
                lib.fp(safock.diagonal()), lib.fp(EXPECTED_SA_FOCK_DIAG), 9
            )
            self.assertAlmostEqual(abs(safock[0, 1]), EXPECTED_SA_FOCK_OFFDIAG, 9)

        with self.subTest("safock unequal"):
            safock = xmspdft.make_fock_mcscf(
                mc, ci=mc.get_ci_adiabats(uci="MCSCF"), weights=[1, 0]
            )
            EXPECTED_SA_FOCK_DIAG = [-4.194714957289011, -3.8317682977263754]
            EXPECTED_SA_FOCK_OFFDIAG = 0.006987847963283834

            self.assertAlmostEqual(
                lib.fp(safock.diagonal()), lib.fp(EXPECTED_SA_FOCK_DIAG), 9
            )
            self.assertAlmostEqual(abs(safock[0, 1]), EXPECTED_SA_FOCK_OFFDIAG, 9)

    def test_lih_xms2ftlda_unequal_weights(self):
        mc = get_lih(1.5, weights=[0.9, 0.1])
        e_mcscf_avg = np.dot(mc.e_mcscf, mc.weights)
        hcoup = abs(mc.heff_mcscf[1, 0])
        ct_mcscf = abs(mc.si_mcscf[0, 0])

        HCOUP_EXPECTED = 0.006844540922301437
        CT_MCSCF_EXPECTED = 0.9990626861718451
        E_MCSCF_AVG_EXPECTED = -7.8479729055935685
        E_STATES_EXPECTED = [-7.869843475893866, -7.696820527732333]

        self.assertAlmostEqual(e_mcscf_avg, E_MCSCF_AVG_EXPECTED, 9)
        self.assertAlmostEqual(hcoup, HCOUP_EXPECTED, 9)
        self.assertAlmostEqual(ct_mcscf, CT_MCSCF_EXPECTED, 9)
        self.assertAlmostEqual(lib.fp(mc.e_states), lib.fp(E_STATES_EXPECTED), 9)

    def test_lih_xms2mc23(self):
        mc = get_lih(1.5)
        e_mcscf_avg = np.dot(mc.e_mcscf, mc.weights)
        hcoup = abs(mc.heff_mcscf[0,1])
        ct_mcscf = abs(mc.si_mcscf[0,0])
        ct_pdft = abs(mc.si_pdft[0,0])

        # Reference values from
        # - PySCF        hash 9a0bb6ddded7049bdacdaf4cfe422f7ce826c2c7
        # - PySCF-forge  hash 40bfc1eb8b8a662c1c57bff5f7ffa7316e5d043d

        E_MCSCF_AVG_EXPECTED = -7.7890218306
        E_STATES_EXPECTED = [-7.85862852, -7.6998051]
        HCOUP_EXPECTED = 0.01996005
        CT_MCSCF_EXPECTED = 0.98867715
        CT_PDFT_EXPECTED = 0.9919416682619435

        self.assertAlmostEqual(e_mcscf_avg, E_MCSCF_AVG_EXPECTED, 8)
        self.assertAlmostEqual(lib.fp(mc.e_states), lib.fp(E_STATES_EXPECTED), 8)
        self.assertAlmostEqual(hcoup, HCOUP_EXPECTED, 8)
        self.assertAlmostEqual(ct_mcscf, CT_MCSCF_EXPECTED, 8)
        self.assertAlmostEqual(ct_pdft, CT_PDFT_EXPECTED, 8)


if __name__ == "__main__":
    print("Full Tests for XMS-PDFT")
    unittest.main()
