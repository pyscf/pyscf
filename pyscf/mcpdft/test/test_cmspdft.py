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

import numpy as np
from pyscf import gto, mcpdft, lib
import unittest

degree = np.pi / 180.0
THETA0 = 30


def u_theta(theta=THETA0):
    # The sign here is consistent with my desired variable convention:
    # lower-triangular positive w/ row idx = initial state and col idx =
    # final state
    ct = np.cos(theta * degree)
    st = np.sin(theta * degree)
    return np.array([[ct, st], [-st, ct]])


def numerical_Q(mc):
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    mo_cas = mc.mo_coeff[:, ncore:][:, :ncas]

    def num_Q(theta):
        ci_theta = mc.get_ci_basis(uci=u_theta(theta))
        states_casdm1 = mc.fcisolver.states_make_rdm1(ci_theta, ncas, nelecas)
        Q = 0
        for casdm1 in states_casdm1:
            dm_cas = np.dot(mo_cas, casdm1)
            dm_cas = np.dot(dm_cas, mo_cas.conj().T)
            vj = mc._scf.get_j(dm=dm_cas)
            Q += np.dot(vj.ravel(), dm_cas.ravel()) / 2
        return Q

    return num_Q


def get_lih(r, fnal="ftLDA,VWN3"):
    global mols
    mol = gto.M(
        atom="Li 0 0 0\nH {} 0 0".format(r),
        basis="sto3g",
        output="/dev/null",
        verbose=0,
    )
    mols.append(mol)
    mf = mol.RHF().run()
    mc = mcpdft.CASSCF(mf, fnal, 2, 2, grids_level=1)
    mc.fix_spin_(ss=0)
    mc = mc.multi_state([0.5, 0.5], "cms").run(conv_tol=1e-8)
    return mc


def setUpModule():
    global mols
    mols = []


def tearDownModule():
    global mols
    [m.stdout.close() for m in mols]
    del mols


class KnownValues(unittest.TestCase):

    def test_lih_cms2ftlda22(self):
        # Reference values from OpenMolcas v22.02, tag 177-gc48a1862b
        # Ignoring the PDFT energies and final states because of grid nonsense
        mc = get_lih(1.5)
        e_mcscf_avg = np.dot(mc.e_mcscf, mc.weights)
        hcoup = abs(mc.heff_mcscf[1, 0])
        ct_mcscf = abs(mc.si_mcscf[0, 0])
        q_max = mc.diabatizer()[0]

        E_MCSCF_AVG_EXPECTED = -7.78902185
        Q_MAX_EXPECTED = 1.76394711
        HCOUP_EXPECTED = 0.0350876533212476
        CT_MCSCF_EXPECTED = 0.96259815333407572

        with self.subTest("diabats"):
            self.assertAlmostEqual(e_mcscf_avg, E_MCSCF_AVG_EXPECTED, 7)
            self.assertAlmostEqual(q_max, Q_MAX_EXPECTED, 5)
            self.assertAlmostEqual(hcoup, HCOUP_EXPECTED, 5)
            self.assertAlmostEqual(ct_mcscf, CT_MCSCF_EXPECTED, 5)

        with self.subTest("e coul"):
            ci_theta0 = mc.get_ci_basis(uci=u_theta(THETA0))
            q_test, dQ_test, d2Q_test = mc.diabatizer(ci=ci_theta0)[:3]
            num_Q = numerical_Q(mc)
            delta = 0.01
            qm = num_Q(THETA0 - delta)
            q0 = num_Q(THETA0)
            qp = num_Q(THETA0 + delta)
            dQ_ref = (qp - qm) / 2 / delta / degree
            d2Q_ref = (qp + qm - 2 * q0) / degree / degree / delta / delta
            with self.subTest(deriv=0):
                self.assertLess(q_test, q_max)
                self.assertAlmostEqual(q_test, q0, 9)
            with self.subTest(deriv=1):
                self.assertAlmostEqual(dQ_test[0], dQ_ref, 6)
            with self.subTest(deriv=2):
                self.assertAlmostEqual(d2Q_test[0, 0], d2Q_ref, 6)

        with self.subTest("e coul old"):
            ci_theta0 = mc.get_ci_basis(uci=u_theta(THETA0))
            q_test, dQ_test, d2Q_test = mc.diabatizer(ci=ci_theta0)[:3]
            from pyscf.mcpdft.cmspdft import e_coul_o0

            q_ref, dQ_ref, d2Q_ref = e_coul_o0(mc, ci_theta0)
            with self.subTest(deriv=0):
                self.assertAlmostEqual(q_test, q_ref, 9)
            with self.subTest(deriv=1):
                self.assertAlmostEqual(dQ_test[0], dQ_ref[0], 9)
            with self.subTest(deriv=2):
                self.assertAlmostEqual(d2Q_test[0, 0], d2Q_ref[0, 0], 9)

        with self.subTest("e coul update"):
            theta_rand = 360 * np.random.rand() - 180
            u_rand = u_theta(theta_rand)
            ci_rand = mc.get_ci_basis(uci=u_rand)
            q, _, _, q_update = mc.diabatizer()
            q_ref, dQ_ref, d2Q_ref = mc.diabatizer(ci=ci_rand)[:3]
            q_test, dQ_test, d2Q_test = q_update(u_rand)
            with self.subTest(deriv=0):
                self.assertLessEqual(q_test, q_max)
                self.assertAlmostEqual(q_test, q_ref, 9)
            with self.subTest(deriv=1):
                self.assertAlmostEqual(dQ_test[0], dQ_ref[0], 9)
            with self.subTest(deriv=2):
                self.assertAlmostEqual(d2Q_test[0, 0], d2Q_ref[0, 0], 9)

    def test_scanner(self):
        mc1 = get_lih(1.5)
        mc2 = get_lih(1.55).as_scanner()

        mc2(mc1.mol)
        self.assertTrue(mc1.converged)
        self.assertTrue(mc2.converged)
        self.assertAlmostEqual(lib.fp(mc1.e_states), lib.fp(mc2.e_states), 6)

    def test_lih_cms2mc2322(self):
        mc = get_lih(1.5, fnal="MC23")
        e_mcscf_avg = np.dot(mc.e_mcscf, mc.weights)
        hcoup = abs(mc.heff_mcscf[0, 1])
        ct_mcscf = abs(mc.si_mcscf[0, 0])
        ct_pdft = abs(mc.si_pdft[0, 0])

        HCOUP_EXPECTED = 0.03508667
        E_MCSCF_AVG_EXPECTED = -7.789021830554006
        E_STATES_EXPECTED = [-7.93513351, -7.77927879]
        CT_MCSCF_EXPECTED = 0.9626004825617019
        CT_PDFT = 0.9728574801328089

        self.assertAlmostEqual(e_mcscf_avg, E_MCSCF_AVG_EXPECTED, 8)
        self.assertAlmostEqual(lib.fp(mc.e_states), lib.fp(E_STATES_EXPECTED), 8)
        self.assertAlmostEqual(hcoup, HCOUP_EXPECTED, 8)
        self.assertAlmostEqual(ct_mcscf, CT_MCSCF_EXPECTED, 8)
        self.assertAlmostEqual(ct_pdft, CT_PDFT, 8)


if __name__ == "__main__":
    print("Full Tests for CMS-PDFT function")
    unittest.main()
