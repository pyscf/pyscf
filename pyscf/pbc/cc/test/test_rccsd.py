#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
import weakref
from unittest import mock

import numpy as np

from pyscf.cc import ccsd, rccsd
from pyscf.pbc import cc, gto, scf


class RCCSDChecks:
    def check_gamma_update(self):
        mycc = cc.RCCSD(self.mf)
        mycc.max_cycle = 1
        real_update = ccsd.update_amps

        def check_update(solver, t1, t2, adapted):
            self.assertIsNot(adapted, self.eris)
            self.assertEqual(adapted.ovvv.ndim, 3)
            self.assertIs(adapted.fock, self.eris.fock)
            self.assertIs(adapted.mo_energy, self.eris.mo_energy)
            self.assertIs(adapted.vvvv, self.eris.vvvv)
            np.testing.assert_allclose(adapted.get_ovvv(), self.eris.ovvv, atol=1e-12, rtol=0)
            expected = rccsd.update_amps(solver, t1, t2, self.eris)
            actual = real_update(solver, t1, t2, adapted)
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
            return actual

        with mock.patch.object(ccsd, 'update_amps', side_effect=check_update) as fast:
            mycc.kernel(eris=self.eris)
        self.assertEqual(fast.call_count, 1)
        self.assertEqual(self.eris.ovvv.ndim, 4)

    def check_gamma_energy_and_eris_reuse(self):
        reference = cc.RCCSD(self.mf)
        ref_energy, ref1, ref2 = rccsd.RCCSD.ccsd(reference, eris=self.eris)
        self.assertTrue(reference.converged)
        original_ovvv = self.eris.ovvv
        before = {name: getattr(self.eris, name).copy() for name in
                  ('oooo', 'ovoo', 'ovov', 'oovv', 'ovvo', 'ovvv', 'vvvv',
                   'fock', 'mo_energy')}
        views = []

        def remember_view(env):
            views.append(weakref.ref(env['eris']))

        mycc = cc.RCCSD(self.mf)
        mycc.callback = remember_view
        energy, t1, t2 = mycc.kernel(eris=self.eris)
        self.assertTrue(mycc.converged)
        self.assertAlmostEqual(energy, ref_energy, 11)
        np.testing.assert_allclose(t1, ref1, atol=1e-11, rtol=0)
        np.testing.assert_allclose(t2, ref2, atol=1e-11, rtol=0)
        self.assertIs(self.eris.ovvv, original_ovvv)
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(self.eris, name), value)
        self.assertTrue(views)
        self.assertTrue(all(view() is None for view in views))
        self.assertAlmostEqual(mycc.ccsd_t(eris=self.eris), reference.ccsd_t(eris=self.eris), 11)
        mycc.solve_lambda(eris=self.eris)
        self.assertTrue(mycc.converged_lambda)

    def check_shifted_update(self):
        mycc = cc.RCCSD(self.shifted_mf)
        mycc.max_cycle = 1
        self.assertTrue(np.iscomplexobj(self.shifted_eris.ovvv))
        with mock.patch.object(ccsd, 'update_amps', wraps=ccsd.update_amps) as fast:
            with mock.patch.object(rccsd, 'update_amps', wraps=rccsd.update_amps) as original:
                mycc.kernel(eris=self.shifted_eris)
        self.assertEqual(fast.call_count, 0)
        self.assertEqual(original.call_count, 1)
        self.assertIs(original.call_args.args[3], self.shifted_eris)

    def check_direct_update(self):
        mycc = cc.RCCSD(self.mf)
        mycc.direct = True
        mycc.max_cycle = 1
        with mock.patch.object(ccsd, 'update_amps',
                               side_effect=AssertionError('PBC direct uses RCCSD')) as fast:
            with mock.patch.object(rccsd, 'update_amps', wraps=rccsd.update_amps) as original:
                mycc.kernel(eris=self.eris)
        self.assertEqual(fast.call_count, 0)
        self.assertEqual(original.call_count, 1)
        self.assertIs(original.call_args.args[3], self.eris)

    def check_gamma_builds_eris(self):
        mycc = cc.RCCSD(self.mf)
        mycc.max_cycle = 1
        with mock.patch.object(mycc, 'ao2mo', return_value=self.eris) as ao2mo:
            with mock.patch.object(ccsd, 'update_amps', wraps=ccsd.update_amps) as fast:
                mycc.kernel()
        ao2mo.assert_called_once_with(mycc.mo_coeff)
        self.assertEqual(fast.call_count, 1)
        self.assertEqual(self.eris.ovvv.ndim, 4)

    def check_custom_update(self):
        mycc = cc.RCCSD(self.mf)
        mycc.max_cycle = 1

        def custom_update(t1, t2, supplied_eris):
            self.assertIs(supplied_eris, self.eris)
            return rccsd.update_amps(mycc, t1, t2, supplied_eris)

        mycc.update_amps = mock.Mock(side_effect=custom_update)
        with mock.patch.object(ccsd, 'update_amps', wraps=ccsd.update_amps) as fast:
            mycc.kernel(eris=self.eris)
        self.assertEqual(mycc.update_amps.call_count, 1)
        self.assertEqual(fast.call_count, 0)

    def check_complex_orbitals_at_gamma(self):
        phases = np.exp(1j * np.linspace(0, .3, self.mf.mo_coeff.shape[1]))
        mycc = cc.RCCSD(self.mf, mo_coeff=self.mf.mo_coeff * phases)
        mycc.max_cycle = 1
        complex_eris = mycc.ao2mo()
        with mock.patch.object(ccsd, 'update_amps', wraps=ccsd.update_amps) as fast:
            with mock.patch.object(rccsd, 'update_amps', wraps=rccsd.update_amps) as original:
                mycc.kernel(eris=complex_eris)
        self.assertEqual(fast.call_count, 0)
        self.assertEqual(original.call_count, 1)
        self.assertIs(original.call_args.args[3], complex_eris)


class KnownValues(RCCSDChecks, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = gto.M(atom='He 0 0 0; He 0 0 1.5', a=np.eye(3)*4,
                         basis=[[0, [.8, 1]], [1, [.5, 1]]],
                         mesh=[15]*3, precision=1e-7, verbose=0)
        cls.mf = scf.RHF(cls.cell).density_fit(auxbasis='weigend').run(conv_tol=1e-11)
        shifted = scf.RHF(cls.cell, kpt=cls.cell.get_abs_kpts([.1]*3))
        cls.shifted_mf = shifted.density_fit(auxbasis='weigend').run(conv_tol=1e-11)
        cls.eris = cc.RCCSD(cls.mf).ao2mo()
        cls.shifted_eris = cc.RCCSD(cls.shifted_mf).ao2mo()

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.mf, cls.eris, cls.shifted_mf, cls.shifted_eris

    test_gamma_update = RCCSDChecks.check_gamma_update
    test_gamma_energy_and_eris_reuse = RCCSDChecks.check_gamma_energy_and_eris_reuse
    test_shifted_update = RCCSDChecks.check_shifted_update
    test_direct_update = RCCSDChecks.check_direct_update
    test_gamma_builds_eris = RCCSDChecks.check_gamma_builds_eris
    test_custom_update = RCCSDChecks.check_custom_update
    test_complex_orbitals_at_gamma = RCCSDChecks.check_complex_orbitals_at_gamma


class HighCostKnownValues(RCCSDChecks, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.M(
            atom='C 0 0 0; C .89175 .89175 .89175',
            a=[[0, 1.7835, 1.7835], [1.7835, 0, 1.7835], [1.7835, 1.7835, 0]],
            basis='cc-pvdz', verbose=0,
        )
        mf = scf.RHF(cell).density_fit().run(conv_tol=1e-11)
        eris = cc.RCCSD(mf).ao2mo()
        kpt = cell.get_abs_kpts([.1, .1, .1])
        shifted_mf = scf.RHF(cell, kpt=kpt).density_fit().run(conv_tol=1e-11)
        shifted_eris = cc.RCCSD(shifted_mf).ao2mo()
        assert mf.converged and shifted_mf.converged
        cls.cell, cls.mf, cls.eris = cell, mf, eris
        cls.shifted_mf, cls.shifted_eris = shifted_mf, shifted_eris

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.mf, cls.eris, cls.shifted_mf, cls.shifted_eris

    test_gamma_update_high_cost = RCCSDChecks.check_gamma_update
    test_gamma_energy_and_eris_reuse_high_cost = RCCSDChecks.check_gamma_energy_and_eris_reuse
    test_shifted_update_high_cost = RCCSDChecks.check_shifted_update
    test_direct_update_high_cost = RCCSDChecks.check_direct_update
    test_gamma_builds_eris_high_cost = RCCSDChecks.check_gamma_builds_eris
    test_custom_update_high_cost = RCCSDChecks.check_custom_update
    test_complex_orbitals_at_gamma_high_cost = RCCSDChecks.check_complex_orbitals_at_gamma


if __name__ == '__main__':
    unittest.main()
