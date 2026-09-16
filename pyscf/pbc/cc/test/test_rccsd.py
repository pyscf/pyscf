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


def setUpModule():
    global cell, mf, eris, shifted_mf, shifted_eris
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


def tearDownModule():
    global cell, mf, eris, shifted_mf, shifted_eris
    del cell, mf, eris, shifted_mf, shifted_eris


class KnownValues(unittest.TestCase):
    def test_gamma_update(self):
        mycc = cc.RCCSD(mf)
        mycc.max_cycle = 1
        real_update = ccsd.update_amps

        def check_update(solver, t1, t2, adapted):
            self.assertIsNot(adapted, eris)
            self.assertEqual(adapted.ovvv.ndim, 3)
            self.assertIs(adapted.fock, eris.fock)
            self.assertIs(adapted.mo_energy, eris.mo_energy)
            self.assertIs(adapted.vvvv, eris.vvvv)
            np.testing.assert_allclose(adapted.get_ovvv(), eris.ovvv, atol=1e-12, rtol=0)
            expected = rccsd.update_amps(solver, t1, t2, eris)
            actual = real_update(solver, t1, t2, adapted)
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
            return actual

        with mock.patch.object(ccsd, 'update_amps', side_effect=check_update) as fast:
            mycc.kernel(eris=eris)
        self.assertEqual(fast.call_count, 1)
        self.assertEqual(eris.ovvv.ndim, 4)

    def test_gamma_energy_and_eris_reuse(self):
        reference = cc.RCCSD(mf)
        ref_energy, ref1, ref2 = rccsd.RCCSD.ccsd(reference, eris=eris)
        self.assertTrue(reference.converged)
        original_ovvv = eris.ovvv
        before = {name: getattr(eris, name).copy() for name in
                  ('oooo', 'ovoo', 'ovov', 'oovv', 'ovvo', 'ovvv', 'vvvv',
                   'fock', 'mo_energy')}
        views = []

        def remember_view(env):
            views.append(weakref.ref(env['eris']))

        mycc = cc.RCCSD(mf)
        mycc.callback = remember_view
        energy, t1, t2 = mycc.kernel(eris=eris)
        self.assertTrue(mycc.converged)
        self.assertAlmostEqual(energy, ref_energy, 11)
        np.testing.assert_allclose(t1, ref1, atol=1e-11, rtol=0)
        np.testing.assert_allclose(t2, ref2, atol=1e-11, rtol=0)
        self.assertIs(eris.ovvv, original_ovvv)
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(eris, name), value)
        self.assertTrue(views)
        self.assertTrue(all(view() is None for view in views))
        self.assertAlmostEqual(mycc.ccsd_t(eris=eris), reference.ccsd_t(eris=eris), 11)
        mycc.solve_lambda(eris=eris)
        self.assertTrue(mycc.converged_lambda)

    def test_shifted_update(self):
        mycc = cc.RCCSD(shifted_mf)
        mycc.max_cycle = 1
        self.assertTrue(np.iscomplexobj(shifted_eris.ovvv))
        with mock.patch.object(ccsd, 'update_amps', wraps=ccsd.update_amps) as fast:
            with mock.patch.object(rccsd, 'update_amps', wraps=rccsd.update_amps) as original:
                mycc.kernel(eris=shifted_eris)
        self.assertEqual(fast.call_count, 0)
        self.assertEqual(original.call_count, 1)
        self.assertIs(original.call_args.args[3], shifted_eris)

    def test_complex_orbitals_at_gamma(self):
        phases = np.exp(1j * np.linspace(0, .3, mf.mo_coeff.shape[1]))
        mycc = cc.RCCSD(mf, mo_coeff=mf.mo_coeff * phases)
        mycc.max_cycle = 1
        complex_eris = mycc.ao2mo()
        with mock.patch.object(ccsd, 'update_amps', wraps=ccsd.update_amps) as fast:
            with mock.patch.object(rccsd, 'update_amps', wraps=rccsd.update_amps) as original:
                mycc.kernel(eris=complex_eris)
        self.assertEqual(fast.call_count, 0)
        self.assertEqual(original.call_count, 1)
        self.assertIs(original.call_args.args[3], complex_eris)


if __name__ == '__main__':
    unittest.main()
