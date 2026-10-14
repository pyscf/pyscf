# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Larger excitation spaces and eight-electron CCSDTQ validation."""

from math import sqrt
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pyscf import cc, gto, lib, scf
from pyscf.cc import _uccsdtq_blocked as blocked
from pyscf.cc import _uccsdtq_native as direct
from pyscf.cc import _uccsdtq_utils as storage
from pyscf.cc import uccsdtq
from pyscf.cc.test.test_uccsdtq_blocked import _direct_family
from pyscf.cc.test import uccsdtq_reference as reference


def _native_case(seed=241):
    """All five nonempty T4 sectors without constructing a spin-orbital T4."""
    rng = np.random.default_rng(seed)
    mycc = SimpleNamespace(nocc=(4, 4), nmo=(8, 8), cc_order=4,
                           do_tri_max_t=True, do_diis_max_t=True, max_memory=2000)
    tamps = storage.vector_to_amplitudes(
        mycc, rng.normal(scale=.02, size=storage.vector_size(mycc)))
    fock = tuple(rng.normal(scale=.1, size=(8, 8)) for _ in range(2))
    same_spin = []
    for _ in range(2):
        g = rng.normal(scale=.1, size=(8,) * 4)
        g = g - g.swapaxes(0, 1)
        g = g - g.swapaxes(2, 3)
        same_spin.append(g)
    mixed = rng.normal(scale=.1, size=(8,) * 4)
    imds = SimpleNamespace(t1_fock=fock, t1_eris=(same_spin[0], mixed, same_spin[1]))
    return mycc, imds, tamps[1:]


class ExpandedContractions(unittest.TestCase):
    def setUp(self):
        self.previous_threads = lib.num_threads()
        lib.num_threads(1)

    def tearDown(self):
        lib.num_threads(self.previous_threads)

    def assert_close(self, actual, expected):
        for rank_a, rank_b in zip(actual, expected):
            for a, b in zip(rank_a, rank_b):
                np.testing.assert_allclose(a, b, atol=2e-11, rtol=2e-11)

    def test_all_five_quadruple_sectors_full_compact_and_threads(self):
        mycc, imds, (t2, t3, packed) = _native_case()
        self.assertEqual(tuple(t.size for t in packed), (1, 256, 1296, 256, 1))
        self.assertTrue(all(np.linalg.norm(t) > 1e-8 for t in packed))
        expected_packed = direct.residuals(mycc, imds, t2, t3, packed)
        full = storage.unpack_t4(mycc, packed)
        for compact, t4 in ((True, packed), (False, full)):
            mycc.do_tri_max_t = compact
            expected = (expected_packed if compact else expected_packed[:2] +
                        (storage.unpack_t4(mycc, expected_packed[2]),))
            for threads in (1, 4):
                with self.subTest(compact=compact, threads=threads):
                    lib.num_threads(threads)
                    actual = blocked.residuals(mycc, imds, t2, t3, t4)
                    self.assert_close(actual, expected)

    def test_nonzero_t4_families_in_eight_electron_space(self):
        mycc, imds, (t2, t3, t4) = _native_case(seed=242)
        for family in ('F*T4', 'V*T4', 'V*T2*T4'):
            with self.subTest(family=family):
                expected = _direct_family(mycc, imds, t2, t3, t4, family)
                actual = blocked.residuals(mycc, imds, t2, t3, t4, _families=(family,))
                self.assert_close(actual, expected)
                for block in actual[2]:
                    self.assertGreater(np.linalg.norm(block), 1e-10)

    def test_all_five_sectors_under_small_fused_workspace(self):
        mycc, imds, (t2, t3, t4) = _native_case(seed=245)
        expected = direct.residuals(mycc, imds, t2, t3, t4)
        blocked._plans((4, 4, 4, 4), None)
        resident = lib.current_memory()[0]
        mycc.max_memory = resident + .35
        with patch.object(blocked.lib, 'current_memory', return_value=(resident, 0)):
            actual = blocked.residuals(mycc, imds, t2, t3, t4)
        self.assert_close(actual, expected)
        stats = mycc._uccsdtq_stats
        self.assertGreater(stats['fused_calls'], 0)
        self.assertLessEqual(stats['max_cache_bytes'], stats['cache_limit_bytes'])
        self.assertLessEqual(stats['max_fused_buffer_bytes'], stats['workspace_bytes'])

    def test_nh_arbitrary_nonzero_t4_against_exact_bch(self):
        mol = gto.M(atom='N 0 0 0; H 0 0 1.036', basis='sto-3g', spin=2, verbose=0)
        mf = scf.UHF(mol).run(conv_tol=1e-13)
        solver = cc.UCCSDTQ(mf)
        solver.quadruples_backend = 'blocked'
        solver.do_diis_max_t = True
        eris = solver.ao2mo()
        rng = np.random.default_rng(243)
        tamps = solver.vector_to_amplitudes(rng.normal(scale=.01, size=solver.vector_size()))
        self.assertEqual(solver.nocc, (5, 3))
        self.assertGreater(np.linalg.norm(tamps[3][3]), 1e-5)
        energy, expected, details = reference.kernel(solver, eris, tamps, True)
        actual_native = uccsdtq.residuals(solver, tamps, eris)
        actual = reference.spinorbital_inputs(solver, eris, actual_native)['tamps']
        self.assertEqual(len(details['determinants']), 120)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, atol=2e-10, rtol=2e-10)
        self.assertAlmostEqual(solver.energy(tamps, eris), energy, 11)

    def test_virtual_truncated_nh_eight_electrons_bch(self):
        # This validation model retains seven MOs per spin from 6-31G.
        # Only virtuals are frozen; all eight electrons remain correlated.
        # Its maximum excitation rank is five, so CCSDTQ is not FCI.
        mol = gto.M(atom='N 0 0 0; H 0 0 1.036', basis='6-31g', spin=2, verbose=0)
        mf = scf.UHF(mol).run(conv_tol=1e-13)
        solver = cc.UCCSDTQ(mf, frozen=[7, 8, 9, 10])
        solver.quadruples_backend = 'blocked'
        solver.do_diis_max_t = True
        eris = solver.ao2mo()
        rng = np.random.default_rng(244)
        tamps = solver.vector_to_amplitudes(rng.normal(scale=.01, size=solver.vector_size()))
        self.assertEqual(solver.nocc, (5, 3))
        self.assertEqual(solver.nmo, (7, 7))
        energy, expected, details = reference.kernel(solver, eris, tamps, True)
        actual_native = uccsdtq.residuals(solver, tamps, eris)
        actual = reference.spinorbital_inputs(solver, eris, actual_native)['tamps']
        ranks = np.array([reference._bit_count(details['reference_bits'] & ~bits)
                          for bits in details['determinants']])
        self.assertEqual(len(ranks), 735)
        self.assertEqual(ranks.max(), 5)
        self.assertGreater(np.linalg.norm(details['exp_t'][ranks == 5]), 1e-9)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, atol=2e-10, rtol=2e-10)
        self.assertAlmostEqual(solver.energy(tamps, eris), energy, 11)

    def test_eight_correlated_electron_ch4_restricted_limit(self):
        a = 1.09 / sqrt(3)
        atom = [('C', (0, 0, 0))] + [
            ('H', tuple(a * x for x in signs)) for signs in
            ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]
        mol = gto.M(atom=atom, basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-13)
        expected = cc.RCCSDTQ(mf, frozen=1).run(conv_tol=1e-12,
                                              conv_tol_normt=1e-10, max_cycle=100)
        seed = cc.UCCSDT(mf.to_uhf(), frozen=1, compact_tamps=True).run(
            conv_tol=1e-11, conv_tol_normt=1e-9, max_cycle=100)
        solver = cc.UCCSDTQ(mf.to_uhf(), frozen=1)
        solver.quadruples_backend = 'blocked'
        solver.conv_tol, solver.conv_tol_normt = 1e-11, 1e-9
        solver.max_cycle = 100
        solver.kernel(tamps=seed.tamps)
        self.assertTrue(expected.converged and seed.converged and solver.converged)
        self.assertEqual(solver.nocc, (4, 4))
        self.assertEqual(solver.nmo, (8, 8))
        self.assertEqual(tuple(t.size for t in solver.t4), (1, 256, 1296, 256, 1))
        self.assertGreater(sum(np.linalg.norm(t) for t in solver.t4), 1e-5)
        self.assertAlmostEqual(solver.e_tot, expected.e_tot, 9)


if __name__ == '__main__':
    unittest.main()
