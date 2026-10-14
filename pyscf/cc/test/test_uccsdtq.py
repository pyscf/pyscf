#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

"""End-to-end UHF CCSDTQ energy and residual validation."""

import copy
from itertools import combinations, product
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from pyscf import cc, fci, gto, lib, scf
from pyscf.cc import uccsdtq, _uccsdtq_utils as utils
from pyscf.cc.test import uccsdtq_reference as reference


def _solve(mf, compact=True, frozen=None, **kwargs):
    solver = cc.UCCSDTQ(mf, frozen=frozen, compact_tamps=compact)
    solver.conv_tol = 1e-11
    solver.conv_tol_normt = 1e-9
    solver.max_cycle = 80
    solver.verbose = 0
    for key, value in kwargs.items():
        setattr(solver, key, value)
    solver.kernel()
    return solver


def setUpModule():
    global old_threads, rhf_be, uhf_be, uhf_b, be_compact
    old_threads = lib.num_threads()
    lib.num_threads(1)
    be = gto.M(atom='Be', basis='sto-3g', verbose=0)
    rhf_be = scf.RHF(be).run(conv_tol=1e-13)
    uhf_be = rhf_be.to_uhf()
    boron = gto.M(atom='B', basis='sto-3g', spin=1, verbose=0)
    uhf_b = scf.UHF(boron).run(conv_tol=1e-13)
    be_compact = _solve(uhf_be)


def tearDownModule():
    lib.num_threads(old_threads)


class KnownValues(unittest.TestCase):
    def test_four_electrons_matches_fci(self):
        self.assertTrue(be_compact.converged)
        exact = fci.FCI(uhf_be)
        exact.conv_tol = 1e-13
        e_fci = exact.kernel()[0]
        self.assertAlmostEqual(be_compact.e_tot, e_fci, 9)
        self.assertGreater(sum(np.linalg.norm(x) for x in be_compact.tamps[3]), 1e-8)
        _, residuals = reference.kernel(be_compact)
        self.assertLess(max(np.max(np.abs(x), initial=0) for x in residuals), 2e-8)

    def test_closed_shell_rccsdtq_limit(self):
        restricted = cc.RCCSDTQ(rhf_be).run(conv_tol=1e-11, conv_tol_normt=1e-9)
        self.assertTrue(restricted.converged)
        self.assertAlmostEqual(be_compact.e_tot, restricted.e_tot, 9)

    def test_compact_full_agree(self):
        full = _solve(uhf_be, compact=False)
        self.assertTrue(full.converged)
        self.assertAlmostEqual(full.e_tot, be_compact.e_tot, 10)
        expanded = be_compact.tamps_tri2full(be_compact.tamps)
        for a, b in zip(full.tamps, expanded):
            for x, y in zip(a, b):
                np.testing.assert_allclose(x, y, atol=2e-9)
        np.testing.assert_allclose(full.amplitudes_to_vector(full.tamps),
                                   be_compact.amplitudes_to_vector(be_compact.tamps),
                                   atol=2e-9)

    def test_open_shell_and_spin_flip(self):
        positive = _solve(uhf_b)
        self.assertTrue(positive.converged)
        exact = fci.FCI(uhf_b)
        exact.conv_tol = 1e-13
        # B/STO-3G has 5 electrons but at most four excitations in this
        # finite orbital space; CCSDTQ is therefore exactly FCI here.
        self.assertAlmostEqual(positive.e_tot, exact.kernel()[0], 9)
        flipped = uhf_b.copy()
        flipped.mol = uhf_b.mol.copy()
        flipped.mol.spin = -uhf_b.mol.spin
        flipped.mo_coeff = uhf_b.mo_coeff[::-1].copy()
        flipped.mo_occ = uhf_b.mo_occ[::-1].copy()
        flipped.mo_energy = uhf_b.mo_energy[::-1].copy()
        negative = _solve(flipped)
        self.assertTrue(negative.converged)
        self.assertEqual(negative.nocc, positive.nocc[::-1])
        self.assertAlmostEqual(positive.e_tot, negative.e_tot, 9)

    def test_frozen_two_electron_limit(self):
        frozen_integer = _solve(uhf_be, frozen=1)
        frozen_list = _solve(uhf_be, frozen=([0], [0]))
        triples = cc.UCCSDT(uhf_be, frozen=1).run(conv_tol=1e-11,
                                               conv_tol_normt=1e-9)
        self.assertTrue(frozen_integer.converged)
        self.assertTrue(frozen_list.converged)
        self.assertAlmostEqual(frozen_integer.e_tot, triples.e_tot, 10)
        self.assertAlmostEqual(frozen_integer.e_tot, frozen_list.e_tot, 11)
        self.assertEqual(sum(x.size for x in frozen_integer.tamps[3]), 0)

    def test_diis_including_t4(self):
        all_ranks = _solve(uhf_be, do_diis_max_t=True)
        self.assertTrue(all_ranks.converged)
        self.assertAlmostEqual(all_ranks.e_tot, be_compact.e_tot, 9)
        vector = all_ranks.amplitudes_to_vector(all_ranks.tamps)
        restored = all_ranks.vector_to_amplitudes(vector)
        self.assertEqual(len(restored), 4)
        np.testing.assert_array_equal(all_ranks.amplitudes_to_vector(restored), vector)

    def test_factory_dispatch_and_rejections(self):
        self.assertIsInstance(cc.CCSDTQ(uhf_be), uccsdtq.UCCSDTQ)
        self.assertIsInstance(uhf_be.CCSDTQ(), uccsdtq.UCCSDTQ)
        with self.assertRaises(NotImplementedError):
            cc.CCSDTQ(uhf_be.to_ghf())
        with self.assertRaises(NotImplementedError):
            cc.UCCSDTQ(uhf_be.to_ghf())
        complex_coeff = uhf_be.mo_coeff.astype(complex)
        with self.assertRaises(NotImplementedError):
            cc.UCCSDTQ(uhf_be, mo_coeff=complex_coeff)

    def test_rohf_reference_has_noncanonical_spin_fock(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .9; H 0 0 2.1',
                    basis='sto-3g', spin=1, verbose=0)
        rohf = scf.ROHF(mol).run(conv_tol=1e-13)
        solver = cc.CCSDTQ(rohf)
        self.assertIsInstance(solver, uccsdtq.UCCSDTQ)
        eris = solver.ao2mo()
        fov = max(np.linalg.norm(f[:no, no:]) for f, no in zip(eris.fock, solver.nocc))
        self.assertGreater(fov, 1e-5)
        solver.conv_tol, solver.conv_tol_normt = 1e-11, 1e-9
        solver.kernel(eris=eris)
        self.assertTrue(solver.converged)
        exact = fci.FCI(rohf.to_uhf())
        exact.conv_tol = 1e-13
        self.assertAlmostEqual(solver.e_tot, exact.kernel()[0], 9)

    def test_frozen_virtual_asymmetric_spaces(self):
        compact = _solve(uhf_be, frozen=([4], []))
        full = _solve(uhf_be, compact=False, frozen=([4], []))
        self.assertEqual(compact.nmo, (4, 5))
        self.assertTrue(compact.converged)
        self.assertTrue(full.converged)
        self.assertAlmostEqual(compact.e_tot, full.e_tot, 10)
        _, residuals = reference.kernel(compact)
        self.assertLess(max(np.max(np.abs(x), initial=0) for x in residuals), 2e-8)

    def test_compact_uccsdt_seed(self):
        seed = cc.UCCSDT(uhf_be, compact_tamps=True).run(conv_tol=1e-11,
                                                       conv_tol_normt=1e-9)
        self.assertEqual(seed.tamps[2][0].ndim, 2)
        solver = cc.UCCSDTQ(uhf_be)
        solver.conv_tol, solver.conv_tol_normt = 1e-11, 1e-9
        solver.kernel(tamps=copy.deepcopy(seed.tamps))
        self.assertTrue(solver.converged)
        self.assertEqual(solver.tamps[2][0].ndim, 2)
        self.assertAlmostEqual(solver.e_tot, be_compact.e_tot, 9)

    def test_packed_t3_storage_and_full_update(self):
        rng = np.random.default_rng(317)
        for backend in ('direct', 'blocked'):
            with self.subTest(backend=backend):
                packed = cc.UCCSDTQ(uhf_b)
                packed.quadruples_backend = backend
                packed.do_diis_max_t = True
                eris = packed.ao2mo()
                _, initial = packed.init_amps(eris)
                for rank in (3, 4):
                    self.assertEqual([a.shape for a in initial[rank-1]],
                                     [utils.block_shape(packed, rank, b, True) for b in range(rank+1)])
                tamps = packed.vector_to_amplitudes(rng.normal(scale=.007, size=packed.vector_size()))
                full_tamps = packed.tamps_tri2full(tamps)
                full = cc.UCCSDTQ(uhf_b, compact_tamps=False)
                full.quadruples_backend = backend
                self.assertLess(packed.memory_estimate()['T3'], full.memory_estimate()['T3'])
                with mock.patch.object(uccsdtq.uccsdt_highm, 'compute_r3_uhf',
                                       side_effect=AssertionError('Full T3 residual allocation')):
                    actual_norms = packed.update_amps_(tamps, eris)
                expected_norms = full.update_amps_(full_tamps, eris)
                np.testing.assert_allclose(actual_norms, expected_norms, atol=3e-12, rtol=3e-12)
                for actual_rank, expected_rank in zip(packed.tamps_tri2full(tamps), full_tamps):
                    for actual, expected in zip(actual_rank, expected_rank):
                        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)

    def test_arbitrary_noncanonical_residuals_match_bch(self):
        beh = gto.M(atom='Be 0 0 0; H 0 0 1.342', basis='sto-3g', spin=1, verbose=0)
        uhf_beh = scf.UHF(beh).run(conv_tol=1e-13)
        for name, mf in (('Be', uhf_be), ('BeH', uhf_beh)):
            with self.subTest(system=name):
                solver = cc.UCCSDTQ(mf)
                solver.do_diis_max_t = True
                eris = solver.ao2mo()
                rng = np.random.default_rng(64)
                tamps = solver.vector_to_amplitudes(
                    rng.normal(scale=.012, size=solver.vector_size()))
                # Include occupied-virtual Fock entries to exercise the
                # explicit-T1 and T1-dressed terms away from convergence.
                eris.fock = tuple(f + .025 * (x + x.T) for f, x in
                                  zip(eris.fock, rng.normal(size=(2,) + eris.fock[0].shape)))
                eris.focka, eris.fockb = eris.fock
                eris.mo_energy = tuple(f.diagonal().copy() for f in eris.fock)
                before = copy.deepcopy(tamps)
                energy, expected = reference.kernel(solver, eris, tamps)
                actual_native = uccsdtq.residuals(solver, tamps, eris)
                actual = reference.spinorbital_inputs(solver, eris, actual_native)['tamps']
                for rank, (x, y) in enumerate(zip(actual, expected), 1):
                    np.testing.assert_allclose(x, y, atol=2e-10, rtol=2e-10,
                                               err_msg='rank-%d residual' % rank)
                self.assertAlmostEqual(solver.energy(tamps, eris), energy, 11)
                for old_rank, new_rank in zip(before, tamps):
                    for x, y in zip(old_rank, new_rank):
                        np.testing.assert_array_equal(x, y)

    def test_four_spin_specific_denominators(self):
        rng = np.random.default_rng(62)
        model = SimpleNamespace(nocc=(4, 4), nmo=(8, 8), level_shift=.13,
                                do_tri_max_t=True, max_memory=2000)
        energies = (np.r_[-np.arange(1., 5), .7 * np.arange(1., 5)],
                    np.r_[-1.2 * np.arange(1., 5), .9 * np.arange(1., 5)])
        packed = tuple(rng.normal(size=utils.block_shape(model, 4, block, True))
                       for block in range(5))
        expected = copy.deepcopy(packed)
        for block, tensor in enumerate(expected):
            na, nb = 4 - block, block
            first = 0 if na >= nb else 1
            ranks = (max(na, nb), min(na, nb))
            groups = []
            signs = []
            for spin, rank in zip((first, 1 - first), ranks):
                if rank:
                    groups.extend((tuple(combinations(range(4), rank)),) * 2)
                    signs.extend(((spin, 1), (spin, -1)))
            for idx in product(*(range(len(x)) for x in groups)):
                denominator = -4 * model.level_shift
                for axis, coord in enumerate(idx):
                    spin, sign = signs[axis]
                    offset = 0 if sign == 1 else 4
                    denominator += sign * sum(energies[spin][offset + p]
                                              for p in groups[axis][coord])
                tensor[idx] /= denominator
        full = utils.unpack_t4(model, packed)
        uccsdtq.r4_divide_e_(model, packed, energies)
        for actual, ref in zip(packed, expected):
            np.testing.assert_allclose(actual, ref, atol=1e-15)
        model.do_tri_max_t = False
        uccsdtq.r4_divide_e_(model, full, energies)
        for actual, ref in zip(utils.pack_t4(model, full), expected):
            np.testing.assert_allclose(actual, ref, atol=1e-15)

    def test_openmp_thread_consistency(self):
        solver = cc.UCCSDTQ(uhf_b)
        solver.do_diis_max_t = True
        eris = solver.ao2mo()
        rng = np.random.default_rng(71)
        tamps = solver.vector_to_amplitudes(rng.normal(scale=.012, size=solver.vector_size()))
        previous_threads = lib.num_threads()
        try:
            lib.num_threads(1)
            expected = uccsdtq.residuals(solver, tamps, eris)
            for threads in (2, 4):
                lib.num_threads(threads)
                actual = uccsdtq.residuals(solver, tamps, eris)
                for rank_a, rank_b in zip(actual, expected):
                    for a, b in zip(rank_a, rank_b):
                        np.testing.assert_allclose(a, b, atol=1e-13, rtol=1e-13)
        finally:
            lib.num_threads(previous_threads)


if __name__ == '__main__':
    unittest.main()
