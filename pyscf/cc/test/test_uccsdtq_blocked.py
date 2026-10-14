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

"""Blocked binary-contraction backend versus the direct native evaluator."""

from functools import lru_cache
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from pyscf import cc, fci, gto, lib, scf
from pyscf.cc import _uccsdtq_blocked as blocked
from pyscf.cc import _uccsdtq_native as direct
from pyscf.cc import _uccsdtq_utils as storage
from pyscf.cc import uccsdtq
from pyscf.cc.test.test_uccsdtq_native import _case


FAMILIES = ('F*T4', 'F*T2*T3', 'V*T3', 'V*T4', 'V*T2*T2',
            'V*T2*T3', 'V*T2*T4', 'V*T3*T3', 'V*T2*T2*T2')


def _family(term):
    names = []
    for operand in term.factor[:term.nfactor]:
        number = operand.array
        names.append('F' if number < 2 else 'V' if number < 5 else
                     'T2' if number < 8 else 'T3' if number < 12 else 'T4')
    return '*'.join(sorted(names, key=lambda name: ('F', 'V', 'T2', 'T3', 'T4').index(name)))


def _direct_family(mycc, imds, t2, t3, t4, family):
    descriptors = direct._descriptors

    @lru_cache(maxsize=None)
    def filtered(rank, nalpha):
        rows = [row for row in descriptors(rank, nalpha) if _family(row) == family]
        return (direct._Term * len(rows))(*rows)

    # Filter the independently evaluated native records rather than sharing
    # the blocked backend's expression planner or contraction code.
    with patch.object(direct, '_descriptors', filtered):
        return direct.residuals(mycc, imds, t2, t3, t4)


class KnownValues(unittest.TestCase):
    def setUp(self):
        self.threads = lib.num_threads()
        lib.num_threads(1)

    def tearDown(self):
        lib.num_threads(self.threads)

    def assert_blocks_close(self, actual, expected):
        for a, b in zip(actual, expected):
            for x, y in zip(a, b):
                np.testing.assert_allclose(x, y, atol=3e-12, rtol=3e-12)

    def test_auto_backend_selection_accounts_for_thread_count(self):
        cases = (((2, 2, 4, 4), 1, direct),
                 ((2, 2, 4, 4), 4, direct),
                 ((3, 2, 6, 7), 1, blocked),
                 ((3, 2, 6, 7), 4, blocked),
                 ((4, 4, 4, 4), 4, blocked),
                 ((3, 3, 8, 8), 4, blocked))
        previous_threads = lib.num_threads()
        try:
            for (oa, ob, va, vb), threads, expected in cases:
                with self.subTest(nocc=(oa, ob), nvir=(va, vb), threads=threads):
                    lib.num_threads(threads)
                    mycc = SimpleNamespace(nocc=(oa, ob), nmo=(oa + va, ob + vb),
                                           quadruples_backend='auto', verbose=0, stdout=None)
                    self.assertIs(uccsdtq._quadruples_backend(mycc), expected)
        finally:
            lib.num_threads(previous_threads)

    def test_each_family_in_all_five_spin_sectors(self):
        for nalpha in range(5):
            dims = (nalpha, 4 - nalpha, nalpha, 4 - nalpha)
            mycc, imds, amps, _ = _case(dims, seed=302 + nalpha)
            mycc.max_memory = 2000
            t4 = storage.pack_t4(mycc, amps[2])
            for family in FAMILIES:
                with self.subTest(nalpha=nalpha, family=family):
                    expected = _direct_family(mycc, imds, amps[0], amps[1], t4, family)
                    actual = blocked.residuals(mycc, imds, amps[0], amps[1], t4,
                                               _families=(family,))
                    self.assert_blocks_close(actual, expected)

    def test_full_compact_asymmetric_spaces_and_threads(self):
        for dims in ((3, 2, 2, 3), (2, 3, 3, 2), (4, 1, 4, 1)):
            mycc, imds, amps, _ = _case(dims, seed=88)
            mycc.max_memory = 2000
            original = tuple(tuple(a.copy() for a in blocks) for blocks in amps)
            for compact in (False, True):
                mycc.do_tri_max_t = compact
                t4 = storage.pack_t4(mycc, amps[2]) if compact else amps[2]
                expected = direct.residuals(mycc, imds, amps[0], amps[1], t4)
                for threads in (1, 4):
                    with self.subTest(dims=dims, compact=compact, threads=threads):
                        lib.num_threads(threads)
                        actual = blocked.residuals(mycc, imds, amps[0], amps[1], t4)
                        self.assert_blocks_close(actual, expected)
            for blocks, saved in zip(amps, original):
                for actual, expected in zip(blocks, saved):
                    np.testing.assert_array_equal(actual, expected)

    def test_strided_inputs_and_accumulating_outputs(self):
        mycc, imds, amps, _ = _case((3, 2, 2, 3), seed=233)
        mycc.max_memory = 2000
        imds.t1_fock = tuple(np.asfortranarray(a) for a in imds.t1_fock)
        imds.t1_eris = tuple(np.asfortranarray(a) for a in imds.t1_eris)
        t2, t3 = (tuple(np.asfortranarray(a) for a in blocks) for blocks in amps[:2])
        for compact in (False, True):
            mycc.do_tri_max_t = compact
            t4 = storage.pack_t4(mycc, amps[2]) if compact else amps[2]
            t4 = tuple(np.asfortranarray(a) for a in t4)
            expected = direct.residuals(mycc, imds, t2, t3, t4)
            out = tuple(tuple(np.full(a.shape, .375, order='F') for a in rank)
                        for rank in expected)
            returned = blocked.residuals(mycc, imds, t2, t3, t4, out=out)
            self.assertIs(returned, out)
            self.assert_blocks_close(out, tuple(tuple(a + .375 for a in rank)
                                                for rank in expected))

    def test_fused_worker_count_respects_minimum_panel_budget(self):
        # Exercise 112 requested workers with a 21-double per-buffer budget.
        # The native call is replaced, so this checks allocation planning
        # without launching workers or doing tensor contractions.
        mycc = SimpleNamespace(nocc=(8, 8), nmo=(10, 10), do_tri_max_t=True)
        shapes = [(10, 10)]*2 + [(10,)*4]*3
        shapes += [storage.block_shape(mycc, rank, block, rank == 4)
                   for rank in (2, 3, 4) for block in range(rank+1)]
        arrays = [np.zeros(shape) for shape in shapes]
        seen = {}

        def capture_plan(*arguments):
            plan = arguments[8]._obj
            seen.update(workers=plan.nworkers, capacity=plan.capacity)
            return 0

        with patch.object(blocked, '_fused_kernel', return_value=capture_plan):
            executor = blocked._Executor(mycc, arrays, direct._metadata(mycc, arrays), 2048)
        executor.threads, executor.blasserial = 112, True
        left = blocked._Node(0, (8, 0))
        right = blocked._Node(14, (8, 1, 4, 5, 2, 3, 6, 7))
        dimensions = {i: (2 if 4 <= i < 8 else 8) for i in range(9)}
        job = (4, 2, 1., left.labels, left, right, {}, dimensions)
        output = np.zeros(storage.block_shape(mycc, 4, 2, True))
        executor._final(job, np.arange(12, dtype=np.int32)[None, :],
                        np.ones(1), output)
        self.assertGreater(seen['workers'], 1)
        self.assertLessEqual(seen['workers']*seen['capacity'], executor.limit)
        self.assertLessEqual(sum(buffer.nbytes for buffer in executor.buffers),
                             3*executor.limit*8)

    def test_workspace_budget_guard(self):
        mycc, imds, amps, _ = _case((2, 2, 2, 2), seed=301)
        mycc.max_memory = 0
        t4 = storage.pack_t4(mycc, amps[2])
        with self.assertRaises(MemoryError):
            blocked.residuals(mycc, imds, amps[0], amps[1], t4)

    def test_changed_amplitudes_do_not_reuse_old_intermediates(self):
        mycc, imds, amps, _ = _case((3, 1, 3, 1), seed=903)
        mycc.max_memory = 2000
        t4 = storage.pack_t4(mycc, amps[2])
        first = blocked.residuals(mycc, imds, amps[0], amps[1], t4)
        changed_t2 = tuple(1.07 * t for t in amps[0])
        changed_t4 = tuple(t + .017 for t in t4)
        second = blocked.residuals(mycc, imds, changed_t2, amps[1], changed_t4)
        expected = direct.residuals(mycc, imds, changed_t2, amps[1], changed_t4)
        self.assert_blocks_close(second, expected)
        difference = max((np.max(np.abs(a - b), initial=0)
                          for rank_a, rank_b in zip(first, second)
                          for a, b in zip(rank_a, rank_b)), default=0)
        self.assertGreater(difference, 1e-8)

    def test_low_workspace_tiling_matches_large_workspace(self):
        mycc, imds, amps, _ = _case((2, 2, 4, 4), seed=931)
        t4 = storage.pack_t4(mycc, amps[2])
        # Plan construction is resident metadata, outside this explicit
        # work-panel/cache budget comparison.
        blocked._plans((2, 2, 4, 4), None)
        resident = lib.current_memory()[0]
        mycc.max_memory = resident + 2
        expected = blocked.residuals(mycc, imds, amps[0], amps[1], t4)
        high_stats = mycc._uccsdtq_stats.copy()
        original_caps = blocked._tile_caps
        split = dict(free=False, contracted=False)

        def record_caps(rows, cols, shared, ranges, limit, **kwargs):
            caps = original_caps(rows, cols, shared, ranges, limit, **kwargs)
            split['free'] |= any(caps[i] < ranges[i][1] for i in rows + cols)
            split['contracted'] |= any(caps[i] < ranges[i][1] for i in shared)
            return caps

        # Stabilize only the RSS reading so allocator caching from the first
        # evaluation does not change the explicit 0.06 MB work budget.
        mycc.max_memory = resident + .06
        with patch.object(blocked.lib, 'current_memory', return_value=(resident, 0)), \
                patch.object(blocked, '_tile_caps', record_caps):
            actual = blocked.residuals(mycc, imds, amps[0], amps[1], t4)
        stats = mycc._uccsdtq_stats
        self.assert_blocks_close(actual, expected)
        self.assertTrue(split['free'])
        self.assertTrue(split['contracted'])
        self.assertGreater(stats['six_index_batches'], high_stats['six_index_batches'])
        self.assertLessEqual(stats['max_panel_elements'] * 8, stats['panel_limit_bytes'])
        self.assertLessEqual(stats['max_cache_bytes'], stats['cache_limit_bytes'])

    def test_threaded_occupied_batches_match_direct_with_small_workspace(self):
        if lib.load_library('libccsdt').UCCSDTQblas_parallel():
            self.skipTest('Occupied batches use outer threads with serial BLAS')
        dims = (4, 2, 3, 2)
        mycc, imds, amps, _ = _case(dims, seed=932)
        t4 = storage.pack_t4(mycc, amps[2])
        expected = direct.residuals(mycc, imds, amps[0], amps[1], t4)
        blocked._plans(dims, None)
        resident = lib.current_memory()[0]
        lib.num_threads(4)
        stats = []
        for budget in (2, .06):
            with self.subTest(workspace_mb=budget):
                # Fix only the RSS measurement, retaining actual panel
                # allocation and threaded contractions at both budgets.
                mycc.max_memory = resident + budget
                with patch.object(blocked.lib, 'current_memory',
                                  return_value=(resident, 0)):
                    actual = blocked.residuals(mycc, imds, amps[0], amps[1], t4)
                self.assert_blocks_close(actual, expected)
                current = mycc._uccsdtq_stats.copy()
                self.assertGreater(current['six_index_batches'], 0)
                self.assertLessEqual(current['max_panel_elements'] * 8,
                                     current['panel_limit_bytes'])
                self.assertLessEqual(current['max_cache_bytes'],
                                     current['cache_limit_bytes'])
                stats.append(current)
        self.assertGreater(stats[1]['six_index_batches'], stats[0]['six_index_batches'])

    def test_plans_bound_contraction_order_and_intermediate_rank(self):
        def check_node(node):
            if node.array is not None:
                return
            self.assertLessEqual(len(node.labels), 6)
            self.assertLessEqual(len(set(node.left.labels) | set(node.right.labels)), 10)
            check_node(node.left)
            check_node(node.right)

        # All five spin sectors are populated; differing virtual dimensions
        # exercise dimension-dependent choices between equivalent trees.
        for node, _, _, _, jobs in blocked._plans((4, 4, 7, 9), None):
            check_node(node)
            for (_, _, _, axes, left, right, _, _), _, _ in jobs:
                self.assertLessEqual(len(set(axes) | set(right.labels)), 10)
                check_node(left)

    def test_forced_blocked_solver_matches_four_electron_fci(self):
        mol = gto.M(atom='Be', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-13).to_uhf()
        exact = fci.FCI(mf)
        exact.conv_tol = 1e-13
        e_fci = exact.kernel()[0]
        seed = cc.UCCSDT(mf, compact_tamps=True).run(conv_tol=1e-11,
                                                  conv_tol_normt=1e-9)
        for compact, initial in ((True, None), (False, seed.tamps)):
            with self.subTest(compact=compact, initial='MP2' if initial is None else 'CCSDT'):
                solver = cc.UCCSDTQ(mf, compact_tamps=compact)
                solver.quadruples_backend = 'blocked'
                solver.conv_tol, solver.conv_tol_normt = 1e-11, 1e-9
                solver.max_cycle = 80
                solver.kernel(tamps=initial)
                self.assertTrue(solver.converged)
                self.assertEqual(solver._quadruples_backend_used, 'blocked')
                self.assertAlmostEqual(solver.e_tot, e_fci, 9)
                self.assertGreater(sum(np.linalg.norm(t) for t in solver.t4), 1e-8)


if __name__ == '__main__':
    unittest.main()
