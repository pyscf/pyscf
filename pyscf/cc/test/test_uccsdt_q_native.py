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

"""Native UCCSDT(Q) contractions, strided storage, and OpenMP execution."""

import copy
import itertools
import math
import tempfile
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from pyscf import cc, lib, gto, scf
from pyscf.cc import uccsdt_q
from pyscf.cc.test import test_uccsdt_q as fixtures
from pyscf.cc.test import uccsdt_q_reference as reference


def python_moments(mycc, eris, nalpha, holes, starts, shape, terms=None):
    """Evaluate the unchanged NumPy implementation on the requested tile."""
    nocc = mycc.nocc
    nvir = tuple(n-o for n, o in zip(mycc.nmo, nocc))
    spins = (0,) * nalpha + (1,) * (4-nalpha)
    ranges = {(x, s): np.arange((nocc if x in 'mn' else nvir)[s])
              for x in 'mnef' for s in (0, 1)}
    for x, s, i in zip('ijkl', spins, holes):
        ranges[x, s] = np.array([i])
    for x, s, start, size in zip('abcd', spins, starts, shape):
        ranges[x, s] = np.arange(start, start+size)
    return uccsdt_q._moments(
        uccsdt_q._Amplitudes(mycc.tamps, nocc, nvir, mycc.do_tri_max_t),
        uccsdt_q._IntegralBlocks(eris, nocc),
        uccsdt_q._spin_terms(nalpha) if terms is None else terms,
        ranges, shape, lambda script, *a: np.einsum(script, *a, optimize=True))


def strided_copy(array, layout):
    """Preserve every value while changing the underlying memory strides."""
    if layout == 'negative':
        reverse = (slice(None, None, -1),) * array.ndim
        return array[reverse].copy()[reverse]
    if layout == 'transpose':
        return array.T.copy().T
    shape = list(array.shape)
    shape[-1] *= 2
    storage = np.empty(shape, dtype=array.dtype)
    view = storage[..., ::2]
    view[...] = array
    return view


class KnownValues(unittest.TestCase):
    def test_public_kernel_uses_native_contractions(self):
        mycc, eris = fixtures.random_native(2, compact=True)
        expected = reference.kernel(mycc, eris)
        with lib.with_omp_threads(2), mock.patch.object(
                uccsdt_q, '_moments', side_effect=AssertionError('Python contraction called')):
            actual = uccsdt_q.kernel(mycc, eris)
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)

    def test_native_per_seed_all_spin_sectors(self):
        # Separate contractions prevent compensating sign/factor errors.
        for nalpha in range(5):
            for compact in (False, True):
                mycc, eris = fixtures.random_native(nalpha, compact)
                spins = (0,) * nalpha + (1,) * (4-nalpha)
                holes = tuple(range(nalpha)) + tuple(range(4-nalpha))
                shape = tuple(mycc.nocc[s] for s in spins)
                for name in ('voov', 'oooo', 'vvvv', 'vvov', 'vooo', 'l2'):
                    with self.subTest(nalpha=nalpha, compact=compact, seed=name):
                        terms = tuple(t for t in uccsdt_q._spin_terms(nalpha) if t[0] == name)
                        expected = python_moments(mycc, eris, nalpha, holes, (0,)*4, shape, terms)
                        actual = uccsdt_q._native_moments(
                            mycc, eris, mycc.tamps, nalpha, holes, (0,)*4, shape, terms=terms)
                        for a, e in zip(actual, expected):
                            np.testing.assert_allclose(a, e, atol=1e-11, rtol=1e-10)

    def test_nonzero_virtual_tile_origins(self):
        for nalpha in range(5):
            for compact in (False, True):
                with self.subTest(nalpha=nalpha, compact=compact):
                    mycc, eris = fixtures.random_native(nalpha, compact)
                    holes = tuple(range(nalpha)) + tuple(range(4-nalpha))
                    starts = holes
                    expected = python_moments(mycc, eris, nalpha, holes, starts, (1,)*4)
                    actual = uccsdt_q._native_moments(
                        mycc, eris, mycc.tamps, nalpha, holes, starts, (1,)*4)
                    for a, e in zip(actual, expected):
                        np.testing.assert_allclose(a, e, atol=1e-11, rtol=1e-10)

    def test_one_two_four_thread_energies(self):
        for nalpha in range(5):
            for compact in (False, True):
                mycc, eris = fixtures.random_native(nalpha, compact)
                expected = reference.kernel(mycc, eris)
                for nthreads in (1, 2, 4):
                    with self.subTest(nalpha=nalpha, compact=compact, nthreads=nthreads):
                        with lib.with_omp_threads(nthreads):
                            actual = uccsdt_q.kernel(mycc, eris)
                        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)

    def test_openmp_team_really_executes(self):
        # BeH has multiple distinct quadruples after permutation reduction.
        # Check the team size inside C, not merely the Python thread setting.
        mol = gto.M(atom='Be 0 0 0; H 0 0 1.342', basis='sto-3g', spin=1, verbose=0)
        mf = scf.UHF(mol).run(conv_tol=1e-12)
        mycc = cc.UCCSDT(mf, compact_tamps=True).run(conv_tol=1e-10)
        eris = mycc.ao2mo()
        expected = reference.kernel(mycc, eris)
        native_batch = uccsdt_q._native_batch
        for nthreads in (1, 2, 4):
            observed = []

            def record_batch(*args, **kwargs):
                result = native_batch(*args, **kwargs)
                observed.append(result[2])
                return result

            with self.subTest(nthreads=nthreads), lib.with_omp_threads(nthreads):
                with mock.patch.object(uccsdt_q, '_native_block_size', return_value=1), \
                        mock.patch.object(uccsdt_q, '_native_batch', side_effect=record_batch):
                    actual = uccsdt_q.kernel(mycc, eris)
                self.assertIn(nthreads, observed)
                self.assertTrue(all(1 <= n <= nthreads for n in observed))
                np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)

    def test_native_tasks_use_one_call_when_budget_allows(self):
        mycc, eris = fixtures.random_native(2, compact=True)
        expected = reference.kernel(mycc, eris)
        native_batch = uccsdt_q._native_batch
        with lib.with_omp_threads(2), mock.patch.object(
                uccsdt_q, '_native_batch', wraps=native_batch) as wrapped:
            actual = uccsdt_q.kernel(mycc, eris)
        self.assertEqual(wrapped.call_count, 1)
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)

    def test_default_tile_uses_cache_aware_target(self):
        records, _, offsets = uccsdt_q._native_term_table()
        mycc = SimpleNamespace(max_memory=4000.)
        with mock.patch.object(lib, 'current_memory', return_value=(0., 0.)):
            block = uccsdt_q._native_block_size(
                mycc, (3, 3), (24, 24), records, offsets, 1)
        self.assertEqual(block, 12)

    def test_native_strided_inputs(self):
        # Reversed backing storage yields the same tensors with negative
        # strides; this tests indexing without changing the physical data.
        for nalpha in (1, 2, 3):
            for compact in (False, True):
                original, original_eris = fixtures.random_native(nalpha, compact)
                expected = reference.kernel(original, original_eris)
                for layout in ('slice', 'negative', 'transpose'):
                    with self.subTest(nalpha=nalpha, compact=compact, layout=layout):
                        mycc, eris = copy.copy(original), copy.copy(original_eris)
                        mycc.tamps = [tuple(strided_copy(a, layout) for a in group)
                                      for group in original.tamps]
                        for name in ('pppp', 'PPPP', 'pPpP'):
                            setattr(eris, name, strided_copy(getattr(original_eris, name), layout))
                        for name in ('mo_energy', 'fock'):
                            setattr(eris, name, tuple(strided_copy(a, layout)
                                                    for a in getattr(original_eris, name)))
                        with lib.with_omp_threads(2):
                            actual = uccsdt_q.kernel(mycc, eris)
                        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)

    def test_tile_size_independence(self):
        for compact in (False, True):
            mycc, eris = fixtures.random_native(1, compact)
            expected = reference.kernel(mycc, eris)
            for block_size in (1, 2, 3):
                with self.subTest(compact=compact, block_size=block_size):
                    with lib.with_omp_threads(4), mock.patch.object(
                            uccsdt_q, '_native_block_size', return_value=block_size):
                        actual = uccsdt_q.kernel(mycc, eris)
                    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)

    def test_native_thread_workspace_budget(self):
        nocc, nvir = (3, 3), (12, 12)
        sizes = np.asarray((*nocc, *nvir), dtype=np.int64)
        records, _, offsets = uccsdt_q._native_term_table()
        workspace = uccsdt_q._native_library().UCCSDTQ_workspace

        def required(block):
            return workspace(sizes.ctypes.data, records.ctypes.data, offsets.ctypes.data, block)

        # The full tile fits one worker, while four workers must reduce the
        # tile. Use C's actual workspace estimate rather than a test formula.
        resident = 37.
        budget = required(max(nvir)) + 16384 + 8192*4 + 100
        mycc = SimpleNamespace(max_memory=resident+budget/1e6)
        blocks = {}
        with mock.patch.object(lib, 'current_memory', return_value=(resident, 0.)):
            for nthreads in (1, 4):
                blocks[nthreads] = uccsdt_q._native_block_size(
                    mycc, nocc, nvir, records, offsets, nthreads)
                ntasks = uccsdt_q._task_count(nocc, nvir, blocks[nthreads])
                self.assertLessEqual(required(blocks[nthreads])*min(ntasks, nthreads),
                                     budget-16384-8192*nthreads)
            self.assertLess(blocks[4], blocks[1])
            mycc.max_memory = resident + (required(1)*4 + 16384 + 8192*4 - 1)/1e6
            with self.assertRaises(MemoryError):
                uccsdt_q._native_block_size(mycc, nocc, nvir, records, offsets, 4)

    def test_same_spin_tile_coverage_and_multiplicity(self):
        nocc, nvir = (3, 2), (5, 4)
        for block in (1, 2, 3, 4, 5):
            with self.subTest(block=block):
                expected, ordered_volume = set(), 0
                for na in range(5):
                    nb = 4-na
                    spins = (0,)*na + (1,)*nb
                    for ha in itertools.combinations(range(nocc[0]), na):
                        for hb in itertools.combinations(range(nocc[1]), nb):
                            for va in itertools.combinations(range(nvir[0]), na):
                                for vb in itertools.combinations(range(nvir[1]), nb):
                                    starts = tuple(i//block*block for i in va+vb)
                                    sizes = tuple(min(block, nvir[s]-i) for s, i in zip(spins, starts))
                                    expected.add((na, *ha, *hb, *starts, *sizes))
                                    ordered_volume += math.factorial(na)*math.factorial(nb)
                tasks = list(uccsdt_q._native_tasks(nocc, nvir, block))
                self.assertEqual(set(tasks), expected)
                self.assertEqual(len(tasks), len(expected))
                self.assertEqual(uccsdt_q._task_count(nocc, nvir, block), len(tasks))
                volume = 0
                for task in tasks:
                    na = task[0]
                    ways = uccsdt_q._tile_multiplicity(task)
                    for spin, starts in enumerate((task[5:5+na], task[5+na:9])):
                        for start, group in itertools.groupby(starts):
                            count = sum(1 for _ in group)
                            capacity = min(block, nvir[spin]-start)
                            ways *= math.factorial(capacity)//math.factorial(capacity-count)
                    volume += ways
                self.assertEqual(volume, ordered_volume)

    def test_fine_virtual_tiles_are_streamed(self):
        tracemalloc.start()
        try:
            task = next(uccsdt_q._native_tasks((3, 1), (80, 1), 1))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(task[5:9], (0, 1, 2, 0))
        # Materializing the 82,160 alpha triples takes several MB, although
        # this request only needs the first constant-size task.
        self.assertLess(peak, 500000)

    def test_native_rejects_insufficient_batch_workspace(self):
        mycc, eris = fixtures.random_native(2, compact=True)
        nocc, nvir = uccsdt_q._validate(mycc, eris, mycc.tamps)
        inputs = uccsdt_q._NativeInputs(mycc, eris, mycc.tamps, nocc, nvir)
        records, coefficients, offsets = uccsdt_q._native_term_table()
        block = max(nvir)
        required = uccsdt_q._native_library().UCCSDTQ_workspace(
            inputs.sizes.ctypes.data, records.ctypes.data, offsets.ctypes.data, block)
        tasks = [next(uccsdt_q._native_tasks(nocc, nvir, block))]
        with self.assertRaisesRegex(MemoryError, 'workspace'):
            uccsdt_q._native_batch(inputs, records, coefficients, offsets, tasks, 1, required-1)

    def test_malformed_integrals_rejected_before_c(self):
        mycc, eris = fixtures.random_native(2, compact=True)
        for name in ('pppp', 'PPPP', 'pPpP'):
            for axis in range(4):
                with self.subTest(block=name, axis=axis):
                    invalid = copy.copy(eris)
                    index = [slice(None)] * 4
                    index[axis] = slice(1, None)
                    setattr(invalid, name, getattr(eris, name)[tuple(index)])
                    with mock.patch.object(uccsdt_q, '_native_batch') as native:
                        with self.assertRaises(ValueError):
                            uccsdt_q.kernel(mycc, invalid)
                        native.assert_not_called()

    def test_mismatched_eri_occupied_counts_rejected_before_c(self):
        mycc, eris = fixtures.random_native(2, compact=True)
        eris.nocc = (eris.nocc[0] - 1, eris.nocc[1])
        with mock.patch.object(uccsdt_q, '_native_batch') as native:
            with self.assertRaisesRegex(ValueError, 'occupied counts'):
                uccsdt_q.kernel(mycc, eris)
            native.assert_not_called()

    def test_hdf5_storage_rejected_before_c(self):
        for kind in ('eri', 't2', 't3', 'energy'):
            with self.subTest(storage=kind), lib.H5TmpFile() as handle:
                mycc, eris = fixtures.random_native(2, compact=True)
                if kind == 'eri':
                    eris.pPpP = handle.create_dataset('input', data=eris.pPpP)
                elif kind == 'energy':
                    dataset = handle.create_dataset('input', data=eris.mo_energy[0])
                    eris.mo_energy = (dataset, eris.mo_energy[1])
                else:
                    rank = int(kind[-1])
                    mycc.tamps = [list(group) for group in mycc.tamps]
                    mycc.tamps[rank-1][1] = handle.create_dataset(
                        'input', data=mycc.tamps[rank-1][1])
                with mock.patch.object(uccsdt_q, '_native_batch') as native:
                    with self.assertRaisesRegex(TypeError, 'ndarray or memmap'):
                        uccsdt_q.kernel(mycc, eris)
                    native.assert_not_called()

    def test_readonly_memmap_storage(self):
        mycc, eris = fixtures.random_native(2, compact=True)
        expected = reference.kernel(mycc, eris)
        with tempfile.TemporaryDirectory() as directory:
            def mapped(array, name):
                path = '%s/%s.npy' % (directory, name)
                np.save(path, array)
                return np.load(path, mmap_mode='r')

            for name in ('pppp', 'pPpP', 'PPPP'):
                setattr(eris, name, mapped(getattr(eris, name), name))
            mycc.tamps = [[mapped(array, 't%d_%d' % (rank, spin))
                           for spin, array in enumerate(group)]
                          for rank, group in enumerate(mycc.tamps)]
            eris.mo_energy = tuple(mapped(array, 'energy%d' % spin)
                                   for spin, array in enumerate(eris.mo_energy))
            inputs = uccsdt_q._NativeInputs(mycc, eris, mycc.tamps,
                                           mycc.nocc,
                                           tuple(n-o for n, o in zip(mycc.nmo, mycc.nocc)))
            for pointer, array in zip(inputs.pointers, inputs.arrays):
                self.assertEqual(pointer, array.ctypes.data)
                self.assertFalse(array.flags.writeable)
            with lib.with_omp_threads(2):
                actual = uccsdt_q.kernel(mycc, eris)
            np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)

    def test_unaligned_inputs_rejected_before_c(self):
        for rank in (0, 2, 3):
            with self.subTest(tensor='ERI' if rank == 0 else 'T%d' % rank):
                mycc, eris = fixtures.random_native(2, compact=True)
                original = eris.pPpP if rank == 0 else mycc.tamps[rank-1][1]
                unaligned = np.ndarray(original.shape, dtype=np.float64,
                                       buffer=bytearray(original.nbytes+1), offset=1)
                unaligned[...] = original
                self.assertFalse(unaligned.flags.aligned)
                if rank == 0:
                    eris.pPpP = unaligned
                else:
                    mycc.tamps = [list(group) for group in mycc.tamps]
                    mycc.tamps[rank-1][1] = unaligned
                with mock.patch.object(uccsdt_q, '_native_batch') as native:
                    with self.assertRaises(ValueError):
                        uccsdt_q.kernel(mycc, eris)
                    native.assert_not_called()
        for spin in (0, 1):
            with self.subTest(orbital_energy_spin=spin):
                mycc, eris = fixtures.random_native(2, compact=True)
                original = eris.mo_energy[spin]
                unaligned = np.ndarray(original.shape, dtype=np.float64,
                                       buffer=bytearray(original.nbytes+1), offset=1)
                unaligned[...] = original
                self.assertFalse(unaligned.flags.aligned)
                self.assertTrue(unaligned.flags.c_contiguous)
                eris.mo_energy = tuple(unaligned if s == spin else e
                                       for s, e in enumerate(eris.mo_energy))
                with mock.patch.object(uccsdt_q, '_native_batch') as native:
                    with self.assertRaises(ValueError):
                        uccsdt_q.kernel(mycc, eris)
                    native.assert_not_called()

    def test_simultaneous_native_calls(self):
        cases = [fixtures.random_native(1, compact=False),
                 fixtures.random_native(2, compact=True)]
        expected = [reference.kernel(mycc, eris) for mycc, eris in cases]
        # Compile the immutable formula table before launching two callers.
        uccsdt_q._native_term_table()
        barrier = Barrier(2)

        def calculate(case):
            mycc, eris = case
            with lib.with_omp_threads(2):
                barrier.wait(timeout=30)
                return [uccsdt_q.kernel(mycc, eris) for _ in range(3)]

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(calculate, cases))
        for attempts, energy in zip(results, expected):
            for actual in attempts:
                np.testing.assert_allclose(actual, energy, atol=1e-11, rtol=1e-10)


class OpenShellValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with lib.with_omp_threads(1):
            fixtures.setUpModule()

    def test_beh_bh2_native_threaded(self):
        for mycc, eris in ((fixtures.beh, fixtures.beh_eris), (fixtures.bh2, fixtures.bh2_eris)):
            self.assertTrue(mycc.converged)
            expected = reference.kernel(mycc, eris)
            full = cc.UCCSDT(mycc._scf, frozen=mycc.frozen, compact_tamps=False)
            full.tamps = [*mycc.tamps[:2], mycc.tamps_tri2full(mycc.tamps[2])]
            for solver in (mycc, full):
                for nthreads in (1, 2, 4):
                    with self.subTest(atom=mycc.mol.atom, compact=solver.do_tri_max_t,
                                      nthreads=nthreads):
                        with lib.with_omp_threads(nthreads):
                            actual = solver.ccsdt_q(eris=eris)
                        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
