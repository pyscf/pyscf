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

"""Fused final CCSDTQ contraction: origins, maps, sparse outputs and workers."""
import ctypes
from types import SimpleNamespace
import unittest

import numpy as np
from pyscf import lib
from pyscf.cc import _uccsdtq_utils as storage
from pyscf.cc.test.test_uccsdtq_blocked_native import _ptr, _dense_meta, _factor, _i32


class _Final(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int32) for name in
                ('rank', 'nalpha', 'oa', 'ob', 'va', 'vb', 'compact', 'nworkers',
                 'nrows', 'ncols', 'nshared', 'nmaps')]
    _fields_ += [(name, ctypes.c_int32*size) for name, size in
                 (('rows', 8), ('cols', 8), ('shared', 4),
                  ('starts', 12), ('counts', 12), ('caps', 12))]
    _fields_ += [('capacity', ctypes.c_int64)]


def _native_order(raw, rank, na):
    nb = rank-na
    if na >= nb:
        order = list(range(na)) + list(range(rank, rank+na))
        order += list(range(na, rank)) + list(range(rank+na, 2*rank))
    else:
        order = list(range(na, rank)) + list(range(rank+na, 2*rank))
        order += list(range(na)) + list(range(rank, rank+na))
    return raw.transpose(order)


def _mapped_result(raw, rank, maps, coefficients):
    result = np.zeros_like(raw)
    for mapping, coefficient in zip(maps, coefficients):
        inverse = np.argsort(np.array(mapping[4:4+rank])-4)
        result += coefficient * raw.transpose(tuple(range(rank)) + tuple(rank+i for i in inverse))
    return result


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = lib.load_library('libccsdt')
        assert cls.library.UCCSDTQfused_abi_version() == 1
        cls.library.UCCSDTQfinal_contract.argtypes = [ctypes.c_void_p]*17
        cls.library.UCCSDTQfinal_contract.restype = ctypes.c_int

    def setUp(self):
        self.threads = lib.num_threads()
        lib.num_threads(1)

    def tearDown(self):
        lib.num_threads(self.threads)

    def run_case(self, rank, na, dims, left, lm, lf, lo, right, rm, rf, ro,
                 rows, cols, shared, starts, counts, caps, maps, coefficients,
                 raw, capacity=4096, thread_counts=(1, 4)):
        plan = _Final()
        plan.rank, plan.nalpha = rank, na
        plan.oa, plan.ob, plan.va, plan.vb = dims
        plan.nrows, plan.ncols, plan.nshared, plan.nmaps = len(rows), len(cols), len(shared), len(maps)
        plan.rows[:len(rows)], plan.cols[:len(cols)], plan.shared[:len(shared)] = rows, cols, shared
        plan.starts[:], plan.counts[:], plan.caps[:] = starts, counts, caps
        plan.capacity = capacity
        mapped = _native_order(_mapped_result(raw, rank, maps, coefficients), rank, na)
        mycc = SimpleNamespace(nocc=dims[:2], nmo=(dims[0]+dims[2], dims[1]+dims[3]))
        packed = storage._pack_block(mycc, mapped, rank, rank-na)
        maps, coefficients, lo, ro = _i32(maps), np.asarray(coefficients), _i32(lo), _i32(ro)
        last = None
        for compact in (False, True):
            plan.compact = compact
            expected = packed if compact else storage._unpack_block(mycc, packed, rank, rank-na)
            results = []
            for threads in thread_counts:
                lib.num_threads(threads)
                self.assertEqual(self.library.UCCSDTQomp_threads(), threads)
                plan.nworkers = threads
                output = np.full(expected.shape, .125, order='F')
                strides = np.asarray(output.strides, dtype=np.int64)//8
                buffers = [np.full(threads*capacity+8, 1729.) for _ in range(3)]
                stats = np.zeros(6, dtype=np.int64)
                arguments = (_ptr(left), ctypes.byref(lm), ctypes.byref(lf), _ptr(lo),
                             _ptr(right), ctypes.byref(rm), ctypes.byref(rf), _ptr(ro),
                             ctypes.byref(plan), _ptr(maps), _ptr(coefficients),
                             *(_ptr(a) for a in buffers), _ptr(output), _ptr(strides), _ptr(stats))
                self.assertEqual(self.library.UCCSDTQfinal_contract(*arguments), 0)
                np.testing.assert_allclose(output, .125+expected, atol=2e-11, rtol=2e-11)
                for buffer in buffers:
                    np.testing.assert_array_equal(buffer[-8:], 1729.)
                self.assertLessEqual(stats[2], capacity)
                self.assertLessEqual(stats[5], threads)
                self.last_stats = stats.copy()
                results.append(output)
                last = plan, output, arguments
            np.testing.assert_allclose(*results, atol=2e-12, rtol=2e-12)
        return last

    def test_complete_occupied_tuples_use_more_workers_than_one_axis(self):
        if self.library.UCCSDTQblas_parallel() != 0:
            self.skipTest('Outer teams require a serial linked BLAS')
        rng = np.random.default_rng(933)
        no, nv = 4, 3
        left = rng.normal(size=(no, no, nv, nv, no, nv))
        right = rng.normal(size=(no, no, nv, nv, no, nv))
        raw = np.einsum('ijabme,klcdme->ijklabcd', left, right, optimize=True)
        counts = [no]*4 + [nv]*4 + [no, 0, nv, 0]
        self.run_case(4, 2, (no, no, nv, nv), left, _dense_meta(left),
                      _factor([0, 1, 4, 5, 8, 10]), [0]*12,
                      right, _dense_meta(right), _factor([2, 3, 6, 7, 8, 10]), [0]*12,
                      [0, 1, 4, 5], [2, 3, 6, 7], [8, 10], [0]*12, counts,
                      [max(1, c) for c in counts], [list(range(12))], [1.], raw,
                      thread_counts=(1, 16))
        self.assertEqual(self.last_stats[3], 36)
        self.assertEqual(self.last_stats[5], 16)

    def test_six_index_origins_cyclic_maps_full_and_partial_virtuals(self):
        rng = np.random.default_rng(181)
        no, nv = 6, 5
        left = rng.normal(size=(no, no, nv, nv, no, nv))
        right = rng.normal(size=(no, no, no, nv, nv, nv))
        cut = left[:, :, :, :, 1:5, 1:4]
        raw = np.einsum('ijabme,klmcde->ijklabcd', cut, right[:, :, 1:5, :, :, 1:4], optimize=True)
        maps = [list(range(12)) for _ in range(3)]
        maps[1][4:8], maps[2][4:8] = [5, 6, 4, 7], [4, 5, 7, 6]
        starts, origin = [0]*12, [0]*12
        starts[8] = starts[10] = origin[8] = origin[10] = 1
        counts = [no]*4 + [nv]*4 + [4, 0, 3, 0]
        for tile in (2, nv):
            caps = [1]*12
            for label in (4, 5, 6, 7, 8, 10):
                caps[label] = tile
            self.run_case(4, 4, (no, 0, nv, 0), cut, _dense_meta(cut),
                          _factor([0, 1, 4, 5, 8, 10]), origin,
                          right, _dense_meta(right), _factor([2, 3, 8, 6, 7, 10]), [0]*12,
                          [0, 1, 4, 5], [2, 3, 6, 7], [8, 10], starts, counts, caps,
                          maps, [.25, -.75, 1.5], raw, capacity=64 if tile == 2 else 4096)

    def test_mixed_spin_native_layout_and_full_virtual_fast_scatter(self):
        rng = np.random.default_rng(781)
        oa, ob, va, vb = 3, 3, 4, 3
        left = np.asfortranarray(rng.normal(size=(oa, oa, va, va, oa, va)))
        right = rng.normal(size=(ob, ob, vb, vb, oa, va))[:, :, ::-1]
        raw = np.einsum('ijabme,klcdme->ijklabcd', left, right, optimize=True)
        maps = [list(range(12)) for _ in range(4)]
        maps[1][4:6], maps[2][6:8] = [5, 4], [7, 6]
        maps[3][4:8] = [5, 4, 7, 6]
        counts = [oa, oa, ob, ob, va, va, vb, vb, oa, 0, va, 0]
        caps = [max(1, c) for c in counts]
        self.run_case(4, 2, (oa, ob, va, vb), left, _dense_meta(left),
                      _factor([0, 1, 4, 5, 8, 10]), [0]*12,
                      right, _dense_meta(right), _factor([2, 3, 6, 7, 8, 10]), [0]*12,
                      [0, 1, 4, 5], [2, 3, 6, 7], [8, 10], [0]*12, counts, caps,
                      maps, [.25, -.5, .75, 1.25], raw)

    def test_packed_t4_feedback_all_contracted_indices_and_capacity_guard(self):
        rng = np.random.default_rng(221)
        mycc = SimpleNamespace(nocc=(5, 0), nmo=(10, 0))
        packed = np.asfortranarray(rng.normal(size=(5, 5)))
        full = storage._unpack_block(mycc, packed, 4, 0)
        g = np.asfortranarray(rng.normal(size=(10,)*4))
        gm = _dense_meta(g)
        gm.hamiltonian, gm.offset[:4] = 1, (5, 5, 5, 5)
        tm = _dense_meta(packed)
        tm.compact, tm.ngroup, tm.group_size[:2], tm.group_dim[:2] = 1, 2, (4, 4), (5, 5)
        raw = np.einsum('mnef,ijmnabef->ijab', g[:5, :5, 5:, 5:], full, optimize=True)
        maps = [list(range(12)) for _ in range(2)]
        maps[1][4:6] = [5, 4]
        counts = [5, 5, 0, 0, 5, 5, 0, 0, 5, 5, 5, 5]
        caps = [1]*12
        caps[4] = caps[5] = 5
        plan, output, arguments = self.run_case(2, 2, (5, 0, 5, 0), g, gm,
                          _factor([8, 9, 10, 11]), [0]*12, packed, tm,
                          _factor([0, 1, 8, 9, 4, 5, 10, 11]), [0]*12,
                          [], [0, 1, 4, 5], [8, 9, 10, 11], [0]*12, counts, caps,
                          maps, [.25, -.75], raw)
        saved = output.copy()
        plan.capacity = 1
        self.assertEqual(self.library.UCCSDTQfinal_contract(*arguments), -2)
        np.testing.assert_array_equal(output, saved)


if __name__ == '__main__':
    unittest.main()
