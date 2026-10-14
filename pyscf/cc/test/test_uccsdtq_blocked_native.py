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

"""Bounded native gather/GEMM/scatter primitives for CCSDTQ."""
import ctypes
from types import SimpleNamespace
import unittest

import numpy as np
from pyscf import lib
from pyscf.cc import _uccsdtq_native as native
from pyscf.cc import _uccsdtq_utils as storage
from pyscf.cc.test.test_uccsdtq_native import _case


def _ptr(array):
    return array.ctypes.data_as(ctypes.c_void_p)


def _i32(values):
    return np.asarray(values, dtype=np.int32)


def _factor(labels):
    result = native._Factor()
    result.ndim = len(labels)
    result.indices[:len(labels)] = labels
    return result


def _dense_meta(array):
    result = native._Array()
    result.ndim = array.ndim
    result.shape[:array.ndim] = array.shape
    result.stride[:array.ndim] = tuple(s // 8 for s in array.strides)
    return result


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = lib.load_library('libccsdt')
        assert cls.library.UCCSDTQblocked_abi_version() == 1
        p, i, ll, d = ctypes.c_void_p, ctypes.c_int32, ctypes.c_int64, ctypes.c_double
        cls.library.UCCSDTQgather.argtypes = [p, p, p, i, p, p, p, p, p]
        cls.library.UCCSDTQgemm.argtypes = [ll, ll, ll, d, p, p, d, p]
        cls.library.UCCSDTQscatter_tile.argtypes = [p, i, p, p, p, p, i, i, i, i, i, i, i, d, p, p]
        cls.mycc, cls.imds, cls.amps, _ = _case((3, 2, 3, 2), seed=927)

    def setUp(self):
        self.threads = lib.num_threads()
        lib.num_threads(1)

    def tearDown(self):
        lib.num_threads(self.threads)

    def gather(self, array, meta, factor, labels, starts, counts, fixed):
        labels, starts, counts, fixed = map(_i32, (labels, starts, counts, fixed))
        output = np.empty(tuple(counts))
        status = self.library.UCCSDTQgather(_ptr(array), ctypes.byref(meta), ctypes.byref(factor),
                    len(labels), _ptr(labels), _ptr(starts), _ptr(counts), _ptr(fixed), _ptr(output))
        self.assertEqual(status, 0)
        return output

    def test_thread_control_reaches_native_runtime(self):
        for count in (1, 4, 2, 1):
            lib.num_threads(count)
            self.assertEqual(self.library.UCCSDTQomp_threads(), lib.num_threads())

    def test_full_and_compact_spin_block_gather(self):
        for compact in (False, True):
            t4 = storage.pack_t4(self.mycc, self.amps[2]) if compact else self.amps[2]
            arrays = list(self.imds.t1_fock) + list(self.imds.t1_eris)
            arrays += list(self.amps[0]) + list(self.amps[1]) + list(t4)
            arrays = [np.asfortranarray(a) for a in arrays]
            metas = native._metadata(self.mycc, arrays)
            for aid in range(5, 17):
                rank, start = (2, 5) if aid < 8 else (3, 8) if aid < 12 else (4, 12)
                block = aid-start
                n = max(rank-block, block)
                full = self.amps[rank-2][block]
                if not full.size:
                    continue
                labels = list(range(n)) + list(range(4, 4+n))
                labels += list(range(n, rank)) + list(range(4+n, 4+rank))
                output = self.gather(arrays[aid], metas[aid], _factor(labels), labels[::-1],
                                    [0]*(2*rank), full.shape[::-1], [-1]*12)
                expected = full.transpose(tuple(range(full.ndim-1, -1, -1)))
                np.testing.assert_allclose(output, expected, atol=1e-15, rtol=1e-14)

    def test_fixed_labels_starts_negative_strides_and_hamiltonian_offsets(self):
        rng = np.random.default_rng(544)
        dense = rng.normal(size=(2, 3, 2, 3, 2, 4))[:, ::-1]
        factor_labels = [0, 1, 8, 4, 5, 10]
        labels, starts, counts = [10, 0, 8, 4], [1, 0, 0, 1], [2, 2, 2, 2]
        fixed = [-1]*12
        fixed[1], fixed[5] = 1, 0
        output = self.gather(dense, _dense_meta(dense), _factor(factor_labels),
                             labels, starts, counts, fixed)
        expected = np.empty(tuple(counts))
        for ix in np.ndindex(expected.shape):
            values = fixed.copy()
            for label, start, value in zip(labels, starts, ix):
                values[label] = start+value
            expected[ix] = dense[tuple(values[label] for label in factor_labels)]
        np.testing.assert_array_equal(output, expected)
        gab = np.asfortranarray(self.imds.t1_eris[1])
        meta = _dense_meta(gab)
        meta.hamiltonian = 1
        meta.offset[:4] = (3, 2, 3, 2)
        output = self.gather(gab, meta, _factor([0, 8, 4, 10]), [10, 4, 8, 0],
                             [0]*4, [2, 3, 2, 3], [-1]*12)
        np.testing.assert_array_equal(output, gab[:3, :2, 3:, 2:].transpose(3, 2, 1, 0))

    def test_dense_gather_long_rows_strides_and_threads(self):
        dense = np.arange(17*130*19, dtype=float).reshape(17, 130, 19)
        labels, starts, counts = [0, 4, 10], [1, 3, 2], [15, 123, 15]
        for threads in (1, 4):
            lib.num_threads(threads)
            for array in (dense, np.asfortranarray(dense), dense[::-1, ::-1, ::-1]):
                expected = array[1:16, 3:126, 2:17]
                for order in ((0, 1, 2), (2, 0, 1), (1, 2, 0)):
                    output = self.gather(array, _dense_meta(array), _factor(labels),
                                         [labels[i] for i in order],
                                         [starts[i] for i in order],
                                         [counts[i] for i in order], [-1]*12)
                    np.testing.assert_array_equal(output, expected.transpose(order))
            fixed = self.gather(dense, _dense_meta(dense), _factor(labels),
                                labels, starts, [1, 1, 1], [-1]*12)
            np.testing.assert_array_equal(fixed, dense[1:2, 3:4, 2:3])
            vector = np.arange(8193., dtype=float)[::-1]
            gathered = self.gather(vector, _dense_meta(vector), _factor([10]),
                                   [10], [0], [vector.size], [-1]*12)
            np.testing.assert_array_equal(gathered, vector)

    def test_parallel_gather_nonzero_rank4_combination_indices(self):
        mycc = SimpleNamespace(nocc=(5, 0), nmo=(10, 0))
        rng = np.random.default_rng(443)
        packed = rng.normal(size=(5, 5))
        full = storage._unpack_block(mycc, packed, 4, 0)
        meta = _dense_meta(packed)
        meta.compact, meta.ngroup = 1, 2
        meta.group_size[:2], meta.group_dim[:2] = (4, 4), (5, 5)
        factor = _factor(list(range(8)))
        for threads in (1, 4):
            lib.num_threads(threads)
            output = self.gather(packed, meta, factor, list(range(8)), [0]*8, [5]*8, [-1]*12)
            np.testing.assert_array_equal(output, full)
            labels, starts, counts, fixed = map(_i32, (list(range(8)), [0]*8, [5]*8, [-1]*12))
            for compact, expected in ((False, full), (True, packed)):
                scattered = np.zeros(expected.shape, order='F')
                strides = np.asarray(scattered.strides, dtype=np.int64)//8
                status = self.library.UCCSDTQscatter_tile(_ptr(full), 8, _ptr(labels), _ptr(starts),
                            _ptr(counts), _ptr(fixed), 4, 4, 5, 0, 5, 0,
                            compact, 1., _ptr(scattered), _ptr(strides))
                self.assertEqual(status, 0)
                np.testing.assert_array_equal(scattered, expected)

    def test_unit_axes_and_fully_fixed_packed_groups(self):
        mycc = SimpleNamespace(nocc=(5, 0), nmo=(10, 0))
        packed = np.random.default_rng(803).normal(size=(5, 5))
        full = storage._unpack_block(mycc, packed, 4, 0)
        meta = _dense_meta(packed)
        meta.compact, meta.ngroup = 1, 2
        meta.group_size[:2], meta.group_dim[:2] = (4, 4), (5, 5)
        for holes in ((0, 1, 2, 4), (1, 0, 2, 4), (1, 1, 2, 4)):
            output = self.gather(packed, meta, _factor(list(range(8))), list(range(8)),
                                 list(holes)+[0]*4, [1]*4+[5]*4, [-1]*12)
            np.testing.assert_array_equal(output.reshape((5,)*4), full[holes])
            labels, starts, counts = map(_i32, ([4, 5, 6, 7], [0]*4, [5]*4))
            fixed = _i32(list(holes)+[-1]*8)
            tile = np.ascontiguousarray(full[holes])
            wanted = np.zeros_like(packed)
            if holes == (0, 1, 2, 4):
                wanted[1] = packed[1]
            for compact in (False, True):
                expected = wanted if compact else storage._unpack_block(mycc, wanted, 4, 0)
                result = np.zeros_like(expected, order='F')
                strides = np.asarray(result.strides, dtype=np.int64)//8
                status = self.library.UCCSDTQscatter_tile(_ptr(tile), 4, _ptr(labels), _ptr(starts),
                            _ptr(counts), _ptr(fixed), 4, 4, 5, 0, 5, 0,
                            compact, 1., _ptr(result), _ptr(strides))
                self.assertEqual(status, 0)
                np.testing.assert_array_equal(result, expected)

    def test_scatter_all_spins_and_storage_with_fixed_occupied_batches(self):
        for rank in (2, 3, 4):
            for block in range(rank+1):
                na, nb = rank-block, block
                full = self.amps[rank-2][block]
                if not full.size:
                    continue
                if na >= nb:
                    order = list(range(na)) + list(range(2*na, 2*na+nb))
                    order += list(range(na, 2*na)) + list(range(2*na+nb, 2*rank))
                else:
                    order = list(range(2*nb, 2*nb+na)) + list(range(nb))
                    order += list(range(2*nb+na, 2*rank)) + list(range(nb, 2*nb))
                tensor = np.ascontiguousarray(full.transpose(order))
                for compact in (False, True):
                    expected = storage._pack_block(self.mycc, full, rank, block) if compact else full
                    output = np.full(expected.shape, .123, order='F')
                    strides = np.asarray(output.strides, dtype=np.int64)//8
                    for first in range(tensor.shape[0]):
                        tile = np.ascontiguousarray(tensor[first])
                        labels = _i32(list(range(1, rank)) + list(range(4, 4+rank)))
                        starts, counts = _i32([0]*len(labels)), _i32(tile.shape)
                        fixed = _i32([-1]*12)
                        fixed[0] = first
                        status = self.library.UCCSDTQscatter_tile(_ptr(tile), len(labels), _ptr(labels),
                            _ptr(starts), _ptr(counts), _ptr(fixed), rank, na, 3, 2, 3, 2,
                            compact, .25, _ptr(output), _ptr(strides))
                        self.assertEqual(status, 0)
                    np.testing.assert_allclose(output, .123+.25*expected, atol=1e-15, rtol=1e-14)

    def test_row_major_gemm_scaling_empty_and_threads(self):
        rng = np.random.default_rng(19)
        for threads in (1, 4):
            lib.num_threads(threads)
            for m, n, k in ((1, 2, 3), (11, 17, 29), (160, 127, 257), (128, 97, 129), (129, 97, 129), (512, 257, 129), (11, 17, 0), (0, 11, 3)):
                a, b, c = rng.normal(size=(m, k)), rng.normal(size=(k, n)), rng.normal(size=(m, n))
                expected = 1.75*(a@b)-.5*c
                status = self.library.UCCSDTQgemm(m, n, k, 1.75, _ptr(a), _ptr(b), -.5, _ptr(c))
                self.assertEqual(status, 0)
                np.testing.assert_allclose(c, expected, atol=1e-12, rtol=1e-12)

    def test_reject_invalid_bounds_before_output_write(self):
        array = np.arange(8., dtype=float).reshape(2, 4)
        output = np.full((2, 4), 12.5)
        meta, factor = _dense_meta(array), _factor([0, 4])
        labels, starts, counts, fixed = map(_i32, ([0, 4], [0, 0], [3, 4], [-1]*12))
        status = self.library.UCCSDTQgather(_ptr(array), ctypes.byref(meta), ctypes.byref(factor),
                    2, _ptr(labels), _ptr(starts), _ptr(counts), _ptr(fixed), _ptr(output))
        self.assertEqual(status, -1)
        np.testing.assert_array_equal(output, 12.5)
        self.assertEqual(self.library.UCCSDTQgemm(2**31, 1, 1, 1., _ptr(array), _ptr(array), 0., _ptr(output)), -1)
        np.testing.assert_array_equal(output, 12.5)


if __name__ == '__main__':
    unittest.main()
