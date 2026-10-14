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

"""Packed T3 reads and residuals against independent full-tensor equations."""

import unittest
from unittest.mock import patch

import numpy as np
from pyscf import lib
from pyscf.cc import _uccsdtq_blocked as blocked
from pyscf.cc import _uccsdtq_native as direct
from pyscf.cc import _uccsdtq_utils as storage
from pyscf.cc.test.test_uccsdtq_native import _case


def _strided(array):
    shape = array.shape[:-1] + (2 * array.shape[-1],)
    result = np.empty(shape)[..., ::2][::-1]
    result[...] = array
    return result


class KnownValues(unittest.TestCase):
    def setUp(self):
        self.threads = lib.num_threads()
        lib.num_threads(1)

    def tearDown(self):
        lib.num_threads(self.threads)

    def assert_blocks_close(self, actual, expected):
        for blocks, refs in zip(actual, expected):
            for array, ref in zip(blocks, refs):
                np.testing.assert_allclose(array, ref, atol=3e-12, rtol=3e-12)

    def test_packed_t3_all_spin_sectors_and_t4_layouts(self):
        for dims in ((4, 0, 4, 0), (3, 1, 3, 1), (2, 2, 2, 2),
                     (1, 3, 1, 3), (0, 4, 0, 4), (3, 2, 2, 3)):
            mycc, imds, amps, reference = _case(dims, seed=443)
            mycc.max_memory = 2000
            t3 = storage.pack_blocks(mycc, amps[1], 3)
            saved = tuple(a.copy() for a in t3)
            for compact_t4 in (True, False):
                mycc.do_tri_max_t = compact_t4
                t4 = storage.pack_t4(mycc, amps[2]) if compact_t4 else amps[2]
                expected = (reference[0], storage.pack_blocks(mycc, reference[1], 3),
                            storage.pack_t4(mycc, reference[2]) if compact_t4 else reference[2])
                for backend in (direct, blocked):
                    with self.subTest(dims=dims, compact_t4=compact_t4, backend=backend.__name__):
                        actual = backend.residuals(mycc, imds, amps[0], t3, t4)
                        self.assertEqual([a.shape for a in actual[1]], [a.shape for a in t3])
                        self.assert_blocks_close(actual, expected)
            for array, original in zip(t3, saved):
                np.testing.assert_array_equal(array, original)

    def test_strided_packed_inputs_and_accumulating_outputs(self):
        mycc, imds, amps, reference = _case((3, 2, 3, 2), seed=445)
        mycc.max_memory = 2000
        imds.t1_fock = tuple(_strided(a) for a in imds.t1_fock)
        imds.t1_eris = tuple(np.asfortranarray(a) for a in imds.t1_eris)
        t2 = tuple(np.asfortranarray(a) for a in amps[0])
        t3 = tuple(_strided(a) for a in storage.pack_blocks(mycc, amps[1], 3))
        t4 = tuple(_strided(a) for a in storage.pack_t4(mycc, amps[2]))
        expected = (reference[0], storage.pack_blocks(mycc, reference[1], 3),
                    storage.pack_t4(mycc, reference[2]))
        for backend in (direct, blocked):
            for threads in (1, 4):
                with self.subTest(backend=backend.__name__, threads=threads):
                    lib.num_threads(threads)
                    out = tuple(tuple(_strided(np.full(a.shape, .375)) for a in blocks)
                                for blocks in expected)
                    returned = backend.residuals(mycc, imds, t2, t3, t4, out=out)
                    self.assertIs(returned, out)
                    self.assert_blocks_close(out, tuple(tuple(a + .375 for a in blocks)
                                                        for blocks in expected))
        invalid = (tuple(np.zeros_like(a) for a in reference[0]),
                   tuple(np.zeros_like(a) for a in reference[1]),
                   tuple(np.zeros_like(a) for a in expected[2]))
        for backend in (direct, blocked):
            with self.assertRaises(ValueError):
                backend.residuals(mycc, imds, t2, t3, t4, out=invalid)

    def test_packed_t3_small_blocked_workspace(self):
        dims = (2, 2, 4, 4)
        mycc, imds, amps, _ = _case(dims, seed=447)
        t3 = storage.pack_blocks(mycc, amps[1], 3)
        t4 = storage.pack_t4(mycc, amps[2])
        expected = direct.residuals(mycc, imds, amps[0], t3, t4)
        blocked._plans(dims, None)
        resident = lib.current_memory()[0]
        for budget in (2, .06):
            with self.subTest(workspace_mb=budget):
                mycc.max_memory = resident + budget
                with patch.object(blocked.lib, 'current_memory', return_value=(resident, 0)):
                    actual = blocked.residuals(mycc, imds, amps[0], t3, t4)
                self.assert_blocks_close(actual, expected)
                stats = mycc._uccsdtq_stats
                self.assertLessEqual(stats['max_panel_elements'] * 8, stats['panel_limit_bytes'])
                self.assertLessEqual(stats['max_cache_bytes'], stats['cache_limit_bytes'])


if __name__ == '__main__':
    unittest.main()
