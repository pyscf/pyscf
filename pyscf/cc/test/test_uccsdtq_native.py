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

"""Native CCSDTQ contraction, packing, stride and OpenMP checks."""
from math import factorial
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from pyscf import lib
from pyscf.cc import _uccsdtq_native as native
from pyscf.cc import _uccsdtq_terms as terms
from pyscf.cc import _uccsdtq_utils as storage


def _blocks(tensor, rank, dims):
    oa, ob, va, vb = dims
    result = []
    for block in range(rank+1):
        na, nb = rank-block, block
        majority = 0 if na >= nb else 1
        hs = (range(oa), range(oa, oa+ob))
        vs = (range(va), range(va, va+vb))
        spin_order = [majority]*max(na, nb) + [1-majority]*min(na, nb)
        full = tensor[np.ix_(*([hs[s] for s in spin_order] + [vs[s] for s in spin_order]))]
        n = max(na, nb)
        order = list(range(n)) + list(range(rank, rank+n))
        order += list(range(n, rank)) + list(range(rank+n, 2*rank))
        result.append(np.asarray(full.transpose(order), order='C'))
    return tuple(result)


def _case(dims, seed=945):
    oa, ob, va, vb = dims
    no, nv = oa+ob, va+vb
    spin = np.array([0]*oa + [1]*ob + [0]*va + [1]*vb)
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(no+nv,)*2)*.1
    f *= spin[:, None] == spin[None, :]
    g = rng.normal(size=(no+nv,)*4)*.1
    g = g - g.swapaxes(0, 1)
    g = g - g.swapaxes(2, 3)
    sums = spin[:, None] + spin[None, :]
    g *= sums[:, :, None, None] == sums[None, None, :, :]
    amps = []
    for rank in (2, 3, 4):
        shape = (no,)*rank + (nv,)*rank
        a = terms.antisymmetrize(rng.normal(size=shape)*.1, rank)/factorial(rank)**2
        grid = np.indices(shape)
        hspin = sum(spin[grid[x]] for x in range(rank))
        pspin = sum(spin[no+grid[x]] for x in range(rank, 2*rank))
        a *= hspin == pspin
        amps.append(a)
    alpha = np.r_[np.arange(oa), no+np.arange(va)]
    beta = np.r_[oa+np.arange(ob), no+va+np.arange(vb)]
    fblocks = [f[np.ix_(s, s)] for s in (alpha, beta)]
    gblocks = [g[np.ix_(alpha, alpha, alpha, alpha)],
               g[np.ix_(alpha, beta, alpha, beta)],
               g[np.ix_(beta, beta, beta, beta)]]
    imds = SimpleNamespace(t1_fock=fblocks, t1_eris=gblocks)
    mycc = SimpleNamespace(nocc=(oa, ob), nmo=(oa+va, ob+vb), do_tri_max_t=True)
    native_amps = tuple(_blocks(a, r, dims) for r, a in zip((2, 3, 4), amps))
    reference = tuple(_blocks(a, r, dims) for r, a in zip((2, 3, 4), terms.residuals(f, g, *amps)))
    return mycc, imds, native_amps, reference


class KnownValues(unittest.TestCase):
    def setUp(self):
        self.threads = lib.num_threads()
        lib.num_threads(1)

    def tearDown(self):
        lib.num_threads(self.threads)

    def assert_blocks_close(self, actual, expected):
        for a, b in zip(actual, expected):
            for x, y in zip(a, b):
                np.testing.assert_allclose(x, y, atol=2e-12, rtol=2e-12)

    def test_all_spin_blocks_full_compact_and_threads(self):
        for dims in ((4,0,4,0), (3,1,3,1), (2,2,2,2), (1,3,1,3), (0,4,0,4), (3,2,2,2), (5,0,5,0), (4,1,4,1)):
            with self.subTest(dims=dims):
                mycc, imds, amps, ref = _case(dims)
                original = tuple(tuple(a.copy() for a in blocks) for blocks in amps)
                for compact in (False, True):
                    mycc.do_tri_max_t = compact
                    t4 = storage.pack_t4(mycc, amps[2]) if compact else amps[2]
                    results = []
                    for threads in (1, 4):
                        lib.num_threads(threads)
                        result = native.residuals(mycc, imds, amps[0], amps[1], t4)
                        full = result[:2] + (storage.unpack_t4(mycc, result[2]),) if compact else result
                        self.assert_blocks_close(full, ref)
                        results.append(result)
                    self.assert_blocks_close(results[0], results[1])
                for blocks, saved in zip(amps, original):
                    for a, b in zip(blocks, saved):
                        np.testing.assert_array_equal(a, b)

    def test_strided_inputs_outputs_and_accumulation(self):
        mycc, imds, amps, ref = _case((3,2,2,2), seed=288)
        imds.t1_fock = tuple(np.asfortranarray(a) for a in imds.t1_fock)
        imds.t1_eris = tuple(np.asfortranarray(a) for a in imds.t1_eris)
        t2, t3 = tuple(tuple(np.asfortranarray(a) for a in blocks) for blocks in amps[:2])
        for compact in (False, True):
            mycc.do_tri_max_t = compact
            t4 = storage.pack_t4(mycc, amps[2]) if compact else amps[2]
            t4 = tuple(np.asfortranarray(a) for a in t4)
            expected = ref[:2] + (storage.pack_t4(mycc, ref[2]),) if compact else ref
            out = tuple(tuple(np.full(a.shape, .375, order='F') for a in blocks) for blocks in expected)
            returned = native.residuals(mycc, imds, t2, t3, t4, out=out)
            self.assertIs(returned, out)
            for blocks, wanted in zip(out, expected):
                for a, b in zip(blocks, wanted):
                    np.testing.assert_allclose(a, b+.375, atol=2e-12, rtol=2e-12)

    def test_invalid_arrays_rejected_before_c(self):
        mycc, imds, amps, _ = _case((2,2,2,2))
        t4 = storage.pack_t4(mycc, amps[2])
        with self.assertRaises(ValueError):
            native.residuals(mycc, imds, amps[0][:-1], amps[1], t4)
        bad = list(amps[0])
        bad[0] = bad[0][:1]
        with self.assertRaises(ValueError):
            native.residuals(mycc, imds, bad, amps[1], t4)
        bad = list(amps[0])
        bad[0] = bad[0].astype(complex)
        with self.assertRaises(ValueError):
            native.residuals(mycc, imds, bad, amps[1], t4)

    def test_stale_library_rejected(self):
        native._native_kernel.cache_clear()
        try:
            with patch.object(native.lib, 'load_library', return_value=SimpleNamespace()):
                with self.assertRaisesRegex(RuntimeError, 'rebuild libccsdt'):
                    native._native_kernel()
        finally:
            native._native_kernel.cache_clear()


if __name__ == '__main__':
    unittest.main()
