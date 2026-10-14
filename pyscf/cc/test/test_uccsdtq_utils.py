#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from itertools import product

import numpy as np

from pyscf.cc import _uccsdtq_utils as u
from pyscf import lib


class TestUCCSDTQStorage(unittest.TestCase):
    def setUp(self):
        self.cc = SimpleNamespace(nocc=(3, 2), nmo=(7, 6), cc_order=4,
                                  do_tri_max_t=True, do_diis_max_t=True,
                                  diis_space=3, chkfile=None, verbose=0,
                                  stdout=sys.stdout)

    def _antisymmetric_full(self, rank, block):
        shape = u.block_shape(self.cc, rank, block, compact=False)
        arr = np.zeros(shape)
        groups, vals, _ = u._canonical_groups(self.cc, rank, block)
        for coord in product(*vals):
            value = 0.1 + sum(x * 0.01 for group in coord for x in group)
            psets = []
            signs = []
            for x in coord:
                pp = tuple(tuple(x[j] for j in p) for p in __import__('itertools').permutations(range(len(x))))
                psets.append(pp)
                signs.append(tuple(u._perm_sign(p) for p in __import__('itertools').permutations(range(len(x)))))
            for choices in product(*[range(len(x)) for x in psets]):
                native = tuple(y for k, i in enumerate(choices) for y in psets[k][i])
                sign = np.prod([signs[k][i] for k, i in enumerate(choices)])
                arr[native] = sign * value
        return arr

    def test_t4_roundtrip_and_signs(self):
        full = tuple(self._antisymmetric_full(4, b) for b in range(5))
        packed = u.pack_t4(self.cc, full)
        self.assertEqual([x.shape for x in packed], [u.block_shape(self.cc, 4, b, True) for b in range(5)])
        restored = u.unpack_t4(self.cc, packed)
        for ref, got in zip(full, restored):
            np.testing.assert_allclose(ref, got)

    def test_rank_generic_t3_compact_input(self):
        full = tuple(self._antisymmetric_full(3, b) for b in range(4))
        packed = u.pack_blocks(self.cc, full, 3)
        self.assertEqual(tuple(x.shape for x in u.unpack_blocks(self.cc, packed, 3)),
                         tuple(x.shape for x in full))
        # UCCSDT can hand us compact T3; conversion to the DIIS vector is idempotent.
        tamps = [tuple(np.ones(u.block_shape(self.cc, 1, b, False)) for b in range(2)),
                 tuple(np.ones(u.block_shape(self.cc, 2, b, False)) for b in range(3)), packed]
        self.cc.do_diis_max_t = False
        vector = u.amplitudes_to_vector(self.cc, tamps)
        restored = u.vector_to_amplitudes(self.cc, vector)
        self.assertEqual(len(restored), 3)
        self.assertEqual([a.shape for a in restored[2]], [a.shape for a in packed])
        for a, b in zip(packed, u.pack_blocks(self.cc, restored[2], 3)):
            np.testing.assert_allclose(a, b)
        from pyscf.cc import uccsdt
        ccsdt = SimpleNamespace(nocc=self.cc.nocc, nmo=self.cc.nmo, cc_order=3)
        for a, b in zip(packed, uccsdt.tamps_full2tri_uhf(ccsdt, full)):
            np.testing.assert_allclose(a, b)

    def test_vector_with_and_without_t4(self):
        tamps = []
        for rank in range(1, 5):
            tamps.append(tuple(self._antisymmetric_full(rank, b) for b in range(rank + 1)))
        self.cc.do_diis_max_t = False
        v3 = u.amplitudes_to_vector(self.cc, tamps[:3])
        self.assertEqual(v3.size, u.vector_size(self.cc, include_t4=False))
        self.assertEqual(len(u.vector_to_amplitudes(self.cc, v3)), 3)
        self.cc.do_diis_max_t = True
        v4 = u.amplitudes_to_vector(self.cc, tamps)
        self.assertEqual(v4.size, u.vector_size(self.cc))
        out = u.vector_to_amplitudes(self.cc, v4)
        self.assertEqual(len(out), 4)
        for rank in (3, 4):
            self.assertEqual([a.shape for a in out[rank - 1]],
                             [u.block_shape(self.cc, rank, b, True) for b in range(rank + 1)])
            for a, b in zip(tamps[rank - 1], u.unpack_blocks(self.cc, out[rank - 1], rank)):
                np.testing.assert_allclose(a, b)
        self.cc.do_tri_max_t = False
        full = u.vector_to_amplitudes(self.cc, v4)
        for rank in (3, 4):
            for a, b in zip(tamps[rank - 1], full[rank - 1]):
                np.testing.assert_allclose(a, b)
        np.testing.assert_array_equal(u.amplitudes_to_vector(self.cc, full), v4)
        self.cc.do_diis_max_t = False
        np.testing.assert_array_equal(u.amplitudes_to_vector(self.cc, tamps), v4)
        self.assertEqual(len(u.vector_to_amplitudes(self.cc, v4)), 4)
        for rank in (1, 2):
            vec = u.amplitudes_to_vector(self.cc, tamps[:rank])
            self.assertEqual(len(u.vector_to_amplitudes(self.cc, vec)), rank)

    def test_empty_t4_vector_ambiguity(self):
        cc = SimpleNamespace(nocc=(1, 1), nmo=(3, 3), cc_order=4,
                             do_tri_max_t=True, do_diis_max_t=False)
        vec = np.zeros(u.vector_size(cc))
        self.assertEqual(len(u.vector_to_amplitudes(cc, vec)), 3)
        cc.do_diis_max_t = True
        self.assertEqual(len(u.vector_to_amplitudes(cc, vec)), 4)

    def test_empty_spin_blocks(self):
        cc = SimpleNamespace(nocc=(1, 4), nmo=(4, 8), cc_order=4,
                             do_tri_max_t=True, do_diis_max_t=True, diis_space=2)
        for rank in (1, 2, 3, 4):
            for block in range(rank + 1):
                shape = u.block_shape(cc, rank, block, True)
                arr = np.zeros(shape)
                self.assertEqual(arr.size, int(np.prod(shape)))
        vector = u.amplitudes_to_vector(cc, [
            tuple(np.zeros(u.block_shape(cc, 1, b, False)) for b in range(2)),
            tuple(np.zeros(u.block_shape(cc, 2, b, False)) for b in range(3)),
            tuple(np.zeros(u.block_shape(cc, 3, b, False)) for b in range(4)),
            tuple(np.zeros(u.block_shape(cc, 4, b, True)) for b in range(5)),
        ])
        self.assertEqual(vector.size, u.vector_size(cc))

    def test_memory_components_and_checkpoint(self):
        parts = u.memory_estimate(self.cc, diis_space=3, include_t4_diis=False)
        for rank in (3, 4):
            packed_size = 8 * sum(int(np.prod(u.block_shape(self.cc, rank, b, True)))
                                 for b in range(rank + 1))
            self.assertEqual(parts['T%d' % rank], packed_size)
            self.assertEqual(parts['R%d' % rank], packed_size)
        self.assertGreater(parts['T4'], 0)
        self.assertNotEqual(parts['total_components'], parts['T4'])
        self.assertEqual(parts['DIIS'], 2 * 3 * 8 * u.vector_size(self.cc, include_t4=False))
        parts4 = u.memory_estimate(self.cc, diis_space=3, include_t4_diis=True)
        self.assertEqual(parts4['DIIS'], 2 * 3 * 8 * u.vector_size(self.cc, include_t4=True))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'uccsdtq.chk')
            self.cc.chkfile = path
            self.cc.tamps = [tuple(np.zeros(u.block_shape(self.cc, r, b, True))
                                   for b in range(r + 1)) for r in range(1, 5)]
            u.dump_chk(self.cc)
            payload = u.load_chk(path)
            self.assertEqual(payload['order'], 4)
            self.assertEqual(payload['storage'], 'compact')
            self.assertEqual(len(payload['tamps']), 4)

    def test_checkpoint_converts_each_rank_from_actual_shape(self):
        full = [tuple(self._antisymmetric_full(rank, b) for b in range(rank + 1))
                for rank in range(1, 5)]
        vector = u.amplitudes_to_vector(self.cc, full)
        with tempfile.TemporaryDirectory() as td:
            self.cc.chkfile = os.path.join(td, 'uccsdtq.chk')
            # The first case matches historical compact checkpoints with
            # full T3 and compact T4. The others cover each rank separately.
            for compact_t3, compact_t4 in ((False, True), (True, False), (True, True), (False, False)):
                self.cc.do_tri_max_t = True
                self.cc.tamps = full[:2] + [
                    u.pack_blocks(self.cc, full[2], 3) if compact_t3 else full[2],
                    u.pack_t4(self.cc, full[3]) if compact_t4 else full[3]]
                u.dump_chk(self.cc)
                for compact_target in (True, False):
                    with self.subTest(source=(compact_t3, compact_t4), target=compact_target):
                        self.cc.do_tri_max_t = compact_target
                        u.load_chk(self.cc.chkfile, self.cc)
                        for rank in (3, 4):
                            self.assertEqual([a.shape for a in self.cc.tamps[rank - 1]],
                                             [u.block_shape(self.cc, rank, b, compact_target)
                                              for b in range(rank + 1)])
                        np.testing.assert_array_equal(u.amplitudes_to_vector(self.cc, self.cc.tamps), vector)

    def test_full_t3_memory_estimate(self):
        self.cc.do_tri_max_t = False
        parts = u.memory_estimate(self.cc)
        for rank in (3, 4):
            full_size = 8 * sum(int(np.prod(u.block_shape(self.cc, rank, b, False)))
                               for b in range(rank + 1))
            self.assertEqual(parts['T%d' % rank], full_size)
            self.assertEqual(parts['R%d' % rank], full_size)

    def test_packed_index_lexicographic(self):
        from itertools import combinations
        for n in range(1, 7):
            for rank in range(1, min(n, 4) + 1):
                for i, coords in enumerate(combinations(range(n), rank)):
                    self.assertEqual(u.packed_index(coords, n), i)
        for coords in ((0, 0), (2, 1), (-1, 2), (0, 4)):
            with self.assertRaises(ValueError):
                u.packed_index(coords, 4)

    def test_all_t4_blocks_antisymmetry(self):
        cc = SimpleNamespace(nocc=(4, 4), nmo=(8, 8))
        rng = np.random.default_rng(24)
        packed = tuple(rng.normal(size=u.block_shape(cc, 4, b, True)) for b in range(5))
        full = u.unpack_t4(cc, packed)
        for b, tensor in enumerate(full):
            groups, _, _ = u._canonical_groups(cc, 4, b)
            for group in groups:
                for a, c in zip(group, group[1:]):
                    np.testing.assert_array_equal(tensor, -tensor.swapaxes(a, c))
                    np.testing.assert_array_equal(np.diagonal(tensor, axis1=a, axis2=c), 0)
        for p, r in zip(packed, u.pack_t4(cc, full)):
            np.testing.assert_array_equal(p, r)

    def test_diis_restart_preserves_or_initializes_t4(self):
        self.cc.do_diis_max_t = False
        self.cc.incore_complete = True
        vec = np.linspace(0.1, 0.9, u.vector_size(self.cc, include_t4=False))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'diis.h5')
            adiis = lib.diis.DIIS(filename=path)
            adiis.update(vec, np.ones(vec.size))
            self.cc.tamps = [None] * 4
            u.restore_from_diis_(self.cc, path, inplace=False)
            np.testing.assert_allclose(u.amplitudes_to_vector(self.cc, self.cc.tamps[:3]), vec)
            self.assertEqual(sum(np.linalg.norm(x) for x in self.cc.tamps[3]), 0)
            t4 = self.cc.tamps[3]
            t4[1][...] = 0.3
            u.restore_from_diis_(self.cc, path, inplace=False)
            self.assertIs(self.cc.tamps[3], t4)

    def test_full_expansion_memory_guard(self):
        packed = tuple(np.zeros(u.block_shape(self.cc, 4, b, True)) for b in range(5))
        self.cc.max_memory = 0
        with self.assertRaises(MemoryError):
            u.unpack_t4(self.cc, packed)

    def test_spin_flip_unequal_spaces(self):
        cc = SimpleNamespace(nocc=(4, 3), nmo=(8, 8))
        flipped = SimpleNamespace(nocc=cc.nocc[::-1], nmo=cc.nmo[::-1])
        rng = np.random.default_rng(25)
        packed = tuple(rng.normal(size=u.block_shape(cc, 4, b, True)) for b in range(5))
        swapped = (packed[4], packed[3], packed[2].transpose(2, 3, 0, 1), packed[1], packed[0])
        full = u.unpack_t4(cc, packed)
        swapped_full = u.unpack_t4(flipped, swapped)
        expected = (full[4], full[3], full[2].transpose(4, 5, 6, 7, 0, 1, 2, 3), full[1], full[0])
        for a, b in zip(expected, swapped_full):
            np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
