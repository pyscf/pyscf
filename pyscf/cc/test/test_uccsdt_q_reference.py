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

"""Independent operator and contraction checks for the small-system oracles."""

from itertools import combinations, permutations
from types import SimpleNamespace
import unittest

import numpy as np

from pyscf.cc.test import uccsdt_q_reference as reference
from pyscf.cc.test import uccsdt_q_slow as slow


def random_inputs(nalpha, seed=21):
    """Four electrons in eight spin orbitals, with one quadruple per sector."""
    rng = np.random.default_rng(seed)
    spin = np.array(([0] * nalpha + [1] * (4 - nalpha)) * 2)
    factors = rng.normal(size=(5, 8, 8))
    factors += factors.transpose(0, 2, 1)
    factors *= spin[:, None] == spin[None, :]
    bare = np.einsum('xpr,xqs->pqrs', factors, factors)
    v = bare - bare.transpose(0, 1, 3, 2)
    amplitudes = []
    for rank in (2, 3):
        tensor = np.zeros((4,) * (2 * rank))
        for holes in combinations(range(4), rank):
            for particles in combinations(range(4), rank):
                if sum(spin[list(holes)]) != sum(spin[4 + np.array(particles)]):
                    continue
                value = rng.normal() * 0.02
                for hp in permutations(range(rank)):
                    for pp in permutations(range(rank)):
                        idx = tuple(holes[i] for i in hp) + tuple(particles[a] for a in pp)
                        tensor[idx] = reference._parity(hp) * reference._parity(pp) * value
        amplitudes.append(tensor)
    eps = np.r_[-np.arange(1., 5), np.arange(1., 5)]
    return v, amplitudes[0], amplitudes[1], eps, spin, bare


class KnownValues(unittest.TestCase):
    def test_fermion_anticommutators(self):
        determinants = list(range(16))
        annihilators = [reference._operator_matrix([(1., (i,), ())], determinants).toarray()
                        for i in range(4)]
        for i, ai in enumerate(annihilators):
            for j, aj in enumerate(annihilators):
                expected = np.eye(16) if i == j else np.zeros((16, 16))
                np.testing.assert_array_equal(ai @ aj.T + aj.T @ ai, expected)
                np.testing.assert_array_equal(ai @ aj + aj @ ai, np.zeros((16, 16)))

    def test_five_spin_sectors(self):
        for nalpha in range(5):
            with self.subTest(nalpha=nalpha):
                v, t2, t3, eps, spin, _ = random_inputs(nalpha)
                bracket, paren, details = reference.determinant_kernel(
                    v, t2, t3, eps, spin, 4, True)
                sb, sp, terms = slow.kernel(v, t2, t3, eps, True)
                np.testing.assert_allclose([bracket, paren], [sb, sp], atol=1e-11, rtol=1e-10)
                self.assertAlmostEqual(details['reference_vn'], 0., 11)
                qidx = np.flatnonzero(details['ranks'] == 4)[0]
                _, sign = reference._apply(15, (0, 1, 2, 3), (4, 5, 6, 7))
                idx = (0, 1, 2, 3) * 2
                for key, expected in (
                        ('w_t3', terms['vvov'][idx] + terms['vooo'][idx]),
                        ('w_t2t2', sum(terms[k][idx] for k in ('voov', 'oooo', 'vvvv'))),
                        ('left2', terms['l2'][idx]), ('left3', terms['l3'][idx])):
                    self.assertAlmostEqual(details[key][qidx] / sign, expected, 11)

    def test_each_contraction_class(self):
        v, t2, t3, eps, spin, _ = random_inputs(2, seed=52)
        indices = np.indices(v.shape)
        nvir_first = (indices[0] >= 4).astype(int) + (indices[1] >= 4)
        nvir_second = (indices[2] >= 4).astype(int) + (indices[3] >= 4)
        nvir = nvir_first + nvir_second
        masks = dict(oooo=nvir == 0, vooo=nvir == 1,
                     voov=(nvir_first == 1) & (nvir_second == 1),
                     vvov=nvir == 3, vvvv=nvir == 4,
                     l2=(nvir == 2) & (nvir_first != 1))
        idx = (0, 1, 2, 3) * 2
        for key, mask in masks.items():
            with self.subTest(contraction=key):
                v_block = v * mask
                _, _, details = reference.determinant_kernel(
                    v_block, t2, t3, eps, spin, 4, True)
                qidx = np.flatnonzero(details['ranks'] == 4)[0]
                _, sign = reference._apply(15, (0, 1, 2, 3), (4, 5, 6, 7))
                terms = slow.terms(v_block, t2, t3)
                if key == 'l2':
                    actual = details['left2'][qidx] / sign
                elif key in ('vvov', 'vooo'):
                    actual = details['w_t3'][qidx] / sign
                else:
                    actual = details['w_t2t2'][qidx] / sign
                self.assertAlmostEqual(actual, terms[key][idx], 11)
                self.assertGreater(abs(actual), 1e-8)
                for other, tensor in terms.items():
                    if other != key:
                        np.testing.assert_allclose(tensor, 0., atol=1e-13)

    def test_native_full_and_compact_conversion(self):
        for nalpha in range(5):
            with self.subTest(nalpha=nalpha):
                v, t2, t3, eps, spin, bare = random_inputs(nalpha)
                a, b = np.arange(nalpha), np.arange(nalpha, 4)
                alpha, beta = np.r_[a, a + 4], np.r_[b, b + 4]
                mycc = SimpleNamespace(nocc=(len(a), len(b)), nmo=(2 * len(a), 2 * len(b)))
                eris = SimpleNamespace(
                    pppp=bare[np.ix_(alpha, alpha, alpha, alpha)],
                    PPPP=bare[np.ix_(beta, beta, beta, beta)],
                    pPpP=bare[np.ix_(alpha, beta, alpha, beta)],
                    mo_energy=(eps[alpha], eps[beta]))
                t1_native = (np.zeros((len(a), len(a))), np.zeros((len(b), len(b))))
                t2_native = (t2[np.ix_(a, a, a, a)],
                             t2[np.ix_(a, b, a, b)].transpose(0, 2, 1, 3),
                             t2[np.ix_(b, b, b, b)])
                t3_native = (t3[np.ix_(a, a, a, a, a, a)],
                             t3[np.ix_(a, a, b, a, a, b)].transpose(0, 1, 3, 4, 2, 5),
                             t3[np.ix_(b, b, a, b, b, a)].transpose(0, 1, 3, 4, 2, 5),
                             t3[np.ix_(b, b, b, b, b, b)])
                compact = []
                for i, block in enumerate(t3_native):
                    rank = 3 if i in (0, 3) else 2
                    holes = list(combinations(range(block.shape[0]), rank))
                    particles = list(combinations(range(block.shape[rank]), rank))
                    tail = () if rank == 3 else block.shape[4:]
                    packed = np.zeros((len(holes), len(particles)) + tail)
                    for ih, hs in enumerate(holes):
                        for ip, ps in enumerate(particles):
                            packed[ih, ip] = block[hs + ps]
                    compact.append(packed)
                for native in (t3_native, compact):
                    inputs = reference.spinorbital_inputs(
                        mycc, eris, (t1_native, t2_native, native))
                    np.testing.assert_array_equal(inputs['v'], v)
                    np.testing.assert_array_equal(inputs['t2'], t2)
                    np.testing.assert_array_equal(inputs['t3'], t3)
                    np.testing.assert_array_equal(inputs['mo_energy'], eps)
                    np.testing.assert_array_equal(inputs['orbspin'], spin)


if __name__ == '__main__':
    unittest.main()
