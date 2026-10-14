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

from itertools import combinations, permutations, product
from math import factorial
from types import SimpleNamespace
import unittest

import numpy as np
from scipy import linalg

from pyscf.cc.test import uccsdtq_reference as reference


def _spin_conserving_hamiltonian(nocc=4, nmo=8, seed=41):
    """Small real UHF-like f/v pair for the determinant BCH oracle."""
    rng = np.random.default_rng(seed)
    # Occupied alpha,beta orbitals precede virtual alpha,beta orbitals.
    spin = np.array(([0] * (nocc // 2) + [1] * (nocc - nocc // 2) +
                     [0] * ((nmo - nocc) // 2) +
                     [1] * (nmo - nocc - (nmo - nocc) // 2)))
    f = np.diag(np.linspace(-1.8, 1.4, nmo))
    x = rng.normal(scale=.06, size=(nmo, nmo))
    f += (x + x.T) * (spin[:, None] == spin[None, :])
    bare = rng.normal(scale=.08, size=(nmo, nmo, nmo, nmo))
    mask = ((spin[:, None, None, None] == spin[None, None, :, None]) &
            (spin[None, :, None, None] == spin[None, None, None, :]))
    bare *= mask
    v = bare - bare.transpose(0, 1, 3, 2)
    v = v - v.transpose(1, 0, 2, 3)
    v = .5 * (v + v.transpose(2, 3, 0, 1))
    return f, v, spin


def _random_full_tamps(nocc, nvir, seed=7):
    rng = np.random.default_rng(seed)
    tamps = []
    for rank in range(1, 5):
        tensor = rng.normal(scale=.015, size=(nocc,) * rank + (nvir,) * rank)
        # Explicit signed antisymmetrization keeps the independent tensor
        # entries easy to inspect while exercising all BCH products.
        anti = np.zeros_like(tensor)
        for hp in permutations(range(rank)):
            for pp in permutations(range(rank)):
                anti += reference._parity(hp) * reference._parity(pp) * tensor.transpose(
                    hp + tuple(rank + i for i in pp))
        tensor = anti / float(factorial(rank) ** 2)
        tamps.append(tensor)
    return tuple(tamps)


class BCHOracle(unittest.TestCase):
    def test_native_full_and_compact_adapter(self):
        f, _, spin = _spin_conserving_hamiltonian(nocc=5, nmo=10)
        no, nv, oa, va = 5, 5, 2, 2
        ob, vb = no - oa, nv - va
        rng = np.random.default_rng(51)
        factors = rng.normal(size=(3, 10, 10))
        factors += factors.transpose(0, 2, 1)
        factors *= spin[:, None] == spin[None, :]
        bare = np.einsum('xpr,xqs->pqrs', factors, factors)
        occ = (np.arange(oa), oa + np.arange(ob))
        vir = (np.arange(va), va + np.arange(vb))
        alpha, beta = np.r_[occ[0], no + vir[0]], np.r_[occ[1], no + vir[1]]
        eris = SimpleNamespace(fock=(f[np.ix_(alpha, alpha)], f[np.ix_(beta, beta)]),
                               pppp=bare[np.ix_(alpha, alpha, alpha, alpha)],
                               PPPP=bare[np.ix_(beta, beta, beta, beta)],
                               pPpP=bare[np.ix_(alpha, beta, alpha, beta)])
        mycc = SimpleNamespace(nocc=(oa, ob), nmo=(oa + va, ob + vb))
        full, native, compact = [], [], []
        for rank, tensor in enumerate(_random_full_tamps(no, nv), 1):
            axes = np.indices(tensor.shape)
            spin_h = sum(spin[axes[k]] for k in range(rank))
            spin_p = sum(spin[no + axes[rank + k]] for k in range(rank))
            tensor = tensor * (spin_h == spin_p)
            full.append(tensor)
            blocks, packed = [], []
            for na in range(rank, -1, -1):
                nb = rank - na
                first = 0 if na >= nb else 1
                r1, r2 = max(na, nb), min(na, nb)
                inds = [occ[first]] * r1 + [occ[1 - first]] * r2
                inds += [vir[first]] * r1 + [vir[1 - first]] * r2
                order = tuple(range(r1)) + tuple(range(rank, rank + r1))
                order += tuple(range(r1, rank)) + tuple(range(rank + r1, 2 * rank))
                block = tensor[np.ix_(*inds)].transpose(order)
                blocks.append(block)
                vals = [tuple(combinations(range(len(occ[first])), r1)),
                        tuple(combinations(range(len(vir[first])), r1))]
                if r2:
                    vals += [tuple(combinations(range(len(occ[1 - first])), r2)),
                             tuple(combinations(range(len(vir[1 - first])), r2))]
                pack = np.zeros(tuple(map(len, vals)))
                for idx in product(*(range(len(x)) for x in vals)):
                    pack[idx] = block[sum((vals[i][j] for i, j in enumerate(idx)), ())]
                packed.append(pack if rank >= 3 else block)
            native.append(blocks)
            compact.append(packed)
        for tamps in (native, compact):
            inputs = reference.spinorbital_inputs(mycc, eris, tamps)
            np.testing.assert_array_equal(inputs['f'], f)
            np.testing.assert_array_equal(inputs['v'], bare - bare.transpose(0, 1, 3, 2))
            for actual, expected in zip(inputs['tamps'], full):
                np.testing.assert_allclose(actual, expected, atol=1e-15)

    def test_finite_bch_series_matches_dense_exponentials(self):
        f, v, spin = _spin_conserving_hamiltonian()
        tamps = _random_full_tamps(4, 4)
        energy, residuals, details = reference.bch_residuals(
            f, v, tamps, spin, 4, return_details=True)
        cluster = details['cluster'].toarray()
        h_normal = details['h_normal'].toarray()
        ref = np.zeros(cluster.shape[0])
        ref[details['determinant_index'][details['reference_bits']]] = 1
        expected_ket = linalg.expm(cluster) @ ref
        expected_similarity = linalg.expm(-cluster) @ h_normal @ expected_ket
        np.testing.assert_allclose(details['exp_t'], expected_ket, atol=2e-12,
                                   rtol=2e-12)
        np.testing.assert_allclose(details['similarity'], expected_similarity,
                                   atol=2e-12, rtol=2e-12)
        ridx = details['determinant_index'][details['reference_bits']]
        self.assertAlmostEqual(energy, expected_similarity[ridx], 14)
        self.assertEqual(tuple(x.ndim for x in residuals), (2, 4, 6, 8))
        self.assertAlmostEqual(float(details['h_normal'][ridx, ridx]), 0., 14)

    def test_normal_ordered_reference_and_zero_cluster(self):
        f, v, spin = _spin_conserving_hamiltonian()
        nocc, nvir = 4, 4
        tamps = tuple(np.zeros((nocc,) * rank + (nvir,) * rank)
                      for rank in range(1, 5))
        energy, residuals, details = reference.bch_residuals(
            f, v, tamps, spin, nocc, return_details=True)
        ridx = details['determinant_index'][details['reference_bits']]
        self.assertAlmostEqual(float(details['h_normal'][ridx, ridx]), 0., 13)
        self.assertAlmostEqual(energy, 0., 13)
        # A normal-ordered Hamiltonian has only one- and two-body projection
        # at T=0, hence rank-3 and rank-4 residuals vanish exactly.
        self.assertLess(np.linalg.norm(residuals[2]), 1e-13)
        self.assertLess(np.linalg.norm(residuals[3]), 1e-13)

    def test_two_electron_fci_state_has_zero_bch_residual(self):
        """A two-electron FCI eigenvector is exactly representable by T1+T2."""
        f, v, spin = _spin_conserving_hamiltonian(nocc=2, nmo=6, seed=19)
        nocc, nvir = 2, 4
        zeros = tuple(np.zeros((nocc,) * rank + (nvir,) * rank)
                      for rank in range(1, 5))
        _, _, ref_details = reference.bch_residuals(
            f, v, zeros, spin, nocc, return_details=True)
        ham = ref_details['h_normal'].toarray()
        eig, vec = linalg.eigh(ham)
        reference_index = ref_details['determinant_index'][ref_details['reference_bits']]
        ci = vec[:, np.argmin(eig)] / vec[reference_index, np.argmin(eig)]
        t1 = np.zeros((nocc, nvir))
        for holes in combinations(range(nocc), 1):
            for particles in combinations(range(nvir), 1):
                bits, sign = reference._apply(ref_details['reference_bits'], holes,
                                               tuple(nocc + a for a in particles))
                if bits in ref_details['determinant_index']:
                    t1[holes + particles] = ci[ref_details['determinant_index'][bits]] / sign
        t1_matrix = reference._operator_matrix(
            [(t1[i, a], (i,), (nocc + a,))
             for i in range(nocc) for a in range(nvir) if t1[i, a]],
            ref_details['determinants'])
        ref_vec = np.zeros(len(ref_details['determinants']))
        ref_vec[reference_index] = 1
        disconnected = .5 * t1_matrix.dot(t1_matrix.dot(ref_vec))
        t2 = np.zeros((nocc, nocc, nvir, nvir))
        for holes in combinations(range(nocc), 2):
            for particles in combinations(range(nvir), 2):
                bits, sign = reference._apply(ref_details['reference_bits'], holes,
                                               tuple(nocc + a for a in particles))
                if bits not in ref_details['determinant_index']:
                    continue
                idx = ref_details['determinant_index'][bits]
                t2[holes + particles] = (ci[idx] - disconnected[idx]) / sign
                t2[holes[::-1] + particles] = -t2[holes + particles]
                t2[holes + particles[::-1]] = -t2[holes + particles]
                t2[holes[::-1] + particles[::-1]] = t2[holes + particles]
        tamps = (t1, t2, np.zeros((nocc,) * 3 + (nvir,) * 3),
                 np.zeros((nocc,) * 4 + (nvir,) * 4))
        energy, residuals = reference.bch_residuals(f, v, tamps, spin, nocc)
        self.assertAlmostEqual(energy, eig.min(), 10)
        self.assertLess(max(np.linalg.norm(x) for x in residuals), 2e-9)

    def test_four_electron_fci_state_has_zero_bch_residual(self):
        """Extract connected amplitudes recursively from a full CI vector."""
        f, v, spin = _spin_conserving_hamiltonian(seed=29)
        nocc, nvir = 4, 4
        tamps = [np.zeros((nocc,) * rank + (nvir,) * rank)
                 for rank in range(1, 5)]
        _, _, details = reference.bch_residuals(f, v, tamps, spin, nocc, True)
        eig, vectors = linalg.eigh(details['h_normal'].toarray())
        index = details['determinant_index']
        ref_bits = details['reference_bits']
        ci = vectors[:, 0] / vectors[index[ref_bits], 0]
        for rank in range(1, 5):
            # With lower-rank amplitudes fixed, the rank-n connected
            # coefficient is CI_n minus the known disconnected exp(T)_n.
            _, _, lower = reference.bch_residuals(f, v, tamps, spin, nocc, True)
            difference = ci - lower['exp_t']
            for holes in combinations(range(nocc), rank):
                for particles in combinations(range(nvir), rank):
                    bits, sign = reference._apply(
                        ref_bits, holes, tuple(nocc + a for a in particles))
                    if bits not in index:
                        continue
                    value = difference[index[bits]] / sign
                    for hp in permutations(range(rank)):
                        for pp in permutations(range(rank)):
                            idx = tuple(holes[i] for i in hp)
                            idx += tuple(particles[a] for a in pp)
                            tamps[rank - 1][idx] = (
                                reference._parity(hp) * reference._parity(pp) * value)
        energy, residuals, final = reference.bch_residuals(
            f, v, tamps, spin, nocc, True)
        np.testing.assert_allclose(final['exp_t'], ci, atol=2e-12)
        self.assertAlmostEqual(energy, eig[0], 11)
        self.assertLess(max(np.linalg.norm(x) for x in residuals), 2e-10)
        self.assertGreater(np.linalg.norm(tamps[3]), 1e-8)

    def test_more_than_four_electrons_keeps_unprojected_fock_space(self):
        f, v, spin = _spin_conserving_hamiltonian(nocc=5, nmo=10, seed=35)
        tamps = _random_full_tamps(5, 5, seed=3)
        _, _, details = reference.bch_residuals(f, v, tamps, spin, 5, True)
        matrix = details['cluster'].toarray()
        ref = np.zeros(len(details['determinants']))
        ref[details['determinant_index'][details['reference_bits']]] = 1
        expected = linalg.expm(-matrix) @ details['h_normal'] @ linalg.expm(matrix) @ ref
        np.testing.assert_allclose(details['similarity'], expected, atol=2e-12)
        ranks = np.array([reference._bit_count(details['reference_bits'] & ~bits)
                          for bits in details['determinants']])
        self.assertEqual(ranks.max(), 5)
        # CCSDTQ has no T5, but exp(T1+...+T4) contains disconnected rank-5
        # components.  Discarding them before similarity transformation
        # would corrupt the low-rank residuals.
        self.assertGreater(np.linalg.norm(details['exp_t'][ranks == 5]), 1e-10)


if __name__ == '__main__':
    unittest.main()
