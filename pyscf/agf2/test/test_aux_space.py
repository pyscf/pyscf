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

"""Auxiliary-space identities on small, deterministic matrices."""
import unittest

import numpy as np

from pyscf import lib
from pyscf.agf2 import aux_space


class KnownValues(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.energy = np.array([2., -3., .8, -1.5, 3., -2., 1., -.5])
        self.coupling = rng.normal(size=(2, 8)) * .2
        self.phys = np.array([[-1., .15], [.15, .7]])
        self.se = aux_space.SelfEnergy(self.energy, self.coupling, chempot=.1)

    def test_sort_partition_and_copy(self):
        order = np.argsort(self.energy)
        np.testing.assert_array_equal(self.se.energy, self.energy[order])
        np.testing.assert_array_equal(self.se.coupling, self.coupling[:, order])
        self.assertEqual((self.se.nphys, self.se.naux), (2, 8))
        for part, mask in ((self.se.get_occupied(), self.se.energy < .1),
                           (self.se.get_virtual(), self.se.energy >= .1),
                           (self.se.copy(), np.ones(8, dtype=bool))):
            self.assertIsInstance(part, aux_space.SelfEnergy)
            self.assertEqual(part.chempot, .1)
            np.testing.assert_array_equal(part.energy, self.se.energy[mask])
            np.testing.assert_array_equal(part.coupling, self.se.coupling[:, mask])
            self.assertFalse(np.shares_memory(part.energy, self.se.energy))
            self.assertFalse(np.shares_memory(part.coupling, self.se.coupling))
        at_mu = aux_space.AuxiliarySpace([0., 1.], [[1., 2.]], chempot=0.)
        self.assertEqual(at_mu.get_occupied().naux, 0)
        self.assertEqual(at_mu.get_virtual().naux, 2)

    def test_array_dot_and_eigenvectors(self):
        rng = np.random.default_rng(8)
        for complex_ in (False, True):
            with self.subTest(complex=complex_):
                coupling = self.coupling.copy()
                phys = self.phys.copy()
                if complex_:
                    coupling = coupling + .1j * rng.normal(size=coupling.shape)
                    phys = phys + np.array([[0, .2j], [-.2j, 0]])
                se = aux_space.SelfEnergy(self.energy, coupling)
                expected = np.block([[phys, se.coupling],
                                     [se.coupling.conj().T,
                                      np.diag(se.energy - .3)]])
                np.testing.assert_allclose(se.get_array(phys, chempot=.3), expected)
                buffer = np.full_like(expected, 7.)
                self.assertIs(se.get_array(phys, out=buffer, chempot=.3), buffer)
                np.testing.assert_allclose(buffer, expected)
                for shape in ((10,), (10, 3)):
                    vec = rng.normal(size=shape)
                    if complex_:
                        vec = vec + 1j * rng.normal(size=shape)
                    product = expected @ vec
                    np.testing.assert_allclose(se.dot(phys, vec, chempot=.3), product,
                                               atol=1e-14)
                    out = np.empty_like(product)
                    actual = se.dot(phys, vec, out=out, chempot=.3)
                    self.assertTrue(np.shares_memory(actual, out))
                    np.testing.assert_allclose(out, product, atol=1e-14)
                w, v = se.eig(phys, out=buffer, chempot=.3)
                np.testing.assert_allclose(expected @ v, v * w, atol=1e-13)
                np.testing.assert_allclose(v.conj().T @ v, np.eye(10), atol=1e-13)

    def test_moments_and_remove_uncoupled(self):
        coupling = self.coupling.astype(complex)
        coupling[0] += .1j
        se = aux_space.SelfEnergy(self.energy, coupling)
        orders = [0, 1, 3]
        expected = np.array([sum(e**n * np.outer(v, v.conj())
                                 for e, v in zip(se.energy, se.coupling.T))
                             for n in orders])
        np.testing.assert_allclose(se.moment(orders), expected, atol=1e-14)
        np.testing.assert_allclose(se.moment(1), expected[1], atol=1e-14)
        self.assertEqual(se.moment(1, squeeze=False).shape, (1, 2, 2))
        se = aux_space.SelfEnergy([-1., 0., 1.], [[0., .5, 1.], [0., 0., 1.]])
        se.remove_uncoupled(.25)
        np.testing.assert_array_equal(se.energy, [0., 1.])
        np.testing.assert_array_equal(se.coupling, [[.5, 1.], [0., 1.]])

    def test_dot_dtype_from_physical_matrix(self):
        phys = self.phys + np.array([[0., .2j], [-.2j, 0.]])
        vector = np.arange(10.)
        expected = self.se.get_array(phys) @ vector
        np.testing.assert_allclose(self.se.dot(phys, vector), expected, atol=1e-14)

    def test_checkpoint(self):
        with lib.NamedTemporaryFile(suffix='.h5') as tmp:
            for key in (None, 'nested/self_energy'):
                with self.subTest(key=key):
                    path = tmp.name
                    self.se.save(path, key=key)
                    restored = aux_space.SelfEnergy.load(path, key=key)
                    self.assertEqual(restored.chempot, self.se.chempot)
                    np.testing.assert_array_equal(restored.energy, self.se.energy)
                    np.testing.assert_array_equal(restored.coupling, self.se.coupling)

    def test_combine(self):
        combined = aux_space.combine(self.se.get_virtual(), self.se.get_occupied())
        np.testing.assert_array_equal(combined.energy, self.se.energy)
        np.testing.assert_array_equal(combined.coupling, self.se.coupling)
        self.assertEqual(combined.chempot, self.se.chempot)
        with self.assertRaisesRegex(ValueError, 'physical space'):
            aux_space.combine(self.se, aux_space.SelfEnergy([1.], [[1.]]))

    def test_greens_function_and_density(self):
        gf = self.se.get_greens_function(self.phys)
        h = self.se.get_array(self.phys)
        w, v = np.linalg.eigh(h)
        np.testing.assert_allclose(gf.energy, w)
        for mu in (None, -1., 10., -10.):
            for occupancy in (1, 2):
                cutoff = self.se.chempot if mu is None else mu
                occupied = v[:2, w < cutoff]
                expected = occupancy * occupied @ occupied.T
                np.testing.assert_allclose(gf.make_rdm1(mu, occupancy), expected,
                                           atol=1e-13)
                np.testing.assert_allclose(
                    self.se.make_rdm1(self.phys, mu, occupancy), expected, atol=1e-13)
        # Dyson's equation independently checks all Green's function residues.
        z = .4 + .2j
        sigma = (self.se.coupling / (z - self.se.energy)) @ self.se.coupling.T
        expected = np.linalg.inv(z * np.eye(2) - self.phys - sigma)
        actual = (gf.coupling / (z - gf.energy)) @ gf.coupling.T
        np.testing.assert_allclose(actual, expected, atol=1e-13)

    def test_spectrum_blocks(self):
        gf = self.se.get_greens_function(self.phys)
        grid = np.linspace(-4, 4, 501)  # Cross the internal 240-point blocks.
        z = grid[:, None] - gf.energy + gf.chempot - .05j
        expected = -(np.sum(abs(gf.coupling)**2, axis=0) / z).sum(axis=1).imag / np.pi
        np.testing.assert_allclose(gf.real_freq_spectrum(grid, eta=.05), expected,
                                   atol=1e-13)
        self.assertEqual(gf.real_freq_spectrum(np.array([])).shape, (0,))
        with self.assertRaisesRegex(ValueError, 'Convert SelfEnergy'):
            self.se.real_freq_spectrum(grid)
        with self.assertRaisesRegex(ValueError, 'SelfEnergy'):
            gf.compress()

    def test_compression_conserves_moments(self):
        for n in (0, 1):
            for method in ('se', 'gf'):
                with self.subTest(n=n, method=method):
                    levels = (None, n) if method == 'se' else (n, None)
                    compressed = self.se.compress(self.phys, n=levels)
                    self.assertLessEqual(compressed.naux, 2 * self.se.nphys * (n + 1))
                    self.assertEqual(compressed.chempot, self.se.chempot)
                    before, after = self.se, compressed
                    if method == 'gf':
                        before = before.get_greens_function(self.phys)
                        after = after.get_greens_function(self.phys)
                    for part in ('get_occupied', 'get_virtual'):
                        orders = list(range(2 * n + 2))
                        np.testing.assert_allclose(
                            getattr(after, part)().moment(orders),
                            getattr(before, part)().moment(orders), atol=1e-11, rtol=1e-11)
        copied = self.se.compress(n=(None, None))
        self.assertIsNot(copied, self.se)
        np.testing.assert_array_equal(copied.coupling, self.se.coupling)
        both = self.se.compress(self.phys, n=(0, 1))
        # The final GF compression preserves the GF moments of the
        # SE-compressed intermediate, not all of its self-energy moments.
        intermediate = self.se.compress(n=(None, 1)).get_greens_function(self.phys)
        gf = both.get_greens_function(self.phys)
        for part in ('get_occupied', 'get_virtual'):
            np.testing.assert_allclose(getattr(gf, part)().moment([0, 1]),
                                       getattr(intermediate, part)().moment([0, 1]),
                                       atol=1e-12)

    def test_compression_empty_and_small_spaces(self):
        for energies in ([], [-2.], [2.]):
            with self.subTest(energies=energies):
                se = aux_space.SelfEnergy(energies, np.ones((2, len(energies))))
                compressed = se.compress()
                np.testing.assert_array_equal(compressed.energy, se.energy)
                np.testing.assert_array_equal(compressed.coupling, se.coupling)
        # Rank-deficient coupling exercises the stabilized Cholesky path.
        se = aux_space.SelfEnergy([-3., -2., -1.], [[1., 2., 3.], [0., 0., 0.]])
        compressed = se.compress()
        np.testing.assert_allclose(compressed.moment([0, 1]), se.moment([0, 1]),
                                   atol=1e-12)

    def test_davidson(self):
        h = self.se.get_array(self.phys)
        exact = np.linalg.eigvalsh(h)
        for which in ('SM', 'LM', 'SA', 'LA'):
            with self.subTest(which=which):
                conv, w, v = aux_space.davidson(self.se, self.phys, nroots=2, which=which)
                self.assertTrue(np.all(conv))
                key = abs(exact) if which.endswith('M') else exact
                order = np.argsort(key)
                if which.startswith('L'):
                    order = order[::-1]
                np.testing.assert_allclose(w, exact[order[:2]], atol=1e-10)
                for energy, vector in zip(w, v):
                    np.testing.assert_allclose(h @ vector, energy * vector, atol=1e-10)
        with self.assertRaises(ValueError):
            aux_space.davidson(self.se, self.phys, which='invalid')
        conv, w, v = aux_space.davidson(self.se, self.phys, which='SA',
                                        maxiter=50, ntrial=10)
        self.assertTrue(np.all(conv))
        np.testing.assert_allclose(w, exact[:1], atol=1e-10)

    def test_invalid_physical_shape(self):
        for method, args in ((self.se.get_array, ()), (self.se.eig, ()),
                             (self.se.dot, (np.ones(10),))):
            with self.subTest(method=method.__name__):
                with self.assertRaisesRegex(ValueError, 'physical space'):
                    method(np.eye(3), *args)


if __name__ == '__main__':
    unittest.main()
