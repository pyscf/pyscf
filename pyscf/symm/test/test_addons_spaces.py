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

"""Symmetry projection and irrep algebra on explicitly constructed spaces."""
import io
import itertools
from types import SimpleNamespace
import unittest

import numpy as np
import scipy.linalg

from pyscf import lib
from pyscf.symm import addons, param


class KnownValues(unittest.TestCase):
    def setUp(self):
        self.overlap = np.diag([1., 2., 3., 4.])
        self.basis = np.diag(1 / np.sqrt(self.overlap.diagonal()))
        self.mol = SimpleNamespace(
            verbose=0, stdout=io.StringIO(), irrep_name=['Ag', 'B1u'],
            irrep_id=[0, 5], symm_orb=[self.basis[:, :2], self.basis[:, 2:]],
            intor_symmetric=lambda name: self.overlap.copy())

    def test_project_orbitals_by_name_and_id(self):
        mixed = self.basis.copy()
        mixed[:, 0] += .2 * self.basis[:, 2]
        for labels in ([0, 0, 5, 5], ['Ag', 'Ag', 'B1u', 'B1u'], None):
            with self.subTest(labels=labels):
                projected = addons.symmetrize_orb(self.mol, mixed, labels)
                np.testing.assert_allclose(projected, self.basis, atol=1e-14)
        labels = addons.label_orb_symm(self.mol, self.mol.irrep_name,
                                       self.mol.symm_orb, mixed, check=False)
        self.assertEqual(list(labels), ['Ag', 'Ag', 'B1u', 'B1u'])
        with self.assertRaisesRegex(ValueError, 'not symmetrized'):
            addons.label_orb_symm(self.mol, self.mol.irrep_id, self.mol.symm_orb, mixed)

    def test_symmetrize_complete_space(self):
        rng = np.random.default_rng(17)
        rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        for scale in (1., 1.2):
            for clean in (False, True):
                with self.subTest(scale=scale, clean=clean):
                    orbitals = self.basis @ rotation * scale
                    actual = addons.symmetrize_space(self.mol, orbitals, clean=clean)
                    np.testing.assert_allclose(actual.T @ self.overlap @ actual,
                                               np.eye(4), atol=1e-12)
                    np.testing.assert_allclose(actual @ actual.T,
                                               np.linalg.inv(self.overlap), atol=1e-12)
                    labels = addons.label_orb_symm(self.mol, self.mol.irrep_id,
                                                   self.mol.symm_orb, actual)
                    self.assertEqual(sorted(labels), [0, 0, 5, 5])

    def test_symmetrize_subspace_and_incomplete_space(self):
        pure = self.basis[:, :2]
        actual = addons.symmetrize_space(self.mol, pure, s=self.overlap, clean=True)
        np.testing.assert_allclose(actual @ actual.T, pure @ pure.T, atol=1e-14)
        # A single vector mixed across irreps is not a symmetry-closed space.
        mixed = self.basis[:, :1] + self.basis[:, 2:3]
        with self.assertRaises(ValueError):
            addons.symmetrize_space(self.mol, mixed)

    def test_find_symmetric_mo_with_redundant_salcs(self):
        projection = np.array([[1., 0.], [0., 0.]])
        overlap = np.diag([1., 0.])
        np.testing.assert_array_equal(addons.find_symmetric_mo(projection, overlap),
                                      [True, False])

    def test_multidimensional_partners(self):
        eye = np.eye(5)
        mol = SimpleNamespace(verbose=0, stdout=io.StringIO(),
                              irrep_name=['A1', 'E_1', 'E_2'], irrep_id=[0, 1, 2],
                              symm_orb=[eye[:, :1], eye[:, 1:3], eye[:, 3:]],
                              intor_symmetric=lambda name: eye)
        angle = .2
        rotate = lambda x: np.array([[np.cos(x), -np.sin(x)],
                                     [np.sin(x), np.cos(x)]])
        first = rotate(angle)
        second = rotate(angle + 1e-3)
        for keep_phase in (False, True):
            with self.subTest(keep_phase=keep_phase):
                mo = scipy.linalg.block_diag(1., first, -second)
                result = addons.symmetrize_multidim(mol, mo, keep_phase=keep_phase)
                np.testing.assert_allclose(result.T @ result, eye, atol=1e-12)
                np.testing.assert_allclose(result[1:3, 1:3], rotate(angle + .0005), atol=1e-12)
                sign = -1 if keep_phase else 1
                np.testing.assert_allclose(result[3:, 3:], sign * result[1:3, 1:3],
                                           atol=1e-12)
        # Missing partner orbitals should produce a meaningful ValueError.
        with self.assertRaisesRegex(ValueError, 'dimensionality'):
            addons.symmetrize_multidim(mol, eye[:, :4])
        with self.assertRaisesRegex(ValueError, 'dimensionality'):
            addons.symmetrize_multidim(mol, eye[:, 1:])

    def test_irrep_round_trips(self):
        for group, table in param.IRREP_ID_TABLE.items():
            for name, ident in table.items():
                with self.subTest(group=group, name=name):
                    self.assertEqual(addons.irrep_name2id(group.lower(), name), ident)
                    self.assertEqual(addons.irrep_id2name(group, ident), name)
        for group, names in (('Coov', ['A1', 'A2', 'E1x', 'E1y', 'E3x', 'E3y']),
                             ('Dooh', ['A1g', 'A2g', 'A1u', 'A2u', 'E2gx', 'E2uy']),
                             ('SO3', ['s+0', 'p-1', 'p+0', 'p+1', 'd+2'])):
            for name in names:
                with self.subTest(group=group, name=name):
                    self.assertEqual(addons.irrep_id2name(group, addons.irrep_name2id(group, name)),
                                     name)

    def test_direct_products(self):
        ids = np.arange(8)
        np.testing.assert_array_equal(addons.direct_prod(ids, ids), ids[:, None] ^ ids)
        for group, ids in (('Coov', [0, 1, 2, 3, 10, 11]),
                           ('Dooh', [0, 1, 2, 3, 4, 5, 6, 7, 10, 11]),
                           ('SO3', [0, 101, 102, 103, 200])):
            with self.subTest(group=group):
                ids = np.array(ids)
                product = addons.direct_prod(ids, ids, group)
                np.testing.assert_array_equal(product, product.T)
                np.testing.assert_array_equal(product[0], ids)
                np.testing.assert_array_equal(product.diagonal(), 0)
                self.assertIn(addons.MULTI_IRREPS, product)

    def test_route_matches_exhaustive_search(self):
        labels = np.array([0, 3, 0, 2, 5, 6])
        for nelec in (1, 2, 3, 6):
            for target in range(8):
                with self.subTest(nelec=nelec, target=target):
                    matches = [list(indices) for indices in itertools.combinations(range(6), nelec)
                               if np.bitwise_xor.reduce(labels[list(indices)]) == target]
                    found = addons.route(target, nelec, labels)
                    self.assertEqual(found, matches[0] if matches else [])

    def test_block_eigenvectors(self):
        matrix = np.array([[2., 0., .3], [0., -1., 0.], [.3, 0., 1.]])
        values, vectors = addons.eigh(matrix, [0, 5, 0])
        np.testing.assert_allclose(values, np.linalg.eigvalsh(matrix), atol=1e-14)
        np.testing.assert_allclose(matrix @ vectors, vectors * values, atol=1e-14)
        metric = np.diag([1., 2., 3.])
        values, vectors = lib.eigh_by_blocks(matrix, metric, labels=[0, 5, 0])
        np.testing.assert_allclose(matrix @ vectors, metric @ vectors * values, atol=1e-14)
        np.testing.assert_allclose(vectors.T @ metric @ vectors, np.eye(3), atol=1e-14)



if __name__ == '__main__':
    unittest.main()
