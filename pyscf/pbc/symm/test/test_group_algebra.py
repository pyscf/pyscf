# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Subgroup restriction and character products for crystallographic groups."""
import itertools
import unittest
import numpy as np
from pyscf.pbc.symm.group import PGElement, PointGroup, Representation


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = PointGroup([PGElement(np.diag(s)) for s in itertools.product((1, -1), repeat=3)])
        cls.sub = PointGroup([PGElement(np.eye(3, dtype=int)), PGElement(-np.eye(3, dtype=int))])

    def test_hash_and_inverse(self):
        for element in self.full.elements:
            np.testing.assert_array_equal(PGElement.decrypt_hash(hash(element)), element.matrix)
            np.testing.assert_array_equal((element * element.inv()).matrix, np.eye(3))

    def test_group_union_and_intersection(self):
        self.assertTrue(self.sub.issubset(self.full))
        self.assertFalse(self.full.issubset(self.sub))
        self.assertTrue(self.sub.issubset(self.sub))
        self.assertEqual(set((self.full & self.sub).hash_table), set(self.sub.hash_table))
        self.assertEqual(set((self.full | self.sub).hash_table), set(self.full.hash_table))
        self.assertIs(self.full & self.full, self.full)
        self.assertIs(self.full | self.full, self.full)

    def test_restriction_and_product(self):
        chi = np.array([np.trace(e.matrix) for e in self.full.elements])
        vector = Representation(self.full, chi=chi)
        np.testing.assert_allclose(Representation(self.full, rep=vector.rep).chi, chi, atol=1e-12)
        restricted = self.full.project_chi(chi, self.sub)
        np.testing.assert_array_equal(restricted, [3, -3])
        odd = Representation(self.sub, chi=np.array([1, -1]))
        product = vector @ odd
        np.testing.assert_array_equal(product.chi, [3, 3])
        self.assertEqual(product.rep.sum(), 3)
        np.testing.assert_array_equal(self.full.project_chi(chi, self.full), chi)

    def test_crystal_class_names(self):
        self.assertEqual(self.full.group_name(), 'mmm')
        self.assertEqual(self.full.group_name('Schoenflies'), 'D2h')
        self.assertEqual(self.sub.group_name('Schoenflies'), 'Ci')
        self.assertNotEqual(self.full.group_index, self.sub.group_index)


if __name__ == '__main__':
    unittest.main()
