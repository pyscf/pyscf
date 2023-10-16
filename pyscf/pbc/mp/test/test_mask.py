#!/usr/bin/env python

import unittest
import numpy as np

import pyscf.pbc.mp.kmp2
from pyscf.pbc.mp.kmp2 import get_nocc, get_nmo, get_frozen_mask

class fake_mp:
    def __init__(self, frozen, mo_occ, nkpts):
        self._nocc = None
        self._nmo = None
        self.frozen = frozen
        self.mo_occ = mo_occ
        self.nkpts = nkpts

    get_nocc = get_nocc
    get_nmo = get_nmo

class KnownValues(unittest.TestCase):
    def test_no_frozen(self):
        mp = fake_mp(frozen=None, mo_occ=[np.array([2, 2, 2, 0, 0]),], nkpts=1)
        nocc = get_nocc(mp)
        nmo = get_nmo(mp)
        self.assertAlmostEqual(nocc, 3)
        self.assertAlmostEqual(nmo, 5)

        nocc = get_nocc(mp, per_kpoint=True)
        nmo = get_nmo(mp, per_kpoint=True)
        self.assertListEqual(nocc, [3])
        self.assertListEqual(nmo, [5])

    def test_frozen_int(self):
        mp = fake_mp(frozen=1, mo_occ=[np.array([2, 2, 2, 0, 0]), np.array([2, 2, 0, 0, 0])], nkpts=2)
        nocc = get_nocc(mp)
        nmo = get_nmo(mp)
        self.assertAlmostEqual(nocc, 2)
        self.assertAlmostEqual(nmo, 5)  # 2 occupied, 3 virtual

        nocc = get_nocc(mp, per_kpoint=True)
        nmo = get_nmo(mp, per_kpoint=True)
        self.assertListEqual(nocc, [2, 1])
        self.assertListEqual(nmo, [4, 4])

    def test_frozen_list1(self):
        mp = fake_mp(frozen=[1,], mo_occ=[np.array([2, 2, 2, 0, 0]), np.array([2, 2, 0, 0, 0])], nkpts=2)
        nocc = get_nocc(mp)
        nmo = get_nmo(mp)
        self.assertAlmostEqual(nocc, 2)
        self.assertAlmostEqual(nmo, 5)  # 2 occupied, 3 virtual

        nocc = get_nocc(mp, per_kpoint=True)
        nmo = get_nmo(mp, per_kpoint=True)
        self.assertListEqual(nocc, [2, 1])
        self.assertListEqual(nmo, [4, 4])

    def test_frozen_list2(self):
        # Freeze virtual not contained in set
        mp = fake_mp(frozen=[4, 5], mo_occ=[np.array([2, 2, 2, 0, 0]), np.array([2, 2, 0, 0, 0])], nkpts=2)
        self.assertRaises(RuntimeError, get_nocc, mp)
        self.assertRaises(RuntimeError, get_nmo, mp)  # Fails because it pads by calling get_nocc

    def test_frozen_repeated_orbital(self):
        mp = fake_mp(frozen=[[1, 1], [0]], mo_occ=[np.array([2, 2, 2, 0, 0]), np.array([2, 2, 0, 0, 0])], nkpts=2)
        self.assertRaises(RuntimeError, get_nocc, mp)
        self.assertRaises(RuntimeError, get_nmo, mp)  # Fails because it pads by calling get_nocc

    def test_frozen_kpt_list1(self):
        mp = fake_mp(frozen=[[0, 1,], [0]], mo_occ=[np.array([2, 2, 2, 0, 0]), np.array([2, 2, 0, 0, 0])], nkpts=2)
        nocc = get_nocc(mp)
        nmo = get_nmo(mp)
        self.assertAlmostEqual(nocc, 1)
        self.assertAlmostEqual(nmo, 4)  # 1 occupied, 3 virtual

        nocc = get_nocc(mp, per_kpoint=True)
        nmo = get_nmo(mp, per_kpoint=True)
        self.assertListEqual(nocc, [1, 1])
        self.assertListEqual(nmo, [3, 4])

    def test_frozen_kpt_list2(self):
        mp = fake_mp(frozen=[[0,1],[],[0]], mo_occ=[np.array([2, 2, 2, 0, 0])] * 3, nkpts=3)
        nocc = get_nocc(mp)
        nmo = get_nmo(mp)
        self.assertAlmostEqual(nocc, 3)
        self.assertAlmostEqual(nmo, 5)  # 2nd k-point has 3 occupied and 2 virtual orbitals

        nocc = get_nocc(mp, per_kpoint=True)
        nmo = get_nmo(mp, per_kpoint=True)
        self.assertListEqual(nocc, [1, 3, 2])
        self.assertListEqual(nmo, [3, 5, 4])

    def test_frozen_kpt_list3(self):
        mp = fake_mp(frozen=[[0,1,3],[3],[0]], mo_occ=[np.array([2, 2, 2, 0, 0])] * 3, nkpts=3)
        nocc = get_nocc(mp)
        nmo = get_nmo(mp)
        self.assertAlmostEqual(nocc, 3)
        self.assertAlmostEqual(nmo, 5)  # 2nd k-point has 3 occupied and 2 virtual orbitals

        nocc = get_nocc(mp, per_kpoint=True)
        nmo = get_nmo(mp, per_kpoint=True)
        self.assertListEqual(nocc, [1, 3, 2])
        self.assertListEqual(nmo, [2, 4, 4])

if __name__ == '__main__':
    print("Full mask test")
    unittest.main()
