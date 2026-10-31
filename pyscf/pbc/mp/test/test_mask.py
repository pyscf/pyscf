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

    def test_invalid_orbital_energy_mask_matches_nmo(self):
        from pyscf.pbc.scf.hf import INVALID_ORBITAL_ENERGY
        from pyscf.pbc.mp.kmp2 import padded_mo_coeff
        inv = INVALID_ORBITAL_ENERGY
        mp = fake_mp(frozen=None, mo_occ=[np.array([2., 2., 0., 0., 0.])], nkpts=1)
        mp.mo_energy = np.array([[-1., -0.4, 0.1, inv, inv]])
        nmo = get_nmo(mp, per_kpoint=True)
        mask = get_frozen_mask(mp)
        self.assertListEqual(nmo, [3])
        self.assertListEqual([int(x.sum()) for x in mask], [3])
        self.assertTrue(np.array_equal(mask[0], np.array([True, True, True, False, False])))
        mp.nmo = get_nmo(mp)
        nao = 4
        mo_coeff = [np.arange(nao * 5, dtype=float).reshape(nao, 5)]
        pad = padded_mo_coeff(mp, mo_coeff)
        self.assertEqual(pad.shape, (1, nao, 3))

        mp_ok = fake_mp(frozen=None, mo_occ=[np.array([2., 2., 0., 0., 0.])], nkpts=1)
        mp_ok.mo_energy = np.array([[-1., -0.4, 0.1, 0.2, 0.3]])
        self.assertListEqual(get_nmo(mp_ok, per_kpoint=True), [5])
        self.assertListEqual([int(x.sum()) for x in get_frozen_mask(mp_ok)], [5])

        mp2 = fake_mp(frozen=None,
                      mo_occ=[np.array([2., 0.]), np.array([2., 0.])], nkpts=2)
        mp2.mo_energy = np.array([[-1., inv], [0.2, 0.3]])
        self.assertListEqual(get_nmo(mp2, per_kpoint=True), [1, 2])
        self.assertListEqual([int(x.sum()) for x in get_frozen_mask(mp2)], [1, 2])

if __name__ == '__main__':
    print("Full mask test")
    unittest.main()
