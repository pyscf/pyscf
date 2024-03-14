#!/usr/bin/env python

import unittest
import numpy as np

import pyscf.pbc.mp.kump2
from pyscf.pbc.mp.kump2 import get_nocc, get_nmo, get_frozen_mask, padding_k_idx

class fake_ump:
    def __init__(self, frozen, mo_occ, nkpts):
        self._nocc = None
        self._nmo = None
        self.frozen = frozen
        self.mo_occ = mo_occ
        self.nkpts = nkpts

    get_nocc = get_nocc
    get_nmo = get_nmo

    @property
    def nocc(self):
        return self.get_nocc()
    @property
    def nmo(self):
        return self.get_nmo()

class KnownValues(unittest.TestCase):
    def test_no_frozen(self):
        mo_occ = [[np.array([1,1,0,0,0]),],
                  [np.array([1,1,1,0,0]),]]
        mp = fake_ump(frozen=None, mo_occ=mo_occ, nkpts=1)
        nocc = get_nocc(mp)
        nmo = get_nmo(mp)
        self.assertAlmostEqual(nocc, (2,3))
        self.assertAlmostEqual(nmo, (5,5))

        nocc = get_nocc(mp, per_kpoint=True)
        nmo = get_nmo(mp, per_kpoint=True)
        self.assertAlmostEqual(nocc, ([2],[3]))
        self.assertAlmostEqual(nmo, ([5],[5]))

    def test_frozen_int(self):
        mo_occ = [[np.array([1,1,0,0,0]),],
                  [np.array([1,1,1,0,0]),]]
        mp = fake_ump(frozen=1, mo_occ=mo_occ, nkpts=1)
        nocc = get_nocc(mp)
        nmo = get_nmo(mp)
        self.assertAlmostEqual(nocc, (1,2))
        self.assertAlmostEqual(nmo, (4,4))  # 2 occupied, 3 virtual

        nocc = get_nocc(mp, per_kpoint=True)
        nmo = get_nmo(mp, per_kpoint=True)
        self.assertAlmostEqual(nocc, ([1],[2]))
        self.assertAlmostEqual(nmo, ([4],[4]))

    def test_frozen_list1(self):
        mo_occ = [[np.array([1,1,0,0,0]),],
                  [np.array([1,1,1,0,0]),]]
        mp = fake_ump(frozen=([1,],[1,]), mo_occ=mo_occ, nkpts=1)
        nocc = get_nocc(mp)
        nmo = get_nmo(mp)
        self.assertAlmostEqual(nocc, (1,2))
        self.assertAlmostEqual(nmo, (4,4))

        nocc = get_nocc(mp, per_kpoint=True)
        nmo = get_nmo(mp, per_kpoint=True)
        self.assertAlmostEqual(nocc, ([1],[2]))
        self.assertAlmostEqual(nmo, ([4],[4]))

    def test_frozen_list2(self):
        # Freeze virtual not contained in set
        mo_occ = [[np.array([1,1,0,0,0]),],
                  [np.array([1,1,1,0,0]),]]
        mp = fake_ump(frozen=([5,], [5,]), mo_occ=mo_occ, nkpts=1)
        self.assertRaises(RuntimeError, get_nocc, mp)
        self.assertRaises(RuntimeError, get_nmo, mp)  # Fails because it pads by calling get_nocc

    def test_frozen_list3(self):
        # Freeze all occupied alpha orbitals
        mo_occ = [[np.array([1,1,0,0,0]),],
                  [np.array([1,1,1,0,0]),]]
        mp = fake_ump(frozen=([0,1], [0,1]), mo_occ=mo_occ, nkpts=1)
        self.assertRaises(AssertionError, get_nocc, mp)  # No occupied alpha
        self.assertRaises(AssertionError, get_nmo, mp)  # Fails because calls get_nocc

    def test_frozen_list4(self):
        frozen = ([0,2], [0,1,3,4])
        mo_occ = [[np.array([1,1,0,0,0]),],
                  [np.array([1,1,1,0,0]),]]
        mp = fake_ump(frozen=frozen, mo_occ=mo_occ, nkpts=1)

        nocc = get_nocc(mp)
        nmo = get_nmo(mp)
        self.assertAlmostEqual(nocc, (1,1))
        self.assertAlmostEqual(nmo, (3,1))

        nocc = get_nocc(mp, per_kpoint=True)
        nmo = get_nmo(mp, per_kpoint=True)
        self.assertAlmostEqual(nocc, ([1],[1]))
        self.assertAlmostEqual(nmo, ([3],[1]))

    def test_frozen_repeated_orbital(self):
        mo_occ = [[np.array([1,1,0,0,0]),],
                  [np.array([1,1,1,0,0]),]]
        mp = fake_ump(frozen=([0,1,1], [0,1]), mo_occ=mo_occ, nkpts=1)
        self.assertRaises(RuntimeError, get_nocc, mp)
        self.assertRaises(RuntimeError, get_nmo, mp)  # Fails because it pads by calling get_nocc

    def test_frozen_kpt_list1(self):
        frozen = ([[2], [0,]],
                  [[0,1,3,4], [0,2]])
        mo_occ = [[np.array([1, 1, 1, 0, 0]),np.array([1, 1, 0, 0, 0]),],
                  [np.array([1, 0, 0, 0, 0]),np.array([1, 1, 1, 0, 0]),]]
        mp = fake_ump(frozen=frozen, mo_occ=mo_occ, nkpts=2)

        nocc = get_nocc(mp)
        nmo = get_nmo(mp)
        self.assertAlmostEqual(nocc, (2, 1))
        self.assertAlmostEqual(nmo, (5, 3))  # Take the max occ and max vir from kpoints -
                                             # spin0: 2 occ (from first kpoint), 3 vir (from second kpoint)
                                             # spin1: 1 occ (from second kpoint),2 vir (from second kpoint)

        nocc = get_nocc(mp, per_kpoint=True)
        nmo = get_nmo(mp, per_kpoint=True)
        self.assertAlmostEqual(nocc, ([2, 1], [0, 1]))
        self.assertAlmostEqual(nmo, ([4, 4], [1, 3]))

        #  Using `O` and `V` as occupied and virtual and `-` as a padded/frozen orbital,
        #  we will have the following orbital list
        #
        #        alpha spin      |  beta spin
        #  k1 :  O O | - V V     |  - | - V
        #  k2 :  O - | V V V     |  O | V V
        split_idx = padding_k_idx(mp, kind='split')
        outa = [[y for x in split_idx[0][idx] for y in x] for idx in range(2)]  # Flatten the list for ease-of-use
        outb = [[y for x in split_idx[1][idx] for y in x] for idx in range(2)]
        expecteda = [[0, 1, 0], [1, 2, 0, 1, 2]]  # [occ_idx, vir_idx] for alpha
        expectedb = [[0], [1, 0, 1]]              # [occ_idx, vir_idx] for beta
        self.assertAlmostEqual(outa, expecteda)
        self.assertAlmostEqual(outb, expectedb)

        outa, outb = padding_k_idx(mp, kind='joint')
        outa = [y for x in outa for y in x]
        outb = [y for x in outb for y in x]
        expecteda = [0,1,3,4,0,2,3,4]  # [occa_idx_k1, occa_idx_k2]
        expectedb = [2,0,1,2]
        self.assertAlmostEqual(outa, expecteda)
        self.assertAlmostEqual(outb, expectedb)

    def test_frozen_kpt_list2(self):
        frozen = ([[0,1], [0,]],
                  [[0], [0,1]])
        mo_occ = [[np.array([1, 1, 1, 0, 0]),np.array([1, 1, 0, 0, 0]),],
                  [np.array([1, 1, 0, 0, 0]),np.array([1, 1, 1, 0, 0]),]]
        mp = fake_ump(frozen=frozen, mo_occ=mo_occ, nkpts=2)

        nocc = get_nocc(mp)
        nmo = get_nmo(mp)
        self.assertAlmostEqual(nocc, (1, 1))
        self.assertAlmostEqual(nmo, (4, 4))  # We deleted the first occ from each of the alpha/beta, so
                                             # we have 1 occupied and 3 virtuals.

        nocc = get_nocc(mp, per_kpoint=True)
        nmo = get_nmo(mp, per_kpoint=True)
        self.assertAlmostEqual(nocc, ([1, 1], [1, 1]))
        self.assertAlmostEqual(nmo, ([3, 4], [4, 3]))

        #  Using `O` and `V` as occupied and virtual and `-` as a padded/frozen orbital,
        #  we will have the following orbital list
        #
        #        alpha spin  |  beta spin
        #  k1 :  O | - V V   |  O | V V V
        #  k2 :  O | V V V   |  O | - V V
        split_idx = padding_k_idx(mp, kind='split')
        outa = [[y for x in split_idx[0][idx] for y in x] for idx in range(2)]  # Flatten the list for ease-of-use
        outb = [[y for x in split_idx[1][idx] for y in x] for idx in range(2)]
        expecteda = [[0, 0], [1, 2, 0, 1, 2]]  # [occ_idx, vir_idx] for alpha
        expectedb = [[0, 0], [0, 1, 2, 1, 2]]  # [occ_idx, vir_idx] for beta
        self.assertAlmostEqual(outa, expecteda)
        self.assertAlmostEqual(outb, expectedb)

        outa, outb = padding_k_idx(mp, kind='joint')
        outa = [y for x in outa for y in x]
        outb = [y for x in outb for y in x]
        expecteda = [0,2,3,0,1,2,3]  # [occa_idx_k1, occa_idx_k2]
        expectedb = [0,1,2,3,0,2,3]
        self.assertAlmostEqual(outa, expecteda)
        self.assertAlmostEqual(outb, expectedb)


if __name__ == '__main__':
    print("Full mask test")
    unittest.main()
