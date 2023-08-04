#!/usr/bin/env python
# Copyright 2022-2023 The PySCF Developers. All Rights Reserved.
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
#
# Authors: Xing Zhang <zhangxing.nju@gmail.com>
#

import unittest
import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.symm.basis import symm_adapted_basis

def get_so(atom, a, basis, kmesh):
    cell = gto.Cell()
    cell.atom = atom
    cell.a = a
    cell.basis = basis
    cell.space_group_symmetry = True
    cell.symmorphic = True
    cell.build()
    kpts = cell.make_kpts(kmesh, with_gamma_point=True,
                          space_group_symmetry=True)
    sos_ks, irrep_ids_ks = symm_adapted_basis(cell, kpts)
    return sos_ks, irrep_ids_ks

def get_norb(sos_ks):
    norb_ks = []
    for sos_k in sos_ks:
        norb = []
        for sos in sos_k:
            norb.append(sos.shape[1])
        norb_ks.append(norb)
    return norb_ks

class KnowValues(unittest.TestCase):
    def test_C2h_symorb(self):
        atom = """
            Cu    1.145000    8.273858   11.128085
        """
        a = [[2.2899999619, 0.0000000000, 0.0000000000],
             [0.0000000000,11.8430004120, 0.0000000000],
             [0.0000000000, 4.7047163727,22.2561701795]]
        basis = 'gth-dzvp-molopt-sr'
        kmesh = [3,] * 3
        sos_ks, irrep_ids_ks = get_so(atom, a, basis, kmesh)
        assert irrep_ids_ks == [[0, 1, 2, 3],
                                [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                                [0], [0], [0], [0]]
        norb_ks = get_norb(sos_ks)
        assert norb_ks == [[8, 4, 8, 5],
                           [16, 9], [16, 9], [16, 9], [16, 9], [13, 12],
                           [25], [25], [25], [25]]

    def test_D3d_symorb(self):
        atom = """
            Li    5.281899    1.082489    0.642479
            Li   18.499727    3.791392    2.250266
        """
        a = [[8.3774538040, 0.0000000000, 0.0000000000],
             [7.7020859454, 3.2953913771, 0.0000000000],
             [7.7020859454, 1.5784896835, 2.8927451749]]
        basis = 'gth-dzvp-molopt-sr'
        kmesh = [2,] * 3
        sos_ks, irrep_ids_ks = get_so(atom, a, basis, kmesh)
        assert irrep_ids_ks == [[0, 3, 4, 5],
                                [0, 1, 2, 3],
                                [0, 1, 2, 3],
                                [0, 3, 4, 5]]
        norb_ks = get_norb(sos_ks)
        assert norb_ks == [[4, 4, 2, 2],
                           [5, 1, 1, 5],
                           [5, 1, 1, 5],
                           [4, 4, 2, 2]]

    def test_D6h_symorb(self):
        atom = """
            Zn    0.000000    1.516699    1.301750
            Zn    1.313500    0.758350    3.905250
        """
        a = [[ 2.627000, 0.000000, 0.000000],
             [-1.313500, 2.275049, 0.000000],
             [ 0.000000, 0.000000, 5.207000]]
        basis = 'gth-dzvp-molopt-sr'
        kmesh = [2,] * 3
        sos_ks, irrep_ids_ks = get_so(atom, a, basis, kmesh)
        assert irrep_ids_ks == [[0, 1, 2, 3, 4, 5],
                                [0, 1, 2, 3, 4, 5],
                                [0, 1, 2, 3],
                                [0, 1, 2, 3]]
        norb_ks = get_norb(sos_ks)
        assert norb_ks == [[8, 8, 1, 1, 16, 16],
                           [8, 8, 1, 1, 16, 16],
                           [16, 9, 16, 9],
                           [16, 9, 16, 9]]

    def test_Td_symorb(self):
        atom = """
            Si  0.0 0.0 0.0
            Si  1.3467560987 1.3467560987 1.3467560987
        """
        a = [[0.0, 2.6935121974, 2.6935121974],
             [2.6935121974, 0.0, 2.6935121974],
             [2.6935121974, 2.6935121974, 0.0]]
        basis = 'gth-dzvp-molopt-sr'
        kmesh = [2,] * 3
        sos_ks, irrep_ids_ks = get_so(atom, a, basis, kmesh)
        assert irrep_ids_ks == [[0, 2, 4],
                                [0, 1, 2, 3, 4],
                                [0, 2]]
        norb_ks = get_norb(sos_ks)
        assert norb_ks == [[4, 4, 18], [6, 1, 1, 6, 12], [10, 16]]

        a = 5.431020511
        xyz = np.array(
                [[0,0,0],
                 [0.25,0.25,0.25],
                 [0, 0.5, 0.5],
                 [0.25, 0.75, 0.75],
                 [0.5, 0, 0.5],
                 [.75, .25, .75],
                 [.5, .5, 0],
                 [.75, .75, .25]]
            ) * a
        atom = []
        for ix in xyz:
            atom.append(['Si', list(ix)])
        a = np.eye(3) * a
        sos_ks, irrep_ids_ks = get_so(atom, a, basis, kmesh)
        assert irrep_ids_ks == [[0, 1, 2, 3, 4],
                                [0, 1, 2, 3, 4],
                                [0, 1, 2, 3, 4],
                                [0, 2, 3, 4]]
        norb_ks = get_norb(sos_ks)
        assert norb_ks == [[10, 1, 18, 21, 54], [18, 9, 8, 19, 50],
                           [18, 8, 18, 8, 52], [10, 16, 24, 54]]

    def test_Oh_symorb(self):
        atom = """
            Si  0.0 0.0 0.0
        """
        a = np.eye(3) * 2.
        basis = 'gth-dzvp-molopt-sr'
        kmesh = [2,] * 3
        sos_ks, irrep_ids_ks = get_so(atom, a, basis, kmesh)
        assert irrep_ids_ks == [[0, 4, 6, 9],
                                [0, 1, 2, 7, 8, 9],
                                [0, 1, 2, 7, 8, 9],
                                [0, 4, 6, 9]]
        norb_ks = get_norb(sos_ks)
        assert norb_ks == [[2, 2, 3, 6], [3, 1, 1, 2, 2, 4],
                           [3, 1, 1, 2, 2, 4], [2, 2, 3, 6]]

if __name__ == "__main__":
    print("Full Tests for pbc.symm.basis")
