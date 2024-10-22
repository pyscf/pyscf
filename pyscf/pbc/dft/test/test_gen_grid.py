#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
from pyscf import dft
from pyscf.pbc import gto as pgto
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint


def get_ovlp(cell, grids=None):
    if grids is None:
        grids = gen_grid.BeckeGrids(cell)
        grids.level = 3
        grids.build()

    aoR = numint.eval_ao(cell, grids.coords)
    s = numpy.dot(aoR.T.conj(), grids.weights.reshape(-1,1)*aoR).real
    return s


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_becke_grids(self):
        L = 4.
        n = 61
        cell = pgto.Cell()
        cell.a = numpy.eye(3)*L
        cell.a[1,0] = cell.a[2,1] = L / 2
        cell.mesh = numpy.array([n,n,n])

        cell.atom =[['He' , ( L/2+0., L/2+0. ,   L/2+1.)],
                    ['He' , ( L/2+1., L/2+0. ,   L/2+1.)]]
        cell.basis = {'He': [[0, (1.0, 1.0)]]}
        cell.rcut = 6.78614042442
        cell.build()
        grids = gen_grid.BeckeGrids(cell)
        grids.level = 3
        grids.build()
        s1 = get_ovlp(cell, grids)
        s2 = cell.pbc_intor('int1e_ovlp_sph')
        self.assertAlmostEqual(numpy.linalg.norm(s1-s2), 0, 4)
        self.assertEqual(grids.weights.size, 15042)

    def test_becke_grids_2d(self):
        L = 4.
        n = 61
        cell = pgto.Cell()
        cell.a = numpy.eye(3)*L
        cell.a[1,0] = cell.a[1,1] = L / 2
        cell.mesh = numpy.array([n,n,n])

        cell.atom = 'He 1 0 1'
        cell.basis = {'He': [[0, (1.0, 1.0)]]}
        cell.rcut = 6.78614042442
        cell.dimension = 2
        cell.low_dim_ft_type = 'inf_vacuum'
        cell.build()
        grids = gen_grid.BeckeGrids(cell)
        grids.level = 3
        grids.build()
        s1 = get_ovlp(cell, grids)
        s2 = cell.pbc_intor('int1e_ovlp_sph')
        self.assertAlmostEqual(numpy.linalg.norm(s1-s2), 0, 5)
        self.assertEqual(grids.weights.size, 7615)

    def test_becke_grids_1d(self):
        L = 4.
        n = 61
        cell = pgto.Cell()
        cell.a = numpy.eye(3)*L
        cell.mesh = numpy.array([n,n,n])

        cell.atom = 'He 1 0 1'
        cell.basis = {'He': [[0, (1.0, 1.0)]]}
        cell.rcut = 6.78614042442
        cell.dimension = 1
        cell.low_dim_ft_type = 'inf_vacuum'
        cell.build()
        grids = gen_grid.BeckeGrids(cell)
        grids.level = 3
        grids.build()
        s1 = get_ovlp(cell, grids)
        s2 = cell.pbc_intor('int1e_ovlp_sph')
        self.assertAlmostEqual(numpy.linalg.norm(s1-s2), 0, 5)
        self.assertEqual(grids.weights.size, 8040)

    def test_becke_grids_2d_low_dim_ft_type(self):
        L = 4.
        n = 61
        cell = pgto.Cell()
        cell.a = numpy.eye(3)*L
        cell.a[1,0] = cell.a[1,1] = L / 2
        cell.a[2,2] = L * 2
        cell.mesh = numpy.array([n,n,n])

        cell.atom = 'He 1 0 1'
        cell.basis = {'He': [[0, (1.0, 1.0)]]}
        cell.rcut = 6.78614042442
        cell.dimension = 2
        cell.build()
        grids = gen_grid.BeckeGrids(cell)
        grids.level = 3
        grids.build()
        s1 = get_ovlp(cell, grids)
        s2 = cell.pbc_intor('int1e_ovlp_sph')
        self.assertAlmostEqual(numpy.linalg.norm(s1-s2), 0, 5)
        self.assertEqual(grids.weights.size, 7251)

        grids = gen_grid.UniformGrids(cell)
        s1 = get_ovlp(cell, grids)
        self.assertAlmostEqual(numpy.linalg.norm(s1-s2), 0, 5)

    def test_becke_grids_round_error(self):
        cell = pgto.Cell()
        cell.atom = '''
            Cu1    1.927800000000   1.927800000000   1.590250000000
            Cu1    5.783400000000   5.783400000000   1.590250000000
            Cu2    1.927800000000   5.783400000000   1.590250000000
            Cu2    5.783400000000   1.927800000000   1.590250000000
            O1     1.927800000000   3.855600000000   1.590250000000
            O1     3.855600000000   5.783400000000   1.590250000000
            O1     5.783400000000   3.855600000000   1.590250000000
            O1     3.855600000000   1.927800000000   1.590250000000
            O1     0.000000000000   1.927800000000   1.590250000000
            O1     1.927800000000   7.711200000000   1.590250000000
            O1     7.711200000000   5.783400000000   1.590250000000
            O1     5.783400000000   0.000000000000   1.590250000000
        '''
        cell.a = numpy.array(
            [[7.7112,    0.,    0.],
             [0.,    7.7112,    0.],
             [0.,    0.,    3.1805]])
        cell.build()

        grids = gen_grid.BeckeGrids(cell)
        grids.level = 1
        grids.build()
        assert abs(cell.vol - grids.weights.sum()) < .1

if __name__ == '__main__':
    print("Full Tests for Becke grids")
    unittest.main()
