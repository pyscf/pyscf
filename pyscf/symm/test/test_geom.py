#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
from functools import reduce
from pyscf import lib
from pyscf import gto
from pyscf import symm
from pyscf.symm import geom


numpy.random.seed(12)
u = numpy.random.random((3,3))
u = numpy.linalg.svd(u)[0]

class KnownValues(unittest.TestCase):
    def test_d5h(self):
        atoms = ringhat(5, u)
        atoms = atoms[5:]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'D5h')

        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(gpname, 'C2v')
        self.assertTrue(geom.check_symm('C2v', atoms))
        self.assertEqual(geom.symm_identical_atoms(gpname, atoms),
                         [[0], [1, 4], [2, 3], [5, 6]])

        atoms = ringhat(5, u)
        atoms = atoms[5:]
        atoms[1][0] = 'C1'
        gpname, orig, axes = geom.detect_symm(atoms, {'C':'ccpvdz','C1':'sto3g','N':'631g'})
        self.assertEqual(gpname, 'C2v')

        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(gpname, 'C2v')
        self.assertTrue(geom.check_symm('C2v', atoms))
        self.assertEqual(geom.symm_identical_atoms(gpname, atoms),
                         [[0,2],[1],[3,4],[5,6]])

    def test_d6h(self):
        atoms = ringhat(6, u)
        atoms = atoms[6:]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'D6h')
        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(geom.symm_identical_atoms(gpname, atoms),
                         [[0,3],[1,2,4,5],[6,7]])
        self.assertTrue(geom.check_symm('D2h', atoms))

    def test_d6h_1(self):
        atoms = ringhat(6, u)
        atoms = atoms[6:]
        atoms[1][0] = 'C1'
        atoms[2][0] = 'C1'
        basis = {'C': 'sto3g', 'N':'sto3g', 'C1':'sto3g'}
        gpname, orig, axes = geom.detect_symm(atoms, basis)
        self.assertEqual(gpname, 'D6h')
        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(geom.symm_identical_atoms(gpname, atoms),
                         [[0,3],[1,2,4,5],[6,7]])
        self.assertTrue(geom.check_symm('D2h', atoms, basis))

    def test_c2v(self):
        atoms = ringhat(6, u)
        atoms = atoms[6:]
        atoms[1][0] = 'C1'
        atoms[2][0] = 'C1'
        basis = {'C': 'sto3g', 'N':'sto3g', 'C1':'ccpvdz'}
        gpname, orig, axes = geom.detect_symm(atoms, basis)
        self.assertEqual(gpname, 'C2v')
        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(geom.symm_identical_atoms(gpname, atoms),
                         [[0, 3], [1, 2], [4, 5], [6, 7]])
        self.assertTrue(geom.check_symm('C2', atoms, basis))

    def test_c2v_planar_mole(self):
        atoms = [
            ['O', [0, 0, 0.1197]],
            ['H', [0, 0.7616, -0.4786]],
            ['H', [0,-0.7616, -0.4786]],
        ]
        gpname, orig, axes = geom.detect_symm(atoms, {'O': 'sto3g', 'H': 'sto3g'})
        self.assertEqual(gpname, 'C2v')
        self.assertAlmostEqual(abs(axes - numpy.diag(axes.diagonal())).max(), 0, 12)

    def test_s4(self):
        atoms = [['C', (  0.5,   0    ,   1)],
                 ['O', (  0.4,   0.2  ,   1)],
                 ['C', ( -0.5,   0    ,   1)],
                 ['O', ( -0.4,  -0.2  ,   1)],
                 ['C', (  0  ,   0.5  ,  -1)],
                 ['O', ( -0.2,   0.4  ,  -1)],
                 ['C', (  0  ,  -0.5  ,  -1)],
                 ['O', (  0.2,  -0.4  ,  -1)]]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'S4')
        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(geom.symm_identical_atoms(gpname, atoms),
                         [[0, 2], [1, 3], [4, 6], [5, 7]])
        self.assertTrue(geom.check_symm('C2', atoms))

    def test_s6(self):
        rotz = random_rotz()
        atoms = ringhat(3, 1)[3:6] + ringhat(3, rotz)[:3]
        rotz[2,2] = -1
        rotz[:2,:2] = numpy.array(((.5, numpy.sqrt(3)/2),(-numpy.sqrt(3)/2, .5)))
        r = numpy.dot([x[1] for x in atoms], rotz) - numpy.array((0,0,3.5))
        atoms += list(zip([x[0] for x in atoms], r))

        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'S6')
        gpname, axes = geom.subgroup(gpname, axes)
        self.assertEqual(gpname, 'C3')

    def test_c5h(self):
        atoms = ringhat(5, u)
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'C5h')

        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(gpname, 'Cs')
        self.assertTrue(geom.check_symm('Cs', atoms))
        self.assertEqual(geom.symm_identical_atoms(gpname, atoms),
                         [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10,11]])

    def test_c5(self):
        atoms = ringhat(5, u)[:-1]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'C5')
        gpname, axes = geom.subgroup(gpname, axes)
        self.assertEqual(gpname, 'C1')

    def test_c5v(self):
        atoms = ringhat(5, u)[5:-1]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'C5v')

        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(gpname, 'Cs')
        self.assertTrue(geom.check_symm('Cs', atoms))
        self.assertEqual(geom.symm_identical_atoms(gpname, atoms),
                         [[0, 4], [1, 3], [2], [5]])

    def test_ih1(self):
        coords = numpy.dot(make60(1.5, 1), u)
        atoms = [['C', c] for c in coords]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'Ih')

        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(gpname, 'Ci')
        self.assertTrue(geom.check_symm('Ci', atoms))
        self.assertEqual(geom.symm_identical_atoms(gpname, atoms),
                         [[0, 55], [1, 56], [2, 57], [3, 58], [4, 59],
                          [5, 30], [6, 31], [7, 32], [8, 33], [9, 34],
                          [10, 35], [11, 36], [12, 37], [13, 38], [14, 39],
                          [15, 40], [16, 41], [17, 42], [18, 43], [19, 44],
                          [20, 45], [21, 46], [22, 47], [23, 48], [24, 49],
                          [25, 50], [26, 51], [27, 52], [28, 53], [29, 54]])

    def test_ih2(self):
        coords1 = numpy.dot(make60(1.5, 3), u)
        coords2 = numpy.dot(make12(1.1), u)
        atoms = [['C', c] for c in coords1] + [['C', c] for c in coords2]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'Ih')

        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(gpname, 'Ci')
        self.assertTrue(geom.check_symm('Ci', atoms))

    def test_ih3(self):
        coords1 = numpy.dot(make20(1.5), u)
        atoms = [['C', c] for c in coords1]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'Ih')

        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(gpname, 'Ci')
        self.assertTrue(geom.check_symm('Ci', atoms))

    def test_ih4(self):
        coords1 = make12(1.5)
        atoms = [['C', c] for c in coords1]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'Ih')

        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(gpname, 'Ci')
        self.assertTrue(geom.check_symm('Ci', atoms))

    def test_d5d_1(self):
        coords1 = numpy.dot(make20(2.0), u)
        coords2 = numpy.dot(make12(1.1), u)
        atoms = [['C', c] for c in coords1] + [['C', c] for c in coords2]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'D5d')

    def test_s10(self):
        numpy.random.seed(19)
        rotz = numpy.eye(3)
        rotz[:2,:2] = numpy.linalg.svd(numpy.random.random((2,2)))[0]
        coords1 = numpy.dot(make60(1.5, 3.0), u)
        coords2 = reduce(numpy.dot, (make20(1.1), rotz, u))
        atoms = [['C', c] for c in coords1] + [['C', c] for c in coords2]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'S10')

    def test_oh1(self):
        coords1 = numpy.dot(make6(1.5), u)
        atoms = [['C', c] for c in coords1]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'Oh')

        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(gpname, 'D2h')
        self.assertTrue(geom.check_symm('D2h', atoms))

    def test_oh2(self):
        coords1 = numpy.dot(make8(1.5), u)
        atoms = [['C', c] for c in coords1]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'Oh')

        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(gpname, 'D2h')
        self.assertTrue(geom.check_symm('D2h', atoms))

    def test_oh3(self):
        coords1 = numpy.dot(make8(1.5), u)
        coords2 = numpy.dot(make6(1.5), u)
        atoms = [['C', c] for c in coords1] + [['C', c] for c in coords2]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'Oh')

        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(gpname, 'D2h')
        self.assertTrue(geom.check_symm('D2h', atoms))

    def test_c4h(self):
        coords1 = numpy.dot(make8(1.5), u)
        coords2 = numpy.dot(make6(1.5).dot(random_rotz()), u)
        atoms = [['C', c] for c in coords1] + [['C', c] for c in coords2]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'C4h')

    def test_c2h(self):
        coords1 = numpy.dot(make8(2.5), u)
        coords2 = numpy.dot(make20(1.2), u)
        atoms = [['C', c] for c in numpy.vstack((coords1,coords2))]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'C2h')

    def test_c1(self):
        coords1 = numpy.dot(make4(2.5), u)
        coords2 = make20(1.2)
        atoms = [['C', c] for c in numpy.vstack((coords1,coords2))]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'C1')

    def test_c2(self):
        coords1 = numpy.dot(make4(2.5), 1)
        coords2 = make12(1.2)
        axes = geom._make_axes(coords2[1]-coords2[0], coords2[2])
        coords2 = numpy.dot(coords2, axes.T)
        atoms = [['C', c] for c in numpy.vstack((coords1,coords2))]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'C2')

    def test_ci(self):
        coords1 = numpy.dot(make8(2.5), u)
        coords2 = numpy.dot(numpy.dot(make20(1.2), random_rotz()), u)
        atoms = [['C', c] for c in numpy.vstack((coords1,coords2))]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'Ci')

    def test_cs(self):
        coords1 = make4(2.5)
        axes = geom._make_axes(coords1[1]-coords1[0], coords1[2])
        coords1 = numpy.dot(coords1, axes.T)
        coords2 = make12(1.2)
        axes = geom._make_axes(coords2[1]-coords2[0], coords2[2])
        coords2 = numpy.dot(coords2, axes.T)
        atoms = [['C', c] for c in numpy.vstack((coords1,coords2))]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'Cs')

        numpy.random.seed(1)
        c0 = numpy.random.random((4,3))
        c0[:,1] *= .5
        c1 = c0.copy()
        c1[:,1] *= -1
        atoms = [['C', c] for c in numpy.vstack((c0,c1))]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'Cs')

    def test_td1(self):
        coords1 = numpy.dot(make4(1.5), u)
        atoms = [['C', c] for c in coords1]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'Td')
        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(gpname, 'D2')
        self.assertTrue(geom.check_symm('D2', atoms))
        self.assertEqual(geom.symm_identical_atoms(gpname, atoms),
                         [[0, 1, 2, 3]])

    def test_td2(self):
        coords1 = numpy.dot(make4(1.5), u)
        coords2 = numpy.dot(make4(1.9), u)
        atoms = [['C', c] for c in coords1] + [['C', c] for c in coords2]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'Td')

        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(gpname, 'D2')
        self.assertTrue(geom.check_symm('D2', atoms))
        self.assertEqual(geom.symm_identical_atoms(gpname, atoms),
                         [[0, 1, 2, 3], [4, 5, 6, 7]])

    def test_td_subgroup_c2v(self):
        atoms = [['C', (0, 0, 0)],
                 ['H', (0, 0, 1)],
                 ['H', (0, 0.9428090415820634, -0.3333333333333333)],
                 ['H', ( 0.8164965809277259, -0.4714045207910317, -0.3333333333333333)],
                 ['H', (-0.8164965809277259, -0.4714045207910317, -0.3333333333333333)],
                ]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'Td')

        gpname, axes = geom.as_subgroup(gpname, axes, subgroup='C2v')
        z_ref = numpy.array([.5**.5, -6**-.5, 3**-.5])
        self.assertAlmostEqual(abs(z_ref - axes[2]).max(), 0, 12)

    def test_c3v(self):
        coords1 = numpy.dot(make4(1.5), u)
        coords2 = numpy.dot(make4(1.9), u)
        atoms = [['C', c] for c in coords1] + [['C', c] for c in coords2]
        atoms[2][0] = 'C1'
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'C3v')

        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(gpname, 'Cs')
        self.assertTrue(geom.check_symm('Cs', atoms))
        self.assertEqual(geom.symm_identical_atoms(gpname, atoms),
                         [[0, 1], [2], [3], [4, 5], [6], [7]])

    def test_c3v_1(self):
        mol = gto.M(atom='''
C   0.948065  -0.081406  -0.007893
C   0.462608  -0.144439   1.364854
N   0.077738  -0.194439   2.453356
H   0.591046   0.830035  -0.495369
H   0.591062  -0.944369  -0.576807
H   2.041481  -0.080642  -0.024174''')
        gpname, orig, axes = geom.detect_symm(mol._atom)
        self.assertEqual(gpname, 'C1')

        with lib.temporary_env(geom, TOLERANCE=1e-3):
            gpname, orig, axes = geom.detect_symm(mol._atom)
        self.assertEqual(gpname, 'C3v')

    def test_t(self):
        atoms = [['C', ( 1.0   ,-1.0   , 1.0   )],
                 ['O', ( 1.0-.1,-1.0+.2, 1.0   )],
                 ['O', ( 1.0   ,-1.0+.1, 1.0-.2)],
                 ['O', ( 1.0-.2,-1.0   , 1.0-.1)],
                 ['C', (-1.0   , 1.0   , 1.0   )],
                 ['O', (-1.0+.1, 1.0-.2, 1.0   )],
                 ['O', (-1.0   , 1.0-.1, 1.0-.2)],
                 ['O', (-1.0+.2, 1.0   , 1.0-.1)],
                 ['C', ( 1.0   , 1.0   ,-1.0   )],
                 ['O', ( 1.0-.2, 1.0   ,-1.0+.1)],
                 ['O', ( 1.0   , 1.0-.1,-1.0+.2)],
                 ['O', ( 1.0-.1, 1.0-.2,-1.0   )],
                 ['C', (-1.0   ,-1.0   ,-1.0   )],
                 ['O', (-1.0   ,-1.0+.1,-1.0+.2)],
                 ['O', (-1.0+.2,-1.0   ,-1.0+.1)],
                 ['O', (-1.0+.1,-1.0+.2,-1.0   )]]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'T')

        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(gpname, 'D2')
        self.assertTrue(geom.check_symm('D2', atoms))
        self.assertEqual(geom.symm_identical_atoms(gpname, atoms),
                         [[0, 4, 8, 12], [1, 5, 11, 15],
                          [2, 6, 10, 13], [3, 7, 9, 14]])

    def test_th(self):
        atoms = [['C', ( 1.0   ,-1.0   , 1.0   )],
                 ['O', ( 1.0-.1,-1.0+.2, 1.0   )],
                 ['O', ( 1.0   ,-1.0+.1, 1.0-.2)],
                 ['O', ( 1.0-.2,-1.0   , 1.0-.1)],
                 ['C', ( 1.0   , 1.0   , 1.0   )],
                 ['O', ( 1.0-.1, 1.0-.2, 1.0   )],
                 ['O', ( 1.0   , 1.0-.1, 1.0-.2)],
                 ['O', ( 1.0-.2, 1.0   , 1.0-.1)],
                 ['C', (-1.0   , 1.0   , 1.0   )],
                 ['O', (-1.0+.1, 1.0-.2, 1.0   )],
                 ['O', (-1.0   , 1.0-.1, 1.0-.2)],
                 ['O', (-1.0+.2, 1.0   , 1.0-.1)],
                 ['C', (-1.0   ,-1.0   , 1.0   )],
                 ['O', (-1.0+.1,-1.0+.2, 1.0   )],
                 ['O', (-1.0   ,-1.0+.1, 1.0-.2)],
                 ['O', (-1.0+.2,-1.0   , 1.0-.1)],
                 ['C', ( 1.0   ,-1.0   ,-1.0   )],
                 ['O', ( 1.0-.2,-1.0   ,-1.0+.1)],
                 ['O', ( 1.0   ,-1.0+.1,-1.0+.2)],
                 ['O', ( 1.0-.1,-1.0+.2,-1.0   )],
                 ['C', ( 1.0   , 1.0   ,-1.0   )],
                 ['O', ( 1.0-.2, 1.0   ,-1.0+.1)],
                 ['O', ( 1.0   , 1.0-.1,-1.0+.2)],
                 ['O', ( 1.0-.1, 1.0-.2,-1.0   )],
                 ['C', (-1.0   , 1.0   ,-1.0   )],
                 ['O', (-1.0+.2, 1.0   ,-1.0+.1)],
                 ['O', (-1.0   , 1.0-.1,-1.0+.2)],
                 ['O', (-1.0+.1, 1.0-.2,-1.0   )],
                 ['C', (-1.0   ,-1.0   ,-1.0   )],
                 ['O', (-1.0   ,-1.0+.1,-1.0+.2)],
                 ['O', (-1.0+.2,-1.0   ,-1.0+.1)],
                 ['O', (-1.0+.1,-1.0+.2,-1.0   )]]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'Th')

        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(gpname, 'D2')
        self.assertTrue(geom.check_symm('D2', atoms))
        self.assertEqual(geom.symm_identical_atoms(gpname, atoms),
                         [[0, 8, 20, 28], [1, 9, 23, 31],
                          [2, 10, 22, 29], [3, 11, 21, 30],
                          [4, 12, 16, 24], [5, 13, 19, 27],
                          [6, 14, 18, 26], [7, 15, 17, 25]])

    def test_s4_1(self):
        coords1 = numpy.dot(make4(1.5), u)
        coords2 = numpy.dot(numpy.dot(make4(2.4), random_rotz()), u)
        atoms = [['C', c] for c in coords1] + [['C', c] for c in coords2]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'S4')

    def test_Dooh(self):
        atoms = [['H', (0,0,0)], ['H', (0,0,-1)], ['H1', (0,0,1)]]
        basis = {'H':'sto3g'}
        gpname, orig, axes = geom.detect_symm(atoms, basis)
        self.assertEqual(gpname, 'Dooh')
        self.assertEqual(geom.symm_identical_atoms(gpname, atoms),
                         [[0], [1,2]])

        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(gpname, 'Dooh')
        self.assertTrue(geom.check_symm('Dooh', atoms, basis))
        self.assertTrue(geom.check_symm('D2h', atoms, basis))
        self.assertEqual(geom.symm_identical_atoms(gpname, atoms),
                         [[0], [1,2]])

        atoms = [['H', (0,0,0)], ['H', (1,0,0)]]
        self.assertFalse(geom.check_symm('Dooh', atoms))

    def test_Coov(self):
        atoms = [['H', (0,0,0)], ['H', (0,0,-1)], ['H1', (0,0,1)]]
        basis = {'H':'sto3g', 'H1':'6-31g'}
        gpname, orig, axes = geom.detect_symm(atoms, basis)
        self.assertEqual(gpname, 'Coov')
        self.assertEqual(geom.symm_identical_atoms(gpname, atoms),
                         [[0], [1] ,[2]])

        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(gpname, 'Coov')
        self.assertTrue(geom.check_symm('Coov', atoms, basis))
        self.assertTrue(geom.check_symm('C2v', atoms, basis))
        self.assertEqual(geom.symm_identical_atoms(gpname, atoms),
                         [[0], [1], [2]])

    def test_d5(self):
        coord1 = ring(5)
        coord2 = ring(5, .1)
        coord1[:,2] = 1
        coord2[:,2] =-1
        atoms = [['H', c] for c in numpy.vstack((coord1,coord2))]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'D5')

        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(gpname, 'C2')
        self.assertTrue(geom.check_symm('C2', atoms))
        self.assertEqual(geom.symm_identical_atoms(gpname, atoms),
                         [[0, 5], [1, 9], [2, 8], [3, 7], [4, 6]])

    def test_d5d(self):
        coord1 = ring(5)
        coord2 = ring(5, numpy.pi/5)
        coord1[:,2] = 1
        coord2[:,2] =-1
        atoms = [['H', c] for c in numpy.vstack((coord1,coord2))]
        gpname, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(gpname, 'D5d')

        gpname, axes = geom.subgroup(gpname, axes)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(gpname, 'C2h')
        self.assertTrue(geom.check_symm('C2h', atoms))
        self.assertEqual(geom.symm_identical_atoms(gpname, atoms),
                         [[0, 3, 5, 7], [1, 2, 8, 9], [4, 6]])

    def test_detect_symm_c2v(self):
        atoms = [['H' , (1., 0., 2.)],
                 ['He', (0., 1., 0.)],
                 ['H' , (-2.,0.,-1.)],
                 ['He', (0.,-1., 0.)]]
        l, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(l, 'C2v')
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertTrue(geom.check_symm('C2v', atoms))
        self.assertEqual(geom.symm_identical_atoms(l,atoms), [[0,2],[1,3]])

    def test_detect_symm_d2h_a(self):
        atoms = [['He', (0., 1., 0.)],
                 ['H' , (1., 0., 0.)],
                 ['H' , (-1.,0., 0.)],
                 ['He', (0.,-1., 0.)]]
        l, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(l, 'D2h')
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertTrue(geom.check_symm('D2h', atoms))
        self.assertEqual(geom.symm_identical_atoms(l,atoms),
                         [[0, 3], [1, 2]])

    def test_detect_symm_d2h_b(self):
        atoms = [['H' , (1., 0., 2.)],
                 ['He', (0., 1., 0.)],
                 ['H' , (-1.,0.,-2.)],
                 ['He', (0.,-1., 0.)]]
        l, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(l, 'D2h')
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertTrue(geom.check_symm('D2h', atoms))
        self.assertEqual(geom.symm_identical_atoms(l,atoms), [[0,2],[1,3]])

    def test_detect_symm_c2h_a(self):
        atoms = [['H' , (1., 0., 2.)],
                 ['He', (0., 1.,-1.)],
                 ['H' , (-1.,0.,-2.)],
                 ['He', (0.,-1., 1.)]]
        l, orig, axes = geom.detect_symm(atoms)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'C2h')
        self.assertEqual(geom.symm_identical_atoms(l,atoms), [[0,2],[1,3]])
        self.assertTrue(geom.check_symm('C2h', atoms))

    def test_detect_symm_c2h(self):
        atoms = [['H' , (1., 0., 2.)],
                 ['He', (0., 1., 0.)],
                 ['H' , (1., 0., 0.)],
                 ['H' , (-1.,0., 0.)],
                 ['H' , (-1.,0.,-2.)],
                 ['He', (0.,-1., 0.)]]
        l, orig, axes = geom.detect_symm(atoms)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'C2h')
        self.assertEqual(geom.symm_identical_atoms(l,atoms), [[0,4],[1,5],[2,3]])
        self.assertTrue(geom.check_symm('C2h', atoms))

        atoms = [['H' , (1., 0., 1.)],
                 ['H' , (1., 0.,-1.)],
                 ['He', (0., 0., 2.)],
                 ['He', (2., 0.,-2.)],
                 ['Li', (1., 1., 0.)],
                 ['Li', (1.,-1., 0.)]]
        l, orig, axes = geom.detect_symm(atoms)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'C2h')
        self.assertEqual(geom.symm_identical_atoms(l,atoms),
                         [[0, 1], [2, 3], [4, 5]])
        self.assertTrue(geom.check_symm('C2h', atoms))

    def test_detect_symm_d2_a(self):
        atoms = [['H' , (1., 0., 1.)],
                 ['H' , (1., 0.,-1.)],
                 ['He', (0., 0., 2.)],
                 ['He', (2., 0., 2.)],
                 ['He', (1., 1.,-2.)],
                 ['He', (1.,-1.,-2.)]]
        l, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(l, 'D2d')
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertTrue(geom.check_symm('D2', atoms))
        self.assertEqual(geom.symm_identical_atoms('D2', atoms),
                         [[0, 1], [2, 3, 4, 5]])

    def test_detect_symm_d2_b(self):
        s2 = numpy.sqrt(.5)
        atoms = [['C', (0., 0., 1.)],
                 ['C', (0., 0.,-1.)],
                 ['H', ( 1, 0., 2.)],
                 ['H', (-1, 0., 2.)],
                 ['H', ( s2, s2,-2.)],
                 ['H', (-s2,-s2,-2.)]]
        l, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(l, 'D2')
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertTrue(geom.check_symm('D2', atoms))
        self.assertEqual(geom.symm_identical_atoms(l,atoms),
                         [[0, 1], [2, 3, 4, 5]])

    def test_detect_symm_s4(self):
        atoms = [['H', (-1,-1.,-2.)],
                 ['H', ( 1, 1.,-2.)],
                 ['C', (-.9,-1.,-2.)],
                 ['C', (.9, 1.,-2.)],
                 ['H', ( 1,-1., 2.)],
                 ['H', (-1, 1., 2.)],
                 ['C', ( 1,-.9, 2.)],
                 ['C', (-1, .9, 2.)],]
        l, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(l, 'S4')
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertTrue(geom.check_symm('C2', atoms))
        self.assertEqual(geom.symm_identical_atoms('C2',atoms),
                         [[0, 1], [2, 3], [4, 5], [6, 7]])

    def test_detect_symm_ci(self):
        atoms = [['H' , ( 1., 0., 0.)],
                 ['He', ( 0., 1., 0.)],
                 ['Li', ( 0., 0., 1.)],
                 ['Be', ( .5, .5, .5)],
                 ['H' , (-1., 0., 0.)],
                 ['He', ( 0.,-1., 0.)],
                 ['Li', ( 0., 0.,-1.)],
                 ['Be', (-.5,-.5,-.5)]]
        l, orig, axes = geom.detect_symm(atoms)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'Ci')
        self.assertTrue(geom.check_symm('Ci', atoms))
        self.assertEqual(geom.symm_identical_atoms(l,atoms),
                         [[0, 4], [1, 5], [2, 6], [3, 7]])

    def test_detect_symm_cs1(self):
        atoms = [['H' , (1., 2., 0.)],
                 ['He', (1., 0., 0.)],
                 ['Li', (2.,-1., 0.)],
                 ['Be', (0., 1., 0.)]]
        l, orig, axes = geom.detect_symm(atoms)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'Cs')
        self.assertTrue(geom.check_symm('Cs', atoms))
        self.assertEqual(geom.symm_identical_atoms(l,atoms),
                         [[0], [1], [2], [3]])

    def test_detect_symm_cs2(self):
        atoms = [['H' , (0., 1., 2.)],
                 ['He', (0., 1., 0.)],
                 ['Li', (0., 2.,-1.)],
                 ['Be', (0., 0., 1.)],
                 ['S' , (-3, 1., .5)],
                 ['S' , ( 3, 1., .5)]]
        coord = numpy.dot([a[1] for a in atoms], u)
        atoms = [[atoms[i][0], c] for i,c in enumerate(coord)]
        l, orig, axes = geom.detect_symm(atoms)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'Cs')
        self.assertTrue(geom.check_symm('Cs', atoms))
        self.assertEqual(geom.symm_identical_atoms(l,atoms),
                         [[0], [1], [2], [3], [4, 5]])

    def test_detect_symm_cs3(self):
        atoms = [['H' , ( 2.,1., 0.)],
                 ['He', ( 0.,1., 0.)],
                 ['Li', (-1.,2., 0.)],
                 ['Be', ( 1.,0., 0.)],
                 ['S' , ( .5,1., -3)],
                 ['S' , ( .5,1.,  3)]]
        coord = numpy.dot([a[1] for a in atoms], u)
        atoms = [[atoms[i][0], c] for i,c in enumerate(coord)]
        l, orig, axes = geom.detect_symm(atoms)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'Cs')
        self.assertTrue(geom.check_symm('Cs', atoms))
        self.assertEqual(geom.symm_identical_atoms(l,atoms),
                         [[0], [1], [2], [3], [4, 5]])

    def test_detect_symm_c1(self):
        atoms = [['H' , ( 1., 0., 0.)],
                 ['He', ( 0., 1., 0.)],
                 ['Li', ( 0., 0., 1.)],
                 ['Be', ( .5, .5, .5)]]
        l, orig, axes = geom.detect_symm(atoms)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'C1')
        self.assertTrue(geom.check_symm('C1', atoms))
        self.assertEqual(geom.symm_identical_atoms(l,atoms),
                         [[0], [1], [2], [3]])

    def test_detect_symm_c2(self):
        atoms = [['H' , ( 1., 0., 1.)],
                 ['H' , ( 1., 0.,-1.)],
                 ['He', ( 0.,-3., 2.)],
                 ['He', ( 0., 3.,-2.)]]
        l, orig, axes = geom.detect_symm(atoms)
        atoms = geom.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'C2')
        self.assertTrue(geom.check_symm('C2', atoms))
        self.assertEqual(geom.symm_identical_atoms(l,atoms), [[0,1],[2,3]])

    def test_detect_symm_d3d(self):
        atoms = [
            ['C', ( 1.25740, -0.72596, -0.25666)],
            ['C', ( 1.25740,  0.72596,  0.25666)],
            ['C', ( 0.00000,  1.45192, -0.25666)],
            ['C', (-1.25740,  0.72596,  0.25666)],
            ['C', (-1.25740, -0.72596, -0.25666)],
            ['C', ( 0.00000, -1.45192,  0.25666)],
            ['H', ( 2.04168, -1.17876,  0.05942)],
            ['H', ( 1.24249, -0.71735, -1.20798)],
            ['H', ( 2.04168,  1.17876, -0.05942)],
            ['H', ( 1.24249,  0.71735,  1.20798)],
            ['H', ( 0.00000,  1.43470, -1.20798)],
            ['H', ( 0.00000,  2.35753,  0.05942)],
            ['H', (-2.04168,  1.17876, -0.05942)],
            ['H', (-1.24249,  0.71735,  1.20798)],
            ['H', (-1.24249, -0.71735, -1.20798)],
            ['H', (-2.04168, -1.17876,  0.05942)],
            ['H', ( 0.00000, -1.43470,  1.20798)],
            ['H', ( 0.00000, -2.35753, -0.05942)], ]
        with lib.temporary_env(geom, TOLERANCE=1e-4):
            l, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(l, 'D3d')

    def test_quasi_c2v(self):
        atoms = [
            ['Fe', ( 0.0000000000,   0.0055197721,   0.0055197721)],
            ['O' , (-1.3265475500,   0.0000000000,  -0.9445024777)],
            ['O' , ( 1.3265475500,   0.0000000000,  -0.9445024777)],
            ['O' , ( 0.0000000000,  -1.3265374484,   0.9444796669)],
            ['O' , ( 0.0000000000,   1.3265374484,   0.9444796669)],]
        l, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(l, 'Cs')
        with lib.temporary_env(geom, TOLERANCE=1e-2):
            l, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(l, 'C2v')

        with lib.temporary_env(geom, TOLERANCE=1e-1):
            l, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(l, 'Td')

    def test_as_subgroup(self):
        axes = numpy.eye(3)
        g, ax = symm.as_subgroup('D2d', axes)
        self.assertEqual(g, 'D2')
        self.assertAlmostEqual(abs(ax-numpy.eye(3)).max(), 0, 12)

        g, ax = symm.as_subgroup('D2d', axes, 'D2')
        self.assertEqual(g, 'D2')
        self.assertAlmostEqual(abs(ax-numpy.eye(3)).max(), 0, 12)

        g, ax = symm.as_subgroup('D2d', axes, 'C2v')
        self.assertEqual(g, 'C2v')
        self.assertAlmostEqual(ax[0,1], numpy.sqrt(.5), 9)
        self.assertAlmostEqual(ax[1,0],-numpy.sqrt(.5), 9)

        g, ax = symm.as_subgroup('C2v', axes, 'Cs')
        self.assertEqual(g, 'Cs')
        self.assertAlmostEqual(ax[2,0], 1, 9)

        g, ax = symm.as_subgroup('D6', axes)
        self.assertEqual(g, 'D2')

        g, ax = symm.as_subgroup('C6h', axes)
        self.assertEqual(g, 'C2h')

        g, ax = symm.as_subgroup('C6v', axes)
        self.assertEqual(g, 'C2v')

        g, ax = symm.as_subgroup('C6', axes)
        self.assertEqual(g, 'C2')

        g, ax = symm.as_subgroup('Dooh', axes, 'D2h')
        self.assertEqual(g, 'D2h')
        g, ax = symm.as_subgroup('Dooh', axes, 'D2')
        self.assertEqual(g, 'D2')
        g, ax = symm.as_subgroup('Dooh', axes, 'C2v')
        self.assertEqual(g, 'C2v')

        g, ax = symm.as_subgroup('Coov', axes, 'C2v')
        self.assertEqual(g, 'C2v')
        g, ax = symm.as_subgroup('Coov', axes, 'C2')
        self.assertEqual(g, 'C2')

        g, ax = symm.as_subgroup('SO3', axes, 'D2h')
        self.assertEqual(g, 'D2h')
        g, ax = symm.as_subgroup('SO3', axes, 'D2')
        self.assertEqual(g, 'D2')
        g, ax = symm.as_subgroup('SO3', axes, 'C2v')
        self.assertEqual(g, 'C2v')

        g, ax = symm.as_subgroup('SO3', axes, 'Dooh')
        self.assertEqual(g, 'Dooh')
        g, ax = symm.as_subgroup('SO3', axes, 'Coov')
        self.assertEqual(g, 'Coov')

    def test_ghost(self):
        atoms = [
            ['Fe'  , ( 0.0,   0.0,   0.0)],
            ['O'   , (-1.3,   0.0,   0.0)],
            ['GHOST-O' , ( 1.3,   0.0,   0.0)],
            ['GHOST-O' , ( 0.0,  -1.3,   0.0)],
            ['O'   , ( 0.0,   1.3,   0.0)],]
        l, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(l, 'C2v')
        self.assertAlmostEqual(axes[2,0]*axes[2,1], -.5, 9)

        atoms = [
            ['Fe'  , ( 0.0,   0.0,   0.0)],
            ['O'   , (-1.3,   0.0,   0.0)],
            ['XO'  , ( 1.3,   0.0,   0.0)],
            ['GHOSTO' , ( 0.0,  -1.3,   0.0)],
            ['O'   , ( 0.0,   1.3,   0.0)],]
        l, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(l, 'C2v')
        self.assertAlmostEqual(axes[2,0]*axes[2,1], -.5, 9)

        atoms = [
            ['Fe'  , ( 0.0,   0.0,   0.0)],
            ['O'   , (-1.3,   0.0,   0.0)],
            ['X' , ( 1.3,   0.0,   0.0)],
            ['X' , ( 0.0,  -1.3,   0.0)],
            ['O'   , ( 0.0,   1.3,   0.0)],]
        l, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(l, 'C2v')
        self.assertAlmostEqual(axes[2,0]*axes[2,1], -.5, 9)

    def test_sort_coords(self):
        c = numpy.random.random((5,3))
        c0 = symm.sort_coords(c)
        idx = symm.argsort_coords(c)
        self.assertAlmostEqual(abs(c[idx] - c0).max(), 0, 9)

    def test_c2v_shifted(self):
        atoms = [
            ["C", [1.0000000, 0.0000000, 0.1238210]],
            ["H", [1.0000000, 0.9620540, -0.3714630]],
            ["H", [1.0000000, -0.9620540, -0.3714630]],
        ]
        l, orig, axes = geom.detect_symm(atoms)
        self.assertEqual(l, 'C2v')
        self.assertAlmostEqual(abs(axes - numpy.diag(axes.diagonal())).max(), 0, 9)

    def test_geometry_small_discrepancy(self):
        # issue 2713
        mol = gto.M(
            atom='''
                O        0.000000    0.000000    0.900000
                H        0.000000    0.000000    0.000000
                H        0.914864    0.000000    1.249646
                H       -0.498887    0.766869    1.249646''',
            charge=1, symmetry=True)
        self.assertEqual(mol.groupname, 'Cs')

def ring(n, start=0):
    r = 1. / numpy.sin(numpy.pi/n)
    coord = []
    for i in range(n):
        theta = i * (2*numpy.pi/n)
        coord.append([r*numpy.cos(theta+start), r*numpy.sin(theta+start), 0])
    return numpy.array(coord)

def ringhat(n, u):
    atoms = [['H', c] for c in ring(n)] \
          + [['C', c] for c in ring(n, .1)] \
          + [['N', [0,0, 1.3]],
             ['N', [0,0,-1.3]]]
    c = numpy.dot([a[1] for a in atoms], u)
    return [[atoms[i][0], c[i]] for i in range(len(atoms))]

def rotmatz(ang):
    c = numpy.cos(ang)
    s = numpy.sin(ang)
    return numpy.array((( c, s, 0),
                        (-s, c, 0),
                        ( 0, 0, 1),))
def rotmaty(ang):
    c = numpy.cos(ang)
    s = numpy.sin(ang)
    return numpy.array((( c, 0, s),
                        ( 0, 1, 0),
                        (-s, 0, c),))

def r2edge(ang, r):
    return 2*r*numpy.sin(ang/2)


def make60(b5, b6):
    theta1 = numpy.arccos(1/numpy.sqrt(5))
    theta2 = (numpy.pi - theta1) * .5
    r = (b5*2+b6)/2/numpy.sin(theta1/2)
    rot72 = rotmatz(numpy.pi*2/5)
    s1 = numpy.sin(theta1)
    c1 = numpy.cos(theta1)
    s2 = numpy.sin(theta2)
    c2 = numpy.cos(theta2)
    p1 = numpy.array(( s2*b5,  0, r-c2*b5))
    p9 = numpy.array((-s2*b5,  0,-r+c2*b5))
    p2 = numpy.array(( s2*(b5+b6),  0, r-c2*(b5+b6)))
    rot1 = reduce(numpy.dot, (rotmaty(theta1), rot72, rotmaty(-theta1)))
    p2s = []
    for i in range(5):
        p2s.append(p2)
        p2 = numpy.dot(p2, rot1)

    coord = []
    for i in range(5):
        coord.append(p1)
        p1 = numpy.dot(p1, rot72)
    for pj in p2s:
        pi = pj
        for i in range(5):
            coord.append(pi)
            pi = numpy.dot(pi, rot72)
    for pj in p2s:
        pi = pj
        for i in range(5):
            coord.append(-pi)
            pi = numpy.dot(pi, rot72)
    for i in range(5):
        coord.append(p9)
        p9 = numpy.dot(p9, rot72)
    return numpy.array(coord)


def make12(b):
    theta1 = numpy.arccos(1/numpy.sqrt(5))
    theta2 = (numpy.pi - theta1) * .5
    r = b/2./numpy.sin(theta1/2)
    rot72 = rotmatz(numpy.pi*2/5)
    s1 = numpy.sin(theta1)
    c1 = numpy.cos(theta1)
    p1 = numpy.array(( s1*r,  0,  c1*r))
    p2 = numpy.array((-s1*r,  0, -c1*r))
    coord = [(  0,  0,    r)]
    for i in range(5):
        coord.append(p1)
        p1 = numpy.dot(p1, rot72)
    for i in range(5):
        coord.append(p2)
        p2 = numpy.dot(p2, rot72)
    coord.append((  0,  0,  -r))
    return numpy.array(coord)


def make20(b):
    theta1 = numpy.arccos(numpy.sqrt(5)/3)
    theta2 = numpy.arcsin(r2edge(theta1,1)/2/numpy.sin(numpy.pi/5))
    r = b/2./numpy.sin(theta1/2)
    rot72 = rotmatz(numpy.pi*2/5)
    s2 = numpy.sin(theta2)
    c2 = numpy.cos(theta2)
    s3 = numpy.sin(theta1+theta2)
    c3 = numpy.cos(theta1+theta2)
    p1 = numpy.array(( s2*r,  0,  c2*r))
    p2 = numpy.array(( s3*r,  0,  c3*r))
    p3 = numpy.array((-s3*r,  0, -c3*r))
    p4 = numpy.array((-s2*r,  0, -c2*r))
    coord = []
    for i in range(5):
        coord.append(p1)
        p1 = numpy.dot(p1, rot72)
    for i in range(5):
        coord.append(p2)
        p2 = numpy.dot(p2, rot72)
    for i in range(5):
        coord.append(p3)
        p3 = numpy.dot(p3, rot72)
    for i in range(5):
        coord.append(p4)
        p4 = numpy.dot(p4, rot72)
    return numpy.array(coord)

def make4(b):
    coord = numpy.ones((4,3)) * b*.5
    coord[1,0] = coord[1,1] = -b*.5
    coord[2,2] = coord[2,1] = -b * .5
    coord[3,0] = coord[3,2] = -b * .5
    return coord

def make6(b):
    coord = numpy.zeros((6,3))
    coord[0,0] = coord[1,1] = coord[2,2] = b * .5
    coord[3,0] = coord[4,1] = coord[5,2] =-b * .5
    return coord

def make8(b):
    coord = numpy.ones((8,3)) * b*.5
    n = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                coord[n,0] = (-1) ** i * b*.5
                coord[n,1] = (-1) ** j * b*.5
                coord[n,2] = (-1) ** k * b*.5
                n += 1
    return coord

def random_rotz(seed=19):
    numpy.random.seed(seed)
    rotz = numpy.eye(3)
    rotz[:2,:2] = numpy.linalg.svd(numpy.random.random((2,2)))[0]
    return rotz


if __name__ == "__main__":
    print("Full Tests geom")
    unittest.main()
