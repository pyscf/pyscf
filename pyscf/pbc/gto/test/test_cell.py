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
import ctypes
import numpy
import numpy as np
from pyscf.pbc import gto as pgto


L = 1.5
n = 41
cl = pgto.Cell()
cl.build(
    a = [[L,0,0], [0,L,0], [0,0,L]],
    mesh = [n,n,n],
    atom = 'He %f %f %f' % ((L/2.,)*3),
    basis = 'ccpvdz')

numpy.random.seed(1)
cl1 = pgto.Cell()
cl1.build(a = numpy.random.random((3,3)).T,
          precision = 1e-9,
          mesh = [n,n,n],
          atom ='''He .1 .0 .0
                   He .5 .1 .0
                   He .0 .5 .0
                   He .1 .3 .2''',
          basis = 'ccpvdz')

def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(a.ravel(), w)

class KnownValues(unittest.TestCase):
    def test_nimgs(self):
        self.assertTrue(list(cl.get_nimgs(9e-1)), [1,1,1])
        self.assertTrue(list(cl.get_nimgs(1e-2)), [2,2,2])
        self.assertTrue(list(cl.get_nimgs(1e-4)), [3,3,3])
        self.assertTrue(list(cl.get_nimgs(1e-6)), [4,4,4])
        self.assertTrue(list(cl.get_nimgs(1e-9)), [5,5,5])

    def test_Gv(self):
        a = cl1.get_Gv()
        self.assertAlmostEqual(finger(a), -99.791927068519939, 10)

    def test_SI(self):
        a = cl1.get_SI()
        self.assertAlmostEqual(finger(a), (16.506917823339265+1.6393578329869585j), 10)

    def test_mixed_basis(self):
        cl = pgto.Cell()
        cl.build(
            a = [[L,0,0], [0,L,0], [0,0,L]],
            mesh = [n,n,n],
            atom = 'C1 %f %f %f; C2 %f %f %f' % ((L/2.,)*6),
            basis = {'C1':'ccpvdz', 'C2':'gthdzv'})

    def test_dumps_loads(self):
        cl1.loads(cl1.dumps())

    def test_get_lattice_Ls(self):
        #self.assertEqual(cl1.get_lattice_Ls([0,0,0]).shape, (1  , 3))
        #self.assertEqual(cl1.get_lattice_Ls([1,1,1]).shape, (13 , 3))
        #self.assertEqual(cl1.get_lattice_Ls([2,2,2]).shape, (57 , 3))
        #self.assertEqual(cl1.get_lattice_Ls([3,3,3]).shape, (137, 3))
        #self.assertEqual(cl1.get_lattice_Ls([4,4,4]).shape, (281, 3))
        #self.assertEqual(cl1.get_lattice_Ls([5,5,5]).shape, (493, 3))

        cell = pgto.M(atom = '''
        C 0.000000000000  0.000000000000  0.000000000000
        C 1.685068664391  1.685068664391  1.685068664391''',
        unit='B',
        basis = 'gth-dzvp',
        pseudo = 'gth-pade',
        a = '''
        0.000000000  3.370137329  3.370137329
        3.370137329  0.000000000  3.370137329
        3.370137329  3.370137329  0.000000000''',
        mesh = [15]*3)
        rcut = max([cell.bas_rcut(ib, 1e-8) for ib in range(cell.nbas)])
        self.assertEqual(cell.get_lattice_Ls(rcut=rcut).shape, (1097, 3))
        rcut = max([cell.bas_rcut(ib, 1e-9) for ib in range(cell.nbas)])
        self.assertEqual(cell.get_lattice_Ls(rcut=rcut).shape, (1241, 3))

    def test_ewald(self):
        cell = pgto.Cell()
        cell.unit = 'B'
        Lx = Ly = Lz = 5.
        cell.a = numpy.diag([Lx,Ly,Lz])
        cell.mesh = numpy.array([41]*3)
        cell.atom = [['He', (2, 0.5*Ly, 0.5*Lz)],
                     ['He', (3, 0.5*Ly, 0.5*Lz)]]
        cell.basis = {'He': [[0, (1.0, 1.0)]]}
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()

        ew_cut = (20,20,20)
        self.assertAlmostEqual(cell.ewald(.05, 100), -0.468640671931, 9)
        self.assertAlmostEqual(cell.ewald(0.1, 100), -0.468640671931, 9)
        self.assertAlmostEqual(cell.ewald(0.2, 100), -0.468640671931, 9)
        self.assertAlmostEqual(cell.ewald(1  , 100), -0.468640671931, 9)

        def check(precision, eta_ref, ewald_ref):
            ew_eta0, ew_cut0 = cell.get_ewald_params(precision, mesh=[41]*3)
            self.assertAlmostEqual(ew_eta0, eta_ref)
            self.assertAlmostEqual(cell.ewald(ew_eta0, ew_cut0), ewald_ref, 9)
        check(0.001, 3.15273336976, -0.468640679947)
        check(1e-05, 2.77596886114, -0.468640671968)
        check(1e-07, 2.50838938833, -0.468640671931)
        check(1e-09, 2.30575091612, -0.468640671931)

        cell = pgto.Cell()
        numpy.random.seed(10)
        cell.a = numpy.random.random((3,3))*2 + numpy.eye(3) * 2
        cell.mesh = [41]*3
        cell.atom = [['He', (1, 1, 2)],
                     ['He', (3, 2, 1)]]
        cell.basis = {'He': [[0, (1.0, 1.0)]]}
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()

        self.assertAlmostEqual(cell.ewald(1, 20), -2.3711356723457615, 9)
        self.assertAlmostEqual(cell.ewald(2, 10), -2.3711356723457615, 9)
        self.assertAlmostEqual(cell.ewald(2,  5), -2.3711356723457615, 9)

    def test_ewald_2d(self):
        cell = pgto.Cell()
        cell.a = numpy.eye(3) * 4
        cell.atom = 'He 0 0 0; He 0 1 1'
        cell.unit = 'B'
        cell.mesh = [9,9,60]
        cell.verbose = 0
        cell.dimension = 2
        cell.rcut = 3.6
        cell.build()
        self.assertAlmostEqual(cell.ewald(), -2.0207698225112987, 6)

    def test_ewald_1d(self):
        cell = pgto.Cell()
        cell.a = numpy.eye(3) * 4
        cell.atom = 'He 0 0 0; He 0 1 1'
        cell.unit = 'B'
        cell.mesh = [9,60,60]
        cell.verbose = 0
        cell.dimension = 1
        cell.rcut = 3.6
        cell.build()
        self.assertAlmostEqual(cell.ewald(), 6.4055183018897317, 4)

    def test_ewald_0d(self):
        cell = pgto.Cell()
        cell.a = numpy.eye(3)
        cell.atom = 'He 0 0 0; He 0 1 1'
        cell.unit = 'B'
        cell.mesh = [60] * 3
        cell.verbose = 0
        cell.dimension = 0
        cell.build()
        eref = cell.to_mol().energy_nuc()
        self.assertAlmostEqual(cell.ewald(), eref, 2)

    def test_pbc_intor(self):
        numpy.random.seed(12)
        kpts = numpy.random.random((4,3))
        kpts[0] = 0
        self.assertEqual(list(cl1.nimgs), [32,21,19])
        s0 = cl1.pbc_intor('int1e_ovlp_sph', hermi=0, kpts=kpts)
        self.assertAlmostEqual(finger(s0[0]), 492.30658304804126, 4)
        self.assertAlmostEqual(finger(s0[1]), 37.812956255000756-28.972806230140314j, 4)
        self.assertAlmostEqual(finger(s0[2]),-26.113285893260819-34.448501789693566j, 4)
        self.assertAlmostEqual(finger(s0[3]), 186.58921213429491+123.90133823378201j, 4)

        s1 = cl1.pbc_intor('int1e_ovlp_sph', hermi=1, kpts=kpts[0])
        self.assertAlmostEqual(finger(s1), 492.30658304804126, 4)

    def test_ecp_pseudo(self):
        from pyscf.pbc.gto import ecp
        cell = pgto.M(
            a = np.eye(3)*5,
            mesh = [9]*3,
            atom = 'Cu 0 0 1; Na 0 1 0',
            ecp = {'Na':'lanl2dz'},
            pseudo = {'Cu': 'gthbp'})
        self.assertTrue(all(cell._ecpbas[:,0] == 1))

        cell = pgto.Cell()
        cell.a = numpy.eye(3) * 8
        cell.mesh = [11] * 3
        cell.atom='''Na 0. 0. 0.
                     H  0.  0.  1.'''
        cell.basis={'Na':'lanl2dz', 'H':'sto3g'}
        cell.ecp = {'Na':'lanl2dz'}
        cell.build()
        v1 = ecp.ecp_int(cell)
        mol = cell.to_mol()
        v0 = mol.intor('ECPscalar_sph')
        self.assertAlmostEqual(abs(v0 - v1).sum(), 0.029005926114411891, 8)

    def test_ecp_keyword_in_pseudo(self):
        cell = pgto.M(
            a = np.eye(3)*5,
            mesh = [9]*3,
            atom = 'S 0 0 1',
            ecp = 'lanl2dz',
            pseudo = {'O': 'gthbp', 'Cu': 'stuttgartrsc'})
        self.assertEqual(cell.ecp, 'lanl2dz')
        self.assertEqual(cell.pseudo, {'O': 'gthbp'})

        cell = pgto.M(
            a = np.eye(3)*5,
            mesh = [9]*3,
            atom = 'S 0 0 1',
            ecp = {'na': 'lanl2dz'},
            pseudo = {'O': 'gthbp', 'Cu': 'stuttgartrsc'})
        self.assertEqual(cell.ecp, {'na': 'lanl2dz', 'Cu': 'stuttgartrsc'})
        self.assertEqual(cell.pseudo, {'O': 'gthbp'})

        cell = pgto.M(
            a = np.eye(3)*5,
            mesh = [9]*3,
            atom = 'S 0 0 1',
            pseudo = {'O': 'gthbp', 'Cu': 'stuttgartrsc'})
        self.assertEqual(cell.ecp, {'Cu': 'stuttgartrsc'})
        self.assertEqual(cell.pseudo, {'O': 'gthbp'})

        cell = pgto.M(
            a = np.eye(3)*5,
            mesh = [9]*3,
            atom = 'S 0 0 1',
            ecp = {'S': 'gthbp', 'na': 'lanl2dz'},
            pseudo = {'O': 'gthbp', 'Cu': 'stuttgartrsc'})
        self.assertEqual(cell.ecp, {'na': 'lanl2dz', 'Cu': 'stuttgartrsc'})
        self.assertEqual(cell.pseudo, {'S': 'gthbp', 'O': 'gthbp'})

    def test_pseudo_suffix(self):
        cell = pgto.M(
            a = np.eye(3)*5,
            mesh = [9]*3,
            atom = 'Mg 0 0 1',
            pseudo = {'Mg': 'gth-lda'})
        self.assertEqual(cell.atom_nelec_core(0), 2)

        cell = pgto.M(
            a = np.eye(3)*5,
            mesh = [9]*3,
            atom = 'Mg 0 0 1',
            pseudo = {'Mg': 'gth-lda q2'})
        self.assertEqual(cell.atom_nelec_core(0), 10)


if __name__ == '__main__':
    print("Full Tests for pbc.gto.cell")
    unittest.main()

