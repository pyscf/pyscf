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

import unittest
import numpy
from pyscf import gto
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import tools
from pyscf.pbc.scf import khf
from pyscf import lib


class KnownValues(unittest.TestCase):
    def test_coulG_ws(self):
        cell = pbcgto.Cell()
        cell.unit = 'A'
        cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
        cell.a = '''0.      1.7834  1.7834
                    1.7834  0.      1.7834
                    1.7834  1.7834  0.    '''
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.mesh = [11]*3
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()
        mf = khf.KRHF(cell, exxdiv='vcut_ws')
        mf.kpts = cell.make_kpts([2,2,2])
        coulG = tools.get_coulG(cell, mf.kpts[2], True, mf, gs=[5,5,5])
        self.assertAlmostEqual(lib.fp(coulG), 1.3245365170998518+0j, 9)

    def test_unconventional_ws_cell(self):
        cell = pbcgto.Cell()
        cell.atom = 'He'
        cell.basis = [[0, (1, 1)]]
        cell.a = '''4.3, 0.7, 1.2
                    0.4, 2.0, 0.1
                    0.5, 0  , 1.8'''
        cell.build()
        kpts = cell.make_kpts([1,1,1])
        self.assertRaises(RuntimeError, tools.precompute_exx, cell, kpts)

    def test_coulG(self):
        numpy.random.seed(19)
        kpt = numpy.random.random(3)
        cell = pbcgto.Cell()
        cell.unit = 'A'
        cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
        cell.a = numpy.array(((0.    , 1.7834, 1.7834),
                              (1.7834, 0.    , 1.7834),
                              (1.7834, 1.7834, 0.    ),)) + numpy.random.random((3,3)).T
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.mesh = [11,9,7]
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()
        coulG = tools.get_coulG(cell, kpt)
        self.assertAlmostEqual(lib.fp(coulG), 62.75448804333378, 9)

        cell.a = numpy.eye(3)
        cell.unit = 'B'
        coulG = tools.get_coulG(cell, numpy.array([0, numpy.pi, 0]))
        self.assertAlmostEqual(lib.fp(coulG), 4.6737453679713905, 9)
        coulG = tools.get_coulG(cell, numpy.array([0, numpy.pi, 0]),
                                wrap_around=False)
        self.assertAlmostEqual(lib.fp(coulG), 4.5757877990664744, 9)
        coulG = tools.get_coulG(cell, exx='ewald')
        self.assertAlmostEqual(lib.fp(coulG), 4.888843468914021, 9)

    #def test_coulG_2d(self):
    #    cell = pbcgto.Cell()
    #    cell.a = numpy.eye(3)
    #    cell.a[2] = numpy.array((0, 0, 20))
    #    cell.atom = 'He 0 0 0'
    #    cell.unit = 'B'
    #    cell.mesh = [9,9,40]
    #    cell.verbose = 5
    #    cell.dimension = 2
    #    cell.output = '/dev/null'
    #    cell.build()
    #    coulG = tools.get_coulG(cell)
    #    self.assertAlmostEqual(lib.fp(coulG), -4.7118365257800496, 9)


    def test_get_lattice_Ls(self):
        numpy.random.seed(2)
        cl1 = pbcgto.M(a = numpy.random.random((3,3))*3,
                       mesh = [3]*3,
                       atom ='''He .1 .0 .0''',
                       basis = 'ccpvdz')
        Ls = tools.get_lattice_Ls(cl1)
        self.assertEqual(Ls.shape, (2275,3))

        Ls = tools.get_lattice_Ls(cl1, rcut=0)
        self.assertEqual(Ls.shape, (1,3))

    def test_get_lattice_Ls1(self):
        cell = pbcgto.Cell()
        cell.verbose = 0
        cell.a = '''
-10.11124892   0.          10.11124892
  0.          10.11124892  10.11124892
-10.11124892  10.11124892   0.        '''
        cell.atom = '''
C   0.          0.          0.
C  13.48166522  6.74083261  6.74083261
C  15.16687337  8.42604076  8.42604076
C  13.48166522 10.11124892 10.11124892
C  15.16687337 11.79645707 11.79645707
C  10.11124892 13.48166522 10.11124892
C  11.79645707 15.16687337 11.79645707
C  13.48166522 13.48166522 13.48166522
C  15.16687337 15.16687337 15.16687337
'''
        cell.unit= 'B'
        cell.basis = 'gth-dzvp'
        cell.pseudo = 'gth-pade'
        cell.precision = 1e-10
        cell.build()
        Ls = cell.get_lattice_Ls()
        self.assertTrue(Ls.shape[0] > 140)

        S = cell.pbc_intor('int1e_ovlp')
        w, v = numpy.linalg.eigh(S)
        self.assertTrue(w.min() > 0)
        self.assertAlmostEqual(abs(S - S.T.conj()).max(), 0, 13)
        self.assertAlmostEqual(w.min(), 0.0007176363230, 8)

    def test_super_cell(self):
        numpy.random.seed(2)
        cl1 = pbcgto.M(a = numpy.random.random((3,3))*3,
                       mesh = [3]*3,
                       atom ='''He .1 .0 .0''',
                       basis = 'ccpvdz')
        _ = cl1.enuc
        ncopy = [2,3,4]
        ncell = ncopy[0]*ncopy[1]*ncopy[2]
        cl2 = tools.super_cell(cl1, ncopy)
        self.assertAlmostEqual(lib.fp(cl2.atom_coords()), -18.946080642714836, 9)
        self.assertAlmostEqual(lib.fp(cl2._bas[:,gto.ATOM_OF]), 16.515144238434807, 9)
        self.assertAlmostEqual(cl1.enuc, cl2.enuc / ncell, 9)

    def test_super_cell_with_symm(self):
        cl1 = pbcgto.M(a = 1.4 * numpy.eye(3),
                       atom ='''He .0 .0 .0''',
                       basis = 'ccpvdz',
                       space_group_symmetry=True,
                       symmorphic=False)
        self.assertEqual(cl1.lattice_symmetry.nop, 48)
        cl2 = tools.super_cell(cl1, [2,2,2])
        self.assertEqual(cl2.lattice_symmetry.nop, 48*8)

    def test_cell_plus_imgs(self):
        numpy.random.seed(2)
        cl1 = pbcgto.M(a = numpy.random.random((3,3))*3,
                       mesh = [3]*3,
                       atom ='''He .1 .0 .0''',
                       basis = 'ccpvdz')
        #self.assertEqual(list(cl1.nimgs), [8, 16, 11])
        cl2 = tools.cell_plus_imgs(cl1, [3,4,5])
        self.assertAlmostEqual(lib.fp(cl2.atom_coords()), 4.791699273649499, 9)
        self.assertAlmostEqual(lib.fp(cl2._bas[:,gto.ATOM_OF]), -681.993543446207, 9)

    def test_madelung(self):
        cell = pbcgto.Cell()
        cell.atom = 'He 0 0 0'
        cell.a = '''0.      1.7834  1.7834
                    1.7834  0.      1.7834
                    1.7834  1.7834  0.    '''
        cell.build()
        scell = tools.super_cell(cell, [2,3,5])
        mad0 = tools.madelung(scell, [0,0,0])
        kpts = cell.make_kpts([2,3,5])
        mad1 = tools.madelung(cell, kpts)
        self.assertAlmostEqual(mad0-mad1, 0, 9)

    def test_fft(self):
        n = 31
        a = numpy.random.random([2,n,n,n])
        ref = numpy.fft.fftn(a, axes=(1,2,3)).ravel()
        v = tools.fft(a, [n,n,n]).ravel()
        self.assertAlmostEqual(abs(ref-v).max(), 0, 10)

        a = numpy.random.random([2,n,n,8])
        ref = numpy.fft.fftn(a, axes=(1,2,3)).ravel()
        v = tools.fft(a, [n,n,8]).ravel()
        self.assertAlmostEqual(abs(ref-v).max(), 0, 10)

        a = numpy.random.random([2,8,n,8])
        ref = numpy.fft.fftn(a, axes=(1,2,3)).ravel()
        v = tools.fft(a, [8,n,8]).ravel()
        self.assertAlmostEqual(abs(ref-v).max(), 0, 10)

    def test_ifft(self):
        n = 31
        a = numpy.random.random([2,n,n,n])
        ref = numpy.fft.ifftn(a, axes=(1,2,3)).ravel()
        v = tools.ifft(a, [n,n,n]).ravel()
        self.assertAlmostEqual(abs(ref-v).max(), 0, 10)

        a = numpy.random.random([2,n,n,8])
        ref = numpy.fft.ifftn(a, axes=(1,2,3)).ravel()
        v = tools.ifft(a, [n,n,8]).ravel()
        self.assertAlmostEqual(abs(ref-v).max(), 0, 10)

        a = numpy.random.random([2,8,n,8])
        ref = numpy.fft.ifftn(a, axes=(1,2,3)).ravel()
        v = tools.ifft(a, [8,n,8]).ravel()
        self.assertAlmostEqual(abs(ref-v).max(), 0, 10)

    def test_mesh_to_cutoff(self):
        a = numpy.array([
            [0.  , 3.37, 3.37],
            [3.37, 0.  , 3.37],
            [3.37, 3.37, 0.  ],])

        ke = tools.mesh_to_cutoff(a, [15]*3)
        self.assertAlmostEqual(ke.min(), 42.58297736648015, 9)
        mesh = tools.cutoff_to_mesh(a, ke.min())
        self.assertAlmostEqual(abs(mesh - [15]*3).max(), 0, 9)

        a = numpy.array([
            [0.  ,10.11, 10.11],
            [3.37, 0.  , 3.37],
            [3.37, 3.37, 0.  ],])
        ke = tools.mesh_to_cutoff(a, [15]*3)
        self.assertAlmostEqual(ke.min(), 4.7314419296089065, 9)
        mesh = tools.cutoff_to_mesh(a, ke.min())
        k1 = tools.mesh_to_cutoff(a, mesh)
        self.assertAlmostEqual(ke.min(), k1.min(), 9)


if __name__ == '__main__':
    print("Full Tests for pbc.tools")
    unittest.main()
