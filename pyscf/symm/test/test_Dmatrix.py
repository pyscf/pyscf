#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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

from functools import reduce
import unittest
import numpy
from pyscf import gto, lib
from pyscf.symm import Dmatrix, geom


class KnownValues(unittest.TestCase):
    def test_Dmatrix(self):
        self.assertAlmostEqual(lib.finger(Dmatrix.Dmatrix(0, -.7, .5, .2)), 1, 12)
        self.assertAlmostEqual(lib.finger(Dmatrix.Dmatrix(1, -.7, .5, .2)), 0.7014811805222106, 12)
        self.assertAlmostEqual(lib.finger(Dmatrix.Dmatrix(2, -.7, .5, .2)), 1.247436140965072 , 12)
        self.assertAlmostEqual(lib.finger(Dmatrix.Dmatrix(3, -.7, .5, .2)), 0.9226598665854279, 12)
        self.assertAlmostEqual(lib.finger(Dmatrix.Dmatrix(4, -.7, .5, .2)), -0.425143083298510, 12)

    def test_real_sph_vec(self):
        c0 = c = numpy.random.random(3)

        mol1 = gto.M(atom=['H1 0 0 0', ['H2', c]],
                     basis = {'H1': [[0, (1, 1)]],
                              'H2': [[l, (1, 1)] for l in range(1,6)]})
        alpha = .2
        beta = .4
        gamma = -.3
        c1 = numpy.dot(geom.rotation_mat((0,0,1), gamma), c0)
        c1 = numpy.dot(geom.rotation_mat((0,1,0), beta), c1)
        c1 = numpy.dot(geom.rotation_mat((0,0,1), alpha), c1)
        mol2 = gto.M(atom=['H1 0 0 0', ['H2', c1]],
                     basis = {'H1': [[0, (1, 1)]],
                              'H2': [[l, (1, 1)] for l in range(1,6)]})

        for l in range(1, 6):
            s1 = mol1.intor('int1e_ovlp', shls_slice=(0,1,l,l+1))
            s2 = mol2.intor('int1e_ovlp', shls_slice=(0,1,l,l+1))

            # Rotating a basis is equivalent to an inversed rotation over the axes.
            # The Eular angles that rotates molecule to a new geometry (axes
            # transformation) corresponds to the inversed rotation over basis.
            #r = small_dmatrix(l, -beta, reorder_p=True)
            r = Dmatrix.Dmatrix(l, -gamma, -beta, -alpha, reorder_p=True)
            self.assertAlmostEqual(abs(numpy.dot(s1, r) - s2).max(), 0, 12)

    def test_euler_angles(self):
        c0 = numpy.random.random(3)
        c2 = numpy.random.random(3)
        self.assertRaises(AssertionError, Dmatrix.get_euler_angles, c0, c2)

        c0 /= numpy.linalg.norm(c0)
        c2 /= numpy.linalg.norm(c2)
        alpha, beta, gamma = Dmatrix.get_euler_angles(c0, c2)
        c1 = numpy.dot(geom.rotation_mat((0,0,1), gamma), c0)
        c1 = numpy.dot(geom.rotation_mat((0,1,0), beta), c1)
        c1 = numpy.dot(geom.rotation_mat((0,0,1), alpha), c1)
        self.assertAlmostEqual(abs(c2 - c1).max(), 0, 12)

        # transform coordinates
        numpy.random.seed(1)
        u, w, vh = numpy.linalg.svd(numpy.random.random((3,3)))
        c1 = u.dot(vh)
        u, w, vh = numpy.linalg.svd(c1+2*numpy.random.random((3,3)))
        c2 = u.dot(vh)
        alpha, beta, gamma = Dmatrix.get_euler_angles(c1, c2)
        yp  = numpy.einsum('j,kj->k', c1[1], geom.rotation_mat(c1[2], alpha))
        tmp = numpy.einsum('ij,kj->ik', c1 , geom.rotation_mat(c1[2], alpha))
        tmp = numpy.einsum('ij,kj->ik', tmp, geom.rotation_mat(yp   , beta ))
        c2p = numpy.einsum('ij,kj->ik', tmp, geom.rotation_mat(c2[2], gamma))
        self.assertAlmostEqual((c2-c2p).max(), 0, 13)


if __name__ == "__main__":
    print("Full Tests for Dmatrix")
    unittest.main()
