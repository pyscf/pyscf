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

import unittest
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import dft
from pyscf.prop.ssc import rhf as rhf_ssc

# Test numerical integration scheme JCP 73, 5718 (1980); DOI:10.1063/1.440051
# Test field gradients integrals

n_gauss = 30

class KnownValues(unittest.TestCase):
    def test_2c_r3(self):
        '''2-center vec{r}/r^3 type integral'''
        mol = gto.M(atom='H 0 1 2; H 1 .3 .1', basis='ccpvdz')
        r0 = (.5, .4, .3)
        mol.set_rinv_orig_(r0)
        ipv = mol.intor('int1e_iprinv_sph', comp=3)
        nablav = ipv + ipv.transpose(0,2,1)
        t, w = numpy.polynomial.legendre.leggauss(n_gauss)
        #t, w = numpy.polynomial.chebyshev.chebgauss(n_gauss)
        a = (1+t)/(1-t) * .8
        w *= 2/(1-t)**2 * .8

        #a, w = numpy.polynomial.hermite.hermgauss(n_gauss)
        #a, w = dft.radi.gauss_chebyshev(n_gauss)
        #a, w = dft.radi.treutler_ahlrichs(n_gauss)
        #a, w = dft.radi.mura_knowles(n_gauss)
        #a, w = dft.radi.delley(n_gauss)

        fakemol = gto.Mole()
        ptr = 0
        fakemol._atm = numpy.asarray([[0, ptr, 0, 0, 0, 0]], dtype=numpy.int32)
        ptr += 3
        fakemol._bas = numpy.asarray([[0, 1, n_gauss, 1, 0, ptr, ptr+n_gauss, 0]], dtype=numpy.int32)
        p_cart2sph_factor = 0.488602511902919921
        fakemol._env = numpy.hstack((r0, a**2, a**2*w*4/numpy.pi**.5/p_cart2sph_factor))
        fakemol._built = True

        pmol = mol + fakemol
        i3c = pmol.intor('int3c1e_sph', shls_slice=(mol.nbas,pmol.nbas,0,mol.nbas,0,mol.nbas))
        self.assertAlmostEqual(abs(i3c - nablav).max(), 0, 9)

    def test_3c_r1(self):
        '''3-center 1/r type integral'''
        mol = gto.M(atom='H 0 1 2; H 1 .3 .1', basis='ccpvdz')
        r0 = (.5, .4, .3)
        mol.set_rinv_orig_(r0)
        ipv = mol.intor('int3c1e_rinv_sph')
        nablav = ipv
        t, w = numpy.polynomial.legendre.leggauss(n_gauss)
        a = (1+t)/(1-t) * .8
        w *= 2/(1-t)**2 * .8

        fakemol = gto.Mole()
        ptr = 0
        fakemol._atm = numpy.asarray([[0, ptr, 0, 0, 0, 0]], dtype=numpy.int32)
        ptr += 3
        fakemol._bas = numpy.asarray([[0, 0, n_gauss, 1, 0, ptr, ptr+n_gauss, 0]], dtype=numpy.int32)
        s_cart2sph_factor = 0.282094791773878143
        fakemol._env = numpy.hstack((r0, a**2, w*2/numpy.pi**.5/s_cart2sph_factor))
        fakemol._built = True

        pmol = mol + fakemol
        i3c = pmol.intor('int4c1e_sph', shls_slice=(mol.nbas,pmol.nbas, 0,mol.nbas,0,mol.nbas,0,mol.nbas))
        nao = mol.nao_nr()
        i3c = i3c.reshape(nao,nao,nao)
        self.assertAlmostEqual(abs(i3c - nablav).max(), 0, 9)

    def test1_3c_r3(self):
        '''3-center vec{r}/r^3 type integral'''
        mol = gto.M(atom='H 0 1 2; H 1 .3 .1', basis='ccpvdz')
        r0 = (.5, .4, .3)
        mol.set_rinv_orig_(r0)
        ipv = mol.intor('int3c1e_iprinv_sph', comp=3)
        nablav = ipv + ipv.transpose(0,2,1,3) + ipv.transpose(0,2,3,1)
        t, w = numpy.polynomial.legendre.leggauss(n_gauss)
        a = (1+t)/(1-t) * .8
        w *= 2/(1-t)**2 * .8

        fakemol = gto.Mole()
        ptr = 0
        fakemol._atm = numpy.asarray([[0, ptr, 0, 0, 0, 0]], dtype=numpy.int32)
        ptr += 3
        fakemol._bas = numpy.asarray([[0, 1, n_gauss, 1, 0, ptr, ptr+n_gauss, 0]], dtype=numpy.int32)
        p_cart2sph_factor = 0.488602511902919921
        fakemol._env = numpy.hstack((r0, a**2, a**2*w*4/numpy.pi**.5/p_cart2sph_factor))
        fakemol._built = True

        pmol = mol + fakemol
        i3c = pmol.intor('int4c1e_sph', shls_slice=(mol.nbas,pmol.nbas, 0,mol.nbas,0,mol.nbas,0,mol.nbas))
        nao = mol.nao_nr()
        i3c = i3c.reshape(3,nao,nao,nao)
        self.assertAlmostEqual(abs(i3c - nablav).max(), 0, 9)


    def test2_3c_r3(self):
        mol = gto.M(atom='H1 0.5 -0.6 0.4; H2 -0.5, 0.4, -0.3; H -0.4 -0.3 0.5; H 0.3 0.5 -0.6',
                    unit='B',
                    basis={'H': [[0,[2., 1]]], 'H1':[[1,[.5, 1]]], 'H2':[[1,[1,1]]]})
        orig1 = mol.atom_coord(2)
        orig2 = mol.atom_coord(3)
        mol._env[mol._bas[:2,6]] = 1
        t, w = numpy.polynomial.legendre.leggauss(n_gauss)
        a = (1+t)/(1-t) * .8
        w *= 2/(1-t)**2 * .8

        fakemol = gto.Mole()
        fakemol._atm = numpy.asarray([[0, 0, 0, 0, 0, 0]], dtype=numpy.int32)
        fakemol._bas = numpy.asarray([[0, 1, n_gauss, 1, 0, 3, 3+n_gauss, 0]],
                                     dtype=numpy.int32)
        p_cart2sph_factor = 0.488602511902919921
        fakemol._env = numpy.hstack((orig2, a**2, a**2*w*4/numpy.pi**.5/p_cart2sph_factor))
        fakemol._built = True

        pmol = mol + fakemol
        pmol.set_rinv_origin(orig1)
        mat1 = pmol.intor('int3c1e_iprinv_sph', comp=3,
                          shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas, pmol.nbas))
        mat  = pmol.intor('int3c1e_iprinv_sph', comp=3,
                          shls_slice=(mol.nbas, pmol.nbas, 0, mol.nbas, 0, mol.nbas))
        mat += mat1.transpose(0,3,1,2) + mat1.transpose(0,3,2,1)
        self.assertAlmostEqual(mat[0,0,0,3], 0.01931792982, 9)
        self.assertAlmostEqual(mat[0,1,0,3], 0.01431132964, 9)


        fakemol = gto.Mole()
        fakemol._atm = numpy.asarray([[0, 0, 0, 0, 0, 0],
                                      [0, 3, 0, 0, 0, 0]], dtype=numpy.int32)
        fakemol._bas = numpy.asarray([[0, 1, n_gauss, 1, 0, 6, 6+n_gauss, 0],
                                      [1, 1, n_gauss, 1, 0, 6, 6+n_gauss, 0]],
                                     dtype=numpy.int32)
        p_cart2sph_factor = 0.488602511902919921
        fakemol._env = numpy.hstack((orig1, orig2, a**2, a**2*w*4/numpy.pi**.5/p_cart2sph_factor))
        fakemol._built = True

        pmol = mol + fakemol
        mat = pmol.intor('int4c1e_sph',
                         shls_slice=(mol.nbas, pmol.nbas, mol.nbas, pmol.nbas, 0, mol.nbas, 0, mol.nbas))
        nao = mol.nao_nr()
        mat = mat.reshape(6,6,nao,nao)
        self.assertAlmostEqual(mat[0,3,0,3], 0.01931792982, 9)
        self.assertAlmostEqual(mat[0,4,0,3], 0.01431132964, 9)

        # With numerical integration
        grids = dft.gen_grid.Grids(mol)
        grids.build()
        ao = dft.numint.eval_ao(mol, grids.coords)
        dr1 = grids.coords - orig1
        dr2 = grids.coords - orig2
        rr1 = numpy.linalg.norm(dr1, axis=1)
        rr2 = numpy.linalg.norm(dr2, axis=1)
        aoao = ao[:,0] * ao[:,3]
        v1 = dr1[:,0] * dr2[:,0] / (rr1**3 * rr2**3)
        v2 = dr1[:,0] * dr2[:,1] / (rr1**3 * rr2**3)
        self.assertAlmostEqual(numpy.dot(aoao, v1*grids.weights), 0.01931792982, 7)
        self.assertAlmostEqual(numpy.dot(aoao, v2*grids.weights), 0.01431132964, 7)


    def test_field_gradients(self):
        mol = gto.M(atom='H1 0.5 -0.6 0.4; H2 -0.5, 0.4, -0.3; H -0.4 -0.3 0.5; H 0.3 0.5 -0.6',
                    unit='B',
                    basis={'H': [[0,[2., 1]]], 'H1':[[1,[.5, 1]]], 'H2':[[1,[1,1]]]})
        grids = dft.gen_grid.Grids(mol)
        grids.build()
        ao = mol.eval_gto('GTOval', grids.coords)
        r0 = mol.atom_coord(0)
        dr = grids.coords - r0
        dd = numpy.linalg.norm(dr, axis=1)
        rr = 3 * numpy.einsum('ix,iy->ixy', dr, dr)
        for i in range(3):
            rr[:,i,i] -= dd**2
        h1ref = lib.einsum('i,ixy,ip,iq->xypq', grids.weights/dd**5, rr, ao, ao)

        h1ao = rhf_ssc._get_integrals_fcsd(mol, 0)
        h1ao[numpy.diag_indices(3)] += rhf_ssc._get_integrals_fc(mol, 0)
        self.assertAlmostEqual(abs(h1ref - h1ao).max(), 0, 5)


if __name__ == "__main__":
    print("Full Tests for ssc integrals")
    unittest.main()

