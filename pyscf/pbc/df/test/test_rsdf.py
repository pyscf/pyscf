# Copyright 2021- The PySCF Developers. All Rights Reserved.
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
from pyscf import gto
from pyscf.pbc.df import rsdf_helper
import numpy
from pyscf import lib
import pyscf.pbc
from pyscf import ao2mo
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.df import rsdf, df
#from mpi4pyscf.pbc.df import df
pyscf.pbc.DEBUG = False

def setUpModule():
    global cell, kmdf, kpts
    L = 5.
    n = 11
    cell = pgto.Cell()
    cell.a = numpy.diag([L,L,L])
    cell.mesh = numpy.array([n,n,n])

    cell.atom = '''He    3.    2.       3.
                   He    1.    1.       1.'''
    cell.basis = 'ccpvdz'
    cell.verbose = 0
    cell.max_memory = 1000
    cell.precision = 1e-9
    cell.build(0,0)

    numpy.random.seed(1)
    kpts = numpy.random.random((5,3))
    kpts[0] = 0
    kpts[3] = kpts[0]-kpts[1]+kpts[2]
    kpts[4] *= 1e-5

    kmdf = rsdf.RSDF(cell)
    kmdf.auxbasis = 'weigend'
    kmdf.kpts = kpts

def tearDownModule():
    global cell, kmdf
    del cell, kmdf


class KnownValues(unittest.TestCase):
    def test_get_eri_gamma(self):
        odf = rsdf.RSDF(cell)
        odf.linear_dep_threshold = 1e-7
        odf.auxbasis = 'weigend'
        odf.mesh = [11]*3
        eri0000 = odf.get_eri()
        self.assertTrue(eri0000.dtype == numpy.double)
        self.assertAlmostEqual(eri0000.real.sum(), 41.61281537829081, 7)
        self.assertAlmostEqual(lib.fp(eri0000), 1.9981475954967156, 8)

    def test_rsgdf_get_eri_gamma1(self):
        eri0000 = kmdf.get_eri()
        self.assertTrue(eri0000.dtype == numpy.double)
        self.assertAlmostEqual(eri0000.real.sum(), 41.61281537829081, 7)
        self.assertAlmostEqual(lib.fp(eri0000), 1.9981475954967156, 8)

        eri1111 = kmdf.get_eri((kpts[0],kpts[0],kpts[0],kpts[0]))
        self.assertTrue(eri1111.dtype == numpy.double)
        self.assertAlmostEqual(eri1111.real.sum(), 41.61281538370225, 7)
        self.assertAlmostEqual(eri1111.imag.sum(), 0, 9)
        self.assertAlmostEqual(lib.fp(eri1111), 1.9981475954967156, 8)
        self.assertAlmostEqual(abs(eri1111-eri0000).max(), 0, 9)

        eri4444 = kmdf.get_eri((kpts[4],kpts[4],kpts[4],kpts[4]))
        self.assertTrue(eri4444.dtype == numpy.complex128)
        self.assertAlmostEqual(eri4444.real.sum(), 62.55123861831842, 7)
        # kpts[4] ~= 0, eri4444.imag should be very closed to 0
        self.assertAlmostEqual(abs(eri4444.imag).sum(), 0, 7)
        self.assertAlmostEqual(lib.fp(eri4444), 0.6205986620420332+0j, 8)
        eri0000 = ao2mo.restore(1, eri0000, cell.nao_nr()).reshape(eri4444.shape)
        self.assertAlmostEqual(abs(eri0000-eri4444).max(), 0, 7)

    def test_get_eri_1111(self):
        eri1111 = kmdf.get_eri((kpts[1],kpts[1],kpts[1],kpts[1]))
        self.assertTrue(eri1111.dtype == numpy.complex128)
        self.assertAlmostEqual(eri1111.real.sum(), 62.54976506061887, 8)
        self.assertAlmostEqual(abs(eri1111.imag).sum(), 0.0018154153948446872, 7)
        self.assertAlmostEqual(lib.fp(eri1111), 0.6203912329366568+8.790493572227777e-05j, 8)
        check2 = kmdf.get_eri((kpts[1]+5e-8,kpts[1]+5e-8,kpts[1],kpts[1]))
        self.assertTrue(numpy.allclose(eri1111, check2, atol=1e-7))

        with lib.temporary_env(kmdf.cell, cart=True):
            eri1111_cart = kmdf.get_eri((kpts[1],kpts[1],kpts[1],kpts[1]))
        self.assertAlmostEqual(abs(eri1111-eri1111_cart).max(), 0, 7)
        kmdf.cell.cart = False

    def test_get_eri_0011(self):
        eri0011 = kmdf.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        self.assertTrue(eri0011.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0011.real.sum(), 62.550501755408035, 7)
        self.assertAlmostEqual(abs(eri0011.imag).sum(), 0.0009080657889720216, 7)
        self.assertAlmostEqual(lib.fp(eri0011), 0.6205470491228497+7.547569375281784e-05j, 8)

    def test_get_eri_0110(self):
        eri0110 = kmdf.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        self.assertTrue(eri0110.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0110.real.sum(), 83.11360960389585, 7)
        self.assertAlmostEqual(abs(eri0110.imag).sum(), 5.083516745552487, 7)
        self.assertAlmostEqual(lib.fp(eri0110), 0.9700462344979466-0.331882616586239j, 8)
        check2 = kmdf.get_eri((kpts[0]+5e-8,kpts[1]+5e-8,kpts[1],kpts[0]))
        self.assertTrue(numpy.allclose(eri0110, check2, atol=1e-7))

    def test_get_eri_0123(self):
        eri0123 = kmdf.get_eri(kpts[:4])
        self.assertTrue(eri0123.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0123.real.sum(), 83.10940286392085, 8)
        self.assertAlmostEqual(abs(eri0123.imag.sum()), 4.9887958509e-5, 7)
        self.assertAlmostEqual(lib.fp(eri0123), 0.9695261296288074-0.33222740818370966j, 8)

    def test_rsdf_build(self):
        cell = pgto.M(a=numpy.eye(3)*1.8,
                      atom='''Li   0.   0.    0.; H    0.   .5   1.2 ''',
                      basis={'Li': [[0, [5., 1.]], [0, [.6, 1.]], [1, [3., 1.]]],
                             'H': [[0, [.3, 1.]]]})
        auxbasis = {'Li': [[0, [5., 1.]], [0, [1.5, 1.]], [1, [.5, 1.]], [2, [2.5, 1.]]],
                    'H':  [[0, [2., 1.]]]}
        numpy.random.seed(2)
        dm = numpy.random.random([cell.nao]*2)

        gdf = df.GDF(cell)
        gdf.auxbasis = auxbasis
        jref, kref = gdf.get_jk(dm)

        gdf = rsdf.RSGDF(cell)
        gdf.auxbasis = auxbasis
        vj, vk = gdf.get_jk(dm)

        self.assertAlmostEqual(abs(vj - jref).max(), 0, 7)
        self.assertAlmostEqual(abs(vk - kref).max(), 0, 7)
        self.assertAlmostEqual(lib.fp(vj), 2.383648833459583, 7)
        self.assertAlmostEqual(lib.fp(vk), 1.553598328349400, 7)


    def test_schwarz_two_center(self):
        basis = [[0, [1.2, .7, .1], [.3, .2, .8]], [1, [.5, 1.]]]
        for keep in (False, True):
            with self.subTest(keep_one_contraction=keep):
                safe = rsdf_helper._get_schwartz_data(basis, .5, keep1ctr=keep, safe=True)
                direct = rsdf_helper._get_schwartz_data(basis, .5, keep1ctr=keep, safe=False)
                numpy.testing.assert_allclose(safe, direct, atol=1e-12)
                self.assertTrue(numpy.all(safe > 0))
                screened = rsdf_helper._get_schwartz_data(basis, 1., keep1ctr=keep, safe=False)
                self.assertTrue(numpy.all(screened < direct))

    def test_cutoff_precision(self):
        mol = gto.M(atom='H 0 0 0; H 1 0 0', unit='Bohr',
                    basis=[[0, [.8, 1]], [1, [.5, 1]]], verbose=0)
        aux = gto.M(atom='He 0 0 0', basis=[[0, [.4, 1]], [1, [.3, 1]]], verbose=0)
        with mol.with_range_coulomb(-.5):
            eri = mol.intor('int2e', shls_slice=(1, 2, 3, 4, 1, 2, 3, 4))
        qij = numpy.linalg.norm(eri)**.5
        for estimator in ('ME', 'ISF', 'ISF0', 'ISFQ0', 'ISFQL'):
            with self.subTest(estimator=estimator):
                args = (mol, aux, 1, 3, 1., .5)
                loose = rsdf_helper._get_3c2e_Rcuts_for_d(*args, 1e-6, estimator, qij, Rprec=.05)
                tight = rsdf_helper._get_3c2e_Rcuts_for_d(*args, 1e-9, estimator, qij, Rprec=.05)
                uncorrected = rsdf_helper._get_3c2e_Rcuts_for_d(
                    *args, 1e-6, estimator, qij, Rprec=.05, eta_correct=False, R_correct=False)
                self.assertTrue(numpy.all(numpy.isfinite(tight)))
                self.assertTrue(numpy.all(tight > loose))
                self.assertTrue(numpy.all(loose >= uncorrected))
                self.assertTrue(numpy.all(uncorrected > 0))

if __name__ == '__main__':
    print("Full Tests for rsdf")
    unittest.main()
