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
from pyscf.pbc import gto as pgto
from pyscf.pbc import dft as pdft
from pyscf.pbc.df import incore
import pyscf.pbc
pyscf.pbc.DEBUG = False

def setUpModule():
    global cell, dfbuilder
    cell = pgto.M(atom='C 1 2 1; C 1 1 1', a=numpy.eye(3)*4, mesh = [5]*3,
                  basis = {'C':[[0, (1, 1)],
                                [1, (.5, 1)],
                                [2, (1.5, 1)]
                               ]})
    numpy.random.seed(1)
    kpts = numpy.random.random((2,3))
    dfbuilder = incore.Int3cBuilder(cell, cell, kpts)

def tearDownModule():
    global cell, dfbuilder
    del cell, dfbuilder

class KnownValues(unittest.TestCase):
    def test_aux_e2(self):
        cell = pgto.Cell()
        cell.unit = 'B'
        cell.a = numpy.eye(3) * 3.
        cell.mesh = numpy.array([41]*3)
        cell.atom = 'He 0 1 1; He 1 1 0'
        cell.basis = { 'He': [[0, (0.8, 1.0)],
                              [0, (1.2, 1.0)]] }
        cell.verbose = 0
        cell.precision = 1e-9
        cell.build(0, 0)
        auxcell = incore.format_aux_basis(cell)
        a1 = incore.aux_e2(cell, auxcell, 'int3c1e_sph')
        self.assertAlmostEqual(lib.fp(a1), 0.1208944790152819, 8)
        a2 = incore.aux_e2(cell, auxcell, 'int3c1e_sph', aosym='s2ij')
        self.assertTrue(numpy.allclose(a1, lib.unpack_tril(a2, axis=0).reshape(a1.shape)))

        numpy.random.seed(3)
        kpt = numpy.random.random(3)
        kptij_lst = numpy.array([[kpt,kpt]])
        a1 = incore.aux_e2(cell, auxcell, 'int3c1e_sph', kptij_lst=kptij_lst)
        self.assertAlmostEqual(lib.fp(a1), -0.073719031689332651-0.054002639392614758j, 8)
        a2 = incore.aux_e2(cell, auxcell, 'int3c1e_sph', aosym='s2', kptij_lst=kptij_lst)
        self.assertTrue(numpy.allclose(a1, lib.unpack_tril(a2, 1, axis=0).reshape(a1.shape)))

        numpy.random.seed(1)
        kptij_lst = numpy.random.random((1,2,3))
        a1 = incore.aux_e2(cell, auxcell, 'int3c1e_sph', aosym='s1', kptij_lst=kptij_lst)
        self.assertAlmostEqual(lib.fp(a1), 0.039329191948685879-0.039836453846241987j, 8)

    @unittest.skip('different to master')
    def test_fill_kk(self):
        # FIXME: Different to master branch
        int3c = dfbuilder.gen_int3c_kernel(aosym='s1', j_only=False, return_complex=True)
        self.assertAlmostEqual(lib.fp(int3c()), -26.25725373336515-0.11215178423742556j, 9)

        int3c = dfbuilder.gen_int3c_kernel(aosym='s2', j_only=False, return_complex=True)
        self.assertAlmostEqual(lib.fp(int3c()), 47.850058515747975+1.2325834591924032j, 9)

    @unittest.skip('different to master')
    def test_fill_k(self):
        # FIXME: Different to master branch
        int3c = dfbuilder.gen_int3c_kernel(aosym='s1', j_only=True, return_complex=True)
        self.assertAlmostEqual(lib.fp(int3c()), -37.82922614615208-0.0822974010349356j, 9)

        int3c = dfbuilder.gen_int3c_kernel(aosym='s2', j_only=True, return_complex=True)
        self.assertAlmostEqual(lib.fp(int3c()), 6.6227481557912071-0.80306690266835279j, 9)

    @unittest.skip('different to master')
    def test_fill_g(self):
        dfbuilder = incore.Int3cBuilder(cell, cell, numpy.zeros((1,3)))
        int3c = dfbuilder.gen_int3c_kernel(aosym='s2', j_only=False, return_complex=True)
        self.assertAlmostEqual(lib.fp(int3c()), 5.199528603910471, 9)

        int3c = dfbuilder.gen_int3c_kernel(aosym='s1', j_only=False, return_complex=True)
        out = int3c()
        # FIXME: Different to master branch
        self.assertAlmostEqual(lib.fp(out), -31.745140510501113, 9)

        mat1 = int3c(shls_slice=(1,4,2,4,2,3)).reshape(9,6,5)
        self.assertAlmostEqual(abs(out.reshape(18,18,18)[1:10,4:10,4:9] - mat1).max(), 0, 9)

    def test_fill_2c(self):
        # test PBCnr2c_fill_ks1
        mat = incore.fill_2c2e(cell, cell, 'int1e_ovlp_sph', hermi=0)
        self.assertAlmostEqual(lib.fp(mat), 2.2144557629971247, 9)

        mat = cell.pbc_intor('int1e_ovlp_sph', kpts=dfbuilder.kpts)
        self.assertAlmostEqual(lib.fp(mat[0]), 2.2137492396285916-0.004739404845627319j, 9)
        self.assertAlmostEqual(lib.fp(mat[1]), 2.2132325548987253+0.0056984781658280699j, 9)

if __name__ == '__main__':
    print("Full Tests for pbc.df.incore")
    unittest.main()
