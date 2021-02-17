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
import pyscf.pbc
from pyscf import ao2mo
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.df import df
#from mpi4pyscf.pbc.df import df
pyscf.pbc.DEBUG = False

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
cell.build(0,0)

mf0 = pscf.RHF(cell)
mf0.exxdiv = 'vcut_sph'


numpy.random.seed(1)
kpts = numpy.random.random((5,3))
kpts[0] = 0
kpts[3] = kpts[0]-kpts[1]+kpts[2]
kpts[4] *= 1e-5

kmdf = df.DF(cell)
kmdf.linear_dep_threshold = 1e-7
kmdf.auxbasis = 'weigend'
kmdf.kpts = kpts
# Note mesh is not dense enough. It breaks the conjugation symmetry between
# the k-points k and -k
kmdf.mesh = (6,)*3

def tearDownModule():
    global cell, mf0, kmdf
    del cell, mf0, kmdf


class KnownValues(unittest.TestCase):
    def test_get_eri_gamma(self):
        odf = df.DF(cell)
        odf.linear_dep_threshold = 1e-7
        odf.auxbasis = 'weigend'
        odf.mesh = (6,)*3
        eri0000 = odf.get_eri()
        self.assertTrue(eri0000.dtype == numpy.double)
        self.assertAlmostEqual(eri0000.real.sum(), 41.61280626625331, 9)
        self.assertAlmostEqual(lib.fp(eri0000), 1.9981472468639465, 9)

        eri1111 = kmdf.get_eri((kpts[0],kpts[0],kpts[0],kpts[0]))
        self.assertTrue(eri1111.dtype == numpy.double)
        self.assertAlmostEqual(eri1111.real.sum(), 41.61280626625331, 9)
        self.assertAlmostEqual(eri1111.imag.sum(), 0, 9)
        self.assertAlmostEqual(lib.fp(eri1111), 1.9981472468639465, 9)
        self.assertAlmostEqual(abs(eri1111-eri0000).max(), 0, 9)

        eri4444 = kmdf.get_eri((kpts[4],kpts[4],kpts[4],kpts[4]))
        self.assertTrue(eri4444.dtype == numpy.complex128)
        self.assertAlmostEqual(eri4444.real.sum(), 62.55120674032798, 9)
        self.assertAlmostEqual(abs(eri4444.imag).sum(), 0.0016507912195378644, 7)
        self.assertAlmostEqual(lib.fp(eri4444), 0.6206014899350296-7.413680313987067e-05j, 8)
        eri0000 = ao2mo.restore(1, eri0000, cell.nao_nr()).reshape(eri4444.shape)
        self.assertAlmostEqual(abs(eri0000-eri4444).max(), 0, 4)

    def test_get_eri_1111(self):
        eri1111 = kmdf.get_eri((kpts[1],kpts[1],kpts[1],kpts[1]))
        self.assertTrue(eri1111.dtype == numpy.complex128)
        self.assertAlmostEqual(eri1111.real.sum(), 62.549733244805395, 9)
        self.assertAlmostEqual(abs(eri1111.imag).sum(), 0.003461793534370973, 9)
        self.assertAlmostEqual(lib.fp(eri1111), 0.6203940639997652+1.3777985976593079e-05j, 9)
        check2 = kmdf.get_eri((kpts[1]+5e-8,kpts[1]+5e-8,kpts[1],kpts[1]))
        self.assertTrue(numpy.allclose(eri1111, check2, atol=1e-7))

        kmdf.cell.cart = True
        eri1111_cart = kmdf.get_eri((kpts[1],kpts[1],kpts[1],kpts[1]))
        self.assertAlmostEqual(abs(eri1111-eri1111_cart).max(), 0, 9)
        kmdf.cell.cart = False

    def test_get_eri_0011(self):
        eri0011 = kmdf.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        self.assertTrue(eri0011.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0011.real.sum(), 62.55046992445563, 9)
        self.assertAlmostEqual(abs(eri0011.imag).sum(), 0.002281807091480516, 9)
        self.assertAlmostEqual(lib.fp(eri0011), 0.6205498957415032+0.00012560844849936946j, 9)

    def test_get_eri_0110(self):
        eri0110 = kmdf.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        self.assertTrue(eri0110.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0110.real.sum(), 83.11379784456005, 9)
        self.assertAlmostEqual(abs(eri0110.imag).sum(), 5.083432749354445, 9)
        self.assertAlmostEqual(lib.fp(eri0110), 0.9700096733262158-0.3318635561567801j, 9)
        check2 = kmdf.get_eri((kpts[0]+5e-8,kpts[1]+5e-8,kpts[1],kpts[0]))
        self.assertTrue(numpy.allclose(eri0110, check2, atol=1e-7))

    def test_get_eri_0123(self):
        eri0123 = kmdf.get_eri(kpts[:4])
        self.assertTrue(eri0123.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0123.real.sum(), 83.11713402928555, 9)
        self.assertAlmostEqual(abs(eri0123.imag.sum()), 0.0006633634465733618, 9)
        self.assertAlmostEqual(lib.fp(eri0123), 0.9693314315640165-0.33152709566516436j, 9)



if __name__ == '__main__':
    print("Full Tests for df")
    unittest.main()

