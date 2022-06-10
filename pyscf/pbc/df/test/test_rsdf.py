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
from pyscf import ao2mo, gto
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.df import rsdf
#from mpi4pyscf.pbc.df import df
pyscf.pbc.DEBUG = False

def setUpModule():
    global cell, mf0, kmdf, kpts
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

    kmdf = rsdf.RSDF(cell)
    kmdf.linear_dep_threshold = 1e-7
    kmdf.auxbasis = 'weigend'
    kmdf.kpts = kpts

def tearDownModule():
    global cell, mf0, kmdf
    del cell, mf0, kmdf


class KnownValues(unittest.TestCase):
    def test_get_eri_gamma(self):
        odf = rsdf.RSDF(cell)
        odf.linear_dep_threshold = 1e-7
        odf.auxbasis = 'weigend'
        eri0000 = odf.get_eri()
        self.assertTrue(eri0000.dtype == numpy.double)
        self.assertAlmostEqual(eri0000.real.sum(), 41.6128153879698, 7)
        self.assertAlmostEqual(lib.fp(eri0000), 1.998147596746171, 7)

        eri1111 = kmdf.get_eri((kpts[0],kpts[0],kpts[0],kpts[0]))
        self.assertTrue(eri1111.dtype == numpy.double)
        self.assertAlmostEqual(eri1111.real.sum(), 41.61281538796979, 7)
        self.assertAlmostEqual(eri1111.imag.sum(), 0, 7)
        self.assertAlmostEqual(lib.fp(eri1111), 1.9981475967461677, 7)
        self.assertAlmostEqual(abs(eri1111-eri0000).max(), 0, 7)

        eri4444 = kmdf.get_eri((kpts[4],kpts[4],kpts[4],kpts[4]))
        self.assertTrue(eri4444.dtype == numpy.complex128)
        self.assertAlmostEqual(eri4444.real.sum(), 62.55123862990809, 7)
        self.assertAlmostEqual(abs(eri4444.imag).sum(), 1.8446941199052448e-07, 7)
        self.assertAlmostEqual(lib.fp(eri4444), 0.6205986625996083-6.3855069621608306e-09j, 8)
        eri0000 = ao2mo.restore(1, eri0000, cell.nao_nr()).reshape(eri4444.shape)
        self.assertAlmostEqual(abs(eri0000-eri4444).max(), 0, 4)

    def test_get_eri_1111(self):
        eri1111 = kmdf.get_eri((kpts[1],kpts[1],kpts[1],kpts[1]))
        self.assertTrue(eri1111.dtype == numpy.complex128)
        self.assertAlmostEqual(eri1111.real.sum(), 62.54954167794108, 7)
        self.assertAlmostEqual(abs(eri1111.imag).sum(), 0.00201257236255183, 7)
        self.assertAlmostEqual(lib.fp(eri1111), 0.6203261847805729+1.771458698423368e-05j, 7)
        check2 = kmdf.get_eri((kpts[1]+5e-8,kpts[1]+5e-8,kpts[1],kpts[1]))
        self.assertTrue(numpy.allclose(eri1111, check2, atol=1e-7))

        kmdf.cell.cart = True
        eri1111_cart = kmdf.get_eri((kpts[1],kpts[1],kpts[1],kpts[1]))
        self.assertAlmostEqual(abs(eri1111-eri1111_cart).max(), 0, 7)
        kmdf.cell.cart = False

    def test_get_eri_0011(self):
        eri0011 = kmdf.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        self.assertTrue(eri0011.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0011.real.sum(), 62.550390074719544, 7)
        self.assertAlmostEqual(abs(eri0011.imag).sum(), 0.001006757670291995, 7)
        self.assertAlmostEqual(lib.fp(eri0011), 0.6202307090207673-1.1365592955173372e-05j, 7)

    def test_get_eri_0110(self):
        eri0110 = kmdf.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        self.assertTrue(eri0110.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0110.real.sum(), 83.10554333963026, 7)
        self.assertAlmostEqual(abs(eri0110.imag).sum(), 5.018709716490495, 7)
        self.assertAlmostEqual(lib.fp(eri0110), 0.9492917585299039-0.32856082029420147j, 7)
        check2 = kmdf.get_eri((kpts[0]+5e-8,kpts[1]+5e-8,kpts[1],kpts[0]))
        self.assertTrue(numpy.allclose(eri0110, check2, atol=1e-7))

    def test_get_eri_0123(self):
        eri0123 = kmdf.get_eri(kpts[:4])
        self.assertTrue(eri0123.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0123.real.sum(), 83.10148514109942, 7)
        self.assertAlmostEqual(abs(eri0123.imag).sum(), 5.0167993897114105, 7)
        self.assertAlmostEqual(lib.fp(eri0123), 0.9495237069199953-0.3295848633939775j, 7)


if __name__ == '__main__':
    print("Full Tests for rsdf")
    unittest.main()
