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
from pyscf.pbc.df import mdf
from pyscf.pbc import df
#from mpi4pyscf.pbc.df import mdf
pyscf.pbc.DEBUG = False
mdf.df.LINEAR_DEP_THR = 1e-7

L = 5.
n = 11
cell = pgto.Cell()
cell.a = numpy.diag([L,L,L])
cell.mesh = numpy.array([n,n,n])

cell.atom = '''C    3.    2.       3.
               C    1.    1.       1.'''
cell.basis = 'ccpvdz'
cell.verbose = 0
cell.rcut = 17
cell.build(0,0)

mf0 = pscf.RHF(cell)
mf0.exxdiv = 'vcut_sph'


numpy.random.seed(1)
kpts = numpy.random.random((5,3))
kpts[0] = 0
kpts[3] = kpts[0]-kpts[1]+kpts[2]
kpts[4] *= 1e-5

kmdf = mdf.MDF(cell)
kmdf.auxbasis = 'weigend'
kmdf.kpts = kpts
kmdf.mesh = (11,)*3
kmdf.eta = 0.154728892598

cell1 = pgto.Cell()
cell1.a = numpy.eye(3) * 3.
cell1.mesh = [10]*3
cell1.atom = '''C    3.    2.       3.
               C    1.    1.       1.'''
cell1.basis = [[0, (3.5, 1)], [0, (1.0, 1)], [1, (0.6, 1)]]
cell1.rcut = 9.5
cell1.build(0,0)

kmdf1 = mdf.MDF(cell1)
kmdf1.auxbasis = df.aug_etb(cell1, 1.8)
kmdf1.kpts = kpts
kmdf1.mesh = [6]*3
kmdf1.eta = 0.1


def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(a.ravel(), w)

class KnowValues(unittest.TestCase):
    def test_vbar(self):
        auxcell = mdf.make_modrho_basis(cell, 'ccpvdz', 1.)
        vbar = mdf.MDF(cell).auxbar(auxcell)
        self.assertAlmostEqual(finger(vbar), -0.00438699039629, 9)

    def test_get_eri_gamma_high_cost(self):
        odf = mdf.MDF(cell)
        odf.auxbasis = 'weigend'
        odf.mesh = (11,)*3
        odf.eta = 0.154728892598
        eri0000 = odf.get_eri()
        self.assertTrue(eri0000.dtype == numpy.double)
        self.assertAlmostEqual(eri0000.real.sum(), 140.52553833398147, 6)
        self.assertAlmostEqual(finger(eri0000), -1.2234059928846319, 6)

        eri1111 = kmdf.get_eri((kpts[0],kpts[0],kpts[0],kpts[0]))
        self.assertTrue(eri1111.dtype == numpy.double)
        self.assertAlmostEqual(eri1111.real.sum(), 140.52553833398153, 6)
        self.assertAlmostEqual(eri1111.imag.sum(), 0, 7)
        self.assertAlmostEqual(finger(eri1111), -1.2234059928846333, 6)
        self.assertTrue(numpy.allclose(eri1111, eri0000))

        eri4444 = kmdf.get_eri((kpts[4],kpts[4],kpts[4],kpts[4]))
        self.assertTrue(eri4444.dtype == numpy.complex128)
        self.assertAlmostEqual(eri4444.real.sum(), 259.46539833377523, 6)
        self.assertAlmostEqual(abs(eri4444.imag).sum(), 0.00044187056294873458, 9)
        self.assertAlmostEqual(finger(eri4444), 1.9705270829923354-3.6097479693720031e-07j, 6)
        eri0000 = ao2mo.restore(1, eri0000, cell.nao_nr()).reshape(eri4444.shape)
        self.assertTrue(numpy.allclose(eri0000, eri4444, atol=1e-7))

    def test_get_eri_1111_high_cost(self):
        eri1111 = kmdf.get_eri((kpts[1],kpts[1],kpts[1],kpts[1]))
        self.assertTrue(eri1111.dtype == numpy.complex128)
        self.assertAlmostEqual(eri1111.real.sum(), 258.81872464108312, 6)
        self.assertAlmostEqual(abs(eri1111.imag).sum(), 16.275864968641145, 6)
        self.assertAlmostEqual(finger(eri1111), 2.2339104732873363+0.10954687420327755j, 7)
        check2 = kmdf.get_eri((kpts[1]+5e-9,kpts[1]+5e-9,kpts[1],kpts[1]))
        self.assertTrue(numpy.allclose(eri1111, check2, atol=1e-7))

    def test_get_eri_0011_high_cost(self):
        eri0011 = kmdf.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        self.assertTrue(eri0011.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0011.real.sum(), 259.13073670793142, 6)
        self.assertAlmostEqual(abs(eri0011.imag).sum(), 8.4042424538275426, 6)
        self.assertAlmostEqual(finger(eri0011), 2.1374953278715552+0.12350314965485282j, 6)

    def test_get_eri_0110_high_cost(self):
        eri0110 = kmdf.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        self.assertTrue(eri0110.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0110.real.sum(), 411.86033298299179, 6)
        self.assertAlmostEqual(abs(eri0110.imag).sum(), 136.58633427242452, 6)
        self.assertAlmostEqual(finger(eri0110), 1.3767132918850329+0.12378724026874122j, 6)
        check2 = kmdf.get_eri((kpts[0]+5e-9,kpts[1]+5e-9,kpts[1],kpts[0]))
        self.assertTrue(numpy.allclose(eri0110, check2, atol=1e-7))

#    def test_get_eri_0123_high_cost(self):
#        eri0123 = kmdf.get_eri(kpts[:4])
#        self.assertTrue(eri0123.dtype == numpy.complex128)
#        self.assertAlmostEqual(eri0123.real.sum(), 410.38308763371651, 6)
#        self.assertAlmostEqual(abs(eri0123.imag.sum()), 0.18510527268199378, 6)
#        self.assertAlmostEqual(finger(eri0123), 1.7644500565943559+0.30677193151572507j, 6)

    def test_get_eri_gamma_1(self):
        odf = mdf.MDF(cell1)
        odf.auxbasis = df.aug_etb(cell1, 1.8)
        odf.mesh = [6]*3
        odf.eta = 0.1
        eri0000 = odf.get_eri()
        self.assertTrue(eri0000.dtype == numpy.double)
        self.assertAlmostEqual(eri0000.real.sum(), 27.271885446069433, 6)
        self.assertAlmostEqual(finger(eri0000), 1.0614085634080137, 6)

        eri1111 = kmdf1.get_eri((kpts[0],kpts[0],kpts[0],kpts[0]))
        self.assertTrue(eri1111.dtype == numpy.double)
        self.assertAlmostEqual(eri1111.real.sum(), 27.271885446069433, 6)
        self.assertAlmostEqual(eri1111.imag.sum(), 0, 7)
        self.assertAlmostEqual(finger(eri0000), 1.0614085634080137, 6)
        self.assertAlmostEqual(abs(eri1111-eri0000).max(), 0, 12)

    def test_get_eri_1111_1(self):
        eri1111 = kmdf1.get_eri((kpts[1],kpts[1],kpts[1],kpts[1]))
        self.assertTrue(eri1111.dtype == numpy.complex128)
        self.assertAlmostEqual(eri1111.real.sum(), 44.106518037762719, 6)
        self.assertAlmostEqual(abs(eri1111.imag).sum(), 11.560980263508144, 6)
        self.assertAlmostEqual(finger(eri1111), (5.8655421128841088+0.034457178081070433j), 7)
        check2 = kmdf1.get_eri((kpts[1]+5e-9,kpts[1]+5e-9,kpts[1],kpts[1]))
        self.assertTrue(numpy.allclose(eri1111, check2, atol=1e-7))

    def test_get_eri_0011_1(self):
        eri0011 = kmdf1.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        self.assertTrue(eri0011.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0011.real.sum(), 44.247892059548818, 6)
        self.assertAlmostEqual(abs(eri0011.imag).sum(), 6.5167066853467244, 6)
        self.assertAlmostEqual(finger(eri0011), (6.2170722376745422-0.10266245792326825j), 6)

    def test_get_eri_0110_1(self):
        eri0110 = kmdf1.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        self.assertTrue(eri0110.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0110.real.sum(), 89.922543439119721, 6)
        self.assertAlmostEqual(abs(eri0110.imag).sum(), 67.573338930409079, 6)
        self.assertAlmostEqual(finger(eri0110), (7.3778033770176341-4.0951535119430975j), 6)
        check2 = kmdf1.get_eri((kpts[0]+5e-9,kpts[1]+5e-9,kpts[1],kpts[0]))
        self.assertTrue(numpy.allclose(eri0110, check2, atol=1e-7))

#    def test_get_eri_0123_1(self):
#        eri0123 = kmdf1.get_eri(kpts[:4])
#        self.assertTrue(eri0123.dtype == numpy.complex128)
#        self.assertAlmostEqual(eri0123.real.sum(), 85.318309473904634, 6)
#        self.assertAlmostEqual(abs(eri0123.imag.sum()), 0.18510527268199378, 6)
#        self.assertAlmostEqual(finger(eri0123), 1.7644500565943559+0.30677193151572507j, 6)


if __name__ == '__main__':
    print("Full Tests for mdf")
    unittest.main()

