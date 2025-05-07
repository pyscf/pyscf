# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
import tempfile
import numpy
import numpy as np
from pyscf import lib
import pyscf.pbc
from pyscf import ao2mo, gto
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.df import df, aug_etb, FFTDF, mdf
from pyscf.pbc.df import gdf_builder
#from mpi4pyscf.pbc.df import df
pyscf.pbc.DEBUG = False

def setUpModule():
    global cell, cell1, kmdf, ccgdf, kpts
    L = 5.
    n = 11
    cell = pgto.Cell()
    cell.a = numpy.diag([L,L,L])
    cell.mesh = numpy.array([n,n,n])

    cell.atom = '''He    3.    2.       3.
                   He    1.    1.       1.'''
    cell.basis = 'ccpvdz'
    cell.precision=1e-12
    cell.verbose = 0
    cell.max_memory = 1000
    cell.build(0,0)

    numpy.random.seed(1)
    kpts = numpy.random.random((5,3))
    kpts[0] = 0
    kpts[3] = kpts[0]-kpts[1]+kpts[2]
    kpts[4] *= 1e-5

    kmdf = df.DF(cell)
    kmdf._prefer_ccdf = False
    kmdf.auxbasis = 'weigend'
    kmdf.kpts = kpts
    kmdf.mesh = [17] * 3

    ccgdf = df.DF(cell)
    ccgdf._prefer_ccdf = True
    ccgdf.auxbasis = 'weigend'
    ccgdf.kpts = kpts
    ccgdf.mesh = [17] * 3

def tearDownModule():
    global cell, kmdf, ccgdf
    del cell, kmdf, ccgdf


class KnownValues(unittest.TestCase):
    def test_aft_get_pp_high_cost(self):
        cell = pgto.Cell()
        cell.verbose = 0
        cell.atom = 'C 0 0 0; C 1 1 1'
        cell.a = numpy.diag([4, 4, 4])
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.build()
        v1 = df.DF(cell).get_pp([.25]*3)
        self.assertAlmostEqual(lib.fp(v1), -0.0533131779366407-0.11895124492447073j, 8)

    def test_get_eri_gamma(self):
        odf = df.DF(cell)
        odf.linear_dep_threshold = 1e-7
        odf.auxbasis = 'weigend'
        odf.mesh = [11]*3
        eri0000 = odf.get_eri()
        self.assertTrue(eri0000.dtype == numpy.double)
        self.assertAlmostEqual(eri0000.real.sum(), 41.612815388042186, 8)
        self.assertAlmostEqual(lib.fp(eri0000), 1.9981475954967156, 8)

    def test_rsgdf_get_eri_gamma1(self):
        eri0000 = kmdf.get_eri()
        self.assertTrue(eri0000.dtype == numpy.double)
        self.assertAlmostEqual(eri0000.real.sum(), 41.612815388042186, 8)
        self.assertAlmostEqual(lib.fp(eri0000), 1.9981475954967156, 8)

        eri1111 = kmdf.get_eri((kpts[0],kpts[0],kpts[0],kpts[0]))
        self.assertTrue(eri1111.dtype == numpy.double)
        self.assertAlmostEqual(eri1111.real.sum(), 41.61281538370225, 8)
        self.assertAlmostEqual(eri1111.imag.sum(), 0, 9)
        self.assertAlmostEqual(lib.fp(eri1111), 1.9981475954967156, 8)
        self.assertAlmostEqual(abs(eri1111-eri0000).max(), 0, 9)

        eri4444 = kmdf.get_eri((kpts[4],kpts[4],kpts[4],kpts[4]))
        self.assertTrue(eri4444.dtype == numpy.complex128)
        self.assertAlmostEqual(eri4444.real.sum(), 62.55123863003902, 8)
        # kpts[4] ~= 0, eri4444.imag should be very closed to 0
        self.assertAlmostEqual(abs(eri4444.imag).sum(), 0, 7)
        self.assertTrue(abs(eri4444.imag).sum() > 1e-8)
        self.assertAlmostEqual(lib.fp(eri4444), 0.6205986620420332+0j, 8)
        eri0000 = ao2mo.restore(1, eri0000, cell.nao_nr()).reshape(eri4444.shape)
        self.assertAlmostEqual(abs(eri0000-eri4444).max(), 0, 8)

    def test_rsgdf_get_eri_1111(self):
        eri1111 = kmdf.get_eri((kpts[1],kpts[1],kpts[1],kpts[1]))
        self.assertTrue(eri1111.dtype == numpy.complex128)
        self.assertAlmostEqual(eri1111.real.sum(), 62.54976506061887, 8)
        self.assertAlmostEqual(abs(eri1111.imag).sum(), 0.0018154474705429095, 8)
        self.assertAlmostEqual(lib.fp(eri1111), 0.6203912329366568+8.790493572227777e-05j, 8)
        check2 = kmdf.get_eri((kpts[1]+5e-8,kpts[1]+5e-8,kpts[1],kpts[1]))
        self.assertTrue(numpy.allclose(eri1111, check2, atol=1e-7))

        with lib.temporary_env(kmdf.cell, cart=True):
            eri1111_cart = kmdf.get_eri((kpts[1],kpts[1],kpts[1],kpts[1]))
        self.assertAlmostEqual(abs(eri1111-eri1111_cart).max(), 0, 9)

    def test_rsgdf_get_eri_0011(self):
        eri0011 = kmdf.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        self.assertTrue(eri0011.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0011.real.sum(), 62.5505017611663, 8)
        self.assertAlmostEqual(abs(eri0011.imag).sum(), 0.0009080830870008819, 8)
        self.assertAlmostEqual(lib.fp(eri0011), 0.6205470491228497+7.547569375281784e-05j, 8)

    def test_rsgdf_get_eri_0110(self):
        eri0110 = kmdf.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        self.assertTrue(eri0110.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0110.real.sum(), 83.11360962348763, 8)
        self.assertAlmostEqual(abs(eri0110.imag).sum(), 5.083516727205381, 8)
        self.assertAlmostEqual(lib.fp(eri0110), 0.9700462344979466-0.331882616586239j, 8)
        check2 = kmdf.get_eri((kpts[0]+5e-8,kpts[1]+5e-8,kpts[1],kpts[0]))
        self.assertTrue(numpy.allclose(eri0110, check2, atol=1e-7))

    def test_rsgdf_get_eri_0123(self):
        eri0123 = kmdf.get_eri(kpts[:4])
        self.assertTrue(eri0123.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0123.real.sum(), 83.10940286392085, 8)
        self.assertAlmostEqual(abs(eri0123.imag.sum()), 4.990140599070436e-05, 8)
        self.assertAlmostEqual(lib.fp(eri0123), 0.9695261296288074-0.33222740818370966j, 8)

    def test_ccgdf_get_eri_gamma1(self):
        eri0000 = ccgdf.get_eri()
        self.assertTrue(eri0000.dtype == numpy.double)
        self.assertAlmostEqual(eri0000.real.sum(), 41.612815388042186, 8)
        self.assertAlmostEqual(lib.fp(eri0000), 1.9981475954967156, 8)

        eri1111 = ccgdf.get_eri((kpts[0],kpts[0],kpts[0],kpts[0]))
        self.assertTrue(eri1111.dtype == numpy.double)
        self.assertAlmostEqual(eri1111.real.sum(), 41.61281538370225, 8)
        self.assertAlmostEqual(eri1111.imag.sum(), 0, 9)
        self.assertAlmostEqual(lib.fp(eri1111), 1.9981475954967156, 8)
        self.assertAlmostEqual(abs(eri1111-eri0000).max(), 0, 9)

        eri4444 = ccgdf.get_eri((kpts[4],kpts[4],kpts[4],kpts[4]))
        self.assertTrue(eri4444.dtype == numpy.complex128)
        self.assertAlmostEqual(eri4444.real.sum(), 62.55123863003902, 8)
        # kpts[4] ~= 0, eri4444.imag should be very closed to 0
        self.assertAlmostEqual(abs(eri4444.imag).sum(), 0, 7)
        self.assertTrue(abs(eri4444.imag).sum() > 1e-8)
        self.assertAlmostEqual(lib.fp(eri4444), 0.6205986620420332+0j, 8)
        eri0000 = ao2mo.restore(1, eri0000, cell.nao_nr()).reshape(eri4444.shape)
        self.assertAlmostEqual(abs(eri0000-eri4444).max(), 0, 8)

    def test_ccgdf_get_eri_1111(self):
        eri1111 = ccgdf.get_eri((kpts[1],kpts[1],kpts[1],kpts[1]))
        self.assertTrue(eri1111.dtype == numpy.complex128)
        self.assertAlmostEqual(eri1111.real.sum(), 62.54976506061887, 8)
        self.assertAlmostEqual(abs(eri1111.imag).sum(), 0.0018154474705429095, 8)
        self.assertAlmostEqual(lib.fp(eri1111), 0.6203912329366568+8.790493572227777e-05j, 8)
        check2 = ccgdf.get_eri((kpts[1]+5e-8,kpts[1]+5e-8,kpts[1],kpts[1]))
        self.assertTrue(numpy.allclose(eri1111, check2, atol=1e-7))

        with lib.temporary_env(ccgdf.cell, cart=True):
            eri1111_cart = ccgdf.get_eri((kpts[1],kpts[1],kpts[1],kpts[1]))
        self.assertAlmostEqual(abs(eri1111-eri1111_cart).max(), 0, 9)

    def test_ccgdf_get_eri_0011(self):
        eri0011 = ccgdf.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        self.assertTrue(eri0011.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0011.real.sum(), 62.5505017611663, 8)
        self.assertAlmostEqual(abs(eri0011.imag).sum(), 0.0009080830870008819, 8)
        self.assertAlmostEqual(lib.fp(eri0011), 0.6205470491228497+7.547569375281784e-05j, 8)

    def test_ccgdf_get_eri_0110(self):
        eri0110 = ccgdf.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        self.assertTrue(eri0110.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0110.real.sum(), 83.11360962348763, 8)
        self.assertAlmostEqual(abs(eri0110.imag).sum(), 5.083516727205381, 8)
        self.assertAlmostEqual(lib.fp(eri0110), 0.9700462344979466-0.331882616586239j, 8)
        check2 = ccgdf.get_eri((kpts[0]+5e-8,kpts[1]+5e-8,kpts[1],kpts[0]))
        self.assertTrue(numpy.allclose(eri0110, check2, atol=1e-7))

    def test_ccgdf_get_eri_0123(self):
        eri0123 = ccgdf.get_eri(kpts[:4])
        self.assertTrue(eri0123.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0123.real.sum(), 83.10940286392085, 8)
        self.assertAlmostEqual(abs(eri0123.imag.sum()), 4.990140599070436e-05, 8)
        self.assertAlmostEqual(lib.fp(eri0123), 0.9695261296288074-0.33222740818370966j, 8)

    def test_modrho_basis(self):
        cell = pgto.Cell(
            atom = 'Li 0 0 0; Li 1.5 1.5 1.5',
            a = numpy.eye(3) * 3,
        )
        cell.build()
        cell1 =  pgto.Cell(
            atom = 'Li1 0 0 0; Li2 1.5 1.5 1.5',
            a = numpy.eye(3) * 3,
        )
        cell1.build()
        auxcell = df.make_modrho_basis(cell, auxbasis='ccpvdz', drop_eta=.1)
        auxcell1 = df.make_modrho_basis(cell1, auxbasis='ccpvdz', drop_eta=.1)
        for ib in range(len(auxcell._bas)):
            nprim = auxcell.bas_nprim(ib)
            nc = auxcell.bas_nctr(ib)
            es = auxcell.bas_exp(ib)
            es1 = auxcell1.bas_exp(ib)
            ptr = auxcell._bas[ib, gto.mole.PTR_COEFF]
            ptr1 = auxcell1._bas[ib, gto.mole.PTR_COEFF]
            cs = auxcell._env[ptr:ptr+nprim*nc]
            cs1 = auxcell1._env[ptr1:ptr1+nprim*nc]
            self.assertAlmostEqual(abs(es - es1).max(), 0, 15)
            self.assertAlmostEqual(abs(cs - cs1).max(), 0, 15)

    # issue #1117
    def test_cell_with_cart(self):
        cell = pgto.M(
            atom='Li 0 0 0; H 2 2 2',
            a=(numpy.ones([3, 3]) - numpy.eye(3)) * 2,
            cart=True,
            basis={'H': '''
H   S
0.5    1''',
                   'Li': '''
Li  S
0.8    1
0.4    1
Li  P
0.8    1
0.4    1'''})

        eri0 = FFTDF(cell).get_eri()
        eri1 = df.GDF(cell).set(auxbasis=aug_etb(cell)).get_eri()
        self.assertAlmostEqual(abs(eri1-eri0).max(), 0, 2)

    def test_kpoints_input(sef):
        cell.space_group_symmetry = True
        cell.build()
        kpts = cell.make_kpts([2,2,2],
                              space_group_symmetry=True,
                              time_reversal_symmetry=True)

        mydf = df.GDF(cell, kpts=kpts)
        assert mydf.kpts.shape == (8,3)

        mydf = FFTDF(cell, kpts=kpts)
        assert mydf.kpts.shape == (8,3)

        mydf = mdf.MDF(cell, kpts=kpts)
        assert mydf.kpts.shape == (8,3)

    # issue 2790: build full-range J and SR K separately
    def test_rsh_df(self):
        cell = pgto.M(
            a = np.eye(3) * 3,
            atom = '''H 0.0000 0.0000 0.0000; H 1.3575 1.3575 1.3575''',
            basis = [[0, [.5, 1]]],
        )
        kpts = cell.make_kpts([2,1,1])
        kmf = cell.KRKS(kpts=kpts, xc='hse06').density_fit().run()
        assert abs(kmf.e_tot - -0.687018457218418) < 1e-8

        kmf = cell.KRKS(kpts=kpts, xc='camb3lyp').density_fit().run()
        assert abs(kmf.e_tot - -0.674692142275221) < 1e-8

        kmf = cell.KRKS(kpts=kpts, xc='wb97').density_fit().run()
        assert abs(kmf.e_tot - -0.678851816639354) < 1e-8

if __name__ == '__main__':
    print("Full Tests for df")
    unittest.main()
