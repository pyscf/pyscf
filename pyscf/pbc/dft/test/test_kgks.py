#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Authors: Chia-Nan Yeh <yehcanon@gmail.com>
#

import unittest
import numpy as np

from pyscf import lib
from pyscf.dft import radi
from pyscf.pbc import gto as gto
from pyscf.pbc import dft as dft
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.df import rsdf_builder, gdf_builder
try:
    import mcfun
except ImportError:
    mcfun = None

def setUpModule():
    global cell, alle_cell, kpts, alle_kpts
    cell = gto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''

    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.mesh = [29] * 3
    cell.build()
    kmesh = [2, 1, 1]
    kpts = cell.make_kpts(kmesh, wrap_around=True)

    alle_cell = gto.Cell()
    alle_cell.unit = 'A'
    alle_cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    alle_cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''

    alle_cell.basis = 'sto-3g'
    alle_cell.verbose = 0
    alle_cell.build()
    kmesh = [2, 1, 1]
    alle_kpts = alle_cell.make_kpts(kmesh, wrap_around=True)

def tearDownModule():
    global cell, alle_cell, kpts, alle_kpts
    del cell, alle_cell

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_KGKS(self):
        # In the absence of off diagonal blocks in the spin space, dft.KGKS should reproduce the dft.KRKS results
        # Reference from dft.KRKS
        # mf = dft.KRKS(cell, kpts)
        # mf.xc = 'lda'
        # mf.conv_tol = 1e-10
        # e_ref = mf.kernel() # -10.38125412115097
        # self.assertAlmostEqual(e_ref, -10.38125412115097, 8)
        mf = dft.KGKS(cell, kpts)
        mf.xc = 'lda'
        mf.conv_tol = 1e-10
        e_kgks = mf.kernel()
        self.assertAlmostEqual(e_kgks, -10.38125412115097, 8)

    def test_veff(self):
        mf = dft.KGKS(cell, kpts)
        n2c = cell.nao * 2
        np.random.seed(1)
        dm = np.random.rand(2, n2c, n2c) * .4 + np.random.rand(2, n2c, n2c) * .2j
        mf.xc = 'pbe'
        v = mf.get_veff(cell, dm)
        self.assertAlmostEqual(lib.fp(v), -99.365338+0j, 5)

    def test_KGKS_sfx2c1e_high_cost(self):
        with lib.light_speed(10) as c:
            # j2c_eig_always is set to make results match old version
            with lib.temporary_env(rsdf_builder._RSGDFBuilder, j2c_eig_always=True):
                mf = dft.KGKS(alle_cell, alle_kpts).density_fit().sfx2c1e()
                mf.xc = 'lda'
                mf.conv_tol = 1e-10
                e_kgks = mf.kernel()
                print(e_kgks)
            self.assertAlmostEqual(e_kgks, -75.67071562222077, 4)

    def test_KGKS_x2c1e_high_cost(self):
        with lib.light_speed(10) as c:
            # j2c_eig_always is set to make results match old version
            with lib.temporary_env(rsdf_builder._RSGDFBuilder, j2c_eig_always=True):
                mf = dft.KGKS(alle_cell, alle_kpts).density_fit().x2c1e()
                mf.xc = 'lda'
                mf.conv_tol = 1e-10
                e_kgks = mf.kernel()
                print(e_kgks)
            self.assertAlmostEqual(e_kgks, -75.66883793093882, 4)

    def test_collinear_kgks_gga(self):
        from pyscf.pbc import gto
        cell = gto.Cell()
        cell.a = '''0.      1.7834  1.7834
                    1.7834  0.      1.7834
                    1.7834  1.7834  0.    '''
        cell.atom = 'He 0.,  0.,  0.; H 0.8917,  0.8917,  0.8917'
        cell.basis = [[0, [2, 1]], [1, [.5, 1]]]
        cell.spin = 1
        cell.build()
        mf = cell.KGKS(kpts=cell.make_kpts([3,1,1]))
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.7335936544788495, 7)

    def test_collinear_kgks_gga(self):
        from pyscf.pbc import gto
        cell = gto.Cell()
        cell.a = '''0.      1.7834  1.7834
                    1.7834  0.      1.7834
                    1.7834  1.7834  0.    '''
        cell.atom = 'He 0.,  0.,  0.; H 0.8917,  0.8917,  0.8917'
        cell.basis = [[0, [2, 1]], [1, [.5, 1]]]
        cell.spin = 1
        cell.build()
        mf = cell.KGKS(kpts=cell.make_kpts([2,1,1]))
        mf.xc = 'pbe'
        mf.collinear = 'col'
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -1.6373456924395708, 7)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_kgks_gga(self):
        from pyscf.pbc import gto
        cell = gto.Cell()
        cell.a = '''0.      1.7834  1.7834
                    1.7834  0.      1.7834
                    1.7834  1.7834  0.    '''
        cell.atom = 'He 0.,  0.,  0.; H 0.8917,  0.8917,  0.8917'
        cell.basis = [[0, [2, 1]], [1, [.5, 1]]]
        cell.spin = 1
        cell.build()
        mf = cell.KGKS(kpts=cell.make_kpts([2,1,1])).density_fit()
        mf.xc = 'pbe'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -1.6312141891350893, 5)

    def test_ncol_x2c_kgks_lda(self):
        from pyscf.pbc import gto
        cell = gto.Cell()
        cell.a = '''0.      1.7834  1.7834
                    1.7834  0.      1.7834
                    1.7834  1.7834  0.    '''
        cell.atom = 'He 0.,  0.,  0.; H 0.8917,  0.8917,  0.8917'
        cell.basis = [[0, [2, 1]], [1, [.5, 1]]]
        cell.spin = 1
        cell.build()
        mf = cell.KGKS(kpts=cell.make_kpts([2,1,1])).x2c()
        mf.xc = 'lda,vwn'
        mf.collinear = 'ncol'
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -1.533809531603591, 6)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_mcol_x2c_kgks_lda(self):
        from pyscf.pbc import gto
        cell = gto.Cell()
        cell.a = '''0.      1.7834  1.7834
                    1.7834  0.      1.7834
                    1.7834  1.7834  0.    '''
        cell.atom = 'He 0.,  0.,  0.; H 0.8917,  0.8917,  0.8917'
        cell.basis = [[0, [2, 1]], [1, [.5, 1]]]
        cell.spin = 1
        cell.build()
        mf = cell.KGKS(kpts=cell.make_kpts([2,1,1])).x2c()
        mf.xc = 'lda,vwn'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 50
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -1.533809531603591, 6)

    def test_to_hf(self):
        mf = dft.KGKS(cell).density_fit()
        mf.with_df._j_only = True
        a_hf = mf.to_hf()
        self.assertTrue(a_hf.with_df._j_only)
        self.assertTrue(isinstance(a_hf, pbcscf.kghf.KGHF))

        mf = dft.KGKS(cell, kpts=cell.make_kpts([2,1,1])).density_fit()
        mf.with_df._j_only = True
        a_hf = mf.to_hf()
        self.assertTrue(not a_hf.with_df._j_only)
        self.assertTrue(isinstance(a_hf, pbcscf.kghf.KGHF))


if __name__ == '__main__':
    print("Full Tests for pbc.dft.kgks")
    unittest.main()
