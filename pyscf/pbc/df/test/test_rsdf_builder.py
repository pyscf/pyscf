# Copyright 2022 The PySCF Developers. All Rights Reserved.
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
import numpy as np
import scipy.linalg
from pyscf import lib
import pyscf.pbc
from pyscf.pbc import gto as pgto
from pyscf.pbc.df import df, aug_etb, FFTDF, make_auxcell
from pyscf.pbc.df import rsdf_builder
from pyscf.pbc.df import ft_ao
from pyscf.pbc.tools import pbc as pbctools
pyscf.pbc.DEBUG = False

cell = pgto.M(
    a = np.eye(3) * 3.5,
    mesh = [11] * 3,
    atom = '''He    3.    2.       3.
              He    1.    1.       1.''',
    basis = 'ccpvdz',
    verbose = 7,
    output = '/dev/null',
    max_memory = 1000,
)

kpts = cell.make_kpts([3,5,6])[[0, 2, 3, 4, 6, 12, 20]]
kpts[3] = kpts[0]-kpts[1]+kpts[2]
nkpts = len(kpts)

auxcell = df.make_auxcell(cell, 'weigend')
auxcell1 = make_auxcell(cell, 'weigend')

cell_sr = cell.copy()
cell_sr.omega = -1.2
auxcell_sr = df.make_auxcell(cell_sr, 'weigend')

def load(filename, kptij):
    with df.cderi_loader(filename, 'j3c', kptij) as cderi:
        return cderi[:]

def tearDownModule():
    global cell, auxcell, auxcell1, cell_sr, auxcell_sr
    del cell, auxcell, auxcell1, cell_sr, auxcell_sr


class KnownValues(unittest.TestCase):
    def test_get_2c2e_gamma(self):
        dfbuilder = rsdf_builder._RSGDFBuilder(cell, auxcell).build()
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))
        self.assertAlmostEqual(lib.fp(j2c), 0.5933533269476654, 9)

        dfbuilder.exclude_d_aux = False
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))
        self.assertAlmostEqual(lib.fp(j2c), 0.5933533269476654, 9)

        dfbuilder = rsdf_builder._RSGDFBuilder(cell, auxcell1).build()
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))
        self.assertAlmostEqual(lib.fp(j2c), -8.240595211369692, 9)

    def test_get_2c2e(self):
        dfbuilder = rsdf_builder._RSGDFBuilder(cell, auxcell, kpts).build()
        j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), -3.9589504442002568+2.882048772368134j, 9)
        self.assertAlmostEqual(lib.fp(j2c[0]), 0.5933533269476654, 9)

        dfbuilder = rsdf_builder._RSGDFBuilder(cell, auxcell1, kpts).build()
        j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), -110.15438692240832+8.528960438609099j, 9)
        self.assertAlmostEqual(lib.fp(j2c[0]), -8.240595211369692, 8)

    def test_get_2c2e_cart(self):
        with lib.temporary_env(cell, cart=True):
            dfbuilder = rsdf_builder._RSGDFBuilder(cell, auxcell, kpts).build()
            j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), -3.9589504442002568+2.882048772368134j, 9)

    def test_make_j3c_gamma(self):
        dfbuilder = rsdf_builder._RSGDFBuilder(cell, auxcell).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name)
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 1.4877735860707935, 8)

            dfbuilder.make_j3c(tmpf.name, aosym='s1')
            v1 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(abs(v1 - lib.unpack_tril(v2).reshape(v1.shape)).max(), 0, 9)

            dfbuilder.exclude_dd_block = True
            dfbuilder.exclude_d_aux = False
            dfbuilder.make_j3c(tmpf.name)
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 1.4877735860707935, 8)

            dfbuilder.exclude_dd_block = False
            dfbuilder.exclude_d_aux = True
            dfbuilder.make_j3c(tmpf.name)
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 1.4877735860707935, 8)

            dfbuilder.exclude_dd_block = False
            dfbuilder.exclude_d_aux = False
            dfbuilder.make_j3c(tmpf.name)
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 1.4877735860707935, 7)

    def test_make_j3c(self):
        dfbuilder = rsdf_builder._RSGDFBuilder(cell, auxcell, kpts).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v_s2 = []
            for ki in range(nkpts):
                for kj in range(nkpts):
                    v_s2.append(load(tmpf.name, kpts[[ki, kj]]))
            self.assertAlmostEqual(lib.fp(v_s2[0]), 1.4877735860707935, 8)
            self.assertAlmostEqual(lib.fp(v_s2[2*nkpts+4]), 4.530919637533813+0.10852447737595214j, 8)
            self.assertAlmostEqual(lib.fp(v_s2[2*nkpts+2]), 1.4492567814298059, 8)

            dfbuilder.make_j3c(tmpf.name, aosym='s1')
            for ki in range(nkpts):
                for kj in range(nkpts):
                    v1 = load(tmpf.name, kpts[[ki, kj]])
                    if ki == kj:
                        v2 = lib.unpack_tril(v_s2[ki*nkpts+kj]).reshape(v1.shape)
                        self.assertAlmostEqual(abs(v1 - v2).max(), 0, 9)
                    else:
                        self.assertAlmostEqual(abs(v1 - v_s2[ki*nkpts+kj]).max(), 0, 9)

    def test_make_j3c_j_only(self):
        dfbuilder = rsdf_builder._RSGDFBuilder(cell, auxcell, kpts).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2', j_only=True)
            v_s2 = []
            for ki in range(nkpts):
                v_s2.append(load(tmpf.name, kpts[[ki, ki]]))
            self.assertAlmostEqual(lib.fp(v_s2[0]), 1.4877735860707935, 8)
            self.assertAlmostEqual(lib.fp(v_s2[2]), 1.4492567814298059+0j, 8)

            dfbuilder.make_j3c(tmpf.name, aosym='s1', j_only=True)
            for ki in range(nkpts):
                v1 = load(tmpf.name, kpts[[ki, ki]])
                v2 = lib.unpack_tril(v_s2[ki]).reshape(v1.shape)
                self.assertAlmostEqual(abs(v1 - v2).max(), 0, 9)

    def test_vs_fft(self):
        cell = pgto.M(
            a = np.eye(3) * 2.8,
            atom = 'He    0.    2.2      1.; He    1.    1.       1.',
            basis = [[0, [1.2, 1.], [.7, .5], [0.4, .5]], [1, [1.1, .5], [0.4, .5]]],
            mesh = [14] * 3,
            verbose = 0,
        )
        auxcell = df.make_auxcell(cell,
            auxbasis=[[0, [1.2, 1.], [.7, .5], [0.4, .5]],
                      [1, [1.1, .5], [0.4, .5]],
                      [2, [1., 1.]],
                     ],
        )

        kpts = np.zeros((1, 3))
        dfbuilder = rsdf_builder._RSGDFBuilder(cell, auxcell, kpts)
        dfbuilder.omega = 0.9
        dfbuilder.build()
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))

        Gv, Gvbase, kws = cell.get_Gv_weights()
        kpt = np.zeros(3)
        auxG = ft_ao.ft_ao(auxcell, Gv).T
        wcoulG = pbctools.get_coulG(auxcell, kpt, mesh=cell.mesh) * kws
        ref = lib.dot(auxG.conj()*wcoulG, auxG.T)
        self.assertAlmostEqual(abs(ref - j2c).max(), 0, 9)

        aopair = ft_ao.ft_aopair(cell, Gv, aosym='s2')
        ngrids = Gv.shape[0]
        j3c = lib.dot(auxG.conj()*wcoulG, aopair.reshape(ngrids,-1))
        j2c = scipy.linalg.cholesky(j2c[0], lower=True)
        ref = scipy.linalg.solve_triangular(j2c, j3c, lower=True)
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2', j_only=True)
            v1 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(abs(ref - v1).max(), 0, 8)

########## SR #########

    def test_get_2c2e_gamma_sr(self):
        dfbuilder = rsdf_builder._RSGDFBuilder(cell_sr, auxcell_sr).build()
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))
        self.assertAlmostEqual(lib.fp(j2c), 0.6260180506383916, 9)

        dfbuilder.exclude_d_aux = False
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))
        self.assertAlmostEqual(lib.fp(j2c), 0.6260180506383916, 9)

    def test_get_2c2e_sr(self):
        dfbuilder = rsdf_builder._RSGDFBuilder(cell_sr, auxcell_sr, kpts).build()
        j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), 1.1905457329390845-0.0017081567010781827j, 9)
        self.assertAlmostEqual(lib.fp(j2c[0]), 0.6260180506383916, 9)

    def test_get_2c2e_cart_sr(self):
        with lib.temporary_env(cell_sr, cart=True):
            dfbuilder = rsdf_builder._RSGDFBuilder(cell_sr, auxcell_sr, kpts).build()
            j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), 1.1905457329390845-0.0017081567010781827j, 9)

    def test_make_j3c_gamma_sr(self):
        dfbuilder = rsdf_builder._RSGDFBuilder(cell_sr, auxcell_sr).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name)
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 1.0826993588706444, 8)

            dfbuilder.make_j3c(tmpf.name, aosym='s1')
            v1 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(abs(v1 - lib.unpack_tril(v2).reshape(v1.shape)).max(), 0, 9)

            dfbuilder.exclude_dd_block = True
            dfbuilder.exclude_d_aux = False
            dfbuilder.make_j3c(tmpf.name)
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 1.0826993588706444, 8)

            dfbuilder.exclude_dd_block = False
            dfbuilder.exclude_d_aux = True
            dfbuilder.make_j3c(tmpf.name)
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 1.0826993588706444, 8)

            dfbuilder.exclude_dd_block = False
            dfbuilder.exclude_d_aux = False
            dfbuilder.make_j3c(tmpf.name)
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 1.0826993588706444, 7)

    def test_make_j3c_sr(self):
        dfbuilder = rsdf_builder._RSGDFBuilder(cell_sr, auxcell_sr, kpts).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v_s2 = []
            for ki in range(nkpts):
                for kj in range(nkpts):
                    v_s2.append(load(tmpf.name, kpts[[ki, kj]]))
            self.assertAlmostEqual(lib.fp(v_s2[0]), 1.0826993588706444, 8)
            self.assertAlmostEqual(lib.fp(v_s2[2*nkpts+4]), 3.4195194139440113-0.00011465992631688187j, 8)
            self.assertAlmostEqual(lib.fp(v_s2[2*nkpts+2]), 1.060425082568402+0j, 9)

            dfbuilder.make_j3c(tmpf.name, aosym='s1')
            for ki in range(nkpts):
                for kj in range(nkpts):
                    v1 = load(tmpf.name, kpts[[ki, kj]])
                    if ki == kj:
                        v2 = lib.unpack_tril(v_s2[ki*nkpts+kj]).reshape(v1.shape)
                        self.assertAlmostEqual(abs(v1 - v2).max(), 0, 8)
                    else:
                        self.assertAlmostEqual(abs(v1 - v_s2[ki*nkpts+kj]).max(), 0, 8)

    def test_make_j3c_j_only_sr(self):
        dfbuilder = rsdf_builder._RSGDFBuilder(cell_sr, auxcell_sr, kpts).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2', j_only=True)
            v_s2 = []
            for ki in range(nkpts):
                v_s2.append(load(tmpf.name, kpts[[ki, ki]]))
            self.assertAlmostEqual(lib.fp(v_s2[0]), 1.0826993588706444, 8)
            self.assertAlmostEqual(lib.fp(v_s2[2]), 1.060425082568402+0j, 9)

            dfbuilder.make_j3c(tmpf.name, aosym='s1', j_only=True)
            for ki in range(nkpts):
                v1 = load(tmpf.name, kpts[[ki, ki]])
                v2 = lib.unpack_tril(v_s2[ki]).reshape(v1.shape)
                self.assertAlmostEqual(abs(v1 - v2).max(), 0, 9)

    def test_vs_fft_sr(self):
        cell_sr = pgto.M(
            a = np.eye(3) * 2.8,
            atom = 'He    0.    2.2      1.; He    1.    1.       1.',
            basis = [[0, [1.2, 1.], [.7, .5], [0.4, .5]], [1, [1.1, .5], [0.4, .5]]],
            mesh = [14] * 3,
            verbose = 0,
        )
        cell_sr.omega = -0.9
        auxcell_sr = df.make_auxcell(cell_sr,
            auxbasis=[[0, [1.2, 1.], [.7, .5], [0.4, .5]],
                      [1, [1.1, .5], [0.4, .5]],
                      [2, [1., 1.]],
                     ],
        )

        kpts = np.zeros((1, 3))
        dfbuilder = rsdf_builder._RSGDFBuilder(cell_sr, auxcell_sr, kpts)
        dfbuilder.build()
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))

        Gv, Gvbase, kws = cell_sr.get_Gv_weights()
        kpt = np.zeros(3)
        auxG = ft_ao.ft_ao(auxcell_sr, Gv).T
        wcoulG = pbctools.get_coulG(auxcell_sr, kpt, mesh=cell_sr.mesh,
                                    omega=cell_sr.omega) * kws
        ref = lib.dot(auxG.conj()*wcoulG, auxG.T)
        self.assertAlmostEqual(abs(ref - j2c).max(), 0, 9)

        aopair = ft_ao.ft_aopair(cell_sr, Gv, aosym='s2')
        ngrids = Gv.shape[0]
        j3c = lib.dot(auxG.conj()*wcoulG, aopair.reshape(ngrids,-1))
        j2c = scipy.linalg.cholesky(j2c[0], lower=True)
        ref = scipy.linalg.solve_triangular(j2c, j3c, lower=True)
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2', j_only=True)
            v1 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(abs(ref - v1).max(), 0, 8)

if __name__ == '__main__':
    print("Full Tests for rsdf_builder")
    unittest.main()
