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
from pyscf.pbc.df import df, aug_etb, FFTDF
from pyscf.pbc.df import gdf_builder
from pyscf.pbc.df import ft_ao
from pyscf.pbc.tools import pbc as pbctools
from pyscf.df.incore import cholesky_eri
pyscf.pbc.DEBUG = False

def setUpModule():
    global cell, auxcell, cell_lr, auxcell_lr, cell_sr, auxcell_sr
    global basis, auxbasis, kpts, nkpts
    basis = '''
    He    S
         38.00    0.05
          5.00    0.25
          0.20    0.60
    He    S
          0.25    1.00
    He    P
          1.27    1.00
       '''
    auxbasis = '''
    He    S
         50.60   0.06
         12.60   0.21
          3.80   0.37
    He    S
          1.40   0.29
    He    S
          0.30   0.06
    He    P
          4.00   1.00
          1.00   1.00
    He    D
          4.00   1.00
    '''
    cell = pgto.M(
        a = np.eye(3) * 3.5,
        mesh = [11] * 3,
        atom = '''He    3.    2.       3.
                  He    1.    1.       1.''',
        basis = basis,
        verbose = 0,
        max_memory = 1000,
        precision = 1e-9,
    )

    kpts = cell.make_kpts([3,5,6])[[0, 2, 3, 4, 6, 12, 20]]
    kpts[3] = kpts[0]-kpts[1]+kpts[2]
    nkpts = len(kpts)

    auxcell = df.make_auxcell(cell, auxbasis)

    cell_lr = cell.copy()
    cell_lr.omega = 1.2
    auxcell_lr = df.make_auxcell(cell_lr, auxbasis)

    cell_sr = cell.copy()
    cell_sr.omega = -1.2
    auxcell_sr = df.make_auxcell(cell_sr, auxbasis)

def load(filename, kptij):
    with df._load3c(filename, 'j3c', kptij) as cderi:
        return cderi[:]

def tearDownModule():
    global cell, auxcell, cell_lr, auxcell_lr, cell_sr, auxcell_sr
    del cell, auxcell, cell_lr, auxcell_lr, cell_sr, auxcell_sr


class KnownValues(unittest.TestCase):
    def test_get_2c2e_gamma(self):
        dfbuilder = gdf_builder._CCGDFBuilder(cell, auxcell).build()
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))
        self.assertAlmostEqual(lib.fp(j2c), 0.7120733223146716, 9)

    def test_get_2c2e(self):
        dfbuilder = gdf_builder._CCGDFBuilder(cell, auxcell, kpts).build()
        j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), -1.6451684819960948+2.889508819643691j, 9)
        self.assertAlmostEqual(lib.fp(j2c[0]), 0.7120733223146716, 9)

    def test_get_2c2e_cart(self):
        with lib.temporary_env(cell, cart=True):
            dfbuilder = gdf_builder._CCGDFBuilder(cell, auxcell, kpts).build()
            j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), -1.6451684819960948+2.889508819643691j, 9)

    def test_make_j3c_gamma(self):
        dfbuilder = gdf_builder._CCGDFBuilder(cell, auxcell).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name)
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 1.5094843470069796, 7)

            dfbuilder.make_j3c(tmpf.name, aosym='s1')
            v1 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(abs(v1 - lib.unpack_tril(v2).reshape(v1.shape)).max(), 0, 9)

            dfbuilder.exclude_dd_block = False
            dfbuilder.make_j3c(tmpf.name)
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 1.5094843470069796, 6)

    def test_make_j3c(self):
        dfbuilder = gdf_builder._CCGDFBuilder(cell, auxcell, kpts).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v_s2 = []
            for ki in range(nkpts):
                for kj in range(nkpts):
                    v_s2.append(load(tmpf.name, kpts[[ki, kj]]))
            self.assertAlmostEqual(lib.fp(v_s2[0]), 1.5094843470069796, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2*nkpts+4]), 3.8063416643507173+0.08901920438689674j, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2*nkpts+2]), 1.2630074629589676+0j, 7)

            dfbuilder.make_j3c(tmpf.name, aosym='s1')
            with df.CDERIArray(tmpf.name) as cderi_array:
                v_s1 = cderi_array[:]
            for ki in range(nkpts):
                for kj in range(nkpts):
                    v1 = v_s1[ki,kj]
                    if ki == kj:
                        v2 = lib.unpack_tril(v_s2[ki*nkpts+kj]).reshape(v1.shape)
                        self.assertAlmostEqual(abs(v1 - v2).max(), 0, 9)
                    else:
                        self.assertAlmostEqual(abs(v1 - v_s2[ki*nkpts+kj]).max(), 0, 9)

    def test_make_j3c_j_only(self):
        dfbuilder = gdf_builder._CCGDFBuilder(cell, auxcell, kpts).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2', j_only=True)
            v_s2 = []
            for ki in range(nkpts):
                v_s2.append(load(tmpf.name, kpts[[ki, ki]]))
            self.assertAlmostEqual(lib.fp(v_s2[0]), 1.5094843470069796, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2]), 1.2630074629589676+0j, 7)

            dfbuilder.make_j3c(tmpf.name, aosym='s1', j_only=True)
            for ki in range(nkpts):
                v1 = load(tmpf.name, kpts[[ki, ki]])
                v2 = lib.unpack_tril(v_s2[ki]).reshape(v1.shape)
                self.assertAlmostEqual(abs(v1 - v2).max(), 0, 9)

    def test_make_j3c_gamma_2d(self):
        cell = pgto.M(atom='He 0 0 0; He 0.9 0 0',
                      basis=basis,
                      a='2.8 0 0; 0 2.8 0; 0 0 15',
                      precision=1e-9,
                      dimension=2)
        auxcell = df.make_auxcell(cell, auxbasis)
        dfbuilder = gdf_builder._CCGDFBuilder(cell, auxcell).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name)
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2.T.dot(v2)), 0.3289627476345819, 8)

    def test_make_j3c_gamma_1d(self):
        cell = pgto.M(atom='He 0 0 0; He 0.9 0 0',
                      basis=basis,
                      a=np.eye(3) * 2.8,
                      dimension=1)
        auxcell = df.make_auxcell(cell, auxbasis)
        dfbuilder = gdf_builder._CCGDFBuilder(cell, auxcell).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name)
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 1.7171975296579753, 6)

    def test_make_j3c_gamma_0d(self):
        from pyscf.df.incore import cholesky_eri
        cell = pgto.M(atom='He 0 0 0; He 0.9 0 0',
                      basis=basis,
                      a=np.eye(3) * 2.8,
                      dimension=0)
        auxcell = df.make_auxcell(cell, auxbasis)
        dfbuilder = gdf_builder._CCGDFBuilder(cell, auxcell).build()
        ref = cholesky_eri(cell, auxmol=auxcell)
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name)
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(abs(v2 - ref).max(), 0, 9)
            self.assertAlmostEqual(lib.fp(v2), 2.4119378731259946, 2)

        ref = cholesky_eri(cell, auxmol=auxcell)
        self.assertAlmostEqual(abs(v2-ref).max(), 0, 2)

    def test_get_nuc(self):
        dfbuilder = gdf_builder._CCNucBuilder(cell).build()
        v1 = dfbuilder.get_nuc()
        self.assertAlmostEqual(lib.fp(v1), -2.9338697931882134, 6)

    def test_get_nuc_2d(self):
        a = np.eye(3) * 2.8
        a[2,2] = 10.
        cell = pgto.M(atom='He 0 0 0; He 0.9 0 0',
                      basis=basis, a=a, dimension=2)
        dfbuilder = gdf_builder._CCNucBuilder(cell).build()
        v1 = dfbuilder.get_nuc()
        self.assertAlmostEqual(lib.fp(v1), -2.9494363868337388, 6)

    def test_get_nuc_0d(self):
        cell = pgto.M(atom='He 0 0 0; He 0.9 0 0',
                      basis=basis,
                      a=np.eye(3) * 2.8,
                      dimension=0)
        ref = cell.to_mol().intor('int1e_nuc')
        dfbuilder = gdf_builder._CCNucBuilder(cell).build()
        v1 = dfbuilder.get_nuc()
        self.assertAlmostEqual(abs(v1-ref).max(), 0, 9)

    def test_get_pp(self):
        L = 7
        a = np.eye(3) * L
        a[1,0] = 5.0
        cell = pgto.M(atom=[['Be', (L/2.,  L/2., L/2.)]],
                      a=a, basis='gth-szv', pseudo='gth-pade-q2')
        dfbuilder = gdf_builder._CCNucBuilder(cell).build()
        vpp = dfbuilder.get_pp()
        self.assertAlmostEqual(lib.fp(vpp), -0.34980233064594995, 7)

        kpts = cell.make_kpts([3,3,2])
        dfbuilder = gdf_builder._CCNucBuilder(cell, kpts).build()
        vpp = dfbuilder.get_pp()
        self.assertAlmostEqual(lib.fp(vpp), 0.08279960176438528+0j, 7)

    def test_vs_fft(self):
        cell = pgto.M(
            a = np.eye(3) * 2.8,
            atom = 'He    0.    2.2      1.; He    1.    1.       1.',
            basis = [[0, [1.2, 1.], [.7, .5], [0.4, .5]], [1, [1.1, .5], [0.4, .5]]],
            mesh = [15] * 3,
            verbose = 0,
        )
        auxcell = df.make_auxcell(cell,
            auxbasis=[[0, [1.2, 1.], [.7, .5], [0.4, .5]],
                      [1, [1.1, .5], [0.4, .5]],
                      [2, [1., 1.]],
                     ],
        )

        kpts = np.zeros((1, 3))
        dfbuilder = gdf_builder._CCGDFBuilder(cell, auxcell, kpts)
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
            self.assertAlmostEqual(abs(ref - v1).max(), 0, 7)

########### LR #########

    def test_get_2c2e_gamma_lr(self):
        dfbuilder = gdf_builder._CCGDFBuilder(cell_lr, auxcell_lr).build()
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))
        self.assertAlmostEqual(lib.fp(j2c), 0.07120226186395502, 9)

    def test_get_2c2e_lr(self):
        dfbuilder = gdf_builder._CCGDFBuilder(cell_lr, auxcell_lr, kpts).build()
        j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), -3.3974428735809807+2.8932134194401593j, 9)
        self.assertAlmostEqual(lib.fp(j2c[0]), 0.07120226186395502, 9)

    def test_get_2c2e_cart_lr(self):
        with lib.temporary_env(cell_lr, cart=True):
            dfbuilder = gdf_builder._CCGDFBuilder(cell_lr, auxcell_lr, kpts).build()
            j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), -3.3974428735809807+2.8932134194401593j, 9)

    def test_make_j3c_gamma_lr(self):
        dfbuilder = gdf_builder._CCGDFBuilder(cell_lr, auxcell_lr).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name)
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 1.0942903795950072, 7)

            dfbuilder.make_j3c(tmpf.name, aosym='s1')
            v1 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(abs(v1 - lib.unpack_tril(v2).reshape(v1.shape)).max(), 0, 9)

            dfbuilder.exclude_dd_block = False
            dfbuilder.make_j3c(tmpf.name)
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 1.0942903795950072, 6)

    def test_make_j3c_lr(self):
        dfbuilder = gdf_builder._CCGDFBuilder(cell_lr, auxcell_lr, kpts).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v_s2 = []
            for ki in range(nkpts):
                for kj in range(nkpts):
                    v_s2.append(load(tmpf.name, kpts[[ki, kj]]))
            self.assertAlmostEqual(lib.fp(v_s2[0]), 1.0942903795950072, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2*nkpts+4]), (2.701808484279018+0.09199879476151751j), 7)
            self.assertAlmostEqual(lib.fp(v_s2[2*nkpts+2]), 0.8939154964856898+0j, 7)

            dfbuilder.make_j3c(tmpf.name, aosym='s1')
            with df.CDERIArray(tmpf.name) as cderi_array:
                for ki in range(nkpts):
                    for kj in range(nkpts):
                        v1 = cderi_array[ki, kj]
                        if ki == kj:
                            v2 = lib.unpack_tril(v_s2[ki*nkpts+kj]).reshape(v1.shape)
                            self.assertAlmostEqual(abs(v1 - v2).max(), 0, 9)
                        else:
                            self.assertAlmostEqual(abs(v1 - v_s2[ki*nkpts+kj]).max(), 0, 9)

    def test_make_j3c_j_only_lr(self):
        dfbuilder = gdf_builder._CCGDFBuilder(cell_lr, auxcell_lr, kpts).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2', j_only=True)
            v_s2 = []
            for ki in range(nkpts):
                v_s2.append(load(tmpf.name, kpts[[ki, ki]]))
            self.assertAlmostEqual(lib.fp(v_s2[0]), 1.0942903795950072, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2]), 0.8939154964856898+0j, 7)

            dfbuilder.make_j3c(tmpf.name, aosym='s1', j_only=True)
            for ki in range(nkpts):
                v1 = load(tmpf.name, kpts[[ki, ki]])
                v2 = lib.unpack_tril(v_s2[ki]).reshape(v1.shape)
                self.assertAlmostEqual(abs(v1 - v2).max(), 0, 9)

    def test_vs_fft_lr(self):
        cell_lr = pgto.M(
            a = np.eye(3) * 2.8,
            atom = 'He    0.    2.2      1.; He    1.    1.       1.',
            basis = [[0, [1.2, 1.], [.7, .5], [0.4, .5]], [1, [1.1, .5], [0.4, .5]]],
            mesh = [15] * 3,
            precision = 1e-9,
            verbose = 0,
        )
        cell_lr.omega = 0.9
        auxcell_lr = df.make_auxcell(cell_lr,
            auxbasis=[[0, [1.2, 1.], [.7, .5], [0.4, .5]],
                      [1, [1.1, .5], [0.4, .5]],
                      [2, [1., 1.]],
                     ],
        )

        kpts = np.zeros((1, 3))
        dfbuilder = gdf_builder._CCGDFBuilder(cell_lr, auxcell_lr, kpts)
        dfbuilder.build()
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))

        Gv, Gvbase, kws = cell_lr.get_Gv_weights()
        kpt = np.zeros(3)
        auxG = ft_ao.ft_ao(auxcell_lr, Gv).T
        wcoulG = pbctools.get_coulG(auxcell_lr, kpt, mesh=cell_lr.mesh,
                                    omega=cell_lr.omega) * kws
        ref = lib.dot(auxG.conj()*wcoulG, auxG.T)
        self.assertAlmostEqual(abs(ref - j2c).max(), 0, 9)

        aopair = ft_ao.ft_aopair(cell_lr, Gv, aosym='s2')
        ngrids = Gv.shape[0]
        j3c = lib.dot(auxG.conj()*wcoulG, aopair.reshape(ngrids,-1))
        j2c = scipy.linalg.cholesky(j2c[0], lower=True)
        ref = scipy.linalg.solve_triangular(j2c, j3c, lower=True)
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2', j_only=True)
            v1 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(abs(ref - v1).max(), 0, 7)

########## SR #########

    def test_get_2c2e_gamma_sr(self):
        dfbuilder = gdf_builder._CCGDFBuilder(cell_sr, auxcell_sr).build()
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))
        self.assertAlmostEqual(lib.fp(j2c), 0.6408710604507251, 9)

    def test_get_2c2e_sr(self):
        dfbuilder = gdf_builder._CCGDFBuilder(cell_sr, auxcell_sr, kpts).build()
        j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), 1.7522743915836274-0.0037045997964663922j, 9)
        self.assertAlmostEqual(lib.fp(j2c[0]), 0.6408710604507251, 9)

    def test_get_2c2e_cart_sr(self):
        with lib.temporary_env(cell_sr, cart=True):
            dfbuilder = gdf_builder._CCGDFBuilder(cell_sr, auxcell_sr, kpts).build()
            j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), 1.7522743915836274-0.0037045997964663922j, 9)

    def test_make_j3c_gamma_sr(self):
        dfbuilder = gdf_builder._CCGDFBuilder(cell_sr, auxcell_sr).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name)
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 0.9647178630555139, 7)

            dfbuilder.make_j3c(tmpf.name, aosym='s1')
            v1 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(abs(v1 - lib.unpack_tril(v2).reshape(v1.shape)).max(), 0, 9)

            dfbuilder.exclude_dd_block = False
            dfbuilder.make_j3c(tmpf.name)
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 0.9647178630555139, 6)

    def test_make_j3c_sr(self):
        dfbuilder = gdf_builder._CCGDFBuilder(cell_sr, auxcell_sr, kpts).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v_s2 = []
            for ki in range(nkpts):
                for kj in range(nkpts):
                    v_s2.append(load(tmpf.name, kpts[[ki, kj]]))
            self.assertAlmostEqual(lib.fp(v_s2[0]), 0.9647178630555139, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2*nkpts+4]), (2.5461640768180303-0.0031169483145069028j), 8)
            self.assertAlmostEqual(lib.fp(v_s2[2*nkpts+2]), 0.8297008398488194+0j, 8)

            dfbuilder.make_j3c(tmpf.name, aosym='s1')
            with df.CDERIArray(tmpf.name) as cderi_array:
                v_s1 = cderi_array[:]
            for ki in range(nkpts):
                for kj in range(nkpts):
                    v1 = v_s1[ki,kj]
                    if ki == kj:
                        v2 = lib.unpack_tril(v_s2[ki*nkpts+kj]).reshape(v1.shape)
                        self.assertAlmostEqual(abs(v1 - v2).max(), 0, 9)
                    else:
                        self.assertAlmostEqual(abs(v1 - v_s2[ki*nkpts+kj]).max(), 0, 9)

    def test_make_j3c_j_only_sr(self):
        dfbuilder = gdf_builder._CCGDFBuilder(cell_sr, auxcell_sr, kpts).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2', j_only=True)
            v_s2 = []
            for ki in range(nkpts):
                v_s2.append(load(tmpf.name, kpts[[ki, ki]]))
            self.assertAlmostEqual(lib.fp(v_s2[0]), 0.9647178630555139, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2]), 0.8297008398488194+0j, 7)

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
            mesh = [15] * 3,
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
        dfbuilder = gdf_builder._CCGDFBuilder(cell_sr, auxcell_sr, kpts)
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
            self.assertAlmostEqual(abs(ref - v1).max(), 0, 7)

if __name__ == '__main__':
    print("Full Tests for gdf_builder")
    unittest.main()
