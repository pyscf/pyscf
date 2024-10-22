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
from pyscf import ao2mo
import pyscf.pbc
from pyscf.pbc import gto as pgto
from pyscf.pbc.df import df, FFTDF
from pyscf.pbc.df import mdf
from pyscf.pbc.df import ft_ao
from pyscf.pbc.tools import pbc as pbctools
pyscf.pbc.DEBUG = False

def setUpModule():
    global cell, auxcell, cell_lr, auxcell_lr, cell_sr, auxcell_sr, kpts, nkpts, mesh
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
        atom = '''He    3.    2.       3.
                  He    1.    1.       1.''',
        basis = basis,
        verbose = 7,
        output = '/dev/null',
        max_memory = 1000,
        precision = 1e-9,
    )

    kpts = cell.make_kpts([3,5,6])[[0, 2, 3, 4, 6, 12, 20]]
    kpts[3] = kpts[0]-kpts[1]+kpts[2]
    nkpts = len(kpts)
    mesh = [11] * 3

    auxcell = df.make_auxcell(cell, auxbasis)

    cell_lr = cell.copy()
    cell_lr.omega = 1.2
    auxcell_lr = df.make_auxcell(cell_lr, auxbasis)

    cell_sr = cell.copy()
    cell_sr.omega = -1.2
    auxcell_sr = df.make_auxcell(cell_sr, auxbasis)

def load(filename, kptij):
    with df._load3c(filename, 'j3c', kptij) as cderi:
        v = cderi[:]
    return v.conj().T.dot(v)

def tearDownModule():
    global cell, auxcell, cell_lr, auxcell_lr, cell_sr, auxcell_sr
    del cell, auxcell, cell_lr, auxcell_lr, cell_sr, auxcell_sr


class KnownValues(unittest.TestCase):
    def test_ccmdf_get_2c2e_gamma(self):
        dfbuilder = mdf._CCMDFBuilder(cell, auxcell).set(mesh=mesh).build()
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))
        self.assertAlmostEqual(lib.fp(j2c), 0.010615787015170507, 9)

    def test_ccmdf_get_2c2e(self):
        dfbuilder = mdf._CCMDFBuilder(cell, auxcell, kpts).set(mesh=mesh).build()
        j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), 0.09006694627014439+0.014552394902466072j, 9)
        self.assertAlmostEqual(lib.fp(j2c[0]), 0.010615787015170507, 9)

    def test_ccmdf_get_2c2e_cart(self):
        with lib.temporary_env(cell, cart=True):
            dfbuilder = mdf._CCMDFBuilder(cell, auxcell, kpts).set(mesh=mesh).build()
            j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), 0.09006694627014439+0.014552394902466072j, 9)

    def test_ccmdf_make_j3c_gamma(self):
        dfbuilder = mdf._CCMDFBuilder(cell, auxcell).set(mesh=mesh).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 0.01486794482668373, 7)

            dfbuilder.make_j3c(tmpf.name, aosym='s1')
            v1 = load(tmpf.name, kpts[[0, 0]])
            v2 = ao2mo.restore(1, v2, cell.nao).reshape(v1.shape)
            self.assertAlmostEqual(abs(v1 - v2).max(), 0, 9)

            dfbuilder.exclude_dd_block = False
            dfbuilder.build()
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 0.01486794482668373, 7)

    def test_ccmdf_make_j3c(self):
        dfbuilder = mdf._CCMDFBuilder(cell, auxcell, kpts).set(mesh=mesh).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v_s2 = []
            for ki in range(nkpts):
                for kj in range(nkpts):
                    v_s2.append(load(tmpf.name, kpts[[ki, kj]]))
            self.assertAlmostEqual(lib.fp(v_s2[0]), 0.01486794482668373, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2*nkpts+4]), 0.016564674723605698+0.0005557577420328895j, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2*nkpts+2]), 0.01509142892728263+0j, 7)

            dfbuilder.make_j3c(tmpf.name, aosym='s1')
            idx, idy = np.tril_indices(cell.nao)
            idx = idx * cell.nao + idy
            for ki in range(nkpts):
                for kj in range(nkpts):
                    v1 = load(tmpf.name, kpts[[ki, kj]])
                    if ki == kj:
                        v1 = v1[idx[:,None], idx]
                    self.assertAlmostEqual(abs(v1 - v_s2[ki*nkpts+kj]).max(), 0, 9)

    def test_ccmdf_make_j3c_j_only(self):
        dfbuilder = mdf._CCMDFBuilder(cell, auxcell, kpts).set(mesh=mesh).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2', j_only=True)
            v_s2 = []
            for ki in range(nkpts):
                v_s2.append(load(tmpf.name, kpts[[ki, ki]]))
            self.assertAlmostEqual(lib.fp(v_s2[0]), 0.01486794482668373, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2]), 0.01509142892728263+0j, 7)

            dfbuilder.make_j3c(tmpf.name, aosym='s1', j_only=True)
            idx, idy = np.tril_indices(cell.nao)
            idx = idx * cell.nao + idy
            for ki in range(nkpts):
                v1 = load(tmpf.name, kpts[[ki, ki]])
                v1 = v1[idx[:,None], idx]
                self.assertAlmostEqual(abs(v1 - v_s2[ki]).max(), 0, 9)

    def test_ccmdf_vs_fft(self):
        cell = pgto.M(
            a = np.eye(3) * 2.6,
            atom = 'He    0.    2.2      1.; He    1.    1.       1.',
            basis = [[0, [1.2, 1.], [.7, .5], [0.4, .5]], [1, [1.1, .5], [0.3, .5]]],
            mesh = [14] * 3,
            verbose = 0,
            precision = 1e-9,
        )
        auxcell = df.make_auxcell(cell,
            auxbasis=[[0, [1.2, 1.], [.7, .5], [0.4, .5]],
                      [1, [1.1, .5], [0.4, .5]],
                      [2, [1., 1.]],
                     ],
        )

        dfbuilder = mdf._CCMDFBuilder(cell, auxcell, kpts)
        dfbuilder.build()
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))

        Gv, Gvbase, kws = cell.get_Gv_weights()
        kpt = np.zeros(3)
        auxG = ft_ao.ft_ao(auxcell, Gv).T
        coulG = pbctools.get_coulG(auxcell, kpt, mesh=cell.mesh) * kws
        Gv1, Gvbase1, kws1 = cell.get_Gv_weights(mesh=dfbuilder.mesh)
        auxG1 = ft_ao.ft_ao(auxcell, Gv1).T
        coulG1 = pbctools.get_coulG(auxcell, kpt, mesh=dfbuilder.mesh) * kws1
        ref = lib.dot(auxG.conj()*coulG, auxG.T) - lib.dot(auxG1.conj()*coulG1, auxG1.T)
        self.assertAlmostEqual(abs(ref - j2c[0]).max(), 0, 9)

        j3c = (lib.dot(auxG.conj()*coulG, ft_ao.ft_aopair(cell, Gv, aosym='s2')) -
               lib.dot(auxG1.conj()*coulG1, ft_ao.ft_aopair(cell, Gv1, aosym='s2')))
        j2c = dfbuilder.eigenvalue_decomposed_metric(j2c[0])
        ref = lib.dot(j2c[0], j3c)
        ref = ref.T.dot(ref)
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2', j_only=True)
            v1 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(abs(ref - v1).max(), 0, 8)

    def test_rsmdf_get_2c2e_gamma(self):
        dfbuilder = mdf._RSMDFBuilder(cell, auxcell).set(mesh=mesh).build()
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))
        self.assertAlmostEqual(lib.fp(j2c), 0.010615787015170507, 7)

    def test_rsmdf_get_2c2e(self):
        dfbuilder = mdf._RSMDFBuilder(cell, auxcell, kpts).set(mesh=mesh).build()
        j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), 0.09006694627014439+0.014552394902466072j, 7)
        self.assertAlmostEqual(lib.fp(j2c[0]), 0.010615787015170507, 7)

    def test_rsmdf_get_2c2e_cart(self):
        with lib.temporary_env(cell, cart=True):
            dfbuilder = mdf._RSMDFBuilder(cell, auxcell, kpts).set(mesh=mesh).build()
            j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), 0.09006694627014439+0.014552394902466072j, 7)

    def test_rsmdf_make_j3c_gamma(self):
        dfbuilder = mdf._RSMDFBuilder(cell, auxcell).set(mesh=mesh).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 0.01486794482668373, 7)

            dfbuilder.make_j3c(tmpf.name, aosym='s1')
            v1 = load(tmpf.name, kpts[[0, 0]])
            v2 = ao2mo.restore(1, v2, cell.nao).reshape(v1.shape)
            self.assertAlmostEqual(abs(v1 - v2).max(), 0, 9)

            dfbuilder.exclude_dd_block = True
            dfbuilder.exclude_d_aux = False
            dfbuilder.build()
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 0.01486794482668373, 7)

            dfbuilder.exclude_dd_block = False
            dfbuilder.exclude_d_aux = True
            dfbuilder.build()
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 0.01486794482668373, 7)

            dfbuilder.exclude_dd_block = False
            dfbuilder.exclude_d_aux = False
            dfbuilder.build()
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 0.01486794482668373, 7)

    def test_rsmdf_make_j3c(self):
        dfbuilder = mdf._RSMDFBuilder(cell, auxcell, kpts).set(mesh=mesh).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v_s2 = []
            for ki in range(nkpts):
                for kj in range(nkpts):
                    v_s2.append(load(tmpf.name, kpts[[ki, kj]]))
            self.assertAlmostEqual(lib.fp(v_s2[0]), 0.01486794482668373, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2*nkpts+4]), 0.016564674723605698+0.0005557577420328895j, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2*nkpts+2]), 0.01509142892728263+0j, 6)

            dfbuilder.make_j3c(tmpf.name, aosym='s1')
            idx, idy = np.tril_indices(cell.nao)
            idx = idx * cell.nao + idy
            for ki in range(nkpts):
                for kj in range(nkpts):
                    v1 = load(tmpf.name, kpts[[ki, kj]])
                    if ki == kj:
                        v1 = v1[idx[:,None], idx]
                    self.assertAlmostEqual(abs(v1 - v_s2[ki*nkpts+kj]).max(), 0, 9)

    def test_rsmdf_make_j3c_j_only(self):
        dfbuilder = mdf._RSMDFBuilder(cell, auxcell, kpts).set(mesh=mesh).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2', j_only=True)
            v_s2 = []
            for ki in range(nkpts):
                v_s2.append(load(tmpf.name, kpts[[ki, ki]]))
            self.assertAlmostEqual(lib.fp(v_s2[0]), 0.01486794482668373, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2]), 0.01509142892728263+0j, 6)

            dfbuilder.make_j3c(tmpf.name, aosym='s1', j_only=True)
            idx, idy = np.tril_indices(cell.nao)
            idx = idx * cell.nao + idy
            for ki in range(nkpts):
                v1 = load(tmpf.name, kpts[[ki, ki]])
                v1 = v1[idx[:,None], idx]
                self.assertAlmostEqual(abs(v1 - v_s2[ki]).max(), 0, 9)

#### Test _CCMDFBuilder with omega > 0 ####

    def test_ccmdf_get_2c2e_gamma_lr(self):
        dfbuilder = mdf._CCMDFBuilder(cell_lr, auxcell_lr).set(mesh=mesh).build()
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))
        self.assertAlmostEqual(lib.fp(j2c), -3.0980016665293594e-05, 9)

    def test_ccmdf_get_2c2e_lr(self):
        dfbuilder = mdf._CCMDFBuilder(cell_lr, auxcell_lr, kpts).set(mesh=mesh).build()
        j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), -0.00014085584845558628+6.0044990636448106e-05j, 9)
        self.assertAlmostEqual(lib.fp(j2c[0]), -3.0980016665293594e-05, 9)

    def test_ccmdf_get_2c2e_cart_lr(self):
        with lib.temporary_env(cell_lr, cart=True):
            dfbuilder = mdf._CCMDFBuilder(cell_lr, auxcell_lr, kpts).set(mesh=mesh).build()
            j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), -0.00014085584845558628+6.0044990636448106e-05j, 9)

    def test_ccmdf_make_j3c_gamma_lr(self):
        dfbuilder = mdf._CCMDFBuilder(cell_lr, auxcell_lr).set(mesh=mesh).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 1.0439710349332878e-05, 7)

            dfbuilder.make_j3c(tmpf.name, aosym='s1')
            v1 = load(tmpf.name, kpts[[0, 0]])
            v2 = ao2mo.restore(1, v2, cell_lr.nao).reshape(v1.shape)
            self.assertAlmostEqual(abs(v1 - v2).max(), 0, 9)

            dfbuilder.exclude_dd_block = False
            dfbuilder.build()
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 1.0439710349332878e-05, 7)

    def test_ccmdf_make_j3c_lr(self):
        dfbuilder = mdf._CCMDFBuilder(cell_lr, auxcell_lr, kpts).set(mesh=mesh).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v_s2 = []
            for ki in range(nkpts):
                for kj in range(nkpts):
                    v_s2.append(load(tmpf.name, kpts[[ki, kj]]))
            self.assertAlmostEqual(lib.fp(v_s2[0]), 1.0439710349332878e-05, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2*nkpts+4]), -1.4267057869237345e-05+1.6315647164194292e-06j, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2*nkpts+2]), 1.062711221387176e-05+0j, 7)

            dfbuilder.make_j3c(tmpf.name, aosym='s1')
            idx, idy = np.tril_indices(cell.nao)
            idx = idx * cell.nao + idy
            for ki in range(nkpts):
                for kj in range(nkpts):
                    v1 = load(tmpf.name, kpts[[ki, kj]])
                    if ki == kj:
                        v1 = v1[idx[:,None], idx]
                    self.assertAlmostEqual(abs(v1 - v_s2[ki*nkpts+kj]).max(), 0, 9)

    def test_ccmdf_make_j3c_j_only_lr(self):
        dfbuilder = mdf._CCMDFBuilder(cell_lr, auxcell_lr, kpts).set(mesh=mesh).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2', j_only=True)
            v_s2 = []
            for ki in range(nkpts):
                v_s2.append(load(tmpf.name, kpts[[ki, ki]]))
            self.assertAlmostEqual(lib.fp(v_s2[0]), 1.0439710349332878e-05, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2]), 1.062711221387176e-05+0j, 7)

            dfbuilder.make_j3c(tmpf.name, aosym='s1', j_only=True)
            idx, idy = np.tril_indices(cell_lr.nao)
            idx = idx * cell.nao + idy
            for ki in range(nkpts):
                v1 = load(tmpf.name, kpts[[ki, ki]])
                v1 = v1[idx[:,None], idx]
                self.assertAlmostEqual(abs(v1 - v_s2[ki]).max(), 0, 9)

    def test_ccmdf_vs_fft_lr(self):
        cell_lr = pgto.M(
            a = np.eye(3) * 2.6,
            atom = 'He    0.    2.2      1.; He    1.    1.       1.',
            basis = [[0, [1.2, 1.], [.7, .5], [0.4, .5]], [1, [1.1, .5], [0.3, .5]]],
            mesh = [14] * 3,
            verbose = 0,
            precision = 1e-9,
        )
        cell_lr.omega = 0.9
        auxcell_lr = df.make_auxcell(cell_lr,
            auxbasis=[[0, [1.2, 1.], [.7, .5], [0.4, .5]],
                      [1, [1.1, .5], [0.4, .5]],
                      [2, [1., 1.]],
                     ],
        )

        dfbuilder = mdf._CCMDFBuilder(cell_lr, auxcell_lr, kpts)
        dfbuilder.mesh = [7] * 3
        dfbuilder.build()
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))

        Gv, Gvbase, kws = cell_lr.get_Gv_weights()
        kpt = np.zeros(3)
        auxG = ft_ao.ft_ao(auxcell_lr, Gv).T
        coulG = pbctools.get_coulG(auxcell_lr, kpt, mesh=cell_lr.mesh,
                                   omega=cell_lr.omega) * kws
        Gv1, Gvbase1, kws1 = cell_lr.get_Gv_weights(mesh=dfbuilder.mesh)
        auxG1 = ft_ao.ft_ao(auxcell_lr, Gv1).T
        coulG1 = pbctools.get_coulG(auxcell_lr, kpt, mesh=dfbuilder.mesh,
                                    omega=cell_lr.omega) * kws1
        ref = lib.dot(auxG.conj()*coulG, auxG.T) - lib.dot(auxG1.conj()*coulG1, auxG1.T)
        self.assertAlmostEqual(abs(ref - j2c[0]).max(), 0, 9)

        j3c = (lib.dot(auxG.conj()*coulG, ft_ao.ft_aopair(cell_lr, Gv, aosym='s2')) -
               lib.dot(auxG1.conj()*coulG1, ft_ao.ft_aopair(cell_lr, Gv1, aosym='s2')))
        j2c = dfbuilder.eigenvalue_decomposed_metric(j2c[0])
        ref = lib.dot(j2c[0], j3c)
        ref = ref.T.dot(ref)
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2', j_only=True)
            v1 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(abs(ref - v1).max(), 0, 9)

#### Test _CCMDFBuilder with omega < 0 ####

    def test_ccmdf_get_2c2e_gamma_sr(self):
        dfbuilder = mdf._CCMDFBuilder(cell_sr, auxcell_sr).set(mesh=mesh).build()
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))
        self.assertAlmostEqual(lib.fp(j2c), 0.010646767031704538, 9)

    def test_ccmdf_get_2c2e_sr(self):
        dfbuilder = mdf._CCMDFBuilder(cell_sr, auxcell_sr, kpts).set(mesh=mesh).build()
        j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), 0.0902078021182187+0.01449234991201075j, 9)
        self.assertAlmostEqual(lib.fp(j2c[0]), 0.010646767031704538, 9)

    def test_ccmdf_get_2c2e_cart_sr(self):
        with lib.temporary_env(cell_sr, cart=True):
            dfbuilder = mdf._CCMDFBuilder(cell_sr, auxcell_sr, kpts).set(mesh=mesh).build()
            j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), 0.0902078021182187+0.01449234991201075j, 9)

    def test_ccmdf_make_j3c_gamma_sr(self):
        dfbuilder = mdf._CCMDFBuilder(cell_sr, auxcell_sr).set(mesh=mesh).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 0.014857466177913803, 7)

            dfbuilder.make_j3c(tmpf.name, aosym='s1')
            v1 = load(tmpf.name, kpts[[0, 0]])
            v2 = ao2mo.restore(1, v2, cell_sr.nao).reshape(v1.shape)
            self.assertAlmostEqual(abs(v1 - v2).max(), 0, 9)

            dfbuilder.exclude_dd_block = False
            dfbuilder.build()
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 0.014857466177913803, 7)

    def test_ccmdf_make_j3c_sr(self):
        dfbuilder = mdf._CCMDFBuilder(cell_sr, auxcell_sr, kpts).set(mesh=mesh).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v_s2 = []
            for ki in range(nkpts):
                for kj in range(nkpts):
                    v_s2.append(load(tmpf.name, kpts[[ki, kj]]))
            self.assertAlmostEqual(lib.fp(v_s2[0]), 0.014857466177913803, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2*nkpts+4]), 0.016578073155690577+0.0005539785824510145j, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2*nkpts+2]), 0.015080760401588148+0j, 7)

            dfbuilder.make_j3c(tmpf.name, aosym='s1')
            idx, idy = np.tril_indices(cell.nao)
            idx = idx * cell.nao + idy
            for ki in range(nkpts):
                for kj in range(nkpts):
                    v1 = load(tmpf.name, kpts[[ki, kj]])
                    if ki == kj:
                        v1 = v1[idx[:,None], idx]
                    self.assertAlmostEqual(abs(v1 - v_s2[ki*nkpts+kj]).max(), 0, 9)

    def test_ccmdf_make_j3c_j_only_sr(self):
        dfbuilder = mdf._CCMDFBuilder(cell_sr, auxcell_sr, kpts).set(mesh=mesh).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2', j_only=True)
            v_s2 = []
            for ki in range(nkpts):
                v_s2.append(load(tmpf.name, kpts[[ki, ki]]))
            self.assertAlmostEqual(lib.fp(v_s2[0]), 0.014857466177913803, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2]), 0.015080760401588148+0j, 7)

            dfbuilder.make_j3c(tmpf.name, aosym='s1', j_only=True)
            idx, idy = np.tril_indices(cell_sr.nao)
            idx = idx * cell_sr.nao + idy
            for ki in range(nkpts):
                v1 = load(tmpf.name, kpts[[ki, ki]])
                v1 = v1[idx[:,None], idx]
                self.assertAlmostEqual(abs(v1 - v_s2[ki]).max(), 0, 9)

    def test_ccmdf_vs_fft_sr(self):
        cell_sr = pgto.M(
            a = np.eye(3) * 2.6,
            atom = 'He    0.    2.2      1.; He    1.    1.       1.',
            basis = [[0, [1.2, 1.], [.7, .5], [0.4, .5]], [1, [1.1, .5], [0.3, .5]]],
            mesh = [14] * 3,
            verbose = 0,
            precision = 1e-9,
        )
        cell_sr.omega = -0.9
        auxcell_sr = df.make_auxcell(cell_sr,
            auxbasis=[[0, [1.2, 1.], [.7, .5], [0.4, .5]],
                      [1, [1.1, .5], [0.4, .5]],
                      [2, [1., 1.]],
                     ],
        )

        dfbuilder = mdf._CCMDFBuilder(cell_sr, auxcell_sr, kpts)
        dfbuilder.build()
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))

        Gv, Gvbase, kws = cell_sr.get_Gv_weights()
        kpt = np.zeros(3)
        auxG = ft_ao.ft_ao(auxcell_sr, Gv).T
        coulG = pbctools.get_coulG(auxcell_sr, kpt, mesh=cell_sr.mesh,
                                   omega=cell_sr.omega) * kws
        Gv1, Gvbase1, kws1 = cell_sr.get_Gv_weights(mesh=dfbuilder.mesh)
        auxG1 = ft_ao.ft_ao(auxcell_sr, Gv1).T
        coulG1 = pbctools.get_coulG(auxcell_sr, kpt, mesh=dfbuilder.mesh,
                                    omega=cell_sr.omega) * kws1
        ref = lib.dot(auxG.conj()*coulG, auxG.T) - lib.dot(auxG1.conj()*coulG1, auxG1.T)
        self.assertAlmostEqual(abs(ref - j2c[0]).max(), 0, 9)

        j3c = (lib.dot(auxG.conj()*coulG, ft_ao.ft_aopair(cell_sr, Gv, aosym='s2')) -
               lib.dot(auxG1.conj()*coulG1, ft_ao.ft_aopair(cell_sr, Gv1, aosym='s2')))
        j2c = dfbuilder.eigenvalue_decomposed_metric(j2c[0])
        ref = lib.dot(j2c[0], j3c)
        ref = ref.T.dot(ref)
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2', j_only=True)
            v1 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(abs(ref - v1).max(), 0, 8)

#### Test _RSMDFBuilder with omega < 0 ####

    def test_rsmdf_get_2c2e_gamma_sr(self):
        dfbuilder = mdf._RSMDFBuilder(cell_sr, auxcell_sr).set(mesh=mesh).build()
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))
        self.assertAlmostEqual(lib.fp(j2c), 0.010646767031704538, 6)

    def test_rsmdf_get_2c2e_sr(self):
        dfbuilder = mdf._RSMDFBuilder(cell_sr, auxcell_sr, kpts).set(mesh=mesh).build()
        j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), 0.0902078021182187+0.01449234991201075j, 6)
        self.assertAlmostEqual(lib.fp(j2c[0]), 0.010646767031704538, 6)

    def test_rsmdf_get_2c2e_cart_sr(self):
        with lib.temporary_env(cell_sr, cart=True):
            dfbuilder = mdf._RSMDFBuilder(cell_sr, auxcell_sr, kpts).set(mesh=mesh).build()
            j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), 0.0902078021182187+0.01449234991201075j, 6)

    def test_rsmdf_make_j3c_gamma_sr(self):
        dfbuilder = mdf._RSMDFBuilder(cell_sr, auxcell_sr).set(mesh=mesh).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 0.014857466177913803, 7)

            dfbuilder.make_j3c(tmpf.name, aosym='s1')
            v1 = load(tmpf.name, kpts[[0, 0]])
            v2 = ao2mo.restore(1, v2, cell_sr.nao).reshape(v1.shape)
            self.assertAlmostEqual(abs(v1 - v2).max(), 0, 9)

            dfbuilder.exclude_dd_block = True
            dfbuilder.exclude_d_aux = False
            dfbuilder.build()
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 0.014857466177913803, 7)

            dfbuilder.exclude_dd_block = False
            dfbuilder.exclude_d_aux = True
            dfbuilder.build()
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 0.014857466177913803, 7)

            dfbuilder.exclude_dd_block = False
            dfbuilder.exclude_d_aux = False
            dfbuilder.build()
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 0.014857466177913803, 7)

    def test_rsmdf_make_j3c_sr(self):
        dfbuilder = mdf._RSMDFBuilder(cell_sr, auxcell_sr, kpts).set(mesh=mesh).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            v_s2 = []
            for ki in range(nkpts):
                for kj in range(nkpts):
                    v_s2.append(load(tmpf.name, kpts[[ki, kj]]))
            self.assertAlmostEqual(lib.fp(v_s2[0]), 0.014857466177913803, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2*nkpts+4]), 0.016578073155690577+0.0005539785824510145j, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2*nkpts+2]), 0.015080760401588148+0j, 7)

            dfbuilder.make_j3c(tmpf.name, aosym='s1')
            idx, idy = np.tril_indices(cell.nao)
            idx = idx * cell.nao + idy
            for ki in range(nkpts):
                for kj in range(nkpts):
                    v1 = load(tmpf.name, kpts[[ki, kj]])
                    if ki == kj:
                        v1 = v1[idx[:,None], idx]
                    self.assertAlmostEqual(abs(v1 - v_s2[ki*nkpts+kj]).max(), 0, 9)

    def test_rsmdf_make_j3c_j_only_sr(self):
        dfbuilder = mdf._RSMDFBuilder(cell_sr, auxcell_sr, kpts).set(mesh=mesh).build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2', j_only=True)
            v_s2 = []
            for ki in range(nkpts):
                v_s2.append(load(tmpf.name, kpts[[ki, ki]]))
            self.assertAlmostEqual(lib.fp(v_s2[0]), 0.014857466177913803, 7)
            self.assertAlmostEqual(lib.fp(v_s2[2]), 0.015080760401588148+0j, 7)

            dfbuilder.make_j3c(tmpf.name, aosym='s1', j_only=True)
            idx, idy = np.tril_indices(cell_sr.nao)
            idx = idx * cell_sr.nao + idy
            for ki in range(nkpts):
                v1 = load(tmpf.name, kpts[[ki, ki]])
                v1 = v1[idx[:,None], idx]
                self.assertAlmostEqual(abs(v1 - v_s2[ki]).max(), 0, 9)

if __name__ == '__main__':
    print("Full Tests for mdf_builder")
    unittest.main()
