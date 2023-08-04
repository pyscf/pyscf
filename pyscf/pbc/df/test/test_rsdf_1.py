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
import tempfile
import numpy as np
import scipy.linalg
from pyscf import lib
import pyscf.pbc
from pyscf import ao2mo, gto
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.df import df
from pyscf.pbc.df import rsdf
from pyscf.pbc.df import ft_ao
from pyscf.pbc.tools import pbc as pbctools
pyscf.pbc.DEBUG = False

def setUpModule():
    global cell, kmdf, auxcell, dfobj, kpts
    cell = pgto.M(
        a = np.eye(3) * 3.5,
        mesh = [11] * 3,
        atom = '''He    3.    2.       3.
                  He    1.    1.       1.''',
        basis = 'ccpvdz',
        verbose = 0,
        max_memory = 1000,
    )

    kpts = cell.make_kpts([3,5,6])[[0, 2, 3, 4, 6, 12, 20]]
    kpts[3] = kpts[0]-kpts[1]+kpts[2]

    kmdf = rsdf.RSDF(cell)
    kmdf.linear_dep_threshold = 1e-7
    kmdf.auxbasis = 'weigend'
    kmdf.kpts = kpts

    dfobj = rsdf.RSGDF(cell, kpts)
    dfobj.auxbasis = 'weigend'
    dfobj._rs_build()
    auxcell = dfobj.auxcell

def load(filename, kptij):
    with df._load3c(filename, 'j3c', kptij) as cderi:
        return cderi[:]

def tearDownModule():
    global cell, kmdf, auxcell, dfobj
    del cell, kmdf, auxcell, dfobj

class KnownValues(unittest.TestCase):
    def test_get_2c2e_gamma(self):
        dfbuilder = rsdf._RSGDFBuilder(cell, auxcell)
        dfbuilder.__dict__.update(dfobj.__dict__)
        dfbuilder.build()
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))
        self.assertAlmostEqual(lib.fp(j2c), -8.240595211369692, 9)

    def test_get_2c2e(self):
        dfbuilder = rsdf._RSGDFBuilder(cell, auxcell, kpts)
        dfbuilder.__dict__.update(dfobj.__dict__)
        dfbuilder.build()
        j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), -110.15438692240832+8.528960438609099j, 7)
        self.assertAlmostEqual(lib.fp(j2c[0]), -8.240595211369692, 9)

    def test_get_2c2e_cart(self):
        with lib.temporary_env(cell, cart=True):
            dfbuilder = rsdf._RSGDFBuilder(cell, auxcell, kpts)
            dfbuilder.__dict__.update(dfobj.__dict__)
            dfbuilder.build()
            j2c = dfbuilder.get_2c2e(kpts)
        self.assertAlmostEqual(lib.fp(j2c), -110.15438692240832+8.528960438609099j, 7)

    def test_make_j3c_gamma(self):
        dfbuilder = rsdf._RSGDFBuilder(cell, auxcell, kpts)
        dfbuilder.__dict__.update(dfobj.__dict__)
        dfbuilder.build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name)
            v2 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(lib.fp(v2), 1.4877735852543206, 8)

    def test_make_j3c(self):
        dfbuilder = rsdf._RSGDFBuilder(cell, auxcell, kpts)
        dfbuilder.__dict__.update(dfobj.__dict__)
        dfbuilder.build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2')
            self.assertAlmostEqual(lib.fp(load(tmpf.name, kpts[[0, 0]])), 1.4877735860707935, 7)
            self.assertAlmostEqual(lib.fp(load(tmpf.name, kpts[[2, 4]])), 4.530919637533813+0.10852447737595214j, 7)
            self.assertAlmostEqual(lib.fp(load(tmpf.name, kpts[[2, 2]])), 1.4492567814298059, 7)

    def test_make_j3c_j_only(self):
        dfbuilder = rsdf._RSGDFBuilder(cell, auxcell, kpts)
        dfbuilder.__dict__.update(dfobj.__dict__)
        dfbuilder.build()
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2', j_only=True)
            self.assertAlmostEqual(lib.fp(load(tmpf.name, kpts[[0, 0]])), 1.4877735860707935, 7)
            self.assertAlmostEqual(lib.fp(load(tmpf.name, kpts[[2, 2]])), 1.4492567814298059, 7)

    def test_vs_fft(self):
        cell = pgto.M(
            a = np.eye(3) * 2.6,
            atom = 'He    0.    2.2      1.; He    1.    1.       1.',
            basis = [[0, [1.2, 1.], [.7, .5], [0.4, .5]], [1, [1.1, .5], [0.3, .5]]],
            mesh = [14] * 3,
            verbose = 0,
        )

        kpts = np.zeros((1, 3))
        dfobj = rsdf.RSGDF(cell, kpts)
        dfobj.auxbasis = [[0, [1.2, 1.], [.7, .5], [0.4, .5]],
                          [1, [1.1, .5], [0.3, .5]],
                          [2, [1., 1.]],
                         ]
        dfobj._rs_build()
        auxcell = dfobj.auxcell

        dfbuilder = rsdf._RSGDFBuilder(cell, auxcell, kpts)
        dfbuilder.__dict__.update(dfobj.__dict__)
        dfbuilder.build()
        j2c = dfbuilder.get_2c2e(np.zeros((1, 3)))

        Gv, Gvbase, kws = cell.get_Gv_weights()
        kpt = np.zeros(3)
        auxG = ft_ao.ft_ao(auxcell, Gv).T
        coulG = pbctools.get_coulG(auxcell, kpt, mesh=cell.mesh) * kws
        ref = lib.dot(auxG.conj()*coulG, auxG.T)
        self.assertAlmostEqual(abs(ref - j2c[0]).max(), 0, 8)

        aopair = ft_ao.ft_aopair(cell, Gv, aosym='s2')
        ngrids = Gv.shape[0]
        j3c = lib.dot(auxG.conj()*coulG, aopair.reshape(ngrids,-1))
        j2c = scipy.linalg.cholesky(j2c[0], lower=True)
        ref = scipy.linalg.solve_triangular(j2c, j3c, lower=True)
        with tempfile.NamedTemporaryFile() as tmpf:
            dfbuilder.make_j3c(tmpf.name, aosym='s2', j_only=True)
            v1 = load(tmpf.name, kpts[[0, 0]])
            self.assertAlmostEqual(abs(ref - v1).max(), 0, 9)


if __name__ == '__main__':
    print("Full Tests for rsdf._RSGDFBuilder")
    unittest.main()
