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
from pyscf.pbc import scf as pscf
import pyscf.pbc
from pyscf.pbc.df import df
from pyscf.pbc.df import df_jk
#from mpi4pyscf.pbc.df import df
#from mpi4pyscf.pbc.df import df_jk
pyscf.pbc.DEBUG = False

def setUpModule():
    global cell, cell0, n
    L = 5.
    n = 11
    cell = pgto.Cell()
    cell.a = numpy.diag([L,L,L])
    cell.mesh = numpy.array([n, n, n])

    cell.atom = '''C    3.    2.       3.
                   C    1.    1.       1.'''
    cell.basis = 'ccpvdz'
    cell.verbose = 0
    cell.max_memory = 0
    cell.precision = 1e-9
    cell.build()

    cell0 = pgto.Cell()
    cell0.a = numpy.eye(3) * L
    cell0.atom = '''C    3.    2.       3.
                    C    1.    1.       1.'''
    cell0.basis = 'sto-3g'
    cell0.verbose = 0
    cell0.precision = 1e-9
    cell0.build()

def tearDownModule():
    global cell, cell0
    del cell, cell0


class KnownValues(unittest.TestCase):
    def test_jk_single_kpt(self):
        mf = cell0.RHF().density_fit(auxbasis='weigend')
        mf.with_df.mesh = [n, n, n]
        mf.with_df.eta = 0.15
        mf.with_df.exp_to_discard = 0.3
        dm = mf.get_init_guess()
        vj, vk = mf.get_jk(cell0, dm)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        j_ref = 50.52980612772263  # rsjk result
        k_ref = 38.84221371860046  # rsjk result
        self.assertAlmostEqual(ej1, j_ref, 2)
        self.assertAlmostEqual(ek1, k_ref, 2)
        self.assertAlmostEqual(ej1, 50.52815082169177, 6)
        self.assertAlmostEqual(ek1, 38.83812022834723, 6)

        numpy.random.seed(12)
        nao = cell0.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        vj1, vk1 = mf.get_jk(cell0, dm, hermi=0)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 25.8129854469354, 6)
        self.assertAlmostEqual(ek1, 72.6088517709998, 6)

    def test_jk_single_kpt_complex_dm(self):
        scaled_center = [0.3728,0.5524,0.7672]
        kpt = cell0.make_kpts([1,1,1], scaled_center=scaled_center)[0]
        mf = pscf.RHF(cell0, kpt=kpt).density_fit('weigend')
        dm = mf.init_guess_by_1e()
        with lib.temporary_env(mf.cell, incore_anyway=True):
            vj1, vk1 = mf.get_jk(dm=dm) # from mol_hf.dot_eri_dm
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        vj, vk = mf.with_df.get_jk(dm=dm, kpts=kpt, exxdiv=mf.exxdiv)
        ej = numpy.einsum('ij,ji->', vj, dm)
        ek = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, ej, 10)
        self.assertAlmostEqual(ek1, ek, 10)

    def test_jk_single_kpt_high_cost(self):
        mf0 = pscf.RHF(cell)
        mf0.exxdiv = None
        mf = df_jk.density_fit(mf0, auxbasis='weigend', mesh=(11,)*3)
        mf.with_df.mesh = cell.mesh
        mf.with_df.eta = 0.15
        mf.with_df.exp_to_discard = 0.3
        dm = mf.get_init_guess()
        vj, vk = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        j_ref = 48.283789539266174  # rsjk result
        k_ref = 32.30441176447805   # rsjk result
        self.assertAlmostEqual(ej1, j_ref, 2)
        self.assertAlmostEqual(ek1, k_ref, 2)
        self.assertAlmostEqual(ej1, 48.283745538383684, 6)
        self.assertAlmostEqual(ek1, 32.30260871417842 , 6)

        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        vj1, vk1 = mf.get_jk(cell, dm, hermi=0)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 242.04678166341, 6)
        self.assertAlmostEqual(ek1, 280.15934886749, 6)

        numpy.random.seed(1)
        kpt = numpy.random.random(3)
        mydf = df.DF(cell, [kpt]).set(auxbasis='weigend')
        mydf.linear_dep_threshold = 1e-7
        mydf.mesh = cell.mesh
        mydf.eta = 0.15
        mydf.exp_to_discard = 0.3
        vj, vk = mydf.get_jk(dm, 1, kpt, exxdiv=None)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 241.15121826587+0j, 5)
        self.assertAlmostEqual(ek1, 279.64649158710+0j, 5)
        vj, vk = mydf.get_jk(dm, 1, kpt, with_j=False, exxdiv='ewald')
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ek1, 691.64624420983+0j, 5)

    def test_jk_hermi0(self):
        numpy.random.seed(12)
        nao = cell0.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        dm[:2,-3:] *= .5
        jkdf = df.DF(cell0).set(auxbasis='weigend')
        jkdf.linear_dep_threshold = 1e-7
        jkdf.mesh = (11,)*3
        jkdf.eta = 0.15
        jkdf.exp_to_discard = 0.3
        vj0, vk0 = jkdf.get_jk(dm, hermi=0, exxdiv=None)
        ej0 = numpy.einsum('ij,ji->', vj0, dm)
        ek0 = numpy.einsum('ij,ji->', vk0, dm)
        self.assertAlmostEqual(ej0, 25.775008146164968, 6)
        self.assertAlmostEqual(ek0, 30.814023529860112, 6)

    def test_jk_hermi0_high_cost(self):
        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        dm[:2,-3:] *= .5
        jkdf = df.DF(cell).set(auxbasis='weigend')
        jkdf.linear_dep_threshold = 1e-7
        jkdf.mesh = (11,)*3
        jkdf.eta = 0.15
        jkdf.exp_to_discard = 0.3
        vj0, vk0 = jkdf.get_jk(dm, hermi=0, exxdiv=None)
        ej0 = numpy.einsum('ij,ji->', vj0, dm)
        ek0 = numpy.einsum('ij,ji->', vk0, dm)
        self.assertAlmostEqual(ej0, 242.04151204237999, 5)
        self.assertAlmostEqual(ek0, 280.58443171612089, 5)

    def test_j_kpts(self):
        numpy.random.seed(1)
        nao = cell0.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = df.DF(cell0).set(auxbasis='weigend')
        mydf.linear_dep_threshold = 1e-7
        mydf.kpts = numpy.random.random((4,3))
        mydf.auxbasis = 'weigend'
        mydf.eta = 0.15
        mydf.exp_to_discard = 0.3
        vj = df_jk.get_j_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(lib.fp(vj[0]), (7.240207870630442-0.001010622364950332j) , 7)
        self.assertAlmostEqual(lib.fp(vj[1]), (7.248745538469966-0.001562604522803734j) , 7)
        self.assertAlmostEqual(lib.fp(vj[2]), (7.241193241602369-0.002518439407055759j) , 7)
        self.assertAlmostEqual(lib.fp(vj[3]), (7.2403591406956185+0.001475803952777666j), 7)

    def test_j_kpts_high_cost(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = df.DF(cell).set(auxbasis='weigend')
        mydf.linear_dep_threshold = 1e-7
        mydf.kpts = numpy.random.random((4,3))
        mydf.mesh = [n, n, n]
        mydf.auxbasis = 'weigend'
        mydf.eta = 0.15
        mydf.exp_to_discard = 0.3
        vj = df_jk.get_j_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(lib.fp(vj[0]), (0.49176180692009197-0.11891083594538684j ), 5)
        self.assertAlmostEqual(lib.fp(vj[1]), (0.54900852073326378-0.04600354345316908j ), 5)
        self.assertAlmostEqual(lib.fp(vj[2]), (0.53648110926681891-0.083507522327029265j), 5)
        self.assertAlmostEqual(lib.fp(vj[3]), (0.5489650266265671 +0.007695733246577244j), 5)

    def test_k_kpts(self):
        numpy.random.seed(1)
        nao = cell0.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = df.DF(cell0).set(auxbasis='weigend')
        mydf.linear_dep_threshold = 1e-7
        kpts = numpy.random.random((4,3))
        kpts -= cell0.get_abs_kpts(cell0.get_scaled_kpts(kpts).round(0))
        mydf.kpts = kpts
        mydf.exxdiv = None
        mydf.eta = 0.15
        mydf.exp_to_discard = 0.3
        mydf.auxbasis = 'weigend'
        vk = df_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(lib.fp(vk[0]), (4.831027586092549-0.12376435978940196j) , 7)
        self.assertAlmostEqual(lib.fp(vk[1]), (4.783208264204395-0.00585421470169705j) , 7)
        self.assertAlmostEqual(lib.fp(vk[2]), (4.823839360632854+0.002511545727704362j), 7)
        self.assertAlmostEqual(lib.fp(vk[3]), (4.833891390413435+0.0208696082684768j)  , 7)

    def test_k_kpts_high_cost(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = df.DF(cell).set(auxbasis='weigend')
        mydf.linear_dep_threshold = 1e-7
        mydf.kpts = numpy.random.random((4,3))
        mydf.mesh = [n, n, n]
        mydf.exxdiv = None
        mydf.eta = 0.15
        mydf.exp_to_discard = 0.3
        mydf.auxbasis = 'weigend'
        vk = df_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(lib.fp(vk[0]), (-2.8332400193836929 -1.0578696472684668j  ), 5)
        self.assertAlmostEqual(lib.fp(vk[1]), (-7.440432864374058  +0.10233777556396761j ), 5)
        self.assertAlmostEqual(lib.fp(vk[2]), (-2.5718862399533897 -1.4487403259747005j  ), 5)
        self.assertAlmostEqual(lib.fp(vk[3]), (-0.79223093737594863+0.011694427945090839j), 5)

    def test_k_kpts_1(self):
        cell = pgto.Cell()
        cell.atom = 'He 1. .5 .5; He .1 1.3 2.1'
        cell.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))]}
        cell.a = numpy.eye(3) * 2.5
        cell.mesh = [11] * 3
        cell.build()
        kpts = cell.get_abs_kpts([[-.25,-.25,-.25],
                                  [-.25,-.25, .25],
                                  [-.25, .25,-.25],
                                  [-.25, .25, .25],
                                  [ .25,-.25,-.25],
                                  [ .25,-.25, .25],
                                  [ .25, .25,-.25],
                                  [ .25, .25, .25]])

        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((8,nao,nao))
        mydf = df.DF(cell).set(auxbasis='weigend')
        mydf.linear_dep_threshold = 1e-7
        mydf.kpts = kpts
        mydf.auxbasis = {'He': [(0, (4.096, 1)), (0, (2.56, 1)), (0, (1.6, 1)), (0, (1., 1))]}
        mydf.exxdiv = None
        mydf.mesh = cell.mesh
        mydf.eta = 0.15
        mydf.exp_to_discard = 0.3
        vk = df_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(lib.fp(vk[0]), (0.54220010040518218-0.00787204295681934j  ), 6)
        self.assertAlmostEqual(lib.fp(vk[1]), (0.35987105007103914+0.0036047438452865574j), 6)
        self.assertAlmostEqual(lib.fp(vk[2]), (0.46287057223452965-0.0065045318150024475j), 6)
        self.assertAlmostEqual(lib.fp(vk[3]), (0.63677390788341914+0.0075132081533213447j), 6)
        self.assertAlmostEqual(lib.fp(vk[4]), (0.53680188658523353-0.0076414750780774933j), 6)
        self.assertAlmostEqual(lib.fp(vk[5]), (0.49613855046499666+0.0060603767383680838j), 6)
        self.assertAlmostEqual(lib.fp(vk[6]), (0.45430752211150049-0.0068611602260866128j), 6)
        self.assertAlmostEqual(lib.fp(vk[7]), (0.41856931218763038+0.0051073315205987522j), 6)

    def test_k_kpts_2(self):
        cell = pgto.Cell()
        cell.atom = 'He 1. .5 .5; He .1 1.3 2.1'
        cell.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))]}
        cell.a = numpy.eye(3) * 2.5
        cell.mesh = [11] * 3
        cell.build()
        kpts = cell.get_abs_kpts([[-.25,-.25,-.25],
                                  [-.25,-.25, .25],
                                  [-.25, .25,-.25],
                                  [-.25, .25, .25],
                                  [ .25,-.25,-.25],
                                  [ .25,-.25, .25],
                                  [ .25, .25,-.25],
                                  [ .25, .25, .25]])
        mydf = df.DF(cell).set(auxbasis='weigend')
        mydf.linear_dep_threshold = 1e-7
        mydf.kpts = kpts
        mydf.auxbasis = {'He': [(0, (4.096, 1)), (0, (2.56, 1)), (0, (1.6, 1)), (0, (1., 1))]}
        mydf.exxdiv = None
        mydf.mesh = cell.mesh
        mydf.eta = 0.15
        mydf.exp_to_discard = 0.3
        nao = cell.nao_nr()
        numpy.random.seed(1)
        dm = numpy.random.random((8,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        vk = df.df_jk.get_k_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(lib.fp(vk[0]), (1.0940331326660724 -0.01474246983191657j ), 6)
        self.assertAlmostEqual(lib.fp(vk[1]), (0.72106828546205248+0.008683360062569572j), 6)
        self.assertAlmostEqual(lib.fp(vk[2]), (0.89868267009698988-0.011091489111877838j), 6)
        self.assertAlmostEqual(lib.fp(vk[3]), (1.2604941401190835 +0.015979544115384041j), 6)
        self.assertAlmostEqual(lib.fp(vk[4]), (1.0492129520812594 -0.012424653667344821j), 6)
        self.assertAlmostEqual(lib.fp(vk[5]), (0.99271107721956797+0.012696925711370165j), 6)
        self.assertAlmostEqual(lib.fp(vk[6]), (0.92184754518871648-0.012035727588110348j), 6)
        self.assertAlmostEqual(lib.fp(vk[7]), (0.8518483148628242 +0.010084767506077213j), 6)


if __name__ == '__main__':
    print("Full Tests for df_jk")
    unittest.main()
