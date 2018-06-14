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
df.LINEAR_DEP_THR = 1e-7

L = 5.
n = 11
cell = pgto.Cell()
cell.a = numpy.diag([L,L,L])
cell.mesh = numpy.array([n,n,n])

cell.atom = '''C    3.    2.       3.
               C    1.    1.       1.'''
cell.basis = 'ccpvdz'
cell.verbose = 0
cell.max_memory = 0
cell.rcut = 28.3458918685
cell.build()

mf0 = pscf.RHF(cell)
mf0.exxdiv = None


def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(a.ravel(), w)

class KnowValues(unittest.TestCase):
    def test_jk_single_kpt(self):
        mf = df_jk.density_fit(mf0, auxbasis='weigend', mesh=(11,)*3)
        mf.with_df.mesh = cell.mesh
        mf.with_df.eta = 0.3
        dm = mf.get_init_guess()
        vj, vk = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 46.69888588120217, 8)
        self.assertAlmostEqual(ek1, 31.72349032270801, 8)

        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        vj1, vk1 = mf.get_jk(cell, dm, hermi=0)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 242.04678235140264, 8)
        self.assertAlmostEqual(ek1, 280.15934926575903, 8)

        numpy.random.seed(1)
        kpt = numpy.random.random(3)
        mydf = df.DF(cell, [kpt]).set(auxbasis='weigend')
        mydf.mesh = cell.mesh
        mydf.eta = 0.3
        vj, vk = mydf.get_jk(dm, 1, kpt, exxdiv=None)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 241.15121903857556+0j, 8)
        self.assertAlmostEqual(ek1, 279.64649194057051+0j, 8)
        vj, vk = mydf.get_jk(dm, 1, kpt, with_j=False, exxdiv='ewald')
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ek1, 691.64624456329909+0j, 6)

    def test_jk_hermi0(self):
        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        dm[:2,-3:] *= .5
        jkdf = df.DF(cell).set(auxbasis='weigend')
        jkdf.mesh = (11,)*3
        jkdf.eta = 0.3
        vj0, vk0 = jkdf.get_jk(dm, hermi=0, exxdiv=None)
        ej0 = numpy.einsum('ij,ji->', vj0, dm)
        ek0 = numpy.einsum('ij,ji->', vk0, dm)
        self.assertAlmostEqual(ej0, 242.04151204237999, 8)
        self.assertAlmostEqual(ek0, 280.58443171612089, 8)

    def test_j_kpts(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = df.DF(cell).set(auxbasis='weigend')
        mydf.kpts = numpy.random.random((4,3))
        mydf.mesh = numpy.asarray((11,)*3)
        mydf.auxbasis = 'weigend'
        mydf.mesh = cell.mesh
        mydf.eta = 0.3
        vj = df_jk.get_j_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(finger(vj[0]), (0.49176180692009197-0.11891083594538684j ), 9)
        self.assertAlmostEqual(finger(vj[1]), (0.54900852073326378-0.04600354345316908j ), 9)
        self.assertAlmostEqual(finger(vj[2]), (0.53648110926681891-0.083507522327029265j), 9)
        self.assertAlmostEqual(finger(vj[3]), (0.5489650266265671 +0.007695733246577244j), 9)

    def test_k_kpts(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = df.DF(cell).set(auxbasis='weigend')
        mydf.kpts = numpy.random.random((4,3))
        mydf.mesh = numpy.asarray((11,)*3)
        mydf.exxdiv = None
        mydf.mesh = cell.mesh
        mydf.eta = 0.3
        mydf.auxbasis = 'weigend'
        vk = df_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (-2.8332400193836929 -1.0578696472684668j  ), 9)
        self.assertAlmostEqual(finger(vk[1]), (-7.440432864374058  +0.10233777556396761j ), 9)
        self.assertAlmostEqual(finger(vk[2]), (-2.5718862399533897 -1.4487403259747005j  ), 9)
        self.assertAlmostEqual(finger(vk[3]), (-0.79223093737594863+0.011694427945090839j), 9)

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
        mydf.kpts = kpts
        mydf.auxbasis = {'He': [(0, (4.096, 1)), (0, (2.56, 1)), (0, (1.6, 1)), (0, (1., 1))]}
        mydf.exxdiv = None
        mydf.mesh = cell.mesh
        mydf.eta = 0.3
        vk = df_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (0.54220010040518218-0.00787204295681934j  ), 9)
        self.assertAlmostEqual(finger(vk[1]), (0.35987105007103914+0.0036047438452865574j), 9)
        self.assertAlmostEqual(finger(vk[2]), (0.46287057223452965-0.0065045318150024475j), 9)
        self.assertAlmostEqual(finger(vk[3]), (0.63677390788341914+0.0075132081533213447j), 9)
        self.assertAlmostEqual(finger(vk[4]), (0.53680188658523353-0.0076414750780774933j), 9)
        self.assertAlmostEqual(finger(vk[5]), (0.49613855046499666+0.0060603767383680838j), 9)
        self.assertAlmostEqual(finger(vk[6]), (0.45430752211150049-0.0068611602260866128j), 9)
        self.assertAlmostEqual(finger(vk[7]), (0.41856931218763038+0.0051073315205987522j), 9)

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
        mydf.kpts = kpts
        mydf.auxbasis = {'He': [(0, (4.096, 1)), (0, (2.56, 1)), (0, (1.6, 1)), (0, (1., 1))]}
        mydf.exxdiv = None
        mydf.mesh = cell.mesh
        mydf.eta = 0.3
        nao = cell.nao_nr()
        numpy.random.seed(1)
        dm = numpy.random.random((8,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        vk = df.df_jk.get_k_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (1.0940331326660724 -0.01474246983191657j ), 9)
        self.assertAlmostEqual(finger(vk[1]), (0.72106828546205248+0.008683360062569572j), 9)
        self.assertAlmostEqual(finger(vk[2]), (0.89868267009698988-0.011091489111877838j), 9)
        self.assertAlmostEqual(finger(vk[3]), (1.2604941401190835 +0.015979544115384041j), 9)
        self.assertAlmostEqual(finger(vk[4]), (1.0492129520812594 -0.012424653667344821j), 9)
        self.assertAlmostEqual(finger(vk[5]), (0.99271107721956797+0.012696925711370165j), 9)
        self.assertAlmostEqual(finger(vk[6]), (0.92184754518871648-0.012035727588110348j), 9)
        self.assertAlmostEqual(finger(vk[7]), (0.8518483148628242 +0.010084767506077213j), 9)


if __name__ == '__main__':
    print("Full Tests for df_jk")
    unittest.main()

