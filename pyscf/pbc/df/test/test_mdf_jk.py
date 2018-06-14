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
from pyscf.pbc.df import mdf
from pyscf.pbc.df import mdf_jk
#from mpi4pyscf.pbc.df import mdf
#from mpi4pyscf.pbc.df import mdf_jk
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
cell.max_memory = 0
cell.rcut = 28.3
cell.build()

mf0 = pscf.RHF(cell)
mf0.exxdiv = None


def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(a.ravel(), w)

class KnowValues(unittest.TestCase):
    def test_jk_single_kpt(self):
        mf = mdf_jk.density_fit(mf0, auxbasis='weigend', mesh=(11,)*3)
        mf.with_df.mesh = [11]*3
        mf.with_df.eta = 0.3
        dm = mf.get_init_guess()
        vj, vk = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 46.698950904264514, 8)
        self.assertAlmostEqual(ek1, 31.724297945923094, 8)

        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        vj1, vk1 = mf.get_jk(cell, dm, hermi=0)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 242.19379703364774, 8)
        self.assertAlmostEqual(ek1, 280.28450527230103, 8)

        numpy.random.seed(1)
        kpt = numpy.random.random(3)
        mydf = mdf.MDF(cell, [kpt]).set(auxbasis='weigend')
        mydf.mesh = [11]*3
        mydf.eta = 0.3
        vj, vk = mydf.get_jk(dm, 1, kpt)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 241.29955504573206+0j, 8)
        self.assertAlmostEqual(ek1, 691.76854602384913+0j, 8)

    def test_jk_hermi0(self):
        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        dm[:2,-3:] *= .5
        jkdf = mdf.MDF(cell).set(auxbasis='weigend')
        jkdf.mesh = (11,)*3
        jkdf.eta = 0.3
        vj0, vk0 = jkdf.get_jk(dm, hermi=0, exxdiv=None)
        ej0 = numpy.einsum('ij,ji->', vj0, dm)
        ek0 = numpy.einsum('ij,ji->', vk0, dm)
        self.assertAlmostEqual(ej0, 242.18855706416096, 8)
        self.assertAlmostEqual(ek0, 280.70982164657647, 8)

    def test_j_kpts(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = mdf.MDF(cell).set(auxbasis='weigend')
        mydf.kpts = numpy.random.random((4,3))
        mydf.mesh = numpy.asarray((11,)*3)
        mydf.eta = 0.3
        mydf.auxbasis = 'weigend'
        vj = mdf_jk.get_j_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(finger(vj[0]), (0.48227579581461733-0.11872579745444795j ), 9)
        self.assertAlmostEqual(finger(vj[1]), (0.54073632897787327-0.046133464893148166j), 9)
        self.assertAlmostEqual(finger(vj[2]), (0.52806708811400505-0.083705157508446218j), 9)
        self.assertAlmostEqual(finger(vj[3]), (0.5435189277058412 +0.008843567739405876j), 9)

    def test_k_kpts(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = mdf.MDF(cell).set(auxbasis='weigend')
        mydf.kpts = numpy.random.random((4,3))
        mydf.mesh = numpy.asarray((11,)*3)
        mydf.eta = 0.3
        mydf.exxdiv = None
        mydf.auxbasis = 'weigend'
        vk = mdf_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (-2.8420706204318527 -1.0520028601696778j  ), 9)
        self.assertAlmostEqual(finger(vk[1]), (-7.4484096949300751 +0.10323425122156138j ), 9)
        self.assertAlmostEqual(finger(vk[2]), (-2.580181288621187  -1.4470150314314312j  ), 9)
        self.assertAlmostEqual(finger(vk[3]), (-0.79660123464892396+0.011973030805184665j), 9)

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
        mydf = mdf.MDF(cell).set(auxbasis='weigend')
        mydf.kpts = kpts
        mydf.auxbasis = 'weigend'
        mydf.exxdiv = None
        mydf.mesh = numpy.asarray((11,)*3)
        mydf.eta = 0.3
        vk = mdf_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (0.54208542933016668-0.007872205456027636j ), 9)
        self.assertAlmostEqual(finger(vk[1]), (0.35976730327192219+0.0036055469686362227j), 9)
        self.assertAlmostEqual(finger(vk[2]), (0.46276307618592272-0.006504349523994527j ), 9)
        self.assertAlmostEqual(finger(vk[3]), (0.63667731843923825+0.0075118647005158034j), 9)
        self.assertAlmostEqual(finger(vk[4]), (0.53670632359622572-0.00764236264065816j  ), 9)
        self.assertAlmostEqual(finger(vk[5]), (0.4960454361832054 +0.0060590376596187257j), 9)
        self.assertAlmostEqual(finger(vk[6]), (0.45421052168235576-0.006861624162215218j ), 9)
        self.assertAlmostEqual(finger(vk[7]), (0.41848054629487041+0.0051096775483082746j), 9)

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
        mydf = mdf.MDF(cell).set(auxbasis='weigend')
        mydf.kpts = kpts
        mydf.auxbasis = 'weigend'
        mydf.exxdiv = None
        mydf.mesh = numpy.asarray((11,)*3)
        mydf.eta = 0.3
        nao = cell.nao_nr()
        numpy.random.seed(1)
        dm = numpy.random.random((8,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        vk = mdf_jk.get_k_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (1.0938028454012594 -0.014742352047969521j), 9)
        self.assertAlmostEqual(finger(vk[1]), (0.72086205228975953+0.008685417852198867j), 9)
        self.assertAlmostEqual(finger(vk[2]), (0.89846608130483796-0.011091006902191843j), 9)
        self.assertAlmostEqual(finger(vk[3]), (1.260302267937254  +0.015976908047169756j), 9)
        self.assertAlmostEqual(finger(vk[4]), (1.0490207113210688 -0.012426436820904021j), 9)
        self.assertAlmostEqual(finger(vk[5]), (0.99252601243537697+0.012694645170334074j), 9)
        self.assertAlmostEqual(finger(vk[6]), (0.92165252496655681-0.012036431811316108j), 9)
        self.assertAlmostEqual(finger(vk[7]), (0.85167195537981   +0.010089165459944484j), 9)


if __name__ == '__main__':
    print("Full Tests for mdf_jk")
    unittest.main()

