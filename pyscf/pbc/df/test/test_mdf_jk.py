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
from pyscf import scf
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
import pyscf.pbc
import pyscf.pbc.df as pbcdf
from pyscf.pbc.df import mdf
from pyscf.pbc.df import mdf_jk
#from mpi4pyscf.pbc.df import mdf
#from mpi4pyscf.pbc.df import mdf_jk
pyscf.pbc.DEBUG = False

def setUpModule():
    global cell, cell0, mf0, n
    L = 5.
    n = 11
    cell = pgto.Cell()
    cell.a = numpy.diag([L,L,L])
    cell.mesh = numpy.array([n,n,n])

    cell.atom = '''C    3.    2.       3.
                   C    1.    1.       1.'''
    cell.basis = 'ccpvdz'
    #cell.verbose = 0
    #cell.max_memory = 0
    cell.rcut = 28.3
    cell.build()

    cell0 = pgto.Cell()
    cell0.a = numpy.eye(3) * L
    cell0.atom = '''C    3.    2.       3.
                    C    1.    1.       1.'''
    cell0.basis = 'sto-3g'
    cell0.verbose = 0
    cell0.build()

    mf0 = pscf.RHF(cell)
    mf0.exxdiv = None

def tearDownModule():
    global cell, cell0, mf0
    del cell, cell0, mf0


class KnownValues(unittest.TestCase):
    def test_jk_single_kpt(self):
        mf = cell0.RHF().mix_density_fit(auxbasis='weigend')
        mf.with_df.mesh = [n, n, n]
        mf.with_df.eta = 0.3
        dm = mf.get_init_guess()
        vj, vk = mf.get_jk(cell0, dm)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        j_ref = 50.52980612772263  # rsjk result
        k_ref = 38.84221371860046  # rsjk result
        self.assertAlmostEqual(ej1, j_ref, 2)
        self.assertAlmostEqual(ek1, k_ref, 2)
        self.assertAlmostEqual(ej1, 50.52819501637896, 6)
        self.assertAlmostEqual(ek1, 38.83943428046173, 6)

        numpy.random.seed(12)
        nao = cell0.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        vj1, vk1 = mf.get_jk(cell0, dm, hermi=0)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 25.81378242361819, 6)
        self.assertAlmostEqual(ek1, 72.6092971275962 , 5)

    def test_jk_single_kpt_high_cost(self):
        mf = mdf_jk.density_fit(mf0, auxbasis='weigend', mesh=(11,)*3)
        mf.with_df.mesh = [11]*3
        mf.with_df.eta = 0.3
        mf.with_df.linear_dep_threshold = 1e-7
        dm = mf.get_init_guess()
        vj, vk = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 48.2837895391989, 5)  # ref from rsjk
        self.assertAlmostEqual(ek1, 32.3033084452672, 8)

        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        vj1, vk1 = mf.get_jk(cell, dm, hermi=0)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 242.19379703364774, 6)
        self.assertAlmostEqual(ek1, 280.28450527230103, 5)

        numpy.random.seed(1)
        kpt = numpy.random.random(3)
        mydf = mdf.MDF(cell, [kpt]).set(auxbasis='weigend')
        mydf.linear_dep_threshold = 1e-7
        mydf.mesh = [11]*3
        mydf.eta = 0.3
        vj, vk = mydf.get_jk(dm, 1, kpt, exxdiv='ewald')
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 241.29955504573206+0j, 6)
        self.assertAlmostEqual(ek1, 691.76854602384913+0j, 5)

    def test_jk_hermi0(self):
        numpy.random.seed(12)
        nao = cell0.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        dm[:2,-3:] *= .5
        jkdf = mdf.MDF(cell0).set(auxbasis='weigend')
        jkdf.linear_dep_threshold = 1e-7
        jkdf.mesh = (11,)*3
        jkdf.eta = 0.3
        vj0, vk0 = jkdf.get_jk(dm, hermi=0, exxdiv=None)
        ej0 = numpy.einsum('ij,ji->', vj0, dm)
        ek0 = numpy.einsum('ij,ji->', vk0, dm)
        self.assertAlmostEqual(ej0, 25.77582611607388 , 6)
        self.assertAlmostEqual(ek0, 30.814555613338555, 6)

    def test_jk_hermi0_high_cost(self):
        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        dm[:2,-3:] *= .5
        jkdf = mdf.MDF(cell).set(auxbasis='weigend')
        jkdf.linear_dep_threshold = 1e-7
        jkdf.mesh = (11,)*3
        jkdf.eta = 0.3
        vj0, vk0 = jkdf.get_jk(dm, hermi=0, exxdiv=None)
        ej0 = numpy.einsum('ij,ji->', vj0, dm)
        ek0 = numpy.einsum('ij,ji->', vk0, dm)
        self.assertAlmostEqual(ej0, 242.18855706416096, 6)
        self.assertAlmostEqual(ek0, 280.70982164657647, 5)

    def test_j_kpts(self):
        numpy.random.seed(1)
        nao = cell0.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = mdf.MDF(cell0).set(auxbasis='weigend')
        mydf.linear_dep_threshold = 1e-7
        mydf.kpts = numpy.random.random((4,3))
        mydf.mesh = numpy.asarray((11,)*3)
        mydf.eta = 0.3
        mydf.auxbasis = 'weigend'
        vj = mdf_jk.get_j_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(lib.fp(vj[0]), (7.240247126035314-0.0010092876216366933j), 7)
        self.assertAlmostEqual(lib.fp(vj[1]), (7.248775589325954-0.0015611883008615822j), 7)
        self.assertAlmostEqual(lib.fp(vj[2]), (7.241230472941957-0.002515541792204466j) , 7)
        self.assertAlmostEqual(lib.fp(vj[3]), (7.240398079284901+0.0014737107502212023j), 7)

    def test_j_kpts_high_cost(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = mdf.MDF(cell).set(auxbasis='weigend')
        mydf.linear_dep_threshold = 1e-7
        mydf.kpts = numpy.random.random((4,3))
        mydf.mesh = numpy.asarray((11,)*3)
        mydf.eta = 0.3
        mydf.auxbasis = 'weigend'
        vj = mdf_jk.get_j_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(lib.fp(vj[0]), (0.48227579581461733-0.11872579745444795j ), 7)
        self.assertAlmostEqual(lib.fp(vj[1]), (0.54073632897787327-0.046133464893148166j), 7)
        self.assertAlmostEqual(lib.fp(vj[2]), (0.52806708811400505-0.083705157508446218j), 7)
        self.assertAlmostEqual(lib.fp(vj[3]), (0.5435189277058412 +0.008843567739405876j), 7)

    def test_k_kpts(self):
        numpy.random.seed(1)
        nao = cell0.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = mdf.MDF(cell0).set(auxbasis='weigend')
        mydf.linear_dep_threshold = 1e-7
        mydf.kpts = numpy.random.random((4,3))
        mydf.mesh = numpy.asarray((11,)*3)
        mydf.eta = 0.3
        mydf.exxdiv = None
        mydf.auxbasis = 'weigend'
        vk = mdf_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(lib.fp(vk[0]), (4.831240836863933-0.12373190618477338j) , 6)
        self.assertAlmostEqual(lib.fp(vk[1]), (4.783417745841964-0.005852945569928365j), 6)
        self.assertAlmostEqual(lib.fp(vk[2]), (4.82403824304899+0.002483686050043201j) , 6)
        self.assertAlmostEqual(lib.fp(vk[3]), (4.834089219093252+0.020822434708267664j), 6)

    def test_k_kpts_high_cost(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = mdf.MDF(cell).set(auxbasis='weigend')
        mydf.linear_dep_threshold = 1e-7
        mydf.kpts = numpy.random.random((4,3))
        mydf.mesh = numpy.asarray((11,)*3)
        mydf.eta = 0.3
        mydf.exxdiv = None
        mydf.auxbasis = 'weigend'
        vk = mdf_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(lib.fp(vk[0]), (-2.8420706204318527 -1.0520028601696778j  ), 6)
        self.assertAlmostEqual(lib.fp(vk[1]), (-7.4484096949300751 +0.10323425122156138j ), 6)
        self.assertAlmostEqual(lib.fp(vk[2]), (-2.580181288621187  -1.4470150314314312j  ), 6)
        self.assertAlmostEqual(lib.fp(vk[3]), (-0.79660123464892396+0.011973030805184665j), 6)

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
        mydf.linear_dep_threshold = 1e-7
        mydf.kpts = kpts
        mydf.auxbasis = 'weigend'
        mydf.exxdiv = None
        mydf.mesh = numpy.asarray((11,)*3)
        mydf.eta = 0.3
        vk = mdf_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(lib.fp(vk[0]), (0.54208542933016668-0.007872205456027636j ), 9)
        self.assertAlmostEqual(lib.fp(vk[1]), (0.35976730327192219+0.0036055469686362227j), 9)
        self.assertAlmostEqual(lib.fp(vk[2]), (0.46276307618592272-0.006504349523994527j ), 9)
        self.assertAlmostEqual(lib.fp(vk[3]), (0.63667731843923825+0.0075118647005158034j), 9)
        self.assertAlmostEqual(lib.fp(vk[4]), (0.53670632359622572-0.00764236264065816j  ), 9)
        self.assertAlmostEqual(lib.fp(vk[5]), (0.4960454361832054 +0.0060590376596187257j), 9)
        self.assertAlmostEqual(lib.fp(vk[6]), (0.45421052168235576-0.006861624162215218j ), 9)
        self.assertAlmostEqual(lib.fp(vk[7]), (0.41848054629487041+0.0051096775483082746j), 9)

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
        mydf.linear_dep_threshold = 1e-7
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
        self.assertAlmostEqual(lib.fp(vk[0]), (1.0938028454012594 -0.014742352047969521j), 8)
        self.assertAlmostEqual(lib.fp(vk[1]), (0.72086205228975953+0.008685417852198867j), 8)
        self.assertAlmostEqual(lib.fp(vk[2]), (0.89846608130483796-0.011091006902191843j), 8)
        self.assertAlmostEqual(lib.fp(vk[3]), (1.260302267937254  +0.015976908047169756j), 8)
        self.assertAlmostEqual(lib.fp(vk[4]), (1.0490207113210688 -0.012426436820904021j), 8)
        self.assertAlmostEqual(lib.fp(vk[5]), (0.99252601243537697+0.012694645170334074j), 8)
        self.assertAlmostEqual(lib.fp(vk[6]), (0.92165252496655681-0.012036431811316108j), 8)
        self.assertAlmostEqual(lib.fp(vk[7]), (0.85167195537981   +0.010089165459944484j), 8)

    def test_mdf_jk_rsh(self):
        L = 4.
        cell = pgto.Cell()
        cell.verbose = 0
        cell.a = numpy.eye(3)*L
        cell.atom =[['He' , ( L/2+0., L/2+0. ,   L/2+1.)],]
        cell.basis = {'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
        cell.build()
        nao = cell.nao
        kpts = [[0.2, 0.2, 0.4]]
        numpy.random.seed(1)
        dm = numpy.random.random((nao,nao)) + .2j*numpy.random.random((nao,nao))
        dm = dm.dot(dm.conj().T).reshape(1,nao,nao)

        vj0, vk0 = pbcdf.FFTDF(cell, kpts).get_jk(dm, hermi=0, kpts=kpts, omega=0.3, exxdiv='ewald')
        vj1, vk1 = pbcdf.GDF(cell, kpts).get_jk(dm, hermi=0, kpts=kpts, omega=0.3, exxdiv='ewald')
        vj2, vk2 = pbcdf.MDF(cell, kpts).get_jk(dm, hermi=0, kpts=kpts, omega=0.3, exxdiv='ewald')
        vj3, vk3 = pbcdf.AFTDF(cell, kpts).get_jk(dm, hermi=0, kpts=kpts, omega=0.3, exxdiv='ewald')
        self.assertAlmostEqual(lib.fp(vj0), 0.007500219791944259, 9)
        self.assertAlmostEqual(lib.fp(vk0), 0.13969453408250163-0.009249150979351648j, 9)
        self.assertAlmostEqual(abs(vj0-vj1).max(), 0, 8)
        self.assertAlmostEqual(abs(vj0-vj2).max(), 0, 8)
        self.assertAlmostEqual(abs(vj0-vj3).max(), 0, 8)
        self.assertAlmostEqual(abs(vk0-vk1).max(), 0, 8)
        self.assertAlmostEqual(abs(vk0-vk2).max(), 0, 8)
        self.assertAlmostEqual(abs(vk0-vk3).max(), 0, 8)

    def test_mdf_jk_0d(self):
        L = 4.
        cell = pgto.Cell()
        cell.verbose = 0
        cell.a = numpy.eye(3)*L
        cell.atom =[['He' , ( L/2+0., L/2+0. ,   L/2+1.)],]
        cell.basis = {'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
        cell.dimension = 0
        cell.mesh = [60]*3
        cell.build()
        nao = cell.nao
        numpy.random.seed(1)
        dm = numpy.random.random((nao,nao))
        dm = dm.dot(dm.conj().T).reshape(1,nao,nao)

        vj0, vk0 = scf.hf.get_jk(cell, dm, hermi=0, omega=0.5)
        self.assertAlmostEqual(lib.fp(vj0), 0.08265798268352553, 9)
        self.assertAlmostEqual(lib.fp(vk0), 0.2375705823780625 , 9)
        vj1, vk1 = pbcdf.GDF(cell).get_jk(dm, hermi=0, omega=0.5, exxdiv=None)
        self.assertAlmostEqual(abs(vj0-vj1).max(), 0, 9)
        self.assertAlmostEqual(abs(vk0-vk1).max(), 0, 9)
        # The previous implementation of LR Coulomb kernel for dimension=0 is incorrect
        #vj2, vk2 = pbcdf.MDF(cell).get_jk(dm, hermi=0, omega=0.5, exxdiv=None)
        #vj3, vk3 = pbcdf.AFTDF(cell).get_jk(dm, hermi=0, omega=0.5, exxdiv=None)
        #self.assertAlmostEqual(abs(vj0-vj2).max(), 0, 3)
        #self.assertAlmostEqual(abs(vj0-vj3).max(), 0, 3)
        #self.assertAlmostEqual(abs(vk0-vk2).max(), 0, 3)
        #self.assertAlmostEqual(abs(vk0-vk3).max(), 0, 3)


if __name__ == '__main__':
    print("Full Tests for mdf_jk")
    unittest.main()
