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

L = 5.
n = 5
cell = pgto.Cell()
cell.a = numpy.diag([L,L,L])
cell.gs = numpy.array([n,n,n])

cell.atom = '''C    3.    2.       3.
               C    1.    1.       1.'''
cell.basis = 'ccpvdz'
cell.verbose = 0
cell.max_memory = 0
cell.build()
cell.rcut = 28.3458918685

cell1 = cell.copy()
cell1.rcut = 17
mf0 = pscf.RHF(cell)
mf0.exxdiv = None


def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(a.ravel(), w)

class KnowValues(unittest.TestCase):
    def test_jk(self):
        mf = df_jk.density_fit(mf0, auxbasis='weigend', gs=(5,)*3)
        dm = mf.get_init_guess()
        vj, vk = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 46.698942503368144, 8)
        self.assertAlmostEqual(ek1, 31.723584951501   , 8)

        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        vj1, vk1 = mf.get_jk(cell, dm, hermi=0)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 242.04814846354407, 8)
        self.assertAlmostEqual(ek1, 280.16142282936778, 8)

        numpy.random.seed(1)
        kpt = numpy.random.random(3)
        mydf = df.DF(cell, [kpt])
        vj, vk = mydf.get_jk(dm, 1, kpt)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 231.44815444295386+0j, 8)
        self.assertAlmostEqual(ek1, 682.70050207852307+0j, 8)

    def test_jk_hermi0(self):
        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        dm[:2,-3:] *= .5
        jkdf = df.DF(cell)
        jkdf.auxbasis = 'weigend'
        jkdf.gs = (5,)*3
        vj0, vk0 = jkdf.get_jk(dm, hermi=0, exxdiv=None)
        ej0 = numpy.einsum('ij,ji->', vj0, dm)
        ek0 = numpy.einsum('ij,ji->', vk0, dm)
        self.assertAlmostEqual(ej0, 242.04287865428188, 8)
        self.assertAlmostEqual(ek0, 280.58648990708701, 8)

    def test_j_kpts(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = df.DF(cell)
        mydf.kpts = numpy.random.random((4,3))
        mydf.gs = numpy.asarray((5,)*3)
        mydf.auxbasis = 'weigend'
        vj = df_jk.get_j_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(finger(vj[0]), (0.49071716424863876-0.11866593764325209j ), 9)
        self.assertAlmostEqual(finger(vj[1]), (0.54805564025672737-0.045780766515833224j), 9)
        self.assertAlmostEqual(finger(vj[2]), (0.53548588134342712-0.083269294794914239j), 9)
        self.assertAlmostEqual(finger(vj[3]), (0.54778931331075764+0.007709304083260985j), 9)

    def test_k_kpts(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = df.DF(cell)
        mydf.kpts = numpy.random.random((4,3))
        mydf.gs = numpy.asarray((5,)*3)
        mydf.exxdiv = None
        mydf.auxbasis = 'weigend'
        vk = df_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (-2.8338631448322422 -1.0571235846726912j ), 9)
        self.assertAlmostEqual(finger(vk[1]), (-7.4393097135258062 +0.10265042873875296j), 9)
        self.assertAlmostEqual(finger(vk[2]), (-2.5706492707654505 -1.4482274480570361j ), 9)
        self.assertAlmostEqual(finger(vk[3]), (-0.78973046762474153+0.01154399375447295j), 9)

    def test_k_kpts_1(self):
        cell = pgto.Cell()
        cell.atom = 'He 1. .5 .5; He .1 1.3 2.1'
        cell.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))]}
        cell.a = numpy.eye(3) * 2.5
        cell.gs = [5] * 3
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
        mydf = df.DF(cell)
        mydf.kpts = kpts
        mydf.auxbasis = 'weigend'
        mydf.exxdiv = None
        vk = df_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (0.54015197313324537-0.0079752072085274676j), 9)
        self.assertAlmostEqual(finger(vk[1]), (0.3582520398839657 +0.0036801628264896014j), 9)
        self.assertAlmostEqual(finger(vk[2]), (0.46108289884408604-0.0066080922040621061j), 9)
        self.assertAlmostEqual(finger(vk[3]), (0.63466063181822774+0.0075857948337662746j), 9)
        self.assertAlmostEqual(finger(vk[4]), (0.53482612780118588-0.0077463092279198443j), 9)
        self.assertAlmostEqual(finger(vk[5]), (0.49433633153410605+0.0061326611410119211j), 9)
        self.assertAlmostEqual(finger(vk[6]), (0.45244565342177845-0.0069645425083198157j), 9)
        self.assertAlmostEqual(finger(vk[7]), (0.41675926649079686+0.0051848485986134649j), 9)

    def test_k_kpts_2(self):
        cell = pgto.Cell()
        cell.atom = 'He 1. .5 .5; He .1 1.3 2.1'
        cell.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))]}
        cell.a = numpy.eye(3) * 2.5
        cell.gs = [5] * 3
        cell.build()
        kpts = cell.get_abs_kpts([[-.25,-.25,-.25],
                                  [-.25,-.25, .25],
                                  [-.25, .25,-.25],
                                  [-.25, .25, .25],
                                  [ .25,-.25,-.25],
                                  [ .25,-.25, .25],
                                  [ .25, .25,-.25],
                                  [ .25, .25, .25]])
        mydf = df.DF(cell)
        mydf.kpts = kpts
        mydf.auxbasis = 'weigend'
        mydf.exxdiv = None
        nao = cell.nao_nr()
        numpy.random.seed(1)
        dm = numpy.random.random((8,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        vk = df_jk.get_k_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (1.0898939666590253 -0.014906612874026948j), 9)
        self.assertAlmostEqual(finger(vk[1]), (0.71766985010790085+0.008841808731355736j), 9)
        self.assertAlmostEqual(finger(vk[2]), (0.89500717500394655-0.011255970881451824j), 9)
        self.assertAlmostEqual(finger(vk[3]), (1.2561632091272155 +0.016131846517374523j), 9)
        self.assertAlmostEqual(finger(vk[4]), (1.0452056696649636 -0.012591023244846416j), 9)
        self.assertAlmostEqual(finger(vk[5]), (0.98906700489431287+0.012847978938583308j), 9)
        self.assertAlmostEqual(finger(vk[6]), (0.91811462227257712-0.012200120679750136j), 9)
        self.assertAlmostEqual(finger(vk[7]), (0.84819570235790342+0.010245262542632964j), 9)


if __name__ == '__main__':
    print("Full Tests for df_jk")
    unittest.main()
