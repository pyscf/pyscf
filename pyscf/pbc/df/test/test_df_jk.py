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
cell.rcut = 28.3458918685
cell.build()

mf0 = pscf.RHF(cell)
mf0.exxdiv = None


def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(a.ravel(), w)

class KnowValues(unittest.TestCase):
    def test_jk_single_kpt(self):
        mf = df_jk.density_fit(mf0, auxbasis='weigend', gs=(5,)*3)
        mf.with_df.gs = cell.gs
        mf.with_df.eta = 0.3
        dm = mf.get_init_guess()
        vj, vk = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 46.69888588854387, 8)
        self.assertAlmostEqual(ek1, 31.72349032270801, 8)

        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        vj1, vk1 = mf.get_jk(cell, dm, hermi=0)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 242.04675958582985, 8)
        self.assertAlmostEqual(ek1, 280.15934765887238, 8)

        numpy.random.seed(1)
        kpt = numpy.random.random(3)
        mydf = df.DF(cell, [kpt]).set(auxbasis='weigend')
        mydf.gs = cell.gs
        mydf.eta = 0.3
        vj, vk = mydf.get_jk(dm, 1, kpt, exxdiv=None)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 241.1511965365826 +0j, 8)
        self.assertAlmostEqual(ek1, 279.64649180140344+0j, 8)
        vj, vk = mydf.get_jk(dm, 1, kpt, with_j=False, exxdiv='ewald')
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ek1, 691.64624442413174+0j, 6)

    def test_jk_hermi0(self):
        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        dm[:2,-3:] *= .5
        jkdf = df.DF(cell).set(auxbasis='weigend')
        jkdf.gs = (5,)*3
        jkdf.eta = 0.3
        vj0, vk0 = jkdf.get_jk(dm, hermi=0, exxdiv=None)
        ej0 = numpy.einsum('ij,ji->', vj0, dm)
        ek0 = numpy.einsum('ij,ji->', vk0, dm)
        self.assertAlmostEqual(ej0, 242.04148926868635, 8)
        self.assertAlmostEqual(ek0, 280.58443078536345, 8)

    def test_j_kpts(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = df.DF(cell).set(auxbasis='weigend')
        mydf.kpts = numpy.random.random((4,3))
        mydf.gs = numpy.asarray((5,)*3)
        mydf.auxbasis = 'weigend'
        mydf.gs = cell.gs
        mydf.eta = 0.3
        vj = df_jk.get_j_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(finger(vj[0]), (0.49176565003906081-0.11891097171192379j  ), 9)
        self.assertAlmostEqual(finger(vj[1]), (0.54901219181920746-0.046003618294119357j ), 9)
        self.assertAlmostEqual(finger(vj[2]), (0.53648483057579632-0.083507647862342937j ), 9)
        self.assertAlmostEqual(finger(vj[3]), (0.54896892736259639+0.0076958020023144037j), 9)

    def test_k_kpts(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = df.DF(cell).set(auxbasis='weigend')
        mydf.kpts = numpy.random.random((4,3))
        mydf.gs = numpy.asarray((5,)*3)
        mydf.exxdiv = None
        mydf.gs = cell.gs
        mydf.eta = 0.3
        mydf.auxbasis = 'weigend'
        vk = df_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (-2.8332405542541035 -1.0578703659011017j  ), 9)
        self.assertAlmostEqual(finger(vk[1]), (-7.4404337144965327 +0.10233793208110664j ), 9)
        self.assertAlmostEqual(finger(vk[2]), (-2.5718873173308587 -1.4487392786249167j  ), 9)
        self.assertAlmostEqual(finger(vk[3]), (-0.79223061014463669+0.011694649225196911j), 9)

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
        mydf = df.DF(cell).set(auxbasis='weigend')
        mydf.kpts = kpts
        mydf.auxbasis = {'He': [(0, (4.096, 1)), (0, (2.56, 1)), (0, (1.6, 1)), (0, (1., 1))]}
        mydf.exxdiv = None
        mydf.gs = cell.gs
        mydf.eta = 0.3
        vk = df_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (0.54220010040518085-0.0078720429568215483j), 9)
        self.assertAlmostEqual(finger(vk[1]), (0.35987105007103337+0.0036047438452914572j), 9)
        self.assertAlmostEqual(finger(vk[2]), (0.46287057223452033-0.0065045318150074175j), 9)
        self.assertAlmostEqual(finger(vk[3]), (0.63677390788341892+0.007513208153323357j ), 9)
        self.assertAlmostEqual(finger(vk[4]), (0.53680188658523043-0.0076414750780819194j), 9)
        self.assertAlmostEqual(finger(vk[5]), (0.49613855046498667+0.006060376738372739j ), 9)
        self.assertAlmostEqual(finger(vk[6]), (0.45430752211148873-0.0068611602260907067j), 9)
        self.assertAlmostEqual(finger(vk[7]), (0.41856931218761884+0.0051073315206036857j), 9)

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
        mydf = df.DF(cell).set(auxbasis='weigend')
        mydf.kpts = kpts
        mydf.auxbasis = {'He': [(0, (4.096, 1)), (0, (2.56, 1)), (0, (1.6, 1)), (0, (1., 1))]}
        mydf.exxdiv = None
        mydf.gs = cell.gs
        mydf.eta = 0.3
        nao = cell.nao_nr()
        numpy.random.seed(1)
        dm = numpy.random.random((8,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        vk = df.df_jk.get_k_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (1.0940331326660706 -0.014742469831921495j), 9)
        self.assertAlmostEqual(finger(vk[1]), (0.72106828546203339+0.008683360062579130j), 9)
        self.assertAlmostEqual(finger(vk[2]), (0.89868267009697655-0.011091489111887209j), 9)
        self.assertAlmostEqual(finger(vk[3]), (1.2604941401190817 +0.015979544115388003j), 9)
        self.assertAlmostEqual(finger(vk[4]), (1.0492129520812483 -0.012424653667353895j), 9)
        self.assertAlmostEqual(finger(vk[5]), (0.99271107721955065+0.012696925711379314j), 9)
        self.assertAlmostEqual(finger(vk[6]), (0.92184754518869783-0.012035727588119237j), 9)
        self.assertAlmostEqual(finger(vk[7]), (0.851848314862798  +0.010084767506087691j), 9)


if __name__ == '__main__':
    print("Full Tests for df_jk")
    unittest.main()

