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
    def test_jk(self):
        mf = df_jk.density_fit(mf0, auxbasis='weigend', gs=(5,)*3)
        dm = mf.get_init_guess()
        vj, vk = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 46.698885887058495, 8)
        self.assertAlmostEqual(ek1, 31.723490322441389, 8)

        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        vj1, vk1 = mf.get_jk(cell, dm, hermi=0)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 242.04675958581976, 8)
        self.assertAlmostEqual(ek1, 280.15934765885805, 8)

        numpy.random.seed(1)
        kpt = numpy.random.random(3)
        mydf = df.DF(cell, [kpt])
        vj, vk = mydf.get_jk(dm, 1, kpt, exxdiv=None)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 241.15119653657837+0j, 8)
        self.assertAlmostEqual(ek1, 279.64649180139776+0j, 8)
        vj, vk = mydf.get_jk(dm, 1, kpt, with_j=False, exxdiv='ewald')
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ek1, 691.6462444241248+0j, 6)

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
        self.assertAlmostEqual(ej0, 242.0414892686764 , 8)
        self.assertAlmostEqual(ek0, 280.58443078534907, 8)

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
        self.assertAlmostEqual(finger(vj[0]), (0.49176565003742523-0.11891097171204273j ), 9)
        self.assertAlmostEqual(finger(vj[1]), (0.54901219181783034-0.046003618294160657j), 9)
        self.assertAlmostEqual(finger(vj[2]), (0.53648483057543039-0.083507647862465353j), 9)
        self.assertAlmostEqual(finger(vj[3]), (0.54896892736198399+0.007695802002441080j), 9)

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
        self.assertAlmostEqual(finger(vk[0]), (-2.8332405542539822 -1.0578703659011341j ), 9)
        self.assertAlmostEqual(finger(vk[1]), (-7.4404337144963018 +0.10233793208112751j), 9)
        self.assertAlmostEqual(finger(vk[2]), (-2.5718873173309404 -1.4487392786249131j ), 9)
        self.assertAlmostEqual(finger(vk[3]), (-0.79223061014449159+0.01169464922520067j), 9)

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
        mydf.auxbasis = {'He': [(0, (4.096, 1)), (0, (2.56, 1)), (0, (1.6, 1)), (0, (1., 1))]}
        mydf.exxdiv = None
        vk = df_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (0.5420051395935308 -0.0078629895406937093j), 9)
        self.assertAlmostEqual(finger(vk[1]), (0.35967745300781129+0.0036013741011200534j), 9)
        self.assertAlmostEqual(finger(vk[2]), (0.46267875636311051-0.0064934210041723787j), 9)
        self.assertAlmostEqual(finger(vk[3]), (0.63657133813146305+0.0075058298866949071j), 9)
        self.assertAlmostEqual(finger(vk[4]), (0.53658686350689644-0.0076348273858712692j), 9)
        self.assertAlmostEqual(finger(vk[5]), (0.49596594079072709+0.0060524166304689085j), 9)
        self.assertAlmostEqual(finger(vk[6]), (0.45414482973739956-0.0068539768439524994j), 9)
        self.assertAlmostEqual(finger(vk[7]), (0.41836048318983377+0.0051095605455669692j), 9)

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
        mydf.auxbasis = {'He': [(0, (4.096, 1)), (0, (2.56, 1)), (0, (1.6, 1)), (0, (1., 1))]}
        mydf.exxdiv = None
        nao = cell.nao_nr()
        numpy.random.seed(1)
        dm = numpy.random.random((8,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        vk = df.df_jk.get_k_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (1.0936431020073591 -0.014722943557931367j), 9)
        self.assertAlmostEqual(finger(vk[1]), (0.72071018138503629+0.008672611686305542j), 9)
        self.assertAlmostEqual(finger(vk[2]), (0.89832293233163618-0.011069550356504438j), 9)
        self.assertAlmostEqual(finger(vk[3]), (1.2601082118723275 +0.015960560764962047j), 9)
        self.assertAlmostEqual(finger(vk[4]), (1.0488000478468562 -0.012410766448146473j), 9)
        self.assertAlmostEqual(finger(vk[5]), (0.99237174968627428+0.012676287846870899j), 9)
        self.assertAlmostEqual(finger(vk[6]), (0.92151251108591792-0.012017833726331088j), 9)
        self.assertAlmostEqual(finger(vk[7]), (0.85142552451225129+0.010081726860909411j), 9)


if __name__ == '__main__':
    print("Full Tests for df_jk")
    unittest.main()
