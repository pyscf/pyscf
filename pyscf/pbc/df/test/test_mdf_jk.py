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
cell.rcut = 28.3
cell.build()

mf0 = pscf.RHF(cell)
mf0.exxdiv = None


def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(a.ravel(), w)

class KnowValues(unittest.TestCase):
    def test_jk(self):
        mf = mdf_jk.density_fit(mf0, auxbasis='weigend', gs=(5,)*3)
        dm = mf.get_init_guess()
        vj, vk = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 46.698952864038205, 8)
        self.assertAlmostEqual(ek1, 31.724297969654103, 8)

        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        vj1, vk1 = mf.get_jk(cell, dm, hermi=0)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 242.19367653513686, 8)
        self.assertAlmostEqual(ek1, 280.28452000317549, 8)

        numpy.random.seed(1)
        kpt = numpy.random.random(3)
        mydf = mdf.MDF(cell, [kpt])
        vj, vk = mydf.get_jk(dm, 1, kpt)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 241.29943526257196+0j, 8)
        self.assertAlmostEqual(ek1, 691.7685639227974 +0j, 8)

    def test_jk_hermi0(self):
        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        dm[:2,-3:] *= .5
        jkdf = mdf.MDF(cell)
        jkdf.auxbasis = 'weigend'
        jkdf.gs = (5,)*3
        vj0, vk0 = jkdf.get_jk(dm, hermi=0, exxdiv=None)
        ej0 = numpy.einsum('ij,ji->', vj0, dm)
        ek0 = numpy.einsum('ij,ji->', vk0, dm)
        self.assertAlmostEqual(ej0, 242.18843656969429, 8)
        self.assertAlmostEqual(ek0, 280.70983914362279, 8)

    def test_j_kpts(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = mdf.MDF(cell)
        mydf.kpts = numpy.random.random((4,3))
        mydf.gs = numpy.asarray((5,)*3)
        mydf.auxbasis = 'weigend'
        vj = mdf_jk.get_j_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(finger(vj[0]), (0.48230415575990992-0.1187259816712561j  ), 9)
        self.assertAlmostEqual(finger(vj[1]), (0.54076441691691635-0.046133570735633858j), 9)
        self.assertAlmostEqual(finger(vj[2]), (0.52809528766588443-0.083705362039349696j), 9)
        self.assertAlmostEqual(finger(vj[3]), (0.54354803346397418+0.008843828278287412j), 9)

    def test_k_kpts(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = mdf.MDF(cell)
        mydf.kpts = numpy.random.random((4,3))
        mydf.gs = numpy.asarray((5,)*3)
        mydf.exxdiv = None
        mydf.auxbasis = 'weigend'
        vk = mdf_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (-2.8420690776723521-1.0520032225125369j ), 9)
        self.assertAlmostEqual(finger(vk[1]), (-7.4484073115980589+0.10323612296497409j), 9)
        self.assertAlmostEqual(finger(vk[2]), (-2.5801797461396694-1.4470142156554828j ), 9)
        self.assertAlmostEqual(finger(vk[3]), (-0.7965970307739134+0.01197284205215638j), 9)

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
        mydf = mdf.MDF(cell)
        mydf.kpts = kpts
        mydf.auxbasis = 'weigend'
        mydf.exxdiv = None
        vk = mdf_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (0.54208523897216732-0.0078722056426767627j), 9)
        self.assertAlmostEqual(finger(vk[1]), (0.35976712320322424+0.0036055471233112256j), 9)
        self.assertAlmostEqual(finger(vk[2]), (0.46276289905549617-0.0065043496646110801j), 9)
        self.assertAlmostEqual(finger(vk[3]), (0.63667714700745526+0.0075118648848031978j), 9)
        self.assertAlmostEqual(finger(vk[4]), (0.53670615076993111-0.0076423628684680065j), 9)
        self.assertAlmostEqual(finger(vk[5]), (0.49604529705337863+0.0060590377882828398j), 9)
        self.assertAlmostEqual(finger(vk[6]), (0.45421039104866701-0.0068616242937238692j), 9)
        self.assertAlmostEqual(finger(vk[7]), (0.41848041777106704+0.0051096776806240215j), 9)

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
        mydf = mdf.MDF(cell)
        mydf.kpts = kpts
        mydf.auxbasis = 'weigend'
        mydf.exxdiv = None
        nao = cell.nao_nr()
        numpy.random.seed(1)
        dm = numpy.random.random((8,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        vk = mdf_jk.get_k_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (1.093802464678856  -0.014742352431919676j), 9)
        self.assertAlmostEqual(finger(vk[1]), (0.72086169197091587+0.008685418217133528j), 9)
        self.assertAlmostEqual(finger(vk[2]), (0.89846572708244654-0.011091007223188223j), 9)
        self.assertAlmostEqual(finger(vk[3]), (1.2603019232380572 +0.015976908426091924j), 9)
        self.assertAlmostEqual(finger(vk[4]), (1.0490203655653567 -0.012426437226316521j), 9)
        self.assertAlmostEqual(finger(vk[5]), (0.99252573352507678+0.012694645461974137j), 9)
        self.assertAlmostEqual(finger(vk[6]), (0.92165226277159285-0.012036432073368958j), 9)
        self.assertAlmostEqual(finger(vk[7]), (0.85167169839559342+0.010089165714668009j), 9)


if __name__ == '__main__':
    print("Full Tests for mdf_jk")
    unittest.main()
