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
    def test_jk_single_kpt(self):
        mf = mdf_jk.density_fit(mf0, auxbasis='weigend', gs=(5,)*3)
        mf.with_df.gs = [5]*3
        mf.with_df.eta = 0.3
        dm = mf.get_init_guess()
        vj, vk = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 46.698952864035398, 8)
        self.assertAlmostEqual(ek1, 31.724297969652138, 8)

        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        vj1, vk1 = mf.get_jk(cell, dm, hermi=0)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 242.19367653513683, 8)
        self.assertAlmostEqual(ek1, 280.28452000317401, 8)

        numpy.random.seed(1)
        kpt = numpy.random.random(3)
        mydf = mdf.MDF(cell, [kpt]).set(auxbasis='weigend')
        mydf.gs = [5]*3
        mydf.eta = 0.3
        vj, vk = mydf.get_jk(dm, 1, kpt)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 241.29943526257165+0j, 8)
        self.assertAlmostEqual(ek1, 691.76856392279456+0j, 8)

    def test_jk_hermi0(self):
        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        dm[:2,-3:] *= .5
        jkdf = mdf.MDF(cell).set(auxbasis='weigend')
        jkdf.gs = (5,)*3
        jkdf.eta = 0.3
        vj0, vk0 = jkdf.get_jk(dm, hermi=0, exxdiv=None)
        ej0 = numpy.einsum('ij,ji->', vj0, dm)
        ek0 = numpy.einsum('ij,ji->', vk0, dm)
        self.assertAlmostEqual(ej0, 242.1884365696942 , 8)
        self.assertAlmostEqual(ek0, 280.70983914362148, 8)

    def test_j_kpts(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = mdf.MDF(cell).set(auxbasis='weigend')
        mydf.kpts = numpy.random.random((4,3))
        mydf.gs = numpy.asarray((5,)*3)
        mydf.eta = 0.3
        mydf.auxbasis = 'weigend'
        vj = mdf_jk.get_j_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(finger(vj[0]), (0.48230415575983976-0.11872598167125974j  ), 9)
        self.assertAlmostEqual(finger(vj[1]), (0.5407644169168595 -0.046133570735630479j ), 9)
        self.assertAlmostEqual(finger(vj[2]), (0.52809528766575564-0.083705362039349793j ), 9)
        self.assertAlmostEqual(finger(vj[3]), (0.54354803346388358+0.0088438282782892438j), 9)

    def test_k_kpts(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = mdf.MDF(cell).set(auxbasis='weigend')
        mydf.kpts = numpy.random.random((4,3))
        mydf.gs = numpy.asarray((5,)*3)
        mydf.eta = 0.3
        mydf.exxdiv = None
        mydf.auxbasis = 'weigend'
        vk = mdf_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (-2.8420690776723045-1.0520032225121236j ), 9)
        self.assertAlmostEqual(finger(vk[1]), (-7.4484073115981797+0.10323612296501217j), 9)
        self.assertAlmostEqual(finger(vk[2]), (-2.5801797461394251-1.4470142156560293j ), 9)
        self.assertAlmostEqual(finger(vk[3]), (-0.7965970307738880+0.01197284205213897j), 9)

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
        mydf = mdf.MDF(cell).set(auxbasis='weigend')
        mydf.kpts = kpts
        mydf.auxbasis = 'weigend'
        mydf.exxdiv = None
        mydf.gs = numpy.asarray((5,)*3)
        mydf.eta = 0.3
        vk = mdf_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (0.54208542933016868-0.007872205456027688j ), 9)
        self.assertAlmostEqual(finger(vk[1]), (0.35976730327192064+0.0036055469686364274j), 9)
        self.assertAlmostEqual(finger(vk[2]), (0.46276307618592205-0.0065043495239945522j), 9)
        self.assertAlmostEqual(finger(vk[3]), (0.63667731843923825+0.0075118647005158069j), 9)
        self.assertAlmostEqual(finger(vk[4]), (0.53670632359622705-0.0076423626406581808j), 9)
        self.assertAlmostEqual(finger(vk[5]), (0.49604543618320496+0.0060590376596186381j), 9)
        self.assertAlmostEqual(finger(vk[6]), (0.45421052168235398-0.006861624162215172j ), 9)
        self.assertAlmostEqual(finger(vk[7]), (0.41848054629486886+0.00510967754830822j  ), 9)

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
        mydf = mdf.MDF(cell).set(auxbasis='weigend')
        mydf.kpts = kpts
        mydf.auxbasis = 'weigend'
        mydf.exxdiv = None
        mydf.gs = numpy.asarray((5,)*3)
        mydf.eta = 0.3
        nao = cell.nao_nr()
        numpy.random.seed(1)
        dm = numpy.random.random((8,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        vk = mdf_jk.get_k_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (1.0938028454012594 -0.014742352047969667j), 9)
        self.assertAlmostEqual(finger(vk[1]), (0.72086205228976086+0.008685417852198991j), 9)
        self.assertAlmostEqual(finger(vk[2]), (0.89846608130483663-0.011091006902191652j), 9)
        self.assertAlmostEqual(finger(vk[3]), (1.2603022679372518 +0.015976908047169988j), 9)
        self.assertAlmostEqual(finger(vk[4]), (1.0490207113210683 -0.0124264368209042j  ), 9)
        self.assertAlmostEqual(finger(vk[5]), (0.99252601243537519+0.012694645170333901j), 9)
        self.assertAlmostEqual(finger(vk[6]), (0.92165252496655681-0.012036431811316016j), 9)
        self.assertAlmostEqual(finger(vk[7]), (0.85167195537980778+0.010089165459944104j), 9)


if __name__ == '__main__':
    print("Full Tests for mdf_jk")
    unittest.main()

