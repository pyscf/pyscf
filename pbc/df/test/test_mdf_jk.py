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
cell.build()
cell.rcut = 28.3

cell1 = cell.copy()
cell1.rcut = 17
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
        self.assertAlmostEqual(ej1, 46.698944426560104, 8)
        self.assertAlmostEqual(ek1, 31.723136985575099, 8)

        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        vj1, vk1 = mf.get_jk(cell, dm, hermi=0)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 242.18066893105836, 8)
        self.assertAlmostEqual(ek1, 280.27749933492305, 8)

        numpy.random.seed(1)
        kpt = numpy.random.random(3)
        mydf = mdf.MDF(cell, [kpt])
        vj, vk = mydf.get_jk(dm, 1, kpt)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 240.96370951556852, 8)
        self.assertAlmostEqual(ek1, 691.41117108086291, 8)

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
        self.assertAlmostEqual(ej0, 242.17542855854242, 8)
        self.assertAlmostEqual(ek0, 280.7028167826453 , 8)

    def test_jk_metric(self):
        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        jkdf = mdf.MDF(cell)
        jkdf.metric = 'S'
        jkdf.auxbasis = 'weigend'
        jkdf.gs = (5,)*3
        vj1, vk1 = jkdf.get_jk(dm, exxdiv=None)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 242.1754481772021, 8)
        self.assertAlmostEqual(ek1, 280.2751263818186, 8)

        jkdf = mdf.MDF(cell)
        jkdf.metric = 'T'
        jkdf.auxbasis = 'weigend'
        jkdf.gs = (5,)*3
        vj1, vk1 = jkdf.get_jk(dm, exxdiv=None)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 242.18066893105842, 8)
        self.assertAlmostEqual(ek1, 280.27749933492305, 8)

    def test_jk_pbc_local_fit(self):
        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        jkdf = mdf.MDF(cell)
        jkdf.metric = 'S'
        jkdf.auxbasis = 'weigend'
        jkdf.approx_sr_level = 2
        jkdf.gs = (5,)*3
        mf = mdf_jk.density_fit(mf0, with_df=jkdf)
        vj1, vk1 = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 242.17560414134402, 8)
        self.assertAlmostEqual(ek1, 280.27525544272237, 8)

        jkdf = mdf.MDF(cell)
        jkdf.metric = 'T'
        jkdf.approx_sr_level = 3
        jkdf.auxbasis = 'weigend'
        jkdf.gs = (5,)*3
        mf = mdf_jk.density_fit(mf0, with_df=jkdf)
        vj1, vk1 = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 242.1790506110278, 8)
        self.assertAlmostEqual(ek1, 280.2763653230468, 8)

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
        self.assertAlmostEqual(finger(vj[0]), (0.48110100011270873-0.1187207516057024j  ), 9)
        self.assertAlmostEqual(finger(vj[1]), (0.53956267439270444-0.046130125909028916j), 9)
        self.assertAlmostEqual(finger(vj[2]), (0.52689349636674265-0.083698423012819212j), 9)
        self.assertAlmostEqual(finger(vj[3]), (0.54234901696511795+0.00883535738463977j ), 9)

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
        self.assertAlmostEqual(finger(vk[0]), (-2.841991690391537  -1.0531936354491773j  ), 9)
        self.assertAlmostEqual(finger(vk[1]), (-7.4479409613933898 +0.10325698693704466j ), 9)
        self.assertAlmostEqual(finger(vk[2]), (-2.5789885244178263 -1.4459525565105991j  ), 9)
        self.assertAlmostEqual(finger(vk[3]), (-0.79603867462917455+0.011991290978492469j), 9)

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
        self.assertAlmostEqual(finger(vk[0]), (0.5418385019142542 -0.0078724013505413332j), 9)
        self.assertAlmostEqual(finger(vk[1]), (0.35952124697656695+0.0036057189081886336j), 9)
        self.assertAlmostEqual(finger(vk[2]), (0.4625120878413842 -0.0065045494016886737j), 9)
        self.assertAlmostEqual(finger(vk[3]), (0.63641999973765473+0.0075120497797564077j), 9)
        self.assertAlmostEqual(finger(vk[4]), (0.53644934892347451-0.0076425730500473956j), 9)
        self.assertAlmostEqual(finger(vk[5]), (0.49579571224501451+0.006059214200031502j ), 9)
        self.assertAlmostEqual(finger(vk[6]), (0.4539578168628049 -0.0068618292524707204j), 9)
        self.assertAlmostEqual(finger(vk[7]), (0.41822536109677522+0.0051098609317817099j), 9)

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
        self.assertAlmostEqual(finger(vk[0]), (1.0933071616067864 -0.01474271237640193j ), 9)
        self.assertAlmostEqual(finger(vk[1]), (0.72036827661735581+0.008685775722653022j), 9)
        self.assertAlmostEqual(finger(vk[2]), (0.8979620015620533 -0.011091372770818691j), 9)
        self.assertAlmostEqual(finger(vk[3]), (1.2597854134048272 +0.015977292598694546j), 9)
        self.assertAlmostEqual(finger(vk[4]), (1.0485047112408323 -0.012426816811955727j), 9)
        self.assertAlmostEqual(finger(vk[5]), (0.99202495287727066+0.012695003848138148j), 9)
        self.assertAlmostEqual(finger(vk[6]), (0.92114488719642162-0.012036800944772899j), 9)
        self.assertAlmostEqual(finger(vk[7]), (0.85115980145319536+0.010089537647891509j), 9)


if __name__ == '__main__':
    print("Full Tests for mdf_jk")
    unittest.main()
