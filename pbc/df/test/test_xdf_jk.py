import unittest
import numpy
from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
import pyscf.pbc
from pyscf.pbc.df import xdf
from pyscf.pbc.df import xdf_jk
pyscf.pbc.DEBUG = False

L = 5.
n = 5
cell = pgto.Cell()
cell.h = numpy.diag([L,L,L])
cell.gs = numpy.array([n,n,n])

cell.atom = '''C    3.    2.       3.
               C    1.    1.       1.'''
cell.basis = 'ccpvdz'
cell.verbose = 0
cell.max_memory = 0
cell.build(0,0)
cell.nimgs = [2,2,2]

mf0 = pscf.RHF(cell)
mf0.exxdiv = None


def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(a.ravel(), w)

class KnowValues(unittest.TestCase):
    def test_jk(self):
        mf = xdf_jk.density_fit(mf0, auxbasis='weigend', gs=(5,)*3)
        dm = mf.get_init_guess()
        vj, vk = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 46.697454798109035, 9)
        self.assertAlmostEqual(ek1, 31.712466152524634, 9)

        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        vj1, vk1 = mf.get_jk(cell, dm, hermi=0)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 241.76026924116567, 9)
        self.assertAlmostEqual(ek1, 279.18873067513761, 9)

    def test_hcore(self):
        mf = pscf.RHF(cell)
        odf = xdf.XDF(cell)
        odf.auxbasis = 'weigend'
        odf.gs = (10,)*3
        dm = mf.get_init_guess()
        self.assertAlmostEqual(numpy.einsum('ij,ji->', odf.get_nuc(), dm),
                               -150.6108333022498, 9)

    def test_jk_hermi0(self):
        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        dm[:2,-3:] *= .5
        jkdf = xdf.XDF(cell)
        jkdf.auxbasis = 'weigend'
        jkdf.gs = (5,)*3
        mf = xdf_jk.density_fit(mf0, with_df=jkdf)
        vj0, vk0 = mf.get_jk(cell, dm, hermi=0)
        ej0 = numpy.einsum('ij,ji->', vj0, dm)
        ek0 = numpy.einsum('ij,ji->', vk0, dm)
        mf = xdf_jk.density_fit(mf0, with_df=jkdf)
        vj1, vk1 = mf.get_jk(cell, dm, hermi=0)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertTrue(numpy.allclose(vj0, vj1))
        self.assertTrue(numpy.allclose(vk0, vk1))
        self.assertAlmostEqual(ej1, 241.75501486411383, 9)
        self.assertAlmostEqual(ek1, 279.59327636982499, 9)

    def test_jk_metric(self):
        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        jkdf = xdf.XDF(cell)
        jkdf.metric = 'S'
        jkdf.auxbasis = 'weigend'
        jkdf.gs = (5,)*3
        mf = xdf_jk.density_fit(mf0, with_df=jkdf)
        vj1, vk1 = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 240.98642886823077, 9)
        self.assertAlmostEqual(ek1, 278.11153195242434, 9)

        jkdf = xdf.XDF(cell)
        jkdf.metric = 'T'
        jkdf.auxbasis = 'weigend'
        jkdf.gs = (5,)*3
        mf = xdf_jk.density_fit(mf0, with_df=jkdf)
        vj1, vk1 = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 241.76026924116567, 9)
        self.assertAlmostEqual(ek1, 279.18873067513761, 9)

    def test_jk_pbc_local_fit(self):
        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        jkdf = xdf.XDF(cell)
        jkdf.metric = 'S'
        jkdf.auxbasis = 'weigend'
        jkdf.approx_sr_level = 0
        jkdf.gs = (5,)*3
        mf = xdf_jk.density_fit(mf0, with_df=jkdf)
        vj1, vk1 = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 240.98642886823077, 9)
        self.assertAlmostEqual(ek1, 278.11153195242434, 9)

        jkdf = xdf.XDF(cell)
        jkdf.metric = 'T'
        jkdf.approx_sr_level = 0
        jkdf.auxbasis = 'weigend'
        jkdf.gs = (5,)*3
        mf = xdf_jk.density_fit(mf0, with_df=jkdf)
        vj1, vk1 = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 241.76026924116567, 9)
        self.assertAlmostEqual(ek1, 279.18873067513761, 9)

    def test_j_kpts(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = xdf.XDF(cell)
        mydf.kpts = numpy.random.random((4,3))
        mydf.gs = numpy.asarray((5,)*3)
        mydf.auxbasis = 'weigend'
        vj = xdf_jk.get_j_kpts(mydf, dm, 1, None, mydf.kpts)
        self.assertAlmostEqual(finger(vj[0]), (0.75809296258436376-0.47108542280677335j)/4, 9)
        self.assertAlmostEqual(finger(vj[1]), (0.99241082422673799-0.18223267233036788j)/4, 9)
        self.assertAlmostEqual(finger(vj[2]), (0.94179957013540516-0.33286536411836043j)/4, 9)
        self.assertAlmostEqual(finger(vj[3]), (1.0171725763002968+0.039929046643852481j)/4, 9)

    def test_k_kpts(self):
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((4,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mydf = xdf.XDF(cell)
        mydf.kpts = numpy.random.random((4,3))
        mydf.gs = numpy.asarray((5,)*3)
        mydf.exxdiv = None
        mydf.auxbasis = 'weigend'
        vk = xdf_jk.get_k_kpts(mydf, dm, 0, mydf, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (-11.344881397590214-4.2410705105331532j)  /4, 9)
        self.assertAlmostEqual(finger(vk[1]), (-29.79023682381176+0.29675103120189483j)  /4, 9)
        self.assertAlmostEqual(finger(vk[2]), (-10.326273314290356-5.7194950445332005j)  /4, 9)
        self.assertAlmostEqual(finger(vk[3]), (-3.1797509893478519+0.040327565044949137j)/4, 9)

    def test_k_kpts_2(self):
        import pyscf.pbc.tools.pyscf_ase as pyscf_ase
        cell = pgto.Cell()
        cell.atom = 'He 1. .5 .5; He .1 1.3 2.1'
        cell.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))]}
        cell.h = numpy.eye(3) * 2.5
        cell.gs = [5] * 3
        cell.build()
        kpts = pyscf_ase.make_kpts(cell, (2,2,2))

        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((8,nao,nao))
        mydf = xdf.XDF(cell)
        mydf.kpts = kpts
        mydf.auxbasis = 'weigend'
        mydf.exxdiv = None
        vk = xdf_jk.get_k_kpts(mydf, dm, 0, mydf, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (4.3347841925298098-0.062977291500864604j)/8, 9)
        self.assertAlmostEqual(finger(vk[1]), (2.8761015271990602+0.028849511945164479j)/8, 9)
        self.assertAlmostEqual(finger(vk[2]), (3.7000171799264021-0.052042703595784535j)/8, 9)
        self.assertAlmostEqual(finger(vk[3]), (5.0914370273725105+0.06009717904159529j )/8, 9)
        self.assertAlmostEqual(finger(vk[4]), (4.2916074730699894-0.061141166602384132j)/8, 9)
        self.assertAlmostEqual(finger(vk[5]), (3.9664272019734872+0.048476723428293463j)/8, 9)
        self.assertAlmostEqual(finger(vk[6]), (3.6317277588494328-0.0549011310888909j  )/8, 9)
        self.assertAlmostEqual(finger(vk[7]), (3.3457845751478734+0.040873646630991085j)/8, 9)

        numpy.random.seed(1)
        dm = numpy.random.random((8,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        vk = xdf_jk.get_k_kpts(mydf, dm, 1, mydf, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (8.7466094134745447-0.11794451574393096j) /8, 9)
        self.assertAlmostEqual(finger(vk[1]), (5.7628099655244203+0.069496695923544183j)/8, 9)
        self.assertAlmostEqual(finger(vk[2]), (7.1835369226567716-0.088737617515686484j)/8, 9)
        self.assertAlmostEqual(finger(vk[3]), (10.078436185541014+0.12781764583216956j) /8, 9)
        self.assertAlmostEqual(finger(vk[4]), (8.3880632334563821-0.099410252805433641j)/8, 9)
        self.assertAlmostEqual(finger(vk[5]), (7.9363214754495335+0.10156364000105772j) /8, 9)
        self.assertAlmostEqual(finger(vk[6]), (7.36928967735646-0.096308680253493123j)  /8, 9)
        self.assertAlmostEqual(finger(vk[7]), (6.8092429879974627+0.08071095923767789j) /8, 9)


if __name__ == '__main__':
    print("Full Tests for xdf_jk")
    unittest.main()


