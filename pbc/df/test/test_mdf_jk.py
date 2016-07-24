import unittest
import numpy
from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
import pyscf.pbc
from pyscf.pbc.df import mdf
from pyscf.pbc.df import mdf_jk
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
        mf = mdf_jk.density_fit(mf0, auxbasis='weigend', gs=(5,)*3)
        dm = mf.get_init_guess()
        vj, vk = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj, dm)
        ek1 = numpy.einsum('ij,ji->', vk, dm)
        self.assertAlmostEqual(ej1, 46.698923699173982, 9)
        self.assertAlmostEqual(ek1, 31.722934801807845, 9)

        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        vj1, vk1 = mf.get_jk(cell, dm, hermi=0)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 242.17738846073865, 9)
        self.assertAlmostEqual(ek1, 280.27434674577881, 9)

    def test_hcore(self):
        mf = pscf.RHF(cell)
        odf = mdf.MDF(cell)
        odf.auxbasis = 'weigend'
        odf.gs = (10,)*3
        dm = mf.get_init_guess()
        self.assertAlmostEqual(numpy.einsum('ij,ji->', odf.get_nuc(), dm),
                               -150.61096872500249, 9)

    def test_jk_hermi0(self):
        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        dm[:2,-3:] *= .5
        jkdf = mdf.MDF(cell)
        jkdf.auxbasis = 'weigend'
        jkdf.gs = (5,)*3
        mf = mdf_jk.density_fit(mf0, with_df=jkdf)
        vj0, vk0 = mf.get_jk(cell, dm, hermi=0)
        ej0 = numpy.einsum('ij,ji->', vj0, dm)
        ek0 = numpy.einsum('ij,ji->', vk0, dm)
        mf = mdf_jk.density_fit(mf0, with_df=jkdf)
        vj1, vk1 = mf.get_jk(cell, dm, hermi=0)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertTrue(numpy.allclose(vj0, vj1))
        self.assertTrue(numpy.allclose(vk0, vk1))
        self.assertAlmostEqual(ej1, 242.17214791834408, 9)
        self.assertAlmostEqual(ek1, 280.69967132047987, 9)

    def test_jk_metric(self):
        numpy.random.seed(12)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        jkdf = mdf.MDF(cell)
        jkdf.metric = 'S'
        jkdf.auxbasis = 'weigend'
        jkdf.gs = (5,)*3
        mf = mdf_jk.density_fit(mf0, with_df=jkdf)
        vj1, vk1 = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 242.15871646261877, 9)
        self.assertAlmostEqual(ek1, 280.25499678912536, 9)

        jkdf = mdf.MDF(cell)
        jkdf.metric = 'T'
        jkdf.auxbasis = 'weigend'
        jkdf.gs = (5,)*3
        mf = mdf_jk.density_fit(mf0, with_df=jkdf)
        vj1, vk1 = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 242.17738846073865, 9)
        self.assertAlmostEqual(ek1, 280.27434674577881, 9)

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
        self.assertAlmostEqual(ej1, 242.15904061299113, 9)
        self.assertAlmostEqual(ek1, 280.2553652304353, 9)

        jkdf = mdf.MDF(cell)
        jkdf.metric = 'T'
        jkdf.approx_sr_level = 3
        jkdf.auxbasis = 'weigend'
        jkdf.gs = (5,)*3
        mf = mdf_jk.density_fit(mf0, with_df=jkdf)
        vj1, vk1 = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 242.17554856213894, 9)
        self.assertAlmostEqual(ek1, 280.27281092754305, 9)

        jkdf = mdf.MDF(cell)
        jkdf.metric = 'T'
        jkdf.approx_sr_level = 4
        jkdf.auxbasis = 'weigend'
        jkdf.gs = (5,)*3
        mf = mdf_jk.density_fit(mf0, with_df=jkdf)
        vj1, vk1 = mf.get_jk(cell, dm)
        ej1 = numpy.einsum('ij,ji->', vj1, dm)
        ek1 = numpy.einsum('ij,ji->', vk1, dm)
        self.assertAlmostEqual(ej1, 242.17944693058223, 9)
        self.assertAlmostEqual(ek1, 280.27697544586277, 9)

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
        self.assertAlmostEqual(finger(vj[0]), (0.48124673545723651-0.11872368821666528j  ), 9)
        self.assertAlmostEqual(finger(vj[1]), (0.53969554495566907-0.046131953496875822j ), 9)
        self.assertAlmostEqual(finger(vj[2]), (0.52703012199370924-0.083702303894662933j ), 9)
        self.assertAlmostEqual(finger(vj[3]), (0.54249742665727529+0.0088412437015989095j), 9)

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
        self.assertAlmostEqual(finger(vk[0]), (-2.8420747650454392 -1.0532080235680654j  ), 9)
        self.assertAlmostEqual(finger(vk[1]), (-7.4479964653027597 +0.10327182213618791j ), 9)
        self.assertAlmostEqual(finger(vk[2]), (-2.5790215490804433 -1.4459077150581894j  ), 9)
        self.assertAlmostEqual(finger(vk[3]), (-0.79608989192947033+0.012002060547759118j), 9)

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
        mydf = mdf.MDF(cell)
        mydf.kpts = kpts
        mydf.auxbasis = 'weigend'
        mydf.exxdiv = None
        vk = mdf_jk.get_k_kpts(mydf, dm, 0, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (0.54167993564799133-0.0078722013632614562j), 9)
        self.assertAlmostEqual(finger(vk[1]), (0.3593714867663208 +0.0036054974436528706j), 9)
        self.assertAlmostEqual(finger(vk[2]), (0.46234927585203783-0.0065043520189423856j), 9)
        self.assertAlmostEqual(finger(vk[3]), (0.63626519337979948+0.0075118282426430818j), 9)
        self.assertAlmostEqual(finger(vk[4]), (0.53630228285345427-0.0076423773076679524j), 9)
        self.assertAlmostEqual(finger(vk[5]), (0.4956577130753681 +0.0060590034846796439j), 9)
        self.assertAlmostEqual(finger(vk[6]), (0.45380714659933519-0.0068616367194157396j), 9)
        self.assertAlmostEqual(finger(vk[7]), (0.41808297427780505+0.0051096506509061114j), 9)

        numpy.random.seed(1)
        dm = numpy.random.random((8,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        vk = mdf_jk.get_k_kpts(mydf, dm, 1, mydf.kpts)
        self.assertAlmostEqual(finger(vk[0]), (1.0929902110020242 -0.014742288503285355j ), 9)
        self.assertAlmostEqual(finger(vk[1]), (0.72006803096300986+0.0086853082996361259j), 9)
        self.assertAlmostEqual(finger(vk[2]), (0.89763598759314234-0.011090943227218j    ), 9)
        self.assertAlmostEqual(finger(vk[3]), (1.2594746237554506 +0.015976836949610319j ), 9)
        self.assertAlmostEqual(finger(vk[4]), (1.0482108645807622 -0.012426406003793576j ), 9)
        self.assertAlmostEqual(finger(vk[5]), (0.99174830962018801+0.012694552375379626j ), 9)
        self.assertAlmostEqual(finger(vk[6]), (0.9208432538335829 -0.01203638564702492j  ), 9)
        self.assertAlmostEqual(finger(vk[7]), (0.8508739289672812 +0.010089099047683993j ), 9)


if __name__ == '__main__':
    print("Full Tests for mdf_jk")
    unittest.main()

