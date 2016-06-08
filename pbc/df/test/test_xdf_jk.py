import unittest
import numpy
from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
#from pyscf.pbc.df import poisson_jk
from pyscf.pbc.df import xdf
from pyscf.pbc.df import xdf_jk

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
        #mf = poisson_jk.with_poisson_(mf, auxbasis='weigend')
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



if __name__ == '__main__':
    print("Full Tests for xdf_jk")
    unittest.main()


