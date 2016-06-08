import unittest
import numpy
from pyscf import gto, scf
#from pyscf.df import poisson_jk_o2 as poisson_jk
from pyscf.df import xdf
from pyscf.df import xdf_jk

mol = gto.M(
    atom = '''C    3.    2.       3.
              C    1.    1.       1.''',
    basis = 'sto3g'
)


class KnowValues(unittest.TestCase):
    def test_j(self):
        numpy.random.seed(12)
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        mf = xdf_jk.density_fit(scf.RHF(mol), auxbasis='weigend', gs=(10,)*3)
        vj1 = mf.get_j(mol, dm)
        self.assertAlmostEqual(numpy.einsum('ij,ij->', vj1, dm), 50.470715659848707, 9)

    def test_jk(self):
        numpy.random.seed(12)
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        #mf = poisson_jk.with_poisson_(scf.RHF(mol), auxbasis='weigend', gs=(10,)*3)
        mf = xdf_jk.density_fit(scf.RHF(mol), auxbasis='weigend', gs=(10,)*3)
        vj1, vk1 = mf.get_jk(mol, dm)
        self.assertAlmostEqual(numpy.einsum('ij,ij->', vj1, dm), 50.470715659848707, 9)
        self.assertAlmostEqual(numpy.einsum('ij,ij->', vk1, dm), 66.448255856461131, 9)
        vk1 = mf.get_k(mol, dm, hermi=0)
        self.assertAlmostEqual(numpy.einsum('ij,ij->', vk1, dm), 66.448255856461131, 9)

    def test_jk_hermi0(self):
        numpy.random.seed(12)
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        dm[:2,-3:] *= .5
        vj0, vk0 = scf.RHF(mol).get_jk(mol, dm, hermi=0)
        ej0 = numpy.einsum('ij,ij->', vj0, dm)
        ek0 = numpy.einsum('ij,ij->', vk0, dm)
        mf = xdf_jk.density_fit(scf.RHF(mol), auxbasis='weigend', gs=(10,)*3)
        vj1, vk1 = mf.get_jk(mol, dm, hermi=0)
        self.assertAlmostEqual(numpy.einsum('ij,ij->', vj1, dm)-ej0, 0.00010173096167420681, 9)
        self.assertAlmostEqual(numpy.einsum('ij,ij->', vk1, dm)-ek0, -6.4503261583581661e-5, 9)

    def test_jk_metric(self):
        numpy.random.seed(12)
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        vj0, vk0 = scf.RHF(mol).get_jk(mol, dm)
        ej0 = numpy.einsum('ij,ij->', vj0, dm)
        ek0 = numpy.einsum('ij,ij->', vk0, dm)
        jkdf = xdf.XDF(mol)
        jkdf.metric = 'S'
        jkdf.auxbasis = 'weigend'
        jkdf.gs = (10,)*3
        mf = xdf_jk.density_fit(scf.RHF(mol), with_xdf=jkdf)
        vj1, vk1 = mf.get_jk(mol, dm)
        self.assertAlmostEqual(numpy.einsum('ij,ij->', vj1, dm)-ej0, 9.5034429456575253e-5, 9)
        self.assertAlmostEqual(numpy.einsum('ij,ij->', vk1, dm)-ek0, -4.759826947520196e-5, 9)

        jkdf = xdf.XDF(mol)
        jkdf.metric = 'T'
        jkdf.auxbasis = 'weigend'
        jkdf.gs = (10,)*3
        mf = xdf_jk.density_fit(scf.RHF(mol), with_xdf=jkdf)
        vj1, vk1 = mf.get_jk(mol, dm)
        self.assertAlmostEqual(numpy.einsum('ij,ij->', vj1, dm)-ej0, 0.005304754655668375, 9)
        self.assertAlmostEqual(numpy.einsum('ij,ij->', vk1, dm)-ek0, 0.076335586801803856, 9)

        jkdf = xdf.XDF(mol)
        jkdf.metric = 'J'
        jkdf.auxbasis = 'weigend'
        jkdf.gs = (10,)*3
        mf = xdf_jk.density_fit(scf.RHF(mol), with_xdf=jkdf)
        vj1, vk1 = mf.get_jk(mol, dm)
        self.assertAlmostEqual(numpy.einsum('ij,ij->', vj1, dm)-ej0, -6.56455759013852e-5, 9)
        self.assertAlmostEqual(numpy.einsum('ij,ij->', vk1, dm)-ek0, -4.965601597461955e-5, 9)

    def test_jk_charge_constraint(self):
        numpy.random.seed(12)
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        vj0, vk0 = scf.RHF(mol).get_jk(mol, dm)
        ej0 = numpy.einsum('ij,ij->', vj0, dm)
        ek0 = numpy.einsum('ij,ij->', vk0, dm)
        jkdf = xdf.XDF(mol)
        jkdf.charge_constraint = False
        jkdf.auxbasis = 'weigend'
        jkdf.gs = (10,)*3
        mf = xdf_jk.density_fit(scf.RHF(mol), with_xdf=jkdf)
        vj1, vk1 = mf.get_jk(mol, dm)
        self.assertAlmostEqual(numpy.einsum('ij,ij->', vj1, dm)-ej0, 0.0003166388141693232, 9)
        self.assertAlmostEqual(numpy.einsum('ij,ij->', vk1, dm)-ek0, 0.0005799101835037845, 9)

        jkdf = xdf.XDF(mol)
        jkdf.charge_constraint = False
        jkdf.metric = 'T'
        jkdf.auxbasis = 'weigend'
        jkdf.gs = (10,)*3
        mf = xdf_jk.density_fit(scf.RHF(mol), with_xdf=jkdf)
        vj1, vk1 = mf.get_jk(mol, dm)
        self.assertAlmostEqual(numpy.einsum('ij,ij->', vj1, dm)-ej0, 0.12360821384800857, 9)
        self.assertAlmostEqual(numpy.einsum('ij,ij->', vk1, dm)-ek0, 0.4701984522179572, 9)

        jkdf = xdf.XDF(mol)
        jkdf.charge_constraint = False
        jkdf.metric = 'J'
        jkdf.auxbasis = 'weigend'
        jkdf.gs = (10,)*3
        mf = xdf_jk.density_fit(scf.RHF(mol), with_xdf=jkdf)
        vj1, vk1 = mf.get_jk(mol, dm)
        self.assertAlmostEqual(numpy.einsum('ij,ij->', vj1, dm)-ej0, -6.3466650786381251e-5, 9)
        self.assertAlmostEqual(numpy.einsum('ij,ij->', vk1, dm)-ek0, -4.5425207332527862e-5, 9)



if __name__ == '__main__':
    print('Full Tests for xdf_jk')
    unittest.main()
