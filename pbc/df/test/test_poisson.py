import unittest
import numpy
import pyscf.gto
from pyscf.pbc import gto
from pyscf.pbc.df import df
from pyscf.pbc.df import poisson
from pyscf.pbc import dft as pdft
import pyscf.dft

cell = gto.Cell()
cell.h = numpy.eye(3) * 4
cell.gs = [20,20,20]
cell.unit = 'B'
cell.atom = '''He     2.    2.       3.
               He     3.    2.       3.'''
cell.basis = {'He': [[0, (1.0, 1.0)]]}
cell.verbose = 5
cell.output = '/dev/null'
cell.build()
mf = pyscf.dft.RKS(cell)
mf.kernel()
dm = mf.make_rdm1()
mf = pdft.RKS(cell)
mf.xc = 'LDA,VWN'
auxbasis = {'He': pyscf.gto.expand_etbs([[0, 11, .1, 2.],
                                         [1, 3 , .3, 3.],
                                         [2, 3 , .3, 3.]])}
auxcell = df.format_aux_basis(cell, auxbasis)
auxcell.nimgs = [3,3,3]

def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(a.ravel(), w)

class KnowValues(unittest.TestCase):
    def test_j_uniform_mod(self):
        vj = poisson.get_j_uniform_mod(cell, dm, auxcell)
        self.assertAlmostEqual(finger(vj), 0.11651805186830155, 6)

    def test_nuc_uniform_mod(self):
        vj = poisson.get_nuc_uniform_mod(cell, auxcell)
        self.assertAlmostEqual(finger(vj), -0.23226970826678728, 6)

    def test_j_gaussian_mod(self):
        modcell = df.format_aux_basis(cell, {'He': [[0, [3, 1]]]})
        vj = poisson.get_j_gaussian_mod(cell, dm, auxcell, modcell)
        self.assertAlmostEqual(finger(vj), -0.040340371628397542, 6)

    def test_nuc_gaussian_mod(self):
        modcell = df.format_aux_basis(cell, {'He': [[0, [3, 1]]]})
        vj = poisson.get_nuc_gaussian_mod(cell, auxcell, modcell)
        self.assertAlmostEqual(finger(vj), -0.076168456359574366, 6)

    def test_jmod_pw_poisson(self):
        modcell = df.format_aux_basis(cell, {'He': [[0, [3, 1]]]})
        vjmod = poisson.get_jmod_pw_poisson(cell, modcell)
        self.assertAlmostEqual(finger(vjmod), 0.15610007121209812, 6)


if __name__ == '__main__':
    print("Full Tests for poisson")
    unittest.main()


