import unittest
import numpy as np

from pyscf import gto
from pyscf.dft import rks

from pyscf.pbc import gto as pbcgto
from pyscf.pbc.dft import rks as pbcrks


mol = gto.Mole()
mol.unit = 'B'
L = 60
mol.atom.extend([['He', (L/2.,L/2.,L/2.)], ])
# these are some exponents which are not hard to integrate
mol.basis = { 'He': [[0, (0.8, 1.0)],
                     [0, (1.0, 1.0)],
                     [0, (1.2, 1.0)]] }
mol.verbose = 5
mol.output = '/dev/null'
mol.build()

m = rks.RKS(mol)
m.xc = 'LDA,VWN_RPA'
e0 =  (m.scf()) # -2.64096172441

def make_cell(n):
    pseudo = None
    cell = pbcgto.Cell()
    cell.unit = 'B'
    cell.h = np.diag([L,L,L])
    cell.gs = np.array([n,n,n])
    cell.nimgs = [0,0,0]

    cell.atom = mol.atom
    cell.basis = mol.basis
    cell.pseudo = pseudo
    cell.build()
    return cell

class KnowValues(unittest.TestCase):
#    def test_aux_e2(self):
#        cell = make_cell(30)
#        mf = pbcrks.RKS(cell)
#        mf.xc = 'LDA,VWN_RPA'
#        mf.kpt = np.ones((3,1))
#        e1 = mf.scf()
#        self.assertAlmostEqual(e1, -2.3464949914151378, 8)

    def test_poisson(self):
        pass
#    def test_lda_grid80(self):
#        cell = make_cell(80)
#        mf = pbcrks.RKS(cell)
#        mf.xc = 'LDA,VWN_RPA'
#        mf.kpt = np.ones((3,1))
#        e1 = mf.scf()
#        self.assertAlmostEqual(e1, -2.63907898485, 8)
#
#    def test_lda_grid90(self):
#        cell = make_cell(90)
#        mf = pbcrks.RKS(cell)
#        mf.xc = 'LDA,VWN_RPA'
#        mf.kpt = np.ones((3,1))
#        e1 = mf.scf()
#        self.assertAlmostEqual(e1, -2.64065784113, 8)
#
#    def test_lda_grid100(self):
#        cell = make_cell(100)
#        mf = pbcrks.RKS(cell)
#        mf.xc = 'LDA,VWN_RPA'
#        mf.kpt = np.ones((3,1))
#        e1 = mf.scf()
## python 2.6, numpy 1.6.2, mkl 10.3 got -2.6409238455955433
#        self.assertAlmostEqual(e1, -2.64086844062, 8)


if __name__ == '__main__':
    print("Full Tests for pbc.dft.rks")
    unittest.main()

