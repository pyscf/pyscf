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


def make_cell(n):
    pseudo = None
    cell = pbcgto.Cell()
    cell.unit = 'B'
    cell.h = np.diag([L,L,L])
    cell.gs = np.array([n,n,n])
    #cell.nimgs = [0,0,0]

    cell.atom = mol.atom
    cell.basis = mol.basis
    cell.pseudo = pseudo
    cell.build()
    return cell

class KnowValues(unittest.TestCase):
    def test_lda_grid30(self):
        cell = make_cell(30)
        mf = pbcrks.RKS(cell)
        mf.xc = 'LDA,VWN_RPA'
        mf.kpt = np.ones(3)
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -2.3464949914151378, 8)

    """
    def test_lda_grid80(self):
        cell = make_cell(80)
        mf = pbcrks.RKS(cell)
        mf.xc = 'LDA,VWN_RPA'
        mf.kpt = np.ones(3)
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -2.63907898485, 8)

    def test_lda_grid90(self):
        cell = make_cell(90)
        mf = pbcrks.RKS(cell)
        mf.xc = 'LDA,VWN_RPA'
        mf.kpt = np.ones(3)
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -2.64065784113, 8)

    def test_lda_grid100(self):
        cell = make_cell(100)
        mf = pbcrks.RKS(cell)
        mf.xc = 'LDA,VWN_RPA'
        mf.kpt = np.ones(3)
        e1 = mf.scf()
    # python 2.6, numpy 1.6.2, mkl 10.3 got -2.6409238455955433
        self.assertAlmostEqual(e1, -2.64086844062, 8)
    """

    def test_pp_RKS(self):
        cell = pbcgto.Cell()

        cell.unit = 'A'
        cell.atom = '''
            Si    0.000000000    0.000000000    0.000000000;
            Si    0.000000000    2.715348700    2.715348700;
            Si    2.715348700    2.715348700    0.000000000;
            Si    2.715348700    0.000000000    2.715348700;
            Si    4.073023100    1.357674400    4.073023100;
            Si    1.357674400    1.357674400    1.357674400;
            Si    1.357674400    4.073023100    4.073023100;
            Si    4.073023100    4.073023100    1.357674400
        '''
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'

        Lx = Ly = Lz = 5.430697500
        cell.h = np.diag([Lx,Ly,Lz])
        cell.gs = np.array([10,10,10])

        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()

        self.assertTrue(np.all(cell.nimgs == [4, 4, 4]))
        kmf = pbcrks.RKS(cell)
        kmf.xc = 'lda,vwn'
        self.assertAlmostEqual(kmf.scf(), -31.0816167311604, 8)

if __name__ == '__main__':
    print("Full Tests for pbc.dft.rks")
    unittest.main()
