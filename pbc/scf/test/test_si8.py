import unittest
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft

class KnowValues(unittest.TestCase):
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
        kmf = pbcdft.RKS(cell)
        kmf.xc = 'lda,vwn'
        self.assertAlmostEqual(kmf.scf(), -31.0816167311604, 8)


if __name__ == '__main__':
    print("Tests for Si8")
    unittest.main()
