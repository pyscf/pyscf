import unittest
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft

# may have bug in PP integral

class KnowValues(unittest.TestCase):
    #def test_RKS(self):
    #    pass

    def test_pp_RKS(self):
        cell = pbcgto.Cell()

        cell.unit = 'A'
        cell.atom = '''
          Si    0.000000000    2.715348700    2.715348700;
          Si    1.357674400    1.357674400    1.357674400;
        '''
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'

        Lx = Ly = Lz = 5.430697500
        cell.h = np.diag([Lx,Ly,Lz])
        cell.gs = np.array([8,8,8])

        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()

        self.assertTrue(np.all(cell.nimgs == [4, 4, 4]))
        kmf = pbcdft.RKS(cell)
        kmf.xc = 'b88, lyp'
        self.assertAlmostEqual(kmf.scf(), -10.2539335672, 8)


if __name__ == '__main__':
    print("Tests for Si2")
    unittest.main()
