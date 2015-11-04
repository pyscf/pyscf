import unittest
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft

def run_lda(atom,L,n): 
    cell = pbcgto.Cell()

    cell.unit = 'A'
    cell.atom = atom
    cell.h = np.diag([L,L,L])
    cell.gs = np.array([n,n,n])

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'

    #cell.verbose = 7
    cell.build()

    print "Cell nimgs =", cell.nimgs
    print "Cell _basis =", cell._basis
    print "Cell _pseudo =", cell._pseudo
    print "Cell nelectron =", cell.nelectron

    kmf = pbcdft.RKS(cell)
    kmf.xc = 'lda,vwn'

    return kmf.scf()

class KnowValues(unittest.TestCase):
    def test_he(self):
        atom = 'He  0.0  0.0  0.0'
        e1 = run_lda(atom, 2.0, 12)
        self.assertAlmostEqual(e1, -3.0570803321165, 8)

    def test_h2(self):
        atom = '''
            H    0.0000    0.0000    0.0000;
            H    0.7414    0.0000    0.0000;
        '''
        # Slightly greater than 2*(bond length)
        L = 1.5 
        e1 = run_lda(atom, L, 8)
        self.assertAlmostEqual(e1, -1.93426844243467, 8)

    def test_c8(self):
        atom = '''
            C    0.           0.           0.        ;
            C    0.           1.78339997   1.78339997;
            C    1.78339997   1.78339997   0.        ;
            C    1.78339997   0.           1.78339997;
            C    2.67509998   0.89170002   2.67509998;
            C    0.89170002   0.89170002   0.89170002;
            C    0.89170002   2.67509998   2.67509998;
            C    2.67509998   2.67509998   0.89170002
        '''
        L = 3.5668 
        e1 = run_lda(atom, L, 10)
        self.assertAlmostEqual(e1, -44.8811199403019, 8)

if __name__ == '__main__':
    print("Full Tests for pbc.gto.pseudo")
    unittest.main()
