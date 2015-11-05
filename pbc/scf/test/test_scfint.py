import unittest
import numpy as np
from pyscf.pbc.scf import scfint
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.scf import hf as pbchf


def make_cell1(L, n):
    pseudo = None
    cell = pbcgto.Cell()
    cell.output = '/dev/null'
    cell.verbose = 5
    cell.unit = 'B'
    cell.h = ((L,0,0),(0,L,0),(0,0,L))
    cell.gs = [n,n,n]

    cell.atom = [['He', (L/2.,L/2.,L/2.)], ]
    cell.basis = { 'He': [[0, (0.8, 1.0)],
                         [0, (1.0, 1.0)],
                         [0, (1.2, 1.0)]] }
    cell.pseudo = None
    cell.build()
    return cell

def make_cell2(L, n):
    cell = pbcgto.Cell()
    cell.build(unit = 'B',
               output = '/dev/null',
               verbose = 5,
               h = ((L,0,0),(0,L,0),(0,0,L)),
               gs = [n,n,n],
               atom = [['He', (L/2.-.5,L/2.,L/2.-.5)],
                       ['He', (L/2.   ,L/2.,L/2.+.5)]],
               basis = { 'He': [[0, (0.8, 1.0)],
                                [0, (1.0, 1.0)],
                                [0, (1.2, 1.0)]] })
    return cell

def finger(mat):
    w = np.cos(np.arange(mat.size))
    return np.dot(mat.ravel(), w)

class KnowValues(unittest.TestCase):
    def test_olvp(self):
        cell = make_cell1(4, 20)
        cell.nimgs = [2,2,2]
        s1 = scfint.get_ovlp(cell)
        self.assertAlmostEqual(finger(s1), 1.3229918679678208, 10)



if __name__ == '__main__':
    print("Full Tests for pbc.scf.hf.scfint")
    unittest.main()

