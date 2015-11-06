import unittest
import numpy
from pyscf.pbc import gto
from pyscf.pbc import dft as pdft
import pyscf.dft

cell = gto.Cell()
cell.h = numpy.eye(3) * 6
cell.gs = [20,20,20]
cell.unit = 'B'
cell.atom = '''He     2.    2.       3.
               He     3.    2.       3.'''
cell.basis = {'He': 'ccpvdz'}
cell.verbose = 5
cell.output = '/dev/null'
cell.build()
cell.nimgs = [2,2,2]

def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(a.ravel(), w)

class KnowValues(unittest.TestCase):
    def test_chkfile_gamma_point(self):
        cell1 = gto.Cell()
        cell1.h = numpy.eye(3) * 6
        cell1.gs = [10,10,10]
        cell1.unit = 'B'
        cell1.atom = '''He     2.    2.       3.
                       He     3.    2.       3.'''
        cell1.basis = {'He': 'sto3g'}
        cell1.verbose = 0
        cell1.build()
        mf1 = pdft.RKS(cell1)
        mf1.kernel()

        mf = pdft.RKS(cell)
        dm = mf.from_chk(mf1.chkfile)
        self.assertAlmostEqual(mf.scf(dm), -4.6500700885377455, 6)


if __name__ == '__main__':
    print("Full Tests for he2")
    unittest.main()


