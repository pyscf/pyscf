import unittest
import numpy
from pyscf.pbc import gto as pgto


L = 1.5
n = 20
cl = pgto.Cell()
cl.build(
    h = [[L,0,0], [0,L,0], [0,0,L]],
    gs = [n,n,n],
    atom = 'He %f %f %f' % ((L/2.,)*3),
    basis = 'ccpvdz')

class KnowValues(unittest.TestCase):
    def test_nimgs(self):
        self.assertTrue(numpy.all(cl.get_nimgs(9e-1)==[2,2,2]))
        self.assertTrue(numpy.all(cl.get_nimgs(1e-2)==[4,4,4]))
        self.assertTrue(numpy.all(cl.get_nimgs(1e-4)==[4,4,4]))
        self.assertTrue(numpy.all(cl.get_nimgs(1e-6)==[4,4,4]))
        self.assertTrue(numpy.all(cl.get_nimgs(1e-9)==[5,5,5]))


if __name__ == '__main__':
    print("Full Tests for pbc.gto.cell")
    unittest.main()

