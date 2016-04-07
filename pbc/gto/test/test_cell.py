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

numpy.random.seed(1)
cl1 = pgto.Cell()
cl1.build(h = numpy.random.random((3,3)),
          gs = [n,n,n],
          atom ='''He .1 .0 .0
                   He .5 .1 .0
                   He .0 .5 .0
                   He .1 .3 .2''',
          basis = 'ccpvdz')

def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(a.ravel(), w)

class KnowValues(unittest.TestCase):
    def test_nimgs(self):
        self.assertTrue(numpy.all(cl.get_nimgs(9e-1)==[3,3,3]))
        self.assertTrue(numpy.all(cl.get_nimgs(1e-2)==[4,4,4]))
        self.assertTrue(numpy.all(cl.get_nimgs(1e-4)==[4,4,4]))
        self.assertTrue(numpy.all(cl.get_nimgs(1e-6)==[4,4,4]))
        self.assertTrue(numpy.all(cl.get_nimgs(1e-9)==[5,5,5]))

    def test_Gv(self):
        a = cl1.get_Gv()
        self.assertAlmostEqual(finger(a), -99.791927068519939, 10)

    def test_SI(self):
        a = cl1.get_SI()
        self.assertAlmostEqual(finger(a), (16.506917823339265+1.6393578329869585j), 10)

    def test_bounding_sphere(self):
        self.assertTrue(numpy.all(cl1.get_bounding_sphere(4.5)==[12,8,7]))

    def test_mixed_basis(self):
        cl = pgto.Cell()
        cl.build(
            h = [[L,0,0], [0,L,0], [0,0,L]],
            gs = [n,n,n],
            atom = 'C1 %f %f %f; C2 %f %f %f' % ((L/2.,)*6),
            basis = {'C1':'ccpvdz', 'C2':'gthdzv'})

    def test_dumps_loads(self):
        cl1.loads(cl1.dumps())


if __name__ == '__main__':
    print("Full Tests for pbc.gto.cell")
    unittest.main()

