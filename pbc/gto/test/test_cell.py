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

    def test_ewald(self):
        cell = pgto.Cell()
        cell.unit = 'B'
        Lx = Ly = Lz = 5.
        cell.h = numpy.diag([Lx,Ly,Lz])
        cell.gs = numpy.array([20,20,20])
        cell.atom = [['He', (2, 0.5*Ly, 0.5*Lz)],
                     ['He', (3, 0.5*Ly, 0.5*Lz)]]
        cell.basis = {'He': [[0, (1.0, 1.0)]]}
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()

        ew_cut = (20,20,20)
        self.assertAlmostEqual(cell.ewald(.05, ew_cut), -0.468640671931, 9)
        self.assertAlmostEqual(cell.ewald(0.1, ew_cut), -0.468640671931, 9)
        self.assertAlmostEqual(cell.ewald(0.2, ew_cut), -0.468640671931, 9)
        self.assertAlmostEqual(cell.ewald(1  , ew_cut), -0.468640671931, 9)

        def check(precision, eta_ref, ewald_ref):
            ew_eta0, ew_cut0 = cell.get_ewald_params(precision)
            self.assertAlmostEqual(ew_eta0, eta_ref)
            self.assertAlmostEqual(cell.ewald(ew_eta0, ew_cut0), ewald_ref, 9)
        check(0.001, 4.78124933741, -0.469112631739)
        check(1e-05, 3.70353981157, -0.468642153932)
        check(1e-07, 3.1300624293 , -0.468640678042)
        check(1e-09, 2.76045559201, -0.468640671959)

if __name__ == '__main__':
    print("Full Tests for pbc.gto.cell")
    unittest.main()

