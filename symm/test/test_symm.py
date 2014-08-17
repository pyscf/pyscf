#
# File: test_symm.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
from pyscf import gto
from pyscf import symm

h2o = gto.Mole()
h2o.verbose = 0
h2o.output = None#"out_h2o"
h2o.atom.extend([
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ])

h2o.basis = {"H": '6-31g',
             "O": '6-31g',}
h2o.build()


class KnowValues(unittest.TestCase):
    def test_mass_center(self):
        ct = symm.get_mass_center(h2o.atom)
        self.assertAlmostEqual(ct[0], 0., 9)
        self.assertAlmostEqual(ct[1], 0., 9)
        self.assertAlmostEqual(ct[2], 0.06522222222, 9)

    def test_charge_center(self):
        ct = symm.get_charge_center(h2o.atom)
        self.assertAlmostEqual(ct[0], 0., 9)
        self.assertAlmostEqual(ct[1], 0., 9)
        self.assertAlmostEqual(ct[2], 0.11740000000, 9)

    def test_detect_symm(self):
        atoms = [[1, (0., 0., 0.)],
                 [1, (1., 0., 0.)],
                 [1, (0., 0., 1.)],
                 [1, (-1, 0., 0.)],
                 [1, (0.,-1., 0.)],
                 [1, (0., 0.,-1.)]]
        ops = symm.SymmOperator(atoms)
        self.assertEqual(ops.detect_icenter(), False)
        v = numpy.linalg.norm(ops.detect_C2() - numpy.array((0.,1.,0.)))
        self.assertAlmostEqual(v, 0., 14)
        v = ops.detect_mirror()
        self.assertAlmostEqual(numpy.dot(v[0],v[1]), 0., 14)
        self.assertAlmostEqual(numpy.dot(v[2],v[3]), 0., 14)
        l,c = symm.detect_symm(atoms)
        self.assertEqual(l, 'C2v')
        self.assertAlmostEqual(numpy.linalg.norm(c[1][1]-numpy.array((-1.,0.,1./6))), 0., 14)

    def test_detect_symm_d2h(self):
        atoms = [[1, (0., 0., 0.)],
                 [1, (1., 0., 0.)],
                 [1, (0., 1., 0.)],
                 [1, (0., 0., 1.)],
                 [1, (-1, 0., 0.)],
                 [1, (0.,-1., 0.)],
                 [1, (0., 0.,-1.)]]
        l,c = symm.detect_symm(atoms)
        self.assertEqual(l, 'D2h')
        self.assertEqual(symm.symm_identical_atoms(l,c), \
                         [[0],[1,4],[2,5],[3,6]])

    def test_detect_symm_c2v(self):
        atoms = [[1, (1., 0., 2.)],
                 [2, (0., 1., 0.)],
                 [1, (-2.,0.,-1.)],
                 [2, (0.,-1., 0.)]]
        l,c = symm.detect_symm(atoms)
        self.assertEqual(l, 'C2v')
        self.assertEqual(symm.symm_identical_atoms(l,c), [[0,2],[1,3]])

    def test_detect_symm_d2h_a(self):
        atoms = [[1, (1., 0., 2.)],
                 [2, (0., 1., 0.)],
                 [1, (-1.,0.,-2.)],
                 [2, (0.,-1., 0.)]]
        l,c = symm.detect_symm(atoms)
        self.assertEqual(l, 'D2h')
        self.assertEqual(symm.symm_identical_atoms(l,c), [[0,2],[1,3]])

    def test_detect_symm_c2v(self):
        atoms = [[1, (1., 0., 2.)],
                 [2, (0., 1., 0.)],
                 [1, (1., 0., 0.)],
                 [1, (-1.,0., 0.)],
                 [1, (-1.,0.,-2.)],
                 [2, (0.,-1., 0.)]]
        l,c = symm.detect_symm(atoms)
        self.assertEqual(l, 'C2h')
        self.assertEqual(symm.symm_identical_atoms(l,c), [[0,4],[1,5],[2,3]])

        atoms = [[1, (1., 0., 1.)],
                 [1, (1., 0.,-1.)],
                 [2, (0., 0., 2.)],
                 [2, (2., 0.,-2.)],
                 [3, (1., 1., 0.)],
                 [3, (1.,-1., 0.)]]
        l,c = symm.detect_symm(atoms)
        self.assertEqual(l, 'C2h')
        self.assertEqual(symm.symm_identical_atoms(l,c), [[0,1],[2,3],[4,5]])

    def test_detect_symm_d2(self):
        atoms = [[1, (1., 0., 1.)],
                 [1, (1., 0.,-1.)],
                 [2, (0., 0., 2.)],
                 [2, (2., 0., 2.)],
                 [2, (1., 1.,-2.)],
                 [2, (1.,-1.,-2.)]]
        l,c = symm.detect_symm(atoms)
        self.assertEqual(l, 'D2')
        self.assertEqual(symm.symm_identical_atoms(l,c), [[0,1],[2,3,4,5]])

    def test_detect_symm_ci(self):
        atoms = [[1, ( 1., 0., 0.)],
                 [2, ( 0., 1., 0.)],
                 [3, ( 0., 0., 1.)],
                 [4, ( .5, .5, .5)],
                 [1, (-1., 0., 0.)],
                 [2, ( 0.,-1., 0.)],
                 [3, ( 0., 0.,-1.)],
                 [4, (-.5,-.5,-.5)]]
        l,c = symm.detect_symm(atoms)
        self.assertEqual(l, 'Ci')
        self.assertEqual(symm.symm_identical_atoms(l,c), [[i,i+4] for i in range(4)])

    def test_detect_symm_cs(self):
        atoms = [[1, (1., 0., 2.)],
                 [2, (1., 0., 0.)],
                 [3, (2., 0.,-1.)],
                 [4, (0., 0., 1.)]]
        l,c = symm.detect_symm(atoms)
        self.assertEqual(l, 'Cs')
        self.assertEqual(symm.symm_identical_atoms(l,c), [[0],[1],[2],[3]])

    def test_detect_symm_c1(self):
        atoms = [[1, ( 1., 0., 0.)],
                 [2, ( 0., 1., 0.)],
                 [3, ( 0., 0., 1.)],
                 [4, ( .5, .5, .5)]]
        l,c = symm.detect_symm(atoms)
        self.assertEqual(l, 'C1')
        self.assertEqual(symm.symm_identical_atoms(l,c), [(i,) for i in range(4)])


if __name__ == "__main__":
    print "Full Tests geom"
    unittest.main()
