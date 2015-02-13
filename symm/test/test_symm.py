#
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
        atoms = [['H', (0., 0., 0.)],
                 ['H', (1., 0., 0.)],
                 ['H', (0., 0., 1.)],
                 ['H', (-1, 0., 0.)],
                 ['H', (0.,-1., 0.)],
                 ['H', (0., 0.,-1.)]]
        ops = symm.SymmSys(atoms)
        self.assertEqual(ops.detect_icenter(), False)
        v = numpy.linalg.norm(ops.detect_C2() - numpy.array((0.,1.,0.)))
        self.assertAlmostEqual(v, 0., 14)
        v = ops.detect_mirror()
        self.assertAlmostEqual(numpy.dot(v[0],v[1]), 0., 14)
        self.assertAlmostEqual(numpy.dot(v[2],v[3]), 0., 14)
        l, orig, axes = symm.detect_symm(atoms)
        atoms = symm.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'C2v')
        self.assertTrue(symm.check_given_symm('C2v', atoms))

    def test_detect_symm_d2h(self):
        atoms = [['H', (0., 0., 0.)],
                 ['H', (1., 0., 0.)],
                 ['H', (0., 1., 0.)],
                 ['H', (0., 0., 1.)],
                 ['H', (-1, 0., 0.)],
                 ['H', (0.,-1., 0.)],
                 ['H', (0., 0.,-1.)]]
        l, orig, axes = symm.detect_symm(atoms)
        atoms = symm.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'D2h')
        self.assertEqual(symm.symm_identical_atoms(l,atoms), \
                         [[1, 4], [2, 5], [3, 6], [0]])
        self.assertTrue(symm.check_given_symm('D2h', atoms))

    def test_detect_symm_c2v(self):
        atoms = [['H' , (1., 0., 2.)],
                 ['He', (0., 1., 0.)],
                 ['H' , (-2.,0.,-1.)],
                 ['He', (0.,-1., 0.)]]
        l, orig, axes = symm.detect_symm(atoms)
        atoms = symm.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'C2v')
        self.assertEqual(symm.symm_identical_atoms(l,atoms), [[0,2],[1,3]])
        self.assertTrue(symm.check_given_symm('C2v', atoms))

    def test_detect_symm_d2h_a(self):
        atoms = [['He', (0., 1., 0.)],
                 ['H' , (1., 0., 0.)],
                 ['H' , (-1.,0., 0.)],
                 ['He', (0.,-1., 0.)]]
        l, orig, axes = symm.detect_symm(atoms)
        atoms = symm.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'D2h')
        self.assertEqual(symm.symm_identical_atoms(l,atoms),
                         [[1, 2], [0, 3]])
        self.assertTrue(symm.check_given_symm('D2h', atoms))

    def test_detect_symm_d2h_b(self):
        atoms = [['H' , (1., 0., 2.)],
                 ['He', (0., 1., 0.)],
                 ['H' , (-1.,0.,-2.)],
                 ['He', (0.,-1., 0.)]]
        l, orig, axes = symm.detect_symm(atoms)
        atoms = symm.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'D2h')
        self.assertEqual(symm.symm_identical_atoms(l,atoms), [[0,2],[1,3]])
        self.assertTrue(symm.check_given_symm('D2h', atoms))

    def test_detect_symm_c2h_a(self):
        atoms = [['H' , (1., 0., 2.)],
                 ['He', (0., 1.,-1.)],
                 ['H' , (-1.,0.,-2.)],
                 ['He', (0.,-1., 1.)]]
        l, orig, axes = symm.detect_symm(atoms)
        atoms = symm.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'C2h')
        self.assertEqual(symm.symm_identical_atoms(l,atoms), [[0,2],[1,3]])
        self.assertTrue(symm.check_given_symm('C2h', atoms))

    def test_detect_symm_c2h(self):
        atoms = [['H' , (1., 0., 2.)],
                 ['He', (0., 1., 0.)],
                 ['H' , (1., 0., 0.)],
                 ['H' , (-1.,0., 0.)],
                 ['H' , (-1.,0.,-2.)],
                 ['He', (0.,-1., 0.)]]
        l, orig, axes = symm.detect_symm(atoms)
        atoms = symm.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'C2h')
        self.assertEqual(symm.symm_identical_atoms(l,atoms), [[2,3],[0,4],[1,5]])
        self.assertTrue(symm.check_given_symm('C2h', atoms))

        atoms = [['H' , (1., 0., 1.)],
                 ['H' , (1., 0.,-1.)],
                 ['He', (0., 0., 2.)],
                 ['He', (2., 0.,-2.)],
                 ['Li', (1., 1., 0.)],
                 ['Li', (1.,-1., 0.)]]
        l, orig, axes = symm.detect_symm(atoms)
        atoms = symm.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'C2h')
        self.assertEqual(symm.symm_identical_atoms(l,atoms),
                         [[2, 3], [4, 5], [0, 1]])
        self.assertTrue(symm.check_given_symm('C2h', atoms))

    def test_detect_symm_d2(self):
        atoms = [['H' , (1., 0., 1.)],
                 ['H' , (1., 0.,-1.)],
                 ['He', (0., 0., 2.)],
                 ['He', (2., 0., 2.)],
                 ['He', (1., 1.,-2.)],
                 ['He', (1.,-1.,-2.)]]
        l, orig, axes = symm.detect_symm(atoms)
        atoms = symm.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'D2')
        self.assertEqual(symm.symm_identical_atoms(l,atoms),
                         [[2, 3, 4, 5], [0, 1]])
        self.assertTrue(symm.check_given_symm('D2', atoms))

    def test_detect_symm_ci(self):
        atoms = [['H' , ( 1., 0., 0.)],
                 ['He', ( 0., 1., 0.)],
                 ['Li', ( 0., 0., 1.)],
                 ['Be', ( .5, .5, .5)],
                 ['H' , (-1., 0., 0.)],
                 ['He', ( 0.,-1., 0.)],
                 ['Li', ( 0., 0.,-1.)],
                 ['Be', (-.5,-.5,-.5)]]
        l, orig, axes = symm.detect_symm(atoms)
        atoms = symm.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'Ci')
        self.assertEqual(symm.symm_identical_atoms(l,atoms),
                         [[0, 4], [3, 7], [1, 5], [2, 6]])
        self.assertTrue(symm.check_given_symm('Ci', atoms))

    def test_detect_symm_cs(self):
        atoms = [['H' , (1., 0., 2.)],
                 ['He', (1., 0., 0.)],
                 ['Li', (2., 0.,-1.)],
                 ['Be', (0., 0., 1.)]]
        l, orig, axes = symm.detect_symm(atoms)
        atoms = symm.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'Cs')
        self.assertEqual(symm.symm_identical_atoms(l,atoms),
                         [[3], [0], [1], [2]])
        self.assertTrue(symm.check_given_symm('Cs', atoms))

    def test_detect_symm_c1(self):
        atoms = [['H' , ( 1., 0., 0.)],
                 ['He', ( 0., 1., 0.)],
                 ['Li', ( 0., 0., 1.)],
                 ['Be', ( .5, .5, .5)]]
        l, orig, axes = symm.detect_symm(atoms)
        atoms = symm.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'C1')
        self.assertEqual(symm.symm_identical_atoms(l,atoms),
                         [[2], [1], [3], [0]])
        self.assertTrue(symm.check_given_symm('C1', atoms))

    def test_detect_symm_line(self):
        atoms = [['H' , ( 0., 0., 0.)],
                 ['He', ( 0., 2., 0.)], ]
        l = symm.detect_symm(atoms)[0]
        self.assertEqual(l, 'C2v')
        atoms = [['H' , ( 0., 0., 0.)],
                 ['He', ( 0., 2., 0.)],
                 ['He', ( 0., 4., 0.)],
                 ['H' , ( 0., 6., 0.)], ]
        l, orig, axes = symm.detect_symm(atoms)
        atoms = symm.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'D2h')
        self.assertEqual(symm.symm_identical_atoms(l,atoms), [[0,3],[1,2]])
        self.assertTrue(symm.check_given_symm('D2h', atoms))

    def test_detect_symm_c2(self):
        atoms = [['H' , ( 1., 0., 1.)],
                 ['H' , ( 1., 0.,-1.)],
                 ['He', ( 0.,-3., 2.)],
                 ['He', ( 0., 3.,-2.)]]
        l, orig, axes = symm.detect_symm(atoms)
        atoms = symm.shift_atom(atoms, orig, axes)
        self.assertEqual(l, 'C2')
        self.assertEqual(symm.symm_identical_atoms(l,atoms), [[2,3], [0,1]])
        self.assertTrue(symm.check_given_symm('C2', atoms))

if __name__ == "__main__":
    print("Full Tests geom")
    unittest.main()
