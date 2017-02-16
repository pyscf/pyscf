import unittest
import numpy
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import tools
from pyscf.pbc.scf import khf
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
from pyscf import lib

from ase.build import bulk

def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(w, a.ravel())


class KnowValues(unittest.TestCase):
    def test_coulG_ws(self):
        ase_atom = bulk('C', 'diamond', a=3.5668)
        cell = pbcgto.Cell()
        cell.unit = 'A'
        cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
        cell.a = ase_atom.cell
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.gs = [5]*3
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()
        mf = khf.KRHF(cell, exxdiv='vcut_ws')
        mf.kpts = cell.make_kpts([2,2,2])
        coulG = tools.get_coulG(cell, mf.kpts[2], True, mf)
        self.assertAlmostEqual(finger(coulG), 1.3245117871351604, 9)

    def test_coulG(self):
        numpy.random.seed(19)
        kpt = numpy.random.random(3)
        ase_atom = bulk('C', 'diamond', a=3.5668)
        cell = pbcgto.Cell()
        cell.unit = 'A'
        cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
        cell.a = ase_atom.cell + numpy.random.random((3,3)).T
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.gs = [5,4,3]
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()
        coulG = tools.get_coulG(cell, kpt)
        self.assertAlmostEqual(finger(coulG), 62.75448804333378, 9)

        cell.a = numpy.eye(3)
        cell.unit = 'B'
        coulG = tools.get_coulG(cell, numpy.array([0, numpy.pi, 0]))
        self.assertAlmostEqual(finger(coulG), 4.6737453679713905, 9)
        coulG = tools.get_coulG(cell, numpy.array([0, numpy.pi, 0]),
                                wrap_around=False)
        self.assertAlmostEqual(finger(coulG), 4.5757877990664744, 9)
        coulG = tools.get_coulG(cell, exx='ewald')
        self.assertAlmostEqual(finger(coulG), 4.888843468914021, 9)

    #def test_coulG_2d(self):
    #    cell = pbcgto.Cell()
    #    cell.a = numpy.eye(3)
    #    cell.a[2] = numpy.array((0, 0, 20))
    #    cell.atom = 'He 0 0 0'
    #    cell.unit = 'B'
    #    cell.gs = [4,4,20]
    #    cell.verbose = 5
    #    cell.dimension = 2
    #    cell.output = '/dev/null'
    #    cell.build()
    #    coulG = tools.get_coulG(cell)
    #    self.assertAlmostEqual(finger(coulG), -4.7118365257800496, 9)


    def test_get_lattice_Ls(self):
        numpy.random.seed(2)
        cl1 = pbcgto.M(a = numpy.random.random((3,3))*3,
                       gs = [1]*3,
                       atom ='''He .1 .0 .0''',
                       basis = 'ccpvdz')
        Ls = tools.get_lattice_Ls(cl1)
        self.assertEqual(Ls.shape, (609,3))

    def test_super_cell(self):
        numpy.random.seed(2)
        cl1 = pbcgto.M(a = numpy.random.random((3,3))*3,
                       gs = [1]*3,
                       atom ='''He .1 .0 .0''',
                       basis = 'ccpvdz')
        cl2 = tools.super_cell(cl1, [2,3,4])
        self.assertAlmostEqual(finger(cl2.atom_coords()), -18.946080642714836, 9)

    def test_cell_plus_imgs(self):
        numpy.random.seed(2)
        cl1 = pbcgto.M(a = numpy.random.random((3,3))*3,
                       gs = [1]*3,
                       atom ='''He .1 .0 .0''',
                       basis = 'ccpvdz')
        cl2 = tools.cell_plus_imgs(cl1, cl1.nimgs)
        self.assertAlmostEqual(finger(cl2.atom_coords()), -75.354530112453105, 9)


if __name__ == '__main__':
    print("Full Tests for pbc.tools")
    unittest.main()

