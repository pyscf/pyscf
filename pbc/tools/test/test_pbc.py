import unittest
import numpy
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import tools
from pyscf.pbc.scf import khf
import pyscf.pbc.tools.pyscf_ase as pyscf_ase

from ase.lattice import bulk

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
        self.assertAlmostEqual(finger(coulG), 166.15891996685517, 9)

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



if __name__ == '__main__':
    print("Full Tests for pbc.tools")
    unittest.main()

