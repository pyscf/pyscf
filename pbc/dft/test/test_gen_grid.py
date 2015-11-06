import unittest
import numpy
from pyscf.pbc import gto as pgto
from pyscf.pbc.scf import scfint
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint
import time


def get_ovlp(cell, kpt=None, grids=None):
    if grids is None:
        grids = gen_grid.BeckeGrids(cell)
        grids.level = 2
        grids.build_()

    aoR = numint.eval_ao(cell, grids.coords, kpt)
    s = numpy.dot(aoR.T.conj(), grids.weights.reshape(-1,1)*aoR).real
    return s


class KnowValues(unittest.TestCase):
    def test_becke_grids(self):
        L = 4.
        n = 30
        cell = pgto.Cell()
        cell.h = numpy.diag([L,L,L])
        cell.gs = numpy.array([n,n,n])

        cell.atom =[['He' , ( L/2+0., L/2+0. ,   L/2+1.)],
                    ['He' , ( L/2+1., L/2+0. ,   L/2+1.)]]
        cell.basis = {'He': [[0, (1.0, 1.0)]]}
        cell.build()
        kpt = None
        s1 = get_ovlp(cell, kpt)
        s2 = scfint.get_ovlp(cell, kpt)
        self.assertAlmostEqual(numpy.linalg.norm(s1-s2), 0, 5)


if __name__ == '__main__':
    print("Full Tests for Becke grids")
    unittest.main()


