#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
from pyscf.pbc import gto as pgto
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint


def get_ovlp(cell, grids=None):
    if grids is None:
        grids = gen_grid.BeckeGrids(cell)
        grids.level = 3
        grids.build()

    aoR = numint.eval_ao(cell, grids.coords)
    s = numpy.dot(aoR.T.conj(), grids.weights.reshape(-1,1)*aoR).real
    return s


class KnowValues(unittest.TestCase):
    def test_becke_grids(self):
        L = 4.
        n = 30
        cell = pgto.Cell()
        cell.h = numpy.eye(3)*L
        cell.h[0,1] = cell.h[1,2] = L / 2
        cell.gs = numpy.array([n,n,n])

        cell.atom =[['He' , ( L/2+0., L/2+0. ,   L/2+1.)],
                    ['He' , ( L/2+1., L/2+0. ,   L/2+1.)]]
        cell.basis = {'He': [[0, (1.0, 1.0)]]}
        cell.build()
        grids = gen_grid.BeckeGrids(cell)
        grids.level = 3
        grids.build()
        s1 = get_ovlp(cell, grids)
        s2 = cell.pbc_intor('cint1e_ovlp_sph')
        self.assertAlmostEqual(numpy.linalg.norm(s1-s2), 0, 5)
        self.assertEqual(grids.weights.size, 14829)


if __name__ == '__main__':
    print("Full Tests for Becke grids")
    unittest.main()


