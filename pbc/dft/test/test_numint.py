import unittest
import numpy as np

from pyscf import gto
from pyscf.dft import rks

from pyscf.pbc import gto as pbcgto
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint


def make_grids(n):
    L = 60
    cell = pbcgto.Cell()
    cell.verbose = 0
    cell.output = '/dev/null'
    cell.unit = 'B'
    cell.h = ((L,0,0),(0,L,0),(0,0,L))
    cell.gs = [n,n,n]
    cell.nimgs = [0,0,0]

    cell.atom = [['He', (L/2.,L/2.,L/2.)], ]
    cell.basis = {'He': [[0, (0.8, 1.0)],
                         [0, (1.0, 1.0)],
                         [0, (1.2, 1.0)]] }
    cell.pseudo = None
    cell.build(False, False)
    grids = gen_grid.UniformGrids(cell)
    grids.setup_grids_()
    return cell, grids

class KnowValues(unittest.TestCase):
    def test_eval_ao(self):
        cell, grids = make_grids(30)
        ao1 = numint.eval_ao(cell, grids.coords)
        w = np.arange(ao1.size) * .01
        self.assertAlmostEqual(np.dot(w,ao1.ravel()), (44072.276638371265+0j), 8)

#    def test_eval_ao_gga(self):
#        cell, grids = make_grids(30)
#        ao1 = numint.eval_ao(cell, grids.coords, isgga=True)
#        w = np.arange(ao1.size)
#        self.assertAlmostEqual(np.dot(w,ao1), 0, 8)
#
#    def test_eval_rho(self):
#        self.assertAlmostEqual(e1, 0, 8)
#
    def test_eval_mat(self):
        cell, grids = make_grids(30)
        ng = grids.weights.size
        np.random.seed(1)
        rho = np.random.random(ng)
        rho *= 1/np.linalg.norm(rho)
        vrho = np.random.random(ng)
        ao1 = numint.eval_ao(cell, grids.coords)
        mat1 = numint.eval_mat(cell, ao1, grids.weights, rho, vrho)
        w = np.arange(mat1.size) * .01
        self.assertAlmostEqual(np.dot(w,mat1.ravel()), (.14777107967912118+0j), 8)


if __name__ == '__main__':
    print("Full Tests for pbc.dft.numint")
    unittest.main()

