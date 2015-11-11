import unittest
import numpy as np

from pyscf.pbc import gto as pgto
import pyscf.pbc.dft as pdft
from pyscf.pbc.df import df


class KnowValues(unittest.TestCase):
    def test_aux_e2_uniform(self):
        cell = pgto.Cell()
        cell.unit = 'B'
        cell.h = np.eye(3) * 3.
        cell.gs = np.array([20,20,20])
        cell.atom = 'He 0 1 1; He 1 1 0'
        cell.basis = { 'He': [[0, (0.8, 1.0)],
                              [0, (1.2, 1.0)]] }
        cell.verbose = 0
        cell.build(0, 0)
        auxcell = df.format_aux_basis(cell)
        auxcell.nimgs = [3,3,3]
        a1 = df.aux_e2(cell, auxcell, 'cint3c1e_sph')
        #grids = pdft.gen_grid.BeckeGrids(cell)
        #grids.level = 3
        grids = pdft.gen_grid.UniformGrids(cell)
        grids.build_()
        a2 = df.aux_e2_grid(cell, auxcell, grids)
        self.assertAlmostEqual(np.linalg.norm(a1-a2), 0, 6)

    def test_aux_e2_becke(self):
        cell = pgto.Cell()
        cell.unit = 'B'
        cell.h = np.eye(3) * 3.
        cell.gs = np.array([20,20,20])
        cell.atom = 'He 0 1 1; He 1 1 0'
        cell.basis = { 'He': [[0, (10.8, 1.0)],
                              [0, (300.2, 1.0)]] }
        cell.verbose = 0
        cell.build(0, 0)
        auxcell = df.format_aux_basis(cell)
        auxcell.nimgs = [3,3,3]
        a1 = df.aux_e2(cell, auxcell, 'cint3c1e_sph')
        grids = pdft.gen_grid.BeckeGrids(cell)
        grids.level = 3
        #grids = pdft.gen_grid.UniformGrids(cell)
        grids.build_()
        a2 = df.aux_e2_grid(cell, auxcell, grids)
        self.assertAlmostEqual(np.linalg.norm(a1-a2), 0, 4)


if __name__ == '__main__':
    print("Full Tests for pbc.df")
    unittest.main()

