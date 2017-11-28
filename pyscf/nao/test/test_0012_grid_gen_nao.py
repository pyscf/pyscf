from __future__ import print_function, division
import unittest
from pyscf.nao import nao 
from pyscf.nao.m_gauleg import leggauss_ab 
from pyscf import dft
import numpy as np

class KnowValues(unittest.TestCase):

  def test_bilocal(self):
    """ Build 3d integration scheme for two centers."""
    sv=nao(xyz_list=[ [8, [0.0, 0.0, 0.0]], [1, [1.0, 1.0, 1.0] ]])
    atom2rcut=np.array([5.0, 4.0])
    grids = dft.gen_grid.Grids(sv)
    grids.level = 2 # precision as implemented in pyscf
    grids.radi_method=leggauss_ab
    grids.build(atom2rcut=atom2rcut)
    self.assertEqual(len(grids.weights), 20648)

  def test_one_center(self):
    """ Build 3d integration coordinates and weights for just one center. """
    sv=nao(xyz_list=[ [8, [0.0, 0.0, 0.0]]])
    atom2rcut=np.array([5.0])
    g = dft.gen_grid.Grids(sv)
    g.level = 1 # precision as implemented in pyscf
    g.radi_method=leggauss_ab
    g.build(atom2rcut=atom2rcut)

    #print(  max(  np.linalg.norm(g.coords, axis=1)  )  )
    #print(  g.weights.sum(), 4.0 *np.pi*5.0**3 / 3.0 )
    self.assertAlmostEqual(max(  np.linalg.norm(g.coords, axis=1)  ), 4.9955942742763986)
    self.assertAlmostEqual(g.weights.sum(), 4.0 *np.pi*5.0**3 / 3.0)
    self.assertEqual(len(g.weights), 6248)
    

if __name__ == "__main__": unittest.main()
