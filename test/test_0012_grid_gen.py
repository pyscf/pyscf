from __future__ import print_function, division
import unittest

class KnowValues(unittest.TestCase):

  def test_ao_eval(self):
    from pyscf.nao import system_vars_c 
    from pyscf.nao.m_gauleg import leggauss_ab 
    from pyscf import dft
    
    import numpy as np
    """  """
    sv=system_vars_c().init_xyzlike([ [8, [0.0, 0.0, 0.0]], [1, [1.0, 1.0, 1.0] ]])
    atom2rcut=np.array([5.0, 4.0])
    grids = dft.gen_grid.Grids(sv)
    grids.level = 2 # precision as implemented in pyscf
    grids.radi_method=leggauss_ab
    grids.build(atom2rcut=atom2rcut)
    self.assertEqual(len(grids.weights), 20648)

if __name__ == "__main__":
  unittest.main()
