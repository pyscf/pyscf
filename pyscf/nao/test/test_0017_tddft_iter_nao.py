from __future__ import print_function, division
import os,unittest
from pyscf.nao import scf, prod_basis_c, tddft_iter_c
from numpy import allclose, float32, einsum

dname = os.path.dirname(os.path.abspath(__file__))
sv = scf(label='water', cd=dname)
pb = sv.pb

class KnowValues(unittest.TestCase):
  
  def test_tddft_iter(self):
    """ This is iterative TDDFT with SIESTA starting point """
    td = tddft_iter_c(pb.sv, pb)
    self.assertTrue(hasattr(td, 'xocc'))
    self.assertTrue(hasattr(td, 'xvrt'))
    self.assertTrue(td.ksn2f.sum()==8.0) # water: O -- 6 electrons in the valence + H2 -- 2 electrons
    self.assertEqual(td.xocc.shape[0], 4)
    self.assertEqual(td.xvrt.shape[0], 19)
    dn0 = td.apply_rf0(td.moms1[:,0])
    

if __name__ == "__main__":
  unittest.main()
