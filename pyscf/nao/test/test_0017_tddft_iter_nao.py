from __future__ import print_function, division
import os,unittest
from pyscf.nao import tddft_iter

dname = os.path.dirname(os.path.abspath(__file__))
td = tddft_iter(label='water', cd=dname)

class KnowValues(unittest.TestCase):
  
  def test_tddft_iter(self):
    """ This is iterative TDDFT with SIESTA starting point """
    self.assertTrue(hasattr(td, 'xocc'))
    self.assertTrue(hasattr(td, 'xvrt'))
    self.assertTrue(td.ksn2f.sum()==8.0) # water: O -- 6 electrons in the valence + H2 -- 2 electrons
    self.assertEqual(td.xocc.shape[0], 4)
    self.assertEqual(td.xvrt.shape[0], 19)
    dn0 = td.apply_rf0(td.moms1[:,0])
    

if __name__ == "__main__":
  unittest.main()
