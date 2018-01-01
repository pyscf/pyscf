from __future__ import print_function, division
import os,unittest
from pyscf.nao import tddft_iter

dname = os.path.dirname(os.path.abspath(__file__))

td = tddft_iter(label='water', cd=dname)
try:
    from pyscf.lib import misc
    libnao_gpu = misc.load_library("libnao_gpu")
    td_gpu = tddft_iter(label='water', cd=dname, GPU=True)
except:
    td_gpu = None

class KnowValues(unittest.TestCase):
  
  def test_tddft_iter(self):
    """ This is iterative TDDFT with SIESTA starting point """
    self.assertTrue(hasattr(td, 'xocc'))
    self.assertTrue(hasattr(td, 'xvrt'))
    self.assertTrue(td.ksn2f.sum()==8.0) # water: O -- 6 electrons in the valence + H2 -- 2 electrons
    self.assertEqual(td.xocc[0].shape[0], 4)
    self.assertEqual(td.xvrt[0].shape[0], 19)
    dn0 = td.apply_rf0(td.moms1[:,0])

  def test_tddft_iter_gpu(self):
    """ Test GPU version """
    if td_gpu is not None:
      self.assertTrue(hasattr(td_gpu, 'xocc'))
      self.assertTrue(hasattr(td_gpu, 'xvrt'))
      self.assertTrue(td_gpu.ksn2f.sum()==8.0) # water: O -- 6 electrons in the valence + H2 -- 2 electrons
      self.assertEqual(td_gpu.xocc.shape[0], 4)
      self.assertEqual(td_gpu.xvrt.shape[0], 19)
      dn0 = td_gpu.apply_rf0(td_gpu.moms1[:,0])

   

if __name__ == "__main__": unittest.main()
