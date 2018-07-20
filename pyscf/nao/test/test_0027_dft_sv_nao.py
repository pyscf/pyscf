from __future__ import print_function, division
import os,unittest,numpy as np
from pyscf.nao import mf

class KnowValues(unittest.TestCase):

  def test_dft_sv(self):
    """ Try to compute the xc potential """
    
    sv = mf(label='water', cd=os.path.dirname(os.path.abspath(__file__)))
    vxc = sv.vxc_lil()
  
if __name__ == "__main__": unittest.main()
