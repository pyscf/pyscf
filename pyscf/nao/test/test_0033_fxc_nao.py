from __future__ import print_function, division
import os,unittest,numpy as np
from pyscf.nao import tddft_iter

class KnowValues(unittest.TestCase):

  def test_fxc(self):
    """ Compute TDDFT interaction kernel  """
    
    td = tddft_iter(label='water', cd=os.path.dirname(os.path.abspath(__file__)))
    fxc = td.comp_fxc_lil(xc_code='1.0*LDA,1.0*PZ', level=4)
    #self.assertAlmostEqual(fxc.sum(), -64.8139811684)

if __name__ == "__main__": unittest.main()
