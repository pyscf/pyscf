from __future__ import print_function, division
import os,unittest,numpy as np

class KnowValues(unittest.TestCase):

  def test_read_siesta_bulk_spin(self):
    """ Test reading of bulk, spin-resolved SIESTA calculation  """
    from pyscf.nao import mf
    
    chdir = os.path.dirname(os.path.abspath(__file__))+'/ice'
    sv  = mf(label='siesta', cd=chdir)
    sv.diag_check()

if __name__ == "__main__": unittest.main()

