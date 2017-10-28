from __future__ import print_function, division
import os,unittest,numpy as np

class KnowValues(unittest.TestCase):

  def test_exc(self):
    """ Compute exchange-correlation energy """
    from timeit import default_timer as timer

    from pyscf.nao import scf
    from timeit import default_timer as timer
    
    sv = scf(label='water', cd=os.path.dirname(os.path.abspath(__file__)))
    dm = sv.comp_dm()
    exc = sv.exc(dm, xc_code='1.0*LDA,1.0*PZ', level=4)
    self.assertAlmostEqual(exc, -4.14222392763)

if __name__ == "__main__": unittest.main()
