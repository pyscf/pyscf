from __future__ import print_function, division
import os,unittest,numpy as np

class KnowValues(unittest.TestCase):

  def test_exc(self):
    """ Compute exchange-correlation energy """
    from timeit import default_timer as timer

    from pyscf.nao import mf
    from timeit import default_timer as timer
    
    sv = mf(label='water', cd=os.path.dirname(os.path.abspath(__file__)))
    dm = sv.make_rdm1()
    exc = sv.exc(dm, xc_code='1.0*LDA,1.0*PZ', level=4)
    #self.assertAlmostEqual(exc, -4.1422234271159333) ? redone water?
    self.assertAlmostEqual(exc, -4.1422239276270201)
    

if __name__ == "__main__": unittest.main()
