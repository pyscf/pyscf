from __future__ import print_function, division
import os,unittest,numpy as np

class KnowValues(unittest.TestCase):

  def test_exc(self):
    """ Compute exchange-correlation energy """
    from timeit import default_timer as timer

    from pyscf.nao import system_vars_c
    from pyscf.nao.m_comp_dm import comp_dm
    from timeit import default_timer as timer
    
    sv = system_vars_c().init_siesta_xml(label='water', cd=os.path.dirname(os.path.abspath(__file__)))
    dm = comp_dm(sv.wfsx.x, sv.get_occupations())
    exc = sv.exc(dm, xc_code='1.0*LDA,1.0*PZ', level=4)
    self.assertAlmostEqual(exc, -4.14222392763)

if __name__ == "__main__": unittest.main()
