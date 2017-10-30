from __future__ import print_function, division
import os,unittest,numpy as np

class KnowValues(unittest.TestCase):

  def test_fxc(self):
    """ Compute TDDFT interaction kernel  """
    from pyscf.nao import system_vars_c, prod_basis_c
    from pyscf.nao.m_comp_dm import comp_dm
    from timeit import default_timer as timer
    
    sv = system_vars_c().init_siesta_xml(label='water', cd=os.path.dirname(os.path.abspath(__file__)))
    pb = prod_basis_c().init_prod_basis_pp(sv)
    dm = comp_dm(sv.wfsx.x, sv.get_occupations())
    fxc = pb.comp_fxc_lil(dm=dm, xc_code='1.0*LDA,1.0*PZ', level=4)
    #self.assertAlmostEqual(fxc.sum(), -64.8139811684)
    
if __name__ == "__main__": unittest.main()
