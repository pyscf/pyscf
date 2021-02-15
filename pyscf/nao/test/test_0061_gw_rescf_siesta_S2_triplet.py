from __future__ import print_function, division
import os,unittest,numpy as np
from pyscf.nao import gw as gw_c

class KnowValues(unittest.TestCase):

    def test_0061_gw_rescf_siesta_s2_triplet (self):
        """ reSCF then G0W0 """
        dname = os.path.dirname(os.path.abspath(__file__))+'/S2_triplet'
        gw = gw_c(label='S2', cd=dname, verbosity=1, rescf=True, tol_ia=1e-6, magnetization=2)    #2 Unpaired electron
        gw.kernel_gw()
        #gw.report()
    
if __name__ == "__main__": 
    unittest.main()
