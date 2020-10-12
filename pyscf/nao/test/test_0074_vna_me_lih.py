from __future__ import print_function, division
import os,unittest,numpy as np
from pyscf.nao import mf as mf_c
from pyscf.data.nist import HARTREE2EV

class KnowValues(unittest.TestCase):

  def test_0074_vna_vnl_LiH(self):
    """ reSCF then G0W0 """

    dname = os.path.dirname(os.path.abspath(__file__))
    mf = mf_c(label='lih', cd=dname)
    vna = mf.vna_coo(level=1).toarray()
    rdm = mf.make_rdm1()[0,0,:,:,0]
    Ena = HARTREE2EV*(-0.5)*(vna*rdm).sum()
    self.assertAlmostEqual(Ena, 6.0251828965429937) # This compares badly with lih.out 
    # siesta: Ena     =         9.253767 

    vnl = mf.vnl_coo().toarray()
    Enl = HARTREE2EV*(vnl*rdm).sum()
    self.assertAlmostEqual(Enl, -2.8533506650656162) # This compares ok with lih.out 
    #siesta: Enl     =        -2.853393
    
    

if __name__ == "__main__": unittest.main()
