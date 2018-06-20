from __future__ import print_function, division
import os,unittest,numpy as np
from pyscf.nao import mf as mf_c

class KnowValues(unittest.TestCase):

  def test_vneutral_atom_matrix_elements(self):
    """ reSCF then G0W0 """
    
    dname = os.path.dirname(os.path.abspath(__file__))
    mf = mf_c(label='lih', cd=dname)
    vna = mf.vna_coo(level=1).toarray()
    rdm = mf.make_rdm1()[0,0,:,:,0]
    Ena = 27.2114*(-0.5)*(vna*rdm).sum()
    #print('Ena   = ', Ena)
    self.assertAlmostEqual(Ena, 6.0251859727456596) # this does compares badly with lih.out 
    #ove = mf.overlap_coo().toarray()
    #print(ove.shape)
    #print(rdm.shape)    
    #print('nelec = ', (ove*rdm).sum())    

if __name__ == "__main__": unittest.main()
