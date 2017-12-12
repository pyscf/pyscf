from __future__ import print_function, division
import os,unittest,numpy as np


class KnowValues(unittest.TestCase):

  def test_vneutral_atom_matrix_elements(self):
    """ reSCF then G0W0 """
    from pyscf.nao import mf as mf_c
    
    dname = os.path.dirname(os.path.abspath(__file__))
    mf = mf_c(label='n2', cd=dname)
    vna = mf.vna_coo().toarray()
    rdm = mf.make_rdm1()[0,0,:,:,0]
    ove = mf.overlap_coo().toarray()
    #print(vna.shape)
    #print(ove.shape)
    #print(rdm.shape)
    
    #print('nelec = ', (ove*rdm).sum())
    print('vna   = ', (vna*rdm).sum()*27.2114)
    

if __name__ == "__main__": unittest.main()
