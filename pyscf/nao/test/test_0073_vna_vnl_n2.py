from __future__ import print_function, division
import os,unittest,numpy as np
from pyscf.nao import mf as mf_c
from pyscf.data.nist import HARTREE2EV
from pyscf.nao.m_overlap_ni import overlap_ni

class KnowValues(unittest.TestCase):

  def test_0073_vna_vnl_N2(self):
    """ Test the Ena energy and indirectly VNA matrix elements """
    
    dname = os.path.dirname(os.path.abspath(__file__))
    mf = mf_c(label='n2', cd=dname)
    vna = mf.vna_coo(level=3).toarray()
    rdm = mf.make_rdm1()[0,0,:,:,0]
    Ena = HARTREE2EV*(-0.5)*(vna*rdm).sum()
    self.assertAlmostEqual(Ena, 133.24212864149359)
    # siesta: Ena     =       133.196299

    vnl = mf.vnl_coo().toarray()
    Enl = HARTREE2EV*(vnl*rdm).sum()
    self.assertAlmostEqual(Enl, -61.604522776730128)
    #siesta: Enl     =       -61.601204

if __name__ == "__main__": unittest.main()
