from __future__ import print_function, division
import os,unittest,numpy as np


class KnowValues(unittest.TestCase):

  def test_rescf(self):
    """ reSCF then G0W0 """
    from pyscf.nao import gw as gw_c
    
    dname = os.path.dirname(os.path.abspath(__file__))
    gw = gw_c(label='water', cd=dname, verbosity=1)
    gw.kernel_scf()
    #self.assertAlmostEqual(gw.mo_energy[0,0,0], -1.327560859909974)
    #self.assertAlmostEqual(gw.mo_energy[0,0,22], 3.9299990455381715)
    #print(gw.mo_energy[0,0,0:8])
    gw.kernel_g0w0()
    for e,eref in zip(gw.mo_energy_g0w0[1:7], [-0.50497589,-0.41390245,-0.2008947,0.21671625,0.31001297]):
      self.assertAlmostEqual(e,eref)
    
    
    

if __name__ == "__main__": unittest.main()
