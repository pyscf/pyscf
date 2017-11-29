from __future__ import print_function, division
import os,unittest,numpy as np


class KnowValues(unittest.TestCase):

  def test_rescf(self):
    """ reSCF """
    from pyscf.nao import scf as scf_c
    from pyscf.nao import gw as gw_c
    
    dname = os.path.dirname(os.path.abspath(__file__))
    gw = gw_c(label='water', cd=dname)
    gw.kernel_scf()
    #self.assertAlmostEqual(gw.mo_energy[0,0,0], -1.327560859909974)
    #self.assertAlmostEqual(gw.mo_energy[0,0,22], 3.9299990455381715)
    #print(gw.mo_energy[0,0,0:8])
    gw.kernel_g0w0()
    for e,eref in zip(gw.mo_energy_g0w0[0:6], [-2.32001863,-0.66298796,-0.5049757,-0.41390237,0.21671625,0.31001297]):
      self.assertAlmostEqual(e,eref)
    
    
    

if __name__ == "__main__": unittest.main()
