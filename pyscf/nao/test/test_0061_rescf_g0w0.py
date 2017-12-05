from __future__ import print_function, division
import os,unittest,numpy as np


class KnowValues(unittest.TestCase):

  def test_rescf(self):
    """ reSCF then G0W0 """
    from pyscf.nao import gw as gw_c
    
    dname = os.path.dirname(os.path.abspath(__file__))
    gw = gw_c(label='water', cd=dname, verbosity=1, nocc_conv=4, nvrt_conv=4)
    gw.kernel_scf()
    #self.assertAlmostEqual(gw.mo_energy[0,0,0], -1.327560859909974)
    #self.assertAlmostEqual(gw.mo_energy[0,0,22], 3.9299990455381715)
    #print(gw.mo_energy[0,0,0:8])
    gw.kernel_g0w0()
    for e,eref in zip(gw.mo_energy_g0w0, [-1.1779330853563,-0.67209042,-0.51326308,-0.43210297,0.207645,0.3093115,0.53504053, 0.58625492]):
      self.assertAlmostEqual(e,eref)
    
    
    

if __name__ == "__main__": unittest.main()
