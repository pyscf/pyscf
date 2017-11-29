from __future__ import print_function, division
import os,unittest,numpy as np


class KnowValues(unittest.TestCase):

  def test_rescf(self):
    """ reSCF than G0W0 N2 example is marked with level change """
    from pyscf.nao import gw as gw_c
    
    dname = os.path.dirname(os.path.abspath(__file__))
    gw = gw_c(label='n2', cd=dname, verbosity=1, jcutoff=9, nff_ia=64, tol_ia=1e-6)
    gw.kernel_scf()
    #self.assertAlmostEqual(gw.mo_energy[0,0,0], -1.327560859909974)
    #self.assertAlmostEqual(gw.mo_energy[0,0,22], 3.9299990455381715)
    print(gw.mo_energy[0,0,0:8])    
    gw.kernel_g0w0()
    print(gw.mo_energy_g0w0)

#    for e,eref in zip(gw.mo_energy_g0w0[0:6], [-2.32001863,-0.66298796,-0.5049757,-0.41390237,0.21671625,0.31001297]):
#      self.assertAlmostEqual(e,eref)
    
    
    

if __name__ == "__main__": unittest.main()
