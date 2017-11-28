from __future__ import print_function, division
import os,unittest,numpy as np


class KnowValues(unittest.TestCase):

  def test_rescf(self):
    """ reSCF """
    from pyscf.nao.scf import scf
    
    dname = os.path.dirname(os.path.abspath(__file__))
    myhf = scf(label='water', cd=dname)
    myhf.kernel_scf()
    #self.assertAlmostEqual(myhf.mo_energy[0,0,0], -1.3274696934511327)
    #self.assertAlmostEqual(myhf.mo_energy[0,0,22], 3.9299990077335423)

    self.assertAlmostEqual(myhf.mo_energy[0,0,0], -1.327560859909974)
    self.assertAlmostEqual(myhf.mo_energy[0,0,22], 3.9299990455381715)
    #print(myhf.mo_energy)

if __name__ == "__main__": unittest.main()
