from __future__ import print_function, division
import os,unittest,numpy as np


class KnowValues(unittest.TestCase):

  def test_rescf(self):
    """ reSCF """
    from pyscf.nao.hf import RHF
    
    dname = os.path.dirname(os.path.abspath(__file__))
    myhf = RHF(label='water', cd=dname)
    myhf.kernel()
    self.assertAlmostEqual(myhf.mo_energy[0], -1.3275608669649857)
    self.assertAlmostEqual(myhf.mo_energy[22], 3.9299990077335423)
    #print(myhf.mo_energy)
    #self.assertAlmostEqual(myhf.mo_energy[0], -1.3274696934511327)

if __name__ == "__main__": unittest.main()
