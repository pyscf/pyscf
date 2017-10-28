from __future__ import print_function, division
import os,unittest,numpy as np


class KnowValues(unittest.TestCase):

  def test_rescf(self):
    """ reSCF """
    from pyscf.nao.hf import RHF
    from pyscf.nao import scf
    
    dname = os.path.dirname(os.path.abspath(__file__))
    sv = scf(label='water', cd=dname)
    myhf = RHF(sv)
    myhf.kernel()
    self.assertAlmostEqual(myhf.mo_energy[0], -1.327471)
    self.assertAlmostEqual(myhf.mo_energy[22], 3.92999633)
    #print(myhf.mo_energy)

if __name__ == "__main__": unittest.main()
