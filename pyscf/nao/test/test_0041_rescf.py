from __future__ import print_function, division
import os,unittest,numpy as np


class KnowValues(unittest.TestCase):

  def test_rescf(self):
    """ reSCF """
    from pyscf.nao.hf import RHF
    from pyscf.nao import system_vars_c
    
    dname = os.path.dirname(os.path.abspath(__file__))
    sv = system_vars_c().init_siesta_xml(label='water', cd=dname)
    #print(sv.get_eigenvalues())
    myhf = RHF(sv)
    myhf.kernel()
    self.assertAlmostEqual(myhf.mo_energy[0], -1.327471)
    self.assertAlmostEqual(myhf.mo_energy[22], 3.92999633)
    #print(myhf.mo_energy)

if __name__ == "__main__": unittest.main()
