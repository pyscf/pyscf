from __future__ import print_function, division
import os,unittest,numpy as np
from pyscf.nao import system_vars_c

#def mycall(d):
#  print(d.keys(), d['dm'].shape, type(d))
  

class KnowValues(unittest.TestCase):

  def test_rescf(self):
    """ reSCF """
    from pyscf import scf
    
    dname = os.path.dirname(os.path.abspath(__file__))
    sv = system_vars_c().init_siesta_xml(label='water', cd=dname)
    #print(sv.get_eigenvalues())
    myhf = scf.RHF(sv)
    myhf.dump_chkfile=False
    #myhf.callback = mycall
    myhf.kernel()
    #print(myhf.mo_energy)

if __name__ == "__main__": unittest.main()
