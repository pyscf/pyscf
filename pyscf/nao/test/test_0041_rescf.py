from __future__ import print_function, division
import os,unittest,numpy as np
from pyscf.nao import system_vars_c, prod_basis_c, tddft_iter_c

class KnowValues(unittest.TestCase):

  def test_rescf(self):
    """ reSCF """
    from pyscf.nao.m_hf import RHF

    dname = os.path.dirname(os.path.abspath(__file__))
    sv = system_vars_c().init_siesta_xml(label='water', cd=dname)
    pb = prod_basis_c().init_prod_basis_pp(sv, jcutoff=7)

    myhf = RHF(sv)
    myhf.kernel()

if __name__ == "__main__": unittest.main()
