from __future__ import print_function, division
import unittest
from pyscf import gto
import numpy as np

mol = gto.M(
    verbose = 1,
    atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)

class KnowValues(unittest.TestCase):

  def test_prdred(self):
    """  """
    from pyscf.nao import system_vars_c, prod_basis_c
    from pyscf.nao.m_prod_talman import prod_talman_c
    
    sv = system_vars_c().init_pyscf_gto(mol)
    pt = prod_talman_c(sv.ao_log)
    self.assertEqual(pt.ngl, 96)
    jtb,clbdtb,lbdtb=pt.prdred_terms(2,4)
    self.assertEqual(len(jtb), 475)
    la, lb = 0, 0
    ra,rb = np.array([0.0,0.1,0.5]),np.array([1.5,0.99,0.5])
    rcen = (ra+rb)*0.5
    #pt.prdred(sv.ao_log.psi_log[0][0,:],la,ra,sv.ao_log.psi_log[0][1,:],lb,rb, rcen)
    

if __name__ == "__main__":
  unittest.main()
