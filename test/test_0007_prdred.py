from __future__ import print_function, division
import unittest
from pyscf import gto

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
    

if __name__ == "__main__":
  unittest.main()
