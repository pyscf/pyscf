from __future__ import print_function, division
import unittest
import numpy as np
from pyscf import gto
from pyscf.nao import system_vars_c

mol = gto.M(
    verbose = 1,
    atom = '''
       Li     0    0        0
        H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)

class KnowValues(unittest.TestCase):
    
  def test_overlap_gto_vs_nao(self):
    """ Test computation of overlaps computed between NAOs against overlaps computed between GTOs"""
    from pyscf.nao import conv_yzx2xyz_c
    from pyscf.nao.m_overlap_am import overlap_am
    sv = system_vars_c().init_pyscf_gto(mol)
    oref = conv_yzx2xyz_c(mol).conv_yzx2xyz_2d(mol.intor_symmetric('cint1e_ovlp_sph'), direction='pyscf2nao')
    over = sv.overlap_coo(funct=overlap_am).toarray()
    self.assertTrue(abs(over-oref).sum()<5e-9)

if __name__ == "__main__":
  print("Tests for system_vars_c")
  unittest.main()
