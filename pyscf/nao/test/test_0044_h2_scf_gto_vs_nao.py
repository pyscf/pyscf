from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import system_vars_c, conv_yzx2xyz_c

mol = gto.M( verbose = 1,
    atom = '''
        H     0    0        0
        H     0    0.757    0.587''', basis = 'cc-pvdz',)
conv = conv_yzx2xyz_c(mol)
gto_hf = scf.RHF(mol)
gto_hf.kernel()
sv = system_vars_c().init_pyscf_gto(mol)

class KnowValues(unittest.TestCase):
    
  def test_scf_gto_vs_nao(self):
    """ Test computation of overlaps between NAOs against overlaps computed between GTOs"""
    from pyscf.nao.m_hf import RHF
    nao_hf = RHF(sv)
    #nao_hf.kernel()
    #print(dir(nao_hf), nao_hf.mo_energy)
    #print(dir(gto_hf), gto_hf.mo_energy)
    
if __name__ == "__main__":
  print("Test of SCF done via NAOs")
  unittest.main()
