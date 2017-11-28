from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import nao
from pyscf.nao.hf import RHF

mol = gto.M( verbose = 1,
    atom = '''
        H     0    0        0
        H     0    0.757    0.587''', basis = 'cc-pvdz',)

class KnowValues(unittest.TestCase):
    
  def test_scf_gto_vs_nao(self):
    """ Test computation of overlaps between NAOs against overlaps computed between GTOs"""
    gto_hf = scf.RHF(mol)
    gto_hf.kernel()
    
    nao_hf = RHF(mf=gto_hf, gto=mol)
    nao_hf.dump_chkfile=False
    nao_hf.kernel()
    self.assertAlmostEqual(gto_hf.e_tot, nao_hf.e_tot, 4)
    for e1,e2 in zip(nao_hf.mo_energy,gto_hf.mo_energy): self.assertAlmostEqual(e1, e2, 3)
    for o1,o2 in zip(nao_hf.mo_occ,gto_hf.mo_occ): self.assertAlmostEqual(o1, o2)

if __name__ == "__main__":
  print("Test of SCF done via NAOs")
  unittest.main()
