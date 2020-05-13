from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw

mol = gto.M( verbose = 1,
    atom = '''
        H     0    0        0
        H     0    0.757    0.587''', basis = 'def2-TZVP',)

class KnowValues(unittest.TestCase):
    
  def test_scf_gto_vs_nao(self):
    """ Test computation of overlaps between NAOs against overlaps computed between GTOs"""
    gto_hf = scf.RHF(mol)
    gto_hf.kernel()
    #print(gto_hf.mo_energy)
    
    nao_gw = gw(mf=gto_hf, gto=mol, pb_algorithm='fp', verbosity=0, rescf=True, perform_gw=True)
    nao_gw.dump_chkfile=False
    for e1,egw,e2 in zip(nao_gw.mo_energy[0,0], nao_gw.mo_energy_gw[0,0], gto_hf.mo_energy):
      self.assertAlmostEqual(e1,e2)

if __name__ == "__main__": unittest.main()
