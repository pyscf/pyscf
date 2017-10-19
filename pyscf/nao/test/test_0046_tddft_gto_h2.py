from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, tddft, scf as scf_gto
from pyscf.nao import nao
from pyscf.nao import scf as scf_nao

mol = gto.M( verbose = 1,
    atom = '''
        H     0    0        0
        H     0    0.757    0.587''', basis = 'cc-pvdz',)

class KnowValues(unittest.TestCase):
    
  def test_tddft_gto_vs_nao(self):
    """ """
    gto_mf = scf.RKS(mol)
    gto_mf.kernel()
    print(dir(gto_mf))
    print(gto_mf.xc)
    print(gto_mf.pop())

    gto_td = tddft.TDDFT(gto_mf)
    gto_td.nstates = 9
    gto_td.kernel()
    print('Excitation energy (eV)', gto_td.e * 27.2114)

    nao_mol = nao(gto=mol, verbose=0)
    nao_mf  = scf
    

if __name__ == "__main__":
  print("Test of TDDFT GTO versus NAO")
  unittest.main()
