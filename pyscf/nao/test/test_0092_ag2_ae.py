from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import scf as scf_c

class KnowValues(unittest.TestCase):

  def test_gw(self):
    """ This is GW """
    mol = gto.M( verbose = 0, atom = '''Ag 0.0 0.0 -0.3707;  Ag 0.0 0.0 0.3707''', basis = 'cc-pvdz-pp',)
    gto_mf = scf.RHF(mol)
    gto_mf.kernel()
    print('gto_mf.mo_energy:', gto_mf.mo_energy)
    s = scf_c(mf=gto_mf, gto=mol, verbosity=0)
    s.kernel_scf()
    print('s.mo_energy:', s.mo_energy)
    
        
if __name__ == "__main__": unittest.main()
