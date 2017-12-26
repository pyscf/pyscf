from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw

class KnowValues(unittest.TestCase):

  def test_gw(self):
    """ This is GW """
    mol = gto.M( verbose = 1, atom = '''H 0 0 0;  H 0.17 0.7 0.587''', basis = 'cc-pvdz', spin=2)
    #mol = gto.M( verbose = 0, atom = '''H 0.0 0.0 -0.3707;  H 0.0 0.0 0.3707''', basis = 'cc-pvdz',)
    gto_mf = scf.UHF(mol)
    gto_mf.kernel()
    #print('gto_mf.mo_energy:', gto_mf.mo_energy)
    b = gw(mf=gto_mf, gto=mol, verbosity=1, nvrt=4)
    b.kernel_gw()
    
        
if __name__ == "__main__": unittest.main()
