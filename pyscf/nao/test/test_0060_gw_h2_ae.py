from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw as gw_c

class KnowValues(unittest.TestCase):

  def test_gw(self):
    """ This is GW """
    mol = gto.M( verbose = 0, atom = '''H 0.0 0.0 -0.3707;  H 0.0 0.0 0.3707''', basis = 'cc-pvdz', )
    gto_mf = scf.RHF(mol)
    gto_mf.kernel()
    #print('gto_mf.mo_energy:', gto_mf.mo_energy)
    gw = gw_c(mf=gto_mf, gto=mol, verbosity=0,)
    gw.kernel_gw()
    self.assertAlmostEqual(gw.mo_energy_gw[0,0,0], -0.59709476270318296)
    self.assertAlmostEqual(gw.mo_energy_gw[0,0,1], 0.19071318743971943)
    
        
if __name__ == "__main__": unittest.main()
