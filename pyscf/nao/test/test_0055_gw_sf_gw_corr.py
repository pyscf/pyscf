from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw as gw_c

class KnowValues(unittest.TestCase):

  def test_sf_gw_corr(self):
    """ This is choice of wmin and wmax in GW """
    mol = gto.M( verbose = 1, atom = '''H 0 0 0;  H 0.17 0.7 0.587''', basis = 'cc-pvdz',)
    gto_mf = scf.RHF(mol)
    gto_mf.kernel()
    gw = gw_c(mf=gto_mf, gto=mol)
    sf = gw.get_snmw2sf()
    self.assertEqual(len(sf), 1)
    self.assertEqual(sf[0].shape, (7,10,32))
    
    
if __name__ == "__main__": unittest.main()
