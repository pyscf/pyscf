from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw_iter
from pyscf.nao import gw as gw_c
import numpy as np

class KnowValues(unittest.TestCase):

  def test_sf_iter(self):
    """ This compares matrix element of W calculated by G0W0 and G0W0_iter """
    mol = gto.M(atom='''C 0.0, 0.0, -0.611046 ; N 0.0, 0.0, 0.52375''', basis='ccpvdz',spin=1)
    gto_mf = scf.UHF(mol)
    gto_mf.kernel()
    
    gw = gw_c(mf=gto_mf, gto=mol, verbosity=0, niter_max_ev=20, kmat_algo='sm0_sum')
    gwi = gw_iter(mf=gto_mf, gto=mol, verbosity=0, niter_max_ev=20, kmat_algo='sm0_sum')

    sf = gw.get_snmw2sf()
    sf_i = gwi.get_snmw2sf_iter()
    self.assertEqual(len(sf), len(sf_i))
    self.assertEqual(sf[0].shape, sf_i[0].shape)
    self.assertTrue(np.allclose(sf, sf_i,atol = 1e-05))
    
    
if __name__ == "__main__": unittest.main()
