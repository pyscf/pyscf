from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw as gw_c

class KnowValues(unittest.TestCase):

  def test_wmin_wmax(self):
    """ This is choice of wmin and wmax in GW """
    mol = gto.M( verbose = 1, atom = '''H 0 0 0;  H 0.17 0.7 0.587''', basis = 'cc-pvdz',)
    gto_mf = scf.RHF(mol)
    gto_mf.kernel()
    gw = gw_c(mf=gto_mf, gto=mol, tol_ia=1e-3)
    self.assertEqual(gw.nocc, 1)
    self.assertEqual(gw.nvrt, 6)
    self.assertEqual(gw.start_st, 0)
    self.assertEqual(gw.finish_st, 7)
    self.assertAlmostEqual(gw.wmin_ia, 0.010963536607965261)
    self.assertAlmostEqual(gw.wmax_ia, 10.396997096859502)
    self.assertAlmostEqual(gw.ww_ia.sum(), 60.588528092301765)
    self.assertAlmostEqual(gw.tt_ia.sum(), 57.489242777049725)
    
if __name__ == "__main__": unittest.main()
