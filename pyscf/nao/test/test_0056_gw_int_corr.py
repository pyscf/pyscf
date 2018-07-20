from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw as gw_c

class KnowValues(unittest.TestCase):

  def test_sf_gw_corr(self):
    """ This is choice of wmin and wmax in GW """
    mol = gto.M( verbose = 1, atom = '''H 0.0 0.0 -0.3707;  H 0.0 0.0 0.3707''', basis = 'cc-pvdz',)
    gto_mf = scf.RHF(mol)
    gto_mf.kernel()
    gw = gw_c(mf=gto_mf, gto=mol)
    sn2eval_gw = np.copy(gw.ksn2e[0,:,gw.nn]).T    
    gw_corr_int = gw.gw_corr_int(sn2eval_gw)
    for c,cref in zip(gw_corr_int[0], [ -4.16459719e-02, -4.79778564e-03,-3.26848826e-03,-4.06138735e-05,
      -5.13947211e-03,  -5.13947211e-03, 8.42977051e-03]):
      self.assertAlmostEqual(c[0], cref)

if __name__ == "__main__": unittest.main()
