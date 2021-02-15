from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw as gw_c

class KnowValues(unittest.TestCase):

  def test_sf_gw_res_corr(self):
    """ This is choice of wmin and wmax in GW """
    mol = gto.M( verbose = 1, atom = '''H 0 0 0;  H 0.17 0.7 0.587''', basis = 'cc-pvdz',)
    gto_mf = scf.RHF(mol)
    gto_mf.kernel()
    gw = gw_c(mf=gto_mf, gto=mol, tol_ia=1e-6)
    sn2eval_gw = [np.copy(gw.ksn2e[0,s,nn]) for s,nn in enumerate(gw.nn) ]
    gw_corr_res = gw.gw_corr_res(sn2eval_gw)
    
    fc = """0.03105265 -0.00339984 -0.01294826 -0.06934852 -0.03335821 -0.03335821 0.46324024"""
    for e,eref_str in zip(gw_corr_res[0],fc.split(' ')):
      self.assertAlmostEqual(e,float(eref_str),7)    

if __name__ == "__main__": unittest.main()
