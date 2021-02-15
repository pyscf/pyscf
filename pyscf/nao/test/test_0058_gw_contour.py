from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import mf

class KnowValues(unittest.TestCase):

  def test_sf_gw_res_corr(self):
    """ This is choice of wmin and wmax in GW """
    mol = gto.M( verbose = 1, atom = '''H 0 0 0;  H 0.17 0.7 0.587''', basis = 'cc-pvdz',)
    gto_mf = scf.RHF(mol)
    gto_mf.kernel()
    s = mf(mf=gto_mf, gto=mol)
    #s.plot_contour(s.mo_energy[0])
    #s.plot_contour(s.mo_energy[1])

if __name__ == "__main__": unittest.main()
