from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import gw as gw_c

class KnowValues(unittest.TestCase):

  def test_gw(self):
    """ This is GW """
    mol = gto.M( verbose = 1, atom = '''H 0.0 0.0 -0.3707;  H 0.0 0.0 0.3707''', basis = 'cc-pvdz',)
    gto_mf = scf.RHF(mol)
    gto_mf.kernel()
    print('gto_mf.mo_energy:', gto_mf.mo_energy)
    gw = gw_c(mf=gto_mf, gto=mol)
    print('       gw.nff_ia:', gw.nff_ia)
    print('      gw.wmin_ia:', gw.wmin_ia)
    print('      gw.wmax_ia:', gw.wmax_ia)
    expv = gw.get_h0_vh_x_expval()
    print(' expv ', expv)
    gw_expv = gw.correct_ev()
    print(' gw_expv ', gw_expv)
    
        
if __name__ == "__main__": unittest.main()
