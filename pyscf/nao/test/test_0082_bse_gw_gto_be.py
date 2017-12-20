from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, tddft, scf
from pyscf.nao import bse_iter
from pyscf.nao import polariz_inter_ave, polariz_nonin_ave

mol = gto.M( verbose = 1, atom = '''Be 0 0 0;''', basis = 'cc-pvdz',)
gto_mf = scf.RHF(mol)
gto_mf.kernel()

class KnowValues(unittest.TestCase):

  def test_bse_gto_vs_nao_inter_0082(self):
    """ Interacting case """
    print(__name__, 'gto.mo_energy', gto_mf.mo_energy)
    nao_td = bse_iter(mf=gto_mf, gto=mol, verbosity=0, xc_code='GW', perform_gw=True)
    
    omegas = np.linspace(0.0,2.0,450)+1j*0.04
    p_iter = -nao_td.polariz_inter_ave(omegas).imag
    data = np.array([omegas.real*27.2114, p_iter])
    np.savetxt('be.bse_iter.omega.inter.ave.txt', data.T, fmt=['%f','%f'])

if __name__ == "__main__": unittest.main()
