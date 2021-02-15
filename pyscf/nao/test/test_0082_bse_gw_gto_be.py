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
    #dm1 = gto_mf.make_rdm1()
    #o1 = gto_mf.get_ovlp()
    #print(__name__, 'dm1*o1', (dm1*o1).sum())
    nao_td = bse_iter(mf=gto_mf, gto=mol, verbosity=0, xc_code='GW', perform_gw=True)
    
    #dm2 = nao_td.make_rdm1()
    #o2 = nao_td.get_ovlp()
    #n = nao_td.norbs
    #print(__name__, 'dm2*o2', (dm2.reshape((n,n))*o2).sum())
    
    omegas = np.linspace(0.0,2.0,450)+1j*0.04
    p_iter = -nao_td.comp_polariz_inter_ave(omegas).imag
    data = np.array([omegas.real*27.2114, p_iter])
    np.savetxt('be.bse_iter.omega.inter.ave.txt', data.T, fmt=['%f','%f'])

if __name__ == "__main__": unittest.main()
