from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import bse_iter

class KnowValues(unittest.TestCase):

  def test_gw(self):
    """ This is GW """
    mol = gto.M( verbose = 1, atom = '''H 0 0 0;  H 0.17 0.7 0.587''', basis = 'cc-pvdz',)
    #mol = gto.M( verbose = 0, atom = '''H 0.0 0.0 -0.3707;  H 0.0 0.0 0.3707''', basis = 'cc-pvdz',)
    gto_mf = scf.RHF(mol)
    gto_mf.kernel()
    #print('gto_mf.mo_energy:', gto_mf.mo_energy)
    b = bse_iter(mf=gto_mf, gto=mol, perform_gw=True, xc_code='GW', verbosity=0, nvrt=4)
    #self.assertAlmostEqual(b.mo_energy[0], -0.5967647)
    #self.assertAlmostEqual(b.mo_energy[1], 0.19072719)
    omegas = np.linspace(0.0,2.0,450)+1j*0.04
    p_iter = -b.comp_polariz_inter_ave(omegas).imag
    data = np.array([omegas.real*27.2114, p_iter])
    np.savetxt('h2_gw_bse_iter.omega.inter.ave.txt', data.T)
    data_ref = np.loadtxt('h2_gw_bse_iter.omega.inter.ave.txt-ref').T
    #print(__name__, abs(data_ref-data).sum()/data.size)
    self.assertTrue(np.allclose(data_ref, data, 5))

    p_iter = -b.comp_polariz_nonin_ave(omegas).imag
    data = np.array([omegas.real*27.2114, p_iter])
    np.savetxt('h2_gw_bse_iter.omega.nonin.ave.txt', data.T)
        
if __name__ == "__main__": unittest.main()
