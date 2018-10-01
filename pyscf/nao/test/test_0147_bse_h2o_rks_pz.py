from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, tddft, scf
from pyscf.nao import bse_iter
from pyscf.nao import polariz_inter_ave, polariz_nonin_ave
from pyscf.data.nist import HARTREE2EV

class KnowValues(unittest.TestCase):

  def test_0147_bse_h2o_rks_pz(self):
    """ Interacting case """
    mol=gto.M(verbose=0,atom='O 0 0 0;H 0 0.489 1.074;H 0 0.489 -1.074',basis='cc-pvdz',)
    gto_hf = scf.RKS(mol)
    gto_hf.kernel()
    gto_td = tddft.TDDFT(gto_hf)
    gto_td.nstates = 95
    gto_td.kernel()

    omegas = np.arange(0.0, 2.0, 0.01) + 1j*0.03
    p_ave = -polariz_inter_ave(gto_hf, mol, gto_td, omegas).imag
    data = np.array([omegas.real*HARTREE2EV, p_ave])
    np.savetxt('test_0147_bse_h2o_rks_pz_pyscf.txt', data.T, fmt=['%f','%f'])
    data_ref = np.loadtxt('test_0147_bse_h2o_rks_pz_pyscf.txt-ref').T
    self.assertTrue(np.allclose(data_ref, data, 5))
    
    nao_td  = bse_iter(mf=gto_hf, gto=mol, verbosity=0, xc_code='LDA',)

    p_iter = -nao_td.comp_polariz_inter_ave(omegas).imag
    data = np.array([omegas.real*HARTREE2EV, p_iter])
    np.savetxt('test_0147_bse_h2o_rks_pz_nao.txt', data.T, fmt=['%f','%f'])
    data_ref = np.loadtxt('test_0147_bse_h2o_rks_pz_nao.txt-ref').T
    self.assertTrue(np.allclose(data_ref, data, 5))
    
if __name__ == "__main__": unittest.main()
