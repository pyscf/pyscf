from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, tddft, scf
from pyscf.nao import bse_iter
from pyscf.nao import polariz_inter_ave
from pyscf.data.nist import HARTREE2EV
from pyscf.nao.m_polariz_inter_ave import polariz_freq_osc_strength

class KnowValues(unittest.TestCase):

  def test_0150_bse_h2o_uhf_rpa(self):
    """ Interacting case """
    mol=gto.M(verbose=0,atom='O 0 0 0;H 0 0.489 1.074;H 0 0.489 -1.074',basis='cc-pvdz',)
    gto_mf = scf.UKS(mol)
    gto_mf.xc = 'hf'
    gto_mf.kernel()
    gto_td = tddft.dRPA(gto_mf)
    gto_td.nstates = 95
    gto_td.kernel()

    omegas = np.arange(0.0, 2.0, 0.01) + 1j*0.03
    p_ave = -polariz_freq_osc_strength(gto_td.e, gto_td.oscillator_strength(), omegas).imag
    data = np.array([omegas.real*HARTREE2EV, p_ave])
    np.savetxt('test_0150_bse_h2o_uhf_rpa_pyscf.txt', data.T, fmt=['%f','%f'])
    data_ref = np.loadtxt('test_0150_bse_h2o_uhf_rpa_pyscf.txt-ref').T
    self.assertTrue(np.allclose(data_ref, data, atol=1e-6, rtol=1e-3))
    
    nao_td  = bse_iter(mf=gto_mf, gto=mol, verbosity=0, xc_code='RPA')

    polariz = -nao_td.comp_polariz_inter_ave(omegas).imag
    data = np.array([omegas.real*HARTREE2EV, polariz])
    np.savetxt('test_0150_bse_h2o_uhf_rpa_nao.txt', data.T, fmt=['%f','%f'])
    data_ref = np.loadtxt('test_0150_bse_h2o_uhf_rpa_nao.txt-ref').T
    self.assertTrue(np.allclose(data_ref, data, atol=1e-6, rtol=1e-3))
        
if __name__ == "__main__": unittest.main()
