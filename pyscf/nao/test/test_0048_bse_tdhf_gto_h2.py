from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, tddft, scf
from pyscf.nao import bse_iter
from pyscf.nao import polariz_inter_ave, polariz_nonin_ave

mol = gto.M( verbose = 0, atom = '''H 0 0 0;  H 0.17 0.7 0.587''', basis = 'cc-pvdz',)

gto_mf = scf.RHF(mol)
gto_mf.kernel()
#print(gto_mf.mo_energy)
gto_td = tddft.TDDFT(gto_mf)
gto_td.nstates = 9
gto_td.singlet = True # False
gto_td.kernel()

nao_td  = bse_iter(mf=gto_mf, gto=mol, verbosity=0)

class KnowValues(unittest.TestCase):

  #def test_bse_iter_vs_tdhf_pyscf(self):
  #  """ Interacting case test """
  #  cdip = np.random.rand(nao_td.norbs,nao_td.norbs)+1j*np.random.rand(nao_td.norbs,nao_td.norbs)
  #  nao_td.apply_l0_exp(cdip, comega=0.2+1j*0.01)
    
  def test_tddft_gto_vs_nao_inter(self):
    """ Interacting case """
    omegas = np.linspace(0.0,2.0,450)+1j*0.04
    p_ave = -polariz_inter_ave(gto_mf, mol, gto_td, omegas).imag
    data = np.array([omegas.real*27.2114, p_ave])
    np.savetxt('hydrogen.tdhf.omega.inter.pav.txt', data.T, fmt=['%f','%f'])
    p_iter = -nao_td.comp_polariz_inter_ave(omegas).imag
    data = np.array([omegas.real*27.2114, p_iter])
    np.savetxt('hydrogen.bse_iter_hf.omega.inter.pav.txt', data.T, fmt=['%f','%f'])
    #print('inter', abs(p_ave-p_iter).sum()/omegas.size, nao_td.l0_ncalls)
    #import matplotlib.pyplot as plt
    #plt.plot(omegas.real, p_ave, "b")
    #plt.plot(omegas.real, p_iter, "--r")
    #plt.show()
    self.assertTrue(abs(p_ave-p_iter).sum()/omegas.size<0.01)
    
  def test_tddft_gto_vs_nao_nonin(self):
    """ Non-interacting case """
    omegas = np.linspace(0.0,2.0,450)+1j*0.04
    p_ave = -polariz_nonin_ave(gto_mf, mol, omegas).imag
    data = np.array([omegas.real*27.2114, p_ave])
    np.savetxt('hydrogen.tdhf.omega.nonin.pav.txt', data.T, fmt=['%f','%f'])
    p_iter = -nao_td.comp_polariz_nonin_ave(omegas).imag
    data = np.array([omegas.real*27.2114, p_iter])
    np.savetxt('hydrogen.bse_iter.omega.nonin.pav.txt', data.T, fmt=['%f','%f'])
    #print('nonin', abs(p_ave-p_iter).sum()/omegas.size)
    self.assertTrue(abs(p_ave-p_iter).sum()/omegas.size<0.03)

if __name__ == "__main__": unittest.main()
