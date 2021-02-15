from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, tddft, scf
from pyscf.nao import tddft_iter
from pyscf.nao import polariz_inter_ave, polariz_nonin_ave

mol = gto.M( verbose = 1,
    atom = '''
        H     0    0        0
        H     0.17    0.7    0.587''', basis = 'cc-pvdz',)

gto_mf = scf.RKS(mol)
gto_mf.kernel()
gto_td = tddft.TDDFT(gto_mf)
gto_td.nstates = 9
gto_td.kernel()

nao_td  = tddft_iter(mf=gto_mf, gto=mol)

class KnowValues(unittest.TestCase):
    
  def test_tddft_gto_vs_nao_inter(self):
    """ Interacting case """
    omegas = np.linspace(0.0,2.0,150)+1j*0.04
    p_ave = -polariz_inter_ave(gto_mf, mol, gto_td, omegas).imag
    data = np.array([omegas.real*27.2114, p_ave])
    np.savetxt('hydrogen.tddft_lda.omega.inter.pav.txt', data.T, fmt=['%f','%f'])
    p_iter = -nao_td.comp_polariz_inter_ave(omegas).imag
    data = np.array([omegas.real*27.2114, p_iter])
    np.savetxt('hydrogen.tddft_iter_lda.omega.inter.pav.txt', data.T, fmt=['%f','%f'])
    #print('inter', abs(p_ave-p_iter).sum()/omegas.size)
    self.assertTrue(abs(p_ave-p_iter).sum()/omegas.size<0.03)
    
  def test_tddft_gto_vs_nao_nonin(self):
    """ Non-interacting case """
    omegas = np.linspace(0.0,2.0,150)+1j*0.04
    p_ave = -polariz_nonin_ave(gto_mf, mol, omegas).imag
    data = np.array([omegas.real*27.2114, p_ave])
    np.savetxt('hydrogen.tddft_lda.omega.nonin.pav.txt', data.T, fmt=['%f','%f'])
    nao_td  = tddft_iter(mf=gto_mf, gto=mol)
    p_iter = -nao_td.comp_polariz_nonin_ave(omegas).imag
    data = np.array([omegas.real*27.2114, p_iter])
    np.savetxt('hydrogen.tddft_iter_lda.omega.nonin.pav.txt', data.T, fmt=['%f','%f'])
    #print('nonin', abs(p_ave-p_iter).sum()/omegas.size)
    self.assertTrue(abs(p_ave-p_iter).sum()/omegas.size<0.03)

if __name__ == "__main__": unittest.main()
