from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, tddft, scf
from pyscf.nao import tddft_tem
from pyscf.nao import polariz_inter_ave, polariz_nonin_ave

mol = gto.M( verbose = 1,
    atom = ''' H  -0.5  -0.5  -0.5; H  0.5  0.5  0.5''', basis = 'cc-pvdz',)

gto_mf = scf.RKS(mol)
gto_mf.kernel()
gto_td = tddft.TDDFT(gto_mf)
gto_td.nstates = 9
gto_td.kernel()

nao_td  = tddft_tem(mf=gto_mf, gto=mol)

class KnowValues(unittest.TestCase):
    
  def test_tddft_tem(self):
    """ EELS for the hydrogen dimer """
    p_iter = -nao_td.get_spectrum_inter().imag
    data = np.array([nao_td.freq.real*27.2114, p_iter])
    np.savetxt('hydrogen.tddft_tem_lda.omega.inter.pav.txt', data.T, fmt=['%f','%f'])

if __name__ == "__main__": unittest.main()
