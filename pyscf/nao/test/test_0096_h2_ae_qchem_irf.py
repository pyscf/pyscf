from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf, dft
from pyscf.gw import GW
from pyscf.tddft import TDDFT
from pyscf.nao.qchem_inter_rf import qchem_inter_rf
from pyscf.nao import tddft_iter

mol = gto.M( verbose = 1,
    atom = '''
        H    0.0   2.0      0.0
        N    2.0   0.0      0.0''', basis = 'cc-pVDZ',)

class KnowValues(unittest.TestCase):
    
  def test_qchem_irf(self):
    """ Test """
    gto_hf = dft.DFT(mol, xc_code='LDA')
    gto_hf.kernel()
    print(gto_hf.mo_energy)
    gto_gw = GW(gto_hf)
    gto_gw.kernel()
    

    nao_td  = tddft_iter(mf=gto_hf, gto=mol, xc_code='RPA')
    eps = 0.02
    omegas = np.arange(0.0,2.0,eps/2.0)+1j*eps
    p_iter = -nao_td.comp_polariz_inter_ave(omegas).imag
    data = np.array([omegas.real*27.2114, p_iter])
    np.savetxt('NH.tddft_iter_rpa.omega.inter.pav.txt', data.T, fmt=['%f','%f'])
    
    np.set_printoptions(linewidth=180)
    qrf = qchem_inter_rf(mf=gto_hf, gto=mol, pb_algorithm='fp', verbosity=1)
    

if __name__ == "__main__": unittest.main()
