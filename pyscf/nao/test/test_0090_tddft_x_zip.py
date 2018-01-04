from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import tddft_iter
from pyscf.nao.tddft_iter_x_zip import tddft_iter_x_zip

class KnowValues(unittest.TestCase):

  def test_x_zip_feature(self):
    """ This a test for compression of the eigenvectos at higher energies """
    mol = gto.M( verbose = 1, 
      atom = '''H 0 0 0;  H 0 0 0.5; H 0 0 1.0; H 0 0 1.5; H 0 0 2.0; H 0 0 2.5; H 0 0 3.0; H 0 0 3.5;''', 
      basis = 'cc-pvdz', spin=0)
    gto_mf = scf.RHF(mol)
    etot = gto_mf.kernel()

    n = tddft_iter(mf=gto_mf, gto=mol, verbosity=1, jcutoff=10, xc_code='RPA', nr=256)    
    print(__name__, 'n.ksn2e.shape', n.ksn2e.shape)
    x = tddft_iter_x_zip(mf=gto_mf, gto=mol, verbosity=1, x_zip=True, jcutoff=7, xc_code='RPA', nr=256)
    print(__name__, 'x.ksn2e.shape', x.ksn2e.shape)
    
    ww = np.arange(0.0, 1.0, 0.1)+1j*0.2

    data = np.array([ww.real*27.2114, -n.comp_polariz_inter_ave(ww).imag])
    fname = 'h8_chain.tddft_iter_rpa.omega.inter.ave.norma.txt'
    np.savetxt(fname, data.T, fmt=['%f','%f'])
    print(__name__, fname)

    data = np.array([ww.real*27.2114, -x.comp_polariz_inter_ave(ww).imag])
    fname = 'h8_chain.tddft_iter_rpa.omega.inter.ave.x_zip.txt'
    np.savetxt(fname, data.T, fmt=['%f','%f'])
    print(__name__, fname)
        
if __name__ == "__main__": unittest.main()
