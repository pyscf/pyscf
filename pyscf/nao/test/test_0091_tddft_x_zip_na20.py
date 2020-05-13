from __future__ import print_function, division
import unittest, numpy as np
from pyscf.nao import tddft_iter
from pyscf.nao.tddft_iter_x_zip import tddft_iter_x_zip as td_c
from os.path import dirname, abspath

class KnowValues(unittest.TestCase):

  def test_x_zip_feature_na20_chain(self):
    """ This a test for compression of the eigenvectos at higher energies """
    dname = dirname(abspath(__file__))
    siesd = dname+'/sodium_20'
    x = td_c(label='siesta', cd=siesd,x_zip=True, x_zip_emax=0.25,x_zip_eps=0.05,jcutoff=7,xc_code='RPA',nr=128, fermi_energy=-0.0913346431431985)
    
    eps = 0.005
    ww = np.arange(0.0, 0.5, eps/2.0)+1j*eps
    data = np.array([ww.real*27.2114, -x.comp_polariz_inter_ave(ww).imag])
    fname = 'na20_chain.tddft_iter_rpa.omega.inter.ave.x_zip.txt'
    np.savetxt(fname, data.T, fmt=['%f','%f'])
    #print(__file__, fname)
    data_ref = np.loadtxt(dname+'/'+fname+'-ref')
    #print('    x.rf0_ncalls ', x.rf0_ncalls)
    #print(' x.matvec_ncalls ', x.matvec_ncalls)
    self.assertTrue(np.allclose(data_ref,data.T, rtol=1.0e-1, atol=1e-06))

if __name__ == "__main__": unittest.main()
