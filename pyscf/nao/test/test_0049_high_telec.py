from __future__ import print_function, division
import os,unittest,numpy as np
from pyscf.nao import tddft_iter

dname = os.path.dirname(os.path.abspath(__file__))
td = tddft_iter(label='water', cd=dname, jcutoff=7, iter_broadening=1e-2, xc_code='RPA', telec=0.03)

class KnowValues(unittest.TestCase):

  def test_non_inter_polariz(self):
    """ This is non-interacting polarizability TDDFT with SIESTA starting point """
    omegas = np.linspace(0.0,2.0,500)+1j*td.eps
    pave = -td.comp_polariz_nonin_ave(omegas).imag
    data = np.array([27.2114*omegas.real, pave])
    np.savetxt('water.tddft_iter.telec-0.03.omega-nonin.pav.txt', data.T, fmt=['%f','%f'])
    data_ref = np.loadtxt(dname+'/water.tddft_iter.telec-0.03.omega-nonin.pav.txt-ref')
    self.assertTrue(np.allclose(data_ref,data.T, rtol=1.0, atol=1e-05))
    
  def test_inter_polariz(self):
    """ This is interacting polarizability with SIESTA starting point """
    omegas = np.linspace(0.0,2.0,150)+1j*td.eps
    pxx = -td.comp_polariz_inter_ave(omegas).imag
    data = np.array([omegas.real*27.2114, pxx])
    np.savetxt('water.tddft_iter_rpa.telec-0.03.omega.inter.pav.txt', data.T, fmt=['%f','%f'])
    data_ref = np.loadtxt(dname+'/water.tddft_iter_rpa.telec-0.03.omega.inter.pav.txt-ref')
    #print('    td.rf0_ncalls ', td.rf0_ncalls)
    #print(' td.matvec_ncalls ', td.matvec_ncalls)
    self.assertTrue(np.allclose(data_ref,data.T, rtol=1.0, atol=1e-05))


if __name__ == "__main__": unittest.main()
