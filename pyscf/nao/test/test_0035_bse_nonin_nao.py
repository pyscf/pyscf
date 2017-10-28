from __future__ import print_function, division
import os,unittest,numpy as np

class KnowValues(unittest.TestCase):

  def test_bse_iter_nonin(self):
    """ Compute polarization with LDA TDDFT  """
    from timeit import default_timer as timer
    from pyscf.nao import bse_iter
    
    dname = os.path.dirname(os.path.abspath(__file__))
    bse = bse_iter(label='water', cd=dname, iter_broadening=1e-2)
    omegas = np.linspace(0.0,2.0,500)+1j*bse.eps
    
    pxx = np.zeros(len(omegas))
    for iw,omega in enumerate(omegas):
      for ixyz in range(1):
        vab = bse.apply_l0(bse.dab[ixyz], omega)
        pxx[iw] = pxx[iw] - (vab.imag*bse.dab[ixyz]).sum()
        
    data = np.array([omegas.real*27.2114, pxx])
    np.savetxt('water.bse_iter_rpa.omega.nonin.pxx.txt', data.T, fmt=['%f','%f'])
    data_ref = np.loadtxt(dname+'/water.bse_iter_rpa.omega.nonin.pxx.txt-ref')
    #print('    bse.l0_ncalls ', bse.l0_ncalls)
    self.assertTrue(np.allclose(data_ref,data.T, rtol=1.0, atol=1e-05))


if __name__ == "__main__": unittest.main()
