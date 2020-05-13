from __future__ import print_function, division
import unittest, numpy as np

from pyscf.nao.log_mesh import funct_log_mesh
from pyscf.nao.m_log_interp import log_interp_c
from timeit import default_timer as timer

class KnowValues(unittest.TestCase):

  def test_log_interp_vv_speed(self):
    """ Test the interpolation facility for an array arguments from the class log_interp_c """
    rr,pp = funct_log_mesh(1024, 0.01, 200.0)
    lgi = log_interp_c(rr)

    gcs = np.array([1.2030, 3.2030, 0.7, 10.0, 5.3])
    ff = np.array([[np.exp(-gc*r**2) for r in rr] for gc in gcs])

    rr = np.linspace(0.05, 250.0, 2000000)
    t1 = timer()
    fr2yy = lgi.interp_rcut(ff, rr, rcut=16.0)
    t2 = timer()
    #print(__name__, 't2-t1: ', t2-t1)
    yyref = np.exp(-(gcs.reshape(gcs.size,1)) * (rr.reshape(1,rr.size)**2))
      
    self.assertTrue(np.allclose(fr2yy, yyref) )    

  def test_log_interp_sparse_coeffs(self):
    """ Test the computation of interpolation coefficients """
    rr,pp = funct_log_mesh(512, 0.01, 200.0)
    lgi = log_interp_c(rr)
    rrs = np.linspace(0.05, 250.0, 20000)
    kr2c_csr,j2r = lgi.coeffs_csr(rrs, rcut=10.0)
    j2r,j2k,ij2c = lgi.coeffs_rcut(rrs, rcut=10.0)
    for r,k,cc in zip(j2r[0], j2k, ij2c.T):
      for i in range(6): 
        self.assertEqual(cc[i], kr2c_csr[k+i,r])

if __name__ == "__main__": unittest.main()
