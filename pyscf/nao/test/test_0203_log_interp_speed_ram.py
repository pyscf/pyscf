from __future__ import print_function, division
import unittest, numpy as np

from pyscf.nao.log_mesh import funct_log_mesh
from pyscf.nao.m_log_interp import log_interp_c
from timeit import default_timer as timer
from scipy.sparse import csr_matrix

class KnowValues(unittest.TestCase):


  def test_log_interp_vv_speed_and_space(self):
    """ Test the interpolation facility for an array arguments from the class log_interp_c """
    rr,pp = funct_log_mesh(1024, 0.01, 200.0)
    lgi = log_interp_c(rr)

    gcs = np.array([1.2030, 3.2030, 0.7, 10.0, 5.3])
    ff = np.array([[np.exp(-gc*r**2) for r in rr] for gc in gcs])

    rrs = np.linspace(0.05, 250.0, 2000000)
    t1 = timer()
    fr2yy1 = lgi.interp_csr(ff, rrs, rcut=16.0)
    t2 = timer()
    
    #print(__name__, 't1: ', t2-t1)
    #print(fr2yy1.shape, fr2yy1.size)
    yyref = np.exp(-(gcs.reshape(gcs.size,1)) * (rrs.reshape(1,rrs.size)**2))
  
    self.assertTrue(np.allclose(fr2yy1.toarray(), yyref) )    

if __name__ == "__main__": unittest.main()
