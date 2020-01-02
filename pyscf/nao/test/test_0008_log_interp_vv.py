from __future__ import print_function, division
import unittest, numpy as np

from pyscf.nao.log_mesh import funct_log_mesh
from pyscf.nao.m_log_interp import log_interp_c

class KnowValues(unittest.TestCase):

  def test_log_interp_vv(self):
    """ Test the interpolation facility for an array arguments from the class log_interp_c """
    rr,pp = funct_log_mesh(256, 0.01, 20.0)
    lgi = log_interp_c(rr)
    gcs = np.array([1.2030, 3.2030, 0.7, 10.0])
    ff = np.array([[np.exp(-gc*r**2) for r in rr] for gc in gcs])
    rvecs = np.linspace(0.05, 25.0, 200)
        
    fr2yy = lgi.interp_vv(ff, rvecs)
    yy_vv1 = np.zeros((len(ff),len(rvecs)))
    for ir, r in enumerate(rvecs):
      yyref = np.exp(-gcs*r**2)
      yy = yy_vv1[:,ir] = lgi.interp_vv(ff, r)
      for y1,yref,y2 in zip(yy, yyref,fr2yy[:,ir]):
        self.assertAlmostEqual(y1,yref)
        self.assertAlmostEqual(y2,yref)    


  def test_log_interp_coeffs_vec(self):
    """ Test the interpolation facility for an array arguments from the class log_interp_c """
    rr,pp = funct_log_mesh(1024, 0.01, 20.0)
    lgi = log_interp_c(rr)
    rvecs = np.linspace(0.00, 25.0, 20)    
    kk1,cc1 = np.zeros(len(rvecs), dtype=np.int32), np.zeros((6,len(rvecs)))
    for i,rv in enumerate(rvecs): kk1[i],cc1[:,i] = lgi.coeffs(rv)
    kk2,cc2 = lgi.coeffs_vv(rvecs)
    for k1,c1,k2,c2 in zip(kk1,cc1,kk2,cc2):
      self.assertEqual(k1,k2)
      for y1,y2 in zip(c1,c2):
        self.assertAlmostEqual(y1,y2)
  
  def test_log_interp_vv_call(self):
    """ Test the interpolation facility for an array arguments from the class log_interp_c """
    rr,pp = funct_log_mesh(256, 0.01, 20.0)
    lgi = log_interp_c(rr)
    gcs = np.array([1.2030, 3.2030, 0.7, 10.0])
    ff = np.array([[np.exp(-gc*r**2) for r in rr] for gc in gcs])
    rvecs = np.linspace(0.05, 25.0, 200)
        
    fr2yy = lgi(ff, rvecs)
    yy_vv1 = np.zeros((len(ff),len(rvecs)))
    for ir, r in enumerate(rvecs):
      yyref = np.exp(-gcs*r**2)
      yy = yy_vv1[:,ir] = lgi(ff, r)
      for y1,yref,y2 in zip(yy, yyref,fr2yy[:,ir]):
        self.assertAlmostEqual(y1,yref)
        self.assertAlmostEqual(y2,yref)    

  def test_log_interp_sv_call(self):
    """ Test the interpolation facility for an array arguments from the class log_interp_c """
    rr,pp = funct_log_mesh(256, 0.01, 20.0)
    lgi = log_interp_c(rr)
    gc = 1.2030
    ff = np.array([np.exp(-gc*r**2) for r in rr])
    rvecs = np.linspace(0.05, 25.0, 200)
        
    r2yy = lgi(ff, rvecs)
    for ir, r in enumerate(rvecs):
      yref = np.exp(-gc*r**2)
      y1 = lgi(ff, r)
      self.assertAlmostEqual(y1,yref)
      self.assertAlmostEqual(r2yy[ir],yref)    

    
if __name__ == "__main__": unittest.main()
