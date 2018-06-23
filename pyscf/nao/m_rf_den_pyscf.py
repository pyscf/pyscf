from __future__ import print_function, division
import sys, numpy as np
from numpy import stack, dot, zeros, einsum, array
from timeit import default_timer as timer

def rf_ov_subr(self, ww):
  """ Full matrix interacting response from tdscf class in occupied-virtual indices, for a given frequency"""
  nov = self.tdscf.xy[0][0].size
  rf_ov = np.zeros([len(ww), nov], dtype=self.dtypeComplex)
  for e,xy in zip(self.tdscf.e, self.tdscf.xy):
    for iw,w in enumerate(ww):
      xpy = xy[0]+xy[1]
      rf_ov[iw] = rf_ov[iw] + xpy/(w-e)-xpy/(w+e)
  return rf_ov

def rf_den_pyscf(self, ww):
  """ Full matrix interacting response from  class"""
  assert hasattr(self, 'tdscf')
  pov = self.get_vertex_pov()[0]
  nprd = pov.shape[0]
  rf_ov = rf_ov_subr(self, ww).reshape([len(ww),pov.shape[1],pov.shape[2]])
  rf = np.zeros([len(ww),nprd,nprd], dtype=self.dtypeComplex)
  for iw, ov in enumerate(rf_ov):
    rf[iw] = np.einsum('pov,ov,qov->pq', pov, ov, pov)
  return rf


