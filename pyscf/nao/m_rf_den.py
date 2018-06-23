from __future__ import print_function, division
import sys, numpy as np
from numpy import stack, dot, zeros, einsum, array
from timeit import default_timer as timer

def rf_den_via_rf0(self, rf0, v):
  """ Full matrix interacting response via non-interacting response and interaction"""
  rf = np.zeros_like(rf0)
  I  = np.identity(rf0.shape[1])
  for ir,r in enumerate(rf0):
    rf[ir] = np.linalg.inv(I-r*v)*r
  return rf


def rf_den(self, ww):
  """ Full matrix interacting response from NAO GW class"""
  rf0 = self.rf0(ww)
  return rf_den_via_rf0(self, rf0, self.kernel_sq)


def rf_den_pyscf(self, ww):
  """ Full matrix interacting response from NAO GW class"""
  rf0 = self.rf0(ww)
  return rf_den_via_rf0(self, rf0, self.kernel_sq)
