from __future__ import print_function, division
import numpy as np
from numpy import identity, dot, zeros, zeros_like

def rf_den_via_rf0(self, rf0, v):
  """ Whole matrix of the interacting response via non-interacting response and interaction"""
  rf = zeros_like(rf0)
  I  = identity(rf0.shape[1])
  for ir,r in enumerate(rf0):
    rf[ir] = dot(np.linalg.inv(I-dot(r,v)), r)
  return rf


def rf_den(self, ww):
  """ Full matrix interacting response from NAO GW class"""
  rf0 = self.rf0(ww)
  return rf_den_via_rf0(self, rf0, self.kernel_sq)

