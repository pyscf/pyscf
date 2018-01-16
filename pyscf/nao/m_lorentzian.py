from __future__ import print_function, division
import sys, numpy as np
from numpy import complex128, zeros

def lorentzian(x, w, e):
  """ An elementary Lorentzian """
  return 1.0/(x-w+1j*e) - 1.0/(x+w-1j*e);

def llc_real(x, w1, w2, e1, e2):
  """ Product of two Lorentzians one of which is conjugated """
  return (lorentzian(x, w1, e1)*np.conj(lorentzian(x, w2, e2))).real

def llc_imag(x, w1, w2, e1, e2):
  """ Product of two Lorentzians one of which is conjugated """
  res = (lorentzian(x, w1, e1)*np.conj(lorentzian(x, w2, e2))).imag
  return res


def overlap(ww, eps):
  """ Overlap matrix between a set of Lorentzians using numerical integration """
  from scipy.integrate import quad 
  n = len(ww)
  mat = zeros((n,n), dtype=np.complex128)
  for i,w1 in enumerate(ww):
    for j,w2 in enumerate(ww):
      re = quad(llc_real, -np.inf, np.inf, args=(w1,w2,eps,eps))
      im = quad(llc_imag, -np.inf, np.inf, args=(w1,w2,eps,eps))
      mat[i,j] = re[0]+1j*im[0]
  return mat
