from __future__ import print_function, division
import sys, numpy as np
from numpy import complex128, zeros

def sf2f_rf(ff, eps, w2f, wab2sf):
  from pyscf.nao.m_lorentzian import lorentzian
  """ Get a function F(w) from it's spectral function A(w) in case of response-like function/spectral function
      ff -- frequencies list or array at which the function F(w) needs to be computed.
      eps -- broadening
      w2f -- frequencies at which the spectral function is given
      wab2sf -- the collection of spectral functions
  """
  f2f = zeros([len(ff)]+list(wab2sf.shape[1:]), dtype=complex128)
  for i,f in enumerate(ff):
    for w,fp in enumerate(w2f):
      f2f[i] += wab2sf[w]*lorentzian(f, fp, eps)

  return f2f

