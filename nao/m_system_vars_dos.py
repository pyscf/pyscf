from __future__ import print_function, division
import numpy as np
from numpy import zeros_like 


def eigenvalues2dos(ksn2e, zomegas):
  """ Compute the Density of States using the eigenvalues """
  zdos = zeros_like(zomegas)
  for iw,zw in enumerate(zomegas):
    zdos[iw] = (1.0/(zw - ksn2e)).sum()
    
  return zdos.imag
  
