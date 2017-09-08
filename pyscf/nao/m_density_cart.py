from __future__ import print_function, division
import numpy as np

#
#
#
def density_cart(crds):
  """
    Compute the values of atomic orbitals on given grid points
    Args:
      sv     : instance of system_vars_c class
      crds   : vector where the atomic orbitals from "ao" are centered
      sab2dm : density matrix
    Returns:
      res[ncoord] : array of density
  """
  
  return np.zeros_like(crds)
