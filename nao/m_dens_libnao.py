from __future__ import print_function, division
import numpy as np

#
#
#
def dens_libnao(crds):
  """
    Compute the values of atomic orbitals on given grid points
    Args:
      crds   : vector where the atomic orbitals from "ao" are centered
    Returns:
      res[ncoord] : array of density
  """
  
  return np.zeros_like(crds)
