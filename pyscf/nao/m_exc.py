from __future__ import print_function, division
import numpy as np
from pyscf.dft import libxc

#
#
#
def exc(sv, dm, xc_code, **kvargs):
  """
    Computes the exchange-correlation energy for a given density matrix
    Args:
      sv : (System Variables), this must have arrays of coordinates and species, etc
      xc_code : is a string must comply with pySCF's convention PZ  
        "LDA,PZ"
        "0.8*LDA+0.2*B88,PZ"
    Returns:
      exc x+c energy
  """
  grid = sv.build_3dgrid_pp(**kvargs)
  dens = sv.dens_elec(grid.coords, dm)
  exc, vxc, fxc, kxc = libxc.eval_xc(xc_code, dens.T, spin=sv.nspin-1, deriv=0)
  nelec = np.dot(dens[:,0]*exc, grid.weights)
  return nelec
