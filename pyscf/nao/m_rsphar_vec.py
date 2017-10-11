import numpy as np
from pyscf.symm import sph

#
def rsphar_vec(rvecs, lmax):
    """
    Computes (all) real spherical harmonics up to the angular momentum lmax
    Args:
      rvecs : A list of Cartesian coordinates defining the theta and phi angles for spherical harmonic
      lmax  : Integer, maximal angular momentum
    Result:
      2-d numpy array of float64 elements with all spherical harmonics stored in order 0,0; 1,-1; 1,0; 1,+1 ... lmax,lmax, althogether 0 : (lmax+1)**2 elements.
    """
    assert lmax>-1
    ylm = sph.real_sph_vec(rvecs, lmax)
    res = np.vstack(ylm).T.copy('C')
    return res

