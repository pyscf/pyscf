from __future__ import division, print_function
import numpy as np
from numpy import pi, sqrt

#
def rsphar_vec(rvecs,lmax):
  """
    Computes (all) real spherical harmonics up to the angular momentum lmax
    Args:
      rvecs : A list of Cartesian coordinates defining the theta and phi angles for spherical harmonic
      lmax  : Integer, maximal angular momentum
    Result:
      2-d numpy array of float64 elements with all spherical harmonics stored in order 0,0; 1,-1; 1,0; 1,+1 ... lmax,lmax, althogether 0 : (lmax+1)**2 elements.
  """
  assert lmax>-1
  
  norms = np.linalg.norm(rvecs, axis=1)
  urv = 1.0*rvecs
  urv[norms>0] /= norms[norms>0, None]
  

  res = np.zeros((norms.size, (lmax+1)**2))
  
  xx = urv[:,0]
  yy = urv[:,1]
  zz = urv[:,2]
  
  res[:,0] = 0.5 / sqrt( pi )

  if lmax>0 :
    res[:,1] = 0.5 * sqrt( 3.0 / pi )*yy
    res[:,2] = 0.5 * sqrt( 3.0 / pi )*zz
    res[:,3] = 0.5 * sqrt( 3.0 / pi )*xx

  if lmax>1:
    res[:,4] =  0.5  * sqrt( 15.0 / pi )*xx*yy
    res[:,5] =  0.5  * sqrt( 15.0 / pi )*yy*zz
    res[:,6] =  0.25 * sqrt( 5.0 / pi )*(3*zz**2-1)
    res[:,7] =  0.5  * sqrt( 15.0 / pi )*xx*zz
    res[:,8] =  0.25 * sqrt( 15.0 / pi )*(xx**2-yy**2)

  if lmax>2:
    res[:,9]  =  0.125 * sqrt( 70.0 / pi  )*yy*(3*xx**2-yy**2)
    res[:,10] =  0.5   * sqrt( 105.0 / pi )*xx*yy*zz
    res[:,11] =  0.125 * sqrt( 42.0 / pi  )*yy*(-1.0+5*zz**2)
    res[:,12] =  0.25  * sqrt( 7.0 / pi   )*zz*(5*zz**2-3.0)
    res[:,13] =  0.125 * sqrt( 42.0 / pi  )*xx*(-1.0+5*zz**2)
    res[:,14] =  0.25  * sqrt( 105.0 / pi )*zz*(xx**2-yy**2)
    res[:,15] =  0.125 * sqrt( 70.0 / pi  )*xx*(xx**2-3.0*yy**2)

  return res
