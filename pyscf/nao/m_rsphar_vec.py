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
  
  x = urv[:,0]
  y = urv[:,1]
  z = urv[:,2]
  
  res[:,0] = 0.5 / sqrt( pi )

  if lmax>0 :
    res[:,1] = 0.5 * sqrt( 3.0 / pi )*y
    res[:,2] = 0.5 * sqrt( 3.0 / pi )*z
    res[:,3] = 0.5 * sqrt( 3.0 / pi )*x

  if lmax>1:
    res[:,4] =  0.5  * sqrt( 15.0 / pi )*x*y
    res[:,5] =  0.5  * sqrt( 15.0 / pi )*y*z
    res[:,6] =  0.25 * sqrt( 5.0 / pi )*(3*z**2-1)
    res[:,7] =  0.5  * sqrt( 15.0 / pi )*x*z
    res[:,8] =  0.25 * sqrt( 15.0 / pi )*(x**2-y**2)

  if lmax>2:
    res[:,9]  =  0.125 * sqrt( 70.0 / pi  )*y*(3*x**2-y**2)
    res[:,10] =  0.5   * sqrt( 105.0 / pi )*x*y*z
    res[:,11] =  0.125 * sqrt( 42.0 / pi  )*y*(-1.0+5*z**2)
    res[:,12] =  0.25  * sqrt( 7.0 / pi   )*z*(5*z**2-3.0)
    res[:,13] =  0.125 * sqrt( 42.0 / pi  )*x*(-1.0+5*z**2)
    res[:,14] =  0.25  * sqrt( 105.0 / pi )*z*(x**2-y**2)
    res[:,15] =  0.125 * sqrt( 70.0 / pi  )*x*(x**2-3.0*y**2)

  if lmax>3:
    res[:,16] = 105*2**((-7.0)/2.0)*(4*x**3*y-4*x*y**3)/(sqrt(70)*sqrt(pi))
    res[:,17] = 3*2**((-5.0)/2.0)*sqrt(35)*(3*x**2*y-y**3)*z/sqrt(pi)
    res[:,18] = x*y*(105.0*z**2/2.0+(-15.0)/2.0)/(sqrt(2)*sqrt(10)*sqrt(pi))
    res[:,19] = 3*2**((-3.0)/2.0)*y*(35.0*z**3/2.0+(-15.0)*z/2.0)/(sqrt(5)*sqrt(pi))
    res[:,20] = 3.0*(35.0*z**4/8.0+(-15.0)*z**2/4.0+3.0/8.0)/(2.0*sqrt(pi))
    res[:,21] = 3*2**((-3.0)/2.0)*x*(35.0*z**3/2.0+(-15.0)*z/2.0)/(sqrt(5)*sqrt(pi))
    res[:,22] = 2**((-3.0)/2.0)*(x**2-y**2)*(105.0*z**2/2.0+(-15.0)/2.0)/(sqrt(10)*sqrt(pi))
    res[:,23] = 3*2**((-5.0)/2.0)*sqrt(35)*(x**3-3*x*y**2)*z/sqrt(pi)
    res[:,24] = 105*2**((-7.0)/2.0)*(y**4-6*x**2*y**2+x**4)/(sqrt(70)*sqrt(pi))

  if lmax>4:
    res[:,25] = 3*2**((-9.0)/2.0)*sqrt(7)*sqrt(11)*(y**5-10*x**2*y**3+5*x**4*y)/sqrt(pi)
    res[:,26] = 105*2**((-7.0)/2.0)*sqrt(11)*(4*x**3*y-4*x*y**3)*z/(sqrt(70)*sqrt(pi))
    res[:,27] = 2**((-7.0)/2.0)*sqrt(11)*(3*x**2*y-y**3)*(945.0*z**2/2.0+(-105.0)/2.0)/(sqrt(35)*sqrt(pi))/3.0
    res[:,28] = sqrt(11)*x*y*(315.0*z**3/2.0+(-105.0)*z/2.0)/(sqrt(2)*sqrt(210)*sqrt(pi))
    res[:,29] = sqrt(11)*y*(315.0*z**4/8.0+(-105.0)*z**2/4.0+15.0/8.0)/(sqrt(2)*sqrt(30)*sqrt(pi))
    res[:,30] = sqrt(11)*(63.0*z**5/8.0+(-35.0)*z**3/4.0+15.0*z/8.0)/sqrt(pi)/2.0
    res[:,31] = sqrt(11)*x*(315.0*z**4/8.0+(-105.0)*z**2/4.0+15.0/8.0)/(sqrt(2)*sqrt(30)*sqrt(pi))
    res[:,32] = 2**((-3.0)/2.0)*sqrt(11)*(x**2-y**2)*(315.0*z**3/2.0+(-105.0)*z/2.0)/(sqrt(210)*sqrt(pi))
    res[:,33] = 2**((-7.0)/2.0)*sqrt(11)*(x**3-3*x*y**2)*(945.0*z**2/2.0+(-105.0)/2.0)/(sqrt(35)*sqrt(pi))/3.0
    res[:,34] = 105*2**((-7.0)/2.0)*sqrt(11)*(y**4-6*x**2*y**2+x**4)*z/(sqrt(70)*sqrt(pi))
    res[:,35] = 3*2**((-9.0)/2.0)*sqrt(7)*sqrt(11)*(5*x*y**4-10*x**3*y**2+x**5)/sqrt(pi)

  return res
