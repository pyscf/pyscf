# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

#
#
#
def log_interp(ff, r, rho_min_jt, dr_jt):
  """
    Interpolation of a function given on the logarithmic mesh (see m_log_mesh how this is defined)
    6-point interpolation on the exponential mesh (James Talman)
    Args:
      ff : function values to be interpolated
      r  : radial coordinate for which we want intepolated value
      rho_min_jt : log(rr[0]), i.e. logarithm of minimal coordinate in the logarithmic mesh
      dr_jt : log(rr[1]/rr[0]) logarithmic step of the grid
    Result: 
      Interpolated value

    Example:
      nr = 1024
      rr,pp = log_mesh(nr, rmin, rmax, kmax)
      rho_min, dr = log(rr[0]), log(rr[1]/rr[0])
      y = interp_log(ff, 0.2, rho, dr)
  """
  if r<=0.0: return ff[0]

  lr = np.log(r)
  k=int((lr-rho_min_jt)/dr_jt)
  nr = len(ff)
  k = min(max(k,2), nr-4)
  dy=(lr-rho_min_jt-k*dr_jt)/dr_jt

  fv = (-dy*(dy**2-1.0)*(dy-2.0)*(dy-3.0)*ff[k-2] 
       +5.0*dy*(dy-1.0)*(dy**2-4.0)*(dy-3.0)*ff[k-1] 
       -10.0*(dy**2-1.0)*(dy**2-4.0)*(dy-3.0)*ff[k]
       +10.0*dy*(dy+1.0)*(dy**2-4.0)*(dy-3.0)*ff[k+1]
       -5.0*dy*(dy**2-1.0)*(dy+2.0)*(dy-3.0)*ff[k+2]
       +dy*(dy**2-1.0)*(dy**2-4.0)*ff[k+3])/120.0 

  return fv

#
#
#
def comp_coeffs_(self, r, i2coeff):
  """
    Interpolation of a function given on the logarithmic mesh (see m_log_mesh how this is defined)
    6-point interpolation on the exponential mesh (James Talman)
    Args:
      r  : radial coordinate for which we want the intepolated value
    Result: 
      Array of weights to sum with the functions values to obtain the interpolated value coeff
      and the index k where summation starts sum(ff[k:k+6]*coeffs)
  """
  if r<=0.0:
    i2coeff.fill(0.0)
    i2coeff[0] = 1
    return 0

  lr = np.log(r)
  k  = int((lr-self.gammin_jt)/self.dg_jt)
  k  = min(max(k,2), self.nr-4)
  dy = (lr-self.gammin_jt-k*self.dg_jt)/self.dg_jt
  
  i2coeff[0] =     -dy*(dy**2-1.0)*(dy-2.0)*(dy-3.0)/120.0
  i2coeff[1] = +5.0*dy*(dy-1.0)*(dy**2-4.0)*(dy-3.0)/120.0
  i2coeff[2] = -10.0*(dy**2-1.0)*(dy**2-4.0)*(dy-3.0)/120.0
  i2coeff[3] = +10.0*dy*(dy+1.0)*(dy**2-4.0)*(dy-3.0)/120.0
  i2coeff[4] = -5.0*dy*(dy**2-1.0)*(dy+2.0)*(dy-3.0)/120.0
  i2coeff[5] =      dy*(dy**2-1.0)*(dy**2-4.0)/120.0

  return k-2

#
#
#
def comp_coeffs(self, r):
  i2coeff = np.zeros(6)
  k = comp_coeffs_(self, r, i2coeff)
  return k,i2coeff


class log_interp_c():
  """    Interpolation of radial orbitals given on a log grid (m_log_mesh)  """
  def __init__(self, gg):
    """
      gg: one-dimensional array defining a logarithmic grid
    """
    #assert(type(rr)==np.ndarray)
    assert(len(gg)>2)
    self.gg = gg
    self.nr = len(gg)
    self.gammin_jt = np.log(gg[0])
    self.dg_jt = np.log(gg[1]/gg[0])

  def __call__(self, ff, r):
    """ Interpolation right away """
    assert ff.shape[-1]==self.nr
    k,cc = comp_coeffs(self, r)
    result = np.zeros(ff.shape[0:-2])
    for j,c in enumerate(cc): result = result + c*ff[...,j+k]
    return result
    
  comp_coeffs=comp_coeffs
  """ Interpolation right away """
  
  def diff(self, za):
    """
      Return array with differential 
      za :  input array to be differentiated 
    """
    ar = self.gg
    dr = self.dg_jt
    nr = self.nr
    zb = np.zeros_like(za)
        
    zb[0]=(za[0]-za[1])/(ar[0]-ar[1]) # forward to improve
    zb[1]=(za[2]-za[0])/(ar[2]-ar[0]) # central? to improve
    zb[2]=(za[3]-za[1])/(ar[3]-ar[1]) # central? to improve 
    
    for i in range(3,nr-3):
      zb[i]=(45.0*(za[i+1]-za[i-1])-9.0*(za[i+2]-za[i-2])+za[i+3]-za[i-3])/(60.0*dr*ar[i])
    
    zb[nr-3]=(za[nr-1]-za[nr-5]+8.0*(za[nr-2]-za[nr-4]))/ ( 12.0*self.dg_jt*self.gg[nr-3] )
    zb[nr-2]=(za[nr-1]-za[nr-3])/(2.0*dr*ar[nr-2])
    zb[nr-1]=( 4.0*za[nr-1]-3.0*za[nr-2]+za[nr-3])/(2.0*dr*ar[nr-1] );
    return zb
    
#    Example:
#      loginterp =log_interp_c(rr)

if __name__ == '__main__':
  from pyscf.nao.m_log_interp import log_interp, log_interp_c, comp_coeffs_
  from pyscf.nao.m_log_mesh import log_mesh
  rr,pp = log_mesh(1024, 0.01, 20.0)
  interp_c = log_interp_c(rr)
  gc = 0.234450
  ff = np.array([np.exp(-gc*r**2) for r in rr])
  rho_min_jt, dr_jt = np.log(rr[0]), np.log(rr[1]/rr[0]) 
  for r in np.linspace(0.01, 25.0, 100):
    yref = log_interp(ff, r, rho_min_jt, dr_jt)
    k,coeffs = comp_coeffs(interp_c, r)
    y = sum(coeffs*ff[k:k+6])
    if(abs(y-yref)>1e-15): print(r, yref, y, np.exp(-gc*r**2))
