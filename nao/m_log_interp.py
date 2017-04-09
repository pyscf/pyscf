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
  
  #print(r, dy, k, rho_min_jt, dr_jt)
  
  #sys.exit(0)

  fv = (-dy*(dy**2-1.0)*(dy-2.0)*(dy-3.0)*ff[k-2] 
       +5.0*dy*(dy-1.0)*(dy**2-4.0)*(dy-3.0)*ff[k-1] 
       -10.0*(dy**2-1.0)*(dy**2-4.0)*(dy-3.0)*ff[k]
       +10.0*dy*(dy+1.0)*(dy**2-4.0)*(dy-3.0)*ff[k+1]
       -5.0*dy*(dy**2-1.0)*(dy+2.0)*(dy-3.0)*ff[k+2]
       +dy*(dy**2-1.0)*(dy**2-4.0)*ff[k+3])/120.0 

  return fv
