from __future__ import division
import numpy as np
from pyscf.nao.m_fact import sgn, onedivsqrt4pi

#
#
#
def csphar(r,lmax):
  """
    Computes (all) complex spherical harmonics up to the angular momentum lmax
    
    Args:
      r : Cartesian coordinates defining correct theta and phi angles for spherical harmonic
      lmax : Integer, maximal angular momentum
    Result:
      1-d numpy array of complex128 elements with all spherical harmonics stored in order 0,0; 1,-1; 1,0; 1,+1 ... lmax,lmax, althogether 0 : (lmax+1)**2 elements.
  """
  x=r[0]
  y=r[1] 
  z=r[2] 
  dd=np.sqrt(x*x+y*y+z*z)
  res = np.zeros(((lmax+1)**2), dtype=np.complex128)

  res[0] = onedivsqrt4pi

  if dd < 1.0e-10 :
     ll=(lmax+1)**2
     return res

  if x == 0.0 :
    phi=0.5*np.pi
    if y<0.0: phi=-phi
  else:
    phi = np.arctan( y/x ) 
    if x<0.0: phi=phi+np.pi 

  ss=np.sqrt(x*x+y*y)/dd
  cc=z/dd
  
  if lmax<1 : return res
  
  for l in range(1,lmax+1):
     al=1.0*l 
     il2=(l+1)**2-1 
     il1=l**2-1
     res[il2] = -ss*np.sqrt((al-0.5)/al)*res[il1] 
     res[il2-1] = cc*np.sqrt(2.0*al-1.0)*res[il1]

  if lmax>1:
    for m in range(lmax-1):
      if m<lmax:
        for l in range(m+1,lmax):
          ind=l*(l+1)+m 
          aa=1.0*(l**2-m**2)
          bb=1.0*((l+1)**2-m**2)
          zz=(l+l+1.0)*cc*res[ind].real-np.sqrt(aa)*res[ind-2*l].real 
          res[ind+2*(l+1)]=zz/np.sqrt(bb) 

  for l in range(lmax+1):
     ll2=l*(l+1)
     rt2lp1=np.sqrt(l+l+1.0)
     for m in range(l+1):
        cs=np.sin(m*phi)*rt2lp1
        cc=np.cos(m*phi)*rt2lp1
        res[ll2+m]=np.complex(cc,cs)*res[ll2+m]
        res[ll2-m]=sgn[m]*np.conj(res[ll2+m])
  
  return res;
  
  
