from __future__ import print_function, division
import numpy as np

#try:
  #import numba
  #use_numba = True
#except:
  #use_numba = False

use_numba = False # This Numba solutions is slower than apply_inv_G2()

if use_numba:
  @numba.jit(nopython=True)
  def apply_inv_G2(vh, gg0, gg1, gg2):
  
    for i in range(len(gg0)):
      g0 = gg0[i]
      for j in range(len(gg1)):
        g1 = gg1[j]
        for k in range(len(gg2)):
          g2 = gg2[k]
          Gsq = ((g0+g1+g2)**2).sum()
          if abs(Gsq)<1e-14:
            vh[i,j,k] = 0.0
          else:
            vh[i,j,k] = vh[i,j,k] / Gsq
    return vh
else:

  def apply_inv_G2(vh, gg0, gg1, gg2):
    """ Only critics to this solution is the memory concerns """          
    gg0 = gg0.reshape((len(gg0),1,1,3))
    gg1 = gg1.reshape((1,len(gg1),1,3))
    gg2 = gg2.reshape((1,1,len(gg2),3))
    gg = gg0+gg1+gg2
    gg = (gg**2).sum(axis=3)
    vh = np.where(gg>1e-14, vh, 0.0)
    gg = np.where(gg>1e-14, gg, 1.0)
    return vh/gg

  def apply_inv_G2_ref(vh, gg0, gg1, gg2):
      
    for i,g0 in enumerate(gg0):
      for j,g1 in enumerate(gg1):
        for k,g2 in enumerate(gg2):
          Gsq = ((g0+g1+g2)**2).sum()
          if abs(Gsq)<1e-14:
            vh[i,j,k] = 0.0
          else:
            vh[i,j,k] = vh[i,j,k] / Gsq
    return vh
  

def vhartree_pbc(self, dens, **kw): 
  """  Compute Hartree potential for the density given in an equidistant grid  """
  from scipy.fftpack import fftn, ifftn
  sh = self.mesh3d.shape
  dens = dens.reshape(sh)
  vh = fftn(dens)    
  umom = self.ucell_mom()
  ii = [np.array([i-sh[j] if i>sh[j]//2 else i for i in range(sh[j])]) for j in range(3)]
  gg = [np.array([umom[j]*i for i in ii[j]]) for j in range(3)]
  vh = apply_inv_G2(vh, gg[0], gg[1], gg[2])
  vh = ifftn(vh).real*(4*np.pi)
  return vh
