from __future__ import print_function, division
import sys, numpy as np

class mesh_affine_equ():

  def __init__(self, **kw):
    """  
      Constructor of affine, equidistant 3d mesh class
      ucell : unit cell vectors (in coordinate space)
      Ecut  : Energy cutoff to parametrize the discretization 
    """
    from scipy.fftpack import next_fast_len
    
    self.ucell = kw['ucell'] if 'ucell' in kw else 30.0*np.eye(3) # Not even unit cells vectors are required by default
    self.Ecut = Ecut = kw['Ecut'] if 'Ecut' in kw else 50.0 # 50.0 Hartree by default
    luc = np.sqrt(np.einsum('ix,ix->i', self.ucell, self.ucell))
    self.shape = nn = np.array([next_fast_len( int(np.rint(l * np.sqrt(Ecut)/2))) for l in luc], dtype=int)
    self.size  = np.prod(self.shape)
    gc = self.ucell/(nn) # This is probable the best for finite systems, for PBC use nn, not (nn-1)
    self.dv = np.abs(np.dot(gc[0], np.cross(gc[1], gc[2] )))
    rr = [np.array([gc[i]*j for j in range(nn[i])]) for i in range(3)]
    rr[0] = rr[0].reshape((nn[0],1,1,3))
    rr[1] = rr[1].reshape((1,nn[1],1,3))
    rr[2] = rr[2].reshape((1,1,nn[2],3))
    self.rr = rr
    self.center = (rr[0][-1,0,0]+rr[1][0,-1,0]+rr[2][0,0,-1])/2.0

  def get_all_coords(self, center=0.0):
    return (self.rr[0]+self.rr[1]+self.rr[2])-self.center+center

  def get_3dgrid(self, center=0.0):
    """ Generate 3d Grid a la PySCF with .coords and .weights  fields """
    self.coords = self.get_all_coords(center=center).reshape((self.size, 3))
    self.weights = self.dv
    return self
