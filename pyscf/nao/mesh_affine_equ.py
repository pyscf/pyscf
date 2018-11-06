from __future__ import print_function, division
import numpy as np

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
    self.rr = rr
    self.origin = kw['origin'] if 'origin' in kw else np.zeros(3)

  def get_mesh_center(self):
    return (self.rr[0][-1]+self.rr[1][-1]+self.rr[2][-1])/2.0
  
  def get_all_coords(self):
    rr0 = self.rr[0].reshape((self.shape[0],1,1,3))
    rr1 = self.rr[1].reshape((1,self.shape[1],1,3))
    rr2 = self.rr[2].reshape((1,1,self.shape[2],3))
    return (rr0+rr1+rr2)-self.get_mesh_center()+self.origin

  def get_3dgrid(self):
    """ Generate 3d Grid a la PySCF with .coords and .weights  fields """
    self.coords = self.get_all_coords().reshape((self.size, 3))
    self.weights = self.dv
    return self

  def write(self, fname, **kw):
    import time
    import pyscf
    from pyscf import lib
    """  Result: .cube file with the field in the file fname.  """
    
    if 'mol' in kw:
      mol = kw['mol'] # Obligatory argument
      coord = mol.atom_coords()
      zz = [mol.atom_charge(ia) for ia in range(mol.natm)]
      natm = mol.natm
    else:
      zz = kw['a2z']
      natm = len(zz)
      coord = kw['a2xyz']

    field = kw['field'] # Obligatory argument?
    
    comment = kw['comment'] if 'comment' in kw else 'none'
    
    with open(fname, 'w') as f:
      f.write(comment+'\n')
      f.write('PySCF Version: {:s}  Date: {:s}\n'.format(pyscf.__version__, time.ctime()))
      bo = self.origin-self.get_mesh_center()
      f.write(('{:5d}'+'{:12.6f}'*3+'\n').format(natm, *bo))

      for s,rr in zip(self.shape, self.rr): 
        f.write(('{:5d}'+'{:12.6f}'*3+'\n').format(s, *rr[1]))

      for chg,xyz in zip(zz,coord): 
        f.write(('{:5d}'+'{:12.6f}'*4+'\n').format(chg, chg, *xyz))

      for ix in range(self.shape[0]):
        for iy in range(self.shape[1]):
          for iz0,iz1 in lib.prange(0, self.shape[2], 6):
            f.write(('{:13.5e}' * (iz1-iz0) + '\n').format(*field[ix,iy,iz0:iz1]))
