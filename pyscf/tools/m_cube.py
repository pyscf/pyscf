from __future__ import print_function, division
import numpy
import time
import pyscf
from pyscf import lib

#
#
#
class cube_c():
  '''  Read-write of the Gaussian CUBE files  '''
  def __init__(self, mol, nx=80, ny=80, nz=80):
    self.nx = nx
    self.ny = ny
    self.nz = nz
    self.mol = mol
    coord = mol.atom_coords()
    self.box = numpy.max(coord,axis=0) - numpy.min(coord,axis=0) + 6.0
    self.boxorig = numpy.min(coord,axis=0) - 3.0
    self.xs = numpy.arange(nx) * (self.box[0]/nx)
    self.ys = numpy.arange(ny) * (self.box[1]/ny)
    self.zs = numpy.arange(nz) * (self.box[2]/nz)
    
  def get_coords(self) :
    """  Result: set of coordinates to compute a field which is to be stored in the file.  """
    coords = lib.cartesian_prod([self.xs,self.ys,self.zs])
    coords = numpy.asarray(coords, order='C') - (-self.boxorig)
    return coords
  
  def get_ngrids(self):
    return self.nx * self.ny * self.nz

  def get_volume_element(self):
    return (self.xs[1]-self.xs[0])*(self.ys[1]-self.ys[0])*(self.zs[1]-self.zs[0])
    
  def write(self, field, fname, comment='Generic field? Supply the optional argument "comment" to define this line'):
    """  Result: .cube file with the field in the file fname.  """
    assert len(field.shape)==3
    assert field.shape==(self.nx,self.ny,self.nz)
    
    mol = self.mol
    coord = mol.atom_coords()
    with open(fname, 'w') as f:
      f.write(comment+'\n')
      f.write('PySCF Version: %s  Date: %s\n' % (pyscf.__version__, time.ctime()))
      f.write('%5d' % mol.natm)
      f.write('%12.6f%12.6f%12.6f\n' % tuple(self.boxorig.tolist()))
      f.write('%5d%12.6f%12.6f%12.6f\n' % (self.nx, self.xs[1], 0, 0))
      f.write('%5d%12.6f%12.6f%12.6f\n' % (self.ny, 0, self.ys[1], 0))
      f.write('%5d%12.6f%12.6f%12.6f\n' % (self.nz, 0, 0, self.zs[1]))
      for ia in range(mol.natm):
        chg = mol.atom_charge(ia)
        f.write('%5d%12.6f'% (chg, chg))
        f.write('%12.6f%12.6f%12.6f\n' % tuple(coord[ia]))

      for ix in range(self.nx):
        for iy in range(self.ny):
          for iz in range(0,self.nz,6):
            remainder  = (self.nz-iz)
            if (remainder > 6 ):
              fmt = '%13.5E' * 6 + '\n'
              f.write(fmt % tuple(field[ix,iy,iz:iz+6].tolist()))
            else:
              fmt = '%13.5E' * remainder + '\n'
              f.write(fmt % tuple(field[ix,iy,iz:iz+remainder].tolist()))
              break

if __name__ == '__main__':
    from pyscf import gto, scf
    from pyscf.tools.m_cube import cube_c
    from pyscf.dft import numint, gen_grid
    
    import numpy
    mol = gto.M(atom='O 0.00000000,  0.000000,  0.000000; H 0.761561, 0.478993, 0.00000000,; H -0.761561, 0.478993, 0.00000000,', basis='6-31g*')

    mf = scf.RHF(mol)
    mf.scf()
    dm = mf.make_rdm1()

    cc = cube_c(mol, nx=20, ny=20, nz=20)  # Initialize the class cube_c
    
    # Compute density on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    rho = numpy.empty(ngrids)
    ao = None
    for ip0, ip1 in gen_grid.prange(0, ngrids, blksize):
      ao = numint.eval_ao(mol, coords[ip0:ip1], out=ao)
      rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)
    rho = rho.reshape(cc.nx,cc.ny,cc.nz)
    
    fname = "h2o_den_cube_c.cube"
    cc.write(rho, fname, comment='Electron density in real space (e/Bohr^3)') # Write out density to the .cube file
    print(fname, ' is written.')
