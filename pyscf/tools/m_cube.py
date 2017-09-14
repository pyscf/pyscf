import numpy
import time
import pyscf
from pyscf import lib


#
#
#
class cube_c():
  '''
  Read-write of the Gaussian CUBE files  
  '''
  def __init__(self, mol, nx, ny, nz):
    coord = mol.atom_coords()
    box = numpy.max(coord,axis=0) - numpy.min(coord,axis=0) + 6
    boxorig = numpy.min(coord,axis=0) - 3
    xs = numpy.arange(nx) * (box[0]/nx)
    ys = numpy.arange(ny) * (box[1]/ny)
    zs = numpy.arange(nz) * (box[2]/nz)
    
  def get_coords(self) :
    """
    Result: set of coordinates to compute a field which is to be stored in the file.
    """
    coords = lib.cartesian_prod([xs,ys,zs])
    coords = numpy.asarray(coords, order='C') - (-boxorig)
    return coords
  
  def dump(self, field):
    """
    Result: set of coordinates to compute a field which is to be stored in the file.
    """
    with open(outfile, 'w') as f:
      f.write('Electron density in real space (e/Bohr^3)\n')
      f.write('PySCF Version: %s  Date: %s\n' % (pyscf.__version__, time.ctime()))
      f.write('%5d' % mol.natm)
      f.write('%12.6f%12.6f%12.6f\n' % tuple(boxorig.tolist()))
      f.write('%5d%12.6f%12.6f%12.6f\n' % (nx, xs[1], 0, 0))
      f.write('%5d%12.6f%12.6f%12.6f\n' % (ny, 0, ys[1], 0))
      f.write('%5d%12.6f%12.6f%12.6f\n' % (nz, 0, 0, zs[1]))
      for ia in range(mol.natm):
        chg = mol.atom_charge(ia)
        f.write('%5d%12.6f'% (chg, chg))
        f.write('%12.6f%12.6f%12.6f\n' % tuple(coord[ia]))

      for ix in range(nx):
        for iy in range(ny):
          for iz in range(0,nz,6):
            remainder  = (nz-iz)
            if (remainder > 6 ):
              fmt = '%13.5E' * 6 + '\n'
              f.write(fmt % tuple(rho[ix,iy,iz:iz+6].tolist()))
            else:
              fmt = '%13.5E' * remainder + '\n'
              f.write(fmt % tuple(rho[ix,iy,iz:iz+remainder].tolist()))
              break

