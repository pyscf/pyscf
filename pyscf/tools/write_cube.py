import pyscf
from pyscf import lib
from pyscf import gto

def write_cube(cell, field, mesh, fname, origin=[0., 0., 0.], comment="Cube file"):
    coord = cell.atom_coords()
    a = cell.lattice_vectors()
    nx, ny, nz = mesh
    dx = a[0] / nx
    dy = a[1] / ny
    dz = a[2] / nz
    with open(fname, 'w') as f:
        f.write(comment+'\n')
        f.write(f'PySCF Version: {pyscf.__version__}  Date: {time.ctime()}\n')
        f.write(f'{cell.natm:5d}')
        f.write('%12.6f%12.6f%12.6f\n' % tuple(origin))
        dx = self.xs[-1] if len(self.xs) == 1 else self.xs[1]
        dy = self.ys[-1] if len(self.ys) == 1 else self.ys[1]
        dz = self.zs[-1] if len(self.zs) == 1 else self.zs[1]
        delta = (self.box.T * [dx,dy,dz]).T
        f.write(f'{self.nx:5d}{dx[0]:12.6f}{dx[1]:12.6f}{dx[2]:12.6f}\n')
        f.write(f'{self.ny:5d}{dy[0]:12.6f}{dy[1]:12.6f}{dy[2]:12.6f}\n')
        f.write(f'{self.nz:5d}{dz[0]:12.6f}{dz[1]:12.6f}{dz[2]:12.6f}\n')
        for ia in range(cell.natm):
            atmsymb = cell.atom_symbol(ia)
            f.write('%5d%12.6f'% (gto.charge(atmsymb), 0.))
            f.write('%12.6f%12.6f%12.6f\n' % tuple(coord[ia]))

        for ix in range(nx):
            for iy in range(ny):
                for iz0, iz1 in lib.prange(0, nz, 6):
                    fmt = '%13.5E' * (iz1-iz0) + '\n'
                    f.write(fmt % tuple(field[ix,iy,iz0:iz1].tolist()))

