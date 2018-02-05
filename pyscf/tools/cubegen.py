#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Gaussian cube file format.  Reference:
http://paulbourke.net/dataformats/cube/
http://gaussian.com/cubegen/

The output cube file has the following format

Comment line
Comment line
N_atom Ox Oy Oz         # number of atoms, followed by the coordinates of the origin
N1 vx1 vy1 vz1          # number of grids along each axis, followed by the step size in x/y/z direction.
N2 vx2 vy2 vz2          # ...
N3 vx3 vy3 vz3          # ...
Atom1 Z1 x y z          # Atomic number, charge, and coordinates of the atom
...                     # ...
AtomN ZN x y z          # ...
Data on grids           # (N1*N2) lines of records, each line has N3 elements
'''

import numpy
import time
import pyscf
from pyscf import lib
from pyscf.dft import numint, gen_grid

def density(mol, outfile, dm, nx=80, ny=80, nz=80):
    """Calculates electron density.

    Args:
        mol (Mole): Molecule to calculate the electron density for.
        outfile (str): Name of Cube file to be written.
        dm (str): Density matrix of molecule.
        nx (int): Number of grid point divisions in x direction.
           Note this is function of the molecule's size; a larger molecule
           will have a coarser representation than a smaller one for the
           same value.
        ny (int): Number of grid point divisions in y direction.
        nz (int): Number of grid point divisions in z direction.


    """

    coord = mol.atom_coords()
    box = numpy.max(coord,axis=0) - numpy.min(coord,axis=0) + 6
    boxorig = numpy.min(coord,axis=0) - 3
    # .../(nx-1) to get symmetric mesh
    # see also the discussion on https://github.com/sunqm/pyscf/issues/154
    xs = numpy.arange(nx) * (box[0] / (nx - 1))
    ys = numpy.arange(ny) * (box[1] / (ny - 1))
    zs = numpy.arange(nz) * (box[2] / (nz - 1))
    coords = lib.cartesian_prod([xs,ys,zs])
    coords = numpy.asarray(coords, order='C') - (-boxorig)

    ngrids = nx * ny * nz
    blksize = min(8000, ngrids)
    rho = numpy.empty(ngrids)
    ao = None
    for ip0, ip1 in gen_grid.prange(0, ngrids, blksize):
        ao = numint.eval_ao(mol, coords[ip0:ip1], out=ao)
        rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)
    rho = rho.reshape(nx,ny,nz)

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

def mep(mol, outfile, dm, nx=80, ny=80, nz=80):
    """Calculates the molecular electrostatic potential (MEP).

    Args:
        mol (Mole): Molecule to calculate the MEP for.
        outfile (str): Name of Cube file to be written.
        dm (str): Density matrix of molecule.
        nx (int): Number of grid point divisions in x direction.
           Note this is function of the molecule's size; a larger molecule
           will have a coarser representation than a smaller one for the
           same value.
        ny (int): Number of grid point divisions in y direction.
        nz (int): Number of grid point divisions in z direction.


    """

    coord = mol.atom_coords()
    box = numpy.max(coord,axis=0) - numpy.min(coord,axis=0) + 6
    boxorig = numpy.min(coord,axis=0) - 3
    # .../(nx-1) to get symmetric mesh
    # see also the discussion on https://github.com/sunqm/pyscf/issues/154
    xs = numpy.arange(nx) * (box[0] / (nx - 1))
    ys = numpy.arange(ny) * (box[1] / (ny - 1))
    zs = numpy.arange(nz) * (box[2] / (nz - 1))
    coords = lib.cartesian_prod([xs,ys,zs])
    coords = numpy.asarray(coords, order='C') - (-boxorig)

    # Nuclear potential at given points
    Vnuc = 0
    for i in range(mol.natm):
       r = mol.atom_coord(i)
       Z = mol.atom_charge(i)
       rp = r - coords
       Vnuc += Z / numpy.einsum('xi,xi->x', rp, rp)**.5

    # Potential of electron density
    Vele = []
    for p in coords:
        mol.set_rinv_orig_(p)
        Vele.append(numpy.einsum('ij,ij', mol.intor('cint1e_rinv_sph'), dm))

    # MEP at each point
    MEP = Vnuc - Vele

    MEP = numpy.asarray(MEP)
    MEP = MEP.reshape(nx,ny,nz)

    with open(outfile, 'w') as f:
        f.write('Molecular electrostatic potential in real space\n')
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
                        f.write(fmt % tuple(MEP[ix,iy,iz:iz+6].tolist()))
                    else:
                        fmt = '%13.5E' * remainder + '\n'
                        f.write(fmt % tuple(MEP[ix,iy,iz:iz+remainder].tolist()))
                        break



if __name__ == '__main__':
    from pyscf import gto, scf
    from pyscf.tools import cubegen
    mol = gto.M(atom='O 0.00000000,  0.000000,  0.000000; H 0.761561, 0.478993, 0.00000000,; H -0.761561, 0.478993, 0.00000000,', basis='6-31g*')
    mf = scf.RHF(mol)
    mf.scf()
    cubegen.density(mol, 'h2o_den.cube', mf.make_rdm1())
    cubegen.mep(mol, 'h2o_pot.cube', mf.make_rdm1())

