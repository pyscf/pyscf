#!/usr/bin/env python
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Peter Koval <koval.peter@gmail.com>
#         Paul J. Robinson <pjrobinson@ucla.edu>
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
from pyscf.dft import numint
from pyscf import __config__

RESOLUTION = getattr(__config__, 'cubegen_resolution', None)
BOX_MARGIN = getattr(__config__, 'cubegen_box_margin', 3.0)


def density(mol, outfile, dm, nx=80, ny=80, nz=80, resolution=RESOLUTION):
    """Calculates electron density and write out in cube format.

    Args:
        mol : Mole
            Molecule to calculate the electron density for.
        outfile : str
            Name of Cube file to be written.
        dm : ndarray
            Density matrix of molecule.

    Kwargs:
        nx : int
            Number of grid point divisions in x direction.
            Note this is function of the molecule's size; a larger molecule
            will have a coarser representation than a smaller one for the
            same value.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
    """

    cc = Cube(mol, nx, ny, nz, resolution)

    # Compute density on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    rho = numpy.empty(ngrids)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = numint.eval_ao(mol, coords[ip0:ip1])
        rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)
    rho = rho.reshape(cc.nx,cc.ny,cc.nz)

    # Write out density to the .cube file
    cc.write(rho, outfile, comment='Electron density in real space (e/Bohr^3)')

def orbital(mol, outfile, coeff, nx=80, ny=80, nz=80, resolution=RESOLUTION):
    """Calculate orbital value on real space grid and write out in cube format.

    Args:
        mol : Mole
            Molecule to calculate the electron density for.
        outfile : str
            Name of Cube file to be written.
        coeff : 1D array
            coeff coefficient.

    Kwargs:
        nx : int
            Number of grid point divisions in x direction.
            Note this is function of the molecule's size; a larger molecule
            will have a coarser representation than a smaller one for the
            same value.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
    """
    cc = Cube(mol, nx, ny, nz, resolution)

    # Compute density on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    orb_on_grid = numpy.empty(ngrids)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = numint.eval_ao(mol, coords[ip0:ip1])
        orb_on_grid[ip0:ip1] = numpy.dot(ao, coeff)
    orb_on_grid = orb_on_grid.reshape(cc.nx,cc.ny,cc.nz)

    # Write out orbital to the .cube file
    cc.write(orb_on_grid, outfile, comment='Orbital value in real space (1/Bohr^3)')


def mep(mol, outfile, dm, nx=80, ny=80, nz=80, resolution=RESOLUTION):
    """Calculates the molecular electrostatic potential (MEP) and write out in
    cube format.

    Args:
        mol : Mole
            Molecule to calculate the electron density for.
        outfile : str
            Name of Cube file to be written.
        dm : ndarray
            Density matrix of molecule.

    Kwargs:
        nx : int
            Number of grid point divisions in x direction.
            Note this is function of the molecule's size; a larger molecule
            will have a coarser representation than a smaller one for the
            same value.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
    """
    cc = Cube(mol, nx, ny, nz, resolution)

    coords = cc.get_coords()

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
        Vele.append(numpy.einsum('ij,ij', mol.intor('int1e_rinv'), dm))

    MEP = Vnuc - Vele     # MEP at each point

    MEP = numpy.asarray(MEP)
    MEP = MEP.reshape(nx,ny,nz)

    # Write the potential
    cc.write(MEP, outfile, 'Molecular electrostatic potential in real space')


class Cube(object):
    '''  Read-write of the Gaussian CUBE files  '''
    def __init__(self, mol, nx=80, ny=80, nz=80, resolution=RESOLUTION,
                 margin=BOX_MARGIN):
        self.mol = mol
        coord = mol.atom_coords()
        self.box = numpy.max(coord,axis=0) - numpy.min(coord,axis=0) + margin*2
        self.boxorig = numpy.min(coord,axis=0) - margin
        if resolution is not None:
            nx, ny, nz = numpy.ceil(self.box / resolution).astype(int)

        self.nx = nx
        self.ny = ny
        self.nz = nz
        # .../(nx-1) to get symmetric mesh
        # see also the discussion on https://github.com/sunqm/pyscf/issues/154
        self.xs = numpy.arange(nx) * (self.box[0] / (nx - 1))
        self.ys = numpy.arange(ny) * (self.box[1] / (ny - 1))
        self.zs = numpy.arange(nz) * (self.box[2] / (nz - 1))

    def get_coords(self) :
        """  Result: set of coordinates to compute a field which is to be stored
        in the file.
        """
        coords = lib.cartesian_prod([self.xs,self.ys,self.zs])
        coords = numpy.asarray(coords, order='C') - (-self.boxorig)
        return coords

    def get_ngrids(self):
        return self.nx * self.ny * self.nz

    def get_volume_element(self):
        return (self.xs[1]-self.xs[0])*(self.ys[1]-self.ys[0])*(self.zs[1]-self.zs[0])

    def write(self, field, fname, comment=None):
        """  Result: .cube file with the field in the file fname.  """
        assert(field.ndim == 3)
        assert(field.shape == (self.nx, self.ny, self.nz))
        if comment is None:
            comment = 'Generic field? Supply the optional argument "comment" to define this line'

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
                    for iz0, iz1 in lib.prange(0, self.nz, 6):
                        fmt = '%13.5E' * (iz1-iz0) + '\n'
                        f.write(fmt % tuple(field[ix,iy,iz0:iz1].tolist()))

del(RESOLUTION, BOX_MARGIN)


if __name__ == '__main__':
    from pyscf import gto, scf
    from pyscf.tools import cubegen
    mol = gto.M(atom='''O 0.00000000,  0.000000,  0.000000
                H 0.761561, 0.478993, 0.00000000
                H -0.761561, 0.478993, 0.00000000''', basis='6-31g*')
    mf = scf.RHF(mol).run()
    cubegen.density(mol, 'h2o_den.cube', mf.make_rdm1()) #makes total density
    cubegen.mep(mol, 'h2o_pot.cube', mf.make_rdm1())
    cubegen.orbital(mol, 'h2o_mo1.cube', mf.mo_coeff[:,0])

