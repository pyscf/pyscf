#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
https://h5cube-spec.readthedocs.io/en/latest/cubeformat.html
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

import time
import numpy
import pyscf
from pyscf import lib
from pyscf import gto
from pyscf import df
from pyscf.dft import numint
from pyscf import __config__

RESOLUTION = getattr(__config__, 'cubegen_resolution', None)
BOX_MARGIN = getattr(__config__, 'cubegen_box_margin', 3.0)
ORIGIN = getattr(__config__, 'cubegen_box_origin', None)
# If given, EXTENT should be a 3-element ndarray/list/tuple to represent the
# extension in x, y, z
EXTENT = getattr(__config__, 'cubegen_box_extent', None)


def density(mol, outfile, dm, nx=80, ny=80, nz=80, resolution=RESOLUTION,
            margin=BOX_MARGIN):
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
            same value. Conflicts to keyword resolution.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
        resolution: float
            Resolution of the mesh grid in the cube box. If resolution is
            given in the input, the input nx/ny/nz have no effects.  The value
            of nx/ny/nz will be determined by the resolution and the cube box
            size.
    """
    from pyscf.pbc.gto import Cell
    cc = Cube(mol, nx, ny, nz, resolution, margin)

    GTOval = 'GTOval'
    if isinstance(mol, Cell):
        GTOval = 'PBC' + GTOval

    # Compute density on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    rho = numpy.empty(ngrids)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = mol.eval_gto(GTOval, coords[ip0:ip1])
        rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)
    rho = rho.reshape(cc.nx,cc.ny,cc.nz)

    # Write out density to the .cube file
    cc.write(rho, outfile, comment='Electron density in real space (e/Bohr^3)')
    return rho


def orbital(mol, outfile, coeff, nx=80, ny=80, nz=80, resolution=RESOLUTION,
            margin=BOX_MARGIN):
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
            same value. Conflicts to keyword resolution.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
        resolution: float
            Resolution of the mesh grid in the cube box. If resolution is
            given in the input, the input nx/ny/nz have no effects.  The value
            of nx/ny/nz will be determined by the resolution and the cube box
            size.
    """
    from pyscf.pbc.gto import Cell
    cc = Cube(mol, nx, ny, nz, resolution, margin)

    GTOval = 'GTOval'
    if isinstance(mol, Cell):
        GTOval = 'PBC' + GTOval

    # Compute density on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    orb_on_grid = numpy.empty(ngrids)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = mol.eval_gto(GTOval, coords[ip0:ip1])
        orb_on_grid[ip0:ip1] = numpy.dot(ao, coeff)
    orb_on_grid = orb_on_grid.reshape(cc.nx,cc.ny,cc.nz)

    # Write out orbital to the .cube file
    cc.write(orb_on_grid, outfile, comment='Orbital value in real space (1/Bohr^3)')
    return orb_on_grid


def mep(mol, outfile, dm, nx=80, ny=80, nz=80, resolution=RESOLUTION,
        margin=BOX_MARGIN):
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
            same value. Conflicts to keyword resolution.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
        resolution: float
            Resolution of the mesh grid in the cube box. If resolution is
            given in the input, the input nx/ny/nz have no effects.  The value
            of nx/ny/nz will be determined by the resolution and the cube box
            size.
    """
    cc = Cube(mol, nx, ny, nz, resolution, margin)

    coords = cc.get_coords()

    # Nuclear potential at given points
    Vnuc = 0
    for i in range(mol.natm):
        r = mol.atom_coord(i)
        Z = mol.atom_charge(i)
        rp = r - coords
        Vnuc += Z / numpy.einsum('xi,xi->x', rp, rp)**.5

    # Potential of electron density
    Vele = numpy.empty_like(Vnuc)
    for p0, p1 in lib.prange(0, Vele.size, 600):
        fakemol = gto.fakemol_for_charges(coords[p0:p1])
        ints = df.incore.aux_e2(mol, fakemol)
        Vele[p0:p1] = numpy.einsum('ijp,ij->p', ints, dm)

    MEP = Vnuc - Vele     # MEP at each point
    MEP = MEP.reshape(cc.nx,cc.ny,cc.nz)

    # Write the potential
    cc.write(MEP, outfile, 'Molecular electrostatic potential in real space')
    return MEP


class Cube:
    '''  Read-write of the Gaussian CUBE files

    Attributes:
        nx : int
            Number of grid point divisions in x direction.
            Note this is function of the molecule's size; a larger molecule
            will have a coarser representation than a smaller one for the
            same value. Conflicts to keyword resolution.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
        resolution: float
            Resolution of the mesh grid in the cube box. If resolution is
            given in the input, the input nx/ny/nz have no effects.  The value
            of nx/ny/nz will be determined by the resolution and the cube box
            size. The unit is Bohr.
    '''
    def __init__(self, mol, nx=80, ny=80, nz=80, resolution=RESOLUTION,
                 margin=BOX_MARGIN, origin=ORIGIN, extent=EXTENT):
        from pyscf.pbc.gto import Cell
        self.mol = mol
        coord = mol.atom_coords()

        # If the molecule is periodic, use lattice vectors as the box
        # and automatically determine margin, origin, and extent
        if isinstance(mol, Cell):
            self.box = mol.lattice_vectors()
            atom_center = (numpy.max(coord, axis=0) + numpy.min(coord, axis=0))/2
            box_center = (self.box[0] + self.box[1] + self.box[2])/2
            self.boxorig = atom_center - box_center
        else:
            # Create a box based on its origin and extents/lengths (rectangular cuboid only)
            # If extent is not supplied, use the coordinates plus a margin on both sides
            if extent is None:
                extent = numpy.max(coord, axis=0) - numpy.min(coord, axis=0) + 2*margin
            self.box = numpy.diag(extent)

            # if origin is not supplied, set it as the minimum coordinate minus a margin.
            if origin is None:
                origin = numpy.min(coord, axis=0) - margin
            self.boxorig = numpy.asarray(origin)

        if resolution is not None:
            nx, ny, nz = numpy.ceil(numpy.diag(self.box) / resolution).astype(int)

        self.nx = nx
        self.ny = ny
        self.nz = nz

        if isinstance(mol, Cell):
            # Use an asymmetric mesh for tiling unit cells
            self.xs = numpy.linspace(0, 1, nx, endpoint=False)
            self.ys = numpy.linspace(0, 1, ny, endpoint=False)
            self.zs = numpy.linspace(0, 1, nz, endpoint=False)
        else:
            # Use endpoint=True to get a symmetric mesh
            # see also the discussion https://github.com/sunqm/pyscf/issues/154
            self.xs = numpy.linspace(0, 1, nx, endpoint=True)
            self.ys = numpy.linspace(0, 1, ny, endpoint=True)
            self.zs = numpy.linspace(0, 1, nz, endpoint=True)

    def get_coords(self) :
        """  Result: set of coordinates to compute a field which is to be stored
        in the file.
        """
        frac_coords = lib.cartesian_prod([self.xs, self.ys, self.zs])
        return frac_coords @ self.box + self.boxorig # Convert fractional coordinates to real-space coordinates

    def get_ngrids(self):
        return self.nx * self.ny * self.nz

    def get_volume_element(self):
        return (self.xs[1]-self.xs[0])*(self.ys[1]-self.ys[0])*(self.zs[1]-self.zs[0])

    def write(self, field, fname, comment=None):
        """  Result: .cube file with the field in the file fname.  """
        assert (field.ndim == 3)
        assert (field.shape == (self.nx, self.ny, self.nz))
        if comment is None:
            comment = 'Generic field? Supply the optional argument "comment" to define this line'

        mol = self.mol
        coord = mol.atom_coords()
        with open(fname, 'w') as f:
            f.write(comment+'\n')
            f.write(f'PySCF Version: {pyscf.__version__}  Date: {time.ctime()}\n')
            f.write(f'{mol.natm:5d}')
            f.write('%12.6f%12.6f%12.6f\n' % tuple(self.boxorig.tolist()))
            dx = self.xs[-1] if len(self.xs) == 1 else self.xs[1]
            dy = self.ys[-1] if len(self.ys) == 1 else self.ys[1]
            dz = self.zs[-1] if len(self.zs) == 1 else self.zs[1]
            delta = (self.box.T * [dx,dy,dz]).T
            f.write(f'{self.nx:5d}{delta[0,0]:12.6f}{delta[0,1]:12.6f}{delta[0,2]:12.6f}\n')
            f.write(f'{self.ny:5d}{delta[1,0]:12.6f}{delta[1,1]:12.6f}{delta[1,2]:12.6f}\n')
            f.write(f'{self.nz:5d}{delta[2,0]:12.6f}{delta[2,1]:12.6f}{delta[2,2]:12.6f}\n')
            for ia in range(mol.natm):
                atmsymb = mol.atom_symbol(ia)
                f.write('%5d%12.6f'% (gto.charge(atmsymb), 0.))
                f.write('%12.6f%12.6f%12.6f\n' % tuple(coord[ia]))

            for ix in range(self.nx):
                for iy in range(self.ny):
                    for iz0, iz1 in lib.prange(0, self.nz, 6):
                        fmt = '%13.5E' * (iz1-iz0) + '\n'
                        f.write(fmt % tuple(field[ix,iy,iz0:iz1].tolist()))

    def read(self, cube_file):
        with open(cube_file, 'r') as f:
            f.readline()
            f.readline()
            data = f.readline().split()
            natm = int(data[0])
            self.boxorig = numpy.array([float(x) for x in data[1:]])
            def parse_nx(data):
                from pyscf.pbc.gto import Cell
                d = data.split()
                nx = int(d[0])
                x_vec = numpy.array([float(x) for x in d[1:]]) * nx
                if isinstance(self.mol, Cell):
                    # Use an asymmetric mesh for tiling unit cells
                    xs = numpy.linspace(0, 1, nx, endpoint=False)
                else:
                    # Use endpoint=True to get a symmetric mesh
                    # see also the discussion https://github.com/sunqm/pyscf/issues/154
                    xs = numpy.linspace(0, 1, nx, endpoint=True)
                return x_vec, nx, xs
            self.box = numpy.zeros((3,3))
            self.box[0], self.nx, self.xs = parse_nx(f.readline())
            self.box[1], self.ny, self.ys = parse_nx(f.readline())
            self.box[2], self.nz, self.zs = parse_nx(f.readline())
            atoms = []
            for ia in range(natm):
                d = f.readline().split()
                atoms.append([int(d[0]), [float(x) for x in d[2:]]])
            self.mol = gto.M(atom=atoms, unit='Bohr')

            data = f.read()
        cube_data = numpy.array([float(x) for x in data.split()])
        return cube_data.reshape([self.nx, self.ny, self.nz])


if __name__ == '__main__':
    from pyscf import scf
    from pyscf.tools import cubegen
    mol = gto.M(atom='''O 0.00000000,  0.000000,  0.000000
                H 0.761561, 0.478993, 0.00000000
                H -0.761561, 0.478993, 0.00000000''', basis='6-31g*')
    mf = scf.RHF(mol).run()
    cubegen.density(mol, 'h2o_den.cube', mf.make_rdm1()) #makes total density
    cubegen.mep(mol, 'h2o_pot.cube', mf.make_rdm1())
    cubegen.orbital(mol, 'h2o_mo1.cube', mf.mo_coeff[:,0])
