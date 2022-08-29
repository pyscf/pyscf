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
# Author: Paul J. Robinson <pjrobinson@ucla.edu>
#         Qiming Sun <osirpt.sun@gmail.com>
#

'''
Vasp CHGCAR file format

See also
https://cms.mpi.univie.ac.at/vasp/vasp/CHGCAR_file.html
'''

import sys
import collections
import time
import numpy
import pyscf
from pyscf import lib
from pyscf import gto
from pyscf.pbc import gto as pbcgto
from pyscf.tools import cubegen

if sys.version_info >= (3,):
    unicode = str

RESOLUTION = cubegen.RESOLUTION
BOX_MARGIN = cubegen.BOX_MARGIN


def density(cell, outfile, dm, nx=60, ny=60, nz=60, resolution=RESOLUTION):
    '''Calculates electron density and write out in CHGCAR format.

    Args:
        cell : Mole or Cell object
            Mole or pbc Cell. If Mole object is given, the program will guess
            a cubic lattice for the molecule.
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

    Returns:
        No return value. This function outputs a VASP chgcarlike file
        (with phase if desired)...it can be opened in VESTA or VMD or
        many other softwares

    Examples:

        >>> # generates the first MO from the list of mo_coefficents
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.tools import chgcar
        >>> cell = gto.M(atom='H 0 0 0; H 0 0 1', a=numpy.eye(3)*3)
        >>> mf = scf.RHF(cell).run()
        >>> chgcar.density(cell, 'h2.CHGCAR', mf.make_rdm1())

    '''
    cc = CHGCAR(cell, nx=nx, ny=ny, nz=nz, resolution=resolution)

    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    rho = numpy.empty(ngrids)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        if isinstance(cell, pbcgto.cell.Cell):
            ao = cell.pbc_eval_gto('GTOval', coords[ip0:ip1])
        else:
            ao = cell.eval_gto('GTOval', coords[ip0:ip1])
        rho[ip0:ip1] = lib.einsum('pi,ij,pj->p', ao, dm, ao)
    rho = rho.reshape(nx,ny,nz)

    cc.write(rho, outfile)


def orbital(cell, outfile, coeff, nx=60, ny=60, nz=60, resolution=RESOLUTION):
    '''Calculate orbital value on real space grid and write out in
    CHGCAR format.

    Args:
        cell : Mole or Cell object
            Mole or pbc Cell. If Mole object is given, the program will guess
            a cubic lattice for the molecule.
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

    Returns:
        No return value. This function outputs a VASP chgcarlike file
        (with phase if desired)...it can be opened in VESTA or VMD or
        many other softwares

    Examples:

        >>> # generates the first MO from the list of mo_coefficents
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.tools import chgcar
        >>> cell = gto.M(atom='H 0 0 0; H 0 0 1', a=numpy.eye(3)*3)
        >>> mf = scf.RHF(cell).run()
        >>> chgcar.orbital(cell, 'h2_mo1.CHGCAR', mf.mo_coeff[:,0])

    '''
    cc = CHGCAR(cell, nx=nx, ny=ny, nz=nz, resolution=resolution)

    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    orb_on_grid = numpy.empty(ngrids)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        if isinstance(cell, pbcgto.cell.Cell):
            ao = cell.pbc_eval_gto('GTOval', coords[ip0:ip1])
        else:
            ao = cell.eval_gto('GTOval', coords[ip0:ip1])
        orb_on_grid[ip0:ip1] = numpy.dot(ao, coeff)
    orb_on_grid = orb_on_grid.reshape(nx,ny,nz)

    cc.write(orb_on_grid, outfile, comment='Orbital value in real space (1/Bohr^3)')


class CHGCAR(cubegen.Cube):
    '''  Read-write of the Vasp CHGCAR files  '''
    def __init__(self, cell, nx=60, ny=60, nz=60, resolution=RESOLUTION,
                 margin=BOX_MARGIN):
        if not isinstance(cell, pbcgto.cell.Cell):
            coord = cell.atom_coords()
            box = numpy.max(coord,axis=0) - numpy.min(coord,axis=0) + margin*2
            boxorig = numpy.min(coord,axis=0) - margin
            if resolution is not None:
                nx, ny, nz = numpy.ceil(box / resolution).astype(int)
            self.box = numpy.diag(box)
            lib.logger.warn(cell, 'Molecular system is found. FFT-grid is not '
                            'available for Molecule. Lattice (in Bohr)\n'
                            '%s\nand FFT grids %s are applied.',
                            self.box, (nx,ny,nz))
            self.mol = cell
            cell = cell.view(pbcgto.Cell)
            if (isinstance(cell.unit, (str, unicode)) and
                cell.unit.startswith(('B','b','au','AU'))):
                cell.a = self.box
            else:
                cell.a = self.box * lib.param.BOHR
            ptr = cell._atm[:,gto.PTR_COORD]
            cell._env[ptr+0] = coord[:,0] - boxorig[0]
            cell._env[ptr+1] = coord[:,1] - boxorig[1]
            cell._env[ptr+2] = coord[:,2] - boxorig[2]

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.cell = cell
        self.box = cell.lattice_vectors()
        self.boxorig = numpy.zeros(3)
        self.vol = cell.vol

    def get_coords(self) :
        """  Result: set of coordinates to compute a field which is to be stored
        in the file.
        """
        xs = numpy.arange(self.nx) * (1./self.nx)
        ys = numpy.arange(self.ny) * (1./self.ny)
        zs = numpy.arange(self.nz) * (1./self.nz)
        xyz = lib.cartesian_prod((xs, ys, zs))
        coords = numpy.dot(xyz, self.box)
        return numpy.asarray(coords, order='C')

    def write(self, field, fname, comment=None):
        """  Result: .vasp file with the field in the file fname.  """
        assert (field.ndim == 3)
        assert (field.shape == (self.nx, self.ny, self.nz))
        if comment is None:
            comment = 'VASP file: Electron density in real space (e/Bohr^3)  '

        cell = self.cell

        # See CHGCAR format https://cms.mpi.univie.ac.at/vasp/vasp/CHGCAR_file.html
        # the value of (total density * volume) was dumped
        field = field * self.vol

        boxA = self.box * lib.param.BOHR
        atomList= [cell.atom_pure_symbol(i) for i in range(cell.natm)]
        Axyz = zip(atomList, cell.atom_coords().tolist())
        Axyz = sorted(Axyz, key = lambda x: x[0])
        swappedCoords = [(vec[1]+self.boxorig) * lib.param.BOHR for vec in Axyz]
        vaspAtomicInfo = collections.Counter([xyz[0] for xyz in Axyz])
        vaspAtomicInfo = sorted(vaspAtomicInfo.items())
        with open(fname, 'w') as f:
            f.write(comment)
            f.write('PySCF Version: %s  Date: %s\n' % (pyscf.__version__, time.ctime()))
            f.write('1.0000000000\n')
            f.write('%14.8f %14.8f %14.8f \n' % (boxA[0,0],boxA[0,1],boxA[0,2]))
            f.write('%14.8f %14.8f %14.8f \n' % (boxA[1,0],boxA[1,1],boxA[1,2]))
            f.write('%14.8f %14.8f %14.8f \n' % (boxA[2,0],boxA[2,1],boxA[2,2]))
            f.write(''.join(['%5.3s'%atomN[0] for atomN in vaspAtomicInfo]) + '\n')
            f.write(''.join(['%5d'%atomN[1] for atomN in vaspAtomicInfo]) + '\n')
            f.write('Cartesian \n')
            for ia in range(cell.natm):
                f.write(' %14.8f %14.8f %14.8f\n' % tuple(swappedCoords[ia]))
            f.write('\n')
            f.write('%6.5s %6.5s %6.5s \n' % (self.nx,self.ny,self.nz))
            fmt = ' %14.8e '
            for iz in range(self.nz):
                for iy in range(self.ny):
                    f.write('\n')
                    for ix in range(self.nx):
                        f.write(fmt % field[ix,iy,iz])

    def read(self, chgcar_file):
        raise NotImplementedError


if __name__ == '__main__':
    from pyscf.pbc import scf
    from pyscf.tools import chgcar
    cell = gto.M(atom='H 0 0 0; H 0 0 1', a=numpy.eye(3)*3)
    mf = scf.RHF(cell).run()
    chgcar.density(cell, 'h2.CHGCAR', mf.make_rdm1()) #makes total density
    chgcar.orbital(cell, 'h2_mo1.CHGCAR', mf.mo_coeff[:,0]) # makes mo#1 (sigma)
    chgcar.orbital(cell, 'h2_mo2.CHGCAR', mf.mo_coeff[:,1]) # makes mo#2 (sigma*)

