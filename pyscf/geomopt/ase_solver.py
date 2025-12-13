#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

'''
Interface to ASE for the lattice and atomic positions optimization
https://ase-lib.org/ase/optimize.html
'''

from ase.optimize import BFGS
from ase.filters import UnitCellFilter, StrainFilter
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import gto
from pyscf.pbc.tools.pyscf_ase import pyscf_to_ase_atoms, PySCF

def kernel(method, target=None, logfile=None, fmax=0.05, max_steps=100):
    '''Optimize the geometry using ASE.

    Kwargs:
        target : str
            Determines which variables to optimize.
            - 'cell': Optimize both the unit-cell lattice and atomic positions.
            - 'lattice': Optimize the lattice while keeping scaled atomic positions fixed.
            - 'atoms': Optimize atomic positions only
            By default, this flag is set to 'cell' for PBC calculations and
            'atoms' for molecular systems.
        logfile: file object, Path, or str
            File to save the ASE output

    Parameters for ASE optimizer:
        fmax : float
            Convergence threshold for atomic forces (in eV/A^3).
        max_steps : int
            Maximum number of optimization steps.
    '''
    if hasattr(method, 'cell'):
        cell = method.cell
    elif hasattr(method, 'mol'):
        cell = method.mol
    else:
        raise RuntimeError(f'{method} not supported')
    is_pbc = isinstance(cell, gto.Cell)

    atoms = pyscf_to_ase_atoms(cell)
    atoms.calc = PySCF(method=method)

    if target is None:
        if is_pbc:
            atoms = UnitCellFilter(atoms)
    elif target == 'cell':
        atoms = UnitCellFilter(atoms)
    elif target == 'lattice':
        atoms = StrainFilter(atoms)

    if logfile is None:
        logfile = '-' # stdout

    opt = BFGS(atoms, logfile=logfile)
    converged = opt.run(fmax=fmax, steps=max_steps)

    if isinstance(atoms, (UnitCellFilter, StrainFilter)):
        atoms = atoms.atoms
    if is_pbc:
        cell = cell.set_geom_(atoms.get_positions(), unit='Ang', a=atoms.cell, inplace=False)
    else:
        cell = cell.set_geom_(atoms.get_positions(), unit='Ang', inplace=False)

    if converged:
        logger.note(cell, 'Geometry optimization converged')
    else:
        logger.note(cell, 'Geometry optimization not converged')
    if cell.verbose >= logger.NOTE:
        coords = cell.atom_coords() * lib.param.BOHR
        for ia in range(cell.natm):
            logger.note(cell, ' %3d %-4s %16.9f %16.9f %16.9f AA',
                        ia+1, cell.atom_symbol(ia), *coords[ia])
        if is_pbc:
            a = cell.lattice_vectors() * lib.param.BOHR
            logger.note(cell, 'lattice vectors  a1 [%.9f, %.9f, %.9f]', *a[0])
            logger.note(cell, '                 a2 [%.9f, %.9f, %.9f]', *a[1])
            logger.note(cell, '                 a3 [%.9f, %.9f, %.9f]', *a[2])
    return converged, cell

class GeometryOptimizer(lib.StreamObject):
    '''Optimize the atomic positions and lattice for the input method.

    Attributes:
        fmax : float
            Convergence threshold for atomic forces (in eV/A^3).
        max_steps : int
            Maximum number of optimization steps.
        target : str
            Determines which variables to optimize.
            - 'cell': Optimize both the unit-cell lattice and atomic positions.
            - 'lattice': Optimize the lattice while keeping scaled atomic positions fixed.
            - 'atoms': Optimize atomic positions only.
            By default, this flag is set to 'cell' for PBC calculations and
            'atoms' for molecular systems.
        logfile: file object, Path, or str
            File to save the ASE output

    Saved results:
        converged : bool
            Whether the geometry optimization is converged

    Note method.cell and method.mol will be modified after calling the .kernel() method.
    '''
    def __init__(self, method):
        self.method = method
        self.converged = False
        self.max_steps = 100
        self.fmax = 0.05
        self.target = None
        self.logfile = None

    @property
    def max_cycle(self):
        return self.max_steps

    @property
    def cell(self):
        return self.method.cell

    @cell.setter
    def cell(self, x):
        assert hasattr(self.method, 'cell')
        self.method.cell = x

    @property
    def mol(self):
        return self.method.mol

    @mol.setter
    def mol(self, x):
        self.method.mol = x

    def kernel(self):
        self.converged, cell = kernel(
            self.method, self.target, self.logfile,
            fmax=self.fmax, max_steps=self.max_steps)
        if isinstance(cell, gto.Cell):
            self.cell = cell
        else:
            self.mol = cell
        return cell

    optimize = kernel
