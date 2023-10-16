#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
Helper functions for geometry optimizer
'''

import numpy
from pyscf import lib

def as_pyscf_method(mol, scan_function):
    '''Creat an wrapper for the given scan_function, to make it work as a
    pyscf gradients scanner. The wrapper can be passed to :func:`optimize`.

    Args:
        scan_function : [mol] => (e_tot, grad)

    Examples::
    >>> mol = gto.M(atom='H; H 1 1.2', basis='ccpvdz')
    >>> scan_fn = scf.RHF(mol).nuc_grad_method().as_scanner()
    >>> m = as_pyscf_method(mol, scan_fn)
    >>> pyscf.geomopt.berny_solver.kernel(m)
    '''
    from pyscf.grad.rhf import GradientsBase
    class OmniGrad(lib.GradScanner):
        def __init__(self, g):
            self.__dict__.update(g.__dict__)
            self.base = g.base
        def __call__(self, mol):
            self.e_tot, grad = scan_function(mol)
            return self.e_tot, grad
        @property
        def converged(self):
            return True

    class Gradients(GradientsBase):
        def as_scanner(self):
            return OmniGrad(self)

    class OmniMethod:
        def __init__(self, mol):
            self.mol = mol
            self.verbose = mol.verbose
            self.stdout = mol.stdout
        def Gradients(self):
            return Gradients(self)
        nuc_grad_method = Gradients
    return OmniMethod(mol)

def dump_mol_geometry(mol, new_coords, log=None):
    '''Dump the molecular geometry (new_coords) and the displacement wrt old
    geometry.

    Args:
        new_coords (ndarray) : Cartesian coordinates in Angstrom
    '''
    if log is None:
        dump = mol.stdout.write
    else:
        dump = log.stdout.write
    old_coords = mol.atom_coords() * lib.param.BOHR
    new_coords = numpy.asarray(new_coords)
    dx = new_coords - old_coords

    dump('Cartesian coordinates (Angstrom)\n')
    dump(' Atom        New coordinates             dX        dY        dZ\n')
    for i in range(mol.natm):
        dump('%4s %10.6f %10.6f %10.6f   %9.6f %9.6f %9.6f\n' %
             (mol.atom_symbol(i),
              new_coords[i,0], new_coords[i,1], new_coords[i,2],
              dx[i,0], dx[i,1], dx[i,2]))

def symmetrize(mol, coords):
    '''Symmetrize the structure of a molecule.'''
    from pyscf.grad.rhf import symmetrize
    # Symmetrizing coordinates is the same to the symmetrization of gradients.
    return symmetrize(mol, coords)
