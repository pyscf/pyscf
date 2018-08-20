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

'''
Interface to geometry optimizer pyberny https://github.com/azag0/pyberny
(In testing)
'''

from __future__ import absolute_import
try:
    from berny import Berny, geomlib, Logger, optimize as optimize_berny
except ImportError:
    raise ImportError('Geometry optimizer pyberny not found.\npyberny library '
                      'can be found on github https://github.com/azag0/pyberny')

import numpy
from pyscf import lib
from pyscf import __config__

INCLUDE_GHOST = getattr(__config__, 'geomopt_berny_solver_optimize_include_ghost', True)
ASSERT_CONV = getattr(__config__, 'geomopt_berny_solver_optimize_assert_convergence', True)


def to_berny_geom(mol, include_ghost=INCLUDE_GHOST):
    atom_charges = mol.atom_charges()
    if include_ghost:
        # Symbol Ghost is not supported in current version of pyberny
        #species = [mol.atom_symbol(i) if z != 0 else 'Ghost'
        #           for i,z in enumerate(atom_charges)]
        species = [mol.atom_symbol(i) if z != 0 else 'H'
                   for i,z in enumerate(atom_charges)]
        coords = mol.atom_coords() * lib.param.BOHR
    else:
        atmlst = numpy.where(atom_charges != 0)[0]  # Exclude ghost atoms
        species = [mol.atom_symbol(i) for i in atmlst]
        coords = mol.atom_coords()[atmlst] * lib.param.BOHR

    # geomlib.Geometry is available in the new version of pyberny solver. (issue #212)
    if hasattr(geomlib, 'Geometry'):
        return geomlib.Geometry(species, coords)
    else:
        return geomlib.Molecule(species, coords)

def _geom_to_atom(mol, geom, include_ghost):
    atoms = list(geom)
    position = numpy.array([x[1] for x in atoms])
    if include_ghost:
        atom_coords = position / lib.param.BOHR
    else:
        atmlst = numpy.where(mol.atom_charges() != 0)[0]
        atom_coords = mol.atom_coords()
        atom_coords[atmlst] = position / lib.param.BOHR
    return atom_coords

def to_berny_log(pyscf_log):
    '''Adapter to allow pyberny to use pyscf.logger
    '''
    class BernyLogger(Logger):
        def __call__(self, msg, level=0):
            if level >= -self.verbosity:
                pyscf_log.info('%d %s', self.n, msg)
    return BernyLogger()

def as_berny_solver(method, assert_convergence=ASSERT_CONV,
                    include_ghost=INCLUDE_GHOST):
    '''Generate a solver for berny optimize function.
    '''
    mol = method.mol.copy()
    if isinstance(method, lib.GradScanner):
        g_scanner = method
    elif hasattr(method, 'nuc_grad_method'):
        g_scanner = method.nuc_grad_method().as_scanner()
    else:
        raise NotImplementedError('Nuclear gradients of %s not available' % method)

    if not include_ghost:
        g_scanner.atmlst = numpy.where(mol.atom_charges() != 0)[0]

    geom = yield
    cout = 0
    while True:
        mol.set_geom_(_geom_to_atom(mol, geom, include_ghost), unit='Bohr')
        energy, gradients = g_scanner(mol)
        if assert_convergence and not g_scanner.converged:
            raise RuntimeError('Nuclear gradients of %s not converged' % method)

        geom = yield energy, gradients

def as_pyscf_method(mol, scan_function):
    '''Creat an wrapper for the given scan_function, to make it work as a
    pyscf gradients scanner. The wrapper can be passed to :func:`optimize`.

    Args:
        scan_function : [mol] => (e_tot, grad)

    Examples::
    >>> mol = gto.M(atom='H; H 1 1.2', basis='ccpvdz')
    >>> scan_fn = scf.RHF(mol).nuc_grad_method().as_scanner()
    >>> m = as_pyscf_method(mol, scan_fn)
    >>> berny_solver.kernel(m)
    '''
    class OmniGrad(lib.GradScanner):
        def __init__(self, g):
            self.base = g.base
        def __call__(self, mol):
            self.e_tot, grad = scan_function(mol)
            return self.e_tot, grad
        @property
        def converged(self):
            return True

    class Grad(object):
        def __init__(self, base):
            self.base = base
        def as_scanner(self):
            return OmniGrad(self)

    class OmniMethod(object):
        def __init__(self, mol):
            self.mol = mol
            self.verbose = mol.verbose
            self.stdout = mol.stdout
        def nuc_grad_method(self):
            return Grad(self)
    return OmniMethod(mol)


def optimize(method, assert_convergence=ASSERT_CONV,
             include_ghost=INCLUDE_GHOST, **kwargs):
    '''Optimize the geometry with the given method.
    '''
    mol = method.mol.copy()
    if 'log' in kwargs:
        log = lib.logger.new_logger(method, kwargs['log'])
    elif 'verbose' in kwargs:
        log = lib.logger.new_logger(method, kwargs['verbose'])
    else:
        log = lib.logger.new_logger(method)
#    geom = optimize_berny(as_berny_solver(method), to_berny_geom(mol),
#                          log=to_berny_log(log), **kwargs)
# temporary interface, taken from berny.py optimize function
    log = to_berny_log(log)
    solver = as_berny_solver(method, assert_convergence, include_ghost)
    geom = to_berny_geom(mol, include_ghost)
    next(solver)
    optimizer = Berny(geom, log=log, **kwargs)
    for geom in optimizer:
        energy, gradients = solver.send(geom)
        optimizer.send((energy, gradients))
    mol.set_geom_(_geom_to_atom(mol, geom, include_ghost), unit='Bohr')
    return mol
kernel = optimize

del(INCLUDE_GHOST, ASSERT_CONV)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf, dft, cc, mp
    mol = gto.M(atom='''
C       1.1879  -0.3829 0.0000
C       0.0000  0.5526  0.0000
O       -1.1867 -0.2472 0.0000
H       -1.9237 0.3850  0.0000
H       2.0985  0.2306  0.0000
H       1.1184  -1.0093 0.8869
H       1.1184  -1.0093 -0.8869
H       -0.0227 1.1812  0.8852
H       -0.0227 1.1812  -0.8852
                ''',
                basis='3-21g')

    mf = scf.RHF(mol)
    mol1 = optimize(mf)
    print(mf.kernel() - -153.219208484874)
    print(scf.RHF(mol1).kernel() - -153.222680852335)

    mf = dft.RKS(mol)
    mf.xc = 'pbe,'
    mf.conv_tol = 1e-7
    mol1 = optimize(mf)

    mymp2 = mp.MP2(scf.RHF(mol))
    mol1 = optimize(mymp2)

    mycc = cc.CCSD(scf.RHF(mol))
    mol1 = optimize(mycc)
