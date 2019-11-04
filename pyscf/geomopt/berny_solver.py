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
Interface to geometry optimizer pyberny https://github.com/azag0/pyberny
'''

from __future__ import absolute_import
try:
    from berny import Berny, geomlib, Logger, coords
except ImportError:
    msg = ('Geometry optimizer pyberny not found.\npyberny library '
           'can be found on github https://github.com/azag0/pyberny.\n'
           'You can install pyberny with "pip install pyberny"')
    raise ImportError(msg)

import time
import numpy
from pyscf import lib
from pyscf.geomopt.addons import (as_pyscf_method, dump_mol_geometry,
                                  symmetrize)
from pyscf import __config__
from pyscf.grad.rhf import GradientsBasics

# Overwrite pyberny's atomic unit
coords.angstrom = 1./lib.param.BOHR

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
    if getattr(geomlib, 'Geometry', None):
        return geomlib.Geometry(species, coords)
    else:
        return geomlib.Molecule(species, coords)

def _geom_to_atom(mol, geom, include_ghost):
    coords = geom.coords
    if include_ghost:
        atom_coords = coords / lib.param.BOHR
    else:
        atmlst = numpy.where(mol.atom_charges() != 0)[0]
        atom_coords = mol.atom_coords()
        atom_coords[atmlst] = coords / lib.param.BOHR
    return atom_coords

def to_berny_log(pyscf_log):
    '''Adapter to allow pyberny to use pyscf.logger
    '''
    class BernyLogger(Logger):
        def __call__(self, msg, level=0):
            if level >= -self.verbosity:
                pyscf_log.info('%d %s', self.n, msg)
    return BernyLogger()


def kernel(method, assert_convergence=ASSERT_CONV,
           include_ghost=INCLUDE_GHOST, callback=None, **kwargs):
    '''Optimize geometry with pyberny for the given method.
    
    To adjust the convergence threshold, parameters can be set in kwargs as
    below:

    .. code-block:: python
        conv_params = {  # They are default settings
            'gradientmax': 0.45e-3,  # Eh/Angstrom
            'gradientrms': 0.15e-3,  # Eh/Angstrom
            'stepmax': 1.8e-3,       # Angstrom
            'steprms': 1.2e-3,       # Angstrom
        }
        from pyscf.geomopt import berny_solver
        opt = berny_solver.GeometryOptimizer(method)
        opt.params = conv_params
        opt.kernel()
    '''
    t0 = time.clock(), time.time()
    mol = method.mol.copy()
    if 'log' in kwargs:
        log = lib.logger.new_logger(method, kwargs['log'])
    elif 'verbose' in kwargs:
        log = lib.logger.new_logger(method, kwargs['verbose'])
    else:
        log = lib.logger.new_logger(method)

    if isinstance(method, lib.GradScanner):
        g_scanner = method
    elif isinstance(method, GradientsBasics):
        g_scanner = method.as_scanner()
    elif getattr(method, 'nuc_grad_method', None):
        g_scanner = method.nuc_grad_method().as_scanner()
    else:
        raise NotImplementedError('Nuclear gradients of %s not available' % method)
    if not include_ghost:
        g_scanner.atmlst = numpy.where(method.mol.atom_charges() != 0)[0]

    # When symmetry is enabled, the molecule may be shifted or rotated to make
    # the z-axis be the main axis. The transformation can cause inconsistency
    # between the optimization steps. The transformation is muted by setting
    # an explict point group to the keyword mol.symmetry (see symmetry
    # detection code in Mole.build function).
    if mol.symmetry:
        mol.symmetry = mol.topgroup

# temporary interface, taken from berny.py optimize function
    berny_log = to_berny_log(log)
    geom = to_berny_geom(mol, include_ghost)
    optimizer = Berny(geom, log=berny_log, **kwargs)

    t1 = t0
    e_last = 0
    for cycle, geom in enumerate(optimizer):
        if log.verbose >= lib.logger.NOTE:
            log.note('\nGeometry optimization cycle %d', cycle+1)
            dump_mol_geometry(mol, geom.coords, log)

        if mol.symmetry:
            geom.coords = symmetrize(mol, geom.coords)

        mol.set_geom_(_geom_to_atom(mol, geom, include_ghost), unit='Bohr')
        energy, gradients = g_scanner(mol)
        log.note('cycle %d: E = %.12g  dE = %g  norm(grad) = %g', cycle+1,
                 energy, energy - e_last, numpy.linalg.norm(gradients))
        e_last = energy
        if callable(callback):
            callback(locals())

        if assert_convergence and not g_scanner.converged:
            raise RuntimeError('Nuclear gradients of %s not converged' % method)
        optimizer.send((energy, gradients))
        t1 = log.timer('geomoetry optimization cycle %d'%cycle, *t1)

    t0 = log.timer('geomoetry optimization', *t0)
    return optimizer._converged, mol

def optimize(method, assert_convergence=ASSERT_CONV,
             include_ghost=INCLUDE_GHOST, callback=None, **kwargs):
    '''Optimize geometry with pyberny for the given method.
    
    To adjust the convergence threshold, parameters can be set in kwargs as
    below:

    .. code-block:: python
        conv_params = {  # They are default settings
            'gradientmax': 0.45e-3,  # Eh/Angstrom
            'gradientrms': 0.15e-3,  # Eh/Angstrom
            'stepmax': 1.8e-3,       # Angstrom
            'steprms': 1.2e-3,       # Angstrom
        }
        from pyscf.geomopt import berny_solver
        newmol = berny_solver.optimize(method, **conv_params)
    '''
    return kernel(method, assert_convergence, include_ghost, callback,
                  **kwargs)[1]

class GeometryOptimizer(lib.StreamObject):
    '''Optimize the molecular geometry for the input method.

    Note the method.mol will be changed after calling .kernel() method.
    '''
    def __init__(self, method):
        self.method = method
        self.callback = None
        self.params = {}
        self.converged = False
        self.max_cycle = 100

    @property
    def mol(self):
        return self.method.mol
    @mol.setter
    def mol(self, x):
        self.method.mol = x

    def kernel(self, params=None):
        if params is not None:
            self.params.update(params)
        params = dict(self.params)
        params['maxsteps'] = self.max_cycle
        self.converged, self.mol = \
                kernel(self.method, callback=self.callback, **params)
        return self.mol
    optimize = kernel

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
    conv_params = {
        'gradientmax': 6e-3,  # Eh/AA
        'gradientrms': 2e-3,  # Eh/AA
        'stepmax': 2e-2,      # AA
        'steprms': 1.5e-2,    # AA
    }
    mol1 = optimize(mf, **conv_params)
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
