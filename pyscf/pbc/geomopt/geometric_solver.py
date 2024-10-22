#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
Interface to geomeTRIC library https://github.com/leeping/geomeTRIC
'''

import os
import tempfile
import numpy
import geometric
import geometric.molecule
from pyscf import lib
from pyscf.geomopt.addons import dump_mol_geometry
from pyscf import __config__
from pyscf.pbc.grad.krhf import GradientsBase

try:
    from geometric import internal, optimize, nifty, engine, molecule
except ImportError:
    msg = ('Geometry optimizer geomeTRIC not found.\ngeomeTRIC library '
           'can be found on github https://github.com/leeping/geomeTRIC.\n'
           'You can install geomeTRIC with "pip install geometric"')
    raise ImportError(msg)

# Overwrite units defined in geomeTRIC
internal.ang2bohr = optimize.ang2bohr = nifty.ang2bohr = 1./lib.param.BOHR
engine.bohr2ang = internal.bohr2ang = molecule.bohr2ang = nifty.bohr2ang = \
        optimize.bohr2ang = lib.param.BOHR
del (internal, optimize, nifty, engine, molecule)


INCLUDE_GHOST = getattr(__config__, 'geomopt_berny_solver_optimize_include_ghost', True)
ASSERT_CONV = getattr(__config__, 'geomopt_berny_solver_optimize_assert_convergence', True)

class PySCFEngine(geometric.engine.Engine):
    def __init__(self, scanner):
        molecule = geometric.molecule.Molecule()
        self.cell = cell = scanner.cell
        molecule.elem = [cell.atom_symbol(i) for i in range(cell.natm)]
        # Molecule is the geometry parser for a bunch of formats which use
        # Angstrom for Cartesian coordinates by default.
        molecule.xyzs = [cell.atom_coords()*lib.param.BOHR]  # In Angstrom
        super().__init__(molecule)

        self.scanner = scanner
        self.cycle = 0
        self.e_last = 0
        self.callback = None
        self.maxsteps = 100
        self.assert_convergence = False

    def calc_new(self, coords, dirname):
        if self.cycle >= self.maxsteps:
            raise NotConvergedError('Geometry optimization is not converged in '
                                    '%d iterations' % self.maxsteps)

        g_scanner = self.scanner
        cell = self.cell
        self.cycle += 1
        lib.logger.note(g_scanner, '\nGeometry optimization cycle %d', self.cycle)

        # geomeTRIC requires coords and gradients in atomic unit
        coords = coords.reshape(-1,3)
        if g_scanner.verbose >= lib.logger.NOTE:
            dump_mol_geometry(cell, coords*lib.param.BOHR)

        cell.set_geom_(coords, unit='Bohr')
        energy, gradients = g_scanner(cell)
        lib.logger.note(g_scanner,
                        'cycle %d: E = %.12g  dE = %g  norm(grad) = %g', self.cycle,
                        energy, energy - self.e_last, numpy.linalg.norm(gradients))
        self.e_last = energy

        if callable(self.callback):
            self.callback(locals())

        if self.assert_convergence and not g_scanner.converged:
            raise RuntimeError('Nuclear gradients of %s not converged' % g_scanner.base)
        return {"energy": energy, "gradient": gradients.ravel()}

def kernel(method, assert_convergence=ASSERT_CONV,
           include_ghost=INCLUDE_GHOST, constraints=None, callback=None,
           maxsteps=100, **kwargs):
    '''Optimize geometry with geomeTRIC library for the given method.

    To adjust the convergence threshold, parameters can be set in kwargs as
    below:

    .. code-block:: python
        conv_params = {  # They are default settings
            'convergence_energy': 1e-6,  # Eh
            'convergence_grms': 3e-4,    # Eh/Bohr
            'convergence_gmax': 4.5e-4,  # Eh/Bohr
            'convergence_drms': 1.2e-3,  # Angstrom
            'convergence_dmax': 1.8e-3,  # Angstrom
        }
        from pyscf import geometric_solver
        opt = geometric_solver.GeometryOptimizer(method)
        opt.params = conv_params
        opt.kernel()
    '''
    if isinstance(method, lib.GradScanner):
        g_scanner = method
    elif isinstance(method, GradientsBase):
        g_scanner = method.as_scanner()
    elif getattr(method, 'nuc_grad_method', None):
        g_scanner = method.nuc_grad_method().as_scanner()
    else:
        raise NotImplementedError('Nuclear gradients of %s not available' % method)
    if not include_ghost:
        g_scanner.atmlst = numpy.where(method.cell.atom_charges() != 0)[0]

    tmpf = tempfile.mktemp(dir=lib.param.TMPDIR)
    engine = PySCFEngine(g_scanner)
    engine.callback = callback
    engine.maxsteps = maxsteps
    # To avoid overwritting method.mol
    engine.cell = g_scanner.cell.copy()

    # When symmetry is enabled, the molecule may be shifted or rotated to make
    # the z-axis be the main axis. The transformation can cause inconsistency
    # between the optimization steps. The transformation is muted by setting
    # an explicit point group to the keyword mol.symmetry (see symmetry
    # detection code in Mole.build function).

    # geomeTRIC library on pypi requires to provide config file log.ini.
    if not os.path.exists(os.path.abspath(
            os.path.join(geometric.optimize.__file__, '..', 'log.ini'))):
        kwargs['logIni'] = os.path.abspath(os.path.join(__file__, '..', 'log.ini'))

    engine.assert_convergence = assert_convergence
    try:
        geometric.optimize.run_optimizer(customengine=engine, input=tmpf,
                                         constraints=constraints, **kwargs)
        conv = True
        # method.mol.set_geom_(m.xyzs[-1], unit='Angstrom')
    except NotConvergedError as e:
        lib.logger.note(method, str(e))
        conv = False
    return conv, engine.cell

def optimize(method, assert_convergence=ASSERT_CONV,
             include_ghost=INCLUDE_GHOST, constraints=None, callback=None,
             maxsteps=100, **kwargs):
    '''Optimize geometry with geomeTRIC library for the given method.

    To adjust the convergence threshold, parameters can be set in kwargs as
    below:

    .. code-block:: python
        conv_params = {  # They are default settings
            'convergence_energy': 1e-6,  # Eh
            'convergence_grms': 3e-4,    # Eh/Bohr
            'convergence_gmax': 4.5e-4,  # Eh/Bohr
            'convergence_drms': 1.2e-3,  # Angstrom
            'convergence_dmax': 1.8e-3,  # Angstrom
        }
        from pyscf import geometric_solver
        newmol = geometric_solver.optimize(method, **conv_params)
    '''
    # MRH, 07/23/2019: name all explicit kwargs for forward compatibility
    return kernel(method, assert_convergence=assert_convergence, include_ghost=include_ghost,
                  constraints=constraints, callback=callback, maxsteps=maxsteps, **kwargs)[1]

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
    def cell(self):
        return self.method.cell

    @cell.setter
    def cell(self, x):
        self.method.cell = x
        self.method.mol = x

    def kernel(self, params=None):
        if params is not None:
            self.params.update(params)
        self.converged, self.cell = \
                kernel(self.method, callback=self.callback,
                       maxsteps=self.max_cycle, **self.params)
        return self.cell
    optimize = kernel

class NotConvergedError(RuntimeError):
    pass

del (INCLUDE_GHOST, ASSERT_CONV)


if __name__ == '__main__':
    from pyscf.pbc import gto, scf, grad
    cell = gto.Cell()
    cell.atom = '''
    C 0.000000000000 0.000000000000 0.000000000000
    C 1.685068664391 1.685068664391 1.685068664391
    '''
    cell.atom= [['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391]]]
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.basis = 'gth-szv'
    cell.verbose= 4
    cell.pseudo = 'gth-pade'
    cell.unit = 'bohr'
    cell.build()

    kpts = cell.make_kpts([1,1,3])
    mf = scf.KRHF(cell, kpts, exxdiv=None)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-8
    conv_params = {
        'convergence_energy': 1e-4,  # Eh
        'convergence_grms': 1e-5,    # Eh/Bohr
        'convergence_gmax': 1e-5,  # Eh/Bohr
        'convergence_drms': 1.2e-2,  # Angstrom
        'convergence_dmax': 1.8e-2,  # Angstrom
    }
    opt = GeometryOptimizer(mf).set(params=conv_params)#.run()
    opt.run()
    cell = opt.cell
