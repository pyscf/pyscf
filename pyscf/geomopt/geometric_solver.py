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
Interface to geomeTRIC library https://github.com/leeping/geomeTRIC
'''

import os
import uuid
import tempfile
import numpy
import geometric
import geometric.molecule
#from geometric import molecule
from pyscf import lib
from pyscf.lib import logger
from pyscf.geomopt.addons import (as_pyscf_method, dump_mol_geometry,
                                  symmetrize)  # noqa
from pyscf import __config__
from pyscf.grad.rhf import GradientsMixin

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
        self.mol = mol = scanner.mol
        molecule.elem = [mol.atom_symbol(i) for i in range(mol.natm)]
        # Molecule is the geometry parser for a bunch of formats which use
        # Angstrom for Cartesian coordinates by default.
        molecule.xyzs = [mol.atom_coords()*lib.param.BOHR]  # In Angstrom
        super(PySCFEngine, self).__init__(molecule)

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
        mol = self.mol
        self.cycle += 1
        logger.note(g_scanner, '\nGeometry optimization cycle %d', self.cycle)

        # geomeTRIC requires coords and gradients in atomic unit
        coords = coords.reshape(-1,3)
        if g_scanner.verbose >= logger.NOTE:
            dump_mol_geometry(mol, coords*lib.param.BOHR)

        if mol.symmetry:
            coords = symmetrize(mol, coords)

        mol.set_geom_(coords, unit='Bohr')
        energy, gradients = g_scanner(mol)
        logger.note(g_scanner, 'cycle %d: E = %.12g  dE = %g  norm(grad) = %g',
                    self.cycle, energy, energy - self.e_last, numpy.linalg.norm(gradients))
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
    elif isinstance(method, GradientsMixin):
        g_scanner = method.as_scanner()
    elif getattr(method, 'nuc_grad_method', None):
        g_scanner = method.nuc_grad_method().as_scanner()
    else:
        raise NotImplementedError('Nuclear gradients of %s not available' % method)
    if not include_ghost:
        g_scanner.atmlst = numpy.where(method.mol.atom_charges() != 0)[0]

    engine = PySCFEngine(g_scanner)
    engine.callback = callback
    engine.maxsteps = maxsteps
    # To avoid overwritting method.mol
    engine.mol = g_scanner.mol.copy()

    # When symmetry is enabled, the molecule may be shifted or rotated to make
    # the z-axis be the main axis. The transformation can cause inconsistency
    # between the optimization steps. The transformation is muted by setting
    # an explict point group to the keyword mol.symmetry (see symmetry
    # detection code in Mole.build function).
    if engine.mol.symmetry:
        engine.mol.symmetry = engine.mol.topgroup
    engine.assert_convergence = assert_convergence

    # geomeTRIC library on pypi requires to provide config file log.ini.
    if not os.path.exists(os.path.abspath(
            os.path.join(geometric.optimize.__file__, '..', 'log.ini'))) and kwargs.get('logIni') is None:
        kwargs['logIni'] = os.path.abspath(os.path.join(__file__, '..', 'log.ini'))

    with tempfile.TemporaryDirectory(dir=lib.param.TMPDIR) as tmpdir:
        tmpf = os.path.join(tmpdir, str(uuid.uuid4()))

        if 'hessian' in kwargs:
            kwargs['hessian'] = _make_hessian(g_scanner, kwargs['hessian'], tmpdir)
            logger.debug(g_scanner, 'Analytical hessian saved in %s', kwargs['hessian'])

        try:
            geometric.optimize.run_optimizer(customengine=engine, input=tmpf,
                                             constraints=constraints, **kwargs)
            conv = True
            # method.mol.set_geom_(m.xyzs[-1], unit='Angstrom')
        except NotConvergedError as e:
            logger.note(method, str(e))
            conv = False
    return conv, engine.mol

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

def _make_hessian(g_scanner, hessian_option, tmpdir):
    '''calculate hessian and saved to a file.
    Returns the filename in the geomeTRIC supported format, e.g.
    file:/path/to/hessian_file
    '''
    if not isinstance(hessian_option, str):
        hessian_option = os.path.join(tmpdir, str(uuid.uuid4()))
    if ':' in hessian_option:
        hessian_file = hessian_option.split(':', 1)[1]
    else:
        hessian_file, hessian_option = hessian_option, f'first:{hessian_file}'

    method = g_scanner.base
    natm = method.mol.natm
    try:
        h = method.Hessian().kernel()
    except (TypeError, NotImplementedError):
        logger.warn(g_scanner, 'Analytical hessian for %s is not available', method)
        hessian_option = False
    else:
        h = h.transpose(0,2,1,3).reshape(3*natm, 3*natm)
        numpy.savetxt(hessian_file, h)
    return hessian_option

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
        self.converged, self.mol = \
                kernel(self.method, callback=self.callback,
                       maxsteps=self.max_cycle, **self.params)
        return self.mol
    optimize = kernel

class NotConvergedError(RuntimeError):
    pass
