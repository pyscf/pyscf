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

import os
import numpy as np
from pyscf import data
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad.rhf import GradientsMixin

AMU_TO_AU = data.nist.ATOMIC_MASS/data.nist.E_MASS

class Frame:
    '''Basic class to hold information at each time step of a MD simulation

    Attributes:
        ekin : float
            Kinetic energy
        epot : float
            Potential energy (electronic energy)
        etot : float
            Total energy, sum of the potential and kinetic energy
        coord : 2D array with shape (natm, 3)
            Geometry of the system at the current time step
        veloc : 2D array with shape (natm, 3)
            Velocities of the system at the current time step
        time : float
            Time for which this frame represents
    '''
    def __init__(self, ekin=None, epot=None, coord=None, veloc=None, time=None):
        self.ekin = ekin
        self.epot = epot
        self.etot = self.ekin + self.epot
        self.coord = coord
        self.veloc = veloc
        self.time = time

def _toframe(integrator):
    '''Convert an Integrator to a Frame given current saved data.
    Args:
        integrator : md.integrator.Integrator object
    '''
    return Frame(ekin=integrator.ekin, epot=integrator.epot, coord=integrator.mol.atom_coords(), veloc=integrator.veloc, time=integrator.time)

def _write(dev, mol, vec, atmlst=None):
    '''Format output of molecular vector quantity.
    Args:
        dev : lib.logger.Logger object
        mol : gto.mol object
        vec : 2D array with shape (mol.natm, 3)
        atmlst : array of indices to pull atoms from. Must be smaller than mol.natm.
    '''
    if atmlst is None:
        atmlst = range(mol.natm)
    dev.stdout.write('         x                y                z\n')
    for k, ia in enumerate(atmlst):
        dev.stdout.write('%d %s  %15.10f  %15.10f  %15.10f\n' %
                         (ia, mol.atom_symbol(ia), vec[k,0], vec[k,1], vec[k,2]))

def kernel(integrator, verbose=logger.NOTE):
    log = logger.new_logger(integrator, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Start BOMD')

    integrator.mol = integrator.mol.copy()

    # Begin the iterations to run molecular dynamics
    t1 = t0
    for iteration, frame in enumerate(integrator):
        log.note('----------- %s final geometry -----------', integrator.__class__.__name__)
        _write(integrator, integrator.mol, frame.coord)
        log.note('----------------------------------------------')
        log.note('------------ %s final velocity -----------', integrator.__class__.__name__)
        _write(integrator, integrator.mol, frame.veloc)
        log.note('----------------------------------------------')
        log.note('Ekin = %17.13f', frame.ekin)
        log.note('Epot = %17.13f', frame.epot)
        log.note('Etot = %17.13f', frame.etot)

        t1 = log.timer('BOMD iteration %d' % iteration, *t1)

    t0 = log.timer('BOMD', *t0)
    return integrator

class Integrator:

    def __init__(self, method, **kwargs):

        if isinstance(method, lib.GradScanner):
            self.scanner = method
        elif isinstance(method, GradientsMixin):
            self.scanner = method.as_scanner()
        elif getattr(method, 'nuc_grad_method', None):
            self.scanner = method.nuc_grad_method().as_scanner()
        else:
            raise NotImplemented('Nuclear gradients of %s not available' % method)

        self.mol = self.scanner.mol
        self.stdout = self.mol.stdout
        self.incore_anyway = self.mol.incore_anyway
        self.veloc = None
        self.verbose = self.mol.verbose
        self.max_iterations = 1
        self.dt = 10
        self.frames = None
        self.epot = None
        self.ekin = None
        self.time = 0
        self.energy_output = None
        self.trajectory_output = None

        self.__dict__.update(kwargs)

    def kernel(self, veloc=None, dump_flags=True, verbose=None):

        if veloc is not None: self.veloc = veloc
        if verbose is None: verbose = self.verbose

        # Default velocities are 0 if none specified
        if self.veloc is None:
            self.veloc = np.full((self.mol.natm, 3), 0.0)

        # avoid opening energy_output file twice
        if type(self.energy_output) is str:
            if self.verbose > logger.QUIET:
                if os.path.isfile(self.energy_output):
                    print('overwrite energy output file: %s' % self.energy_output)
                else:
                    print('energy output file: %s' % self.energy_output)

            if self.energy_output == '/dev/null':
                self.energy_output = open(os.devnull, 'w')

            else:
                self.energy_output = open(self.energy_output, 'w')
                self.energy_output.write('   time          Epot                 Ekin                 Etot\n')

        # avoid opening trajectory_output file twice
        if type(self.trajectory_output) is str:
            if self.verbose > logger.QUIET:
                if os.path.isfile(self.trajectory_output):
                    print('overwrite energy output file: %s' % self.trajectory_output)
                else:
                    print('trajectory output file: %s' % self.trajectory_output)

            if self.trajectory_output == '/dev/null':
                self.trajectory_output = open(os.devnull, 'w')
            else:
                self.trajectory_output = open(self.trajectory_output, 'w')

        log = logger.new_logger(self, verbose)
        self.check_sanity()

        if dump_flags and self.verbose > logger.NOTE:
            self.dump_input()

        kernel(self, verbose=log)

    def dump_input(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** BOMD flags ********')
        log.info('dt = %f', self.dt)
        log.info('Iterations = %d', self.max_iterations)
        log.info('                   Initial Velocity                  ')
        log.info('             vx              vy              vz')
        for i, (e, v) in enumerate(zip(self.mol.elements, self.veloc)):
            log.info('%d %s    %.8E  %.8E  %.8E', i, e, *v)

    # TODO implement sanity check
    def check_sanity(self):
        pass

    def compute_kinetic_energy(self):
        '''Compute the kinetic energy of the current frame.'''
        energy = 0
        for v, m in zip(self.veloc, self.mol.atom_charges()):
            m = data.elements.COMMON_ISOTOPE_MASSES[m]
            energy += 0.5 * m * AMU_TO_AU * np.linalg.norm(v) ** 2

        return energy

    def __iter__(self):
        self._iteration = 0
        self._log = logger.new_logger(self, self.verbose)
        return self

    def __next__(self):
        if self._iteration < self.max_iterations:
            if self._log.verbose >= lib.logger.NOTE:
                self._log.note('\nBOMD Time %d', self.time)

            current_frame = self._next()

            if self.incore_anyway:
                self.frames.append(current_frame)

            if self.energy_output is not None:
                self._write_energy()

            if self.trajectory_output is not None:
                self._write_coord()

            self._iteration += 1
            self.time += self.dt

            return current_frame

        else:
            raise StopIteration

    def _next(self):
        '''Determines the next step in the molecular dynamics simulation. Integrates to
        the next time step. Must be implemented in derived classes.

        Returns:
            'Frame' which contains the new geometry, velocity, time step, and energy.
        '''
        raise NotImplementedError('Method Not Implemented')

    def _write_energy(self):
        '''Writes out the potential, kinetic, and total energy to the self.energy_output stream.'''
        self.energy_output.write('%8.2f  %.12E  %.12E  %.12E\n' % (
            self.time, self.epot, self.ekin, self.ekin+self.epot))

    def _write_coord(self):
        '''Writes out the current geometry to the self.trajectroy_output stream in xyz format.'''
        self.trajectory_output.write('%s\nMD Time %s\n' % (self.mol.natm, self.time))
        self.trajectory_output.write(self.mol.tostring(format='raw') + '\n')


class VelocityVerlot(Integrator):

    def __init__(self, method, **kwargs):
        super().__init__(method, **kwargs)
        self.accel = None

    def _next(self):
        # If no acceleration, compute that first, and then go onto the next step
        if self.accel is None:
            next_epot, next_accel = self._compute_accel()

        else:
            self.mol.set_geom_(self._next_geometry(), unit='B')
            self.mol.build()
            next_epot, next_accel = self._compute_accel()
            self.veloc = self._next_velocity(next_accel)

        self.epot = next_epot
        self.ekin = self.compute_kinetic_energy()
        self.accel = next_accel

        return _toframe(self)

    def _compute_accel(self):
        '''Given the current geometry, computes the acceleration for each atom.'''
        e_tot, grad = self.scanner(self.mol)
        if not self.scanner.converged:
            raise RuntimeError('SCF did not converge!')


        a = []
        for m, g in zip(self.mol.atom_charges(), grad):
            m = data.elements.COMMON_ISOTOPE_MASSES[m]
            a.append(-1 * g / m / AMU_TO_AU)

        return e_tot, np.array(a)

    def _next_atom_position(self, r, v, a):
        return r + self.dt * v + 0.5 * (self.dt ** 2) * a

    def _next_atom_velocity(self, v, a2, a1):
        return v + self.dt * 0.5 * (a2 + a1)

    def _next_geometry(self):
        '''Computes the next geometry using the Velocity Verlet algorithm.'''
        new_coords = []
        for r, v, a in zip(self.mol.atom_coords(), self.veloc, self.accel):
            new_coords.append(self._next_atom_position(r, v, a))

        return np.array(new_coords)

    def _next_velocity(self, next_accel):
        '''Compute the next velocity using the Velocity Verlet algorithm'''
        new_veloc = []
        for v, a2, a1 in zip(self.veloc, next_accel, self.accel):
            new_veloc.append(self._next_atom_velocity(v, a2, a1))

        return np.array(new_veloc)
