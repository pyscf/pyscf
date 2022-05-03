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


class Frame:
    def __init__(self, ekin=None, epot=None, coords=None, time=None):
        self.ekin = ekin
        self.epot = epot
        self.etot = self.ekin + self.epot
        self.coords = coords
        self.time = time

def toframe(integrator):
    return Frame(integrator.ekin, integrator.epot, integrator.mol.atom_coords(), integrator.time)

# COPIEDDDD
def _write(dev, mol, de, atmlst):
    '''Format output of nuclear gradients.
    Args:
        dev : lib.logger.Logger object
    '''
    if atmlst is None:
        atmlst = range(mol.natm)
    dev.stdout.write('         x                y                z\n')
    for k, ia in enumerate(atmlst):
        dev.stdout.write('%d %s  %15.10f  %15.10f  %15.10f\n' %
                         (ia, mol.atom_symbol(ia), de[k,0], de[k,1], de[k,2]))

def kernel(integrator, verbose=logger.NOTE):
    log = logger.new_logger(integrator, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Start BOMD')

    # Begin the iterations to run molecular dynamics
    t1 = t0
    # TODO look at berny_solver.py and others to get an idea of how they do this sort of thing...see if we can adapt
    # for example, use the for iteration, frame in integrator
    while integrator.iteration < integrator.max_iterations:
        if log.verbose >= lib.logger.NOTE:
            log.note('\nBOMD iteration %d', integrator.iteration+1)

        integrator.next()

        if integrator.energy_output is not None:
            integrator.write_energy()

        if integrator.trajectory_output is not None:
            integrator.write_coord()

        if integrator.incore_anyway:
            integrator.frames.append(toframe(integrator))

        t1 = log.timer('BOMD iteration %d' % integrator.iteration, *t1)


class Integrator:

    def __init__(self, scanner, **kwargs):
        self.scanner = scanner
        self.mol = self.scanner.mol
        self.stdout = self.mol.stdout
        self.incore_anyway = self.mol.incore_anyway
        self.veloc = np.full((self.mol.natm, 3), 0.0)
        self.verbose = self.mol.verbose
        self.max_iterations = 1
        self.dt = 10
        self.iteration = 0
        self.frames = None
        self.epot = None
        self.ekin = None
        self.time = 0
        self.energy_output = None
        self.trajectory_output = None

        self.__dict__.update(kwargs)

    def kernel(self, dump_flags=True, veloc=None, energy_output=None, trajectory_output=None, verbose=None, incore_anyway=None):
        if veloc is not None: self.veloc = veloc
        if energy_output is not None: self.energy_output = energy_output
        if trajectory_output is not None: self.trajectory_output = trajectory_output
        if verbose is not None: self.verbose = verbose
        if incore_anyway is not None: self.incore_anyway = incore_anyway

        # avoid opening energy_output file twice
        if (self.energy_output is not None and
                # StringIO() does not have attribute 'name'
                getattr(self.stdout, 'name', None) != self.energy_output):

            if self.verbose > logger.QUIET:
                if os.path.isfile(self.energy_output):
                    print('overwrite energy output file: %s' % self.energy_output)
                else:
                    print('energy output file: %s' % self.energy_output)

            if self.energy_output == '/dev/null':
                self.energy_output = open(os.devnull, 'w')
            else:
                self.energy_output = open(self.energy_output, 'w')

        # avoid opening trajectory_output file twice
        if (self.trajectory_output is not None and
                # StringIO() does not have attribute 'name'
                getattr(self.stdout, 'name', None) != self.trajectory_output and
                getattr(self.energy_output, 'name', None) != trajectory_output):

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

    #TODO implement sanity check
    def check_sanity(self):
        pass

    def next(self):
        raise NotImplementedError("Method Not Implemented")

    def compute_kinetic_energy(self):
        energy = 0
        for v, m in zip(self.veloc, self.mol.atom_mass_list()):
            energy += 0.5 * m * data.nist.MP_ME * np.linalg.norm(v) ** 2

        return energy

    def write_energy(self):
        output = ""
        if not os.path.isfile(self.energy_stdout):
            output += "   time          Epot                 Ekin                 Etot\n"

        output += f"{self.time:8.2f}  {self.epot:.12E}  {self.ekin:.12E}  {self.ekin + self.epot:.12E}\n"

        with open(self.energy_stdout, 'a') as f:
            f.write(output)

    def write_coord(self):
        output = f"{self.mol.natm}\nMolecular Dynamics Time {self.time}\n" + self.mol.tostring(format="raw")
        with open(self.structure_stdout, "a+") as f:
            f.write(output)
            f.write('\n')


class VelocityVerlot(Integrator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.accel = None

    def next(self):
        # For the first iteration, just compute the acceleration
        # and the write out total energies and initial geometry to file
        if self.iteration == 0:
            next_epot, next_accel = self._compute_accel()

        else:
            self.mol.set_geom_(self._next_geometry(), unit="B")
            self.mol.build()
            next_epot, next_accel = self._compute_accel()
            self.veloc = self._next_velocity(next_accel)

        self.epot = next_epot
        self.ekin = self.compute_kinetic_energy()
        self.accel = next_accel

        self.iteration += 1
        self.time += self.dt

    def _compute_accel(self):
        e_tot, grad = self.scanner(self.mol)
        if not self.scanner.converged:
            raise RuntimeError("SCF did not converge!")

        a = []
        for m, g in zip(self.mol.atom_mass_list(), grad):
            a.append(-1 * g / m / data.nist.MP_ME)

        return e_tot, np.array(a)

    def _next_atom_position(self, r, v, a):
        return r + self.dt * v + 0.5 * (self.dt ** 2) * a

    def _next_atom_velocity(self, v, a2, a1):
        return v + self.dt * 0.5 * (a2 + a1)

    def _next_geometry(self):
        new_coords = []
        for r, v, a in zip(self.mol.atom_coords(), self.veloc, self.accel):
            new_coords.append(self._next_atom_position(r, v, a))

        return np.array(new_coords)

    def _next_velocity(self, next_accel):
        new_veloc = []
        for v, a2, a1 in zip(self.veloc, next_accel, self.accel):
            new_veloc.append(self._next_atom_velocity(v, a2, a1))

        return np.array(new_veloc)
