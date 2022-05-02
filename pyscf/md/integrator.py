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

from pyscf import gto, scf, grad, data
import numpy as np
import os

class Frame:
    def __init__(self, ekin=None, epot=None, coords=None, time=None):
        self.ekin = ekin 
        self.epot = epot 
        self.coords = coords
        self.time = time

class Integrator:
    def __init__(self, scanner, veloc=None, dt=10, steps=1):
        self.scanner=scanner
        self.mol=self.scanner.mol
        self.max_steps = steps
        self.dt = dt
        if veloc is None:
            self.veloc = np.full((self.mol.n_atoms, 3), 0.0)
        
        elif veloc == "Boltzmann":
            raise NotImplementedError("Boltzmann method not implemented!")
    
        elif type(veloc) == str:
            raise NotImplementedError("Not implemented velocity from string")

        else:
            self.veloc = veloc

        self.iteration = 0
        self.frames = None
        self.epot = None
        self.ekin = None
        self.flush = True
        self.time = 0
        self.energy_stdout = "bomd.md.energies"
        self.structure_stdout = "bomd.md.xyz"

    def kernel(self):
        while self.iteration < self.max_steps:
            self.next()
            
            if self.flush:
                self.write_energy()
                self.write_coord()

    def next(self):
        raise NotImplementedError("Method Not Implemented")
    
    def compute_kinetic_energy(self):
        energy = 0
        for v, m in zip(self.veloc, self.mol.atom_mass_list()):
            energy += 0.5*m*data.nist.MP_ME * np.linalg.norm(v)**2

        return energy

    def write_energy(self):
        output = ""
        if not os.path.isfile(self.energy_stdout):
            output += "   time          Epot                 Ekin                 Etot\n"
        
        output += f"{self.time:8.2f}  {self.epot:.12E}  {self.ekin:.12E}  {self.ekin+self.epot:.12E}\n"
        
        with open(self.energy_stdout, 'a') as f:
            f.write(output)

    def write_coord(self):
        output = self.mol.tostring(format="XYZ")
        with open(self.structure_stdout, "a+") as f:
            f.write(output)
            f.write('\n')

class VelocityVerlot(Integrator):

    def __init__(self, scanner, dt=10.0, steps=1):
        super().__init__(scanner, dt=dt, steps=steps)
        self.accel = None

    def next(self, mol=None, veloc=None):
        if mol is None:
            mol = self.mol

        if veloc is None:
            veloc = self.veloc

        ##########
        if self.iteration == 0:
            epot, self.accel = self._compute_accel(mol)
            self.epot = epot
            self.ekin = self.compute_kinetic_energy()
            self.write_energy()

        if self.accel is None:
            epot, self.accel = self._compute_accel(mol)

        print("Start")
        print(self.mol.atom_coords())
        print("\nVeloc")
        print(veloc)
        print("\nAccel")
        print(f"const: {data.nist.MP_ME}")
        print(self.accel)


        mol.set_geom_(self._next_geometry(mol, veloc, self.accel), unit="B")
        mol.build()
        next_epot, next_accel = self._compute_accel(mol)
        veloc = self._next_velocity(veloc, next_accel, self.accel)

        self.mol = mol
        self.veloc = veloc
        self.epot = next_epot
        self.ekin = self.compute_kinetic_energy()
        self.accel = next_accel

        self.iteration += 1
        self.time += self.dt

    def _compute_accel(self, mol):
        e_tot, grad = self.scanner(mol)
        if not self.scanner.converged:
            raise RuntimeError("SCF did not converge!")

        a = []
        for m, g in zip(mol.atom_mass_list(), grad):
            accel = -1*g/m/data.nist.MP_ME
            a.append(accel)

        return e_tot, np.array(a)

    def _next_atom_position(self, r, v, a):
        return r + self.dt*v + 0.5*(self.dt**2)*a

    def _next_atom_velocity(self, v, a2, a1):
        return v + self.dt * 0.5*(a2 + a1)

    def _next_geometry(self, mol, veloc, accel):
        new_coords = []
        for r, v, a in zip(mol.atom_coords(), veloc, accel):
            r = self._next_atom_position(r, v, a)
            new_coords.append(r)

        return np.array(new_coords)

    def _next_velocity(self, veloc, next_accel, accel):
        new_veloc = []
        for v, a2, a1 in zip(veloc, next_accel, accel):
            new_v = self._next_atom_velocity(v, a2, a1)
            new_veloc.append(new_v)

        return np.array(new_veloc)


if __name__ == "__main__":
    mol = gto.M(
        #verbose=1,
        atom=[
            ['O', 0., 0., 0],
            ['H', 0., -0.757, 0.587],
            ['H', 0., 0.757, 0.587]],
        basis='def2-svp')

    hf_scanner = scf.RHF(mol).nuc_grad_method().as_scanner()

    hf_scanner.max_cycle = 500

    integrator = VelocityVerlot(hf_scanner)
    for i in range(1):
        integrator.next()
        op = integrator.mol.tostring(format="XYZ")
        with open("tmp.xyz", 'a+') as f:
            f.write(op)
            f.write('\n')





