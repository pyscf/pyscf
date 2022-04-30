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

class Integrator:
    def __init__(self, scanner, mol=None, init_veloc = None, dt=10):
        self.scanner = scanner
        if mol is None:
            self.mol = scanner.mol.copy()
        
        else:
            self.mol = mol
       

        self.dt = dt
        
        # initialize the velocity based off of what is passed.
        # If it is a string, read from file or boltzmann
        # if it empty (none) initialize from  etc. etc.
        if init_veloc is None or init_veloc == 0:
            num_atoms = len(self.mol.atom_coords())
            self.veloc = np.full((num_atoms,3), 0.0)
        
        else:
            self.veloc = np.array([[0.001,0,0],[-0.002,0.001,0],[0,0.004,0.0003]])
        
        self.iteration = 0

class VelocityVerlot(Integrator):

    def __init__(self, scanner, dt=10):
        super().__init__(scanner, dt=dt)
        self.accel = None

    def next(self, mol=None, veloc=None):
        if mol is None:
            mol = self.mol

        if veloc is None:
            veloc = self.veloc

        ##########
        print(f"---------------- Iteration {self.iteration}")
        print("\nCoords:")
        print(mol.atom_coords())
        print("\nVeloc:")
        print(veloc)

        ##########
        if self.iteration == 0:
            # print out coord and velocity to file?
            pass

        if self.accel is None:
            p_energy, self.accel = self.compute_accel(mol)

        mol.set_geom_(self.next_geometry(mol, veloc, self.accel), unit="B")
        mol.build()
        next_p_energy, next_accel = self.compute_accel(mol)
        veloc = self.next_velocity(veloc, next_accel, self.accel)

        self.mol = mol
        self.veloc = veloc
        self.accel = next_accel
        # print out coord and velocity to file?

        self.iteration += 1

    def compute_accel(self, mol):
        e_tot, grad = self.scanner(mol)
        if not self.scanner.converged:
            raise RuntimeError("SCF did not converge!")

        a = []
        for m, g in zip(mol.atom_mass_list(), grad):
            accel = -1*g/m/data.nist.MP_ME
            a.append(accel)

        return e_tot, np.array(a)

    def next_atom_position(self, r, v, a):
        return r + self.dt*v + self.dt*self.dt*a/2.0

    def next_atom_velocity(self, v, a2, a1):
        return v + self.dt * (a2 + a1)/2.0

    def next_geometry(self, mol, veloc, accel):
        new_coords = []
        for r, v, a in zip(mol.atom_coords(), veloc, accel):
            r = self.next_atom_position(r, v, a)
            new_coords.append(r)

        return np.array(new_coords)

    def next_velocity(self, veloc, next_accel, accel):
        new_veloc = []
        for v, a2, a1 in zip(veloc, next_accel, accel):
            new_v = self.next_atom_velocity(v, a2, a1)
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
    for i in range(500):
        integrator.next()
        op = integrator.mol.tostring(format="XYZ")
        with open("tmp.xyz", 'a+') as f:
            f.write(op)
            f.write('\n')





