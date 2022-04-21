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
    def __init__(self, scanner, dt=10):
        self.scanner = scanner
        self.mol = scanner.mol.copy()
        self.dt = dt
        self.veloc = np.array([[0.001,0,0],[-0.002,0.001,0],[0,0.004,0.0003]])
        self.iteration = 0

class VelocityVerlot(Integrator):

    def __init__(self, scanner, dt=10):
        super().__init__(scanner, dt=dt)
        self.prev_grad = None

    def next_atom_position(self, m, F, r, v):
        return r + self.dt*v + self.dt*self.dt*F/m/2.0/ data.nist.MP_ME

    def next_atom_velocity(self, v, m, Fn1, Fn2):
        return v + self.dt * (Fn2 + Fn1)/m/2.0/ data.nist.MP_ME

    def next_geometry(self, mol, veloc, grad):
        new_coords = []
        for r, m, v, F in zip(mol.atom_coords(), mol.atom_mass_list(), veloc, grad):
            r = self.next_atom_position(m, F, r, v)
            new_coords.append(r)

        return np.array(new_coords)

    def next_velocity(self, mol, veloc, grad, prev_grad):
        new_veloc = []
        for v, m, Fn2, Fn1 in zip(veloc, mol.atom_mass_list(), grad, prev_grad):
            new_v = self.next_atom_velocity(v, m, Fn1, Fn2)
            new_veloc.append(new_v)

        return np.array(new_veloc)

    def next(self, mol=None, veloc=None):
        if mol is None:
            mol = self.mol

        if veloc is None:
            veloc = self.veloc
        
        print(f"---------------- Iteration {self.iteration}")
        print("\nCoords:")
        print(mol.atom_coords())
        print("\nVeloc:")
        print(veloc)
        e_tot, grad = self.scanner(mol)
        if not self.scanner.converged:
            raise RuntimeError("SCF did not converge!") 

        grad *= -1
        print(grad)
        if self.prev_grad is None or self.iteration == 0:
            mol.set_geom_(self.next_geometry(mol, veloc, grad), unit="B")
            mol.build()
        
        else:
            #detemine new velocity first
            veloc = self.next_velocity(mol, veloc, grad, self.prev_grad)
            mol.set_geom_(self.next_geometry(mol, veloc, grad), unit="B")
            mol.build()

        self.prev_grad = grad
        self.mol = mol
        self.mol.build()
        self.veloc = veloc
        self.iteration += 1

        #print(f"---------------- Iteration {self.iteration}")
        #print("\nCoords:")
        #print(self.mol.atom_coords())
        #print("\nVeloc:")
        #print(self.veloc)
        #print("\nGrad:")
        #print(self.prev_grad)


if __name__ == "__main__":
    mol = gto.M(
        #verbose=1,
        atom=[
            ['O', 0., 0., 0],
            ['H', 0., -0.757, 0.587],
            ['H', 0., 0.757, 0.587]],
        basis='def2-svp')

    hf_scanner = scf.RHF(mol).nuc_grad_method().as_scanner()

    hf_scanner.max_cycle=500
    #hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()

    integrator = VelocityVerlot(hf_scanner)
    for i in range(500):
        integrator.next()
        op = integrator.mol.tostring(format="XYZ")
        with open("tmp.xyz", 'a+') as f:
            f.write(op)
            f.write('\n')


    #for i in range(2):
    #    integrator.next()
    #    op = integrator.mol.tostring(format="XYZ")
    #    with open("tmp.xyz", 'a+') as f:
    #        f.write(op)
    #        f.write('\n')
   
#    atoms = mol.atom_coords()
#    print(atoms)
#    atoms[0] += [1,0,0]
#    print(atoms)
#    mol.set_geom_(atoms, unit="B")
#    mol.build()
#    print(mol.atom_coords())




