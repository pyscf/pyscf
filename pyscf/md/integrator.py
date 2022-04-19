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

from pyscf import gto, scf, grad

class Integrator:
    def __init__(self, scanner, dt=10):
        self.scanner = scanner
        self.mol = scanner.mol.copy()
        self.dt = dt


class VelocityVerlot(Integrator):

    def next_atom_position(self, m, F, r, v):
        return r + self.dt*v + self.dt*self.dt*F/m/2.0

    def next_atom_velocity(self, v, Fn1, Fn2):
        return v + self.dt * (Fn2 + Fn1)/m/2.0

    def next(self, mol=None, veloc=None):
        if mol is None:
            mol = self.scanner.mol

        e_tot, grad_i = self.scanner(mol)




if __name__ == "__main__":
    mol = gto.M(
        atom=[
            ['O', 0., 0., 0],
            ['H', 0., -0.757, 0.587],
            ['H', 0., 0.757, 0.587]],
        basis='sto-3g')

    hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()

    integrator = VelocityVerlot(hf_scanner)

    atoms = mol.atom_coords()
    print(atoms)
    atoms[0] += [1,0,0]
    mol.set_geom_(atoms)
    mol.build()
    print(mol.atom_coords())




