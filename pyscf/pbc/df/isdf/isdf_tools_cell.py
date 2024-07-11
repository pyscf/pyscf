#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
#
# Author: Ning Zhang <ningzhang1024@gmail.com>
#

import sys

import numpy
import numpy as np
import copy

from pyscf.pbc.gto import Cell
import pyscf.pbc.gto as pbcgto


def build_supercell(prim_atm, 
                    prim_a, 
                    spin=0,
                    charge=0,
                    mesh=None, 
                    Ls = [1,1,1], 
                    basis='gth-dzvp', 
                    pseudo='gth-pade', 
                    ke_cutoff=70, 
                    max_memory=2000, 
                    precision=1e-8,
                    use_particle_mesh_ewald=True,
                    verbose=4):
    
    Cell = pbcgto.Cell()
    
    assert prim_a[0, 1] == 0.0
    assert prim_a[0, 2] == 0.0
    assert prim_a[1, 0] == 0.0
    assert prim_a[1, 2] == 0.0
    assert prim_a[2, 0] == 0.0
    assert prim_a[2, 1] == 0.0
    
    Supercell_a = prim_a * np.array(Ls)
    Cell.a = Supercell_a
    
    atm = []
    
    for ix in range(Ls[0]):
        for iy in range(Ls[1]):
            for iz in range(Ls[2]):
                shift = [ix * prim_a[0, 0], iy * prim_a[1, 1], iz * prim_a[2, 2]]
                for atom in prim_atm:
                    atm.append([atom[0], (atom[1][0] + shift[0], atom[1][1] + shift[1], atom[1][2] + shift[2])])
    
    Cell.atom = atm
    Cell.basis = basis
    Cell.pseudo = pseudo
    Cell.ke_cutoff = ke_cutoff
    Cell.max_memory = max_memory
    Cell.precision = precision
    Cell.use_particle_mesh_ewald = use_particle_mesh_ewald
    Cell.verbose = verbose
    Cell.unit    = 'angstorm'
    Cell.spin    = spin
    Cell.charge  = charge
    
    Cell.build(mesh=mesh)
    
    return Cell

def build_primitive_cell(supercell:Cell, kmesh):
    
    Cell = pbcgto.Cell()
    
    # assert prim_a[0, 1] == 0.0
    # assert prim_a[0, 2] == 0.0
    # assert prim_a[1, 0] == 0.0
    # assert prim_a[1, 2] == 0.0
    # assert prim_a[2, 0] == 0.0
    # assert prim_a[2, 1] == 0.0
    
    prim_a = np.array( [supercell.a[0]/kmesh[0], supercell.a[1]/kmesh[1], supercell.a[2]/kmesh[2]], dtype=np.float64 )
    
    print("supercell.a = ", supercell.a)
    print("prim_a = ", prim_a)
    
    Cell.a = prim_a
    
    atm = supercell.atom[:supercell.natm//np.prod(kmesh)]
    
    Cell.atom = atm
    Cell.basis = supercell.basis
    Cell.pseudo = supercell.pseudo
    Cell.ke_cutoff = supercell.ke_cutoff
    Cell.max_memory = supercell.max_memory
    Cell.precision = supercell.precision
    Cell.use_particle_mesh_ewald = supercell.use_particle_mesh_ewald
    Cell.verbose = supercell.verbose
    Cell.unit = supercell.unit
    
    mesh = np.array(supercell.mesh) // np.array(kmesh)
    
    Cell.build(mesh=mesh)
    
    return Cell

def build_supercell_with_partition(prim_atm, 
                                   prim_a, 
                                   mesh=None, 
                                   Ls = [1,1,1],
                                   partition = None, 
                                   basis='gth-dzvp', 
                                   pseudo='gth-pade', 
                                   ke_cutoff=70, 
                                   max_memory=2000, 
                                   precision=1e-8,
                                   use_particle_mesh_ewald=True,
                                   verbose=4):

    cell = build_supercell(prim_atm, prim_a, mesh=mesh, Ls=Ls, basis=basis, pseudo=pseudo, ke_cutoff=ke_cutoff, max_memory=max_memory, precision=precision, use_particle_mesh_ewald=use_particle_mesh_ewald, verbose=verbose)

    natm_prim = len(prim_atm)
    
    if partition is None:
        partition = []
        for i in range(natm_prim):
            partition.append([i])

    partition_supercell = []

    for ix in range(Ls[0]):
        for iy in range(Ls[1]):
            for iz in range(Ls[2]):
                cell_id = ix * Ls[1] * Ls[2] + iy * Ls[2] + iz
                for sub_partition in partition:
                    partition_supercell.append([x + cell_id * natm_prim for x in sub_partition])

    return cell, partition_supercell
