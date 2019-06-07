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
#
# Authors:  Sandeep Sharma <sanshar@gmail.com>
#           James Smith <james.smith9113@gmail.com>
#
'''
All output is deleted after the run to keep the directory neat. Comment out the
cleanup section to view output.
'''
import os, time
t0 = time.time()
from pyscf import gto, scf, mcscf, dmrgscf
from pyscf.shciscf import shci

# C2 molecule Parameters
b = 1.3119  # Bond length.
dimer_atom = 'C'
norb = 8  # Number of orbitals in active space.
nelec = 8  # Number of electrons in active space.

mol = gto.Mole()
mol.build(
    verbose=4,
    output=None,
    atom=[
        [dimer_atom, (0.000000, 0.000000, -b / 2)],
        [dimer_atom, (0.000000, 0.000000, b / 2)],
    ],
    basis={
        dimer_atom: 'ccpvdz',
    },
    symmetry=1)

# Create HF molecule
mf = scf.RHF(mol)
mf.conv_tol = 1e-9
mf.scf()

# Calculate energy of the molecules with frozen core.
# Active spaces chosen to reflect valence active space.
#mc = shci.SHCISCF( mf, norb, nelec ).state_average_([0.333333, 0.33333, 0.33333])
mc = shci.SHCISCF(mf, norb, nelec).state_average_([0.5, 0.5])
mc.fcisolver.sweep_iter = [0, 3]
mc.fcisolver.sweep_epsilon = [1.e-3, 1.e-4]
mc.kernel()

print("Total Time:    ", time.time() - t0)

# File cleanup
mc.fcisolver.cleanup_dice_files()
