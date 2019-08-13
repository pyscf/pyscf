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
#
# Authors:  Sandeep Sharma <sanshar@gmail.com>
#           James Smith <james.smith9113@gmail.com>
#
"""
All output is deleted after the run to keep the directory neat. Comment out the
cleanup section to view output.
"""
import os, time

from pyscf import gto, scf, mcscf, dmrgscf
from pyscf.shciscf import shci

t0 = time.time()

#
# Mean Field
#
mol = gto.M(verbose=4, atom="C 0 0 0; C 0 0 1.3119", basis="ccpvdz")
mf = scf.RHF(mol).run()

# Create HF molecule
mf = scf.RHF(mol)
mf.conv_tol = 1e-9
mf.scf()

# Calculate energy of the molecules with frozen core.
# Active spaces chosen to reflect valence active space.
# mc = shci.SHCISCF( mf, norb, nelec ).state_average_([0.333333, 0.33333, 0.33333])
ncas = 8
nelecas = 8
mc = shci.SHCISCF(mf, ncas, nelecas).state_average_([0.5, 0.5])
mc.frozen = mc.ncore
mc.fcisolver.sweep_iter = [0, 3]
mc.fcisolver.sweep_epsilon = [1.0e-3, 1.0e-4]
mc.kernel()

print("Total Time:    ", time.time() - t0)

# File cleanup
mc.fcisolver.cleanup_dice_files()
