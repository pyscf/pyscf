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
# Author: James Smith <james.smith9113@gmail.com>
#
# All diatomic bond lengths taken from:
# http://cccbdb.nist.gov/diatomicexpbondx.asp
'''
All output is deleted after the run to keep the directory neat. Comment out the
cleanup section to view output files.
'''
import time
t0 = time.time()
from pyscf import gto, scf, mcscf, dmrgscf
from pyscf.shciscf import shci
import struct, os

# Initialize O2 molecule
b = 1.208
mol = gto.Mole()
mol.build(
    verbose=4,
    output=None,
    atom="O 0 0 %f; O 0 0 %f" % (-b / 2, b / 2),
    basis='ccpvdz',
    symmetry=True)

# Create HF molecule
mf = scf.RHF(mol)
mf.conv_tol = 1e-9
mf.scf()

# Number of orbital and electrons
ncas = 8
nelecas = 12
dimer_atom = 'O'

mc = mcscf.CASSCF(mf, ncas, nelecas)
e_CASSCF = mc.mc1step()[0]

# Create SHCI molecule for just variational opt.
# Active spaces chosen to reflect valence active space.
mc = shci.SHCISCF(mf, ncas, nelecas)
mc.fcisolver.mpiprefix = 'mpirun -np 2'
mc.fcisolver.stochastic = True
mc.fcisolver.nPTiter = 0  # Turn off perturbative calc.
mc.fcisolver.sweep_iter = [0]
# Setting large epsilon1 thresholds highlights improvement from perturbation.
mc.fcisolver.sweep_epsilon = [5e-3]
e_noPT = mc.mc1step()[0]

# Run a single SHCI iteration with perturbative correction.
mc.fcisolver.stochastic = False  # Turns on deterministic PT calc.
mc.fcisolver.epsilon2 = 1e-8
shci.writeSHCIConfFile(mc.fcisolver, [nelecas / 2, nelecas / 2], False)
shci.executeSHCI(mc.fcisolver)
e_PT = shci.readEnergy(mc.fcisolver)  #struct.unpack(format, file1.read())

# Comparison Calculations
del_PT = e_PT - e_noPT
del_shci = e_CASSCF - e_PT

print('\n\nEnergies for %s2 give in E_h.' % dimer_atom)
print('=====================================')
print('SHCI Variational: %6.12f' % e_noPT)
# Prints the total energy including the perturbative component.
print('SHCI Perturbative: %6.12f' % e_PT)
print('Perturbative Change: %6.12f' % del_PT)
print('CASSCF Total Energy: %6.12f' % e_CASSCF)
print('E(CASSCF) - E(SHCI): %6.12f' % del_shci)
print("Total Time:    ", time.time() - t0)

# File cleanup
mc.fcisolver.cleanup_dice_files()
