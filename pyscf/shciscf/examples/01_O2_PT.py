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
output = None,
atom = [
    ['O',(  0.000000,  0.000000, -b/2)],
    ['O',(  0.000000,  0.000000,  b/2)], ],
basis = {'O': 'ccpvdz', },
symmetry = True
)

# Create HF molecule
mf = scf.RHF( mol )
mf.conv_tol = 1e-9
mf.scf()

# Number of orbital and electrons
norb = 8
nelec = 12
dimer_atom = 'O'

mc = mcscf.CASSCF( mf, norb, nelec )
e_CASSCF = mc.mc1step()[0]

# Create SHCI molecule for just variational opt.
# Active spaces chosen to reflect valence active space.
mch = shci.SHCISCF( mf, norb, nelec )
mch.fcisolver.mpiprefix = 'mpirun -np 8' # Turns on mpi if none set in
                                           # settings.py.
mch.fcisolver.nPTiter = 0 # Turn off perturbative calc.
mch.fcisolver.sweep_iter = [ 0, 3 ]
# Setting large epsilon1 thresholds highlights improvement from perturbation.
mch.fcisolver.sweep_epsilon = [ 1e-3, 0.5e-3 ]
e_noPT = mch.mc1step()[0]

# Run a single SHCI iteration with perturbative correction.
mch.fcisolver.stochastic = False # Turns on deterministic PT calc.
mch.fcisolver.epsilon2 = 1e-8
shci.writeSHCIConfFile( mch.fcisolver, [nelec/2,nelec/2] , False )
shci.executeSHCI( mch.fcisolver )

# Open and get the energy from the binary energy file shci.e.
file1 = open(os.path.join(mch.fcisolver.runtimeDir, "%s/shci.e"%(mch.fcisolver.prefix)), "rb")
format = ['d']*1
format = ''.join(format)
e_PT = struct.unpack(format, file1.read())
print "DEBUG:   ", e_PT
file1.close()

# Comparison Calculations
del_PT = e_PT[0] - e_noPT
del_shci = e_CASSCF - e_PT

print( '\n\nEnergies for %s2 give in E_h.' % dimer_atom )
print( '=====================================' )
print( 'SHCI Variational: %6.12f' %e_noPT )
# Prints the total energy including the perturbative component.
print( 'SHCI Perturbative: %6.12f' %e_PT )
print( 'Perturbative Change: %6.12f' %del_PT )
print( 'CASSCF Total Energy: %6.12f' %e_CASSCF )
print( 'E(CASSCF) - E(SHCI): %6.12f' %del_shci )
print "Total Time:    ", time.time() - t0

# File cleanup
os.system("rm *.bkp")
os.system("rm *.txt")
os.system("rm shci.e")
os.system("rm *.dat")
os.system("rm FCIDUMP")
