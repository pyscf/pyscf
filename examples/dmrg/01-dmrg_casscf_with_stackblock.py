#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Block (DMRG) program has two branches.  The OpenMP/MPI hybrid implementation
Block-1.5 (stackblock) code is more efficient than the old pure MPI
implementation Block-1.1 in both the computing time and memory footprint.
This example shows how to input new defined keywords for stackblock program.

Block-1.5 (stackblock) defines two new keywords memory and num_thrds.  The
rest keywords are all compatible to the old Block program.
'''

from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import dmrgscf

import os
from pyscf.dmrgscf import settings
if 'SLURMD_NODENAME' in os.environ:  # slurm system
    settings.MPIPREFIX = 'srun'
elif 'PBS_NODEFILE' in os.environ:   # PBS system
    settings.MPIPREFIX = 'mpirun'
else:  # MPI on single node
    settings.MPIPREFIX = 'mpirun -np 4'

b = 1.2
mol = gto.M(
    verbose = 4,
    atom = 'N 0 0 0; N 0 0 %f'%b,
    basis = 'cc-pvdz',
    symmetry = True,
)
mf = scf.RHF(mol)
mf.kernel()

#
# Pass stackblock keywords memory and num_thrds to fcisolver attributes
#
mc = dmrgscf.DMRGSCF(mf, 8, 8)
mc.fcisolver.memory = 4  # in GB
mc.fcisolver.num_thrds = 8
emc = mc.kernel()[0]
print(emc)

mc = dmrgscf.DMRGSCF(mf, 8, 8)
mc.state_average_([0.5, 0.5])
mc.fcisolver.memory = 4  # in GB
mc.fcisolver.num_thrds = 8
mc.kernel()
print(mc.e_tot)


