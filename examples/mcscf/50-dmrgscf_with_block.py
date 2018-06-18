#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Use BLOCK program as the DMRG solver and parallel DMRGSCF on different nodes.

BLOCK is invoked through system call.  Different settings.MPIPREFIX needs to
be specified for PBS and SLURM systems.
'''

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

mc = dmrgscf.DMRGSCF(mf, 8, 8)
emc = mc.kernel()[0]
print(emc)

mc = dmrgscf.DMRGSCF(mf, 8, 8)
mc.state_average_([0.5, 0.5])
mc.kernel()
print(mc.e_tot)

