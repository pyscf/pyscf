#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import dmrgscf

import os
from pyscf.dmrgscf import settings
if 'SLURMD_NODENAME' in os.environ:  # slurm system
    settings.MPIPREFIX = 'srun'
elif 'PBS_NODEFILE' in os.environ:   # PBS system
    nproc = len(open(os.environ['PBS_NODEFILE']).readlines())
    settings.MPIPREFIX = 'mpirun -np %d --host %s' % (nproc, os.environ['PBS_NODEFILE'])
else:  # MPI on same node
    settings.MPIPREFIX = 'mpirun -np 4'

'''
Use BLOCK program as the DMRG solver and parallel DMRGSCF on different nodes.

BLOCK is invoked through system call.  Different MPIPREFIX needs to be
specified for PBS and SLURM systems.
'''

b = 1.2
mol = gto.Mole()
mol.build(
    verbose = 4,
    atom = [['N', (0.,0.,0.)], ['N', (0.,0.,b)]],
    basis = 'cc-pvdz',
    symmetry = True,
)
m = scf.RHF(mol)
m.kernel()

mc = dmrgscf.dmrgci.DMRGSCF(m, 8, 8)
emc = mc.kernel()[0]
print(emc)

