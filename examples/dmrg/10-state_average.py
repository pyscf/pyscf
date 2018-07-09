#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf.dmrgscf import DMRGCI, DMRGSCF

'''
When Block code is used for active space solver, state-average calculation can
be set up in various ways.
'''

b = 1.2
mol = gto.M(
    verbose = 4,
    atom = 'N 0 0 %f; N 0 0 %f'%(-b*.5,b*.5),
    basis = 'cc-pvdz',
    symmetry = True,
)
m = scf.RHF(mol)
m.kernel()

#
# Simple state-average calculation
#
mc = DMRGSCF(m, 8, 8)
mc.state_average_([0.5, 0.5])
mc.kernel()
print(mc.e_tot)


#
# More general and/or complicated state-average calculations:
#
# Block code does not allow state average over different spin symmetry or
# spatial symmetry.  Mixing different spin or different irreducible
# representations requires multiple passes of DMRG calculations.
# See also pyscf/examples/mcscf/41-hack_state_average.py
#

#
# state-average over different spin states.
#
weights = [.25, .25, .5]  # 0.25 singlet + 0.25 singlet + 0.5 triplet
#
# dmrgsolver1 to handle singlet and dmrgsolver2 to handle triplet
#
dmrgsolver1 = DMRGCI(mol)
dmrgsolver1.nroots = 2
dmrgsolver1.spin = 0
dmrgsolver1.weights = [.5, .5]
dmrgsolver1.mpiprefix = 'mpirun -np 4'
dmrgsolver2 = DMRGCI(mol)
dmrgsolver2.spin = 2
dmrgsolver2.mpiprefix = 'mpirun -np 4'
#
# Note one must assign different scratches to the different DMRG solvers.
# Mixing the scratch directories can cause DMRG program fail.
#
dmrgsolver1.scratchDirectory = '/scratch/dmrg1'
dmrgsolver2.scratchDirectory = '/scratch/dmrg2'

mc = mcscf.CASSCF(m, 8, 8)
mcscf.state_average_mix_(mc, [dmrgsolver1, dmrgsolver2], weights)
mc.kernel()
print(mc.e_tot)




#
# state-average over states of different spatial symmetry
#
mol.build(symmetry='D2h')
m = scf.RHF(mol)
m.kernel()

weights = [.2, .4, .4]
dmrgsolver1 = DMRGCI(mol)
dmrgsolver1.wfnsym = 'Ag'
dmrgsolver1.scratchDirectory = '/scratch/dmrg1'
dmrgsolver2 = DMRGCI(mol)
dmrgsolver2.wfnsym = 'B1g'
dmrgsolver2.scratchDirectory = '/scratch/dmrg2'
dmrgsolver3 = DMRGCI(mol)
dmrgsolver3.wfnsym = 'B1u'
dmrgsolver3.scratchDirectory = '/scratch/dmrg3'

mc = mcscf.CASSCF(m, 8, 8)
mcscf.state_average_mix_(mc, [dmrgsolver1, dmrgsolver2, dmrgsolver3], weights)
mc.kernel()
print(mc.e_tot)
