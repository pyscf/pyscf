#!/usr/bin/env python
#
# Author: Sheng Guo <shengg@princeton.edu>
#         Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import dmrgscf
from pyscf.mrpt.nevpt2 import sc_nevpt

'''
DMRG-CASCI then DMRG-NEVPT2 calculation for excited states.
'''

b = 1.4
mol = gto.Mole()
mol.build(
    verbose = 4,
    output = 'fci_nevpt2.out',
    atom = [['H', (0.,0.,i-3.5)] for i in range(8)],
    basis = '6-31g',
)
m = scf.RHF(mol)
m.kernel()

#
# Optimize orbitals with state-average DMRG-CASSCF
#
mc = dmrgscf.dmrgci.DMRGSCF(m, 8, 8)
mc.state_average_([.5,.5])
e_0 = mc.kernel()[0]

#
# Run DMRGCI for 2 excited states
#
mc = mcscf.CASCI(m, 8, 8)
mc.fcisolver = dmrgscf.drmgci.DMRGCI(mol, maxM=200)
mc.fcisolver.nroots = 2
e_0 = mc.kernel()[0]

#
# Computing NEVPT2 based on state-specific DMRG-CASCI calculation
#
dmrg_nevpt_e1 = sc_nevpt(mc, ci=mc.ci[0])
dmrg_nevpt_e2 = sc_nevpt(mc, ci=mc.ci[1])
