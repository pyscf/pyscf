#!/usr/bin/env python
#
# Author: Sheng Guo <shengg@princeton.edu>
#         Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import dmrgscf
from pyscf import mrpt

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


##################################################
#
# State-average
#
##################################################

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
mc.fcisolver = dmrgscf.dmrgci.DMRGCI(mol, maxM=200)
mc.fcisolver.nroots = 2
e_0 = mc.kernel()[0]

#
# Computing NEVPT2 based on state-specific DMRG-CASCI calculation
#
dmrg_nevpt_e1 = mrpt.NEVPT(mc, root=0).kernel()
dmrg_nevpt_e2 = mrpt.NEVPT(mc, root=1).kernel()
print('DE = %.9g' % (e_0[1]+dmrg_nevpt_e2) - (e_0[0]+dmrg_nevpt_e1))



##################################################
#
# State-specific
#
##################################################

#
# Optimize orbitals for first excited state
#
mc = dmrgscf.dmrgci.DMRGSCF(m, 8, 8)
mc.state_specific_(1)
e_0 = mc.kernel()[0]

#
# Run DMRGCI for 2 excited states
#
mc = mcscf.CASCI(m, 8, 8)
mc.fcisolver = dmrgscf.dmrgci.DMRGCI(mol, maxM=200)
mc.fcisolver.nroots = 2
e_0 = mc.kernel()[0]

#
# Computing NEVPT2 based on state-specific DMRG-CASCI calculation
#
dmrg_nevpt_e1 = mrpt.NEVPT(mc, root=0).kernel()
dmrg_nevpt_e2 = mrpt.NEVPT(mc, root=1).kernel()
print('DE = %.9g' % (e_0[1]+dmrg_nevpt_e2) - (e_0[0]+dmrg_nevpt_e1))
