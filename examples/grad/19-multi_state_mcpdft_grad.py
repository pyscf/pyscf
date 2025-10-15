#!/usr/bin/env python

'''
Gradients for multi-state PDFT
'''

from pyscf import gto, scf, mcpdft

mol = gto.M(
    atom=[
        ['Li', (0., 0., 0.)],
        ['H', (0., 0., 1.7)]
    ], basis='sto-3g',
    symmetry=0  # symmetry enforcement is not recommended for MS-PDFT
)

mf = scf.RHF(mol)
mf.kernel()

mc = mcpdft.CASSCF(mf, 'tpbe', 2, 2)
mc.fix_spin_(ss=0)  # often necessary!

# For CMS-PDFT Gradients
cms = mc.multi_state([.5, .5], method='cms').run()

mc_grad = cms.nuc_grad_method()
de0 = mc_grad.kernel(state=0)
de1 = mc_grad.kernel(state=1)
print("CMS-PDFT Gradients")
print("Gradient of ground state:\n", de0)
print("Gradient of first singlet excited state:\n", de1)

# For L-PDFT Gradients
lpdft = mc.multi_state([0.5, 0.5], method='lin').run()

mc_grad = lpdft.nuc_grad_method()
de0 = mc_grad.kernel(state=0)
de1 = mc_grad.kernel(state=1)
print("L-PDFT Gradients")
print("Gradient of ground state:\n", de0)
print("Gradient of first singlet excited state:\n", de1)
