#!/usr/bin/env python

'''
Analytical nuclear gradients of state-average CASCI

More examples can be found in 11-excited_state_casci_grad.py
'''

import pyscf

mol = pyscf.M(
    atom = 'N 0 0 0; N 0 0 1.2',
    basis = 'ccpvdz',
    verbose = 5)

mf = mol.RHF().run()
sa_mc = mf.CASCI(4, 4).state_average_([0.5, 0.5]).run()
print('State-average CASCI total energy', sa_mc.e_tot)

sa_mc_grad = sa_mc.Gradients()

# The state-averaged nuclear gradients
de_avg = sa_mc_grad.kernel()

# State-specific gradients are computed from the multi-root CASCI object.
# Recent PySCF releases keep the state-average gradient object for the weighted
# average only.
ss_mc = mf.CASCI(4, 4)
ss_mc.fcisolver.nroots = 2
ss_mc.run()

# Nuclear gradients for state 1
de_0 = ss_mc.Gradients().kernel(state=0)

# Nuclear gradients for state 2
de_1 = ss_mc.Gradients().kernel(state=1)
