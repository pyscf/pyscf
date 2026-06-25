#!/usr/bin/env python

'''
Analytical nuclear gradients of state-average CASSCF

More examples can be found in 12-excited_state_casscf_grad.py
'''

import pyscf

mol = pyscf.M(
    atom = 'N 0 0 0; N 0 0 1.2',
    basis = 'ccpvdz',
    verbose = 5)

mf = mol.RHF().run()
sa_mc = mf.CASSCF(4, 4).state_average_([0.5, 0.5]).run()
print('State-average CASSCF total energy', sa_mc.e_tot)

sa_mc_grad = sa_mc.Gradients()

# The state-averaged nuclear gradients
de_avg = sa_mc_grad.kernel()

# State-specific gradients are computed from dedicated state-specific CASSCF
# objects. Recent PySCF releases keep the state-average gradient object for the
# weighted average only.
ss_mc = mf.CASSCF(4, 4).state_specific_(0).run()

# Nuclear gradients for state 1
de_0 = ss_mc.Gradients().kernel()

# Nuclear gradients for state 2
ss_mc = mf.CASSCF(4, 4).state_specific_(1).run()
de_1 = ss_mc.Gradients().kernel()
