#!/usr/bin/env python

'''
Optimize the geometry of excited states using CASSCF or CASCI

Note when optiming the excited states, states may flip and this may cause
convergence issue in geometry optimizer.
'''

from pyscf import gto
from pyscf import scf, mcscf
import copy

mol = gto.Mole()
mol.atom="N; N 1, 1.1"
mol.basis= "6-31g"
mol.build()

mf = scf.RHF(mol).run()

#
# 1. Geometry optimization over a specific state.
#
# Targeting at one excited state
mc = mcscf.CASCI(mf, 4,4)
mc.state_specific_(2)
excited_grad = mc.nuc_grad_method().as_scanner()
mol1 = excited_grad.optimizer().kernel()

# Code above is equivalent to
mc = mcscf.CASCI(mf, 4,4)
mc.fcisolver.nstates = 3
excited_grad = mc.nuc_grad_method().as_scanner(state=2)
mol1 = excited_grad.optimizer().kernel()

# CASSCF for one excited state
mc = mcscf.CASSCF(mf, 4,4)
mc.state_specific_(2)
excited_grad = mc.nuc_grad_method().as_scanner()
mol1 = excited_grad.optimizer().kernel()

#
# 2. Geometry optimization over an averaged state.
# Note the state-averaged gradients are optimized.
#
mc = mcscf.CASCI(mf, 4,4)
mc.state_average_([0.25, 0.25, 0.25, 0.25])
excited_grad = mc.nuc_grad_method().as_scanner()
mol1 = excited_grad.optimizer().kernel()

mc = mcscf.CASSCF(mf, 4,4)
mc.state_average_([0.25, 0.25, 0.25, 0.25])
excited_grad = mc.nuc_grad_method().as_scanner()
mol1 = excited_grad.optimizer().kernel()

#
# 3. Geometry optimization for mixed FCI solvers.
# Note the state-averaged gradients are optimized.
#
mc = mcscf.CASSCF(mf, 4,4)
solver1 = mc.fcisolver
solver2 = copy.copy(mc.fcisolver)
solver2.spin = 2
mc = mc.state_average_mix([solver1, solver2], (.5, .5))
excited_grad = mc.nuc_grad_method().as_scanner()
mol1 = excited_grad.optimizer().kernel()

#
# 4. Geometry optimization of the 3rd of 4 states
#
mc = mcscf.CASSCF(mf, 4,4)
mc.state_average_([0.25, 0.25, 0.25, 0.25])
excited_grad = mc.nuc_grad_method().as_scanner(state=2)
mol1 = excited_grad.optimizer().kernel()

#
# 4. Geometry optimization of the triplet state
# In a triplet-singlet state average
#
mc = mcscf.CASSCF(mf, 4,4)
solver1 = mc.fcisolver
solver2 = copy.copy(mc.fcisolver)
solver2.spin = 2
mc.state_average_mix_([solver1, solver2], (.5, .5))
excited_grad = mc.nuc_grad_method().as_scanner(state=1)
mol1 = excited_grad.optimizer().kernel()



