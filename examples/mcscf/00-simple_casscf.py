#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run CASSCF calculation.
'''

import pyscf

mol = pyscf.M(
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)

myhf = mol.RHF().run()

# 6 orbitals, 8 electrons
mycas = myhf.CASSCF(6, 8).run()
#
# Note this mycas object can also be created using the APIs of mcscf module:
#
# from pyscf import mcscf
# mycas = mcscf.CASSCF(myhf, 6, 8).run()

# Natural occupancy in CAS space, Mulliken population etc.
# See also 00-simple_casci.py for the instruction of the output of analyze()
# method
mycas.verbose = 4
mycas.analyze()

# The default kernel uses the one-step CASSCF algorithm. The two-step
# alternative converges the CI problem between orbital optimization steps.
# Reuse the optimized orbitals here to avoid repeating the initial search.
two_step = myhf.CASSCF(6, 8)
two_step.mc2step(mycas.mo_coeff)

# The Newton solver optimizes CI coefficients and orbitals together using
# second derivatives. It accepts the same orbital and CI initial guesses.
newton = myhf.CASSCF(6, 8).newton()
newton.kernel(mycas.mo_coeff, mycas.ci)

# Active-active orbital rotations are redundant for a full CI active-space
# solver. internal_rotation=True is therefore unnecessary for these examples.
