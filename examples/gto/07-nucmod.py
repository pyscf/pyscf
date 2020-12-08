#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Input nuclear model
'''

import numpy
from pyscf import gto

mol = gto.Mole()
mol.atom = [[8,(0, 0, 0)], ['h',(0, 1, 0)], ['H',(0, 0, 1)]]

# 0 means point nuclear model (default), 1 or 'G' means Gaussian nuclear model.
# nuclear model can be set globally
mol.nucmod = 'G'
# or specified in a dictionary, like mol.basis
mol.nucmod = {'O': 'G'}  # Use Gaussian nuclear model for oxygen

mol.build()

#
# Attribute nucmod can be initialized as a function. The function should
# take the nuc_charge as input argument and return the charge distribution
# value "zeta".
#
mol.nucmod = {'O': gto.filatov_nuc_mod, 'H': 0}
mol.build()

#
# The default Gaussian nuclear model is Dyall's nuclear model
# See L. Visscher and K. Dyall, At. Data Nucl. Data Tables, 67, 207 (1997)
# If other gaussian nuclear charge distribution are required,
#    rho(r) = nuc_charge * Norm * exp(-zeta * r^2).
# You can update the gaussian exponent zeta of particular atoms with
# mol.set_nuc_mod function
#
mol.set_nuc_mod(0, 2.)
