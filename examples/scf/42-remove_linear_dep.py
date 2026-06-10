#!/usr/bin/env python

'''
There is a list of methods of the SCF object one can modify to control the SCF
calculations. You can find these methods in function pyscf.scf.hf.kernel.
This example shows how to remove basis linear dependency by modifying the SCF
eig method.

The pyscf.scf.addons module provides a function remove_linear_dep_ to remove
basis linear dependency in a similar manner, but it also can handle
pathological linear dependencies via the partial Cholesky procedure.
'''

from functools import reduce
import numpy
import numpy.linalg
from pyscf import gto, scf, mcscf

# Adjusting the following settings to control
scf.hf.remove_overlap_zero_eigenvalue = True
scf.hf.overlap_zero_eigenvalue_threshold = 1e-6

mol = gto.M(atom=['H 0 0 %f'%i for i in range(10)], unit='Bohr',
            basis='ccpvtz')
mf = mol.RHF()
mf.verbose = 4
mf.kernel()

#
# The CASSCF solver takes the HF orbital as initial guess.  The MCSCF is solved
# in a reduced space, with (0 core, 10 active, 127 external) orbitals.
#
mc = mcscf.CASSCF(mf, 10, 10)
mc.verbose = 4
mc.kernel()



#
# Linearly dependent orbitals are automatically removed in symmetry adapted
# calculations.
#
mol = gto.M(atom=['H 0 0 %f'%i for i in range(10)], unit='Bohr',
            basis='ccpvtz', symmetry=1)

mf = mol.RHF()
mf.verbose = 4
mf.kernel()

mc = mcscf.CASSCF(mf, 10, 10)
mc.verbose = 4
mc.kernel()
