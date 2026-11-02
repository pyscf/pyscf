#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Use selected CI solver for the given 1-electron and 2-electron Hamiltonians
'''

import numpy
from pyscf import fci

numpy.random.seed(12)
norb = 6
h1 = numpy.random.random((norb,norb))
h2 = numpy.random.random((norb,norb,norb,norb))
# Restore permutation symmetry
h1 = h1 + h1.T
h2 = h2 + h2.transpose(1,0,2,3)
h2 = h2 + h2.transpose(0,1,3,2)
h2 = h2 + h2.transpose(2,3,0,1)

#
# A quick run with selected_ci.kernel
#
e, fcivec = fci.selected_ci.kernel(h1, h2, norb, 8, verbose=5)

# Freeze the selected determinant space for a subsequent diagonalization.
# The selected vector stores alpha and beta determinant strings in _strs.
solver = fci.SCI()
e_fixed, c_fixed = fci.selected_ci.kernel_fixed_space(
    solver, h1, h2, norb, 8, fcivec._strs)
print('Fixed-space energy:', e_fixed)

# Selected-CI density matrices can be compared directly with conventional FCI.
dm1 = solver.make_rdm1(c_fixed, norb, 8)
dm2 = solver.make_rdm2(c_fixed, norb, 8)
print('RDM shapes:', dm1.shape, dm2.shape)
print('Selected-CI spin:', solver.spin_square(c_fixed, norb, 8)[0])
print('Expanded FCI-vector spin:',
      fci.spin_op.spin_square0(fci.selected_ci.to_fci(c_fixed, norb, 8),
                               norb, 8)[0])
print('Largest selected-CI configurations:', solver.large_ci(c_fixed, norb, 8))

#
# More options of SCI object to control the calculation.
#
cisolver = fci.SCI()
cisolver.max_cycle = 100
cisolver.conv_tol = 1e-8
e, fcivec = cisolver.kernel(h1, h2, norb, 8)
e, fcivec = cisolver.kernel(h1, h2, norb, (5,4))  # 5 alpha, 4 beta electrons
e, fcivec = cisolver.kernel(h1, h2, norb, (3,1))  # 3 alpha, 1 beta electrons

#
# If the ground state is singlet, you can use spin0 solver.  Spin symmetry is
# considered in spin0 solver to reduce cimputational cost.
#
cisolver = fci.selected_ci_spin0.SCI()
cisolver.verbose = 5
e, fcivec = cisolver.kernel(h1, h2, norb, 8)
print('Spin-adapted selected-CI energy:', e)
# The spin-adapted and full direct-spin-0 solvers use the same Hamiltonian.
e_fci, _ = fci.direct_spin0.kernel(h1, h2, norb, 8)
print('Difference from direct-spin-0 FCI:', e - e_fci)
