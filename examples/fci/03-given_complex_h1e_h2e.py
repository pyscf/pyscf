#!/usr/bin/env python
#
# Author: Huanchen Zhai <hczhai.ok@gmail.com>
#

'''
Solve FCI problem with given complex 1-electron and 2-electron Hamiltonian
'''

import numpy
from pyscf import fci

numpy.random.seed(12)
norb = 12
h1 = numpy.random.random((norb,norb)) + 1j * numpy.random.random((norb,norb))
h2 = numpy.random.random((norb,norb,norb,norb)) + 1j * numpy.random.random((norb,norb,norb,norb))
# Restore permutation symmetry
h1 = h1 + h1.T.conj()
h2 = h2 + h2.transpose(2, 3, 0, 1)
h2 = h2 + h2.transpose(1, 0, 3, 2).conj()
h2 = h2 + h2.transpose(3, 2, 1, 0).conj()

e, fcivec = fci.fci_dhf_slow.kernel(h1, h2, norb, nelec=8, verbose=5)

#
# A better way is to create a FCI (=FCISolver) object because FCI object offers
# more options to control the calculation.
#
cisolver = fci.fci_dhf_slow.FCI()
cisolver.max_cycle = 100
cisolver.conv_tol = 1e-8
e, fcivec = cisolver.kernel(h1, h2, norb, nelec=8, verbose=5)
e, fcivec = cisolver.kernel(h1, h2, norb, nelec=9, verbose=5)
e, fcivec = cisolver.kernel(h1, h2, norb, nelec=4, verbose=5)
