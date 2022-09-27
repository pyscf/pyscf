#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Generate the entire FCI Hamiltonian for small system
'''

import numpy
from pyscf import fci

numpy.random.seed(1)
norb = 7
nelec = (4,4)
h1 = numpy.random.random((norb,norb))
h2 = numpy.random.random((norb,norb,norb,norb))
# Restore permutation symmetry
h1 = h1 + h1.T
h2 = h2 + h2.transpose(1,0,2,3)
h2 = h2 + h2.transpose(0,1,3,2)
h2 = h2 + h2.transpose(2,3,0,1)

# pspace function computes the FCI Hamiltonian for "primary" determinants.
# Primary determinants are the determinants which have lowest expectation
# value <H>.  np controls the number of primary determinants.
# To get the entire Hamiltonian, np should be larger than the wave-function
# size.  In this example, a (8e,7o) FCI problem has 1225 determinants.
H_fci = fci.direct_spin1.pspace(h1, h2, norb, nelec, np=1225)[1]
e_all, v_all = numpy.linalg.eigh(H_fci)

e, fcivec = fci.direct_spin1.kernel(h1, h2, norb, nelec, nroots=2,
                                    max_space=30, max_cycle=100)

print('First root:')
print('energy', e_all[0], e[0])
print('wfn overlap', v_all[:,0].dot(fcivec[0].ravel()))

print('Second root:')
print('energy', e_all[1], e[1])
print('wfn overlap', v_all[:,1].dot(fcivec[1].ravel()))
