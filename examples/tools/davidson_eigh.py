#!/usr/bin/env python

'''
Calling davidson solver for the lowest eigenvalues of a Hermitian matrix
'''

import numpy
from pyscf import lib

n = 100
a = numpy.random.rand(n,n)
a = a + a.T

# Define the matrix-vector operation
def matvec(x):
    return a.dot(x)

# Define the preconditioner. It can be just the diagonal elements of the
# matrix.
precond = a.diagonal()

# Define the initial guess
x_init = numpy.zeros(n)
x_init[0] = 1

e, c = lib.eigh(matvec, x_init, precond, nroots=4, max_cycle=1000, verbose=5)
print('Eigenvalues', e)
