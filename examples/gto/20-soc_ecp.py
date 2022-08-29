#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
SO-ECP Integrals

See also examples/scf/44-soc_ecp.py how to include SO-ECP in a mean-field
calculation
'''

import numpy
from pyscf import gto, lib

mol = gto.M(
    verbose = 0,
    atom = 'C 0 0 0; O 0 0 1.5',
    basis = {'C': 'crenbl', 'O': 'ccpvdz'},
    ecp = {'C': 'crenbl'}
)

#
# SO-ECP integrals in real spherical Gaussian basis computes
#       <i| 1j * l U(r)|j>
# have three components, corresponding to the lx, ly, lz operators
#
mat_sph = mol.intor('ECPso')

#
# The SOC contribution to Hamiltonian is < s dot l U(r) > (s = 1/2 Pauli matrix)
# Note the phase 1j was introduced to make ECPso integrals real. Removing it
# with multipler -1j.
#
s = .5 * lib.PauliMatrices
H_soc = -1j * lib.einsum('sxy,spq->xpyq', s, mat_sph)

# The SOC integrals < 1j * s dot l U(r) > in spinor basis transformed from
# spherical basis
s = .5 * lib.PauliMatrices
u = mol.sph2spinor_coeff()
mat_spinor = numpy.einsum('sxy,spq,xpi,yqj->ij', s, mat_sph, u.conj(), u)

#
# Evaluating the integrals in spinor basis directly
#
mat = .5 * mol.intor('ECPso_spinor')
print(abs(mat - mat_spinor).max())
