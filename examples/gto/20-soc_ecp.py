#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
SO-ECP Integrals
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
# SO-ECP integrals in real spherical Gaussian basis
#       <i| 1/2 l U(r)|j>
# have three components, corresponding to the lx, ly, lz operators
#
mat_sph = mol.intor('ECPso')

# It can be transformed to integrals in spinor basis
u = mol.sph2spinor_coeff()
mat_spinor = numpy.einsum('sxy,spq,xpi,yqj->ij',
                          lib.PauliMatrices, mat_sph, u.conj(), u)
# Note the SOC Hamiltonian is defined ~  < s dot l U >, the factor 1/2 in
# spin operator (s = 1/2 Pauli matrix) is included in mat_sph. When using 
# the spherical-GTO integrals, you may need to multiply the integrals by 2.
# See also the discussions in https://github.com/pyscf/pyscf/issues/378

#
# Evaluating the integrals in spinor basis directly
#
mat = mol.intor('ECPso_spinor')
print(abs(mat - mat_spinor).max())
