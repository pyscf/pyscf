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
# SO-ECP integrals in real spherical Gaussian basis
#       <i| l U(r)|j>
# have three components, corresponding to the lx, ly, lz operators
#
mat_sph = mol.intor('ECPso')

# The H^{SO} integrals < s dot U_ECP > in spinor basis can be evaluated using
# the code as below.  Note the SOC Hamiltonian is defined ~  < s dot l U >,
# the factor 1/2 in spin operator (s = 1/2 Pauli matrix). See also the
# discussions in https://github.com/pyscf/pyscf/issues/378
s = .5 * lib.PauliMatrices
u = mol.sph2spinor_coeff()
mat_spinor = numpy.einsum('sxy,spq,xpi,yqj->ij', s, mat_sph, u.conj(), u)

#
# Evaluating the integrals in spinor basis directly
#
mat = .5 * mol.intor('ECPso_spinor')
print(abs(mat - mat_spinor).max())
