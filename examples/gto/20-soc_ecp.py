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
# SO-ECP integrals
#       <i| i/2 l U(r)|j>
# in real spherical Gaussian basis have three components, corresponding to the
# lx, ly, lz operators
#
mat_sph = mol.intor('ECPso')

# It can be transformed to spinor basis
u = mol.sph2spinor_coeff()
mat_spinor = numpy.einsum('sxy,spq,xpi,yqj->ij',
                          lib.PauliMatrices, mat_sph, u.conj(), u)

#
# Evaluating the integrals in spinor basis directly
#
mat = mol.intor('ECPso_spinor')

