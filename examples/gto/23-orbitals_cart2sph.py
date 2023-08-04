#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Forward and backward transformations for orbital coefficients between the
representations on Cartesian GTOs and the representations on spherical GTOs.
'''

import numpy
from pyscf import gto, scf

mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvtz', verbose=0)
mo_sph = scf.RHF(mol).run().mo_coeff

#
# mo_sph is the orbital coefficients on spherical GTO basis.  Use c2s matrix
# to transform the coefficients to the representations on Cartesian GTOs.
# Note except s and p functions Cartesian GTOs are not normalized.
#
c2s = mol.cart2sph_coeff(normalized='sp')
mo_cart = numpy.dot(c2s, mo_sph)

#
# Backward transformation from Cartesian basis to spherical basis
#
c2s = mol.cart2sph_coeff(normalized='sp')
t = c2s.T.dot(mol.intor('cint1e_ovlp_cart')).dot(mo_cart)
mo_sph1 = numpy.linalg.solve(mol.intor('cint1e_ovlp_sph'), t)
print(numpy.linalg.norm(mo_sph1 - mo_sph))
