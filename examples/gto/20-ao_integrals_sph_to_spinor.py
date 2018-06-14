#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#


'''
AO integrals from spherical GTO basis representation to spinor-GTO basis
representation.

Generally, the transformation requires two steps.  First is to form a
quaternion matrix (2x2 super matrix) using Pauli matrices (sigma_2x2)

        1_2x2 + 1j*sigma_2x2

Second is to contract to the Clebsch-Gordan coefficients (spherical to spinor
transformation coefficients).
'''

import numpy
from pyscf import gto, lib

mol = gto.M(atom = 'H 0 0 0 \n F 0 0 1.1',
            basis = 'cc-pvdz')

nao = mol.nao_nr()

paulix, pauliy, pauliz = lib.PauliMatrices
iden = numpy.eye(2)

################################################################################
#
# Integrals without pauli matrices.
#
################################################################################
ints_spinor = mol.intor('int1e_nuc_spinor')

ints_sph = mol.intor('int1e_nuc_sph')
ints_sph = numpy.einsum('ij,pq->ijpq', iden, ints_sph)

c = mol.sph2spinor_coeff()

ints_from_sph = numpy.einsum('ipa,ijpq,jqb->ab', numpy.conj(c), ints_sph, c)

print(abs(ints_from_sph - ints_spinor).max())

################################################################################
#
# Integrals with pauli matrices and they are real in the spherical GTO basis
# representation
#
################################################################################
ints_spinor = mol.intor('int1e_spnucsp_spinor')

# When integrals contains Pauli matrices, they spherical representation have
# four components. The first three correspond to the three Pauli matrices and
# the last one corresponds to identity of quaternion.
ints_sph = mol.intor('int1e_spnucsp_sph', comp=4)
ints_sx = ints_sph[0]
ints_sy = ints_sph[1]
ints_sz = ints_sph[2]
ints_1 = ints_sph[3]
ints_sph = (numpy.einsum('ij,pq->ijpq', 1j*paulix, ints_sx) +
            numpy.einsum('ij,pq->ijpq', 1j*pauliy, ints_sy) +
            numpy.einsum('ij,pq->ijpq', 1j*pauliz, ints_sz) +
            numpy.einsum('ij,pq->ijpq', iden     , ints_1 ))

c = mol.sph2spinor_coeff()

ints_from_sph = numpy.einsum('ipa,ijpq,jqb->ab', numpy.conj(c), ints_sph, c)

print(abs(ints_from_sph - ints_spinor).max())

################################################################################
#
# Integrals with pauli matrices and they are pure imaginary numbers in the
# spherical GTO basis representation
#
################################################################################
ints_spinor = mol.intor('int1e_cg_sa10nucsp_spinor', comp=3)

ints_sph = mol.intor('int1e_cg_sa10nucsp_sph', comp=12).reshape(3,4,nao,nao)
ints_sx = ints_sph[:,0]
ints_sy = ints_sph[:,1]
ints_sz = ints_sph[:,2]
ints_1 = ints_sph[:,3]
# In the integral <r \times \sigma V \sigma \dot p >, the coefficients of the
# quaternion basis are pure imaginary numbers.  The integral engine returns
# the imaginary part of the coefficients (thus multiplying them by factor 1j).
ints_sph = 1j * (numpy.einsum('ij,xpq->xijpq', 1j*paulix, ints_sx) +
                 numpy.einsum('ij,xpq->xijpq', 1j*pauliy, ints_sy) +
                 numpy.einsum('ij,xpq->xijpq', 1j*pauliz, ints_sz) +
                 numpy.einsum('ij,xpq->xijpq', iden     , ints_1 ))

c = mol.sph2spinor_coeff()

ints_from_sph = numpy.einsum('ipa,xijpq,jqb->xab', numpy.conj(c), ints_sph, c)

print(abs(ints_from_sph - ints_spinor).max())
