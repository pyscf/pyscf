#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Examples for transform_ci function to transform FCI wave functions with
respect to the change of orbital space:
1. Transform wavefunction wrt orbital rotation/transformation.
2. Transfer a FCI wave function from a smaller orbital space to the wavefunction
of a large orbital space.
'''

import numpy as np
import pyscf
from pyscf import fci, lib

#
# 1. Transform wavefunction wrt orbital rotation/transformation.
#

myhf1 = pyscf.M(atom='H 0 0 0; F 0 0 1.1', basis='6-31g', verbose=0).RHF().run()
e1, ci1 = fci.FCI(myhf1.mol, myhf1.mo_coeff).kernel()
print('FCI energy of mol1', e1)

myhf2 = pyscf.M(atom='H 0 0 0; F 0 0 1.2', basis='6-31g', verbose=0).RHF().run()

s12 = pyscf.gto.intor_cross('cint1e_ovlp_sph', myhf1.mol, myhf2.mol)
s12 = myhf1.mo_coeff.T.dot(s12).dot(myhf2.mo_coeff)
nelec = myhf2.mol.nelectron
ci2 = fci.addons.transform_ci(ci1, nelec, s12)

print('alpha-string, beta-string,  CI coefficients')
norb = myhf2.mo_coeff.shape[1]
for c,stra,strb in fci.addons.large_ci(ci2, norb, nelec):
    print(stra, strb, c)

#
# 2. Transfer wavefunction from small orbital space to large orbital space
#

mol = pyscf.M(atom=['H 0 0 %f'%x for x in range(6)], basis='6-31g')
mf = mol.RHF().run()
h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
h2 = pyscf.lib.einsum('pqrs,pi,qj,rk,sl->ijkl', mf.mol.intor('int2e'),
                      mf.mo_coeff, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff)

norb = 6
nelec = (3, 3)
civec = fci.FCI(mol).kernel(h1[:norb,:norb], h2[:norb,:norb,:norb,:norb], norb, nelec,
                            ecore=mol.energy_nuc())[1]

#
# Expand to 8 orbitals. It can be done with a 6x8 transformation matrix which
# maps the old orbitals to the new orbitals.
#
u = np.zeros((6, 8))
for i in range(6):
    u[i, i] = 1
civec1 = fci.addons.transform_ci(civec, nelec, u)
print(civec1.shape)  # == (56, 56)

#
# Compare to the wavefunction obtained from FCI solver. They should be very
# closed since the determinants of high excitations are less important for the
# ground state.
#
norb = 8
nelec = (3, 3)
civec2 = fci.FCI(mol).kernel(h1[:norb,:norb], h2[:norb,:norb,:norb,:norb], norb, nelec,
                              ecore=mol.energy_nuc())[1]
print(np.dot(civec1.ravel(), civec2.ravel()))

#
# The FCI coefficients are associated with the determinants in strings
# representations. We can find the determinants' addresses in each space and
# sort the determinants accordingly. This sorting algorithm is more efficient
# than the transform_ci function. However, permutation parity have to be
# handled carefully if orbitals are flipped in the other space. If there is no
# orbital flipping in the second orbital space, the code below, without
# the phase due to parity, can be used to transform the FCI wavefunction.
#
# Assuming the following orbital mappings between small space and large space:
#   small space -> large space
#       0       ->      0
#       1       ->      1
#       2       ->      2
#       3       ->      4
#       4       ->      5
#       5       ->      7
#                       3
#                       6

#
# first we get the address of each determinant of CI(6,6) in CI(6,8)
#
strsa = fci.cistring.make_strings([0,1,2,4,5,7], nelec[0])
strsb = fci.cistring.make_strings([0,1,2,4,5,7], nelec[1])
addrsa = fci.cistring.strs2addr(8, nelec[0], strsa)
addrsb = fci.cistring.strs2addr(8, nelec[1], strsa)
civec1 = np.zeros_like(civec2)
civec1[addrsa[:,None], addrsb] = civec

#
# Check against the transform_ci function
#
u = np.zeros((6, 8))
u[0,0] = 1
u[1,1] = 1
u[2,2] = 1
u[3,4] = 1
u[4,5] = 1
u[5,7] = 1
civec1_ref = fci.addons.transform_ci(civec, nelec, u)
print(np.allclose(civec1, civec1_ref))

