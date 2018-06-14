#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import gto, scf, ao2mo, mcscf

'''
MO integrals needed by spin-orbit coupling
'''

mol = gto.M(
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)
myhf = scf.RHF(mol)
myhf.kernel()
mycas = mcscf.CASSCF(myhf, 6, 8) # 6 orbital, 8 electron
mycas.kernel()

# CAS space orbitals
cas_orb = mycas.mo_coeff[:,mycas.ncore:mycas.ncore+mycas.ncas]

# 2-electron part spin-same-orbit coupling
#       [ijkl] = <ik| p1 1/r12 cross p1 |jl>
# JCP, 122, 034107 Eq (3) = h2 * (-i)
# For simplicty, we didn't consider the permutation symmetry k >= l, therefore aosym='s1'
h2 = ao2mo.kernel(mol, cas_orb, intor='int2e_p1vxp1_sph', comp=3, aosym='s1')
print('SSO 2e integrals shape %s' % str(h2.shape))

# 1-electron part for atom A
#       <i| p 1/|r-R_A| cross p |j>
# JCP, 122, 034107 Eq (2) = h1 * (iZ_A)
mol.set_rinv_origin(mol.atom_coord(1))  # set the gauge origin on second atom
h1 = numpy.einsum('xpq,pi,qj->xij', mol.intor('int1e_prinvxp_sph'), cas_orb, cas_orb)
print('1e integral shape %s' % str(h1.shape))

