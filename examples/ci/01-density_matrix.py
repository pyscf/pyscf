#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
CISD density matrix
'''

import numpy
from pyscf import gto, scf, ci, ao2mo

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')
mf = scf.RHF(mol).run()
myci = ci.CISD(mf).run()

#
# CISD density matrix in MO basis
#
dm1 = myci.make_rdm1()
dm2 = myci.make_rdm2()

#
# CISD energy based on density matrices
#
h1 = numpy.einsum('pi,pq,qj->ij', mf.mo_coeff.conj(), mf.get_hcore(), mf.mo_coeff)
nmo = mf.mo_coeff.shape[1]
eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape([nmo]*4)
E = numpy.einsum('pq,qp', h1, dm1)
# Note dm2 is transposed to simplify its contraction to integrals
E+= numpy.einsum('pqrs,pqrs', eri, dm2) * .5
E+= mol.energy_nuc()
print('E(CCSD) = %s, reference %s' % (E, myci.e_tot))

#
# CISD 1-particle transition density matrix
#
myci.nroots = 4
myci.kernel()
# Transition from ground state to 3rd excited state
t_dm1 = myci.trans_rdm1(myci.ci[3], myci.ci[0])

# To AO representation
t_dm1 = numpy.einsum('pi,ij,qj->pq', mf.mo_coeff, t_dm1, mf.mo_coeff.conj())

charge_center = (numpy.einsum('z,zx->x', mol.atom_charges(), mol.atom_coords())
                 / mol.atom_charges().sum())
with mol.with_common_origin(charge_center):
    t_dip = numpy.einsum('xij,ji->x', mol.intor('int1e_r'), t_dm1)
print(t_dip)
