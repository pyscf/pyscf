#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
CCSD density matrix
'''

import numpy
from pyscf import gto, scf, cc, ao2mo

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')
mf = scf.RHF(mol).run()
mycc = cc.CCSD(mf).run()

#
# CCSD density matrix in MO basis
#
dm1 = mycc.make_rdm1()
dm2 = mycc.make_rdm2()

#
# CCSD energy based on density matrices
#
h1 = numpy.einsum('pi,pq,qj->ij', mf.mo_coeff.conj(), mf.get_hcore(), mf.mo_coeff)
nmo = mf.mo_coeff.shape[1]
eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape([nmo]*4)
E = numpy.einsum('pq,qp', h1, dm1)
# Note dm2 is transposed to simplify its contraction to integrals
E+= numpy.einsum('pqrs,pqrs', eri, dm2) * .5
E+= mol.energy_nuc()
print('E(CCSD) = %s, reference %s' % (E, mycc.e_tot))


# When plotting CCSD density on the grids, CCSD density matrices need to be
# first transformed to AO basis.
dm1_ao = numpy.einsum('pi,ij,qj->pq', mf.mo_coeff, dm1, mf.mo_coeff.conj())

from pyscf.tools import cubegen
cubegen.density(mol, 'rho_ccsd.cube', dm1_ao)

