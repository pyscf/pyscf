#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
CCSD and CCSD(T) density matrices
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


# When plotting CCSD density on grids, CCSD density matrices need to be
# transformed to AO basis representation.
dm1_ao = numpy.einsum('pi,ij,qj->pq', mf.mo_coeff, dm1, mf.mo_coeff.conj())

from pyscf.tools import cubegen
cubegen.density(mol, 'rho_ccsd.cube', dm1_ao)


###
#
# Compute CCSD(T) density matrices with ccsd_t-slow implementation
# (as of pyscf v1.7)
#
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
eris = mycc.ao2mo()
conv, l1, l2 = ccsd_t_lambda.kernel(mycc, eris, mycc.t1, mycc.t2)
dm1 = ccsd_t_rdm.make_rdm1(mycc, mycc.t1, mycc.t2, l1, l2, eris=eris)
dm2 = ccsd_t_rdm.make_rdm2(mycc, mycc.t1, mycc.t2, l1, l2, eris=eris)

# Open-shell UCCSD returns alpha/beta 1-RDMs and aa/ab/bb 2-RDMs.
# GCCSD instead returns spin-orbital arrays, including with GHF references.
ucc = cc.UCCSD(mf.to_uhf()).run()
ucc.solve_lambda()
dm1a, dm1b = ucc.make_rdm1()
dm2aa, dm2ab, dm2bb = ucc.make_rdm2()
gcc = cc.GCCSD(mf.to_ghf()).run()
gcc.solve_lambda()
gdm1, gdm2 = gcc.make_rdm1(), gcc.make_rdm2()
print('UCCSD density shapes', dm1a.shape, dm1b.shape, dm2ab.shape)
print('GCCSD density shapes', gdm1.shape, gdm2.shape)

# Perturbative triples also contribute to UCCSD(T) and GCCSD(T) RDMs.
# Their lambda and RDM routines use the same amplitudes and MO integrals.
from pyscf.cc import uccsd_t_lambda, uccsd_t_rdm
from pyscf.cc import gccsd_t_lambda, gccsd_t_rdm
ueris = ucc.ao2mo()
_, ul1, ul2 = uccsd_t_lambda.kernel(ucc, ueris)
udm1 = uccsd_t_rdm.make_rdm1(ucc, ucc.t1, ucc.t2, ul1, ul2, ueris)
udm2 = uccsd_t_rdm.make_rdm2(ucc, ucc.t1, ucc.t2, ul1, ul2, ueris)
geris = gcc.ao2mo()
_, gl1, gl2 = gccsd_t_lambda.kernel(gcc, geris)
gdm1_t = gccsd_t_rdm.make_rdm1(gcc, gcc.t1, gcc.t2, gl1, gl2, geris)
gdm2_t = gccsd_t_rdm.make_rdm2(gcc, gcc.t1, gcc.t2, gl1, gl2, geris)
print('UCCSD(T) and GCCSD(T) RDM shapes',
      udm1[0].shape, udm2[1].shape, gdm1_t.shape, gdm2_t.shape)
