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

# Frozen occupied orbitals are included by default in the returned RDMs.
# Contract them with full-space integrals, including the frozen core.
frozen_cc = cc.CCSD(mf, frozen=1).run()
frozen_dm1, frozen_dm2 = frozen_cc.make_rdm1(), frozen_cc.make_rdm2()
frozen_energy = (numpy.einsum('pq,qp', h1, frozen_dm1)
                 + .5 * numpy.einsum('pqrs,pqrs', eri, frozen_dm2)
                 + mol.energy_nuc())
print('Frozen-core RDM energy and CCSD energy', frozen_energy, frozen_cc.e_tot)
print('RDM particle number', numpy.trace(frozen_dm1), mol.nelectron)
print('Partial-trace error', abs(numpy.einsum('pqrr->pq', frozen_dm2)
                                - (mol.nelectron - 1) * frozen_dm1.T).max())

# The same full-space contraction applies when UCCSD is run with frozen=1;
# its default RDMs also restore frozen occupied orbitals.
# UCCSD uses separate alpha/beta orbitals. The mixed-spin interaction has
# weight one because both ab and ba contributions are represented by dm2ab.
ca, cb = ucc.mo_coeff
ha = ca.conj().T @ mf.get_hcore() @ ca
hb = cb.conj().T @ mf.get_hcore() @ cb
eriaa = ao2mo.kernel(mol, ca, compact=False).reshape(dm2aa.shape)
eriab = ao2mo.general(mol, (ca, ca, cb, cb), compact=False).reshape(dm2ab.shape)
eribb = ao2mo.kernel(mol, cb, compact=False).reshape(dm2bb.shape)
uenergy = (numpy.einsum('pq,qp', ha, dm1a)
           + numpy.einsum('pq,qp', hb, dm1b)
           + .5 * numpy.einsum('pqrs,pqrs', eriaa, dm2aa)
           + numpy.einsum('pqrs,pqrs', eriab, dm2ab)
           + .5 * numpy.einsum('pqrs,pqrs', eribb, dm2bb)
           + mol.energy_nuc())
print('UCCSD RDM energy and CCSD energy', uenergy, ucc.e_tot)

# This GHF reference was converted from RHF and is collinear. Each MO has
# a definite spin, allowing spatial integrals to be filtered with orbspin.
# For general spin-mixed GHF orbitals, this shortcut is not valid.
orbspin = gcc.mo_coeff.orbspin
nao = mol.nao_nr()
c = gcc.mo_coeff[:nao] + gcc.mo_coeff[nao:]
geri = ao2mo.kernel(mol, c, compact=False).reshape(gdm2.shape)
spin_allowed = orbspin[:, None] == orbspin[None, :]
geri *= spin_allowed[:, :, None, None] * spin_allowed[None, None, :, :]
gh1 = c.conj().T @ mf.get_hcore() @ c
gh1 *= spin_allowed
genergy = (numpy.einsum('pq,qp', gh1, gdm1)
           + .5 * numpy.einsum('pqrs,pqrs', geri, gdm2)
           + mol.energy_nuc())
print('GCCSD RDM energy and CCSD energy', genergy, gcc.e_tot)

# Trace out spin in the triples-corrected RDMs. Summing alpha and beta
# spatial matrices assumes matching spatial orbitals, as in this RHF example.
ia, ib = numpy.where(orbspin == 0)[0], numpy.where(orbspin == 1)[0]
spin_sum1 = gdm1_t[numpy.ix_(ia, ia)] + gdm1_t[numpy.ix_(ib, ib)]
mixed = gdm2_t[numpy.ix_(ia, ia, ib, ib)]
spin_sum2 = (gdm2_t[numpy.ix_(ia, ia, ia, ia)]
             + gdm2_t[numpy.ix_(ib, ib, ib, ib)]
             + mixed + mixed.transpose(2, 3, 0, 1))
print('Spin-summed triples RDM shapes', spin_sum1.shape, spin_sum2.shape)
print('Triples RDM energy contraction',
      numpy.einsum('pq,qp', gh1, gdm1_t)
      + .5 * numpy.einsum('pqrs,pqrs', geri, gdm2_t) + mol.energy_nuc())
