#!/usr/bin/env python

"""Frozen-orbital UMP2 and total-energy reconstruction from density matrices."""

import numpy
import pyscf
from pyscf import ao2mo

mol = pyscf.M(
    atom = 'O 0 0 0; H 0 0 0.97',
    basis = 'ccpvdz',
    spin = 1)

# UMP2 accepts separate frozen-orbital lists for alpha and beta spins.
# Its default RDMs include the frozen occupied orbitals, so use full-space
# MO integrals when reconstructing the total energy.
umf = mol.UHF().run()
upt = umf.MP2(frozen=([0], [0])).run()
dm1a, dm1b = upt.make_rdm1()
dm2aa, dm2ab, dm2bb = upt.make_rdm2()
ca, cb = umf.mo_coeff
ha = ca.conj().T @ umf.get_hcore() @ ca
hb = cb.conj().T @ umf.get_hcore() @ cb
aa = ao2mo.kernel(mol, ca, compact=False).reshape(dm2aa.shape)
ab = ao2mo.general(mol, (ca, ca, cb, cb), compact=False).reshape(dm2ab.shape)
bb = ao2mo.kernel(mol, cb, compact=False).reshape(dm2bb.shape)
e_rdm = (numpy.einsum('pq,qp', ha, dm1a) + numpy.einsum('pq,qp', hb, dm1b)
         + .5 * numpy.einsum('pqrs,pqrs', aa, dm2aa)
         + numpy.einsum('pqrs,pqrs', ab, dm2ab)
         + .5 * numpy.einsum('pqrs,pqrs', bb, dm2bb) + mol.energy_nuc())
print('UMP2 density contraction and total energy', e_rdm, upt.e_tot)
# See 10-dfump2.py for density fitting and 01-non_canonical_mp2.py for
# noncanonical references. Out-of-core equivalence is checked in test_ump2.py.
