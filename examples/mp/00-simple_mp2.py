#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run MP2 calculation.

MP2 for closed-shell (RHF), unrestricted (UHF) and generalized (GHF)
references.  For a restricted open-shell (ROHF) reference the restricted
open-shell MP2 (ROMP2) should be used; see 01-non_canonical_mp2.py.
'''

import pyscf

# Restricted MP2 (closed-shell reference)
mol = pyscf.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')

mf = mol.RHF().run()
pt = mf.MP2().run()

# The low-memory MP2 path gives the same correlation energy.
pt.max_memory = 1
print('Low-memory MP2 correlation energy:', pt.kernel()[0])

# Unrestricted and generalized MP2
mol = pyscf.M(
    atom = 'O 0 0 0; H 0 0 0.97',
    basis = 'ccpvdz',
    spin = 1)

mf = mol.UHF().run()
mf.MP2().run()

mf = mol.GHF().run()
mf.MP2().run()

# UMP2 accepts separate frozen-orbital lists for alpha and beta spins.
# Its default RDMs include the frozen occupied orbitals, so use full-space
# MO integrals when reconstructing the total energy.
import numpy
from pyscf import ao2mo
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
