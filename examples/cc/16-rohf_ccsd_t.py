#!/usr/bin/env python

'''
UCCSD(T) with an ROHF reference.

The (T) correction uses only the diagonal of the occupied-occupied and
virtual-virtual blocks of the Fock matrix, so it needs semicanonical orbitals.
ROHF orbitals are not semicanonical: the alpha and beta Fock matrices are not
diagonal in these blocks.  For an ROHF reference, UCCSD therefore rotates the
orbitals before CCSD, within the active occupied and within the active virtual
space of each spin.  Frozen orbitals are not rotated (the Molpro convention).
The CCSD energy does not depend on these rotations; the (T) correction does.
'''

import pyscf

mol = pyscf.M(
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)

mf = mol.ROHF().run()

# semicanonical = None (default): semicanonical orbitals for ROHF references
# only.  mycc.mo_coeff holds the rotated orbitals after the calculation.
mycc = pyscf.cc.UCCSD(mf, frozen=2).run()
et = mycc.ccsd_t()
print('ROHF-UCCSD(T) correlation energy', mycc.e_corr + et)

# semicanonical = False keeps the ROHF orbitals.  CCSD gives the same energy,
# but (T) is not valid for these orbitals and prints a warning.
mycc = pyscf.cc.UCCSD(mf, frozen=2)
mycc.semicanonical = False
mycc.run()
print('CCSD correlation energy with ROHF orbitals', mycc.e_corr)
mycc.ccsd_t()
