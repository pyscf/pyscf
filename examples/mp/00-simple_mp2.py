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
mf.MP2().run()

# Unrestricted and generalized MP2
mol = pyscf.M(
    atom = 'O 0 0 0; H 0 0 0.97',
    basis = 'ccpvdz',
    spin = 1)

mf = mol.UHF().run()
mf.MP2().run()

mf = mol.GHF().run()
mf.MP2().run()
