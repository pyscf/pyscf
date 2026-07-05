#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Break spin symmetry for UHF/UKS by initial guess.

See also examples/dft/32-broken_symmetry_dft.py
     and examples/scf/56-h2_symm_breaking.py
'''

import numpy
from pyscf import gto
from pyscf import scf

mol = gto.Mole()
mol.verbose = 4
mol.atom = [
    ["H", (0., 0.,  2.5)],
    ["H", (0., 0., -2.5)],]
mol.basis = 'cc-pvdz'
mol.build()

mf = scf.UHF(mol)

#
# We can modify the initial guess DM to break spin symmetry.
# For UHF/UKS calculation,  the initial guess DM can be a two-item list
# (alpha,beta).  Assigning alpha-DM and beta-DM to different value can break
# the spin symmetry.
#
# In the following example, the function get_init_guess returns the
# superposition of atomic density matrices in which the alpha and beta
# components are degenerated.  The degeneracy are destroyed by zeroing out the
# beta 1s,2s components.
#
dm_alpha, dm_beta = mf.get_init_guess()
dm_beta[:2,:2] = 0
dm = (dm_alpha,dm_beta)
mf.kernel(dm)

#
# Alternative: use the built-in HOMO-LUMO rotation (breaksym='mix').
# Instead of zeroing atom blocks, this rotates the alpha and beta HOMOs
# by +/-45 degrees into the LUMO:
#   alpha HOMO -> (HOMO + LUMO) / sqrt(2)
#   beta  HOMO -> (HOMO - LUMO) / sqrt(2)
# The orbitals remain delocalized over the full molecule, giving a smoother
# symmetry break that is less likely to collapse back to the RHF solution.
# This option also works for UKS.
#
mf2 = scf.UHF(mol)
mf2.init_guess_breaksym = 'mix'
mf2.kernel()
