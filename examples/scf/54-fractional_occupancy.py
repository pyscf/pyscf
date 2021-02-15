#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
When HOMO and LUMO orbitals are degenerated or quasi-degenerated, fill
fractional number of electrons in the degenerated orbitals.
'''

from pyscf import gto, scf

#
# pi_x*, pi_y* degeneracy in the restricted HF calculation.  The occupancy on
# pi_x* and pi_y* are set to 1.
#
mol = gto.M(atom='O 0 0 0; O 0 0 1')
mf = scf.RHF(mol)
mf.verbose = 4
mf = scf.addons.frac_occ(mf)
mf.kernel()

#
# One electron needs to be put in the degenerated pi_x*, pi_y* orbitals.  The
# occupancy on pi_x* and pi_y* are set to 0.5.
#
mol = gto.M(atom='O 0 0 0; O 0 0 1', charge=1, spin=1)
mf = scf.rhf.RHF(mol)
mf.verbose = 4
mf = scf.addons.frac_occ(mf)
mf.kernel()

#
# In the ROHF method, one alpha electron needs to be put in the degenerated
# pi_x*, pi_y* orbitals.
#
mol = gto.M(atom='O 0 0 0; O 0 0 1', charge=2, spin=2)
mf = scf.rohf.ROHF(mol)
mf.verbose = 4
mf = scf.addons.frac_occ(mf)
mf.kernel()

#
# One alpha electron in the degenerated pi_x*, pi_y* orbitals for UHF method.
#
mol = gto.M(atom='O 0 0 0; O 0 0 1', charge=1, spin=1)
mf = scf.UHF(mol)
mf.verbose = 4
mf = scf.addons.frac_occ(mf)
mf.kernel()

