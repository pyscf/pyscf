#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
When HOMO and LUMO orbitals are degenerated or quasi-degenerated, fill
fractional number of electrons in the degenerated orbitals.
'''

from pyscf import gto, scf
mol = gto.M(atom='O 0 0 0; O 0 0 1')
mf = scf.RHF(mol)
mf.verbose = 4
mf = scf.addons.frac_occ(mf)
mf.kernel()

