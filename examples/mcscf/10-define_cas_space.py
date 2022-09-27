#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, scf, mcscf

'''
Select HF orbitals for CASSCF active space

By reordering the HF orbitals using .sort_mo function, we make an initial guess
that has the pi-orbitals as the highest occupied orbitals and lowest virtual
orbitals.  With this initial guess, the CASSCF solver only correlate pi-electrons.
'''

mol = gto.M(
    verbose= 4,
    atom = 'C 0 0 0; C 0 0 1.2',
    basis = 'ccpvdz',
    symmetry = 1)

myhf = scf.RHF(mol)
myhf.kernel()
#myhf.analyze()

# 4 orbitals, 4 electrons
mycas = mcscf.CASSCF(myhf, 4, 4)
# Note sort_mo by default take the 1-based orbital indices.
mo = mycas.sort_mo([5,6,8,9])
mycas.kernel(mo)
