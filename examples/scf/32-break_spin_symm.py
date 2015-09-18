#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import gto
from pyscf import scf

'''
Break spin symmetry for UHF/UKS by initial guess.
'''

mol = gto.Mole()
mol.verbose = 5
mol.atom = [
    ["H", (0., 0.,  2.5)],
    ["H", (0., 0., -2.5)],]
mol.basis = 'cc-pvdz'
mol.build()

mf = scf.UHF(mol)
dm = mf.get_init_guess()
dm[1][:5,:5] = 0
mf.scf(dm)
