#!/usr/bin/env python
import numpy
from pyscf import gto
from pyscf import scf

mol = gto.Mole()
mol.verbose = 5
mol.atom = [
    ["H", (0., 0.,  2.5)],
    ["H", (0., 0., -2.5)],]

mol.basis = 'cc-pvdz'
mol.build()

dm = scf.hf.get_init_guess(mol)
dmb = dm.copy()
dmb[:5,:5] = 0
mf = scf.UHF(mol)
mf.scf((dm,dmb))
