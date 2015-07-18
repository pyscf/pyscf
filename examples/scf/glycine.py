#!/usr/bin/env python

from pyscf import gto
from pyscf import scf, dft

mol = gto.Mole()
mol.verbose = 5
mol.atom = open('glycine.xyz').read()
mol.basis = '6-31g*'
mol.build()

mf = scf.RHF(mol)
scf.fast_scf(mf)

mf = dft.RKS(mol)
scf.fast_scf(mf)
