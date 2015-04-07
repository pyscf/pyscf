#!/usr/bin/env python

from pyscf import scf
from pyscf import gto

mol = gto.Mole()
mol.verbose = 5
mol.output = 'out_h2o'
mol.atom = '''
O 0 0.     0
H 0 -2.757 2.587
H 0 2.757  2.587'''
mol.basis = 'ccpvdz'
mol.symmetry = 1
mol.build()

mf = scf.RHF(mol)
mf.scf()
