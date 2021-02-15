#!/usr/bin/env python
#
# Author: Xiao Wang <xiaowang314159@gmail.com>
#
"""
This example first runs a general periodic EOM-EE-CCSD with K-point sampling,
a.k.a. EOM-EE-KGCCSD, on one of the three provided systems.

Then a molecular EOM-EE-CCSD based on Gamma-point Hartree-Fock is done, where
a supercell (unit cell with its nmp replicas) is used.

Excitation energies from the two sets of calculations should match.
"""

import numpy as np
from pyscf.pbc import gto, scf, cc
from pyscf.pbc.cc import eom_kccsd_ghf as eom_kgccsd
from pyscf import cc as molcc
from pyscf.cc.gccsd import GCCSD
from pyscf.cc import eom_gccsd as mol_eom_gccsd
from pyscf.pbc.tools.pbc import super_cell


cell = gto.Cell()
cell.verbose = 5
cell.unit = 'B'

#
# Hydrogen crystal
#
# cell.a = np.eye(3) * 7.0
# cell.basis = '3-21g'
# cell.atom = '''
#     H 0.000000000000   0.000000000000   0.000000000000
#     H 0.000000000000   0.000000000000   1.400000000000
#     '''

#
# Helium crystal
#
cell.atom = '''
He 0.000000000000   0.000000000000   0.000000000000
He 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000
'''

#
# Diamond
#
# cell.atom = '''
# C 0.000000000000   0.000000000000   0.000000000000
# C 1.685068664391   1.685068664391   1.685068664391
# '''
# cell.basis = 'gth-szv'
# cell.pseudo = 'gth-pade'
# cell.a = '''
# 0.000000000, 3.370137329, 3.370137329
# 3.370137329, 0.000000000, 3.370137329
# 3.370137329, 3.370137329, 0.000000000'''


cell.build()

nmp = [1,1,2]
nroots_test = 2

# KRHF
kpts = cell.make_kpts(nmp)
kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None).density_fit()
ekrhf = kmf.kernel()

# KGCCSD
mycc = cc.KGCCSD(kmf)
ekgcc, t1, t2 = mycc.kernel()

# EOM-EE-KGCCSD
myeomee = eom_kgccsd.EOMEE(mycc)
eee, vee = myeomee.kernel(nroots=nroots_test)

# Supercell
scell = super_cell(cell, nmp)

# PBC Gamma-point RHF based on supercell
mf = scf.RHF(scell, exxdiv=None).density_fit()
erhf = mf.kernel()

# Molecular GCCSD
mf = scf.addons.convert_to_ghf(mf)
mycc = GCCSD(mf)
egcc, t1, t2 = mycc.kernel()

# Molecular EOM-GCCSD
myeomee = mol_eom_gccsd.EOMEE(mycc)
eee_mol, vee_mol = myeomee.kernel(nroots=nroots_test*np.product(nmp))

print("PBC KRHF Energy:", ekrhf)
print("PBC RHF Energy :", erhf)
print("PBC KGCCSD Energy        :", ekgcc)
print("Mol GCCSD Energy per cell:", egcc / np.product(nmp))
print("PBC EOMEE roots:", eee)
print("Mol EOMEE roots:", eee_mol)
