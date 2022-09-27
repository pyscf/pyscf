#!/usr/bin/env python
#
# Author: Xiao Wang <xiaowang314159@gmail.com>
#
"""
This example first runs a closed-shell periodic EOM-EE-CCSD with K-point
sampling, a.k.a. EOM-EE-KRCCSD, for singlet excited states on one of the
three provided systems.

Then a molecular EOM-EE-RCCSD based on Gamma-point Hartree-Fock is done, where
a supercell (unit cell with its nmp replicas) is used.

Excitation energies from the two sets of calculations should match.
"""

import numpy as np
from pyscf import gto, scf, cc
from pyscf.cc import eom_rccsd as mol_eom_rccsd
from pyscf.pbc import gto, scf, cc
from pyscf.pbc.cc import eom_kccsd_rhf as eom_krccsd
from pyscf.pbc.tools.pbc import super_cell
from pyscf.cc.rccsd import RCCSD

cell = gto.Cell()
cell.verbose = 7
cell.unit = 'B'
cell.max_memory = 4000  # MB

#
# Hydrogen crystal
#
# cell.a = np.eye(3) * 7.0
# cell.basis = 'sto-3g'
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
nroots_test = 4

# KRHF
kpts = cell.make_kpts(nmp)
kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None).density_fit()
ekrhf = kmf.kernel()

# KRCCSD
mycc = cc.KRCCSD(kmf)
ekrcc, t1, t2 = mycc.kernel()

# EOM-EE-KRCCSD
myeomee = eom_krccsd.EOMEESinglet(mycc)
eee, vee = myeomee.kernel(nroots=nroots_test)

# Supercell
scell = super_cell(cell, nmp)

# Gamma-point RHF based on supercell
mf = scf.RHF(scell, exxdiv=None).density_fit()
erhf = mf.kernel()

# Molecular RCCSD
mycc = RCCSD(mf)
ercc, t1, t2 = mycc.kernel()

# Molecular EOM-RCCSD
myeomee = mol_eom_rccsd.EOMEESinglet(mycc)
eee_mol, vee_mol = myeomee.kernel(nroots=nroots_test*np.prod(nmp))

eee = np.sort(np.hstack(eee[:]))
eee_mol = np.array(eee_mol)

print("PBC KRHF Energy:", ekrhf)
print("PBC RHF Energy :", erhf)
print("PBC KRCCSD Energy        :", ekrcc)
print("Mol RCCSD Energy per cell:", ercc / np.prod(nmp))
print("PBC EOM-EE-KRCCSD roots:", repr(eee))
print("Mol EOM-EE-RCCSD  roots:", repr(eee_mol))
