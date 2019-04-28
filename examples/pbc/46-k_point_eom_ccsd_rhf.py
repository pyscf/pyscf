#!/usr/bin/env python
#
# Author: Xiao Wang <xiaowang314159@gmail.com>
#
"""Module documentation goes here.
More discriptions ...
"""

import numpy as np
from pyscf import gto, scf, cc
from pyscf.cc import eom_rccsd as eom_rccsd
from pyscf.cc import eom_uccsd as eom_uccsd
from pyscf.cc import eom_gccsd as eom_gccsd
from pyscf.pbc.cc import eom_kccsd_ghf as eom_kgccsd
from pyscf import cc as molcc
from pyscf.cc.gccsd import GCCSD
from pyscf.cc import eom_gccsd as mol_eom_gccsd
from pyscf.pbc.tools.pbc import super_cell


mol = gto.Mole()
mol.verbose = 1
mol.unit = 'B'

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
# Helium dimer
#
mol.basis = '3-21g'
mol.atom = '''
    He 0.000000000000   0.000000000000   0.000000000000
    He 0.000000000000   0.000000000000   1.400000000000
    '''
#
# Hydrogen molecule
#
# mol.basis = '3-21g'
# mol.atom = '''
#     H 0.000000000000   0.000000000000   0.000000000000
#     H 0.000000000000   0.000000000000   1.400000000000
#     '''
#
# Helium crystal
#
# mol.atom = '''
# He 0.000000000000   0.000000000000   0.000000000000
# He 1.685068664391   1.685068664391   1.685068664391
# '''
# mol.basis = [[0, (1., 1.)], [0, (.5, 1.)]]

# cell.a = '''
# 0.000000000, 3.370137329, 3.370137329
# 3.370137329, 0.000000000, 3.370137329
# 3.370137329, 3.370137329, 0.000000000
# '''

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


mol.build()

# RHF
mf_rhf = scf.RHF(mol)
mf_rhf.kernel()

# RCCSD
mycc = cc.RCCSD(mf_rhf)
ecc, t1, t2 = mycc.kernel()

# EOM-RCCSD
# myeom = eom_rccsd.EOMEE(mycc)
# print("EOMEE-RCCSD vector size:", myeom.vector_size())
# result = myeom.kernel(nroots=99)
#
# myeom = eom_rccsd.EOMEESpinFlip(mycc)
# print("EOMEESpinFlip vector size:", myeom.vector_size())
# result_sf = myeom.kernel(nroots=99)

myeom = eom_rccsd.EOMEESinglet(mycc)
print("\nEOMEESinglet vector size:", myeom.vector_size())
result_singlet = myeom.kernel(nroots=99)

myeom = eom_rccsd.EOMEA(mycc)
print("\nEOMEA:")
result_ea = myeom.kernel(nroots=99)

# myeom = eom_rccsd.EOMEETriplet(mycc)
# print("EOMEETriplet vector size:", myeom.vector_size())
# result_triplet = myeom.kernel(nroots=99)
#
# print("EOMEE-RCCSD roots:", result[0])
# print("EOMEE-RCCSD Spin Flip roots:", result_sf[0])
print("EOMEE-RCCSD Singlet roots:", result_singlet[0])
# print("EOMEE-RCCSD Triplet roots:", result_triplet[0])



# # UHF
# mf = scf.UHF(mol)
# mf.kernel()
#
# # UCCSD
# mycc = cc.UCCSD(mf)
# mycc.kernel()
#
# # EOM-UCCSD
# myeom = eom_uccsd.EOMEE(mycc)
# print("EOMEE-UCCSD vector size:", myeom.vector_size())
# result = myeom.kernel(nroots=4)
# print("EOMEE-UCCSD roots:", result[0])
#
# # GCCSD
# mycc = cc.GCCSD(mf_rhf)
# mycc.kernel()
#
# # GCCSD
# myeom = eom_gccsd.EOMEE(mycc)
# print("EOMEE-GCCSD vector size:", myeom.vector_size())
# result_gccsd = myeom.kernel(nroots=99)
# print("EOMCC-GCCSD roots:", result_gccsd[0])

# nmp = [1,1,2]
# nroots_test = 2
#
# # KRHF
# kpts = cell.make_kpts(nmp)
# kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None).density_fit()
# ekrhf = kmf.kernel()
#
# # KGCCSD
# mycc = cc.KGCCSD(kmf)
# ekgcc, t1, t2 = mycc.kernel()
#
# # EOM-EE-KGCCSD
# myeomee = eom_kgccsd.EOMEE(mycc)
# eee, vee = myeomee.kernel(nroots=nroots_test)
#
# # Supercell
# scell = super_cell(cell, nmp)
#
# # PBC Gamma-point RHF based on supercell
# mf = scf.RHF(scell, exxdiv=None).density_fit()
# erhf = mf.kernel()
#
# # Molecular GCCSD
# mf = scf.addons.convert_to_ghf(mf)
# mycc = GCCSD(mf)
# egcc, t1, t2 = mycc.kernel()
#
# # Molecular EOM-GCCSD
# myeomee = mol_eom_gccsd.EOMEE(mycc)
# eee_mol, vee_mol = myeomee.kernel(nroots=nroots_test*np.product(nmp))
#
# print("PBC KRHF Energy:", ekrhf)
# print("PBC RHF Energy :", erhf)
# print("PBC KGCCSD Energy        :", ekgcc)
# print("Mol GCCSD Energy per cell:", egcc / np.product(nmp))
# print("PBC EOMEE roots:", eee)
# print("Mol EOMEE roots:", eee_mol)