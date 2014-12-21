#!/usr/bin/env python
from pyscf import scf
from pyscf import gto

mol = gto.Mole()
mol.verbose = 5
mol.output = 'out_o2'
mol.atom = [
    ["O", (0., 0.,  0.7)],
    ["O", (0., 0., -0.7)],]

mol.basis = {'O': 'cc-pvdz'}
mol.symmetry = True
mol.build()

m = scf.RHF(mol)
m.irrep_nocc['B2g'] = 2
m.irrep_nocc['B3g'] = 2
m.irrep_nocc['B2u'] = 2
m.irrep_nocc['B3u'] = 2
print('RHF    = %.15g' % m.scf())


m = scf.UHF(mol)
m.irrep_nocc_alpha['B2g'] = 1
m.irrep_nocc_alpha['B3g'] = 1
m.irrep_nocc_alpha['B2u'] = 1
m.irrep_nocc_alpha['B3u'] = 1
m.irrep_nocc_beta['B2g'] = 1
m.irrep_nocc_beta['B3g'] = 1
m.irrep_nocc_beta['B2u'] = 0
m.irrep_nocc_beta['B3u'] = 0
print('UHF    = %.15g' % m.scf())


mol.spin = 2 # triplet
mol.build(False, False)
m = scf.RHF(mol)
m.irrep_nocc_alpha['B2u'] = 1
m.irrep_nocc_alpha['B3u'] = 1
m.irrep_nocc_beta['B2u'] = 0
m.irrep_nocc_beta['B3u'] = 0
print('ROHF   = %.15g' % m.scf())
