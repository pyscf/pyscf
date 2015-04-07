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
m.irrep_nelec = {'B2g': 2, 'B3g': 2, 'B2u': 2, 'B3u': 2}
print('RHF    = %.15g' % m.scf())


m = scf.UHF(mol)
m.irrep_nelec = {'B2g': (1,1), 'B3g': (1,1), 'B2u': (1,0), 'B3u': (1,0)}
print('UHF    = %.15g' % m.scf())


mol.spin = 2 # triplet
mol.build(False, False)
m = scf.RHF(mol)
m.irrep_nelec = {'B2u': (1,0), 'B3u': (1,0)}
print('ROHF   = %.15g' % m.scf())
