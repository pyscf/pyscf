#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import scf
from pyscf import gto

'''
Specify irrep_nelec to control the wave function symmetry
'''

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    ["O", (0., 0.,  0.7)],
    ["O", (0., 0., -0.7)],]

mol.basis = {'O': 'cc-pvdz'}
mol.symmetry = True
mol.build()

mf = scf.RHF(mol)
mf.irrep_nelec = {'B2g': 2, 'B3g': 2, 'B2u': 2, 'B3u': 2}
print('RHF    = %.15g' % mf.scf())


mf = scf.UHF(mol)
mf.irrep_nelec = {'B2g': (1,1), 'B3g': (1,1), 'B2u': (1,0), 'B3u': (1,0)}
print('UHF    = %.15g' % mf.scf())


mol.spin = 2 # triplet
mol.build(False, False)
mf = scf.RHF(mol)
mf.irrep_nelec = {'B2u': (1,0), 'B3u': (1,0)}
print('ROHF   = %.15g' % mf.scf())
