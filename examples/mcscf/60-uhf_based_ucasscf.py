#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, scf, mcscf

'''
UHF-CASSCF does impose the degeneracy between alpha and beta orbitals
'''

mol = gto.Mole()
mol.atom = [
    ["O", (0., 0.,  0.7)],
    ["O", (0., 0., -0.7)],]
mol.basis = 'cc-pvdz'
mol.spin = 2
mol.build()

mf = scf.UHF(mol)
print('E(UHF) = %.15g' % mf.kernel())

mc = mcscf.UCASSCF(mf, 4, (4,2))
emc1 = mc.kernel()[0]
print('* Triplet with UHF-CASSCF, E(UCAS) = %.15g' % emc1)
# Generally, 2s+1 is not 3
print('  S^2 = %.6f, 2S+1 = %.6f' % mcscf.spin_square(mc))


print('\n')
mol.spin = 0
mol.build()
mf = scf.UHF(mol)
print('E(UHF) = %.15g' % mf.kernel())

mc = mcscf.UCASSCF(mf, 4, 6)
emc1 = mc.kernel()[0]
print('* Singlet with UHF-CASSCF, E(UCAS) = %.15g' % emc1)
# Generally, 2s+1 is not 1
print('  S^2 = %.6f, 2S+1 = %.6f' % mcscf.spin_square(mc))
