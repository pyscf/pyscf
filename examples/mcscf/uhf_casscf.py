#!/usr/bin/env python
import numpy
from pyscf import scf
from pyscf import gto
from pyscf import mcscf
from pyscf import fci


mol = gto.Mole()
mol.verbose = 5
mol.output = 'out_o2_uhf'
mol.atom = [
    ["O", (0., 0.,  0.7)],
    ["O", (0., 0., -0.7)],]

mol.basis = {'O': 'cc-pvdz',
             'C': 'cc-pvdz',}
mol.spin = 2
mol.build()

m = scf.UHF(mol)
print('UHF     = %.15g' % m.scf())

mc = mcscf.CASSCF(mol, m, 4, (4,2))
mc.stdout.write('** Triplet with UHF-CASSCF**\n')
emc1 = mc.mc1step()[0]
print('CASSCF = %.15g' % emc1)
# Generally, 2s+1 is not 3
print('s^2 = %.6f, 2s+1 = %.6f' % mcscf.spin_square(mc))


mol.spin = 0
mol.build()
m = scf.UHF(mol)
print('\n')
print('UHF     = %.15g' % m.scf())

mc = mcscf.CASSCF(mol, m, 4, 6)
mc.stdout.write('** Singlet with UHF-CASSCF **\n')
emc1 = mc.mc1step()[0]

print('CASSCF = %.15g' % emc1)
# Generally, 2s+1 is not 1
print('s^2 = %.6f, 2s+1 = %.6f' % mcscf.spin_square(mc))
