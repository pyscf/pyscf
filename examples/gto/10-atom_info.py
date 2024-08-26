#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Access molecule geometry.

Mole.natm is the total number of atoms.  It is initialized in Mole.build()
function.
'''

from pyscf import gto

mol = gto.M(
    atom = '''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161''',
    basis = 'ccpvdz',
)

for i in range(mol.natm):
    print('%s %s  charge %f  xyz %s' % (mol.atom_symbol(i),
                                        mol.atom_pure_symbol(i),
                                        mol.atom_charge(i),
                                        mol.atom_coord(i)))

print("Atomic charges:")
print(mol.atom_charges())
print("Atomic coordinates (Bohr):")
print(mol.atom_coords())
print("Atomic coordinates (Angstrom):")
print(mol.atom_coords(unit='Ang'))
