#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto

'''
Access molecule geometry.

Mole.natm is the total number of atoms.  It is initialized in Mole.build()
function.
'''

mol = gto.M(
    atom = '''
        O  0 0.     0
        H1 0 -2.757 2.587
        H2 0 2.757  2.587''',
    basis = 'ccpvdz',
)

for i in range(mol.natm):
    print('%s %s  charge %f  xyz %s' % (mol.atom_symbol(i),
                                        mol.atom_pure_symbol(i),
                                        mol.atom_charge(i),
                                        mol.atom_coord(i)))
