#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto

'''
Specify symmetry.

Mole.symmetry can be True/False to turn on/off the symmetry (default is off),
or a string to specify the symmetry of molecule.  symmetry_subgroup keyword
can be set to generate a subgroup of the detected symmetry.
symmetry_subgroup has no effect when an explicit label is assigned to
Mole.symmetry.

Symmetry adapted basis are stored in Mole attribute symm_orb.
'''

mol = gto.M(
    atom = 'C 0 .2 0; O 0 0 1.1',
    symmetry = True,
)
print('Symmetry %-4s, subgroup %s.' % (mol.topgroup, mol.groupname))
print('--\n')

mol = gto.M(
    atom = 'C 0 .2 0; O 0 0 1.1',
    symmetry = True,
    symmetry_subgroup = 'C2v',
)
print('Symmetry %-4s, subgroup %s.' % (mol.topgroup, mol.groupname))
print('--\n')

mol = gto.M(
    atom = 'C 0 0 0; O 0 0 1.5',
    symmetry = 'C2v',
)
print('Symmetry %-4s, subgroup %s.' % (mol.topgroup, mol.groupname))
print('If "symmetry=xxx" is specified, the symmetry for the molecule will be set to xxx')
print('--\n')

print('Symmetry adapted orbitals')
for k, ir in enumerate(mol.irrep_name):
    print('Irrep name %s  (ID %d), symm-adapted-basis shape %s' %
          (ir, mol.irrep_id[k], mol.symm_orb[k].shape))
