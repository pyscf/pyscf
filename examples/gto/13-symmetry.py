#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto

'''
Specify symmetry.

Mole.symmetry can be True/False to turn on/off the symmetry (default is off),
or a string to specify the symmetry of molecule.  symmetry_subgroup keyword
can be set to generate a subgroup of the dectected symmetry.
symmetry_subgroup has no effect when an explicit label is assigned to
Mole.symmetry.

Symmetry adapted basis are stored in Mole attribute symm_orb.
'''

mol = gto.M(
    atom = 'C 0 .2 0; O 0 0 1.1',
    symmetry = True,
)
print('Symmetry %-4s, using subgroup %s.  The molecule geometry is changed' %
      (mol.topgroup, mol.groupname))
for x in mol._atom:
    print(x)
print('--\n')

mol = gto.M(
    atom = 'C 0 .2 0; O 0 0 1.1',
    symmetry = True,
    symmetry_subgroup = 'C2v',
)
print('Symmetry %-4s, using subgroup %s.  The molecule geometry is changed' %
      (mol.topgroup, mol.groupname))
for x in mol._atom:
    print(x)
print('--\n')

try:
    mol = gto.M(
        atom = 'C 0 .2 0; O 0 0 1.1',
        symmetry = 'C2v',
    )
except RuntimeWarning as e:
    print('Unable to identify the symmetry with the input geometry.  Error msg:')
    print(e)
print('--\n')

mol = gto.M(
    atom = 'C 0 0 0; O 0 0 1.5',
    symmetry = 'C2v',
)
print('Symmetry %-4s, using subgroup %s.' % (mol.topgroup, mol.groupname))
print('If "symmetry=string" was specified, the string is taken as the '
      'group name and the geometry is kept')
for x in mol._atom:
    print(x)

for k, ir in enumerate(mol.irrep_name):
    print('Irrep name %s  (ID %d), symm-adapted-basis shape %s' %
          (ir, mol.irrep_id[k], mol.symm_orb[k].shape))
