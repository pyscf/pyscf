#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
PySCF supports input molecule geometry in
1. Cartesian coordinates in one string
2. Z-matrix in one string
3. Internal format (not recommended)
4. Mixed inputs
'''

import numpy
from pyscf import gto

#
# Input cartesian coordinates
# * Four columns, the first column is atom, the next three are x,y,z coordinates
# * Coordinates are in Angstrom by default
#
mol = gto.Mole()
mol.atom = '''
O 0 0 0
H 0 1 0
H 0 0 1
'''

#
# Input Z-matrix
#
mol = gto.M(
    atom = '''
    C
    H    1  1.2
    H    1  1.2  2  109.5
    H-1  1  1.2  2  109.5  3  120
    H-2  1  1.2  2  109.5  3  -120
    ''',
)

#
# * Case insensitive for atom symbol
# * If a number is given at the first column, the number is interpreted as nuclear charge
# * Different atoms are separated by ";" or "\n"
# * space " " or "," are used to separate columns
# * Blank lines will be ignored
#
mol.atom = '''
8        0,   0, 0             ; h:1 0.0 1 0

H@2,0 0 1
'''

#
# Internal format (not recommended).
#
# The internal format of atom is a python list:
# atom = [[atom1, (x, y, z)],
#         [atom2, (x, y, z)],
#         ...
#         [atomN, (x, y, z)]]
#
mol.atom = [['O', (0, 0, 0)],
            ['H', (0, 1, 0)],
            ['H', (0, 0, 1)]]


#
# Mixed inputs
#
mol.atom = ['8 1 1 1.5',
            ('H', 0, 2, 2),
            ('H', numpy.random.random(3))]
#
# You can make use of all possible Python language features to create the
# geometry, eg import geometry from external module, parsing geometry
# database.  Following are examples to generate geometry using python snippet
#
mol.atom = ['O 1 1 1.5']
mol.atom.extend([['H', (i, i, i)] for i in range(1,5)])

mol.atom = 'O 1 1 1.5;'
mol.atom += ';'.join(['H '+(' %f'%i)*3 for i in range(1,5)])

mol.build()
#
# No matter which format or symbol used for the input, function Mole.build()
# will convert mol.atom to the internal format
#



#
# Label atoms
# -----------
# If you want to label one atom to distinguish it from the rest, you can prefix
# or suffix number or special characters 1234567890~!@#$%^&*()_+.?:<>[]{}|
# (execept "," and ";") to an atomic symbol.  It allows you specify
# different basis for the labelled atom (see also 04-input_basis.py)
#
# If the decorated atomic symbol is appeared in mol.atom but not mol.basis,
# the basis parser will remove all decorations and seek the pure atomic symbol
# in mol.basis dict, e.g.  in the following example, 6-31G basis will be
# assigned to the second H atom, but STO-3G will be used for the third atom.
#
mol.atom = [[8,(0, 0, 0)], ['h1',(0, 1, 0)], ['H2',(0, 0, 1)]]
mol.basis = {'O': 'sto-3g', 'H': 'sto3g', 'H1': '6-31G'}

