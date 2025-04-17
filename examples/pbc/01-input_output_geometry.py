#!/usr/bin/env python

'''
This example demonstrates some ways to input the Cell geometry, including
lattice vectors, and to write the Cell geometry to file.

Example solid is wurtzite BN.
'''

from pyscf.pbc import gto

# Input Cartesian coordinates for the lattice vectors a and the atomic
# positions. Coordinates are in Angstrom by default

cell = gto.Cell()
cell.a = [[  2.5539395809,  0.0000000000,  0.0000000000],
          [ -1.2769697905,  2.2117765568,  0.0000000000],
          [  0.0000000000,  0.0000000000,  4.2268548012]]
cell.atom = '''
            B     1.276969829         0.737258874         4.225688066
            N     1.276969829         0.737258874         2.642950986
            B     0.000000000         1.474517748         2.112260792
            N     0.000000000         1.474517748         0.529523459
            '''
cell.pseudo = 'gth-pade'
cell.basis = 'gth-szv'
cell.build()

# Write cell geometry to file
# - Format is guessed from filename
# - These can be read by VESTA, Avogadro, etc.
# - XYZ is Extended XYZ file format, which includes lattice vectors in the
#   comment line
cell.tofile('bn.vasp')
cell.tofile('POSCAR')
cell.tofile('bn.xyz')

# Read a and atom from file
from pyscf.pbc.gto.cell import fromfile
a, atom = fromfile('bn.vasp')
a, atom = fromfile('bn.xyz')

# Read a and atom from file directly into Cell
cell = gto.Cell()
cell.fromfile('bn.vasp')
cell.pseudo = 'gth-pade'
cell.basis = 'gth-szv'
cell.build()
