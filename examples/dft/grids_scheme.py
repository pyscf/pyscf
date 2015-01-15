#!/usr/bin/env python

import unittest
from pyscf import gto
from pyscf import lib
from pyscf import dft

mol = gto.Mole()
mol.verbose = 0
mol.atom = '''
    o    0    0.       0.
    h    0    -0.757   0.587
    h    0    0.757    0.587'''

mol.basis = '6-31g'
mol.build()

# the default running calls:
# Bragg radius for atom
# Gauss-Chebeshev radial grids
# Becke partition for grid weight
# 
method = dft.RKS(mol)
print('default DFT(LDA) = %.12f' % method.scf())

# Using covalent radius for atom, and Mura-Knowles radial grids
method = dft.RKS(mol)
method.xc = 'b88, p86'
#method.grids.atomic_radii = dft.radi.BRAGG_RADII
method.grids.atomic_radii = dft.radi.COVALENT_RADII
#grids.radi_method = dft.radi.gauss_chebeshev
#grids.radi_method = dft.radi.delley
method.grids.radi_method = dft.radi.mura_knowles
print('change grids for DFT = %.12f' % method.scf())


# Using grids weights of Stratmann, Scuseria, Frisch. CPL, 257, 213 (1996), eq.11
method = dft.RKS(mol)
method.xc = 'b88, p86'
#method.grids.becke_scheme = dft.grid.original_becke
method.grids.becke_scheme = dft.grid.stratmann
print('change grids partition funciton for DFT = %.12f' % method.scf())

# Dense grids
method = dft.RKS(mol)
method.xc = 'b3lyp'
method.level = 4  # big number indicates dense grids. Default is 3
print('Dense grids for DFT = %.12f' % method.scf())

